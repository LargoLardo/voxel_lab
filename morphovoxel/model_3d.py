"""Three-dimensional neural cellular automaton."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import resolve_hidden_layers
from .perception_3d import perceive_3d


class _PointwiseConv3d(nn.Conv3d):
    """A voxel-wise affine layer with the same checkpoint format as Conv3d."""

    def __init__(self, incoming: int, outgoing: int):
        super().__init__(incoming, outgoing, 1)

    def prepare_conditioning(self, fixed_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split fixed input channels' affine contribution out of a rollout."""
        split = self.in_channels - fixed_inputs.shape[1]
        return self.weight[:, :split].contiguous(), self._affine(
            fixed_inputs.to(self.weight), self.weight[:, split:].contiguous(), self.bias,
        )

    @staticmethod
    def _affine(state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if state.device.type == "mps" and torch.is_grad_enabled():
            # Metal's 3D convolution backward is costly even for a 1x1x1
            # kernel. Flattening two spatial axes lets its 2D kernel perform
            # exactly the same independent affine transform at each voxel.
            # Without backward, native Conv3d is already as fast or faster.
            return F.conv2d(state.flatten(3), weight.squeeze(-1), bias).reshape(
                state.shape[0], weight.shape[0], *state.shape[2:],
            )
        return F.conv3d(state, weight, bias)

    def forward(self, state: torch.Tensor, *, prepared_conditioning=None) -> torch.Tensor:
        if prepared_conditioning is not None:
            weight, offset = prepared_conditioning
            return self._affine(state, weight, None) + offset
        return self._affine(state, self.weight, self.bias)


class NeuralCA3D(nn.Module):
    """Shared local 3x3x3 update rule for a semantic voxel state."""

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        genome_size: int = 0,
        fire_rate: float = 0.5,
        context_channels: int = 0,
        *,
        hidden_layers: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if not 0 < fire_rate <= 1:
            raise ValueError("fire_rate must be in (0, 1]")
        if context_channels < 0:
            raise ValueError("context_channels must be non-negative")
        self.channels, self.genome_size, self.context_channels, self.fire_rate = channels, genome_size, context_channels, fire_rate
        self.hidden_layers = resolve_hidden_layers(hidden, hidden_layers)
        widths = (channels * 5 + genome_size + context_channels, *self.hidden_layers)
        layers = []
        for incoming, outgoing in zip(widths, widths[1:]):
            layers.extend((_PointwiseConv3d(incoming, outgoing), nn.ReLU()))
        self.update = nn.Sequential(*layers, _PointwiseConv3d(widths[-1], channels))
        nn.init.normal_(self.update[-1].weight, std=1e-3)
        nn.init.zeros_(self.update[-1].bias)

    def living_mask(self, state: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(state[:, :1], 3, stride=1, padding=1) > 0.1

    def prepare_conditioning(self, genome: torch.Tensor | None, context: torch.Tensor):
        context = context.to(self.update[0].weight)
        if self.genome_size:
            if genome is None or genome.shape != (context.shape[0], self.genome_size):
                raise ValueError("genome must have shape [B, genome_size]")
            inherited = genome.to(context)[:, :, None, None, None].expand(-1, -1, *context.shape[2:])
            context = torch.cat((inherited, context), 1)
        return self.update[0].prepare_conditioning(context)

    def forward(
        self,
        state: torch.Tensor,
        genome: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        fire_mask: torch.Tensor | None = None,
        *,
        prepared_conditioning: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B,C,D,H,W]")
        features = perceive_3d(state)
        if self.genome_size:
            if genome is None or genome.shape != (state.shape[0], self.genome_size):
                raise ValueError("genome must have shape [B, genome_size]")
            if prepared_conditioning is None:
                features = torch.cat((features, genome[:, :, None, None, None].expand(-1, -1, *state.shape[2:])), 1)
        if self.context_channels:
            expected = (state.shape[0], self.context_channels, *state.shape[2:])
            if context is None or context.shape != expected:
                raise ValueError("context must have shape [B, context_channels, D, H, W]")
            if prepared_conditioning is None:
                features = torch.cat((features, context.to(features)), 1)
        elif context is not None:
            raise ValueError("this model was created without environment context channels")
        if prepared_conditioning is None:
            delta = self.update(features)
        else:
            layers = iter(self.update)
            delta = next(layers)(features, prepared_conditioning=prepared_conditioning)
            for layer in layers:
                delta = layer(delta)
        fire = torch.rand_like(state[:, :1]) <= self.fire_rate if fire_mask is None else fire_mask
        if fire.shape != state[:, :1].shape:
            raise ValueError("fire_mask must have shape [B,1,D,H,W]")
        before = self.living_mask(state)
        updated = state + delta * fire * before
        alive = before & self.living_mask(updated)
        return updated * alive


class TreeFamilyNCA3D(nn.Module):
    """Tree-family NCA with shared perception and explicit genome modulation."""

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        genome_size: int = 15,
        fire_rate: float = 0.5,
        context_channels: int = 0,
        family_count: int = 4,
        *,
        hidden_layers: list[int] | tuple[int, ...] | None = None,
    ):
        super().__init__()
        if not 0 < fire_rate <= 1:
            raise ValueError("fire_rate must be in (0, 1]")
        if family_count < 2 or genome_size <= family_count or context_channels < 0:
            raise ValueError("tree family dimensions are invalid")
        self.channels = channels
        self.genome_size = genome_size
        self.context_channels = context_channels
        self.fire_rate = fire_rate
        self.family_count = family_count
        self.continuous_size = genome_size - family_count
        self.hidden_layers = resolve_hidden_layers(hidden, hidden_layers)
        self.shared = _PointwiseConv3d(channels * 5 + context_channels, self.hidden_layers[0])
        layers = []
        for incoming, outgoing in zip(self.hidden_layers, self.hidden_layers[1:]):
            layers.extend((_PointwiseConv3d(incoming, outgoing), nn.ReLU()))
        self.hidden_update = nn.Sequential(*layers)
        hidden = self.hidden_layers[-1]
        self.film = nn.ModuleList(nn.Linear(self.continuous_size, hidden * 2) for _ in range(family_count))
        self.heads = nn.ModuleList(_PointwiseConv3d(hidden, channels) for _ in range(family_count))
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for head in self.heads:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def living_mask(self, state: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(state[:, :1], 3, stride=1, padding=1) > 0.1

    def prepare_conditioning(self, genome: torch.Tensor | None, context: torch.Tensor):
        return self.shared.prepare_conditioning(context)

    def prepare_genome(self, genome: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Prepare fixed conditioning for one rollout, retaining its autograd graph.

        The caller must discard this after changing weights or genomes. Keeping
        it local to a rollout avoids stale weights and freed backward graphs.
        """
        if genome is None or genome.ndim != 2 or genome.shape[1] != self.genome_size:
            raise ValueError("genome must have shape [B, genome_size]")
        family_ids = genome[:, :self.family_count].argmax(1)
        continuous = genome[:, self.family_count:].to(self.shared.weight)
        film_weight = torch.stack([film.weight for film in self.film])[family_ids]
        film_bias = torch.stack([film.bias for film in self.film])[family_ids]
        modulation = torch.bmm(film_weight, continuous.unsqueeze(2)).squeeze(2) + film_bias
        gamma, beta = modulation.chunk(2, 1)
        head_weight = torch.stack([head.weight[:, :, 0, 0, 0] for head in self.heads])[family_ids]
        head_bias = torch.stack([head.bias for head in self.heads])[family_ids]
        return gamma, beta, head_weight, head_bias

    def forward(
        self,
        state: torch.Tensor,
        genome: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        fire_mask: torch.Tensor | None = None,
        *,
        prepared_genome: tuple[torch.Tensor, ...] | None = None,
        prepared_conditioning: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B,C,D,H,W]")
        if genome is None or genome.shape != (state.shape[0], self.genome_size):
            raise ValueError("genome must have shape [B, genome_size]")
        features = perceive_3d(state)
        if self.context_channels:
            expected = (state.shape[0], self.context_channels, *state.shape[2:])
            if context is None or context.shape != expected:
                raise ValueError("context must have shape [B, context_channels, D, H, W]")
            if prepared_conditioning is None:
                features = torch.cat((features, context.to(features)), 1)
        elif context is not None:
            raise ValueError("this model was created without environment context channels")
        hidden = F.relu(self.shared(features) if prepared_conditioning is None else
                        self.shared(features, prepared_conditioning=prepared_conditioning))
        hidden = self.hidden_update(hidden)
        gamma, beta, head_weight, head_bias = self.prepare_genome(genome) if prepared_genome is None else prepared_genome
        modulated = F.relu(hidden * (1 + gamma[..., None, None, None]) + beta[..., None, None, None])
        delta = torch.bmm(head_weight, modulated.flatten(2)).reshape_as(state) + head_bias[..., None, None, None]
        fire = torch.rand_like(state[:, :1]) <= self.fire_rate if fire_mask is None else fire_mask
        if fire.shape != state[:, :1].shape:
            raise ValueError("fire_mask must have shape [B,1,D,H,W]")
        before = self.living_mask(state)
        updated = state + delta * fire * before
        alive = before & self.living_mask(updated)
        return updated * alive
