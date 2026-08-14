"""Three-dimensional neural cellular automaton."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .perception_3d import perceive_3d


class NeuralCA3D(nn.Module):
    """Shared local 3x3x3 update rule for a semantic voxel state."""

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        genome_size: int = 0,
        fire_rate: float = 0.5,
        context_channels: int = 0,
    ):
        super().__init__()
        if not 0 < fire_rate <= 1:
            raise ValueError("fire_rate must be in (0, 1]")
        if context_channels < 0:
            raise ValueError("context_channels must be non-negative")
        self.channels, self.genome_size, self.context_channels, self.fire_rate = channels, genome_size, context_channels, fire_rate
        self.update = nn.Sequential(
            nn.Conv3d(channels * 5 + genome_size + context_channels, hidden, 1), nn.ReLU(), nn.Conv3d(hidden, channels, 1)
        )
        nn.init.normal_(self.update[-1].weight, std=1e-3)
        nn.init.zeros_(self.update[-1].bias)

    def living_mask(self, state: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(state[:, :1], 3, stride=1, padding=1) > 0.1

    def forward(
        self,
        state: torch.Tensor,
        genome: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        fire_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B,C,D,H,W]")
        features = perceive_3d(state)
        if self.genome_size:
            if genome is None or genome.shape != (state.shape[0], self.genome_size):
                raise ValueError("genome must have shape [B, genome_size]")
            features = torch.cat((features, genome[:, :, None, None, None].expand(-1, -1, *state.shape[2:])), 1)
        if self.context_channels:
            expected = (state.shape[0], self.context_channels, *state.shape[2:])
            if context is None or context.shape != expected:
                raise ValueError("context must have shape [B, context_channels, D, H, W]")
            features = torch.cat((features, context.to(features)), 1)
        elif context is not None:
            raise ValueError("this model was created without environment context channels")
        delta = self.update(features)
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
        self.shared = nn.Conv3d(channels * 5 + context_channels, hidden, 1)
        self.film = nn.ModuleList(nn.Linear(self.continuous_size, hidden * 2) for _ in range(family_count))
        self.heads = nn.ModuleList(nn.Conv3d(hidden, channels, 1) for _ in range(family_count))
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for head in self.heads:
            nn.init.normal_(head.weight, std=1e-3)
            nn.init.zeros_(head.bias)

    def living_mask(self, state: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(state[:, :1], 3, stride=1, padding=1) > 0.1

    def forward(
        self,
        state: torch.Tensor,
        genome: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        fire_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B,C,D,H,W]")
        if genome is None or genome.shape != (state.shape[0], self.genome_size):
            raise ValueError("genome must have shape [B, genome_size]")
        family = genome[:, : self.family_count]
        family_ids = family.argmax(1)
        features = perceive_3d(state)
        if self.context_channels:
            expected = (state.shape[0], self.context_channels, *state.shape[2:])
            if context is None or context.shape != expected:
                raise ValueError("context must have shape [B, context_channels, D, H, W]")
            features = torch.cat((features, context.to(features)), 1)
        elif context is not None:
            raise ValueError("this model was created without environment context channels")
        hidden = F.relu(self.shared(features))
        continuous = genome[:, self.family_count :].to(hidden)
        delta = torch.zeros_like(state)
        for index, (film, head) in enumerate(zip(self.film, self.heads)):
            selected = torch.nonzero(family_ids == index, as_tuple=False).flatten()
            selected_hidden = hidden.index_select(0, selected)
            gamma, beta = film(continuous.index_select(0, selected)).chunk(2, 1)
            family_delta = head(F.relu(
                selected_hidden * (1 + gamma[..., None, None, None]) + beta[..., None, None, None]
            ))
            delta = delta.index_copy(0, selected, family_delta)
        fire = torch.rand_like(state[:, :1]) <= self.fire_rate if fire_mask is None else fire_mask
        if fire.shape != state[:, :1].shape:
            raise ValueError("fire_mask must have shape [B,1,D,H,W]")
        before = self.living_mask(state)
        updated = state + delta * fire * before
        alive = before & self.living_mask(updated)
        return updated * alive
