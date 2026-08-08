"""Two-dimensional neural cellular automaton."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .perception_2d import perceive_2d


class NeuralCA2D(nn.Module):
    """Shared local 3x3 update rule for a semantic 2D state."""

    def __init__(self, channels: int, hidden: int = 64, genome_size: int = 0, fire_rate: float = 0.5):
        super().__init__()
        if not 0 < fire_rate <= 1:
            raise ValueError("fire_rate must be in (0, 1]")
        self.channels, self.genome_size, self.fire_rate = channels, genome_size, fire_rate
        self.update = nn.Sequential(
            nn.Conv2d(channels * 4 + genome_size, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1)
        )
        nn.init.normal_(self.update[-1].weight, std=1e-3)
        nn.init.zeros_(self.update[-1].bias)

    def living_mask(self, state: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(state[:, :1], 3, stride=1, padding=1) > 0.1

    def forward(self, state: torch.Tensor, genome: torch.Tensor | None = None) -> torch.Tensor:
        if state.ndim != 4 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [B,C,H,W]")
        features = perceive_2d(state)
        if self.genome_size:
            if genome is None or genome.shape != (state.shape[0], self.genome_size):
                raise ValueError("genome must have shape [B, genome_size]")
            features = torch.cat((features, genome[:, :, None, None].expand(-1, -1, *state.shape[2:])), 1)
        delta = self.update(features)
        fire = torch.rand_like(state[:, :1]) <= self.fire_rate
        return state + delta * fire * self.living_mask(state)
