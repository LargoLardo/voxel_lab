"""Ecological world state."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EcologyWorld:
    """Separate organism tensors prevent hidden-state and identity mixing."""

    states: torch.Tensor  # [N,C,D,H,W]
    genomes: torch.Tensor  # [N,G]
    substrate: torch.Tensor  # [D,H,W]
    water: torch.Tensor  # [D,H,W]
    light: torch.Tensor  # [D,H,W]
    energy: torch.Tensor  # [N,D,H,W]

    def __post_init__(self) -> None:
        if self.states.ndim != 5 or self.energy.shape != (len(self.states), *self.states.shape[-3:]):
            raise ValueError("ecology state and energy shapes do not match")
        if self.water.shape != self.states.shape[-3:] or self.light.shape != self.water.shape or self.substrate.shape != self.water.shape:
            raise ValueError("environmental fields must match the spatial world shape")

    @property
    def occupancy(self) -> torch.Tensor:
        return self.states[:, 0].clamp(0, 1)

    @property
    def ownership(self) -> torch.Tensor:
        occupied = self.occupancy > 0.5
        scores = torch.where(occupied, self.occupancy, self.occupancy.new_tensor(-1))
        owner = scores.argmax(0) + 1
        return torch.where(occupied.any(0), owner, torch.zeros_like(owner))

