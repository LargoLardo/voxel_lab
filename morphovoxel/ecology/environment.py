"""Ecological world state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..environment import ENVIRONMENT_CHANNELS


@dataclass
class EcologyWorld:
    """Separate organism tensors prevent hidden-state and identity mixing."""

    states: torch.Tensor  # [N,C,D,H,W]
    genomes: torch.Tensor  # [N,G]
    substrate: torch.Tensor  # [D,H,W]
    water: torch.Tensor  # [D,H,W]
    light: torch.Tensor  # [D,H,W]
    energy: torch.Tensor  # [N,D,H,W]
    obstacles: torch.Tensor | None = None  # [D,H,W]
    model_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.states.ndim != 5 or self.energy.shape != (len(self.states), *self.states.shape[-3:]):
            raise ValueError("ecology state and energy shapes do not match")
        if self.genomes.ndim != 2 or len(self.genomes) != len(self.states):
            raise ValueError("one genome vector is required per organism")
        if self.water.shape != self.states.shape[-3:] or self.light.shape != self.water.shape or self.substrate.shape != self.water.shape:
            raise ValueError("environmental fields must match the spatial world shape")
        if self.obstacles is None:
            self.obstacles = torch.zeros_like(self.substrate, dtype=torch.bool)
        elif self.obstacles.shape != self.water.shape:
            raise ValueError("obstacles must match the spatial world shape")
        if self.model_ids is not None:
            self.model_ids = tuple(self.model_ids)
            if len(self.model_ids) != len(self.states):
                raise ValueError("one model id is required per organism")

    @property
    def occupancy(self) -> torch.Tensor:
        return self.states[:, 0].clamp(0, 1)

    @property
    def ownership(self) -> torch.Tensor:
        occupied = self.occupancy > 0.5
        scores = torch.where(occupied, self.occupancy, self.occupancy.new_tensor(-1))
        owner = scores.argmax(0) + 1
        return torch.where(occupied.any(0), owner, torch.zeros_like(owner))


def local_environment_context(
    world: EcologyWorld,
    *,
    light: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
    gravity: Sequence[float] = (-1.0, 0.0, 0.0),
    wind: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Build per-organism ``[N,E,D,H,W]`` context from the current world."""
    gravity_values, wind_values = tuple(map(float, gravity)), tuple(map(float, wind))
    if len(gravity_values) != 3 or len(wind_values) != 3:
        raise ValueError("gravity and wind must each contain [z, y, x]")
    vectors = torch.tensor((*gravity_values, *wind_values))
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("gravity and wind must be finite")

    states, occupancy = world.states, world.occupancy
    count, shape = len(states), states.shape[-3:]

    def shared(field: torch.Tensor) -> torch.Tensor:
        if field.shape != shape:
            raise ValueError("context fields must match the spatial world shape")
        return field.to(states).unsqueeze(0).expand(count, *shape)

    current_energy = world.energy if energy is None else energy
    if current_energy.shape != (count, *shape):
        raise ValueError("energy context must contain one field per organism")
    neighbor_occupancy = (occupancy.sum(0, keepdim=True) - occupancy).clamp(0, 1)
    fields = [
        shared(world.light if light is None else light),
        shared(world.water),
        current_energy.to(states),
        shared(world.substrate),
        shared(world.obstacles),
        neighbor_occupancy,
    ]
    fields.extend(torch.full_like(occupancy, value) for value in gravity_values)
    fields.extend(torch.full_like(occupancy, value) for value in wind_values)
    context = torch.stack(fields, dim=1)
    if context.shape[1] != len(ENVIRONMENT_CHANNELS):
        raise RuntimeError("ecology context no longer matches the environment schema")
    return context
