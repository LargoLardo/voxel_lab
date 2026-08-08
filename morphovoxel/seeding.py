"""Deterministic seed initialization."""
from __future__ import annotations

import torch

from .random_utils import generator
from .state import StateLayout


def seed_state(
    batch: int,
    world_size: int | tuple[int, ...],
    layout: StateLayout,
    *,
    dimensions: int,
    seed_size: int = 1,
    noise: float = 0.0,
    random_seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create centered one-cell or odd-width cubic seeds."""
    shape = (world_size,) * dimensions if isinstance(world_size, int) else world_size
    if len(shape) != dimensions or seed_size < 1 or seed_size % 2 == 0:
        raise ValueError("world dimensions must match and seed_size must be positive and odd")
    state = torch.zeros((batch, layout.channels, *shape), device=device)
    slices = tuple(slice(n // 2 - seed_size // 2, n // 2 + seed_size // 2 + 1) for n in shape)
    state[(slice(None), layout.occupancy, *slices)] = 1.0
    state[(slice(None), layout.material_slice, *slices)] = 1.0
    if layout.energy_index is not None:
        state[(slice(None), layout.energy_index, *slices)] = 1.0
    if noise:
        hidden = state[(slice(None), layout.hidden_slice, *slices)]
        hidden.normal_(generator=generator(random_seed, device), std=noise)
    return state

