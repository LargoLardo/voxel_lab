"""Locally diffusing shared water field."""
from __future__ import annotations

import torch


def diffuse_water(
    water: torch.Tensor,
    substrate: torch.Tensor,
    rate: float = 0.1,
    source: torch.Tensor | None = None,
    absorption: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diffuse pairwise with no-flux boundaries, then apply sources and sinks."""
    if water.ndim != 3 or water.shape != substrate.shape or not 0 <= rate <= 1 / 6:
        raise ValueError("water must be 3D and diffusion rate in [0, 1/6]")
    result = water.clone()
    valid = substrate.bool()
    for axis in range(3):
        left, right = [slice(None)] * 3, [slice(None)] * 3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        pair = valid[left] & valid[right]
        flux = rate * (water[left] - water[right]) * pair
        result[left] -= flux
        result[right] += flux
    if source is not None:
        result = result + source
    if absorption is not None:
        result = result - absorption
    return result.clamp_min(0) * valid


def allocate_water(water: torch.Tensor, demand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fairly allocate a shared field among organism demands [N,D,H,W]."""
    total = demand.sum(0)
    scale = torch.where(total > 0, torch.minimum(torch.ones_like(total), water / total.clamp_min(1e-12)), torch.zeros_like(total))
    absorbed = demand * scale
    return water - absorbed.sum(0), absorbed

