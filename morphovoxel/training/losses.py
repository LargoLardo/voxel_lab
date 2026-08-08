"""Differentiable morphology losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..state import StateLayout


def morphology_loss(
    state: torch.Tensor,
    occupancy_target: torch.Tensor,
    material_target: torch.Tensor,
    layout: StateLayout,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute occupancy, material, and outside-target leakage losses."""
    weights = weights or {}
    occupancy = state[:, layout.occupancy].clamp(0, 1)
    target = occupancy_target.to(occupancy)
    components = {
        "occupancy": F.mse_loss(occupancy, target),
        "leakage": (occupancy * (1 - target)).mean(),
    }
    occupied = target > 0.5
    logits = state[:, layout.material_slice].movedim(1, -1)
    components["material"] = F.cross_entropy(logits[occupied], material_target.to(state.device)[occupied]) if occupied.any() else state.sum() * 0
    total = sum(weights.get(name, 1.0) * value for name, value in components.items())
    return total, components


def stability_loss(state: torch.Tensor, continued_state: torch.Tensor) -> torch.Tensor:
    return (state[:, :1].clamp(0, 1) - continued_state[:, :1].clamp(0, 1)).square().mean()
