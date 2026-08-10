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
    state_limit: float = 4.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute target, range, leakage, and bounded-state losses."""
    if state_limit <= 0:
        raise ValueError("state_limit must be positive")
    weights = weights or {}
    occupancy = state[:, layout.occupancy]
    target = occupancy_target.to(occupancy)
    components = {
        "occupancy": F.mse_loss(occupancy, target),
        "leakage": (occupancy * (1 - target)).square().mean(),
        "occupancy_range": (F.relu(-occupancy).square() + F.relu(occupancy - 1).square()).mean(),
        "magnitude": F.relu(state.abs() - state_limit).square().mean(),
    }
    occupied = target > 0.5
    logits = state[:, layout.material_slice].movedim(1, -1)
    components["material"] = F.cross_entropy(logits[occupied], material_target.to(state.device)[occupied]) if occupied.any() else state.sum() * 0
    total = sum(weights.get(name, 1.0) * value for name, value in components.items())
    return total, components


def stability_loss(state: torch.Tensor, continued_state: torch.Tensor) -> torch.Tensor:
    return (state[:, :1] - continued_state[:, :1]).square().mean()
