"""Differentiable morphology losses."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..state import StateLayout


def _soft_overlap(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, target = prediction.flatten(1), target.flatten(1)
    intersection = (prediction * target).sum(1)
    epsilon = 1e-6
    dice = (2 * intersection + epsilon) / (prediction.sum(1) + target.sum(1) + epsilon)
    iou = (intersection + epsilon) / (prediction.sum(1) + target.sum(1) - intersection + epsilon)
    return (1 - dice).mean(), (1 - iou).mean()


def _distance_field(target: torch.Tensor) -> torch.Tensor:
    """Normalized Chebyshev distance to a target, using only torch pooling."""
    spatial = target.ndim - 1
    pool = F.max_pool3d if spatial == 3 else F.max_pool2d
    reached = target[:, None] > 0.5
    distance = torch.zeros_like(target[:, None])
    remaining = ~reached
    maximum = max(target.shape[1:])
    with torch.no_grad():
        for step in range(1, maximum + 1):
            expanded = pool(reached.float(), 3, stride=1, padding=1) > 0
            shell = expanded & remaining
            distance[shell] = step / maximum
            reached |= shell
            remaining &= ~shell
    return distance[:, 0]


def _shape_components(
    occupancy: torch.Tensor,
    target: torch.Tensor,
    material_logits: torch.Tensor,
    material_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shape = occupancy.shape[1:]
    coordinates = torch.meshgrid(
        *(torch.linspace(-1, 1, length, device=occupancy.device, dtype=occupancy.dtype) for length in shape),
        indexing="ij",
    )

    def descriptors(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = values.flatten(1).sum(1).clamp_min(1e-6)
        centroid = torch.stack([(values * coordinate).flatten(1).sum(1) / mass for coordinate in coordinates], 1)
        variance = torch.stack([
            (values * (coordinate - centroid[:, index].view(-1, *([1] * len(shape)))).square()).flatten(1).sum(1) / mass
            for index, coordinate in enumerate(coordinates)
        ], 1)
        volume = mass / values[0].numel()
        height = variance[:, 0].clamp_min(1e-8).sqrt()
        width = variance[:, 1:].sum(1).clamp_min(1e-8).sqrt()
        return volume, height, width, centroid

    predicted = descriptors(occupancy)
    wanted = descriptors(target)
    branch_probability = material_logits.softmax(1)[:, min(2, material_logits.shape[1] - 1)] * occupancy
    branch_target = (material_target.to(occupancy.device) == 2).to(occupancy)
    branch_loss, _ = _soft_overlap(branch_probability, branch_target)
    return {
        "volume": (predicted[0] - wanted[0]).square().mean(),
        "height": (predicted[1] - wanted[1]).square().mean(),
        "width": (predicted[2] - wanted[2]).square().mean(),
        "centroid": (predicted[3] - wanted[3]).square().mean(),
        "branch_distribution": branch_loss,
    }


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
    squared_error = (occupancy - target).square().flatten(1)
    foreground = (target > 0.5).flatten(1)
    background = ~foreground
    foreground_error = (squared_error * foreground).sum(1) / foreground.sum(1).clamp_min(1)
    background_error = (squared_error * background).sum(1) / background.sum(1).clamp_min(1)
    prediction = occupancy.clamp(0, 1)
    soft_dice, soft_iou = _soft_overlap(prediction, target)
    components = {
        # Equal foreground/background weighting prevents sparse 3D trees from
        # making an empty or averaged organism look deceptively inexpensive.
        "occupancy": ((foreground_error + background_error) * 0.5).mean(),
        "leakage": (occupancy * (1 - target)).square().mean(),
        "occupancy_range": (F.relu(-occupancy).square() + F.relu(occupancy - 1).square()).mean(),
        "magnitude": F.relu(state.abs() - state_limit).square().mean(),
        "soft_dice": soft_dice,
        "soft_iou": soft_iou,
        "distance": (prediction * (1 - target) * _distance_field(target)).mean(),
    }
    occupied = target > 0.5
    logits = state[:, layout.material_slice].movedim(1, -1)
    components["material"] = F.cross_entropy(logits[occupied], material_target.to(state.device)[occupied]) if occupied.any() else state.sum() * 0
    components.update(_shape_components(occupancy.clamp(0, 1), target, state[:, layout.material_slice], material_target))
    structural = {"soft_dice", "soft_iou", "distance", "height", "width", "volume", "centroid", "branch_distribution"}
    total = sum(weights.get(name, 0.0 if name in structural else 1.0) * value for name, value in components.items())
    return total, components


def counterfactual_loss(state: torch.Tensor, occupancy_target: torch.Tensor, layout: StateLayout) -> torch.Tensor:
    """Match the signed target change within adjacent low/high gene pairs."""
    if len(state) < 2 or len(state) % 2 or len(occupancy_target) != len(state):
        raise ValueError("counterfactual loss requires adjacent even-sized pairs")
    occupancy = state[:, layout.occupancy].clamp(0, 1)
    target = occupancy_target.to(occupancy)
    return F.l1_loss(occupancy[1::2] - occupancy[0::2], target[1::2] - target[0::2])


def stability_loss(state: torch.Tensor, continued_state: torch.Tensor) -> torch.Tensor:
    return (state[:, :1] - continued_state[:, :1]).square().mean()
