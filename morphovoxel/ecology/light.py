"""Interpretable top-down light attenuation."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def compute_light(occupancy: torch.Tensor, incident: float = 1.0, attenuation: float = 0.7, lateral_diffusion: float = 0.0) -> torch.Tensor:
    """Compute light [D,H,W] from combined occupancy [D,H,W] or [N,D,H,W]."""
    if occupancy.ndim == 4:
        occupancy = occupancy.max(0).values
    if occupancy.ndim != 3 or not 0 <= attenuation <= 1 or not 0 <= lateral_diffusion <= 1:
        raise ValueError("invalid occupancy shape or light parameters")
    light = torch.empty_like(occupancy)
    incoming = torch.full_like(occupancy[0], incident)
    for z in range(len(occupancy)):
        light[z] = incoming
        incoming = incoming * (1 - attenuation * occupancy[z].clamp(0, 1))
    if lateral_diffusion:
        blurred = F.avg_pool3d(F.pad(light[None, None], (1, 1, 1, 1, 1, 1), mode="replicate"), 3)[0, 0]
        light = torch.lerp(light, blurred, lateral_diffusion)
    return light.clamp(0, incident)

