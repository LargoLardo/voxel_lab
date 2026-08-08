"""Two-dimensional deletion damage."""
from __future__ import annotations

import torch


def damage_2d(state: torch.Tensor, severity: float = 0.25, kind: str = "circle", seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= severity <= 1 or state.ndim != 4:
        raise ValueError("severity must be in [0,1] and state must be [B,C,H,W]")
    h, w = state.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=state.device), torch.arange(w, device=state.device), indexing="ij")
    radius = (severity * h * w / torch.pi) ** 0.5
    if kind == "circle":
        mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2
    elif kind == "dropout":
        mask = torch.rand((h, w), generator=torch.Generator(device=state.device).manual_seed(seed), device=state.device) < severity
    else:
        raise ValueError("unknown 2D damage kind")
    damaged = state.masked_fill(mask[None, None], 0)
    return damaged, mask

