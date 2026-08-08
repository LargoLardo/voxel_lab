"""Three-dimensional deletion damage."""
from __future__ import annotations

import torch


KINDS = ("sphere", "cuboid", "plane", "top", "core", "branch", "dropout")


def damage_3d(state: torch.Tensor, severity: float = 0.25, kind: str = "sphere", seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Erase all channels in a geometric region; severity is world-volume based."""
    if not 0 <= severity <= 1 or state.ndim != 5 or kind not in KINDS:
        raise ValueError(f"state must be [B,C,D,H,W], severity in [0,1], kind in {KINDS}")
    d, h, w = state.shape[-3:]
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=state.device) for n in (d, h, w)), indexing="ij")
    if kind == "sphere":
        radius = (severity * d * h * w * 3 / (4 * torch.pi)) ** (1 / 3)
        mask = (zz - d // 2) ** 2 + (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2
    elif kind == "cuboid":
        side = max(1, round((severity * d * h * w) ** (1 / 3)))
        mask = (abs(zz - d // 2) < side / 2) & (abs(yy - h // 2) < side / 2) & (abs(xx - w // 2) < side / 2)
    elif kind == "plane":
        mask = abs(xx - w // 2) < max(1, round(severity * w / 2))
    elif kind == "top":
        mask = zz < round(severity * d)
    elif kind == "core":
        radius = max(1, (severity * h * w / torch.pi) ** 0.5)
        mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2
    elif kind == "branch":
        mask = (yy < h // 2) & (xx > w // 2) & (zz < round(max(1, severity) * d))
    else:
        mask = torch.rand((d, h, w), generator=torch.Generator(device=state.device).manual_seed(seed), device=state.device) < severity
    return state.masked_fill(mask[None, None], 0), mask

