"""Procedural 2D organism targets."""
from __future__ import annotations

import numpy as np

TARGETS_2D = ("branching", "radial", "elongated", "asymmetric")


def _disk(mask: np.ndarray, y: float, x: float, radius: float) -> None:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(yy - y) ** 2 + (xx - x) ** 2 <= radius**2] = True


def _segment(mask: np.ndarray, start: tuple[float, float], end: tuple[float, float], radius: float) -> None:
    y0, x0 = start
    y1, x1 = end
    for t in np.linspace(0, 1, max(2, int(np.hypot(y1 - y0, x1 - x0) * 2))):
        _disk(mask, y0 + t * (y1 - y0), x0 + t * (x1 - x0), radius)


def make_target_2d(kind: str, size: int = 64, seed: int = 0, **params) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic occupancy and integer material maps."""
    if kind not in TARGETS_2D or size < 12:
        raise ValueError(f"kind must be one of {TARGETS_2D} and size >= 12")
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2
    thickness = float(params.get("thickness", max(1, size / 48)))
    if kind == "radial":
        _disk(mask, cy, cx, size * 0.09)
        for angle in np.linspace(0, 2 * np.pi, int(params.get("branch_count", 6)), endpoint=False):
            length = size * 0.28
            _segment(mask, (cy, cx), (cy + np.sin(angle) * length, cx + np.cos(angle) * length), thickness)
    elif kind == "elongated":
        _segment(mask, (size * 0.78, cx), (size * 0.2, cx), thickness * 1.5)
        for y, direction in ((0.38, -1), (0.52, 1), (0.65, -1)):
            _segment(mask, (size * y, cx), (size * (y - 0.12), cx + direction * size * 0.16), thickness)
    else:
        jitter = rng.uniform(-0.08, 0.08, 5) if kind == "asymmetric" else np.zeros(5)
        _segment(mask, (size * 0.8, cx), (size * 0.32, cx + jitter[0] * size), thickness * 1.4)
        for index, (y, direction) in enumerate(((0.42, -1), (0.5, 1), (0.62, -1), (0.68, 1))):
            length = size * (0.16 + jitter[index + 1])
            start = (size * y, cx)
            end = (size * (y - 0.18), cx + direction * length)
            _segment(mask, start, end, thickness)
            if kind == "branching":
                _segment(mask, end, (end[0] - size * 0.08, end[1] + direction * size * 0.07), thickness * 0.8)
    materials = np.zeros_like(mask, dtype=np.int64)
    materials[mask] = 1
    boundary = mask & ~(
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    )
    materials[boundary] = 2
    return mask.astype(np.float32), materials

