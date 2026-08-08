"""Procedural 3D voxel targets."""
from __future__ import annotations

import numpy as np

TARGETS_3D = ("branching", "conical", "radial", "mushroom", "dome")


def _ball(mask: np.ndarray, z: float, y: float, x: float, radius: float) -> None:
    zz, yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1], : mask.shape[2]]
    mask[(zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius**2] = True


def _segment(mask: np.ndarray, start: tuple[float, float, float], end: tuple[float, float, float], radius: float) -> None:
    distance = np.linalg.norm(np.subtract(end, start))
    for t in np.linspace(0, 1, max(2, int(distance * 2))):
        _ball(mask, *(np.add(start, np.subtract(end, start) * t)), radius)


def make_target_3d(kind: str, size: int = 32, seed: int = 0, **params) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic [D,H,W] occupancy and material maps."""
    if kind not in TARGETS_3D or size < 10:
        raise ValueError(f"kind must be one of {TARGETS_3D} and size >= 10")
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size, size), dtype=bool)
    cx = cy = size // 2
    base, top = size * 0.78, size * 0.24
    radius = max(1.0, size / 22)
    if kind == "conical":
        zz, yy, xx = np.ogrid[:size, :size, :size]
        vertical = (zz >= top) & (zz <= base)
        allowed = (base - zz) / (base - top) * size * 0.24 + 1
        mask[vertical & ((yy - cy) ** 2 + (xx - cx) ** 2 <= allowed**2)] = True
    elif kind == "mushroom":
        _segment(mask, (base, cy, cx), (size * 0.42, cy, cx), radius * 1.4)
        zz, yy, xx = np.ogrid[:size, :size, :size]
        cap = ((zz - size * 0.36) / (size * 0.10)) ** 2 + ((yy - cy) / (size * 0.25)) ** 2 + ((xx - cx) / (size * 0.25)) ** 2 <= 1
        mask |= cap & (zz <= size * 0.4)
    elif kind == "dome":
        zz, yy, xx = np.ogrid[:size, :size, :size]
        r = size * 0.27
        distance = np.sqrt((zz - size * 0.58) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        mask |= (distance <= r) & (distance >= r - 2) & (zz <= size * 0.58)
    else:
        _segment(mask, (base, cy, cx), (size * 0.3, cy, cx), radius * 1.3)
        count = int(params.get("branch_count", 5 if kind == "branching" else 8))
        for index, angle in enumerate(np.linspace(0, 2 * np.pi, count, endpoint=False)):
            start_z = size * (0.38 + 0.25 * (index % 3) / 2) if kind == "branching" else size * 0.5
            length = size * (0.2 + rng.uniform(-0.02, 0.02))
            end = (start_z - (size * 0.12 if kind == "branching" else 0), cy + np.sin(angle) * length, cx + np.cos(angle) * length)
            _segment(mask, (start_z, cy, cx), end, radius)
            _ball(mask, *end, radius * 1.8)
    materials = np.zeros_like(mask, dtype=np.int64)
    materials[mask] = 1
    interior = mask.copy()
    for axis in range(3):
        interior &= np.roll(mask, 1, axis) & np.roll(mask, -1, axis)
    materials[mask & ~interior] = 2
    materials[(mask) & (np.indices(mask.shape)[0] < size * 0.42)] = 3
    return mask.astype(np.float32), materials

