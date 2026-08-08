"""Target persistence and summaries."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_target(directory: str | Path, occupancy: np.ndarray, materials: np.ndarray) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "occupancy.npy", occupancy)
    np.save(directory / "materials.npy", materials)


def load_target(directory: str | Path) -> tuple[np.ndarray, np.ndarray]:
    directory = Path(directory)
    return np.load(directory / "occupancy.npy"), np.load(directory / "materials.npy")


def target_summary(occupancy: np.ndarray, materials: np.ndarray) -> dict[str, object]:
    if occupancy.shape != materials.shape:
        raise ValueError("occupancy and materials must have matching shapes")
    points = np.argwhere(occupancy > 0.5)
    return {
        "occupied": int(len(points)),
        "material_counts": {int(k): int(v) for k, v in zip(*np.unique(materials[occupancy > 0.5], return_counts=True))},
        "bounds": None if not len(points) else [points.min(0).tolist(), points.max(0).tolist()],
    }

