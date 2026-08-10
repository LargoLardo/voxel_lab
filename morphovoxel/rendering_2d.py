"""Standalone 2D rendering."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def occupancy_image(state) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    array = np.asarray(state)
    occupancy = array[0, 0] if array.ndim == 4 else array
    return (np.clip(occupancy, 0, 1) * 255).astype(np.uint8)


def save_gif(frames, path: str | Path, fps: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [occupancy_image(frame) for frame in frames], duration=1000 / fps, loop=0)


def save_comparison(target, state, path: str | Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Target")
    axes[1].imshow(occupancy_image(state), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_hidden_channels(state, hidden_slice: slice, path: str | Path) -> None:
    array = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
    channels = array[0, hidden_slice]
    figure, axes = plt.subplots(1, len(channels), figsize=(2 * len(channels), 2), squeeze=False)
    for index, channel in enumerate(channels):
        axes[0, index].imshow(channel, cmap="coolwarm")
        axes[0, index].set_title(f"h{index}")
        axes[0, index].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
