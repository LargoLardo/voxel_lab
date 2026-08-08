"""Lightweight voxel projections."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def occupancy_volume(state) -> np.ndarray:
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    array = np.asarray(state)
    values = array[0, 0] if array.ndim == 5 else array
    return np.clip(values, 0, 1)


def projection(state) -> np.ndarray:
    volume = occupancy_volume(state)
    top, front, side = volume.max(0), volume.max(1), volume.max(2)
    width = max(view.shape[1] for view in (top, front, side))
    padded = [np.pad(view, ((0, 0), (0, width - view.shape[1]))) for view in (top, front, side)]
    return (np.clip(np.concatenate(padded), 0, 1) * 255).astype(np.uint8)


def save_gif(frames, path: str | Path, fps: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [projection(frame) for frame in frames], duration=1000 / fps, loop=0)


def save_comparison(target, state, path: str | Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(6, 5))
    axes[0].imshow(projection(target), cmap="viridis", vmin=0, vmax=255)
    axes[0].set_title("Target projections")
    axes[1].imshow(projection(state), cmap="viridis", vmin=0, vmax=255)
    axes[1].set_title("Prediction projections")
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
        axes[0, index].imshow(channel.max(0), cmap="coolwarm")
        axes[0, index].set_title(f"h{index} max-z")
        axes[0, index].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_isometric(state, path: str | Path, threshold: float = 0.5) -> None:
    occupied = occupancy_volume(state) > threshold
    figure = plt.figure(figsize=(5, 5))
    axis = figure.add_subplot(projection="3d")
    axis.voxels(occupied, facecolors="#66bb6a", edgecolor="#263238", linewidth=0.15)
    axis.set_box_aspect(occupied.shape)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_views(state, path: str | Path) -> None:
    volume = occupancy_volume(state)
    views = (volume.max(0), volume.max(1), volume.max(2), volume[len(volume) // 2])
    titles = ("top", "front", "side", "central slice")
    figure, axes = plt.subplots(1, 4, figsize=(10, 3))
    for axis, view, title in zip(axes, views, titles):
        axis.imshow(view, cmap="viridis", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_recovery_comparison(pre_damage, damaged, recovered, path: str | Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9, 5))
    for axis, value, title in zip(axes, (pre_damage, damaged, recovered), ("Pre-damage", "Damaged", "Recovered")):
        axis.imshow(projection(value), cmap="viridis", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
