"""Interactive inference sessions for completed MorphoVoxel training runs."""
from __future__ import annotations

import io
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .genomes import MORPHOLOGIES, one_hot_genomes
from .model_2d import NeuralCA2D
from .model_3d import NeuralCA3D
from .random_utils import resolve_device
from .seeding import seed_state
from .state import StateLayout


def find_checkpoint(run: Path) -> Path | None:
    """Return the preferred local inference checkpoint for a run."""
    for name in ("latest.pt", "best.pt"):
        path = run / "checkpoints" / name
        if path.is_file():
            return path
    return None


def _load_model_weights(path: Path, model: torch.nn.Module) -> None:
    """Load only tensor weights from an app checkpoint, never arbitrary pickle code."""
    from numpy._core.multiarray import _reconstruct

    allowed = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    try:
        with torch.serialization.safe_globals(allowed):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError("checkpoint contains unsupported or unsafe data") from error
    weights = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(weights, dict):
        raise ValueError("checkpoint does not contain model weights")
    model.load_state_dict(weights)


@dataclass
class LabSession:
    """One mutable cellular world driven by a trained local update rule."""

    run_name: str
    config: dict
    dimensions: int
    layout: StateLayout
    model: torch.nn.Module
    state: torch.Tensor
    device: torch.device
    conditional: bool
    genome_index: int = 0
    steps: int = 0
    rate: float = 0.0
    version: int = 0

    @classmethod
    def from_run(cls, run: Path, requested_device: str = "auto") -> "LabSession":
        config_path = run / "config.yaml"
        checkpoint = find_checkpoint(run)
        if not config_path.is_file() or checkpoint is None:
            raise ValueError("this run needs config.yaml and a latest/best checkpoint")
        config = load_config(config_path)
        dimensions = int(config.get("dimensions", 3))
        if dimensions not in {2, 3}:
            raise ValueError("only 2D and 3D training runs can be opened in the lab")
        device = resolve_device(requested_device)
        materials = int(config.get("materials", 3 if dimensions == 2 else 4))
        layout = StateLayout(materials, int(config.get("hidden_channels", 8)))
        conditional = bool(config.get("conditional", False))
        model_class = NeuralCA2D if dimensions == 2 else NeuralCA3D
        model = model_class(
            layout.channels,
            int(config.get("model_width", 32)),
            len(MORPHOLOGIES) if conditional else 0,
            float(config.get("fire_rate", 0.5)),
        ).to(device).eval()
        # Keep optimizer/pool tensors off the GPU; load_state_dict copies model weights.
        _load_model_weights(checkpoint, model)
        size = int(config.get("world_size", 32 if dimensions == 2 else 16))
        state = seed_state(
            1, size, layout, dimensions=dimensions,
            seed_size=int(config.get("seed_size", 1)),
            noise=float(config.get("seed_noise", 0)),
            random_seed=int(config.get("seed", 0)), device=device,
        )
        return cls(run.name, config, dimensions, layout, model, state, device, conditional)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.state.shape[2:])

    @property
    def genome(self) -> torch.Tensor | None:
        if not self.conditional:
            return None
        label = torch.tensor([self.genome_index], dtype=torch.long, device=self.device)
        return one_hot_genomes(label)

    def _views(self) -> tuple[list[str], dict[str, int]]:
        if self.dimensions == 2:
            return ["plane"], {"plane": 1}
        depth, height, width = self.shape
        return ["slice", "top", "front", "side"], {
            "slice": depth, "top": depth, "front": height, "side": width,
        }

    def summary(self) -> dict[str, object]:
        views, layers = self._views()
        occupied = int((self.state[0, self.layout.occupancy] > 0.1).sum().item())
        return {
            "run": self.run_name, "dimensions": self.dimensions, "shape": self.shape,
            "device": str(self.device), "conditional": self.conditional,
            "genomes": list(MORPHOLOGIES) if self.conditional else [],
            "genome": self.genome_index, "steps": self.steps,
            "steps_per_second": self.rate, "occupied_cells": occupied,
            "views": views, "layers": layers, "frame_version": self.version,
        }

    @torch.inference_mode()
    def advance(self, steps: int = 1) -> dict[str, object]:
        if not 1 <= steps <= 32:
            raise ValueError("steps must be from 1 to 32")
        started = time.perf_counter()
        candidate = self.state
        genome = self.genome
        for _ in range(steps):
            candidate = self.model(candidate, genome)
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError(f"model became non-finite after {self.steps} steps")
        self.state = candidate
        self.steps += steps
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.rate = round(steps / max(time.perf_counter() - started, 1e-9), 1)
        self.version += 1
        return self.summary()

    @torch.inference_mode()
    def reset(self, *, clear: bool = False) -> dict[str, object]:
        if clear:
            self.state.zero_()
        else:
            self.state = seed_state(
                1, self.shape, self.layout, dimensions=self.dimensions,
                seed_size=int(self.config.get("seed_size", 1)),
                noise=float(self.config.get("seed_noise", 0)),
                random_seed=int(self.config.get("seed", 0)), device=self.device,
            )
        self.steps = 0
        self.rate = 0.0
        self.version += 1
        return self.summary()

    def set_genome(self, index: int) -> dict[str, object]:
        if not self.conditional or not 0 <= index < len(MORPHOLOGIES):
            raise ValueError("genome is unavailable or out of range")
        self.genome_index = index
        self.version += 1
        return self.summary()

    def _position(self, view: str, row: int, column: int, layer: int = 0) -> tuple[int, ...]:
        if self.dimensions == 2:
            position = (row, column)
        elif view in {"slice", "top"}:
            position = (layer, row, column)
        elif view == "front":
            position = (row, layer, column)
        elif view == "side":
            position = (row, column, layer)
        else:
            raise ValueError("unknown lab view")
        if len(position) != len(self.shape) or any(not 0 <= value < size for value, size in zip(position, self.shape)):
            raise ValueError("edit position is outside the cellular world")
        return position

    @torch.inference_mode()
    def place_seed(self, view: str, row: int, column: int, layer: int = 0) -> dict[str, object]:
        position = self._position(view, row, column, layer)
        seed = seed_state(
            1, self.shape, self.layout, dimensions=self.dimensions,
            seed_size=int(self.config.get("seed_size", 1)),
            noise=float(self.config.get("seed_noise", 0)),
            random_seed=int(self.config.get("seed", 0)), device=self.device,
        )
        shifts = tuple(value - size // 2 for value, size in zip(position, self.shape))
        seed = torch.roll(seed, shifts, dims=tuple(range(2, 2 + self.dimensions)))
        for axis, shift in enumerate(shifts, start=2):
            if not shift:
                continue
            wrapped = [slice(None)] * seed.ndim
            wrapped[axis] = slice(0, shift) if shift > 0 else slice(shift, None)
            seed[tuple(wrapped)] = 0
        self.state = torch.where(seed[:, :1] > 0, seed, self.state)
        self.version += 1
        return self.summary()

    @torch.inference_mode()
    def erase(self, view: str, row: int, column: int, layer: int = 0, radius: int = 2) -> dict[str, object]:
        if not 1 <= radius <= 32:
            raise ValueError("eraser radius must be from 1 to 32")
        position = self._position(view, row, column, layer)
        grids = torch.meshgrid(*(torch.arange(size, device=self.device) for size in self.shape), indexing="ij")
        mask = sum((grid - center) ** 2 for grid, center in zip(grids, position)) <= radius**2
        self.state.masked_fill_(mask[None, None], 0)
        self.version += 1
        return self.summary()

    def _view(self, view: str, layer: int) -> np.ndarray:
        volume = torch.nan_to_num(self.state[0, self.layout.occupancy].detach()).clamp(0, 1).cpu()
        if self.dimensions == 2:
            image = volume
        elif view == "slice":
            image = volume[layer]
        elif view == "top":
            image = volume.max(0).values
        elif view == "front":
            image = volume.max(1).values
        elif view == "side":
            image = volume.max(2).values
        else:
            raise ValueError("unknown lab view")
        return image.numpy()

    def frame_png(self, view: str = "plane", layer: int = 0) -> bytes:
        views, layers = self._views()
        if view not in views:
            raise ValueError("unknown lab view")
        count = layers[view]
        if not 0 <= layer < count:
            raise ValueError("layer is outside the cellular world")
        values = self._view(view, layer)[..., None]
        background = np.asarray([3, 8, 6], dtype=np.float32)
        foreground = np.asarray([120, 247, 191], dtype=np.float32)
        rgb = (background + values * (foreground - background)).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(output, format="PNG")
        return output.getvalue()
