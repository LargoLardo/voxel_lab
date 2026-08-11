"""Interactive inference sessions for completed MorphoVoxel training runs."""
from __future__ import annotations

import copy
import hashlib
import io
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .checkpointing import load_model_payload
from .config import load_config
from .environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, environment_context_batch
from .genomes import MORPHOLOGIES, TreeGenome, one_hot_genomes, tree_genome_tensor
from .model_2d import NeuralCA2D
from .model_3d import NeuralCA3D
from .random_utils import resolve_device
from .seeding import seed_state
from .state import StateLayout
from .targets import make_target_2d, make_target_3d, make_tree_target
from .targets.targets_2d import TARGETS_2D
from .validation import ValidationCriteria, ValidationReport, build_candidate_panel, validate_panel


def list_checkpoints(run: Path) -> list[Path]:
    directory = run / "checkpoints"
    if not directory.is_dir():
        return []
    priority = {"best.pt": 0, "latest.pt": 1}
    return sorted(
        (path for path in directory.glob("*.pt") if path.is_file() and not path.name.startswith(".")),
        key=lambda path: (priority.get(path.name, 2), path.name),
    )


def find_checkpoint(run: Path, name: str | None = None) -> Path | None:
    """Return the preferred local inference checkpoint for a run."""
    if name is not None:
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("checkpoint must be a local .pt filename")
        path = run / "checkpoints" / name
        return path if path.is_file() else None
    for name in ("best.pt", "latest.pt"):
        path = run / "checkpoints" / name
        if path.is_file():
            return path
    return None


def _load_model_weights(path: Path, model: torch.nn.Module, expected_model_kind: str) -> str:
    """Load only tensor weights from an app checkpoint, never arbitrary pickle code."""
    from numpy._core.multiarray import _reconstruct

    allowed = [_reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    checkpoint_bytes = path.read_bytes()
    try:
        with torch.serialization.safe_globals(allowed):
            payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError("checkpoint contains unsupported or unsafe data") from error
    load_model_payload(
        payload,
        model,
        checkpoint_path=path,
        expected_model_kind=expected_model_kind,
    )
    return hashlib.sha256(checkpoint_bytes).hexdigest()


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
    model_kind: str = "specialist"
    checkpoint_name: str = ""
    checkpoint_sha256: str = ""
    genome_index: int = 0
    active_tree_genome: TreeGenome | None = None
    pending_tree_genome: TreeGenome | None = None
    environment_spec: EnvironmentSpec | None = None
    context: torch.Tensor | None = field(default=None, repr=False)
    last_validation: dict[str, object] | None = field(default=None, repr=False)
    last_validation_report: ValidationReport | None = field(default=None, repr=False)
    steps: int = 0
    rate: float = 0.0
    version: int = 0
    target_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_run(
        cls,
        run: Path,
        requested_device: str = "auto",
        requested_checkpoint: str | None = None,
    ) -> "LabSession":
        config_path = run / "config.yaml"
        checkpoint = find_checkpoint(run, requested_checkpoint)
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
        model_kind = str(config.get("model_kind", "legacy_conditional" if conditional else "specialist"))
        continuous = model_kind == "tree_family"
        if continuous and dimensions != 3:
            raise ValueError("continuous tree-family checkpoints require a 3D run")
        genome_size = TreeGenome.model_size() if continuous else len(MORPHOLOGIES) if conditional else 0
        context_channels = len(ENVIRONMENT_CHANNELS) if bool(config.get("environment_conditioning", continuous)) else 0
        model_class = NeuralCA2D if dimensions == 2 else NeuralCA3D
        model = model_class(
            layout.channels,
            int(config.get("model_width", 32)),
            genome_size,
            float(config.get("fire_rate", 0.5)),
            context_channels,
        ).to(device).eval()
        # Keep optimizer/pool tensors off the GPU; load_state_dict copies model weights.
        checkpoint_sha256 = _load_model_weights(checkpoint, model, model_kind)
        size = int(config.get("world_size", 32 if dimensions == 2 else 16))
        state = seed_state(
            1, size, layout, dimensions=dimensions,
            seed_size=int(config.get("seed_size", 1)),
            noise=float(config.get("seed_noise", 0)),
            random_seed=int(config.get("seed", 0)), device=device,
        )
        tree_model = model_kind in {"tree_family", "tree_specialist"}
        tree_genome = TreeGenome.from_dict(config.get("tree_genome", {})) if tree_model else None
        environment = EnvironmentSpec.from_dict(config.get("environment", {})) if tree_model or context_channels else None
        context = environment_context_batch([environment], size, device=device) if context_channels and environment else None
        return cls(
            run.name, config, dimensions, layout, model, state, device, conditional,
            model_kind=model_kind, checkpoint_name=checkpoint.name,
            checkpoint_sha256=checkpoint_sha256,
            active_tree_genome=tree_genome, pending_tree_genome=tree_genome,
            environment_spec=environment, context=context,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.state.shape[2:])

    @property
    def genome(self) -> torch.Tensor | None:
        if self.model_kind == "tree_family":
            if self.active_tree_genome is None:
                raise ValueError("tree-family session has no active genome")
            return tree_genome_tensor([self.active_tree_genome], device=self.device)
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

    def _target(self) -> tuple[str, torch.Tensor, torch.Tensor]:
        if self.model_kind in {"tree_family", "tree_specialist"}:
            genome = self.active_tree_genome
            if genome is None:
                raw = self.config.get("tree_genome", {})
                genome = TreeGenome.from_dict(raw if isinstance(raw, dict) else {})
            environment = self.environment_spec or EnvironmentSpec()
            key = f"tree:{genome.to_dict()}:{environment.to_dict()}"
            if key not in self.target_cache:
                occupancy, materials = make_tree_target(genome, int(self.config.get("world_size", self.shape[-1])), environment)
                self.target_cache[key] = torch.from_numpy(occupancy), torch.from_numpy(materials)
            occupancy, materials = self.target_cache[key]
            return f"{genome.family} tree", occupancy, materials
        names = MORPHOLOGIES if self.dimensions == 3 else TARGETS_2D
        index = str(self.genome_index if self.conditional else -1)
        kind = names[self.genome_index] if self.conditional else str(self.config.get("target_kind", names[0]))
        if index not in self.target_cache:
            maker = make_target_3d if self.dimensions == 3 else make_target_2d
            occupancy, materials = maker(kind, int(self.config.get("world_size", self.shape[-1])), int(self.config.get("seed", 0)))
            self.target_cache[index] = torch.from_numpy(occupancy), torch.from_numpy(materials)
        occupancy, materials = self.target_cache[index]
        return kind, occupancy, materials

    def summary(self) -> dict[str, object]:
        views, layers = self._views()
        occupied = int((self.state[0, self.layout.occupancy] > 0.1).sum().item())
        target_name, target, _ = self._target()
        return {
            "run": self.run_name, "dimensions": self.dimensions, "shape": self.shape,
            "device": str(self.device), "conditional": self.conditional,
            "model_kind": self.model_kind, "checkpoint": self.checkpoint_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "continuous_genome": self.model_kind == "tree_family",
            "genomes": list(MORPHOLOGIES) if self.conditional and self.model_kind != "tree_family" else [],
            "genome": self.genome_index, "steps": self.steps,
            "steps_per_second": self.rate, "occupied_cells": occupied,
            "target_name": target_name, "target_cells": int((target > 0.5).sum().item()),
            "views": views, "layers": layers, "frame_version": self.version,
            "active_tree_genome": self.active_tree_genome.to_dict() if self.active_tree_genome else None,
            "pending_tree_genome": self.pending_tree_genome.to_dict() if self.pending_tree_genome else None,
            "genome_pending": self.pending_tree_genome != self.active_tree_genome,
            "environment": self.environment_spec.to_dict() if self.environment_spec else None,
            "environment_overlays": list(ENVIRONMENT_CHANNELS[:6]) if self.context is not None else [],
            "validation": self.last_validation,
        }

    @torch.inference_mode()
    def advance(self, steps: int = 1) -> dict[str, object]:
        if not 1 <= steps <= 32:
            raise ValueError("steps must be from 1 to 32")
        started = time.perf_counter()
        candidate = self.state
        genome = self.genome
        for _ in range(steps):
            candidate = self.model(candidate, genome, self.context) if self.context is not None else self.model(candidate, genome)
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
            self._apply_pending_genome()
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
        if self.model_kind == "tree_family" or not self.conditional or not 0 <= index < len(MORPHOLOGIES):
            raise ValueError("genome is unavailable or out of range")
        self.genome_index = index
        self.version += 1
        return self.summary()

    def _apply_pending_genome(self) -> None:
        if self.pending_tree_genome is not None and self.pending_tree_genome != self.active_tree_genome:
            self.active_tree_genome = self.pending_tree_genome
            self.target_cache.clear()

    def set_tree_genome(self, value: dict[str, object], *, live_remodel: bool = False) -> dict[str, object]:
        if self.model_kind != "tree_family":
            raise ValueError("continuous genome controls require a tree-family checkpoint")
        genome = TreeGenome.from_dict(value)
        self.pending_tree_genome = genome
        if live_remodel:
            self.active_tree_genome = genome
            self.target_cache.clear()
        self.version += 1
        return self.summary()

    def randomize_tree_genome(self, seed: int, *, locked: list[str] | None = None) -> dict[str, object]:
        base = self.pending_tree_genome or self.active_tree_genome
        if base is None:
            raise ValueError("continuous genome controls require a tree-family checkpoint")
        return self.set_tree_genome(
            TreeGenome.random(seed, family=base.family, locked=locked or (), base=base).to_dict(),
        )

    def mutate_tree_genome(self, strength: float, seed: int, *, locked: list[str] | None = None) -> dict[str, object]:
        genome = self.pending_tree_genome or self.active_tree_genome
        if genome is None:
            raise ValueError("continuous genome controls require a tree-family checkpoint")
        return self.set_tree_genome(genome.mutate(strength, seed, locked=locked or ()).to_dict())

    def interpolate_tree_genome(self, left: dict[str, object], right: dict[str, object], amount: float) -> dict[str, object]:
        genome = TreeGenome.from_dict(left).interpolate(TreeGenome.from_dict(right), amount)
        return self.set_tree_genome(genome.to_dict())

    def set_environment(self, value: dict[str, object]) -> dict[str, object]:
        if getattr(self.model, "context_channels", 0) != len(ENVIRONMENT_CHANNELS):
            raise ValueError("this checkpoint was not trained with environment context")
        environment = EnvironmentSpec.from_dict(value)
        self.environment_spec = environment
        self.context = environment_context_batch([environment], self.shape, device=self.device)
        self.target_cache.clear()
        self.version += 1
        return self.summary()

    def validate_tree_candidate(
        self,
        *,
        steps: int = 512,
        recovery_steps: int = 128,
        fire_seeds: tuple[int, ...] = (0, 1, 2),
    ) -> dict[str, object]:
        if self.model_kind not in {"tree_family", "tree_specialist"}:
            raise ValueError("tree validation requires a specialist or tree-family checkpoint")
        genome = self.pending_tree_genome or self.active_tree_genome
        if genome is None:
            raise ValueError("tree validation requires a tree genome")
        environment = self.environment_spec or EnvironmentSpec()
        environments = (environment,) if environment == EnvironmentSpec() else (environment, EnvironmentSpec())
        panel = build_candidate_panel(
            genome,
            seed=int(self.config.get("validation_seed", int(self.config.get("seed", 0)) + 100_000)),
            fire_seeds=fire_seeds,
            environments=environments,
        )
        model = copy.deepcopy(self.model).eval()
        report = validate_panel(
            model,
            panel,
            layout=self.layout,
            world_size=self.shape[-1],
            steps=steps,
            recovery_steps=recovery_steps,
            seed_size=int(self.config.get("seed_size", 1)),
            device=self.device,
            criteria=ValidationCriteria(
                min_steps=int(self.config.get("archive_min_validation_steps", 512)),
                min_recovery_steps=int(self.config.get("archive_min_recovery_steps", 128)),
                state_limit=float(self.config.get("state_limit", 4.0)),
            ),
            aggregation=str(self.config.get("validation_aggregation", "worst")),
            low_percentile=float(self.config.get("validation_low_percentile", 0.1)),
        )
        result = {
            "checkpoint": self.checkpoint_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_kind": self.model_kind,
            "genome": genome.to_dict(),
            "environment": environment.to_dict(),
            "report": report.to_dict(),
        }
        if genome == self.pending_tree_genome and environment == (self.environment_spec or EnvironmentSpec()):
            self.last_validation = result
            self.last_validation_report = report
        return result

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
        self._apply_pending_genome()
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

    def _display_volume(self, source: str) -> tuple[torch.Tensor, torch.Tensor]:
        if source == "organism":
            occupancy = torch.nan_to_num(self.state[0, self.layout.occupancy].detach()).clamp(0, 1).cpu()
            materials = torch.nan_to_num(self.state[0, self.layout.material_slice].detach()).argmax(0).cpu()
            return occupancy, materials
        if source == "target":
            _, occupancy, materials = self._target()
            return occupancy, materials
        if source.startswith("environment:") and self.context is not None:
            name = source.partition(":")[2]
            if name not in ENVIRONMENT_CHANNELS[:6]:
                raise ValueError("unknown environment overlay")
            occupancy = self.context[0, ENVIRONMENT_CHANNELS.index(name)].detach().clamp(0, 1).cpu()
            return occupancy, torch.zeros_like(occupancy, dtype=torch.long)
        raise ValueError("source must be organism, target, or an available environment overlay")

    def _view(self, view: str, layer: int, source: str = "organism") -> np.ndarray:
        volume, _ = self._display_volume(source)
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

    def frame_png(self, view: str = "plane", layer: int = 0, source: str = "organism") -> bytes:
        views, layers = self._views()
        if view not in views:
            raise ValueError("unknown lab view")
        count = layers[view]
        if not 0 <= layer < count:
            raise ValueError("layer is outside the cellular world")
        values = self._view(view, layer, source)[..., None]
        background = np.asarray([3, 8, 6], dtype=np.float32)
        foreground = np.asarray([120, 247, 191], dtype=np.float32)
        rgb = (background + values * (foreground - background)).astype(np.uint8)
        output = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(output, format="PNG")
        return output.getvalue()

    def voxel_data(self, threshold: float = 0.1, source: str = "organism") -> dict[str, object]:
        """Return visible 3D cells for the browser's interactive cube renderer."""
        if self.dimensions != 3:
            raise ValueError("voxel view is only available for 3D runs")
        occupancy, material_map = self._display_volume(source)
        occupied = occupancy > threshold
        interior = occupied.clone()
        interior[[0, -1], :, :] = False
        interior[:, [0, -1], :] = False
        interior[:, :, [0, -1]] = False
        interior[1:-1, 1:-1, 1:-1] &= (
            occupied[:-2, 1:-1, 1:-1] & occupied[2:, 1:-1, 1:-1]
            & occupied[1:-1, :-2, 1:-1] & occupied[1:-1, 2:, 1:-1]
            & occupied[1:-1, 1:-1, :-2] & occupied[1:-1, 1:-1, 2:]
        )
        visible = occupied & ~interior
        coordinates = visible.nonzero()
        materials = material_map[visible]
        voxels = torch.cat((coordinates, occupancy[visible, None], materials[:, None]), dim=1).cpu().tolist()
        return {"shape": list(self.shape), "voxels": voxels, "frame_version": self.version}
