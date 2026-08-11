"""Serializable environment controls and local fields sensed by 3D NCAs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import torch

ENVIRONMENT_SCHEMA_VERSION = 1
ENVIRONMENT_CHANNELS = (
    "light",
    "water",
    "energy",
    "substrate",
    "obstacles",
    "neighbor_occupancy",
    "gravity_z",
    "gravity_y",
    "gravity_x",
    "wind_z",
    "wind_y",
    "wind_x",
)
ENVIRONMENT_PARAMETERS = (
    "light_direction_y",
    "light_direction_x",
    "water_direction_y",
    "water_direction_x",
    "water_level",
    "energy",
    "obstacle_density",
    "neighbor_pressure",
    "wind_direction_y",
    "wind_direction_x",
    "wind_strength",
)


@dataclass(frozen=True)
class EnvironmentSpec:
    """Inherited-independent conditions used to construct local context fields."""

    light_direction_y: float = 0.0
    light_direction_x: float = 0.0
    water_direction_y: float = 0.0
    water_direction_x: float = 0.0
    water_level: float = 1.0
    energy: float = 1.0
    obstacle_density: float = 0.0
    neighbor_pressure: float = 0.0
    wind_direction_y: float = 0.0
    wind_direction_x: float = 0.0
    wind_strength: float = 0.0
    seed: int = 0
    schema_version: int = ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported environment schema version: {self.schema_version}")
        for name in ("light_direction_y", "light_direction_x", "water_direction_y", "water_direction_x", "wind_direction_y", "wind_direction_x"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [-1, 1]")
            object.__setattr__(self, name, value)
        for name in ("water_level", "energy", "neighbor_pressure", "wind_strength"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0, 1]")
            object.__setattr__(self, name, value)
        density = float(self.obstacle_density)
        if not math.isfinite(density) or not 0 <= density <= 0.3:
            raise ValueError("obstacle_density must be finite and within [0, 0.3]")
        object.__setattr__(self, "obstacle_density", density)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("environment seed must be an integer from 0 through 2^31-1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EnvironmentSpec":
        if not isinstance(value, Mapping):
            raise ValueError("environment JSON must be an object")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown environment fields: {', '.join(sorted(map(str, unknown)))}")
        return cls(**value)  # type: ignore[arg-type]

    @classmethod
    def random(cls, seed: int, *, span: float = 1.0) -> "EnvironmentSpec":
        if not 0 <= span <= 1:
            raise ValueError("environment span must be within [0, 1]")
        rng = np.random.default_rng(seed)
        direction = lambda: float(rng.uniform(-span, span))
        return cls(
            light_direction_y=direction(), light_direction_x=direction(),
            water_direction_y=direction(), water_direction_x=direction(),
            water_level=float(rng.uniform(1 - span * 0.7, 1)), energy=float(rng.uniform(1 - span * 0.5, 1)),
            obstacle_density=float(rng.uniform(0, span * 0.12)), neighbor_pressure=float(rng.uniform(0, span * 0.5)),
            wind_direction_y=direction(), wind_direction_x=direction(), wind_strength=float(rng.uniform(0, span)),
            seed=int(rng.integers(0, 2**31)),
        )

    def vector(self) -> torch.Tensor:
        # float64 preserves every permitted 31-bit seed exactly in pool/checkpoint
        # metadata; generated local fields are still float32 by default.
        return torch.tensor([getattr(self, name) for name in ENVIRONMENT_PARAMETERS] + [float(self.seed)], dtype=torch.float64)

    @classmethod
    def from_vector(cls, value: torch.Tensor) -> "EnvironmentSpec":
        flat = value.detach().cpu().flatten().tolist()
        if len(flat) != len(ENVIRONMENT_PARAMETERS) + 1:
            raise ValueError("environment vector has the wrong size")
        values = dict(zip(ENVIRONMENT_PARAMETERS, flat[:-1]))
        return cls(**values, seed=int(round(flat[-1])))


def _obstacle_field(spec: EnvironmentSpec, size: int) -> np.ndarray:
    field = np.zeros((size, size, size), dtype=np.float32)
    if spec.obstacle_density == 0:
        return field
    rng = np.random.default_rng(spec.seed)
    zz, yy, xx = np.ogrid[:size, :size, :size]
    count = max(1, round(size * spec.obstacle_density))
    for _ in range(count):
        center = (rng.uniform(size * 0.3, size * 0.75), rng.uniform(2, size - 2), rng.uniform(2, size - 2))
        radius = rng.uniform(0.7, max(1.0, size * 0.09))
        field[(zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2] = 1
    return field


def _neighbor_field(spec: EnvironmentSpec, size: int) -> np.ndarray:
    """Seeded local neighboring-organism occupancy used during training."""
    field = np.zeros((size, size, size), dtype=np.float32)
    if spec.neighbor_pressure == 0:
        return field
    rng = np.random.default_rng(spec.seed ^ 0x5A17_1EAF)
    zz, yy, xx = np.ogrid[:size, :size, :size]
    count = max(1, round(1 + spec.neighbor_pressure * 4))
    for index in range(count):
        angle = 2 * np.pi * index / count + rng.uniform(-0.35, 0.35)
        radial = size * rng.uniform(0.16, 0.28)
        center = (
            rng.uniform(size * 0.22, size * 0.72),
            (size - 1) / 2 + np.sin(angle) * radial,
            (size - 1) / 2 + np.cos(angle) * radial,
        )
        radius = rng.uniform(0.7, 0.8 + spec.neighbor_pressure * size * 0.08)
        field[(zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2] = 1
    return field


def make_environment_context(
    spec: EnvironmentSpec,
    size: int | tuple[int, int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build ``[E,D,H,W]`` local fields in z/y/x axis order."""
    shape = (size, size, size) if isinstance(size, int) else tuple(size)
    if len(shape) != 3 or min(shape) < 4:
        raise ValueError("environment shape must contain three dimensions of at least four cells")
    z = torch.linspace(-1, 1, shape[0], device=device, dtype=dtype)[:, None, None]
    y = torch.linspace(-1, 1, shape[1], device=device, dtype=dtype)[None, :, None]
    x = torch.linspace(-1, 1, shape[2], device=device, dtype=dtype)[None, None, :]
    light = (1 - (z + 1) * 0.42 + y * spec.light_direction_y * 0.25 + x * spec.light_direction_x * 0.25).clamp(0, 1).expand(shape)
    substrate = torch.zeros(shape, device=device, dtype=dtype)
    substrate[-2:] = 1
    water_focus = (1 - ((y - spec.water_direction_y * 0.65).square() + (x - spec.water_direction_x * 0.65).square()).sqrt() / 2).clamp(0, 1)
    water = substrate * water_focus.expand(shape) * spec.water_level
    energy = torch.full(shape, spec.energy, device=device, dtype=dtype)
    obstacles = torch.as_tensor(_obstacle_field(spec, shape[0]), device=device, dtype=dtype)
    if obstacles.shape != shape:
        # Non-cubic contexts are valid even though the simple seeded obstacle
        # generator uses the depth as its base scale.
        obstacles = torch.nn.functional.interpolate(obstacles[None, None], size=shape, mode="nearest")[0, 0]
    neighbors = torch.as_tensor(_neighbor_field(spec, shape[0]), device=device, dtype=dtype)
    if neighbors.shape != shape:
        neighbors = torch.nn.functional.interpolate(neighbors[None, None], size=shape, mode="nearest")[0, 0]
    gravity_z = torch.full(shape, -1.0, device=device, dtype=dtype)
    zero = torch.zeros(shape, device=device, dtype=dtype)
    wind_y = torch.full(shape, spec.wind_direction_y * spec.wind_strength, device=device, dtype=dtype)
    wind_x = torch.full(shape, spec.wind_direction_x * spec.wind_strength, device=device, dtype=dtype)
    return torch.stack((light, water, energy, substrate, obstacles, neighbors, gravity_z, zero, zero, zero, wind_y, wind_x))


def environment_context_batch(
    specs: Iterable[EnvironmentSpec],
    size: int | tuple[int, int, int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    contexts = [make_environment_context(spec, size, device=device, dtype=dtype) for spec in specs]
    if not contexts:
        raise ValueError("at least one environment specification is required")
    return torch.stack(contexts)
