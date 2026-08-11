"""Procedural 3D voxel targets."""
from __future__ import annotations

import math

import numpy as np

from ..environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, make_environment_context
from ..genomes import TreeGenome

TARGETS_3D = ("branching", "conical", "radial", "mushroom", "dome")
TREE_TARGET_VERSION = 1


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


def _normalized(genome: TreeGenome, name: str) -> float:
    return (genome.value(name) + 1) / 2


def _seed_component(mask: np.ndarray, seed: tuple[int, int, int]) -> np.ndarray:
    """Keep only growth still connected to the planted base after avoidance."""
    if not mask[seed]:
        return np.zeros_like(mask)
    kept = np.zeros_like(mask)
    kept[seed] = True
    stack = [seed]
    while stack:
        z, y, x = stack.pop()
        for point in ((z - 1, y, x), (z + 1, y, x), (z, y - 1, x), (z, y + 1, x), (z, y, x - 1), (z, y, x + 1)):
            if all(0 <= value < limit for value, limit in zip(point, mask.shape)) and mask[point] and not kept[point]:
                kept[point] = True
                stack.append(point)
    return kept


def make_tree_target(
    genome: TreeGenome,
    size: int = 16,
    environment: EnvironmentSpec | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic semantic tree target from inherited genes.

    Axis order is ``z, y, x`` with gravity toward increasing z. Environment
    parameters influence the target but remain separate from the genome.
    """
    if not isinstance(genome, TreeGenome):
        raise ValueError("genome must be a TreeGenome")
    if size < 12:
        raise ValueError("continuous tree targets require size >= 12")
    environment = environment or EnvironmentSpec()
    # Style variation is a smooth function of the same phase encoded in the
    # model's two style inputs. An unrelated PRNG stream would ask the NCA to
    # infer information that is not present in its genome vector.
    style_phase = genome.style_phase
    trunk = np.zeros((size, size, size), dtype=bool)
    branches = np.zeros_like(trunk)
    leaves = np.zeros_like(trunk)
    roots = np.zeros_like(trunk)
    center = np.array((size - 3.0, (size - 1) / 2, (size - 1) / 2))

    resource_scale = 0.72 + 0.14 * environment.water_level + 0.14 * environment.energy
    height = (
        size * (0.38 + 0.34 * _normalized(genome, "height"))
        * resource_scale * (1 - 0.18 * environment.wind_strength)
    )
    thickness = 0.55 + 1.05 * _normalized(genome, "trunk_thickness") + 0.35 * environment.wind_strength
    tropism = genome.value("light_tropism")
    lean = np.array((
        0.0,
        environment.light_direction_y * tropism * size * 0.12 - environment.wind_direction_y * environment.wind_strength * size * 0.08,
        environment.light_direction_x * tropism * size * 0.12 - environment.wind_direction_x * environment.wind_strength * size * 0.08,
    ))
    top = center + np.array((-height, 0.0, 0.0)) + lean
    taper = 0.25 + 0.65 * _normalized(genome, "taper")
    for fraction in np.linspace(0, 1, max(4, int(height * 2))):
        point = center + (top - center) * fraction
        _ball(trunk, *point, max(0.55, thickness * (1 - taper * fraction * 0.72)))

    density = _normalized(genome, "branch_density")
    branch_count = max(3, round((4 + density * 9) * (0.8 + 0.2 * environment.energy)))
    inclination = genome.value("branch_inclination")
    base_length = (
        size * (0.11 + 0.16 * _normalized(genome, "branch_length"))
        * (0.75 + 0.25 * environment.energy)
    )
    spread = 0.75 + 0.7 * _normalized(genome, "canopy_spread")
    asymmetry = genome.value("asymmetry")
    allocation = _normalized(genome, "root_canopy_allocation")
    family_angle = style_phase % (2 * np.pi)
    for index in range(branch_count):
        level = (index + 1) / (branch_count + 1)
        if genome.family == "conifer":
            trunk_fraction = 0.25 + 0.7 * level
            family_scale = 1.25 - 0.7 * level
        elif genome.family == "broad_canopy":
            trunk_fraction = 0.58 + 0.36 * level
            family_scale = 1.25
        elif genome.family == "weeping":
            trunk_fraction = 0.55 + 0.38 * level
            family_scale = 1.12
        else:
            trunk_fraction = 0.32 + 0.62 * level
            family_scale = 1.0
        start = center + (top - center) * trunk_fraction
        branch_offset = index * (2 * np.pi / branch_count)
        angle = family_angle + branch_offset + 0.13 * math.sin(style_phase + branch_offset * 1.7)
        directional = math.atan2(environment.light_direction_y, environment.light_direction_x) if (
            environment.light_direction_y or environment.light_direction_x
        ) else angle
        angle += tropism * 0.22 * math.sin(directional - angle)
        side_bias = 1 + asymmetry * 0.32 * math.cos(angle - family_angle)
        length = base_length * family_scale * side_bias * (1 + 0.12 * math.cos(style_phase + branch_offset * 2.3))
        rise = -length * (0.28 + inclination * 0.22)
        if genome.family == "weeping":
            rise = -length * 0.12
        end = start + np.array((rise, math.sin(angle) * length, math.cos(angle) * length))
        _segment(branches, tuple(start), tuple(end), max(0.5, thickness * 0.55))
        if genome.family == "weeping":
            droop = end + np.array((length * (0.6 + 0.35 * _normalized(genome, "branch_inclination")), 0, 0))
            _segment(branches, tuple(end), tuple(droop), 0.5)
            end = droop
        leaf_radius = max(0.75, thickness * 0.6 + spread * (0.45 + 0.35 * allocation))
        _ball(leaves, *end, leaf_radius)

    root_count = 3 + round((1 - allocation) * 5)
    root_length = size * (0.07 + (1 - allocation) * 0.10) * (1 + 0.25 * (1 - environment.water_level))
    water_bias = np.array((
        0.0,
        environment.water_direction_y * size * 0.08,
        environment.water_direction_x * size * 0.08,
    ))
    for index in range(root_count):
        angle = family_angle + index * 2 * np.pi / root_count
        end = center + np.array((min(1.0, size - 1 - center[0]), math.sin(angle) * root_length, math.cos(angle) * root_length)) + water_bias
        _segment(roots, tuple(center), tuple(end), max(0.5, thickness * 0.48))

    occupancy = trunk | branches | leaves | roots
    context = make_environment_context(environment, size)
    obstacles = context[ENVIRONMENT_CHANNELS.index("obstacles")].numpy() > 0.5
    neighbors = context[ENVIRONMENT_CHANNELS.index("neighbor_occupancy")].numpy() > 0.5
    obstacles[max(0, int(center[0]) - 1) :, max(0, int(center[1]) - 1) : int(center[1]) + 2, max(0, int(center[2]) - 1) : int(center[2]) + 2] = False
    neighbors[max(0, int(center[0]) - 1) :, max(0, int(center[1]) - 1) : int(center[1]) + 2, max(0, int(center[2]) - 1) : int(center[2]) + 2] = False
    occupancy &= ~obstacles & ~neighbors
    occupancy = _seed_component(occupancy, tuple(np.rint(center).astype(int)))
    trunk &= occupancy
    branches &= occupancy
    leaves &= occupancy
    roots &= occupancy
    materials = np.zeros_like(occupancy, dtype=np.int64)
    materials[trunk | roots] = 1
    materials[branches] = 2
    materials[leaves] = 3
    return occupancy.astype(np.float32), materials
