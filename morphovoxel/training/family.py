"""Continuous tree-family curriculum helpers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from ..environment import EnvironmentSpec, make_environment_context
from ..genomes import FAMILY_GENE_NAMES, TREE_FAMILIES, TREE_GENE_SPECS, TreeGenome, tree_genome_from_vector, tree_genome_tensor
from ..targets import make_tree_target
from .losses import _distance_field


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _cached_tree_target(genome: TreeGenome, size: int, environment: EnvironmentSpec):
    # Private CPU cache: packing copies these arrays before callers can edit them.
    occupancy, materials = make_tree_target(genome, size, environment)
    # Neutral examples recur. New varied targets are cheaper to prepare together
    # on the training device in _pack_family_data, then retain in the state pool.
    distance = (_distance_field(torch.from_numpy(occupancy)[None])[0]
                if not any(genome.genes) and environment == EnvironmentSpec() else None)
    return occupancy, materials, distance


@lru_cache(maxsize=8)
@torch.inference_mode(False)
def _cached_environment(environment: EnvironmentSpec, size: int) -> torch.Tensor:
    return make_environment_context(environment, size)


@dataclass
class FamilyData:
    genomes: list[TreeGenome]
    environment_specs: list[EnvironmentSpec]
    model_genomes: torch.Tensor
    target_occupancy: torch.Tensor
    target_materials: torch.Tensor
    environments: torch.Tensor
    environment_vectors: torch.Tensor
    style_seeds: torch.Tensor
    creation_methods: list[str]
    condition_ids: torch.Tensor
    pair_ids: torch.Tensor
    target_distances: torch.Tensor


def _pack_family_data(
    genomes: list[TreeGenome],
    environments: list[EnvironmentSpec],
    methods: list[str],
    size: int,
    device: torch.device | str,
    *,
    condition_ids: Sequence[int] | None = None,
    pair_ids: Sequence[int] | None = None,
    targets: Sequence[tuple[np.ndarray, np.ndarray, torch.Tensor | None]] | None = None,
    minimum_branch_voxels: int = 1,
    minimum_leaf_voxels: int = 1,
) -> FamilyData:
    if targets is None:
        targets = [_cached_tree_target(genome, size, environment) for genome, environment in zip(genomes, environments)]
    if len(targets) != len(genomes):
        raise ValueError("each genome must have one target")
    empty = [index for index, (occupancy, _, _) in enumerate(targets) if not np.asarray(occupancy).any()]
    if empty:
        raise RuntimeError(f"tree target invariant violated: occupancy is empty for sample indices {', '.join(map(str, empty))}")
    for material_index, name, minimum in (
        (2, "branch", minimum_branch_voxels), (3, "leaf", minimum_leaf_voxels),
    ):
        counts = [int(np.count_nonzero(materials == material_index)) for _, materials, _ in targets]
        undersized = [index for index, count in enumerate(counts) if 0 < count < minimum]
        if undersized:
            raise RuntimeError(
                f"tree target invariant violated: positive {name} masks need at least {minimum} voxels "
                f"for sample indices {', '.join(map(str, undersized))}"
            )
    occupancy, materials, distances = zip(*targets)
    target_occupancy = torch.as_tensor(np.stack(occupancy), device=device)
    target_distances = (torch.stack(distances).to(device) if all(value is not None for value in distances)
                        else _distance_field(target_occupancy))
    count = len(genomes)
    return FamilyData(
        genomes=genomes,
        environment_specs=environments,
        model_genomes=tree_genome_tensor(genomes, device=device),
        target_occupancy=target_occupancy,
        target_materials=torch.as_tensor(np.stack(materials), dtype=torch.long, device=device),
        environments=torch.stack([_cached_environment(environment, size) for environment in environments]).to(device),
        # Metadata contains exact seeds in float64; Metal cannot store this dtype.
        environment_vectors=torch.stack([environment.vector() for environment in environments]),
        style_seeds=torch.tensor([genome.style_seed for genome in genomes], dtype=torch.long, device=device),
        creation_methods=methods,
        condition_ids=torch.tensor(condition_ids if condition_ids is not None else [-1] * count, dtype=torch.long, device=device),
        pair_ids=torch.tensor(pair_ids if pair_ids is not None else range(count), dtype=torch.long, device=device),
        target_distances=target_distances,
    )


def _reflect_mutation(genome: TreeGenome, strength: float, seed: int) -> TreeGenome:
    """Mutate without accumulating probability mass at clipped boundaries."""
    rng = np.random.default_rng(seed)
    genes = []
    for spec, current in zip(TREE_GENE_SPECS, genome.genes):
        width = spec.maximum - spec.minimum
        offset = (current + float(rng.normal(0, strength)) - spec.minimum) % (2 * width)
        genes.append(spec.minimum + (offset if offset <= width else 2 * width - offset))
    return TreeGenome(genome.family, tuple(genes), genome.style_seed)


def curriculum_values(step: int, iterations: int, config: dict) -> dict[str, float | str]:
    """Use the selected Phase 2 schedule; preserve later stages' existing recipe."""
    mode = config.get("family_curriculum")
    stage = "variation"
    if mode is not None:
        if mode not in {"full", "basics", "variation", "transition"}:
            raise ValueError("family_curriculum must be full, basics, variation, or transition")
        fraction = float(config.get("basic_family_fraction", 0.25))
        if not 0 < fraction < 1:
            raise ValueError("basic_family_fraction must be between zero and one")
        if mode == "full" and iterations < 2:
            raise ValueError("the full family curriculum requires at least two iterations")
        basic_steps = min(iterations - 1, max(1, int(iterations * fraction))) if mode == "full" else 0
        if mode == "transition":
            stage = "transition"
        elif mode == "basics" or (mode == "full" and step < basic_steps):
            stage = "basics"
        elif mode == "full":
            step, iterations = step - basic_steps, iterations - basic_steps
    progress = min(1.0, max(0.0, (step + 1) / max(1, iterations)))
    initial_span = float(config.get("initial_genome_span", 0.15))
    widen_fraction = float(config.get("genome_widen_fraction", 0.45))
    if not 0 < initial_span <= 1 or not 0 < widen_fraction <= 1:
        raise ValueError("initial_genome_span and genome_widen_fraction must be in (0, 1]")
    span = initial_span + (1 - initial_span) * min(1.0, progress / widen_fraction)
    interpolation_start = float(config.get("interpolation_start_fraction", 0.25))
    mutation_start = float(config.get("mutation_start_fraction", 0.45))
    environment_start = float(config.get("environment_start_fraction", 0.65))
    interpolation = float(config.get("interpolation_fraction", 0.25)) if progress >= interpolation_start else 0.0
    mutation = float(config.get("mutation_fraction", 0.25)) if progress >= mutation_start else 0.0
    environment_span = 0.0 if progress < environment_start else min(1.0, (progress - environment_start) / max(1e-6, 1 - environment_start))
    values = {
        "progress": progress,
        "genome_span": min(1.0, max(0.0, span)),
        "interpolation_fraction": min(1.0, max(0.0, interpolation)),
        "mutation_fraction": min(1.0, max(0.0, mutation)),
        "environment_span": environment_span,
    }
    if mode is not None:
        neutral_fraction = float(config.get("neutral_fraction", 0.25))
        combination_start = float(config.get("combination_start_fraction", 0.25))
        if not 0 < neutral_fraction < 1 or not 0 <= combination_start < 1:
            raise ValueError("neutral_fraction must be in (0, 1) and combination_start_fraction in [0, 1)")
        diversity = min(1.0, max(0.0, (progress - combination_start) / min(widen_fraction, 1 - combination_start)))
        values.update(
            curriculum_stage=stage,
            genome_span=0.0 if stage in {"basics", "transition"} else span,
            background_span=0.0 if stage in {"basics", "transition"} else span * diversity,
            style_random_fraction=0.0 if stage in {"basics", "transition"} else diversity,
            neutral_fraction=1.0 if stage in {"basics", "transition"} else neutral_fraction,
            environment_span=0.0,
            interpolation_fraction=0.0,
            mutation_fraction=0.0,
        )
    return values


def family_style_seeds(config: dict) -> tuple[int, ...]:
    # Quarter-turn phases: adjacent integer seeds rasterize to the same tree.
    seeds = config.get("family_style_seeds", [0, 970806, 1941611, 2912417])
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ValueError("family_style_seeds must be a nonempty list of integer seeds")
    for seed in seeds:
        TreeGenome(style_seed=seed)  # Reuse the genome's seed validation.
    return tuple(dict.fromkeys(seeds))


def curriculum_sampling_options(values: dict, config: dict) -> dict:
    """Keep initial pool creation and subsequent replacements on the same recipe."""
    if "curriculum_stage" not in values:
        return {}
    return {
        name: values[name]
        for name in ("background_span", "style_random_fraction", "neutral_fraction")
    } | {"style_seeds": family_style_seeds(config)}


def sample_transition_destinations(
    genomes: torch.Tensor, style_seeds: torch.Tensor, size: int, seed: int,
    *, minimum_branch_voxels: int = 1, minimum_leaf_voxels: int = 1,
) -> FamilyData:
    """Change only family identity; preserve the organism's genes and style."""
    rng = np.random.default_rng(seed)
    sources = [tree_genome_from_vector(vector, style) for vector, style in zip(genomes.cpu(), style_seeds.cpu().tolist())]
    destinations = [
        replace(source, family=str(rng.choice([family for family in TREE_FAMILIES if family != source.family])))
        for source in sources
    ]
    return _pack_family_data(
        destinations, [EnvironmentSpec()] * len(destinations), ["transition"] * len(destinations),
        size, genomes.device, minimum_branch_voxels=minimum_branch_voxels,
        minimum_leaf_voxels=minimum_leaf_voxels,
    )


def sample_family_data(
    count: int,
    size: int,
    seed: int,
    *,
    genome_span: float = 1.0,
    interpolation_fraction: float = 0.25,
    mutation_fraction: float = 0.25,
    environment_span: float = 0.0,
    mutation_strength: float = 0.15,
    parent: TreeGenome | None = None,
    families: Sequence[str] | None = None,
    train_light_tropism: bool = False,
    minimum_branch_voxels: int = 1,
    minimum_leaf_voxels: int = 1,
    device: torch.device | str = "cpu",
) -> FamilyData:
    """Sample paired genomes, targets, environments, and exact style seeds."""
    if count < 1:
        raise ValueError("family sample count must be positive")
    if interpolation_fraction + mutation_fraction > 1:
        raise ValueError("interpolation and mutation fractions cannot exceed one in total")
    if families is not None and (len(families) != count or any(family not in TREE_FAMILIES for family in families)):
        raise ValueError("families must provide one valid tree family per sample")
    rng = np.random.default_rng(seed)
    genomes: list[TreeGenome] = []
    environments: list[EnvironmentSpec] = []
    methods: list[str] = []
    family_order = list(rng.permutation(TREE_FAMILIES))
    family_schedule = [str(family_order[index % len(TREE_FAMILIES)]) for index in range(count)]
    rng.shuffle(family_schedule)
    for index in range(count):
        sample_seed = int(rng.integers(0, 2**31))
        choice = float(rng.random())
        if families is not None:
            family = families[index]
        elif parent is None:
            family = family_schedule[index]
        elif choice < interpolation_fraction + mutation_fraction:
            family = parent.family
        else:
            family = TREE_FAMILIES[int(rng.integers(len(TREE_FAMILIES)))]
        if choice < interpolation_fraction:
            left = parent or TreeGenome.random(int(rng.integers(0, 2**31)), family=family, span=genome_span)
            right = TreeGenome.random(int(rng.integers(0, 2**31)), family=family, span=genome_span)
            genome = left.interpolate(right, float(rng.random()))
            methods.append("interpolation")
        elif choice < interpolation_fraction + mutation_fraction:
            base = parent or TreeGenome.random(int(rng.integers(0, 2**31)), family=family, span=genome_span)
            genome = (
                _reflect_mutation(base, min(1.0, mutation_strength), sample_seed)
                if parent is not None else base.mutate(min(1.0, mutation_strength), sample_seed)
            )
            methods.append("mutation")
        else:
            genome = TreeGenome.random(sample_seed, family=family, span=genome_span)
            methods.append("random")
        if not train_light_tropism:
            genome = genome.with_values({"light_tropism": 0.0})
        environment = EnvironmentSpec.random(int(rng.integers(0, 2**31)), span=environment_span) if environment_span else EnvironmentSpec()
        genomes.append(genome)
        environments.append(environment)
    return _pack_family_data(
        genomes, environments, methods, size, device,
        minimum_branch_voxels=minimum_branch_voxels,
        minimum_leaf_voxels=minimum_leaf_voxels,
    )


def sample_counterfactual_family_data(
    pair_count: int,
    size: int,
    seed: int,
    *,
    genome_span: float = 1.0,
    background_span: float | None = None,
    style_seeds: Sequence[int] | None = None,
    style_random_fraction: float = 1.0,
    neutral_fraction: float = 0.0,
    environment_span: float = 0.0,
    active_gene_names: Sequence[str] = FAMILY_GENE_NAMES,
    condition_ids: Sequence[int] | None = None,
    pair_id_start: int = 0,
    minimum_branch_voxels: int = 1,
    minimum_leaf_voxels: int = 1,
    device: torch.device | str = "cpu",
) -> FamilyData:
    """Create adjacent pairs differing in exactly one controlled gene."""
    if pair_count < 1 or not 0 <= genome_span <= 1:
        raise ValueError("pair_count must be positive and genome_span within [0, 1]")
    background_span = genome_span if background_span is None else background_span
    if not 0 <= background_span <= genome_span or not 0 <= style_random_fraction <= 1 or not 0 <= neutral_fraction <= 1:
        raise ValueError("background span and sampling fractions are out of range")
    if style_seeds is not None:
        style_seeds = family_style_seeds({"family_style_seeds": list(style_seeds)})
    specs = {spec.name: spec for spec in TREE_GENE_SPECS}
    names = tuple(active_gene_names)
    if not names or len(set(names)) != len(names) or set(names) - set(specs):
        raise ValueError("active_gene_names must contain unique known genes")
    total_conditions = len(TREE_FAMILIES) * len(names)
    chosen = list(condition_ids) if condition_ids is not None else [index % total_conditions for index in range(pair_count)]
    if len(chosen) != pair_count or any(not 0 <= value < total_conditions for value in chosen):
        raise ValueError("condition_ids must provide one in-range condition per pair")
    rng = np.random.default_rng(seed)
    genomes: list[TreeGenome] = []
    environments: list[EnvironmentSpec] = []
    methods: list[str] = []
    item_conditions: list[int] = []
    pair_ids: list[int] = []
    targets: list[tuple[np.ndarray, np.ndarray, torch.Tensor | None]] = []
    locked = {spec.name for spec in TREE_GENE_SPECS if spec.name not in names}
    for pair_index, condition in enumerate(chosen):
        family = TREE_FAMILIES[condition // len(names)]
        gene_name = names[condition % len(names)]
        neutral = genome_span == 0 or (neutral_fraction > 0 and rng.random() < neutral_fraction)
        for _ in range(128):
            sample_seed = int(rng.integers(0, 2**31))
            base = TreeGenome.random(sample_seed, family=family, span=0.0 if neutral else background_span, locked=locked)
            if style_seeds is not None and (neutral or rng.random() >= style_random_fraction):
                base = replace(base, style_seed=int(rng.choice(style_seeds)))
            low = base if neutral else base.with_values({gene_name: -genome_span})
            high = base if neutral else base.with_values({gene_name: genome_span})
            environment = EnvironmentSpec.random(int(rng.integers(0, 2**31)), span=environment_span) if environment_span else EnvironmentSpec()
            low_target = _cached_tree_target(low, size, environment)
            pair_targets = (low_target, low_target if neutral else _cached_tree_target(high, size, environment))
            counts = [
                (int(np.count_nonzero(materials == 2)), int(np.count_nonzero(materials == 3)))
                for _, materials, _ in pair_targets
            ]
            if all(
                (branches == 0 or branches >= minimum_branch_voxels)
                and (leaves == 0 or leaves >= minimum_leaf_voxels)
                for branches, leaves in counts
            ):
                break
        else:
            raise RuntimeError(
                f"could not generate a {family}/{gene_name} counterfactual pair with minimum "
                f"branch={minimum_branch_voxels} and leaf={minimum_leaf_voxels} voxel counts"
            )
        genomes.extend((low, high))
        targets.extend(pair_targets)
        environments.extend((environment, environment))
        methods.extend(("neutral", "neutral") if neutral else (f"counterfactual:{gene_name}:low", f"counterfactual:{gene_name}:high"))
        item_conditions.extend((condition, condition))
        pair_ids.extend((pair_id_start + pair_index, pair_id_start + pair_index))
    return _pack_family_data(
        genomes, environments, methods, size, device,
        condition_ids=item_conditions, pair_ids=pair_ids, targets=targets,
        minimum_branch_voxels=minimum_branch_voxels,
        minimum_leaf_voxels=minimum_leaf_voxels,
    )
