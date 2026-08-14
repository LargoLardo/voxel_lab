"""Continuous tree-family curriculum helpers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..environment import EnvironmentSpec, environment_context_batch
from ..genomes import FAMILY_GENE_NAMES, TREE_FAMILIES, TREE_GENE_SPECS, TreeGenome, tree_genome_tensor
from ..targets import make_tree_target


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


def _pack_family_data(
    genomes: list[TreeGenome],
    environments: list[EnvironmentSpec],
    methods: list[str],
    size: int,
    device: torch.device | str,
    *,
    condition_ids: Sequence[int] | None = None,
    pair_ids: Sequence[int] | None = None,
) -> FamilyData:
    targets = [make_tree_target(genome, size, environment) for genome, environment in zip(genomes, environments)]
    empty = [index for index, (occupancy, _) in enumerate(targets) if not np.asarray(occupancy).any()]
    if empty:
        raise RuntimeError(f"tree target invariant violated: occupancy is empty for sample indices {', '.join(map(str, empty))}")
    occupancy, materials = zip(*targets)
    count = len(genomes)
    return FamilyData(
        genomes=genomes,
        environment_specs=environments,
        model_genomes=tree_genome_tensor(genomes, device=device),
        target_occupancy=torch.as_tensor(np.stack(occupancy), device=device),
        target_materials=torch.as_tensor(np.stack(materials), dtype=torch.long, device=device),
        environments=environment_context_batch(environments, size, device=device),
        environment_vectors=torch.stack([environment.vector() for environment in environments]).to(device),
        style_seeds=torch.tensor([genome.style_seed for genome in genomes], dtype=torch.long, device=device),
        creation_methods=methods,
        condition_ids=torch.tensor(condition_ids if condition_ids is not None else [-1] * count, dtype=torch.long, device=device),
        pair_ids=torch.tensor(pair_ids if pair_ids is not None else range(count), dtype=torch.long, device=device),
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


def curriculum_values(step: int, iterations: int, config: dict) -> dict[str, float]:
    """Widen genes first, then interpolation, mutation, and environments."""
    progress = min(1.0, max(0.0, (step + 1) / max(1, iterations)))
    initial_span = float(config.get("initial_genome_span", 0.15))
    span = initial_span + (1 - initial_span) * min(1.0, progress / float(config.get("genome_widen_fraction", 0.45)))
    interpolation_start = float(config.get("interpolation_start_fraction", 0.25))
    mutation_start = float(config.get("mutation_start_fraction", 0.45))
    environment_start = float(config.get("environment_start_fraction", 0.65))
    interpolation = float(config.get("interpolation_fraction", 0.25)) if progress >= interpolation_start else 0.0
    mutation = float(config.get("mutation_fraction", 0.25)) if progress >= mutation_start else 0.0
    environment_span = 0.0 if progress < environment_start else min(1.0, (progress - environment_start) / max(1e-6, 1 - environment_start))
    return {
        "progress": progress,
        "genome_span": min(1.0, max(0.0, span)),
        "interpolation_fraction": min(1.0, max(0.0, interpolation)),
        "mutation_fraction": min(1.0, max(0.0, mutation)),
        "environment_span": environment_span,
    }


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
    return _pack_family_data(genomes, environments, methods, size, device)


def sample_counterfactual_family_data(
    pair_count: int,
    size: int,
    seed: int,
    *,
    genome_span: float = 1.0,
    environment_span: float = 0.0,
    active_gene_names: Sequence[str] = FAMILY_GENE_NAMES,
    condition_ids: Sequence[int] | None = None,
    pair_id_start: int = 0,
    device: torch.device | str = "cpu",
) -> FamilyData:
    """Create adjacent pairs differing in exactly one controlled gene."""
    if pair_count < 1 or not 0 < genome_span <= 1:
        raise ValueError("pair_count must be positive and genome_span within (0, 1]")
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
    locked = {spec.name for spec in TREE_GENE_SPECS if spec.name not in names}
    for pair_index, condition in enumerate(chosen):
        family = TREE_FAMILIES[condition // len(names)]
        gene_name = names[condition % len(names)]
        sample_seed = int(rng.integers(0, 2**31))
        base = TreeGenome.random(sample_seed, family=family, span=genome_span, locked=locked)
        low = base.with_values({gene_name: -genome_span})
        high = base.with_values({gene_name: genome_span})
        environment = EnvironmentSpec.random(int(rng.integers(0, 2**31)), span=environment_span) if environment_span else EnvironmentSpec()
        genomes.extend((low, high))
        environments.extend((environment, environment))
        methods.extend((f"counterfactual:{gene_name}:low", f"counterfactual:{gene_name}:high"))
        item_conditions.extend((condition, condition))
        pair_ids.extend((pair_id_start + pair_index, pair_id_start + pair_index))
    return _pack_family_data(
        genomes, environments, methods, size, device,
        condition_ids=item_conditions, pair_ids=pair_ids,
    )
