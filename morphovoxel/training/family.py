"""Continuous tree-family curriculum helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..environment import EnvironmentSpec, environment_context_batch
from ..genomes import TREE_FAMILIES, TreeGenome, tree_genome_tensor
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
    device: torch.device | str = "cpu",
) -> FamilyData:
    """Sample paired genomes, targets, environments, and exact style seeds."""
    if count < 1:
        raise ValueError("family sample count must be positive")
    if interpolation_fraction + mutation_fraction > 1:
        raise ValueError("interpolation and mutation fractions cannot exceed one in total")
    rng = np.random.default_rng(seed)
    genomes: list[TreeGenome] = []
    environments: list[EnvironmentSpec] = []
    methods: list[str] = []
    for _ in range(count):
        sample_seed = int(rng.integers(0, 2**31))
        choice = float(rng.random())
        family = parent.family if parent is not None and choice < interpolation_fraction + mutation_fraction else (
            TREE_FAMILIES[0] if genome_span < 0.35 else TREE_FAMILIES[int(rng.integers(len(TREE_FAMILIES)))]
        )
        if choice < interpolation_fraction:
            left = parent or TreeGenome.random(int(rng.integers(0, 2**31)), family=family, span=genome_span)
            right = TreeGenome.random(int(rng.integers(0, 2**31)), family=family, span=genome_span)
            genome = left.interpolate(right, float(rng.random()))
            methods.append("interpolation")
        elif choice < interpolation_fraction + mutation_fraction:
            base = parent or TreeGenome.random(int(rng.integers(0, 2**31)), family=family, span=genome_span)
            genome = base.mutate(min(1.0, mutation_strength), sample_seed)
            methods.append("mutation")
        else:
            genome = TreeGenome.random(sample_seed, family=family, span=genome_span)
            methods.append("random")
        environment = EnvironmentSpec.random(int(rng.integers(0, 2**31)), span=environment_span) if environment_span else EnvironmentSpec()
        genomes.append(genome)
        environments.append(environment)
    occupancy, materials = zip(*(make_tree_target(genome, size, environment) for genome, environment in zip(genomes, environments)))
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
    )
