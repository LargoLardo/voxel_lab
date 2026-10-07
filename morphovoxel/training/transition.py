"""Fixed rehearsal coverage for live-transition training."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch

from ..environment import EnvironmentSpec
from ..genomes import TreeGenome
from ..seeding import seed_state
from .family import FamilyData, curriculum_values, family_style_seeds
from .state_pool import StatePool


def replay_recipe(payload: dict, fallback: dict, _seen: frozenset = frozenset()) -> dict:
    """Keep the input checkpoint's coverage when the edit schedule starts over."""
    source = payload.get("config") or fallback
    previous = copy.deepcopy(source.get("transition_replay"))
    if previous is None and source.get("family_curriculum") in {"transition", "gene_transition"}:
        origin = source.get("initialize_from_checkpoint") or source.get("initialize_from_specialist")
        if origin and Path(origin).is_file() and str(Path(origin).resolve()) not in _seen:
            previous = replay_recipe(torch.load(origin, map_location="cpu", weights_only=False), fallback,
                                     _seen | {str(Path(origin).resolve())})
        elif origin:
            logging.getLogger(__name__).warning("Original transition checkpoint %s unavailable; estimating rehearsal coverage from saved curriculum", origin)
    kind = (payload.get("metadata") or source).get("model_kind")
    if kind == "tree_specialist":
        genome = TreeGenome.from_dict(source.get("tree_genome", {}))
        return {
            "genome_span": 0., "background_span": 0., "style_random_fraction": 0.,
            "style_seeds": [genome.style_seed], "neutral_genome": genome.to_dict(),
            "environment": source.get("environment", {}),
        }
    budget = int(source.get("family_curriculum_iterations", source.get("iterations", 1)))
    step = int(payload.get("step", budget)) - int(source.get("family_curriculum_start_step", 0)) - 1
    values = curriculum_values(max(0, step), budget, source)
    recipe = {
        "genome_span": values["genome_span"],
        "background_span": values.get("background_span", values["genome_span"]),
        "style_random_fraction": values.get("style_random_fraction", 1.),
        "style_seeds": list(family_style_seeds(source)),
    }
    if previous:
        # A handoff retains the older base as well as newly learned variation.
        # True resume uses the stored recipe directly in the trainer.
        for name in ("genome_span", "background_span", "style_random_fraction"):
            previous[name] = max(previous[name], recipe[name])
        if recipe["genome_span"]:
            previous["style_seeds"] = sorted(set(previous["style_seeds"] + recipe["style_seeds"]))
        return previous
    return recipe


def replay_sampling_options(recipe: dict, *, neutral: bool) -> dict:
    return {
        "genome_span": 0. if neutral else recipe["genome_span"],
        "background_span": 0. if neutral else recipe["background_span"],
        "style_random_fraction": 0. if neutral else recipe["style_random_fraction"],
        "style_seeds": recipe["style_seeds"],
        "neutral_fraction": 1. if neutral or not recipe["genome_span"] else 0.,
        "random_gene_values": True,
        "neutral_genome": TreeGenome.from_dict(recipe["neutral_genome"]) if recipe.get("neutral_genome") else None,
        "fixed_environment": EnvironmentSpec.from_dict(recipe.get("environment", {})),
    }


def family_pool(data: FamilyData, size: int, layout, config: dict, seed: int) -> StatePool:
    states = seed_state(
        len(data.genomes) // 2, size, layout, dimensions=3,
        seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)),
        random_seed=seed, device="cpu",
    ).repeat_interleave(2, 0)
    return StatePool(
        states, data.model_genomes, target_occupancy=data.target_occupancy,
        target_materials=data.target_materials, environments=data.environments,
        environment_specs=data.environment_vectors, style_seeds=data.style_seeds,
        condition_ids=data.condition_ids, pair_ids=data.pair_ids,
        target_distances=data.target_distances,
    )


def retention_panel(recipe: dict, families, config: dict):
    """Fixed bases plus one varied pair per gene; distribute fire seeds to limit cost."""
    from ..genomes import FAMILY_GENE_NAMES
    from ..validation import ValidationCase
    from .family import sample_counterfactual_family_data

    fire_seeds = tuple(map(int, config.get("validation_fire_seeds", [100_000, 100_001])))
    if not fire_seeds:
        raise ValueError("transition retention requires at least one validation fire seed")
    base = TreeGenome.from_dict(recipe.get("neutral_genome", {}))
    entries = [
        ("neutral", TreeGenome(family=family, genes=base.genes, style_seed=style))
        for family in families for style in recipe["style_seeds"]
    ]
    if recipe["genome_span"]:
        data = sample_counterfactual_family_data(
            len(families) * len(FAMILY_GENE_NAMES), int(config.get("world_size", 16)),
            int(config.get("seed", 0)) + 40_000, families=families,
            **replay_sampling_options(recipe, neutral=False),
            minimum_branch_voxels=int(config.get("minimum_branch_voxels", 1)),
            minimum_leaf_voxels=int(config.get("minimum_leaf_voxels", 1)),
        )
        entries.extend(("variation", genome) for genome in data.genomes)
    environment = EnvironmentSpec.from_dict(recipe.get("environment", {}))
    return tuple(
        ValidationCase(f"retention-{index}", category, genome, environment, fire_seeds[index % len(fire_seeds)])
        for index, (category, genome) in enumerate(entries)
    )


def retention_metrics(report) -> dict:
    """Separate tree families and bases/variations so gains cannot hide forgetting."""
    groups = {}
    for trial in report.trials:
        key = f"{trial.case.genome.family}/{trial.case.category}"
        groups.setdefault(key, []).append(trial.metrics)
    return {
        key: {**{name: sum(row[name] for row in rows) / len(rows)
                 for name in ("target_iou", "material_accuracy", "late_drift")},
              "finite_state": min(row["finite_state"] for row in rows)}
        for key, rows in groups.items()
    }


def retention_failures(baseline: dict, current: dict, tolerance: float) -> list[str]:
    failures = []
    for group, original in baseline.items():
        candidate = current.get(group)
        if candidate is None or not candidate["finite_state"]:
            failures.append(f"{group}: missing or non-finite state")
            continue
        for metric in ("target_iou", "material_accuracy", "late_drift"):
            regression = ((candidate[metric] - original[metric]) if metric == "late_drift"
                          else (original[metric] - candidate[metric]))
            if regression > tolerance:
                failures.append(f"{group}: {metric} regressed by {regression:.4f}")
    return failures


def transition_rank(report) -> tuple[float, float]:
    """Break strict-score ties with actual destination and edited-voxel quality."""
    destination = sum(trial.metrics["target_iou"] for trial in report.trials) / len(report.trials)
    edits = [trial.metrics["transition_edit_accuracy"] for trial in report.trials
             if trial.metrics.get("transition_edited_voxels", 0) > 0]
    return report.score, min(destination, sum(edits) / len(edits)) if edits else destination
