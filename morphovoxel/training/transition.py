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
