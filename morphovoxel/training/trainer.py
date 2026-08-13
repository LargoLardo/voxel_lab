"""Compact shared trainer for all growth phases."""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..checkpointing import CHECKPOINT_FORMAT_VERSION, convert_specialist_to_family, load_checkpoint, save_checkpoint
from ..config import save_config
from ..environment import ENVIRONMENT_CHANNELS, ENVIRONMENT_SCHEMA_VERSION, EnvironmentSpec, environment_context_batch
from ..genomes import MORPHOLOGIES, TREE_FAMILIES, TREE_GENE_SPECS, TREE_GENOME_VERSION, TreeGenome, one_hot_genomes, tree_genome_tensor
from ..metrics import morphology_metrics, threshold_iou
from ..model_2d import NeuralCA2D
from ..model_3d import NeuralCA3D
from ..random_utils import resolve_device, seed_everything
from ..rollout import rollout
from ..seeding import seed_state
from ..state import StateLayout
from ..targets import make_target_2d, make_target_3d, make_tree_target
from ..targets.targets_3d import TREE_TARGET_VERSION
from ..targets.morphology_library import save_target
from ..utils import create_run_directory, metadata, steps_per_second, write_json, write_live_preview
from ..validation import ValidationCriteria, build_candidate_panel, build_validation_panel, validate_panel
from .losses import morphology_loss
from .family import curriculum_values, sample_family_data
from .state_pool import StatePool

LOGGER = logging.getLogger(__name__)


def _rng_byte_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(payload: dict | None) -> None:
    """Continue the exact stochastic streams stored by a true resume."""
    if not isinstance(payload, dict):
        return
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(_rng_byte_tensor(payload["torch"]))
    cuda_states = payload.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state count does not match available CUDA devices")
        torch.cuda.set_rng_state_all([_rng_byte_tensor(state) for state in cuda_states])


def _targets(dimensions: int, labels: torch.Tensor, size: int, seed: int, conditional: bool, device, target_kind: str | None = None):
    names_2d = ("branching", "radial", "elongated", "asymmetric")
    names = MORPHOLOGIES if dimensions == 3 else names_2d
    maker = make_target_3d if dimensions == 3 else make_target_2d
    occupancy, materials = zip(*(maker(names[int(label)] if conditional else (target_kind or names[0]), size, seed) for label in labels))
    return torch.as_tensor(np.stack(occupancy), device=device), torch.as_tensor(np.stack(materials), device=device, dtype=torch.long)


def _step_range(value, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        if value < 0:
            raise ValueError("step ranges must be non-negative and ordered")
        return value, value
    minimum, maximum = map(int, value)
    if minimum < 0 or maximum < minimum:
        raise ValueError("step ranges must be non-negative and ordered")
    return minimum, maximum


def _tree_settings(config: dict) -> tuple[TreeGenome, EnvironmentSpec]:
    genome_value = config.get("tree_genome", {})
    environment_value = config.get("environment", {})
    if not isinstance(genome_value, dict) or not isinstance(environment_value, dict):
        raise ValueError("tree_genome and environment must be mappings")
    return TreeGenome.from_dict(genome_value), EnvironmentSpec.from_dict(environment_value)


def _restore_pool(saved: dict) -> StatePool:
    optional = {
        name: saved.get(name)
        for name in (
            "target_occupancy", "target_materials", "environments",
            "environment_specs", "style_seeds",
        )
    }
    return StatePool(saved["states"], saved["genomes"], saved.get("ages"), **optional)


def _pool_actions(
    state: torch.Tensor,
    target: torch.Tensor,
    ages: torch.Tensor,
    fresh_count: int,
    damage_fraction: float,
    mature_age: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the worst samples to reseed and the best mature samples to damage."""
    living = state[:, 0].flatten(1).amax(1) > 0.1
    valid_target = target.flatten(1).amax(1) > 0.5
    errors = torch.nan_to_num(
        (state[:, 0] - target).square().flatten(1).mean(1),
        nan=float("inf"), posinf=float("inf"), neginf=float("inf"),
    )
    invalid = ~living | ~valid_target
    errors = errors.masked_fill(invalid, float("inf"))
    # A dead NCA has no living cells from which to regrow and receives no
    # useful gradient through the living mask, so it must be reseeded even if
    # the configured routine fresh fraction would choose fewer samples.
    fresh_count = min(len(state), max(0, fresh_count, int(invalid.sum())))
    reseed = torch.topk(errors, fresh_count).indices if fresh_count else torch.empty(0, dtype=torch.long, device=state.device)
    eligible = (ages >= mature_age) & living & valid_target
    if len(reseed):
        eligible[reseed] = False
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    damage_count = min(len(candidates), round(len(candidates) * max(0.0, min(1.0, damage_fraction))))
    if damage_fraction > 0 and len(candidates) and not damage_count:
        damage_count = 1
    damage = candidates[torch.argsort(errors[candidates])[:damage_count]] if damage_count else torch.empty(0, dtype=torch.long, device=state.device)
    return reseed, damage


def _keep_viable_damage(original: torch.Tensor, damaged: torch.Tensor) -> torch.Tensor:
    """Do not turn a regenerating sample into an unrecoverable all-dead state."""
    if original.shape != damaged.shape:
        raise ValueError("original and damaged states must have the same shape")
    living = damaged[:, 0].flatten(1).amax(1) > 0.1
    return torch.where(living.view(-1, *([1] * (damaged.ndim - 1))), damaged, original)


@torch.no_grad()
def _validate_persistence(
    model,
    *,
    dimensions: int,
    conditional: bool,
    size: int,
    layout: StateLayout,
    device: torch.device,
    target_seed: int,
    validation_seed: int,
    total_steps: int,
    start_step: int,
    interval: int,
    target_kind: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Return the lowest late-horizon IoU for every genome and across genomes."""
    count = len(MORPHOLOGIES) if conditional else 1
    labels = torch.arange(count, device=device) if conditional else torch.zeros(1, dtype=torch.long, device=device)
    genomes = one_hot_genomes(labels) if conditional else None
    targets, _ = _targets(dimensions, labels, size, target_seed, conditional, device, target_kind)
    names = MORPHOLOGIES if conditional else (target_kind or ("branching" if dimensions == 2 else MORPHOLOGIES[0]),)
    scores = {name: 1.0 for name in names}
    start_step = min(total_steps, max(1, start_step))
    interval = max(1, interval)
    checkpoints = list(range(start_step, total_steps + 1, interval))
    if not checkpoints or checkpoints[-1] != total_steps:
        checkpoints.append(total_steps)
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(validation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(validation_seed)
            state = seed_state(count, size, layout, dimensions=dimensions, device=device)
            elapsed = 0
            for checkpoint in checkpoints:
                state, _ = rollout(model, state, checkpoint - elapsed, genomes)
                elapsed = checkpoint
                if not bool(torch.isfinite(state).all()):
                    scores = {name: 0.0 for name in names}
                    break
                for index, name in enumerate(names):
                    scores[name] = min(scores[name], threshold_iou(state[index, layout.occupancy], targets[index]))
    finally:
        model.train(was_training)
    return min(scores.values()), scores


def train(config: dict, *, dimensions: int, conditional: bool = False) -> Path:
    """Train an NCA and persist a reproducible, visualizable run."""
    config = dict(config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    default_kind = "legacy_conditional" if conditional else "specialist"
    model_kind = str(config.get("model_kind", default_kind))
    if model_kind not in {"specialist", "tree_specialist", "legacy_conditional", "tree_family"}:
        raise ValueError("model_kind must be specialist, tree_specialist, legacy_conditional, or tree_family")
    tree_family = model_kind == "tree_family"
    tree_specialist = model_kind == "tree_specialist"
    if (tree_family or tree_specialist) and dimensions != 3:
        raise ValueError("tree models require three dimensions")
    if model_kind == "legacy_conditional" and not conditional:
        raise ValueError("legacy_conditional requires conditional training")
    config.setdefault("model_kind", model_kind)
    if tree_family or tree_specialist:
        config.setdefault("genome_schema_version", TREE_GENOME_VERSION)
        config.setdefault("environment_schema_version", ENVIRONMENT_SCHEMA_VERSION)
        config.setdefault("target_generator", {"name": "procedural_tree", "version": TREE_TARGET_VERSION})
    if tree_family:
        config.setdefault(
            "training_genome_ranges",
            {spec.name: [spec.minimum, spec.maximum] for spec in TREE_GENE_SPECS},
        )
        config.setdefault("validation_panel", {
            "categories": ["default", "boundary", "random", "interpolation", "mutation", "archive"],
            "fire_seeds": config.get("validation_fire_seeds", [seed + 100_000, seed + 100_001]),
            "steps": int(config.get("validation_steps", 256)),
        })
    tree_default, environment_default = _tree_settings(config) if tree_family or tree_specialist else (None, None)
    if tree_specialist:
        config.setdefault(
            "training_genome_ranges",
            {spec.name: [tree_default.value(spec.name), tree_default.value(spec.name)] for spec in TREE_GENE_SPECS},
        )
    size, batch = int(config.get("world_size", 32 if dimensions == 2 else 16)), int(config.get("batch_size", 2 if dimensions == 2 else 1))
    classes = 4 if dimensions == 3 else 3
    layout = StateLayout(materials=int(config.get("materials", classes)), hidden=int(config.get("hidden_channels", 8)))
    genome_size = TreeGenome.model_size() if tree_family else len(MORPHOLOGIES) if conditional else 0
    context_channels = len(ENVIRONMENT_CHANNELS) if bool(config.get("environment_conditioning", tree_family)) else 0
    model_class = NeuralCA3D if dimensions == 3 else NeuralCA2D
    model = model_class(
        layout.channels,
        int(config.get("model_width", 32)),
        genome_size,
        float(config.get("fire_rate", 0.5)),
        context_channels,
    ).to(device)
    initialize_from = config.get("initialize_from_specialist")
    if initialize_from:
        if not tree_family:
            raise ValueError("initialize_from_specialist is only valid for tree-family training")
        if config.get("resume"):
            raise ValueError("choose either resume or initialize_from_specialist, not both")
        source_context_channels = int(config.get("specialist_context_channels", 0))
        source_model = NeuralCA3D(
            layout.channels,
            int(config.get("model_width", 32)),
            0,
            float(config.get("fire_rate", 0.5)),
            source_context_channels,
        ).to(device)
        load_checkpoint(
            initialize_from,
            source_model,
            map_location=device,
            expected_model_kind="tree_specialist",
        )
        convert_specialist_to_family(source_model, model)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(str(config.get("optimizer", "adam")).lower())
    if optimizer_class is None:
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    optimizer = optimizer_class(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config["scheduler"]) if config.get("scheduler") else None
    start, restored = 0, None
    if config.get("resume"):
        restored = load_checkpoint(
            config["resume"],
            model,
            optimizer,
            scheduler,
            map_location=device,
            expected_model_kind=model_kind,
        )
        start = int(restored["step"])
        _restore_rng_state(restored.get("rng"))
    run = create_run_directory(str(config.get("run_name", f"phase{dimensions}d")), config.get("runs_root", "runs"))
    save_config(config, run / "config.yaml")
    run_metadata, started = metadata(seed, model, device), time.perf_counter()
    run_metadata.update({
        "model_kind": model_kind,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "genome_schema_version": TREE_GENOME_VERSION if tree_family or tree_specialist else (1 if conditional else 0),
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION if tree_family or tree_specialist or context_channels else 0,
        "target_generator_version": TREE_TARGET_VERSION if tree_family or tree_specialist else 0,
        "genome_size": genome_size,
        "context_channels": context_channels,
    })
    write_json(run / "metadata.json", run_metadata)
    records, validation_records = [], []
    iterations = int(config.get("iterations", 10))
    minimum, maximum = _step_range(config.get("rollout_steps"), (16, 32) if dimensions == 2 else (8, 16))
    persistence_value = config.get("persistence_steps", config.get("stability_steps", 0))
    persistence_minimum, persistence_maximum = _step_range(persistence_value, (0, 0))
    validation_steps = int(config.get("validation_steps", 256 if conditional else 0))
    validation_every = max(1, int(config.get("validation_every", max(1, iterations))))
    validation_start = int(config.get("validation_start", max(maximum, validation_steps // 4)))
    validation_interval = int(config.get("validation_interval", 32))
    state_limit = float(config.get("state_limit", 4.0))
    default_capture_every = max(1, int(maximum) // 12)
    capture_every = max(1, int(config.get("frame_every", default_capture_every)))
    live_preview = bool(config.get("live_preview", False))
    preview_image = None
    if live_preview:
        if dimensions == 2:
            from ..rendering_2d import occupancy_image as preview_image
        else:
            from ..rendering_3d import projection as preview_image
    pool = None
    configured_pool_size = int(config.get("pool_size", 0))
    if tree_family:
        requested_pool_size = configured_pool_size or max(batch, 64)
        pool_size = max(requested_pool_size, batch)
    elif conditional:
        requested_pool_size = configured_pool_size or max(batch, len(MORPHOLOGIES) * 4)
        pool_size = max(requested_pool_size, batch, len(MORPHOLOGIES))
    else:
        pool_size = max(configured_pool_size, batch) if configured_pool_size else 0
    if tree_family:
        initial = curriculum_values(0, iterations, config)
        family = sample_family_data(
            pool_size, size, seed,
            genome_span=initial["genome_span"],
            interpolation_fraction=initial["interpolation_fraction"],
            mutation_fraction=initial["mutation_fraction"],
            environment_span=initial["environment_span"] if context_channels else 0.0,
            mutation_strength=float(config.get("mutation_strength", 0.15)),
        )
        pool_states = seed_state(pool_size, size, layout, dimensions=dimensions, device="cpu")
        pool = StatePool(
            pool_states,
            family.model_genomes,
            target_occupancy=family.target_occupancy,
            target_materials=family.target_materials,
            environments=family.environments,
            environment_specs=family.environment_vectors,
            style_seeds=family.style_seeds,
        )
    elif pool_size:
        pool_labels = torch.arange(pool_size) % (len(MORPHOLOGIES) if conditional else 1)
        pool_genomes = one_hot_genomes(pool_labels) if conditional else pool_labels[:, None].float()
        pool_states = seed_state(pool_size, size, layout, dimensions=dimensions, device="cpu")
        pool = StatePool(pool_states, pool_genomes)
    if restored and restored.get("pool"):
        initialized_pool = pool
        pool = _restore_pool(restored["pool"])
        if initialized_pool is not None and len(pool.states) < len(initialized_pool.states):
            pool.append_from(initialized_pool, len(pool.states))
        if tree_family and any(getattr(pool, name) is None for name in (
            "target_occupancy", "target_materials", "environments",
            "environment_specs", "style_seeds",
        )):
            raise ValueError("tree-family checkpoint pool is missing paired target/environment/style data")
    specialist_target = specialist_material = specialist_context = None
    if tree_specialist:
        occupancy, materials = make_tree_target(tree_default, size, environment_default)
        specialist_target = torch.as_tensor(np.repeat(occupancy[None], batch, 0), device=device)
        specialist_material = torch.as_tensor(np.repeat(materials[None], batch, 0), dtype=torch.long, device=device)
        if context_channels:
            specialist_context = environment_context_batch([environment_default] * batch, size, device=device)
    final_state = final_target = final_materials = None
    best_score, last_validation = float("-inf"), None
    for step in range(start, start + iterations):
        context = None
        pool_batch = pool.sample(batch, torch.Generator().manual_seed(seed + step), device) if pool else None
        if pool_batch:
            state = pool_batch.states
            if tree_family:
                if any(value is None for value in (
                    pool_batch.target_occupancy, pool_batch.target_materials,
                    pool_batch.environments, pool_batch.style_seeds,
                )):
                    raise ValueError("tree-family pool batch lost its paired identity")
                genomes = pool_batch.genomes
                target = pool_batch.target_occupancy
                material = pool_batch.target_materials
                context = pool_batch.environments if context_channels else None
            elif tree_specialist:
                genomes = None
                target, material, context = specialist_target, specialist_material, specialist_context
            else:
                genomes = pool_batch.genomes if conditional else None
                labels = genomes.argmax(1) if conditional else torch.zeros(batch, dtype=torch.long, device=device)
                target, material = _targets(dimensions, labels, size, seed, conditional, device, config.get("target_kind"))
            fresh_fraction = max(0.0, min(1.0, float(config.get("fresh_fraction", 0.25))))
            expected_fresh = batch * fresh_fraction
            fresh = int(expected_fresh) + int(random.random() < expected_fresh % 1)
            reseed, damage = _pool_actions(
                state,
                target,
                pool_batch.ages,
                fresh,
                float(config.get("damage_probability", 0)) if dimensions == 3 else 0.0,
                int(config.get("damage_min_age", maximum)),
            )
            if len(reseed):
                fresh_states = seed_state(
                    len(reseed), size, layout, dimensions=dimensions,
                    seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)),
                    random_seed=seed + step, device=device,
                )
                state[reseed] = fresh_states
                pool_batch.ages[reseed] = 0
                if tree_family:
                    curriculum = curriculum_values(step - start, iterations, config)
                    replacement_families = [
                        TREE_FAMILIES[int(genomes[index, : len(TREE_FAMILIES)].argmax())]
                        for index in reseed.tolist()
                    ]
                    replacement = sample_family_data(
                        len(reseed), size, seed + 10_000 + step,
                        genome_span=curriculum["genome_span"],
                        interpolation_fraction=curriculum["interpolation_fraction"],
                        mutation_fraction=curriculum["mutation_fraction"],
                        environment_span=curriculum["environment_span"] if context_channels else 0.0,
                        mutation_strength=float(config.get("mutation_strength", 0.15)),
                        families=replacement_families,
                        device=device,
                    )
                    genomes[reseed] = replacement.model_genomes
                    target[reseed] = replacement.target_occupancy
                    material[reseed] = replacement.target_materials
                    if context is not None:
                        context[reseed] = replacement.environments
                    pool_batch.style_seeds[reseed] = replacement.style_seeds
                    if pool_batch.environment_specs is not None:
                        pool_batch.environment_specs[reseed] = replacement.environment_vectors
                    pool.replace_entries(
                        pool_batch.indices[reseed], states=fresh_states, genomes=replacement.model_genomes,
                        target_occupancy=replacement.target_occupancy,
                        target_materials=replacement.target_materials,
                        environments=replacement.environments,
                        environment_specs=replacement.environment_vectors,
                        style_seeds=replacement.style_seeds,
                    )
            if len(damage):
                from ..damage import damage_3d
                for index in damage.tolist():
                    original = state[index : index + 1]
                    damaged, _ = damage_3d(
                        original,
                        random.choice(config.get("damage_severities", [0.1, 0.25, 0.5])),
                        random.choice(config.get("damage_types", ["sphere", "cuboid", "top"])),
                        seed + step + index,
                    )
                    state[index : index + 1] = _keep_viable_damage(original, damaged)
        else:
            state = seed_state(batch, size, layout, dimensions=dimensions, seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)), random_seed=seed + step, device=device)
            if tree_family:
                raise RuntimeError("tree-family training requires its paired state pool")
            if tree_specialist:
                genomes = None
                target, material, context = specialist_target, specialist_material, specialist_context
            else:
                labels = torch.arange(step * batch, (step + 1) * batch, device=device) % (len(MORPHOLOGIES) if conditional else 1)
                genomes = one_hot_genomes(labels) if conditional else None
                target, material = _targets(dimensions, labels, size, seed, conditional, device, config.get("target_kind"))
        steps = random.randint(int(minimum), int(maximum))
        preview_started = time.perf_counter() if preview_image and (step + 1) % 10 == 0 else None
        final_state, _ = rollout(model, state, steps, genomes, context=context)
        loss, components = morphology_loss(
            final_state, target, material, layout, config.get("loss_weights"), state_limit=state_limit,
        )
        persistence_steps = random.randint(persistence_minimum, persistence_maximum)
        committed_state = final_state
        if persistence_steps:
            committed_state, _ = rollout(model, final_state, persistence_steps, genomes, context=context)
            persistence_loss, persistence_components = morphology_loss(
                committed_state, target, material, layout, config.get("loss_weights"), state_limit=state_limit,
            )
            components["persistence"] = persistence_loss
            components.update({f"persistence_{name}": value for name, value in persistence_components.items()})
            loss = loss + float(config.get("persistence_weight", 1.0)) * persistence_loss
        if preview_started is not None:
            total_steps = steps + persistence_steps
            write_live_preview(
                run / "visualizations" / "live.png", preview_image(committed_state[:1]),
                phase="training", iteration=step + 1, step=total_steps, total_steps=total_steps,
                steps_per_second=steps_per_second(total_steps, preview_started),
            )
        if not torch.isfinite(loss):
            save_checkpoint(run / "checkpoints" / "nan.pt", model, optimizer, step=step, config=config)
            raise FloatingPointError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
        optimizer.step()
        if scheduler:
            scheduler.step()
        if pool and pool_batch:
            pool.commit(pool_batch, committed_state, steps + persistence_steps)
        final_state = committed_state
        curriculum_record = curriculum_values(step - start, iterations, config) if tree_family else {}
        records.append({
            "step": step + 1,
            "loss": float(loss.detach()),
            **curriculum_record,
            **{name: float(value.detach()) for name, value in components.items()},
        })
        final_target, final_materials = target, material
        LOGGER.info("step=%d loss=%.6f", step + 1, float(loss.detach()))
        should_validate = validation_steps > 0 and ((step + 1) % validation_every == 0 or step + 1 == start + iterations)
        if should_validate:
            detailed_validation_rows: list[dict[str, object]] | None = None
            if tree_family or tree_specialist:
                fire_seeds = tuple(map(int, config.get("validation_fire_seeds", [seed + 100_000, seed + 100_001])))
                environment_values = config.get("validation_environment_specs")
                environments = (
                    tuple(EnvironmentSpec.from_dict(value) for value in environment_values)
                    if environment_values is not None else
                    ((environment_default,) if tree_specialist else None)
                )
                if not context_channels:
                    environments = (environment_default,)
                if tree_specialist:
                    panel = build_candidate_panel(
                        tree_default, seed=int(config.get("validation_seed", seed + 100_000)),
                        fire_seeds=fire_seeds, environments=environments,
                    )
                else:
                    archived_values = config.get("validation_archived_genomes", [])
                    panel = build_validation_panel(
                        seed=int(config.get("validation_seed", seed + 100_000)),
                        boundary_genes=config.get("validation_boundary_genes", ["height", "canopy_spread"]),
                        random_count=int(config.get("validation_random_count", 2)),
                        interpolation_steps=int(config.get("validation_interpolation_steps", 2)),
                        mutation_count=int(config.get("validation_mutation_count", 2)),
                        mutation_strength=float(config.get("validation_mutation_strength", 0.15)),
                        archived=tuple(TreeGenome.from_dict(value) for value in archived_values),
                        fire_seeds=fire_seeds, environments=environments,
                    )
                criteria = ValidationCriteria(
                    min_steps=int(config.get("validation_min_steps", 256)),
                    min_recovery_steps=int(config.get("validation_min_recovery_steps", 64)),
                    state_limit=state_limit,
                )
                report = validate_panel(
                    model, panel, layout=layout, world_size=size, steps=validation_steps,
                    recovery_steps=int(config.get("validation_recovery_steps", 64)),
                    seed_size=int(config.get("seed_size", 1)), device=device, criteria=criteria,
                    aggregation=str(config.get("validation_aggregation", "worst")),
                    low_percentile=float(config.get("validation_low_percentile", 0.1)),
                )
                worst_score = report.score
                genome_scores = {trial.case.case_id: trial.score for trial in report.trials}
                last_validation = {
                    "step": step + 1,
                    "validation_steps": validation_steps,
                    "validation_panel": [case.to_dict() for case in panel],
                    "persistence_report": report.to_dict(),
                    "worst_genome_persistence_score": report.worst_score,
                    "low_percentile_persistence_score": report.low_percentile_score,
                    "per_genome": genome_scores,
                }
                detailed_validation_rows = [
                    {
                        "step": step + 1,
                        "case": trial.case.case_id,
                        "category": trial.case.category,
                        "family": trial.case.genome.family,
                        "style_seed": trial.case.genome.style_seed,
                        "genome": json.dumps(trial.case.genome.to_dict(), sort_keys=True),
                        "environment": json.dumps(trial.case.environment.to_dict(), sort_keys=True),
                        "fire_seed": trial.case.fire_seed,
                        "validation_steps": trial.steps,
                        "recovery_steps": trial.recovery_steps,
                        "validated": trial.validated,
                        "accepted": trial.accepted,
                        "persistence_score": trial.score,
                        "worst_genome_persistence_score": report.worst_score,
                        "low_percentile_persistence_score": report.low_percentile_score,
                        "failure_reasons": ";".join(trial.failure_reasons),
                        **{f"metric_{name}": value for name, value in trial.metrics.items()},
                        **{f"descriptor_{name}": value for name, value in trial.descriptors.items()},
                    }
                    for trial in report.trials
                ]
            else:
                worst_score, genome_scores = _validate_persistence(
                    model,
                    dimensions=dimensions,
                    conditional=conditional,
                    size=size,
                    layout=layout,
                    device=device,
                    target_seed=seed,
                    validation_seed=int(config.get("validation_seed", seed + 100_000)),
                    total_steps=validation_steps,
                    start_step=validation_start,
                    interval=validation_interval,
                    target_kind=config.get("target_kind"),
                )
                last_validation = {
                    "step": step + 1,
                    "validation_steps": validation_steps,
                    "worst_genome_persistence_score": worst_score,
                    "per_genome": genome_scores,
                }
            if detailed_validation_rows is not None:
                validation_records.extend(detailed_validation_rows)
            else:
                validation_records.extend(
                    {"step": step + 1, "genome": name, "persistence_score": score, "worst_genome_persistence_score": worst_score}
                    for name, score in genome_scores.items()
                )
            # Strict persistence criteria often tie at zero early in training.
            # Keep the newest tied checkpoint instead of freezing best.pt at
            # the first validation window.
            if worst_score >= best_score:
                best_score = worst_score
                best_validation = {**last_validation, "best_worst_genome_persistence_score": best_score}
                save_checkpoint(
                    run / "checkpoints" / "best.pt", model, optimizer, step=step + 1,
                    scheduler=scheduler, config=config, pool=pool,
                    genomes=({"schema_version": 1, "default": tree_default.to_dict()} if tree_family else list(MORPHOLOGIES) if conditional else None),
                    validation=best_validation,
                )
            LOGGER.info("validation step=%d worst_genome_persistence=%.6f", step + 1, worst_score)
    checkpoint = run / "checkpoints" / "latest.pt"
    validation_summary = ({**last_validation, "best_worst_genome_persistence_score": best_score} if last_validation else None)
    save_checkpoint(
        checkpoint, model, optimizer, step=start + iterations, scheduler=scheduler, config=config, pool=pool,
        genomes=({"schema_version": 1, "default": tree_default.to_dict()} if tree_family else list(MORPHOLOGIES) if conditional else None),
        validation=validation_summary,
    )
    pd.DataFrame(records).to_csv(run / "logs.csv", index=False)
    pd.DataFrame(records).to_csv(run / "metrics" / "per_step.csv", index=False)
    if validation_records:
        pd.DataFrame(validation_records).to_csv(run / "metrics" / "persistence_validation.csv", index=False)
    assert final_state is not None and final_target is not None and final_materials is not None
    final_np = final_state.detach().cpu().numpy()
    np.save(run / "rollouts" / "final_state.npy", final_np)
    save_target(run / "targets", final_target[0].cpu().numpy(), final_materials[0].cpu().numpy())
    prediction = final_state[0, 0].clamp(0, 1).detach().cpu().numpy()
    summary = morphology_metrics(prediction, final_target[0].cpu().numpy())
    pd.DataFrame([summary]).to_csv(run / "metrics" / "summary.csv", index=False)
    capture = seed_state(1, size, layout, dimensions=dimensions, device=device)
    if tree_family:
        capture_genome = tree_genome_tensor([tree_default], device=device)
        capture_context = environment_context_batch([environment_default], size, device=device) if context_channels else None
    elif conditional:
        capture_genome = one_hot_genomes(torch.zeros(1, dtype=torch.long, device=device))
        capture_context = None
    else:
        capture_genome = None
        capture_context = specialist_context[:1] if specialist_context is not None else None
    with torch.no_grad():
        _, frames = rollout(model, capture, int(maximum), capture_genome, capture_every, context=capture_context)
    np.savez_compressed(run / "rollouts" / "states.npz", states=np.stack([frame.numpy() for frame in frames]))
    if dimensions == 2:
        from ..rendering_2d import save_comparison, save_gif, save_hidden_channels
    else:
        from ..rendering_3d import save_comparison, save_gif, save_hidden_channels, save_isometric, save_views
    save_gif(frames, run / "visualizations" / "growth.gif")
    save_comparison(final_target[0].cpu().numpy(), final_state[:1], run / "visualizations" / "target_vs_prediction.png")
    save_hidden_channels(final_state[:1], layout.hidden_slice, run / "visualizations" / "hidden_channels.png")
    if dimensions == 3:
        save_isometric(final_state[:1], run / "visualizations" / "isometric.png")
        save_views(final_state[:1], run / "visualizations" / "views.png")
    from ..plotting import plot_metrics
    plot_metrics(run / "logs.csv", run / "visualizations" / "metrics.png")
    run_metadata["training_seconds"] = time.perf_counter() - started
    if last_validation:
        run_metadata["best_worst_genome_persistence_score"] = best_score
        run_metadata["last_persistence_validation"] = last_validation
    write_json(run / "metadata.json", run_metadata)
    return run
