"""Compact shared trainer for all growth phases."""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..checkpointing import load_checkpoint, save_checkpoint
from ..config import save_config
from ..genomes import MORPHOLOGIES, one_hot_genomes
from ..metrics import morphology_metrics, threshold_iou
from ..model_2d import NeuralCA2D
from ..model_3d import NeuralCA3D
from ..random_utils import resolve_device, seed_everything
from ..rollout import rollout
from ..seeding import seed_state
from ..state import StateLayout
from ..targets import make_target_2d, make_target_3d
from ..targets.morphology_library import save_target
from ..utils import create_run_directory, metadata, steps_per_second, write_json, write_live_preview
from .losses import morphology_loss
from .state_pool import StatePool

LOGGER = logging.getLogger(__name__)


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


def _pool_actions(
    state: torch.Tensor,
    target: torch.Tensor,
    ages: torch.Tensor,
    fresh_count: int,
    damage_fraction: float,
    mature_age: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the worst samples to reseed and the best mature samples to damage."""
    errors = torch.nan_to_num(
        (state[:, 0] - target).square().flatten(1).mean(1),
        nan=float("inf"), posinf=float("inf"), neginf=float("inf"),
    )
    fresh_count = min(len(state), max(0, fresh_count))
    reseed = torch.topk(errors, fresh_count).indices if fresh_count else torch.empty(0, dtype=torch.long, device=state.device)
    eligible = ages >= mature_age
    if len(reseed):
        eligible[reseed] = False
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    damage_count = min(len(candidates), round(len(candidates) * max(0.0, min(1.0, damage_fraction))))
    if damage_fraction > 0 and len(candidates) and not damage_count:
        damage_count = 1
    damage = candidates[torch.argsort(errors[candidates])[:damage_count]] if damage_count else torch.empty(0, dtype=torch.long, device=state.device)
    return reseed, damage


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
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(config.get("device", "auto"))
    size, batch = int(config.get("world_size", 32 if dimensions == 2 else 16)), int(config.get("batch_size", 2 if dimensions == 2 else 1))
    classes = 4 if dimensions == 3 else 3
    layout = StateLayout(materials=int(config.get("materials", classes)), hidden=int(config.get("hidden_channels", 8)))
    genome_size = len(MORPHOLOGIES) if conditional else 0
    model_class = NeuralCA3D if dimensions == 3 else NeuralCA2D
    model = model_class(layout.channels, int(config.get("model_width", 32)), genome_size, float(config.get("fire_rate", 0.5))).to(device)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(str(config.get("optimizer", "adam")).lower())
    if optimizer_class is None:
        raise ValueError("optimizer must be 'adam' or 'adamw'")
    optimizer = optimizer_class(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config["scheduler"]) if config.get("scheduler") else None
    start, restored = 0, None
    if config.get("resume"):
        restored = load_checkpoint(config["resume"], model, optimizer, scheduler, map_location=device)
        start = int(restored["step"])
    run = create_run_directory(str(config.get("run_name", f"phase{dimensions}d")), config.get("runs_root", "runs"))
    save_config(config, run / "config.yaml")
    run_metadata, started = metadata(seed, model, device), time.perf_counter()
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
    if conditional:
        requested_pool_size = configured_pool_size or max(batch, len(MORPHOLOGIES) * 4)
        pool_size = max(requested_pool_size, batch, len(MORPHOLOGIES))
    else:
        pool_size = max(configured_pool_size, batch) if configured_pool_size else 0
    if pool_size:
        pool_labels = torch.arange(pool_size) % (len(MORPHOLOGIES) if conditional else 1)
        pool_genomes = one_hot_genomes(pool_labels) if conditional else pool_labels[:, None].float()
        pool_states = seed_state(pool_size, size, layout, dimensions=dimensions, device="cpu")
        pool = StatePool(pool_states, pool_genomes)
        if restored and restored.get("pool"):
            saved = restored["pool"]
            pool = StatePool(saved["states"], saved["genomes"], saved["ages"])
    final_state = final_target = final_materials = None
    best_score, last_validation = float("-inf"), None
    for step in range(start, start + iterations):
        pool_batch = pool.sample(batch, torch.Generator().manual_seed(seed + step), device) if pool else None
        if pool_batch:
            state = pool_batch.states
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
                state[reseed] = seed_state(
                    len(reseed), size, layout, dimensions=dimensions,
                    seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)),
                    random_seed=seed + step, device=device,
                )
                pool_batch.ages[reseed] = 0
            if len(damage):
                from ..damage import damage_3d
                for index in damage.tolist():
                    state[index : index + 1], _ = damage_3d(
                        state[index : index + 1],
                        random.choice(config.get("damage_severities", [0.1, 0.25, 0.5])),
                        random.choice(config.get("damage_types", ["sphere", "cuboid", "top"])),
                        seed + step + index,
                    )
        else:
            labels = torch.arange(step * batch, (step + 1) * batch, device=device) % (len(MORPHOLOGIES) if conditional else 1)
            genomes = one_hot_genomes(labels) if conditional else None
            state = seed_state(batch, size, layout, dimensions=dimensions, seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)), random_seed=seed + step, device=device)
            target, material = _targets(dimensions, labels, size, seed, conditional, device, config.get("target_kind"))
        steps = random.randint(int(minimum), int(maximum))
        preview_started = time.perf_counter() if preview_image and (step + 1) % 10 == 0 else None
        final_state, _ = rollout(model, state, steps, genomes)
        loss, components = morphology_loss(
            final_state, target, material, layout, config.get("loss_weights"), state_limit=state_limit,
        )
        persistence_steps = random.randint(persistence_minimum, persistence_maximum)
        committed_state = final_state
        if persistence_steps:
            committed_state, _ = rollout(model, final_state, persistence_steps, genomes)
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
        records.append({"step": step + 1, "loss": float(loss.detach()), **{name: float(value.detach()) for name, value in components.items()}})
        final_target, final_materials = target, material
        LOGGER.info("step=%d loss=%.6f", step + 1, float(loss.detach()))
        should_validate = validation_steps > 0 and ((step + 1) % validation_every == 0 or step + 1 == start + iterations)
        if should_validate:
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
            validation_records.extend(
                {"step": step + 1, "genome": name, "persistence_score": score, "worst_genome_persistence_score": worst_score}
                for name, score in genome_scores.items()
            )
            if worst_score > best_score:
                best_score = worst_score
                best_validation = {**last_validation, "best_worst_genome_persistence_score": best_score}
                save_checkpoint(
                    run / "checkpoints" / "best.pt", model, optimizer, step=step + 1,
                    scheduler=scheduler, config=config, pool=pool,
                    genomes=list(MORPHOLOGIES) if conditional else None, validation=best_validation,
                )
            LOGGER.info("validation step=%d worst_genome_persistence=%.6f", step + 1, worst_score)
    checkpoint = run / "checkpoints" / "latest.pt"
    validation_summary = ({**last_validation, "best_worst_genome_persistence_score": best_score} if last_validation else None)
    save_checkpoint(
        checkpoint, model, optimizer, step=start + iterations, scheduler=scheduler, config=config, pool=pool,
        genomes=list(MORPHOLOGIES) if conditional else None, validation=validation_summary,
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
    capture_genome = one_hot_genomes(torch.zeros(1, dtype=torch.long, device=device)) if conditional else None
    with torch.no_grad():
        _, frames = rollout(model, capture, int(maximum), capture_genome, capture_every)
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
