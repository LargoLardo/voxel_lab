"""Compact shared trainer for all growth phases."""
from __future__ import annotations

import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..checkpointing import load_checkpoint, save_checkpoint
from ..config import save_config
from ..genomes import MORPHOLOGIES, one_hot_genomes
from ..metrics import morphology_metrics
from ..model_2d import NeuralCA2D
from ..model_3d import NeuralCA3D
from ..random_utils import resolve_device, seed_everything
from ..rollout import rollout
from ..seeding import seed_state
from ..state import StateLayout
from ..targets import make_target_2d, make_target_3d
from ..targets.morphology_library import save_target
from ..utils import create_run_directory, metadata, steps_per_second, write_json, write_live_preview
from .losses import morphology_loss, stability_loss
from .state_pool import StatePool

LOGGER = logging.getLogger(__name__)


def _targets(dimensions: int, labels: torch.Tensor, size: int, seed: int, conditional: bool, device, target_kind: str | None = None):
    names_2d = ("branching", "radial", "elongated", "asymmetric")
    names = MORPHOLOGIES if dimensions == 3 else names_2d
    maker = make_target_3d if dimensions == 3 else make_target_2d
    occupancy, materials = zip(*(maker(names[int(label)] if conditional else (target_kind or names[0]), size, seed) for label in labels))
    return torch.as_tensor(np.stack(occupancy), device=device), torch.as_tensor(np.stack(materials), device=device, dtype=torch.long)


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
    records = []
    iterations = int(config.get("iterations", 10))
    minimum, maximum = config.get("rollout_steps", [16, 32] if dimensions == 2 else [8, 16])
    default_capture_every = max(1, int(maximum) // 12)
    frame_every = max(1, int(config.get("frame_every", default_capture_every)))
    live_preview = bool(config.get("live_preview", False))
    preview_image = None
    if live_preview:
        if dimensions == 2:
            from ..rendering_2d import occupancy_image as preview_image
        else:
            from ..rendering_3d import projection as preview_image
    pool = None
    if int(config.get("pool_size", 0)):
        pool_size = int(config["pool_size"])
        pool_labels = torch.arange(pool_size) % (len(MORPHOLOGIES) if conditional else 1)
        pool_genomes = one_hot_genomes(pool_labels) if conditional else pool_labels[:, None].float()
        pool_states = seed_state(pool_size, size, layout, dimensions=dimensions, device="cpu")
        pool = StatePool(pool_states, pool_genomes)
        if restored and restored.get("pool"):
            saved = restored["pool"]
            pool = StatePool(saved["states"], saved["genomes"], saved["ages"])
    final_state = final_target = final_materials = None
    for step in range(start, start + iterations):
        pool_batch = pool.sample(batch, torch.Generator().manual_seed(seed + step), device) if pool else None
        if pool_batch:
            state = pool_batch.states
            genomes = pool_batch.genomes if conditional else None
            labels = genomes.argmax(1) if conditional else torch.zeros(batch, dtype=torch.long, device=device)
            fresh = max(0, round(batch * float(config.get("fresh_fraction", 0.25))))
            if fresh:
                state[:fresh] = seed_state(fresh, size, layout, dimensions=dimensions, seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)), random_seed=seed + step, device=device)
            if dimensions == 3 and float(config.get("damage_probability", 0)):
                from ..damage import damage_3d
                for index in range(fresh, batch):
                    if random.random() < float(config["damage_probability"]):
                        state[index : index + 1], _ = damage_3d(state[index : index + 1], random.choice(config.get("damage_severities", [0.1, 0.25, 0.5])), random.choice(config.get("damage_types", ["sphere", "cuboid", "top"])), seed + step + index)
        else:
            labels = torch.arange(step * batch, (step + 1) * batch, device=device) % (len(MORPHOLOGIES) if conditional else 1)
            genomes = one_hot_genomes(labels) if conditional else None
            state = seed_state(batch, size, layout, dimensions=dimensions, seed_size=int(config.get("seed_size", 1)), noise=float(config.get("seed_noise", 0)), random_seed=seed + step, device=device)
        target, material = _targets(dimensions, labels, size, seed, conditional, device, config.get("target_kind"))
        steps = random.randint(int(minimum), int(maximum))
        on_step = None
        if preview_image:
            rollout_started = time.perf_counter()

            def on_step(update, current, *, iteration=step + 1, total=steps):
                if update % frame_every == 0 or update == total:
                    image = preview_image(current[:1])
                    write_live_preview(
                        run / "visualizations" / "live.png", image,
                        phase="training", iteration=iteration, step=update, total_steps=total,
                        steps_per_second=steps_per_second(update, rollout_started),
                    )
        final_state, _ = rollout(model, state, steps, genomes, on_step=on_step)
        loss, components = morphology_loss(final_state, target, material, layout, config.get("loss_weights"))
        stability_steps = int(config.get("stability_steps", 0))
        if stability_steps:
            continued, _ = rollout(model, final_state, stability_steps, genomes)
            components["stability"] = stability_loss(final_state, continued)
            loss = loss + float(config.get("loss_weights", {}).get("stability", 0.1)) * components["stability"]
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
            pool.commit(pool_batch, final_state, steps)
        records.append({"step": step + 1, "loss": float(loss.detach()), **{name: float(value.detach()) for name, value in components.items()}})
        final_target, final_materials = target, material
        LOGGER.info("step=%d loss=%.6f", step + 1, float(loss.detach()))
    checkpoint = run / "checkpoints" / "latest.pt"
    save_checkpoint(checkpoint, model, optimizer, step=start + iterations, scheduler=scheduler, config=config, pool=pool, genomes=list(MORPHOLOGIES) if conditional else None)
    shutil.copy2(checkpoint, run / "checkpoints" / "best.pt")
    pd.DataFrame(records).to_csv(run / "logs.csv", index=False)
    pd.DataFrame(records).to_csv(run / "metrics" / "per_step.csv", index=False)
    assert final_state is not None and final_target is not None and final_materials is not None
    final_np = final_state.detach().cpu().numpy()
    np.save(run / "rollouts" / "final_state.npy", final_np)
    save_target(run / "targets", final_target[0].cpu().numpy(), final_materials[0].cpu().numpy())
    prediction = final_state[0, 0].clamp(0, 1).detach().cpu().numpy()
    summary = morphology_metrics(prediction, final_target[0].cpu().numpy())
    pd.DataFrame([summary]).to_csv(run / "metrics" / "summary.csv", index=False)
    capture = seed_state(1, size, layout, dimensions=dimensions, device=device)
    capture_genome = one_hot_genomes(torch.zeros(1, dtype=torch.long, device=device)) if conditional else None
    final_on_step = None
    if preview_image:
        final_started = time.perf_counter()

        def final_on_step(update, current):
            if update % frame_every == 0 or update == int(maximum):
                image = preview_image(current[:1])
                write_live_preview(
                    run / "visualizations" / "live.png", image,
                    phase="final rollout", iteration=start + iterations, step=update, total_steps=int(maximum),
                    steps_per_second=steps_per_second(update, final_started),
                )
    with torch.no_grad():
        _, frames = rollout(model, capture, int(maximum), capture_genome, frame_every, final_on_step)
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
    write_json(run / "metadata.json", run_metadata)
    return run
