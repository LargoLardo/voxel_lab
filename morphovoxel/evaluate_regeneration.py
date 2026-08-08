"""Damage and regeneration evaluation."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpointing import load_checkpoint
from .config import load_config, save_config
from .damage import damage_3d
from .genomes import MORPHOLOGIES, one_hot_genomes
from .metrics import recovery_metrics, threshold_iou
from .model_3d import NeuralCA3D
from .random_utils import resolve_device
from .rendering_3d import projection, save_gif, save_recovery_comparison
from .rollout import rollout
from .seeding import seed_state
from .state import StateLayout
from .targets import make_target_3d
from .utils import create_run_directory, metadata, steps_per_second, write_json, write_live_preview


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run = create_run_directory(config.get("run_name", "phase4_regeneration"), config.get("runs_root", "runs"))
    save_config(config, run / "config.yaml")
    device = resolve_device(config.get("device", "auto"))
    frame_every = max(1, int(config.get("frame_every", 1)))
    live_preview = bool(config.get("live_preview", False))
    size = int(config.get("world_size", 16))
    layout = StateLayout(int(config.get("materials", 4)), int(config.get("hidden_channels", 8)))
    rows, all_frames, comparison = [], [], None
    checkpoints = config.get("checkpoints", {"untrained_reference": None})
    for model_name, checkpoint in checkpoints.items():
        model = NeuralCA3D(layout.channels, int(config.get("model_width", 32)), len(MORPHOLOGIES), float(config.get("fire_rate", 0.5))).to(device).eval()
        if checkpoint:
            load_checkpoint(checkpoint, model, map_location=device)
        if not (run / "metadata.json").exists():
            write_json(run / "metadata.json", metadata(int(config.get("seed", 0)), model, device))
        for label, kind in enumerate(MORPHOLOGIES):
            genome = one_hot_genomes(torch.tensor([label], device=device))
            target, _ = make_target_3d(kind, size, int(config.get("seed", 0)))
            target = torch.as_tensor(target, device=device)
            state = seed_state(1, size, layout, dimensions=3, device=device)
            growth_steps = int(config.get("growth_steps", 16))
            growth_started = time.perf_counter()

            def publish_growth(update, current):
                if live_preview and (update % frame_every == 0 or update == growth_steps):
                    image = projection(current)
                    write_live_preview(
                        run / "visualizations" / "live.png", image,
                        phase="growth", genome=kind, step=update, total_steps=growth_steps,
                        steps_per_second=steps_per_second(update, growth_started),
                    )

            pre, before_frames = rollout(model, state, growth_steps, genome, frame_every, publish_growth)
            for damage_kind in config.get("damage_types", ["sphere", "top", "dropout"]):
                for severity in config.get("severities", [0.1, 0.25, 0.5, 0.75]):
                    damaged, mask = damage_3d(pre, float(severity), damage_kind, int(config.get("seed", 0)))
                    current, recovery_frames, ious = damaged, [damaged.detach().cpu()], []
                    recovery_steps = int(config.get("recovery_steps", 16))
                    if live_preview:
                        write_live_preview(
                            run / "visualizations" / "live.png", projection(damaged),
                            phase="damaged", genome=kind, damage=damage_kind, severity=severity,
                            step=0, total_steps=recovery_steps, steps_per_second=0.0,
                        )
                    recovery_started = time.perf_counter()
                    for update in range(1, recovery_steps + 1):
                        current = model(current, genome)
                        if update % frame_every == 0 or update == recovery_steps:
                            recovery_frames.append(current.detach().cpu())
                            if live_preview:
                                image = projection(current)
                                write_live_preview(
                                    run / "visualizations" / "live.png", image,
                                    phase="recovery", genome=kind, damage=damage_kind, severity=severity,
                                    step=update, total_steps=recovery_steps,
                                    steps_per_second=steps_per_second(update, recovery_started),
                                )
                        ious.append(threshold_iou(current[0, 0].clamp(0, 1), target))
                    values = recovery_metrics(pre[0, 0], damaged[0, 0], current[0, 0], target, mask & (target > 0.5))
                    threshold, consecutive = float(config.get("success_threshold", 0.85)), int(config.get("success_consecutive", 3))
                    recovery_time = next((i + 1 for i in range(max(0, len(ious) - consecutive + 1)) if all(v >= threshold for v in ious[i : i + consecutive])), np.nan)
                    rows.append({"model": model_name, "genome": kind, "damage": damage_kind, "severity": severity, "recovery_time": recovery_time, **values})
                    if not all_frames:
                        all_frames = before_frames + recovery_frames
                        comparison = (pre.detach().cpu(), damaged.detach().cpu(), current.detach().cpu())
    pd.DataFrame(rows).to_csv(run / "metrics" / "episodes.csv", index=False)
    pd.DataFrame(rows).groupby(["model", "damage", "severity"], as_index=False).mean(numeric_only=True).to_csv(run / "metrics" / "summary.csv", index=False)
    if all_frames:
        save_gif(all_frames, run / "visualizations" / "regeneration.gif")
        np.savez_compressed(run / "rollouts" / "regeneration_states.npz", states=np.stack([frame.numpy() for frame in all_frames]))
        save_recovery_comparison(*comparison, run / "visualizations" / "recovery_comparison.png")
    print(run)


if __name__ == "__main__":
    main()
