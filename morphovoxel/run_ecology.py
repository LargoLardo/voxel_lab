"""Phase 5 fixed-morphogenesis ecology command."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from .checkpointing import load_checkpoint
from .config import load_config, save_config
from .environment import ENVIRONMENT_CHANNELS
from .ecology.environment import EcologyWorld
from .ecology.metrics import ecology_metrics
from .ecology.router import ModelRouter
from .ecology.simulator import ProceduralEcologyBaseline, ecology_step
from .genomes import MORPHOLOGIES, TreeGenome, one_hot_genomes, tree_genome_tensor
from .model_3d import NeuralCA3D
from .random_utils import resolve_device
from .state import StateLayout
from .utils import create_run_directory, metadata, steps_per_second, write_json, write_live_preview


def _frame(world: EcologyWorld) -> np.ndarray:
    owner = world.ownership.detach().cpu().numpy().max(0)
    palette = np.asarray([[0, 0, 0], [76, 175, 80], [255, 152, 0], [3, 169, 244], [156, 39, 176]], dtype=np.uint8)
    frame = palette[np.minimum(owner, len(palette) - 1)]
    empty = owner == 0
    frame[empty, 2] = (world.water.detach().cpu().numpy().max(0)[empty].clip(0, 1) * 180).astype(np.uint8)
    frame[empty, 0] = (world.light.detach().cpu().numpy().mean(0)[empty].clip(0, 1) * 60).astype(np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run = create_run_directory(config.get("run_name", "phase5_ecology"), config.get("runs_root", "runs"))
    save_config(config, run / "config.yaml")
    device = resolve_device(config.get("device", "auto"))
    frame_every = max(1, int(config.get("frame_every", 1)))
    live_preview = bool(config.get("live_preview", False))
    size, count = int(config.get("world_size", 16)), int(config.get("organisms", 2))
    layout = StateLayout(int(config.get("materials", 4)), int(config.get("hidden_channels", 8)), energy=False)
    configured_genomes = config.get("genomes")
    tree_genomes = config.get("tree_genomes")
    semantic_tree_genomes = tree_genomes is not None
    if configured_genomes is not None and tree_genomes is not None:
        raise ValueError("choose semantic tree_genomes or raw genomes, not both")
    if tree_genomes is not None:
        if not isinstance(tree_genomes, list) or not all(isinstance(value, dict) for value in tree_genomes):
            raise ValueError("tree_genomes must be a list of genome mappings")
        configured_genomes = tree_genome_tensor([TreeGenome.from_dict(value) for value in tree_genomes]).tolist()
    if configured_genomes is not None and len(configured_genomes) != count:
        raise ValueError("genomes must contain one continuous vector per organism")
    genome_size = len(configured_genomes[0]) if configured_genomes else len(MORPHOLOGIES)
    if configured_genomes is not None and any(len(values) != genome_size for values in configured_genomes):
        raise ValueError("all continuous ecology genomes must have the same size")
    context_channels = int(config.get("context_channels", 0))
    if context_channels not in (0, len(ENVIRONMENT_CHANNELS)):
        raise ValueError(f"context_channels must be 0 or {len(ENVIRONMENT_CHANNELS)}")

    specialist_checkpoints = config.get("specialist_checkpoints")
    organism_model_ids: tuple[str, ...] | None = None
    if specialist_checkpoints:
        if not isinstance(specialist_checkpoints, dict):
            raise ValueError("specialist_checkpoints must map model ids to checkpoint paths")
        specialist_models = {}
        for model_id, checkpoint in specialist_checkpoints.items():
            specialist = NeuralCA3D(
                layout.channels,
                int(config.get("model_width", 32)),
                0,
                float(config.get("fire_rate", 0.5)),
                context_channels,
            ).to(device).eval()
            payload = load_checkpoint(checkpoint, specialist, map_location=device)
            if payload["metadata"]["model_kind"] not in {"specialist", "tree_specialist"}:
                raise ValueError(
                    f"specialist ecology checkpoint {checkpoint} has incompatible model kind "
                    f"{payload['metadata']['model_kind']!r}"
                )
            specialist_models[str(model_id)] = specialist
        configured_ids = config.get("organism_model_ids")
        keys = tuple(specialist_models)
        organism_model_ids = tuple(map(str, configured_ids)) if configured_ids else tuple(keys[index % len(keys)] for index in range(count))
        if len(organism_model_ids) != count:
            raise ValueError("organism_model_ids must contain one specialist model id per organism")
        model = ModelRouter(specialist_models)
    elif config.get("procedural_baseline"):
        model = ModelRouter(ProceduralEcologyBaseline())
    else:
        shared_model = NeuralCA3D(
            layout.channels,
            int(config.get("model_width", 32)),
            genome_size,
            float(config.get("fire_rate", 0.5)),
            context_channels,
        ).to(device).eval()
        if config.get("checkpoint"):
            expected_kind = str(config.get(
                "model_kind",
                "tree_family" if semantic_tree_genomes else "legacy_conditional",
            ))
            load_checkpoint(
                config["checkpoint"], shared_model, map_location=device,
                expected_model_kind=expected_kind,
            )
        model = ModelRouter(shared_model)
    write_json(run / "metadata.json", metadata(int(config.get("seed", 0)), model, device))
    states = torch.zeros((count, layout.channels, size, size, size), device=device)
    positions = config.get("seed_positions")
    if positions and len(positions) != count:
        raise ValueError("seed_positions must contain one [z,y,x] coordinate per organism")
    for organism in range(count):
        point = tuple(positions[organism]) if positions else (size - 3, size // 2, round((organism + 1) * size / (count + 1)))
        states[(organism, layout.occupancy, *point)] = 1
        states[(organism, layout.material_slice, *point)] = 1
    substrate = torch.zeros((size, size, size), dtype=torch.bool, device=device)
    substrate[size - 2 :] = True
    obstacles = torch.zeros_like(substrate)
    for point in config.get("obstacle_voxels", []):
        if len(point) != 3 or any(not 0 <= int(coordinate) < size for coordinate in point):
            raise ValueError("obstacle_voxels must contain in-bounds [z,y,x] coordinates")
        obstacles[tuple(map(int, point))] = True
    water = substrate.float() * float(config.get("initial_water", 1.0))
    for z, y, x, radius, amount in config.get("water_patches", []):
        zz, yy, xx = torch.meshgrid(*(torch.arange(size, device=device) for _ in range(3)), indexing="ij")
        water[((zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= radius**2) & substrate] = float(amount)
    if configured_genomes is None:
        labels = torch.as_tensor(config.get("genome_labels", list(range(count))), dtype=torch.long, device=device) % len(MORPHOLOGIES)
        genomes = one_hot_genomes(labels)
    else:
        genomes = torch.as_tensor(configured_genomes, dtype=states.dtype, device=device)
    world = EcologyWorld(
        states,
        genomes,
        substrate,
        water,
        torch.ones_like(water),
        states[:, 0].clamp(0, 1) * float(config.get("initial_energy", 1.0)),
        obstacles=obstacles,
        model_ids=organism_model_ids,
    )
    frames, rows = [_frame(world)], []
    steps = int(config.get("steps", 16))
    if live_preview:
        write_live_preview(run / "visualizations" / "live.png", frames[0], phase="ecology", step=0, total_steps=steps, steps_per_second=0.0)
    simulation_started = time.perf_counter()
    with torch.no_grad():
        for step in range(steps):
            world, flows = ecology_step(world, model, config)
            rows.extend({"step": step + 1, **values} for values in ecology_metrics(world, flows))
            if (step + 1) % frame_every == 0 or step + 1 == steps:
                frame = _frame(world)
                frames.append(frame)
                if live_preview:
                    write_live_preview(
                        run / "visualizations" / "live.png", frame,
                        phase="ecology", step=step + 1, total_steps=steps,
                        steps_per_second=steps_per_second(step + 1, simulation_started),
                    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "metrics" / "per_step.csv", index=False)
    final = metrics.groupby("organism", as_index=False).last()
    for column in ("light_absorbed", "water_absorbed", "maintenance", "growth", "remodeling", "shading_imposed", "shading_received"):
        final[column] = metrics.groupby("organism")[column].sum().to_numpy()
    final["survival_time"] = metrics.assign(alive=metrics.occupied_volume > 0).groupby("organism")["alive"].sum().to_numpy()
    final.to_csv(run / "metrics" / "summary.csv", index=False)
    np.savez_compressed(
        run / "rollouts" / "ecology_states.npz",
        states=world.states.detach().cpu().numpy(), water=world.water.detach().cpu().numpy(), light=world.light.detach().cpu().numpy(),
        obstacles=world.obstacles.detach().cpu().numpy(),
    )
    imageio.mimsave(run / "visualizations" / "ecology.gif", frames, duration=125, loop=0)
    print(run)


if __name__ == "__main__":
    main()
