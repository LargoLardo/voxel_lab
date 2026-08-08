"""Render pairwise genome interpolations without assuming meaningful hybrids."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch

from morphovoxel.checkpointing import load_checkpoint
from morphovoxel.config import load_config
from morphovoxel.genomes import MORPHOLOGIES, one_hot_genomes
from morphovoxel.metrics import connected_components, morphology_metrics
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.rendering_3d import save_comparison
from morphovoxel.rollout import rollout
from morphovoxel.seeding import seed_state
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_target_3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/interpolation")
    args = parser.parse_args()
    config, output = load_config(args.config), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    size = int(config.get("world_size", 16))
    layout = StateLayout(int(config.get("materials", 4)), int(config.get("hidden_channels", 8)))
    model = NeuralCA3D(layout.channels, int(config.get("model_width", 64)), len(MORPHOLOGIES), float(config.get("fire_rate", 0.5)))
    load_checkpoint(args.checkpoint, model)
    rows = []
    endpoints = one_hot_genomes(torch.arange(len(MORPHOLOGIES)))
    for left, right in combinations(range(len(MORPHOLOGIES)), 2):
        genome = (endpoints[left : left + 1] + endpoints[right : right + 1]) / 2
        state, _ = rollout(model, seed_state(1, size, layout, dimensions=3), int(config.get("evaluation_steps", 48)), genome)
        occupancy = state[0, 0].clamp(0, 1)
        left_target, _ = make_target_3d(MORPHOLOGIES[left], size)
        right_target, _ = make_target_3d(MORPHOLOGIES[right], size)
        components, largest = connected_components(occupancy)
        name = f"{MORPHOLOGIES[left]}_{MORPHOLOGIES[right]}"
        structure = morphology_metrics(occupancy, left_target)
        continued, _ = rollout(model, state, int(config.get("stability_steps", 8)), genome)
        symmetry = float(1 - (occupancy - occupancy.flip(-1)).abs().mean())
        material_confidence = float(state[0, layout.material_slice].softmax(0).max(0).values[occupancy > 0.5].mean()) if (occupancy > 0.5).any() else 0.0
        rows.append({
            "pair": name, "components": components, "largest_component_fraction": largest,
            "similarity_left": structure["iou"], "similarity_right": morphology_metrics(occupancy, right_target)["iou"],
            "volume": structure["occupied_volume"], "bounding_box_error_to_left": structure["bounding_box_error"],
            "symmetry": symmetry, "material_consistency": material_confidence,
            "stability": float(1 - (continued[0, 0].clamp(0, 1) - occupancy).abs().mean()),
            "structural_collapse": bool(structure["occupied_volume"] == 0 or components > 4),
        })
        save_comparison(left_target, state, output / f"{name}.png")
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
