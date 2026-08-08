"""Evaluate one local 3D rule on larger empty worlds without rescaling its target."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from morphovoxel.checkpointing import load_checkpoint
from morphovoxel.config import load_config
from morphovoxel.genomes import MORPHOLOGIES, one_hot_genomes
from morphovoxel.metrics import morphology_metrics
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.rollout import rollout
from morphovoxel.seeding import seed_state
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_target_3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="scaling.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    trained_size = int(config.get("world_size", 32))
    base_target, _ = make_target_3d("branching", trained_size, int(config.get("seed", 0)))
    layout = StateLayout(int(config.get("materials", 4)), int(config.get("hidden_channels", 8)))
    model = NeuralCA3D(layout.channels, int(config.get("model_width", 64)), len(MORPHOLOGIES), float(config.get("fire_rate", 0.5)))
    load_checkpoint(args.checkpoint, model)
    rows = []
    for size in config.get("scaling_sizes", [40, 48, 64]):
        state = seed_state(1, int(size), layout, dimensions=3)
        started = time.perf_counter()
        state, _ = rollout(model, state, int(config.get("evaluation_steps", 48)), one_hot_genomes(torch.tensor([0])))
        target = np.zeros((size, size, size), np.float32)
        offset = (size - trained_size) // 2
        target[offset : offset + trained_size, offset : offset + trained_size, offset : offset + trained_size] = base_target
        rows.append({"world_size": size, "runtime_seconds": time.perf_counter() - started, **morphology_metrics(state[0, 0].clamp(0, 1), target)})
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

