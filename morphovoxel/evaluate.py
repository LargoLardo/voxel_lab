"""Normal-growth checkpoint evaluation."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from .checkpointing import load_checkpoint
from .config import load_config
from .genomes import MORPHOLOGIES, one_hot_genomes
from .metrics import material_accuracy, morphology_metrics
from .model_2d import NeuralCA2D
from .model_3d import NeuralCA3D
from .rollout import rollout
from .seeding import seed_state
from .state import StateLayout
from .targets import make_target_2d, make_target_3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="evaluation.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    dimensions, size = int(config.get("dimensions", 3)), int(config.get("world_size", 16))
    layout = StateLayout(int(config.get("materials", 4 if dimensions == 3 else 3)), int(config.get("hidden_channels", 8)))
    conditional = bool(config.get("conditional", dimensions == 3))
    model_class = NeuralCA3D if dimensions == 3 else NeuralCA2D
    model = model_class(layout.channels, int(config.get("model_width", 32)), len(MORPHOLOGIES) if conditional else 0, 1.0)
    load_checkpoint(args.checkpoint, model)
    rows = []
    kinds = MORPHOLOGIES if conditional else ("branching",)
    maker = make_target_3d if dimensions == 3 else make_target_2d
    for label, kind in enumerate(kinds):
        genome = one_hot_genomes(torch.tensor([label])) if conditional else None
        state = seed_state(1, size, layout, dimensions=dimensions)
        started = time.perf_counter()
        state, _ = rollout(model, state, int(config.get("evaluation_steps", 32)), genome)
        elapsed = time.perf_counter() - started
        target, materials = maker(kind, size, int(config.get("seed", 0)))
        rows.append({"genome": kind, "parameter_count": sum(p.numel() for p in model.parameters()), "inference_seconds": elapsed, "material_accuracy": material_accuracy(state[0, layout.material_slice], materials, target), **morphology_metrics(state[0, 0].clamp(0, 1), target)})
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
