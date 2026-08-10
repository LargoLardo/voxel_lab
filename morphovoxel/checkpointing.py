"""Reproducible training checkpoints."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    step: int = 0,
    scheduler=None,
    config=None,
    pool=None,
    genomes=None,
    validation=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None, "step": step, "config": config,
        "pool": pool.state_dict() if pool else None, "genomes": genomes,
        "validation": validation,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, map_location="cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train the preset that creates it first, "
            "launch the Full experiment, or correct the checkpoint path."
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload
