"""Deterministic random-number setup."""
from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@contextmanager
def fork_rng(device: torch.device):
    """Keep validation from advancing or reseeding the training RNG streams."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # torch.manual_seed also seeds MPS, even when validation runs on the CPU.
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Prefer CUDA, then Apple Metal, then CPU for automatic selection."""
    name = str(requested).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        device = torch.device(name)
    except RuntimeError as error:
        raise ValueError(f"invalid compute device: {requested}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch build cannot access a CUDA GPU")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested, but this PyTorch build cannot access an Apple Metal GPU; use device: cpu")
    return device
