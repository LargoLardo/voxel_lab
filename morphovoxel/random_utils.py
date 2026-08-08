"""Deterministic random-number setup."""
from __future__ import annotations

import random

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


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Choose CUDA when requested automatically and available."""
    name = str(requested).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except RuntimeError as error:
        raise ValueError(f"invalid compute device: {requested}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch build cannot access a CUDA GPU")
    return device
