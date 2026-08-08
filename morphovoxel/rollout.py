"""NCA rollout utilities."""
from __future__ import annotations

import torch


def rollout(
    model,
    state: torch.Tensor,
    steps: int,
    genome: torch.Tensor | None = None,
    capture_every: int = 0,
    on_step=None,
):
    if steps < 0 or capture_every < 0:
        raise ValueError("steps and capture_every must be nonnegative")
    frames = [state.detach().cpu()] if capture_every else []
    if on_step:
        on_step(0, state)
    for step in range(steps):
        state = model(state, genome)
        if capture_every and ((step + 1) % capture_every == 0 or step + 1 == steps):
            frames.append(state.detach().cpu())
        if on_step:
            on_step(step + 1, state)
    return state, frames
