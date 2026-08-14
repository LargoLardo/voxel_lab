"""NCA rollout utilities."""
from __future__ import annotations

from collections.abc import Callable

import torch


def rollout(
    model,
    state: torch.Tensor,
    steps: int,
    genome: torch.Tensor | None = None,
    capture_every: int = 0,
    on_step=None,
    context: torch.Tensor | Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    shared_fire_pairs: bool = False,
):
    if steps < 0 or capture_every < 0:
        raise ValueError("steps and capture_every must be nonnegative")
    frames = [state.detach().cpu()] if capture_every else []
    if on_step:
        on_step(0, state)
    for step in range(steps):
        step_context = context(step, state) if callable(context) else context
        if shared_fire_pairs:
            if len(state) % 2:
                raise ValueError("shared fire pairs require an even batch")
            fire = (torch.rand_like(state[::2, :1]) <= float(model.fire_rate)).repeat_interleave(2, 0)
            state = model(state, genome, step_context, fire) if step_context is not None else model(state, genome, fire_mask=fire)
        else:
            state = model(state, genome, step_context) if step_context is not None else model(state, genome)
        if capture_every and ((step + 1) % capture_every == 0 or step + 1 == steps):
            frames.append(state.detach().cpu())
        if on_step:
            on_step(step + 1, state)
    return state, frames
