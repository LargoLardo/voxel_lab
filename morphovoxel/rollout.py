"""NCA rollout utilities."""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

from .model_3d import NeuralCA3D, TreeFamilyNCA3D


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
    step_model = model
    if steps and isinstance(model, TreeFamilyNCA3D):
        step_model = partial(model, prepared_genome=model.prepare_genome(genome))
    if (steps and on_step is None and isinstance(model, (NeuralCA3D, TreeFamilyNCA3D))
            and state.device.type == "mps" and torch.is_grad_enabled()):
        fixed_context = None
        if model.context_channels and isinstance(context, torch.Tensor):
            expected = (state.shape[0], model.context_channels, *state.shape[2:])
            if context.shape != expected:
                raise ValueError("context must have shape [B, context_channels, D, H, W]")
            fixed_context = context
        elif isinstance(model, NeuralCA3D) and model.genome_size and not model.context_channels and context is None:
            # Genes are spatially constant: calculate a single offset per
            # organism and broadcast it, without materializing a voxel grid.
            fixed_context = state.new_empty((state.shape[0], 0, 1, 1, 1))
        # Local to this rollout: weights, environment and autograd graph must
        # never leak into the next optimizer update or a changing environment.
        if fixed_context is not None:
            step_model = partial(step_model, prepared_conditioning=model.prepare_conditioning(genome, fixed_context))
    for step in range(steps):
        step_context = context(step, state) if callable(context) else context
        if shared_fire_pairs:
            if len(state) % 2:
                raise ValueError("shared fire pairs require an even batch")
            fire = (torch.rand_like(state[::2, :1]) <= float(model.fire_rate)).repeat_interleave(2, 0)
            state = step_model(state, genome, step_context, fire) if step_context is not None else step_model(state, genome, fire_mask=fire)
        else:
            state = step_model(state, genome, step_context) if step_context is not None else step_model(state, genome)
        if capture_every and ((step + 1) % capture_every == 0 or step + 1 == steps):
            frames.append(state.detach().cpu())
        if on_step:
            on_step(step + 1, state)
    return state, frames
