"""Fixed local 3D perception."""
from __future__ import annotations

from functools import lru_cache

import torch
from torch.nn import functional as F


@lru_cache(maxsize=32)
@torch.inference_mode(False)
def _perception_weight(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Build on CPU once, including when first used during inference: autograd
    # needs a normal (non-inference) tensor for subsequent training backwards.
    kernels = torch.zeros((5, 3, 3, 3), dtype=dtype)
    kernels[0, 1, 1, 1] = 1
    kernels[1, 1, 1, 0], kernels[1, 1, 1, 2] = -0.5, 0.5
    kernels[2, 1, 0, 1], kernels[2, 1, 2, 1] = -0.5, 0.5
    kernels[3, 0, 1, 1], kernels[3, 2, 1, 1] = -0.5, 0.5
    kernels[4, 1, 1, 1] = -6
    for index in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
        kernels[(4, *index)] = 1
    return kernels[:, None].repeat(channels, 1, 1, 1, 1).reshape(-1, 1, 3, 3, 3).to(device)


def perceive_3d(state: torch.Tensor) -> torch.Tensor:
    """Apply identity, three central gradients, and a 6-neighbor Laplacian."""
    channels = state.shape[1]
    weight = _perception_weight(channels, state.device, state.dtype)
    return F.conv3d(state, weight, padding=1, groups=channels)
