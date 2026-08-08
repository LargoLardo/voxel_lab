"""Fixed local 3D perception."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def perceive_3d(state: torch.Tensor) -> torch.Tensor:
    """Apply identity, three central gradients, and a 6-neighbor Laplacian."""
    kernels = state.new_zeros((5, 3, 3, 3))
    kernels[0, 1, 1, 1] = 1
    kernels[1, 1, 1, 0], kernels[1, 1, 1, 2] = -0.5, 0.5
    kernels[2, 1, 0, 1], kernels[2, 1, 2, 1] = -0.5, 0.5
    kernels[3, 0, 1, 1], kernels[3, 2, 1, 1] = -0.5, 0.5
    kernels[4, 1, 1, 1] = -6
    for index in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
        kernels[(4, *index)] = 1
    channels = state.shape[1]
    weight = kernels[:, None].repeat(channels, 1, 1, 1, 1).reshape(-1, 1, 3, 3, 3)
    return F.conv3d(state, weight, padding=1, groups=channels)

