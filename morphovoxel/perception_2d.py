"""Fixed local 2D perception."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def perceive_2d(state: torch.Tensor) -> torch.Tensor:
    """Apply identity, Sobel-x/y, and Laplacian per channel."""
    kernels = state.new_tensor([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    ])
    channels = state.shape[1]
    weight = kernels[:, None].repeat(channels, 1, 1, 1).reshape(-1, 1, 3, 3)
    return F.conv2d(state, weight, padding=1, groups=channels)

