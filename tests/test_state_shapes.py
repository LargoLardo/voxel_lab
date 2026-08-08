import torch

from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D


def test_nca_shapes():
    assert NeuralCA2D(5, 8, fire_rate=1)(torch.zeros(2, 5, 12, 13)).shape == (2, 5, 12, 13)
    assert NeuralCA3D(5, 8, fire_rate=1)(torch.zeros(2, 5, 8, 9, 10)).shape == (2, 5, 8, 9, 10)

