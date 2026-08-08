import torch

from morphovoxel.model_3d import NeuralCA3D


def test_larger_grid_does_not_change_parameters():
    model = NeuralCA3D(3, 4, fire_rate=1)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert model(torch.zeros(1, 3, 10, 10, 10)).shape[-1] == 10
    assert model(torch.zeros(1, 3, 14, 14, 14)).shape[-1] == 14
    assert sum(parameter.numel() for parameter in model.parameters()) == count

