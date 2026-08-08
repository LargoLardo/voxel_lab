import torch

from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.perception_3d import perceive_3d


def test_perception_shape_and_locality():
    state = torch.zeros(1, 2, 7, 7, 7)
    assert perceive_3d(state).shape == (1, 10, 7, 7, 7)
    model = NeuralCA3D(2, 8, fire_rate=1)
    state[:, 0, 3, 3, 3] = 1
    changed = (model(state) - model(torch.zeros_like(state))).abs().sum(1)[0]
    assert changed[:2].sum() == changed[5:].sum() == changed[:, :2].sum() == changed[:, 5:].sum() == 0

