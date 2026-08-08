import torch

from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.perception_2d import perceive_2d


def test_perception_shape_and_locality():
    state = torch.zeros(1, 2, 9, 9)
    assert perceive_2d(state).shape == (1, 8, 9, 9)
    model = NeuralCA2D(2, 8, fire_rate=1)
    state[:, 0, 4, 4] = 1
    changed = (model(state) - model(torch.zeros_like(state))).abs().sum(1)[0]
    assert changed[:3].sum() == changed[6:].sum() == changed[:, :3].sum() == changed[:, 6:].sum() == 0


def test_growth_mask_blocks_empty_world():
    model = NeuralCA2D(2, 4, fire_rate=1)
    model.update[-1].bias.data.fill_(1)
    empty = torch.zeros(1, 2, 8, 8)
    assert torch.equal(model(empty), empty)

