import torch

from morphovoxel.ecology.water import allocate_water, diffuse_water


def test_water_nonnegative_conserved_and_shared():
    water = torch.zeros(5, 5, 5)
    water[2, 2, 2] = 1
    substrate = torch.ones_like(water, dtype=torch.bool)
    diffused = diffuse_water(water, substrate, 0.1)
    assert diffused.min() >= 0
    assert torch.isclose(diffused.sum(), water.sum())
    remaining, absorbed = allocate_water(torch.ones_like(water), torch.ones(2, *water.shape))
    assert torch.allclose(absorbed[0], absorbed[1]) and remaining.min() >= 0

