import torch

from morphovoxel.ecology.light import compute_light


def test_light_attenuates_below_opaque_voxel():
    occupancy = torch.zeros(5, 3, 3)
    occupancy[1, 1, 1] = 1
    light = compute_light(occupancy, attenuation=0.8)
    assert 0 <= light.min() and light.max() <= 1
    assert light[2, 1, 1] < light[2, 0, 0]

