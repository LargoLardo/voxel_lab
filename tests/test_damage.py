import torch

from morphovoxel.damage import damage_2d, damage_3d


def test_damage_clears_all_channels_only_inside_mask():
    state = torch.ones(1, 7, 12, 12, 12)
    damaged, mask = damage_3d(state, 0.25, "sphere")
    assert (damaged[:, :, mask] == 0).all()
    assert torch.equal(damaged[:, :, ~mask], state[:, :, ~mask])
    state2 = torch.ones(1, 4, 16, 16)
    damaged2, mask2 = damage_2d(state2, 0.25)
    assert (damaged2[:, :, mask2] == 0).all()

