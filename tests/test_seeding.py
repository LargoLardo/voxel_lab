import torch

from morphovoxel.seeding import seed_state
from morphovoxel.state import StateLayout


def test_seed_is_deterministic():
    args = (2, 15, StateLayout(hidden=4))
    first = seed_state(*args, dimensions=2, seed_size=3, noise=0.1, random_seed=4)
    second = seed_state(*args, dimensions=2, seed_size=3, noise=0.1, random_seed=4)
    assert torch.equal(first, second)
    assert (first[:, 0] > 0.5).sum() == 18

