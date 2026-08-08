import numpy as np

from morphovoxel.targets import make_target_2d, make_target_3d


def test_targets_are_deterministic_and_distinct():
    assert np.array_equal(make_target_2d("asymmetric", 24, 3)[0], make_target_2d("asymmetric", 24, 3)[0])
    targets = [make_target_3d(kind, 12, 2)[0] for kind in ("branching", "conical", "radial", "mushroom")]
    assert all(target.sum() > 0 for target in targets)
    assert len({target.tobytes() for target in targets}) == 4

