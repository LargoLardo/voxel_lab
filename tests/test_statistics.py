from morphovoxel.metrics import paired_bootstrap_interval, paired_permutation_test


def test_paired_statistics_are_reproducible_and_bounded():
    result = paired_bootstrap_interval([2, 3, 4], [1, 1, 1], seed=3, samples=100)
    assert result[1] <= result[0] <= result[2]
    assert 0 <= paired_permutation_test([2, 3, 4], [1, 1, 1], seed=3, samples=100) <= 1
