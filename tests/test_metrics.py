import numpy as np

from morphovoxel.metrics import recovery_metrics, threshold_iou


def test_empty_iou_and_total_destruction_are_safe():
    empty = np.zeros((5, 5, 5))
    target = np.ones_like(empty)
    assert threshold_iou(empty, empty) == 1
    result = recovery_metrics(target, empty, empty, target, target)
    assert result["recovered_target_fraction"] == 0
    assert result["recovery_success"] is False

