from pathlib import Path

from morphovoxel.autoresearch import CANDIDATE, FROZEN_FILES, SCOPES, _ranking


def _metrics(**overrides):
    values = {
        "finite_cases": 8,
        "bounded_cases": 8,
        "max_occupancy_violation": 0.0,
        "worst_iou": 0.4,
        "worst_branch_dice": 0.3,
        "worst_leaf_dice": 0.2,
        "worst_genome_sensitivity": 0.1,
        "max_late_drift": 0.2,
        "worst_regeneration": 0.4,
    }
    values.update(overrides)
    return values


def test_research_ranking_is_graded_but_prioritizes_safety():
    baseline = _ranking(_metrics(), 100, 500)
    assert _ranking(_metrics(bounded_cases=7, worst_iou=1), 1, 1) < baseline
    assert _ranking(_metrics(worst_iou=0.5), 1000, 5000) > baseline
    assert _ranking(_metrics(worst_branch_dice=0.4), 100, 500) > baseline
    assert _ranking(_metrics(worst_leaf_dice=0.3), 100, 500) > baseline
    assert _ranking(_metrics(worst_genome_sensitivity=0.2), 100, 500) > baseline


def test_default_scope_is_config_only_and_frozen_evaluation_is_outside_every_scope():
    assert SCOPES["config"] == (CANDIDATE,)
    assert set(FROZEN_FILES).isdisjoint(SCOPES["config"])
    assert set(FROZEN_FILES).isdisjoint(SCOPES["expanded"])
    assert Path("morphovoxel/model_3d.py") in SCOPES["expanded"]
