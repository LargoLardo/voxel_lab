import numpy as np
import pytest
import torch

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec
from morphovoxel.genomes import TREE_GENE_SPECS, TreeGenome
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.validation import (
    ValidationCriteria,
    build_candidate_panel,
    build_validation_panel,
    validate_candidate,
    validate_panel,
)


def test_validation_panel_is_deterministic_and_covers_required_sources():
    archived = TreeGenome.random(77, family="weeping")
    environments = (EnvironmentSpec(), EnvironmentSpec(wind_direction_x=1, wind_strength=0.5, seed=4))
    options = dict(
        seed=12,
        boundary_genes=("height",),
        random_count=1,
        interpolation_steps=1,
        mutation_count=1,
        archived=(archived,),
        fire_seeds=(3, 9),
        environments=environments,
    )

    first = build_validation_panel(**options)
    second = build_validation_panel(**options)

    assert [case.to_dict() for case in first] == [case.to_dict() for case in second]
    assert {case.category for case in first} == {
        "default", "boundary", "corner", "random", "interpolation", "mutation", "archive",
    }
    assert {case.fire_seed for case in first} == {3, 9}
    assert {case.environment for case in first} == set(environments)
    assert len({case.case_id for case in first}) == len(first)
    assert all(
        spec.minimum <= value <= spec.maximum
        for case in first
        for spec, value in zip(TREE_GENE_SPECS, case.genome.genes)
    )
    assert {case.genome.family for case in first if case.category == "interpolation"} == {"branching"}


class _TargetModel(torch.nn.Module):
    def __init__(self, template: torch.Tensor, *, genome_size: int, context_channels: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("template", template)
        self.channels = template.shape[1]
        self.genome_size = genome_size
        self.context_channels = context_channels
        self.calls: list[tuple[bool, bool, float]] = []

    def forward(self, state, genome=None, context=None):
        assert not torch.is_grad_enabled()
        self.calls.append((genome is not None, context is not None, float(torch.rand((), device=state.device))))
        return self.template.to(state).clone()


def _target_model(case, layout, genome_size):
    occupancy, materials = make_tree_target(case.genome, 12, case.environment)
    template = torch.zeros(1, layout.channels, 12, 12, 12)
    template[0, layout.occupancy] = torch.from_numpy(occupancy)
    for material in range(layout.materials):
        template[0, layout.material_slice.start + material][torch.from_numpy(materials == material)] = 3
    return _TargetModel(template, genome_size=genome_size, context_channels=len(ENVIRONMENT_CHANNELS))


@pytest.mark.parametrize("genome_size,expects_genome", [(TreeGenome.model_size(), True), (0, False)])
def test_candidate_validation_is_no_grad_reproducible_and_supports_family_or_specialist(genome_size, expects_genome):
    layout = StateLayout(materials=4, hidden=1)
    case = build_candidate_panel(TreeGenome(), fire_seeds=(41,), environments=(EnvironmentSpec(),))[0]
    model = _target_model(case, layout, genome_size)
    model.train()

    first = validate_candidate(model, case, layout=layout, world_size=12, steps=256, recovery_steps=64)
    calls_per_trial = len(model.calls)
    second = validate_candidate(model, case, layout=layout, world_size=12, steps=256, recovery_steps=64)

    assert first.validated and first.accepted and first.score > 0.9
    assert first.metrics == second.metrics and first.descriptors == second.descriptors
    assert model.calls[0][0] is expects_genome and model.calls[0][1] is True
    assert model.calls[0][2] == model.calls[calls_per_trial][2]
    assert model.training


def test_panel_aggregation_and_failed_or_short_protocols_are_explicit():
    layout = StateLayout(materials=4, hidden=1)
    cases = build_candidate_panel(TreeGenome(), fire_seeds=(5, 6), environments=(EnvironmentSpec(),))
    model = _target_model(cases[0], layout, TreeGenome.model_size())
    progress = []
    report = validate_panel(
        model,
        cases,
        layout=layout,
        world_size=12,
        steps=256,
        recovery_steps=64,
        aggregation="low_percentile",
        low_percentile=0.25,
        on_trial=lambda completed, total, trial: progress.append((completed, total, trial.case.case_id)),
    )
    assert report.validated and report.accepted
    assert report.score == report.low_percentile_score == report.worst_score
    assert report.to_dict()["criteria"]["min_steps"] == 256
    assert [item[:2] for item in progress] == [(1, 2), (2, 2)]

    short = validate_candidate(model, cases[0], layout=layout, world_size=12, steps=8, recovery_steps=2)
    assert not short.validated and not short.accepted and short.score == 0
    assert {"insufficient_validation_steps", "insufficient_recovery_steps"}.issubset(short.failure_reasons)

    class NonFinite(_TargetModel):
        def forward(self, state, genome=None, context=None):
            return torch.full_like(state, float("nan"))

    broken = NonFinite(model.template, genome_size=TreeGenome.model_size(), context_channels=len(ENVIRONMENT_CHANNELS))
    failed = validate_candidate(
        broken,
        cases[0],
        layout=layout,
        world_size=12,
        steps=256,
        recovery_steps=64,
        criteria=ValidationCriteria(min_target_iou=0),
    )
    assert failed.validated and not failed.accepted and failed.score == 0
    assert "non_finite_state" in failed.failure_reasons
    assert all(np.isfinite(value) for value in failed.metrics.values())

    empty = _TargetModel(
        torch.zeros_like(model.template),
        genome_size=TreeGenome.model_size(),
        context_channels=len(ENVIRONMENT_CHANNELS),
    )
    missing_body = validate_candidate(
        empty,
        cases[0],
        layout=layout,
        world_size=12,
        steps=256,
        recovery_steps=64,
        criteria=ValidationCriteria(min_target_iou=0),
    )
    assert missing_body.metrics["material_accuracy"] == 0
    assert "material_accuracy_below_minimum" in missing_body.failure_reasons
