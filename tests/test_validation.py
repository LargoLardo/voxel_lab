import numpy as np
import pytest
import torch

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec
from morphovoxel.genomes import FAMILY_GENE_NAMES, TREE_FAMILIES, TREE_GENE_SPECS, TreeGenome
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.validation import (
    ValidationCriteria,
    build_candidate_panel,
    build_gene_response_panel,
    build_gene_transition_panel,
    build_transition_panel,
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
    for category in ("default", "boundary", "corner", "random", "interpolation", "mutation"):
        counts = [sum(case.category == category and case.genome.family == family for case in first) for family in TREE_FAMILIES]
        assert min(counts) > 0 and len(set(counts)) == 1


def test_default_variation_validation_checks_every_family_at_both_gene_limits():
    panel = build_validation_panel(environments=(EnvironmentSpec(),), fire_seeds=(1,))
    for family in TREE_FAMILIES:
        for gene in ("height", "canopy_spread"):
            values = {case.genome.value(gene) for case in panel if case.genome.family == family and case.category == "boundary"}
            assert {-1, 1} <= values
    disabled = build_validation_panel(
        boundary_genes=(), random_count=0, interpolation_steps=0, mutation_count=0,
        environments=(EnvironmentSpec(),), fire_seeds=(1,),
    )
    assert len(disabled) == len(TREE_FAMILIES)


def test_gene_response_panel_covers_all_controls_with_matched_backgrounds():
    options = dict(style_seeds=[0, 970806], fire_seeds=[41, 42])
    panel = build_gene_response_panel(**options)
    assert panel == build_gene_response_panel(**options)
    assert len(panel) == len({case.case_id for case in panel}) == 32
    controls = set()
    for case in panel:
        low, high = case.comparison_genome, case.genome
        assert low.family == high.family and low.style_seed == high.style_seed
        assert case.source_genome is None and case.source_steps == 0
        name, = [name for name in FAMILY_GENE_NAMES if low.value(name) != high.value(name)]
        assert low.value(name) == -.75 and high.value(name) == .75
        assert case.to_dict()["comparison_genome"] == low.to_dict()
        controls.add((high.family, name))
    assert controls == {(family, name) for family in TREE_FAMILIES for name in FAMILY_GENE_NAMES}


@pytest.mark.parametrize("target_change", ["shape", "material", "none"])
@pytest.mark.parametrize("responds", [False, True])
def test_gene_response_requires_both_outputs_to_match_changed_voxels(monkeypatch, target_change, responds):
    from dataclasses import replace
    from morphovoxel import validation

    case = next(case for case in build_gene_response_panel(style_seeds=[42], fire_seeds=[41])
                if case.case_id == "gene-response-weeping-canopy_spread")
    layout = StateLayout(4, 1)
    low_case = replace(case, genome=case.comparison_genome, comparison_genome=None)
    high = _target_model(case, layout, TreeGenome.model_size())
    low = _target_model(low_case, layout, TreeGenome.model_size())
    if target_change != "shape":
        occupancy, materials = make_tree_target(case.genome, 12)
        changed_materials = materials.copy()
        if target_change == "material":
            changed_materials[materials == 3] = 2
        def targets(genome, size, environment):
            return occupancy, changed_materials if genome == case.comparison_genome else materials
        monkeypatch.setattr(validation, "make_tree_target", targets)
        low.template.copy_(high.template)
        for label in range(layout.materials):
            low.template[0, layout.material_slice.start + label] = torch.from_numpy((changed_materials == label) * 3)

    class Conditional(_TargetModel):
        def forward(self, state, genome=None, context=None):
            result = super().forward(state, genome, context)
            if responds and genome[0, 9] < 0:  # canopy_spread
                result = low.template.clone()
            return result

    model = Conditional(high.template, genome_size=TreeGenome.model_size(), context_channels=len(ENVIRONMENT_CHANNELS))
    model.train()
    rng = torch.get_rng_state()
    trial = validate_candidate(model, case, layout=layout, world_size=12, steps=4, recovery_steps=1,
                               criteria=ValidationCriteria(min_steps=4, min_recovery_steps=1))
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng)
    # Same stochastic sequence for the independent low/high growth rollouts.
    assert [call[2] for call in model.calls[:4]] == [call[2] for call in model.calls[4:8]]
    assert trial.metrics["target_iou"] == 1
    if target_change == "none":
        assert trial.metrics["gene_response_edited_voxels"] == 0
        assert "gene_response_accuracy" not in trial.metrics
    else:
        assert trial.metrics["gene_response_edited_voxels"] > 0
        assert trial.metrics["gene_response_accuracy"] == float(responds)
        assert ("gene_response_below_minimum" in trial.failure_reasons) != responds
        assert trial.accepted == responds


def test_transition_validation_keeps_source_state_and_tests_every_direction():
    panel = build_transition_panel(style_seeds=[0, 970806], fire_seeds=[41], source_steps=2)
    assert len(panel) == 24
    assert {(case.source_genome.family, case.genome.family) for case in panel} == {
        (source, destination) for source in TREE_FAMILIES for destination in TREE_FAMILIES if source != destination
    }
    assert all(case.source_genome.style_seed == case.genome.style_seed for case in panel)
    case = panel[0]
    layout = StateLayout(4, 1)
    destination = _target_model(case, layout, TreeGenome.model_size())
    from dataclasses import replace
    source = _target_model(replace(case, genome=case.source_genome), layout, TreeGenome.model_size())

    class Switching(_TargetModel):
        def forward(self, state, genome=None, context=None):
            assert not torch.is_grad_enabled()
            if int(genome[0, :4].argmax()) == TREE_FAMILIES.index(case.source_genome.family):
                self.calls.append("source")
                return source.template.clone()
            if self.calls[-1] == "source":
                torch.testing.assert_close(state, source.template)
            self.calls.append("destination")
            return self.template.clone()

    model = Switching(destination.template, genome_size=TreeGenome.model_size(), context_channels=len(ENVIRONMENT_CHANNELS))
    trial = validate_candidate(model, case, layout=layout, world_size=12, steps=4, recovery_steps=1,
                               criteria=ValidationCriteria(min_steps=4, min_recovery_steps=1))
    assert model.calls == ["source"] * 2 + ["destination"] * 5
    assert trial.accepted and trial.metrics["source_target_iou"] == trial.metrics["target_iou"] == 1
    assert trial.case.to_dict()["source_steps"] == 2

    # Perfect destination growth cannot hide a failed source organism.
    source.template.zero_()
    failed = validate_candidate(model, case, layout=layout, world_size=12, steps=4, recovery_steps=1,
                                criteria=ValidationCriteria(min_steps=4, min_recovery_steps=1))
    assert not failed.accepted and failed.score == 0
    assert "source_target_iou_below_minimum" in failed.failure_reasons


def test_gene_transition_panel_covers_every_gene_both_directions_and_edit_sizes():
    options = dict(style_seeds=[0, 970806, 1941611, 2912417], fire_seeds=[41, 42], source_steps=2)
    panel = build_gene_transition_panel(**options)
    assert panel == build_gene_transition_panel(**options)
    assert len(panel) == len({case.case_id for case in panel}) == 128
    assert {case.fire_seed for case in panel} == {41, 42}
    assert {case.genome.style_seed for case in panel} == set(options["style_seeds"])
    edits = set()
    for forward, reverse in zip(panel[::2], panel[1::2]):
        assert forward.genome == reverse.source_genome and forward.source_genome == reverse.genome
        assert forward.fire_seed == reverse.fire_seed
        for case in (forward, reverse):
            assert case.genome.family == case.source_genome.family
            assert case.genome.style_seed == case.source_genome.style_seed
            changed = [spec.name for spec in TREE_GENE_SPECS if case.genome.value(spec.name) != case.source_genome.value(spec.name)]
            assert len(changed) == 1 and changed[0] in FAMILY_GENE_NAMES
            edits.add((case.genome.family, changed[0], case.genome.value(changed[0])))
    assert edits == {(family, gene, value) for family in TREE_FAMILIES for gene in FAMILY_GENE_NAMES for value in (-.75, -.25, .25, .75)}


def test_gene_validation_rejects_ignored_small_edits_despite_high_whole_tree_iou():
    from dataclasses import replace
    case = next(case for case in build_gene_transition_panel(style_seeds=[0], fire_seeds=[41], source_steps=2)
                if case.genome.family == "branching" and case.genome.value("canopy_spread") == .25)
    layout = StateLayout(4, 1)
    source = _target_model(replace(case, genome=case.source_genome), layout, TreeGenome.model_size())
    destination = _target_model(case, layout, TreeGenome.model_size())

    class IgnoresLiveEdits(_TargetModel):
        def forward(self, state, genome=None, context=None):
            # Each genome grows perfectly from a seed. Once grown, keep the old
            # body's identity even when its gene input changes.
            identity = float(state[0, layout.hidden_slice.start].max()) or (1 if genome[0, 9] < 0 else 2)
            result = (source.template if identity == 1 else self.template).clone()
            result[:, layout.hidden_slice.start] = identity
            return result

    model = IgnoresLiveEdits(destination.template, genome_size=TreeGenome.model_size(), context_channels=len(ENVIRONMENT_CHANNELS))
    criteria = ValidationCriteria(min_steps=4, min_recovery_steps=1)
    trial = validate_candidate(model, case, layout=layout, world_size=12, steps=4, recovery_steps=1, criteria=criteria)
    assert trial.metrics["target_iou"] > .8 and trial.metrics["source_target_iou"] == 1
    assert trial.metrics["transition_edited_voxels"] > 0
    assert trial.metrics["transition_edit_accuracy"] == 0
    assert not trial.accepted and trial.score == 0
    assert "transition_edits_below_minimum" in trial.failure_reasons
    for genome in (case.source_genome, case.genome):
        fixed = replace(case, genome=genome, source_genome=None, source_steps=0)
        assert validate_candidate(model, fixed, layout=layout, world_size=12, steps=4, recovery_steps=1, criteria=criteria).accepted

    # Identical rasterized targets are persistence checks, not evidence of edits.
    unchanged = replace(case, genome=case.source_genome)
    trial = validate_candidate(source, unchanged, layout=layout, world_size=12, steps=4, recovery_steps=1, criteria=criteria)
    assert trial.accepted and trial.metrics["transition_edited_voxels"] == 0
    assert "transition_edit_accuracy" not in trial.metrics


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
