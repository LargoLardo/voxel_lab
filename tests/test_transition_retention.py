"""Regression checks for remembering growth during live-edit training."""
import pandas as pd
import pytest
import torch

from morphovoxel.checkpointing import save_checkpoint
from morphovoxel.genomes import TreeGenome
from morphovoxel.model_3d import TreeFamilyNCA3D
from morphovoxel.training import trainer
from morphovoxel.training.family import sample_counterfactual_family_data
from morphovoxel.training.transition import replay_recipe, replay_sampling_options


def test_replay_keeps_learned_coverage_and_exact_specialist_base(tmp_path):
    source = {"config": {"model_kind": "tree_family", "family_curriculum": "variation", "iterations": 8000}, "step": 8000}
    recipe = replay_recipe(source, {})
    assert recipe["genome_span"] == recipe["background_span"] == recipe["style_random_fraction"] == 1
    original = tmp_path / "original.pt"
    torch.save(source, original)
    old_transition = {"config": {"family_curriculum": "gene_transition", "iterations": 8000,
                                 "initialize_from_checkpoint": str(original)}, "step": 1}
    assert replay_recipe(old_transition, {}) == recipe
    specialist = TreeGenome(family="weeping", style_seed=23).with_values({"height": .4})
    recipe = replay_recipe({"config": {"model_kind": "tree_specialist", "tree_genome": specialist.to_dict()}}, {})
    for neutral in (False, True):
        data = sample_counterfactual_family_data(8, 12, 42, families=("weeping",),
                                                **replay_sampling_options(recipe, neutral=neutral))
        assert all(genome == specialist for genome in data.genomes)


@pytest.mark.parametrize("mode", ["transition", "gene_transition"])
def test_rehearsal_pools_are_isolated_and_resume_without_narrowing(tmp_path, monkeypatch, mode):
    source = tmp_path / "source.pt"
    save_checkpoint(source, TreeFamilyNCA3D(6, 4, TreeGenome.model_size(), 1, 0), step=8000,
                    config={"model_kind": "tree_family", "family_curriculum": "variation", "iterations": 8000})
    config = dict(run_name="rehearsal", runs_root=str(tmp_path), model_kind="tree_family", family_curriculum=mode,
                  initialize_from_checkpoint=str(source), device="cpu", world_size=12, batch_size=8, pool_size=64,
                  materials=4, hidden_channels=1, model_width=4, fire_rate=1, environment_conditioning=False,
                  iterations=1, family_curriculum_iterations=8000, transition_source_steps=2, rollout_steps=1,
                  persistence_steps=1, validation_steps=0, family_style_seeds=[0])
    monkeypatch.setattr(trainer.random, "random", lambda: .3)
    run = trainer.train(config, dimensions=3, conditional=True)
    path = run / "checkpoints/latest.pt"
    first = torch.load(path, weights_only=False)
    replay = first["transition_state"]
    assert not first["pool"]["ages"].any()
    assert not replay["pools"]["neutral"]["ages"].any()
    assert replay["pools"]["variation"]["ages"].max() == 2
    assert replay["pools"]["variation"]["genomes"][:, 4:12].abs().max() > .8
    assert replay["cursors"] == {"transition": 0, "neutral": 0, "variation": 4}
    assert pd.read_csv(run / "logs.csv").variation_rehearsal_fraction.iloc[0] == 1
    config.pop("initialize_from_checkpoint")
    monkeypatch.setattr(trainer.random, "random", lambda: 0)
    resumed = trainer.train({**config, "run_name": "resumed", "resume": str(path)}, dimensions=3, conditional=True)
    second = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
    assert second["config"]["transition_replay"] == first["config"]["transition_replay"]
    for key, value in replay["pools"]["variation"].items():
        torch.testing.assert_close(second["transition_state"]["pools"]["variation"][key], value)
    assert second["transition_state"]["pools"]["neutral"]["ages"].max() == 2
    assert second["transition_state"]["cursors"]["neutral"] == 4


@pytest.mark.parametrize("mode", ["transition", "variation"])
def test_best_requires_retention_and_quality_to_break_zero_score_ties(tmp_path, monkeypatch, mode):
    from morphovoxel.validation import ValidationReport, ValidationTrial

    retained_values = iter((.8, .8, .65, .78, .78))  # Baseline, then four validations.
    edit_values = iter((.3, .9, .4, .2))
    def fake_validation(model, panel, **kwargs):
        retention = panel[0].category in {"neutral", "variation"}
        quality = next(retained_values if retention else edit_values)
        return ValidationReport(tuple(ValidationTrial(
            case, 1, 1, True, False, 0., ("state_bound",),
            {"target_iou": quality, "material_accuracy": .8, "late_drift": .01, "finite_state": 1.,
             **({} if retention or mode == "variation" else {"transition_edited_voxels": 12., "transition_edit_accuracy": quality})}, {},
        ) for case in panel))
    monkeypatch.setattr(trainer, "validate_panel", fake_validation)
    config = dict(run_name="guarded", runs_root=str(tmp_path), model_kind="tree_family", family_curriculum=mode,
                  device="cpu", world_size=12, batch_size=2, materials=4, hidden_channels=1, model_width=4,
                  fire_rate=1, iterations=4, transition_source_steps=2, rollout_steps=1, persistence_steps=1,
                  validation_steps=1, validation_recovery_steps=1, validation_every=1,
                  validation_fire_seeds=[71], family_style_seeds=[0])
    run = trainer.train(config, dimensions=3, conditional=True)
    best = torch.load(run / "checkpoints/best.pt", weights_only=False)
    latest = torch.load(run / "checkpoints/latest.pt", weights_only=False)
    assert best["step"] == 3 and latest["step"] == 4
    assert best["transition_state"]["best_rank"] == pytest.approx([0., .4])
    rows = pd.read_csv(run / "metrics/retention_validation.csv")
    assert not rows[rows.step == 2].eligible.any()
    assert rows[rows.step == 3].eligible.all()
    assert set(rows[rows.step == 2].failure_reasons.str.contains("target_iou")) == {True}
    # Resume retains the ORIGINAL baseline, rather than accepting another .05 drop.
    retained_values = iter((.72,))
    edit_values = iter((.99,))
    trainer.train({**config, "resume": str(run / "checkpoints/latest.pt"), "iterations": 1}, dimensions=3, conditional=True)
    resumed = torch.load(run / "checkpoints/latest.pt", weights_only=False)
    assert resumed["step"] == 5 and not resumed["validation"]["retention"]["eligible"]
    assert torch.load(run / "checkpoints/best.pt", weights_only=False)["step"] == 3
    assert resumed["transition_state"]["retention"] == latest["transition_state"]["retention"]
    with pytest.raises(ValueError, match="retention validation settings changed"):
        trainer.train({**config, "resume": str(run / "checkpoints/latest.pt"), "validation_steps": 2}, dimensions=3, conditional=True)


def test_retention_cannot_hide_a_forgotten_family_or_nonfinite_state():
    from morphovoxel.training.transition import retention_failures

    original = {family: dict(target_iou=.7, material_accuracy=.7, late_drift=.05, finite_state=1.)
                for family in ("branching/neutral", "weeping/variation")}
    current = {family: dict(values) for family, values in original.items()}
    current["branching/neutral"]["target_iou"] = .99
    current["weeping/variation"]["target_iou"] = .6
    assert retention_failures(original, current, .05) == ["weeping/variation: target_iou regressed by 0.1000"]
    current["weeping/variation"] = dict(original["weeping/variation"], late_drift=.2)
    assert "late_drift" in retention_failures(original, current, .05)[0]
    current["weeping/variation"]["finite_state"] = 0.
    assert "non-finite" in retention_failures(original, current, .05)[0]


def test_checkpoint_rank_includes_gene_response_even_when_strict_scores_tie():
    from dataclasses import replace
    from morphovoxel.training.transition import transition_rank
    from morphovoxel.validation import ValidationReport, ValidationTrial, build_gene_response_panel

    case = build_gene_response_panel(style_seeds=[42], fire_seeds=[71])[0]
    ignored = ValidationTrial(case, 256, 64, True, False, 0., ("state_bound",),
                              dict(target_iou=.9, gene_response_edited_voxels=12., gene_response_accuracy=0.), {})
    responsive = replace(ignored, metrics={**ignored.metrics, "target_iou": .8, "gene_response_accuracy": .7})
    assert transition_rank(ValidationReport((responsive,))) > transition_rank(ValidationReport((ignored,)))
    # Rasterization can make a gene pair identical. Such pairs don't get a
    # perfect-response bonus or a zero-response penalty in checkpoint ranking.
    skipped = replace(ignored, metrics=dict(target_iou=.9, gene_response_edited_voxels=0.))
    assert transition_rank(ValidationReport((skipped,))) == (0., .9)


def test_variation_rehearses_original_specialist_separately_from_new_styles(tmp_path):
    from morphovoxel.model_3d import NeuralCA3D

    source = tmp_path / "specialist.pt"
    genome = TreeGenome(family="weeping", style_seed=42).with_values({"height": .3})
    save_checkpoint(source, NeuralCA3D(6, 4, 0, 1), step=8000, config={
        "model_kind": "tree_specialist", "tree_genome": genome.to_dict(),
        "hidden_channels": 1, "hidden_layers": [4], "materials": 4, "fire_rate": 1.,
    })
    config = dict(run_name="variation", runs_root=str(tmp_path), model_kind="tree_family",
                  family_curriculum="variation", initialize_from_checkpoint=str(source),
                  device="cpu", world_size=12, batch_size=8, pool_size=16,
                  iterations=8, rollout_steps=1, persistence_steps=1, validation_steps=0,
                  family_style_seeds=[970806], fresh_fraction=.25)
    run = trainer.train(config, dimensions=3, conditional=True)
    payload = torch.load(run / "checkpoints/latest.pt", weights_only=False)
    replay = payload["transition_state"]["pools"]["neutral"]
    assert replay["style_seeds"].eq(42).all()
    assert replay["genomes"][:, 4].eq(.3).all()
    assert set(replay["genomes"][:, :4].argmax(1).tolist()) == {3}  # weeping
    assert replay["ages"].max() == 2
    logs = pd.read_csv(run / "logs.csv")
    assert logs.neutral_rehearsal_fraction.tolist() == [1, 0, 0, 0, 1, 0, 0, 0]
    assert not logs.transition_fraction.any()
    assert payload["transition_state"]["cursors"]["transition"] == 24
    # Resume visits mature bases as well as reseeded examples; it cannot shrink
    # the replay coverage to this new run's update budget or style list.
    config.pop("initialize_from_checkpoint")
    resumed = trainer.train({**config, "resume": str(run / "checkpoints/latest.pt"), "iterations": 1},
                            dimensions=3, conditional=True)
    updated = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
    assert updated["transition_state"]["pools"]["neutral"]["ages"].max() == 4


def test_legacy_variation_recovers_original_reference_and_handoffs_keep_it(tmp_path):
    from morphovoxel.checkpointing import initialize_tree_family
    from morphovoxel.model_3d import NeuralCA3D
    from morphovoxel.training.transition import base_reference

    original = tmp_path / "specialist.pt"
    specialist = NeuralCA3D(6, 4, 0, 1)
    genome = TreeGenome(family="weeping", style_seed=42).with_values({"height": .4})
    save_checkpoint(original, specialist, step=8000, config={
        "model_kind": "tree_specialist", "tree_genome": genome.to_dict(),
    })
    model = NeuralCA3D(6, 4, TreeGenome.model_size(), 1)
    initialize_tree_family(original, model)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        model.update[-1].bias.add_(1)
    current = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "variation.pt"
    save_checkpoint(path, model, step=5000, config={
        "model_kind": "tree_gene", "family_curriculum": "variation", "iterations": 8000,
        "initialize_from_checkpoint": str(original),
    })
    payload = torch.load(path, weights_only=False)
    recipe = replay_recipe(payload, {})
    assert recipe["neutral_style_seeds"] == [42]
    assert recipe["genome_span"] == 1  # New styles/genes do not redefine the base.
    rng = torch.get_rng_state()
    reference = base_reference(payload, path, model, recipe)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    torch.testing.assert_close(model.state_dict(), current)
    torch.testing.assert_close(reference["model"], expected)
    assert reference["checkpoint"] == str(original) and reference["step"] == 8000
    assert reference["recipe"]["neutral_genome"] == genome.to_dict()
    original.unlink()
    payload["transition_state"] = {"base_reference": reference}
    carried = base_reference(payload, path, model, recipe)
    torch.testing.assert_close(carried.pop("model"), reference["model"])
    assert carried == {key: value for key, value in reference.items() if key != "model"}


def test_legacy_resume_does_not_leave_an_unchecked_best_advertised(tmp_path, monkeypatch):
    from morphovoxel.validation import ValidationReport, ValidationTrial

    qualities = iter((.9, .5, .5))  # Original base, varied panel, retained base.
    def validation(model, panel, **kwargs):
        quality = next(qualities)
        return ValidationReport(tuple(ValidationTrial(
            case, 1, 1, True, False, 0., ("state_bound",),
            dict(target_iou=quality, material_accuracy=quality, late_drift=0., finite_state=1.), {},
        ) for case in panel))
    monkeypatch.setattr(trainer, "validate_panel", validation)
    config = dict(run_name="legacy", runs_root=str(tmp_path), model_kind="tree_family", family_curriculum="variation",
                  device="cpu", world_size=12, batch_size=2, materials=4, hidden_channels=1, model_width=4,
                  fire_rate=1, iterations=1, rollout_steps=1, persistence_steps=1,
                  validation_steps=1, validation_recovery_steps=1, validation_every=1,
                  validation_fire_seeds=[71], family_style_seeds=[0])
    path = tmp_path / "legacy/checkpoints/best.pt"
    model = TreeFamilyNCA3D(6, 4, TreeGenome.model_size(), 1, 12)
    save_checkpoint(path, model, step=2, config=config, validation={"curriculum_stage": "variation"})
    run = trainer.train({**config, "resume": str(path)}, dimensions=3, conditional=True)
    assert not path.exists()
    preserved = list(path.parent.glob("best_before_retention_2_*.pt"))
    assert len(preserved) == 1 and torch.load(preserved[0], weights_only=False)["step"] == 2
    latest = torch.load(run / "checkpoints/latest.pt", weights_only=False)
    assert latest["step"] == 3 and not latest["validation"]["retention"]["eligible"]
    assert latest["validation"]["best_worst_genome_persistence_score"] is None


@pytest.mark.parametrize("upgrade", ["targets", "validation"])
def test_target_or_validation_upgrade_remeasures_retention_and_archives_stale_best(tmp_path, monkeypatch, upgrade):
    from morphovoxel.targets.targets_3d import TREE_TARGET_VERSION
    from morphovoxel.validation import ValidationReport, ValidationTrial

    calls = []
    def validation(model, panel, **kwargs):
        calls.append(panel[0].category)
        return ValidationReport(tuple(ValidationTrial(
            case, 1, 1, True, False, 0., ("state_bound",),
            dict(target_iou=.7, material_accuracy=.8, late_drift=0., finite_state=1.), {},
        ) for case in panel))
    monkeypatch.setattr(trainer, "validate_panel", validation)
    config = dict(run_name="upgrade", runs_root=str(tmp_path), model_kind="tree_family", family_curriculum="variation",
                  device="cpu", world_size=12, batch_size=2, pool_size=8, materials=4, hidden_channels=1, model_width=4,
                  fire_rate=1, iterations=1, rollout_steps=1, persistence_steps=1,
                  validation_steps=1, validation_recovery_steps=1, validation_every=1,
                  validation_fire_seeds=[71], family_style_seeds=[0])
    run = trainer.train(config, dimensions=3, conditional=True)
    path = run / "checkpoints/latest.pt"
    payload = torch.load(path, weights_only=False)
    if upgrade == "targets":
        payload["metadata"]["target_generator_version"] = 4
    else:
        payload["config"]["tree_validation_version"] = 1
    state = payload["transition_state"]
    state["best_rank"] = [1., 1.]
    state["retention"]["settings"].pop("target_generator_version")
    for value in state["retention"]["baseline"].values():
        value["target_iou"] = .99
    if upgrade == "targets":
        for pool in [payload["pool"], *state["pools"].values()]:
            pool["target_occupancy"].fill_(-99)
            pool["ages"].fill_(100_000)
    torch.save(payload, path)
    torch.save(payload, run / "checkpoints/best.pt")
    calls.clear()
    trainer.train({**config, "resume": str(path)}, dimensions=3, conditional=True)
    updated = torch.load(path, weights_only=False)
    assert calls.count("neutral") == 2  # Original baseline and current retention.
    assert updated["step"] == 2
    assert {int(value["step"]) for value in updated["optimizer"]["state"].values()} == {2}
    for pool in [updated["pool"], *updated["transition_state"]["pools"].values()]:
        assert pool["target_occupancy"].min() >= 0 and pool["ages"].max() < 100_000
    guard = updated["transition_state"]["retention"]
    assert guard["settings"]["target_generator_version"] == TREE_TARGET_VERSION
    assert all(value["target_iou"] == .7 for value in guard["baseline"].values())
    torch.testing.assert_close(updated["transition_state"]["base_reference"], state["base_reference"])
    assert len(list(path.parent.glob(f"best_before_{upgrade}_1_*.pt"))) == 1
    assert torch.load(path.parent / "best.pt", weights_only=False)["step"] == 2
