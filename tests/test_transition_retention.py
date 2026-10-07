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


def test_best_transition_requires_retention_and_improved_edits(tmp_path, monkeypatch):
    from morphovoxel.validation import ValidationReport, ValidationTrial

    retained_values = iter((.8, .8, .65, .78, .78))  # Baseline, then four validations.
    edit_values = iter((.3, .9, .4, .2))
    def fake_validation(model, panel, **kwargs):
        retention = panel[0].category in {"neutral", "variation"}
        quality = next(retained_values if retention else edit_values)
        return ValidationReport(tuple(ValidationTrial(
            case, 1, 1, True, False, 0., ("state_bound",),
            {"target_iou": quality, "material_accuracy": .8, "late_drift": .01, "finite_state": 1.,
             **({} if retention else {"transition_edited_voxels": 12., "transition_edit_accuracy": quality})}, {},
        ) for case in panel))
    monkeypatch.setattr(trainer, "validate_panel", fake_validation)
    config = dict(run_name="guarded", runs_root=str(tmp_path), model_kind="tree_family", family_curriculum="transition",
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
