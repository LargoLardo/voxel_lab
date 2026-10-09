import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from morphovoxel.genomes import MORPHOLOGIES
from morphovoxel.state import StateLayout
from morphovoxel.training.trainer import (
    _keep_viable_damage,
    _guard_tree_family_cuda_memory,
    _pool_actions,
    _restore_rng_state,
    _step_range,
    _training_horizons,
    _validate_persistence,
    train,
)


@pytest.mark.parametrize("kind,mode", [("tree_specialist", None), ("tree_family", "basics"), ("tree_family", None)])
def test_tree_best_uses_shape_quality_on_zero_score_ties_and_resume(tmp_path, monkeypatch, kind, mode):
    from morphovoxel.training import trainer
    from morphovoxel.validation import ValidationReport, ValidationTrial

    qualities = iter((.8, .2, .1, .9))
    def validation(model, panel, **kwargs):
        quality = next(qualities)
        return ValidationReport(tuple(ValidationTrial(
            case, 1, 1, True, False, 0., ("state_bound",), {"target_iou": quality}, {},
        ) for case in panel))
    monkeypatch.setattr(trainer, "validate_panel", validation)
    config = dict(run_name="ranked", runs_root=str(tmp_path), model_kind=kind,
                  device="cpu", world_size=12, batch_size=2, materials=4, hidden_channels=1,
                  model_width=4, fire_rate=1., environment_conditioning=False, iterations=2,
                  rollout_steps=1, persistence_steps=0, validation_steps=1, validation_every=1,
                  validation_recovery_steps=1, validation_fire_seeds=[1], family_style_seeds=[0])
    if mode:
        config["family_curriculum"] = mode
    run = train(config, dimensions=3, conditional=kind == "tree_family")
    best = run / "checkpoints/best.pt"
    assert torch.load(best, weights_only=False)["step"] == 1
    for expected in (1, 4):
        train({**config, "resume": str(run / "checkpoints/latest.pt"), "iterations": 1},
              dimensions=3, conditional=kind == "tree_family")
        assert torch.load(best, weights_only=False)["step"] == expected


@pytest.mark.parametrize("external_resume", [False, True])
def test_reusing_run_directory_cannot_overwrite_configuration_or_weights(tmp_path, external_resume):
    from shutil import copyfile

    config = dict(run_name="protected", runs_root=str(tmp_path), device="cpu", world_size=12,
                  batch_size=1, materials=3, hidden_channels=1, model_width=4,
                  iterations=1, rollout_steps=1, validation_steps=0)
    run = train(config, dimensions=2)
    paths = [run / "config.yaml", run / "checkpoints/latest.pt"]
    before = [path.read_bytes() for path in paths]
    if external_resume:
        other = tmp_path / "unrelated.pt"
        copyfile(paths[1], other)
        config["resume"] = str(other)
    else:
        config["model_width"] = 6
    with pytest.raises(FileExistsError, match="Choose a new run_name"):
        train(config, dimensions=2)
    assert [path.read_bytes() for path in paths] == before


def test_step_range_rejects_negative_scalar():
    try:
        _step_range(-1, (1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("negative steps must be rejected")


def test_differentiable_horizon_cap_preserves_both_training_losses():
    random.seed(7)
    for _ in range(20):
        growth, persistence = _training_horizons((24, 64), (32, 96), 96)
        assert 24 <= growth <= 64
        assert 32 <= persistence <= 96
        assert growth + persistence <= 96


def test_tree_family_cuda_guard_rejects_the_32_cubed_batch_eight_workload(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=6 * 1024**3),
    )
    with pytest.raises(ValueError, match="world_size: 32 use batch_size: 2"):
        _guard_tree_family_cuda_memory(
            device=torch.device("cuda"), dimensions=3, world_size=32, batch_size=8,
            rollout_maximum=48, persistence_maximum=32, differentiable_step_limit=None,
        )


def test_resume_rng_state_continues_python_numpy_and_torch_streams():
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(30)
    np.random.seed(40)
    torch.manual_seed(50)

    _restore_rng_state(state)

    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected


def test_resume_rng_state_normalizes_cuda_bytes_to_cpu(monkeypatch):
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.extend(states))

    _restore_rng_state({"cuda": [np.arange(8, dtype=np.int64)]})

    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8
    assert restored[0].tolist() == list(range(8))


def test_pool_actions_reseed_worst_and_damage_low_loss_mature_samples():
    state = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(4, 1, 1, 1, 1)
    target = torch.ones(4, 1, 1, 1)
    ages = torch.full((4,), 100)

    reseed, damage = _pool_actions(state, target, ages, fresh_count=1, damage_fraction=0.5, mature_age=48)

    assert reseed.tolist() == [0]
    assert damage.tolist() == [3, 2]


def test_pool_actions_reseed_dead_samples_and_exclude_them_from_damage():
    state = torch.tensor([0.0, 0.0, 0.8, 1.0]).view(4, 1, 1, 1, 1)
    target = torch.ones(4, 1, 1, 1)
    ages = torch.full((4,), 100)

    reseed, damage = _pool_actions(state, target, ages, fresh_count=1, damage_fraction=1.0, mature_age=48)

    assert set(reseed.tolist()) == {0, 1}
    assert not set(reseed.tolist()) & set(damage.tolist())


def test_lethal_damage_keeps_the_original_living_sample():
    original = torch.zeros(1, 3, 5, 5, 5)
    original[0, 0, 2, 2, 2] = 1

    kept = _keep_viable_damage(original, torch.zeros_like(original))

    assert torch.equal(kept, original)


def test_lethal_damage_is_rejected_for_both_members_of_a_counterfactual_pair():
    from morphovoxel.damage import damage_3d

    original = torch.zeros(4, 3, 5, 5, 5)
    original[:, :, 0, 2, 2] = 1
    original[1:, :, 4, 2, 2] = 1
    damaged, _ = damage_3d(original, .4, "top")
    assert not damaged[0].any() and damaged[1:].any()
    kept = _keep_viable_damage(original, damaged, paired=True)
    torch.testing.assert_close(kept[:2], original[:2])
    torch.testing.assert_close(kept[2:], damaged[2:])
    torch.testing.assert_close(_keep_viable_damage(original, damaged)[1:], damaged[1:])
    with pytest.raises(ValueError, match="even-sized pairs"):
        _keep_viable_damage(original[:1], damaged[:1], paired=True)


class _IdentityCA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, genome=None):
        self.calls += 1
        return state


def test_validation_rolls_every_genome_to_requested_horizon():
    model = _IdentityCA()
    worst, scores = _validate_persistence(
        model,
        dimensions=3,
        conditional=True,
        size=10,
        layout=StateLayout(materials=4, hidden=2),
        device=torch.device("cpu"),
        target_seed=7,
        validation_seed=8,
        total_steps=8,
        start_step=4,
        interval=2,
    )

    assert model.calls == 8
    assert set(scores) == set(MORPHOLOGIES)
    assert worst == min(scores.values())


def test_conditional_training_updates_best_checkpoint_on_tied_scores(tmp_path, monkeypatch):
    score = .75
    monkeypatch.setattr(
        "morphovoxel.training.trainer._validate_persistence",
        lambda *args, **kwargs: (score, {name: score for name in MORPHOLOGIES}),
    )
    run = train(
        {
            "run_name": "persistence",
            "runs_root": str(tmp_path),
            "device": "cpu",
            "seed": 3,
            "world_size": 10,
            "batch_size": 4,
            "materials": 4,
            "hidden_channels": 2,
            "model_width": 4,
            "fire_rate": 1.0,
            "iterations": 2,
            "rollout_steps": [1, 1],
            "persistence_steps": [1, 1],
            "pool_size": 4,
            "validation_steps": 4,
            "validation_start": 2,
            "validation_interval": 1,
            "validation_every": 1,
        },
        dimensions=3,
        conditional=True,
    )

    checkpoint = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["validation"]["validation_steps"] == 4
    assert set(checkpoint["validation"]["per_genome"]) == set(MORPHOLOGIES)
    assert len(checkpoint["pool"]["states"]) >= len(MORPHOLOGIES)
    assert (run / "metrics" / "persistence_validation.csv").is_file()

    # Resuming an older file must compare against the actual incumbent best.
    before = (run / "checkpoints/best.pt").read_bytes()
    checkpoint["validation"]["best_worst_genome_persistence_score"] = .1
    older = run / "checkpoints/older.pt"
    torch.save(checkpoint, older)
    score = .25
    resumed_config = {**checkpoint["config"], "iterations": 1, "resume": str(older)}
    train(resumed_config, dimensions=3, conditional=True)
    assert (run / "checkpoints/best.pt").read_bytes() == before
    latest = torch.load(run / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert latest["validation"]["best_worst_genome_persistence_score"] == .75
    assert latest["validation"]["worst_genome_persistence_score"] == .25

    # Changing the validation horizon starts a new, incomparable ranking.
    train({**resumed_config, "validation_steps": 5}, dimensions=3, conditional=True)
    best = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    assert best["validation"]["validation_steps"] == 5
    assert best["validation"]["best_worst_genome_persistence_score"] == .25


@pytest.mark.parametrize("validation_steps", [0, 4])
def test_periodic_recovery_checkpoint_survives_failure_without_validation_improvement(tmp_path, monkeypatch, validation_steps):
    from morphovoxel.training import trainer
    calls = 0
    original = trainer.morphology_loss
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("simulated failure after update four")
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "morphology_loss", interrupted)
    scores = iter((.75, .25, .25))
    def validate(*args, **kwargs):
        # Loss breakdowns must already be on disk before long validation starts.
        assert (tmp_path / "recover/metrics/per_step.csv").is_file()
        score = next(scores)
        return score, {"tree": score}
    monkeypatch.setattr(trainer, "_validate_persistence", validate)
    config = {
        "run_name": "recover", "runs_root": str(tmp_path), "device": "cpu",
        "world_size": 10, "batch_size": 4, "materials": 4, "hidden_channels": 1,
        "model_width": 4, "fire_rate": 1.0, "iterations": 5,
        "rollout_steps": 1, "pool_size": 4, "validation_steps": validation_steps,
        "validation_every": 2, "scheduler": {"step_size": 2, "gamma": .9},
    }
    with pytest.raises(RuntimeError, match="simulated failure"):
        train(config, dimensions=3, conditional=True)
    checkpoint = tmp_path / "recover/checkpoints/latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["step"] == 4
    assert payload["pool"] is not None and payload["rng"] is not None
    assert {int(value["step"]) for value in payload["optimizer"]["state"].values()} == {4}
    assert payload["scheduler"]["last_epoch"] == 4
    assert payload["validation"] is None  # Never attach an older score to new weights.
    run = checkpoint.parent.parent
    logs = pd.read_csv(run / "logs.csv")
    assert logs.step.tolist() == [1, 2, 3, 4]
    assert {"loss", "occupancy", "material", "branch_dice", "magnitude"} <= set(logs)
    pd.testing.assert_frame_equal(logs, pd.read_csv(run / "metrics/per_step.csv"))
    if validation_steps:
        best = torch.load(checkpoint.with_name("best.pt"), map_location="cpu", weights_only=False)
        assert best["step"] == 2
        assert pd.read_csv(run / "metrics/persistence_validation.csv").step.tolist() == [2, 4]
    # An older checkpoint can have newer log rows: discard those on resume.
    pd.concat([logs, logs.tail(1).assign(step=5, loss=999)]).to_csv(run / "logs.csv", index=False)
    monkeypatch.setattr(trainer, "morphology_loss", original)
    train({**config, "resume": str(checkpoint), "iterations": 1}, dimensions=3, conditional=True)
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed["step"] == 5 and resumed["scheduler"]["last_epoch"] == 5
    resumed_logs = pd.read_csv(run / "logs.csv")
    assert resumed_logs.step.tolist() == [1, 2, 3, 4, 5]
    pd.testing.assert_frame_equal(resumed_logs.iloc[:4], logs)
    assert resumed_logs.loss.iloc[-1] != 999
    if validation_steps:
        assert pd.read_csv(run / "metrics/persistence_validation.csv").step.tolist() == [2, 4, 5]
