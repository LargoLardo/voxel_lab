import json

import pytest
import torch

from morphovoxel.training import train


def test_cpu_smoke_training_2d_and_3d(tmp_path):
    common = {"runs_root": str(tmp_path), "device": "cpu", "iterations": 1, "rollout_steps": [1, 1], "batch_size": 1, "hidden_channels": 2, "model_width": 4, "fire_rate": 1.0, "live_preview": True}
    two = train({**common, "run_name": "2d", "world_size": 12, "materials": 3, "iterations": 10}, dimensions=2)
    three = train({**common, "run_name": "3d", "world_size": 10, "materials": 4}, dimensions=3)
    assert (two / "checkpoints" / "latest.pt").exists()
    assert json.loads((two / "visualizations" / "live.json").read_text())["iteration"] == 10
    assert (three / "checkpoints" / "latest.pt").exists()
    assert not (three / "visualizations" / "live.png").exists()


def test_cpu_smoke_training_continuous_tree_family(tmp_path):
    config = {
        "run_name": "tree_family_smoke", "runs_root": str(tmp_path),
        "model_kind": "tree_family", "dimensions": 3, "conditional": True,
        "device": "cpu", "seed": 4, "world_size": 16, "batch_size": 2,
        "pool_size": 4, "materials": 4, "hidden_channels": 2,
        "model_width": 8, "fire_rate": 1, "iterations": 1,
        "rollout_steps": 1, "persistence_steps": 1,
        "validation_steps": 0, "capture_every": 1,
    }
    run = train(config, dimensions=3, conditional=True)
    payload = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["config"]["model_kind"] == "tree_family"
    assert payload["pool"]["target_occupancy"].shape[0] == config["pool_size"]
    assert payload["pool"]["environments"].shape[1] == 12
    assert payload["pool"]["style_seeds"].shape[0] == config["pool_size"]

    resumed = train(
        {
            **config, "run_name": "tree_family_resized", "pool_size": 6,
            "resume": str(run / "checkpoints" / "latest.pt"),
        },
        dimensions=3,
        conditional=True,
    )
    resized = torch.load(resumed / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert all(len(value) == 6 for value in resized["pool"].values())

    broken = tmp_path / "missing-environment-specs.pt"
    payload["pool"].pop("environment_specs")
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="missing paired target/environment/style data"):
        train(
            {**config, "run_name": "broken_resume", "resume": str(broken)},
            dimensions=3,
            conditional=True,
        )


def test_tree_family_without_context_uses_default_environment_compatibility_path(tmp_path):
    run = train({
        "run_name": "tree_family_no_context", "runs_root": str(tmp_path),
        "model_kind": "tree_family", "environment_conditioning": False,
        "device": "cpu", "seed": 5, "world_size": 16, "batch_size": 2,
        "pool_size": 2, "materials": 4, "hidden_channels": 2,
        "model_width": 8, "fire_rate": 1, "iterations": 1,
        "rollout_steps": 1, "persistence_steps": 0, "validation_steps": 0,
    }, dimensions=3, conditional=True)
    payload = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["metadata"]["context_channels"] == 0
    assert payload["pool"]["environment_specs"].shape[0] == 2


def test_gradient_accumulation_keeps_iterations_as_optimizer_updates(tmp_path):
    run = train({
        "run_name": "accumulated", "runs_root": str(tmp_path),
        "model_kind": "tree_family", "environment_conditioning": False,
        "device": "cpu", "seed": 6, "world_size": 12, "batch_size": 2,
        "pool_size": 4, "fresh_fraction": 0, "materials": 4, "hidden_channels": 2,
        "model_width": 4, "fire_rate": 1, "iterations": 2,
        "rollout_steps": 1, "persistence_steps": 0, "validation_steps": 0,
        "gradient_accumulation": True, "gradient_accumulation_steps": 3,
    }, dimensions=3, conditional=True)
    payload = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    optimizer_steps = {int(value["step"]) for value in payload["optimizer"]["state"].values()}
    assert optimizer_steps == {2}
    assert payload["pool"]["ages"].tolist() == [3, 3, 3, 3]
    assert len((run / "metrics" / "per_step.csv").read_text().splitlines()) == 3


def test_gradient_accumulation_yaml_values_are_validated(tmp_path):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train({
            "run_name": "bad_accumulation", "runs_root": str(tmp_path),
            "device": "cpu", "iterations": 1, "gradient_accumulation": True,
            "gradient_accumulation_steps": 0,
        }, dimensions=2)
