"""Exercise the actual Metal backend; skipped on machines without an Apple GPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from morphovoxel.checkpointing import load_checkpoint, save_checkpoint
from morphovoxel.config import load_config, save_config
from morphovoxel.damage import damage_2d, damage_3d
from morphovoxel.environment import EnvironmentSpec
from morphovoxel.genomes import TreeGenome
from morphovoxel.lab import LabSession
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.random_utils import fork_rng
from morphovoxel.seeding import seed_state
from morphovoxel.state import StateLayout
from morphovoxel.training import train
from morphovoxel.training.trainer import _restore_rng_state
from morphovoxel.validation import build_candidate_panel, validate_candidate


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Metal GPU is unavailable")


def test_mps_rng_checkpoint_and_seeded_damage(tmp_path):
    device = torch.device("mps")
    torch.manual_seed(42)
    model = NeuralCA3D(7, 4).to(device)
    checkpoint = tmp_path / "mps.pt"
    save_checkpoint(checkpoint, model)
    expected = torch.rand(8, device=device)
    payload = load_checkpoint(checkpoint, model, map_location=device)
    _restore_rng_state(payload["rng"])
    assert torch.equal(expected, torch.rand(8, device=device))

    before_cpu, before_mps = torch.get_rng_state(), torch.mps.get_rng_state()
    with pytest.raises(RuntimeError, match="trial failed"):
        with fork_rng(device):
            torch.manual_seed(99)
            torch.rand(8, device=device)
            raise RuntimeError("trial failed")
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_mps, torch.mps.get_rng_state())

    for dimensions, damage in ((2, damage_2d), (3, damage_3d)):
        kwargs = dict(dimensions=dimensions, device=device, noise=0.1, random_seed=7)
        first = seed_state(1, 12, StateLayout(4, 2), **kwargs)
        assert torch.equal(first, seed_state(1, 12, StateLayout(4, 2), **kwargs))
        damaged, mask = damage(first, 0.25, "dropout", seed=8)
        assert torch.equal(mask, damage(first, 0.25, "dropout", seed=8)[1])
        assert torch.count_nonzero(damaged[..., mask]) == 0


@pytest.mark.filterwarnings("error:index_put_with_accumulate_mps.*:UserWarning")
def test_mps_tree_pipeline(tmp_path, caplog):
    caplog.set_level("INFO", logger="morphovoxel.training.trainer")
    common = {
        "runs_root": str(tmp_path), "device": "mps", "seed": 4,
        "world_size": 16, "batch_size": 2, "pool_size": 4,
        "materials": 4, "hidden_channels": 2, "model_width": 8,
        "fire_rate": 0.5, "iterations": 1, "rollout_steps": 2,
        "persistence_steps": 1, "validation_steps": 0, "capture_every": 1,
    }
    specialist = train({**common, "run_name": "specialist", "model_kind": "tree_specialist"}, dimensions=3)
    assert "GPU operations are not guaranteed deterministic" in caplog.text
    assert json.loads((specialist / "metadata.json").read_text())["deterministic_algorithms"] is False
    config = {
        **common, "run_name": "family", "model_kind": "tree_family",
        "initialize_from_specialist": str(specialist / "checkpoints/latest.pt"),
        "fresh_fraction": 1.0, "train_light_tropism": True,
        "environment_start_fraction": 0.0,
        "gradient_accumulation": True, "gradient_accumulation_steps": 2,
    }
    family = train(config, dimensions=3, conditional=True)
    config.pop("initialize_from_specialist")
    resumed = train({
        **config, "run_name": "resumed", "resume": str(family / "checkpoints/latest.pt"),
        "fresh_fraction": 0.0, "damage_probability": 1.0, "damage_min_age": 0,
        "damage_types": ["dropout"],
    }, dimensions=3, conditional=True)
    payload = torch.load(resumed / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    specs = payload["pool"]["environment_specs"]
    assert specs.dtype == torch.float64
    assert all(torch.equal(EnvironmentSpec.from_vector(row).vector(), row) for row in specs)
    assert all(bool(torch.isfinite(value).all()) for value in payload["model"].values())
    assert payload["rng"]["mps"] is not None

    lab = LabSession.from_run(resumed, "mps")
    lab.advance(2)
    assert lab.state.device.type == "mps"
    assert lab.frame_png("top")
    case = build_candidate_panel(TreeGenome(), fire_seeds=(3,), environments=(EnvironmentSpec(),))[0]
    before = torch.mps.get_rng_state()
    kwargs = dict(layout=lab.layout, world_size=16, steps=4, recovery_steps=2, device="mps")
    trial = validate_candidate(lab.model, case, **kwargs)
    assert torch.equal(before, torch.mps.get_rng_state())
    assert trial.to_dict() == validate_candidate(lab.model, case, **kwargs).to_dict()
    assert trial.metrics["finite_state"] == 1

    ecology = load_config(Path(__file__).parents[1] / "configs/smoke_tree_ecology.yaml")
    ecology.update(device="mps", runs_root=str(tmp_path), checkpoint=str(resumed / "checkpoints/latest.pt"))
    ecology_config = tmp_path / "ecology.yaml"
    save_config(ecology, ecology_config)
    subprocess.run([sys.executable, "-m", "morphovoxel.run_ecology", "--config", str(ecology_config)], check=True)
    metadata = json.loads((tmp_path / "smoke_tree_ecology/metadata.json").read_text())
    assert metadata["device"] == "mps"
