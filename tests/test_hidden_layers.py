import pytest
import torch

from morphovoxel.checkpointing import CheckpointCompatibilityError, initialize_tree_family, load_checkpoint, save_checkpoint
from morphovoxel.config import load_config, resolve_hidden_layers
from morphovoxel.genomes import TREE_FAMILIES, TreeGenome, tree_genome_tensor
from morphovoxel.lab import LabSession
from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D, TreeFamilyNCA3D
from morphovoxel.rollout import rollout
from morphovoxel.training.trainer import train


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable",
))]


@pytest.mark.parametrize("model_type,dimensions", [(NeuralCA2D, 2), (NeuralCA3D, 3), (TreeFamilyNCA3D, 3)])
@pytest.mark.parametrize("device", DEVICES)
def test_multiple_hidden_layers_train_and_keep_the_legacy_single_layer(model_type, dimensions, device):
    genome_size = 15 if model_type is TreeFamilyNCA3D else 0
    model = model_type(5, hidden_layers=[7, 5, 3], genome_size=genome_size, fire_rate=1).to(device)
    genome = tree_genome_tensor([TreeGenome(family=f) for f in TREE_FAMILIES], device=device) if genome_size else None
    state = torch.rand(4, 5, *([5] * dimensions), device=device, requires_grad=True)
    final, _ = rollout(model, state, 3, genome)
    assert final.shape == state.shape
    final.square().mean().backward()
    assert torch.isfinite(state.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    original = model_type(5, hidden=7, genome_size=genome_size, fire_rate=1).to(device)
    explicit = model_type(5, hidden_layers=[7], genome_size=genome_size, fire_rate=1).to(device)
    explicit.load_state_dict(original.state_dict(), strict=True)
    torch.testing.assert_close(original(state, genome), explicit(state, genome))
    assert not any(name.startswith("hidden_update.") for name in original.state_dict())


@pytest.mark.parametrize("layers", [[], 32, "32,32", [0], [-1], [True], [32.5]])
def test_invalid_layer_specifications_fail_clearly(layers):
    with pytest.raises(ValueError, match="hidden_layers.*positive integers"):
        resolve_hidden_layers(64, layers)


def test_two_by_32_family_has_the_expected_size():
    model = TreeFamilyNCA3D(13, context_channels=12, hidden_layers=[32, 32])
    assert model.hidden_layers == (32, 32)
    assert sum(parameter.numel() for parameter in model.parameters()) == 8340


@pytest.mark.parametrize("device", DEVICES)
def test_deep_specialist_handoff_preserves_behavior_and_rejects_other_architectures(tmp_path, device):
    specialist = NeuralCA3D(5, hidden_layers=[7, 5], fire_rate=1).to(device)
    path = tmp_path / "specialist.pt"
    save_checkpoint(path, specialist, config={"model_kind": "tree_specialist"})
    family = TreeFamilyNCA3D(5, hidden_layers=[7, 5], context_channels=2, fire_rate=1).to(device)
    initialize_tree_family(path, family)
    state = torch.rand(4, 5, 5, 5, 5, device=device)
    genomes = tree_genome_tensor([TreeGenome(family=f) for f in TREE_FAMILIES], device=device)
    torch.testing.assert_close(family(state, genomes, torch.zeros(4, 2, 5, 5, 5, device=device)), specialist(state))
    torch.testing.assert_close(family.hidden_update[0].weight, specialist.update[2].weight)
    assert torch.load(path, weights_only=False, map_location="cpu")["config"]["hidden_layers"] == [7, 5]
    with pytest.raises(CheckpointCompatibilityError, match="matching hidden_layers"):
        load_checkpoint(path, NeuralCA3D(5, hidden_layers=[7, 7], fire_rate=1))


@pytest.mark.parametrize("device", DEVICES)
def test_deep_training_checkpoint_handoff_resume_and_lab(tmp_path, device, monkeypatch):
    from morphovoxel.training import trainer

    config = {
        "runs_root": str(tmp_path), "device": device, "world_size": 12,
        "materials": 4, "hidden_channels": 1, "hidden_layers": [4, 3],
        "batch_size": 2, "pool_size": 2, "fire_rate": 1,
        "iterations": 1, "rollout_steps": 1, "persistence_steps": 1,
        "validation_steps": 0,
    }
    specialist = train({**config, "model_kind": "tree_specialist", "run_name": "specialist"}, dimensions=3)
    family_config = {
        **config, "model_kind": "tree_family", "family_curriculum": "basics", "conditional": True,
    }
    family = train({
        **family_config, "run_name": "family", "initialize_from_checkpoint": str(specialist / "checkpoints/latest.pt"),
    }, dimensions=3, conditional=True)
    sample_family = trainer.sample_counterfactual_family_data
    sampled_counts = []

    def sample(pair_count, *args, **kwargs):
        sampled_counts.append(pair_count)
        return sample_family(pair_count, *args, **kwargs)

    monkeypatch.setattr(trainer, "sample_counterfactual_family_data", sample)
    resumed = train({
        **family_config, "run_name": "resumed", "resume": str(family / "checkpoints/latest.pt"),
    }, dimensions=3, conditional=True)
    assert all(count < 32 for count in sampled_counts)  # No discarded replacement of the saved 64-entry pool.
    payload = torch.load(resumed / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert load_config(resumed / "config.yaml")["hidden_layers"] == [4, 3]
    assert all(value["step"] == 2 for value in payload["optimizer"]["state"].values())
    lab = LabSession.from_run(resumed, device)
    assert lab.model.hidden_layers == (4, 3)
    assert lab.advance(2)["steps"] == 2
    if device == "cpu":
        for pool_size, reset in ((66, False), (64, True)):
            sampled_counts.clear()
            run = train({
                **family_config, "run_name": f"pool_{pool_size}", "pool_size": pool_size,
                "resume": str(family / "checkpoints/latest.pt"), "reset_pool_on_resume": reset,
            }, dimensions=3, conditional=True)
            assert sampled_counts[0] == pool_size // 2
            saved = torch.load(run / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
            assert len(saved["pool"]["states"]) == pool_size
