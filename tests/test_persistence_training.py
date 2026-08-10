import torch

from morphovoxel.genomes import MORPHOLOGIES
from morphovoxel.state import StateLayout
from morphovoxel.training.trainer import _pool_actions, _step_range, _validate_persistence, train


def test_step_range_rejects_negative_scalar():
    try:
        _step_range(-1, (1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("negative steps must be rejected")


def test_pool_actions_reseed_worst_and_damage_low_loss_mature_samples():
    state = torch.tensor([4.0, 3.0, 2.0, 1.0]).view(4, 1, 1, 1, 1)
    target = torch.zeros(4, 1, 1, 1)
    ages = torch.full((4,), 100)

    reseed, damage = _pool_actions(state, target, ages, fresh_count=1, damage_fraction=0.5, mature_age=48)

    assert reseed.tolist() == [0]
    assert damage.tolist() == [3, 2]


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


def test_conditional_training_saves_scored_best_checkpoint(tmp_path):
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
            "iterations": 1,
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
    assert checkpoint["validation"]["validation_steps"] == 4
    assert set(checkpoint["validation"]["per_genome"]) == set(MORPHOLOGIES)
    assert len(checkpoint["pool"]["states"]) >= len(MORPHOLOGIES)
    assert (run / "metrics" / "persistence_validation.csv").is_file()
