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
