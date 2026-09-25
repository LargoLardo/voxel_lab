from collections import Counter

import pytest
import torch

from morphovoxel.checkpointing import CheckpointCompatibilityError, save_checkpoint
from morphovoxel.genomes import TREE_FAMILIES, TREE_GENE_SPECS, TreeGenome
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.state import StateLayout
from morphovoxel.targets.targets_3d import TREE_TARGET_VERSION
from morphovoxel.training.family import curriculum_sampling_options, curriculum_values, sample_counterfactual_family_data, sample_family_data
from morphovoxel.training.state_pool import StatePool
from morphovoxel.training.trainer import _paired_pool_actions, train


def test_family_curriculum_widens_before_environment_randomization():
    early = curriculum_values(0, 100, {})
    late = curriculum_values(99, 100, {})
    assert early["genome_span"] < late["genome_span"] == 1
    assert early["environment_span"] == 0 < late["environment_span"]
    assert early["mutation_fraction"] == 0 < late["mutation_fraction"]


def test_family_samples_keep_genome_target_environment_and_seed_paired():
    parent = TreeGenome.random(8, family="branching")
    data = sample_family_data(
        4, 16, 10, genome_span=0.5, interpolation_fraction=0.5,
        mutation_fraction=0.5, environment_span=0.5, parent=parent,
    )
    assert data.model_genomes.shape == (4, TreeGenome.model_size())
    assert data.target_occupancy.shape == (4, 16, 16, 16)
    assert data.environments.shape[:2] == (4, 12)
    assert torch.equal(data.style_seeds, torch.tensor([genome.style_seed for genome in data.genomes]))
    assert set(data.creation_methods) <= {"interpolation", "mutation"}


def test_initial_family_samples_are_balanced_at_narrow_span():
    first = sample_family_data(
        8, 16, 20, genome_span=0.01, interpolation_fraction=0, mutation_fraction=0,
    )
    second = sample_family_data(
        8, 16, 20, genome_span=0.01, interpolation_fraction=0, mutation_fraction=0,
    )
    families = [genome.family for genome in first.genomes]
    assert Counter(families) == {family: 2 for family in TREE_FAMILIES}
    assert families == [genome.family for genome in second.genomes]

    refreshed = sample_family_data(
        16, 16, 21, genome_span=0.01, interpolation_fraction=0,
        mutation_fraction=0, parent=TreeGenome(family="branching"),
    )
    assert {genome.family for genome in refreshed.genomes} == set(TREE_FAMILIES)


def test_family_samples_have_nonempty_targets():
    data = sample_family_data(
        32, 16, 30, genome_span=1, interpolation_fraction=0.25,
        mutation_fraction=0.25, environment_span=1,
    )
    occupied_cells = torch.count_nonzero(data.target_occupancy.flatten(1), dim=1)
    assert bool((occupied_cells > 0).all())


def test_counterfactual_samples_enforce_minimum_positive_branch_and_leaf_masks():
    from morphovoxel.training.family import sample_counterfactual_family_data

    data = sample_counterfactual_family_data(
        32, 16, 31, minimum_branch_voxels=8, minimum_leaf_voxels=8,
    )
    for material, minimum in ((2, 8), (3, 8)):
        counts = torch.count_nonzero(data.target_materials == material, dim=(1, 2, 3))
        assert bool(((counts == 0) | (counts >= minimum)).all())


def test_family_replacements_can_preserve_pool_family_balance():
    requested = list(TREE_FAMILIES)
    data = sample_family_data(
        len(requested), 16, 35, genome_span=1, interpolation_fraction=0.25,
        mutation_fraction=0.25, families=requested,
    )
    assert [genome.family for genome in data.genomes] == requested


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable",
))])
def test_routine_refresh_reaches_every_healthy_condition_across_pool_resume(device):
    conditions = torch.arange(32).repeat_interleave(2)
    targets = torch.ones(64, 2, 2, 2)
    pool = StatePool(
        targets[:, None], conditions[:, None].float(), target_occupancy=targets,
        target_materials=targets.long() * 2, condition_ids=conditions, pair_ids=conditions,
    )
    refreshed = Counter()
    # One initial pool traversal leaves equal ages; four more refresh every slot.
    for update in range(40):
        if update == 16:
            pool = StatePool(**pool.state_dict())
        batch = pool.sample_stratified_pairs(8, update * 4, device)
        reseed, damage = _paired_pool_actions(batch.states, batch.target_occupancy, batch.ages, 1, 0, 48)
        assert len(reseed) == 2 and len(damage) == 0
        refreshed.update(batch.condition_ids[reseed][::2].tolist())
        batch.ages[reseed] = 0
        pool.commit(batch, batch.states, 60)
    assert set(refreshed) == set(range(32))
    assert max(refreshed.values()) == 2
    torch.testing.assert_close(pool.ages[::2], pool.ages[1::2])


@pytest.mark.parametrize("span,expected_calls", [(0, 4), (.6, 8)])
def test_counterfactual_targets_are_generated_once_and_reused(monkeypatch, span, expected_calls):
    from morphovoxel.targets import make_tree_target
    from morphovoxel.training import family

    calls = []
    def counted(*args):
        calls.append(args)
        return make_tree_target(*args)
    monkeypatch.setattr(family, "make_tree_target", counted)
    data = sample_counterfactual_family_data(4, 16, 42, genome_span=span, condition_ids=[0, 8, 16, 24])
    assert len(calls) == expected_calls
    for index, (genome, environment) in enumerate(zip(data.genomes, data.environment_specs)):
        target, material = make_tree_target(genome, 16, environment)
        torch.testing.assert_close(data.target_occupancy[index], torch.from_numpy(target))
        torch.testing.assert_close(data.target_materials[index], torch.from_numpy(material))


def test_parent_mutations_reflect_instead_of_clipping_to_gene_limits():
    parent = TreeGenome(family="branching", genes=(1.0,) * len(TREE_GENE_SPECS))
    data = sample_family_data(
        8, 16, 40, mutation_fraction=1, interpolation_fraction=0,
        mutation_strength=1, parent=parent,
    )
    assert all(-1 < value < 1 for genome in data.genomes for value in genome.genes)


def test_tree_family_training_can_initialize_from_specialist_checkpoint(tmp_path):
    layout = StateLayout(4, 2)
    specialist = NeuralCA3D(layout.channels, 8, 0, 1.0)
    checkpoint = tmp_path / "specialist.pt"
    save_checkpoint(checkpoint, specialist, config={"model_kind": "tree_specialist"})
    run = train({
        "run_name": "converted", "runs_root": str(tmp_path),
        "model_kind": "tree_family", "dimensions": 3, "conditional": True,
        "initialize_from_specialist": str(checkpoint), "device": "cpu",
        "world_size": 16, "batch_size": 2, "pool_size": 2,
        "materials": 4, "hidden_channels": 2, "model_width": 8,
        "fire_rate": 1.0, "iterations": 1, "rollout_steps": 1,
        "persistence_steps": 0, "validation_steps": 0,
    }, dimensions=3, conditional=True)
    payload = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["metadata"]["model_kind"] == "tree_family"


def test_tree_family_initialization_rejects_a_non_tree_specialist(tmp_path):
    layout = StateLayout(4, 2)
    checkpoint = tmp_path / "generic-specialist.pt"
    save_checkpoint(
        checkpoint,
        NeuralCA3D(layout.channels, 8, 0, 1.0),
        config={"model_kind": "specialist"},
    )

    with pytest.raises(CheckpointCompatibilityError, match="expected 'tree_specialist'"):
        train({
            "run_name": "wrong-conversion", "runs_root": str(tmp_path),
            "model_kind": "tree_family", "initialize_from_specialist": str(checkpoint),
            "device": "cpu", "world_size": 16, "batch_size": 1, "pool_size": 2,
            "materials": 4, "hidden_channels": 2, "model_width": 8,
            "fire_rate": 1.0, "iterations": 1, "rollout_steps": 1,
            "persistence_steps": 0, "validation_steps": 0,
        }, dimensions=3, conditional=True)


def test_phase_two_samples_progress_from_neutral_to_single_genes_to_combinations():
    config = {"family_curriculum": "full", "family_style_seeds": [3, 7]}
    stages = [curriculum_values(step, 100, config) for step in (0, 24, 25, 99)]
    assert [stage["curriculum_stage"] for stage in stages] == ["basics", "basics", "variation", "variation"]
    assert all(stage["environment_span"] == 0 for stage in stages)
    samples = [
        sample_counterfactual_family_data(
            32, 16, 42, genome_span=stage["genome_span"],
            **curriculum_sampling_options(stage, config),
        ) for stage in stages
    ]
    basic, _, early, late = samples
    assert {genome.family for genome in basic.genomes} == set(TREE_FAMILIES)
    assert all(not any(genome.genes) for genome in basic.genomes)
    assert set(basic.style_seeds.tolist()) <= {3, 7}
    assert all(sum(value != 0 for value in genome.genes) <= 1 for genome in early.genomes)
    assert set(early.style_seeds.tolist()) <= {3, 7}
    assert any(sum(value != 0 for value in genome.genes) == 8 for genome in late.genomes)
    assert set(late.style_seeds.tolist()) - {3, 7}
    for data in (early, late):
        assert any(not any(genome.genes) for genome in data.genomes)
        assert any(any(genome.genes) for genome in data.genomes)
        for low, high in zip(data.genomes[::2], data.genomes[1::2]):
            assert low.family == high.family and low.style_seed == high.style_seed
            assert sum(a != b for a, b in zip(low.genes, high.genes)) <= 1
            assert low.value("light_tropism") == high.value("light_tropism") == 0
    assert curriculum_values(99, 100, {"family_curriculum": "basics"})["genome_span"] == 0
    assert curriculum_values(0, 100, {"family_curriculum": "variation"})["genome_span"] > 0


@pytest.mark.parametrize("override", [
    {"family_curriculum": "typo"}, {"basic_family_fraction": 1},
    {"neutral_fraction": -0.1}, {"combination_start_fraction": 1},
])
def test_phase_two_rejects_invalid_schedules(override):
    with pytest.raises(ValueError):
        curriculum_values(0, 100, {"family_curriculum": "full", **override})


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple Metal GPU is unavailable")),
])
def test_phase_two_checkpoint_handoffs_and_full_curriculum_validation(tmp_path, device):
    import pandas as pd

    layout = StateLayout(4, 2)
    specialist = tmp_path / "specialist.pt"
    save_checkpoint(specialist, NeuralCA3D(layout.channels, 4, 0, 1), step=90, config={"model_kind": "tree_specialist"})
    config = {
        "runs_root": str(tmp_path), "model_kind": "tree_family", "device": device,
        "world_size": 12, "batch_size": 8, "pool_size": 4, "materials": 4,
        "hidden_channels": 2, "model_width": 4, "fire_rate": 1,
        "iterations": 4, "rollout_steps": 1, "persistence_steps": 1,
        "validation_steps": 1, "validation_recovery_steps": 1,
        "validation_every": 99, "validation_fire_seeds": [71], "family_style_seeds": [0, 1],
        "validation_random_count": 0, "validation_interpolation_steps": 0,
        "validation_mutation_count": 0, "validation_boundary_genes": [],
    }
    source = specialist
    for mode in ("basics", "variation", "basics", "full"):
        run = train({
            **config, "run_name": f"from_{source.parent.name}_{mode}", "family_curriculum": mode,
            "initialize_from_checkpoint": str(source),
        }, dimensions=3, conditional=True)
        source = run / "checkpoints" / "latest.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        assert payload["step"] == 4  # Each handoff starts its own update budget.
        assert set(payload["pool"]["genomes"][:, :4].argmax(1).tolist()) == {0, 1, 2, 3}
        assert {int(value["step"]) for value in payload["optimizer"]["state"].values()} == {4}
        logs = pd.read_csv(run / "logs.csv")
        assert list(logs.curriculum_stage) == (["basics", "variation", "variation", "variation"] if mode == "full" else [mode] * 4)
        if mode in {"basics", "full"}:
            basic = torch.load(run / "checkpoints" / "basic_families.pt", map_location="cpu", weights_only=False)
            assert not basic["pool"]["genomes"][:, 4:13].any()
            torch.testing.assert_close(basic["pool"]["states"][::2], basic["pool"]["states"][1::2])
            torch.testing.assert_close(basic["pool"]["ages"][::2], basic["pool"]["ages"][1::2])
            panel = basic["validation"]["validation_panel"]
            assert {case["genome"]["family"] for case in panel} == set(TREE_FAMILIES)
            assert all(not any(case["genome"]["genes"].values()) for case in panel)
        best = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        assert best["validation"]["curriculum_stage"] == ("variation" if mode == "full" else mode)
    # A true resume continues the full curriculum's variation phase, even with a
    # new update budget. It must not restart basic training with a varied pool.
    resumed = train({
        **config, "run_name": "resumed_full", "family_curriculum": "full", "iterations": 1,
        "resume": str(source), "validation_steps": 0,
    }, dimensions=3, conditional=True)
    assert set(pd.read_csv(resumed / "logs.csv").curriculum_stage) == {"variation"}


def test_resuming_old_wind_targets_rebuilds_pool_without_resetting_optimizer(tmp_path):
    config = {
        "runs_root": str(tmp_path), "model_kind": "tree_family", "device": "cpu",
        "world_size": 12, "batch_size": 2, "pool_size": 2, "materials": 4,
        "hidden_channels": 1, "model_width": 4, "fire_rate": 1,
        "iterations": 1, "rollout_steps": 1, "validation_steps": 0,
    }
    original = train({**config, "run_name": "original"}, dimensions=3, conditional=True)
    source = original / "checkpoints" / "latest.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["metadata"]["target_generator_version"] = 3
    # Stale targets/ages must never enter the resumed training pool.
    payload["pool"]["target_occupancy"].fill_(-99)
    payload["pool"]["ages"].fill_(100_000)
    torch.save(payload, source)
    resumed = train({
        **config, "run_name": "resumed", "resume": str(source),
        "target_generator": {"name": "procedural_tree", "version": 3},
        "target_generator_version": 3, "reset_pool_on_resume": False,
    }, dimensions=3, conditional=True)
    result = torch.load(resumed / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert result["step"] == 2
    assert {int(value["step"]) for value in result["optimizer"]["state"].values()} == {2}
    assert result["pool"]["ages"].max() < 100_000
    assert result["pool"]["target_occupancy"].min() >= 0
    assert result["metadata"]["target_generator_version"] == TREE_TARGET_VERSION
    assert result["config"]["target_generator"]["version"] == TREE_TARGET_VERSION
