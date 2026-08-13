from collections import Counter

import pytest
import torch

from morphovoxel.checkpointing import CheckpointCompatibilityError, save_checkpoint
from morphovoxel.genomes import TREE_FAMILIES, TreeGenome
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.state import StateLayout
from morphovoxel.training.family import curriculum_values, sample_family_data
from morphovoxel.training.trainer import train


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


def test_family_replacements_can_preserve_pool_family_balance():
    requested = list(TREE_FAMILIES)
    data = sample_family_data(
        len(requested), 16, 35, genome_span=1, interpolation_fraction=0.25,
        mutation_fraction=0.25, families=requested,
    )
    assert [genome.family for genome in data.genomes] == requested


def test_parent_mutations_reflect_instead_of_clipping_to_gene_limits():
    parent = TreeGenome(family="branching", genes=(1.0,) * 10)
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
        "world_size": 16, "batch_size": 1, "pool_size": 2,
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
