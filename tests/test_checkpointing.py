import pytest
import torch

from morphovoxel.checkpointing import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCompatibilityError,
    convert_specialist_to_family,
    load_checkpoint,
    save_checkpoint,
)
from morphovoxel.environment import ENVIRONMENT_CHANNELS, ENVIRONMENT_SCHEMA_VERSION
from morphovoxel.genomes import TREE_GENOME_VERSION, TreeGenome
from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.targets.targets_3d import TREE_TARGET_VERSION


def test_checkpoint_roundtrip(tmp_path):
    model = NeuralCA2D(3, 4)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    save_checkpoint(tmp_path / "model.pt", model, step=9)
    for parameter in model.parameters():
        parameter.data.zero_()
    payload = load_checkpoint(tmp_path / "model.pt", model)
    assert payload["step"] == 9
    assert payload["metadata"]["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["legacy_checkpoint"] is False
    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())


def test_missing_checkpoint_explains_prerequisite(tmp_path):
    with pytest.raises(FileNotFoundError, match="launch the Full experiment"):
        load_checkpoint(tmp_path / "missing.pt", torch.nn.Linear(2, 2))


def test_family_checkpoint_records_explicit_training_and_validation_metadata(tmp_path):
    model = NeuralCA3D(
        6,
        hidden=8,
        genome_size=TreeGenome.model_size(),
        context_channels=len(ENVIRONMENT_CHANNELS),
    )
    ranges = {"height": [-0.25, 0.5], "canopy_spread": [-0.5, 0.75]}
    panel = {"random_genomes": 8, "steps": 512, "fire_mask_seeds": [10, 11]}
    validation = {"best_worst_genome_persistence_score": 0.82}
    path = tmp_path / "family.pt"
    save_checkpoint(
        path,
        model,
        config={
            "model_kind": "tree_family",
            "target_generator": "procedural_tree",
            "target_generator_version": TREE_TARGET_VERSION,
            "training_genome_ranges": ranges,
            "validation_panel": panel,
        },
        validation=validation,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    assert payload["checkpoint_format"] == CHECKPOINT_FORMAT
    assert payload["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert metadata == {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": "tree_family",
        "genome_schema_version": TREE_GENOME_VERSION,
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "context_channels": len(ENVIRONMENT_CHANNELS),
        "target_generator": "procedural_tree",
        "target_generator_version": TREE_TARGET_VERSION,
        "training_genome_ranges": ranges,
        "validation_panel": panel,
        "best_persistence_score": 0.82,
    }


def test_checkpoint_metadata_prefers_the_exact_expanded_validation_panel(tmp_path):
    model = NeuralCA3D(
        6,
        hidden=8,
        genome_size=TreeGenome.model_size(),
        context_channels=len(ENVIRONMENT_CHANNELS),
    )
    recipe = {"random_count": 2, "fire_seeds": [1, 2]}
    exact_panel = [{"case_id": "candidate-000-e00-f00", "fire_seed": 1}]
    path = tmp_path / "expanded-panel.pt"

    save_checkpoint(
        path,
        model,
        config={"model_kind": "tree_family", "validation_panel": recipe},
        validation={"validation_panel": exact_panel},
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["validation_panel"] == exact_panel


def test_tree_specialist_metadata_keeps_target_schemas_separate_from_context_width(tmp_path):
    model = NeuralCA3D(6, hidden=8, genome_size=0, context_channels=0)
    path = tmp_path / "tree-specialist.pt"
    save_checkpoint(path, model, config={"model_kind": "tree_specialist"})

    payload = load_checkpoint(path, NeuralCA3D(6, hidden=8), expected_model_kind="tree_specialist")
    metadata = payload["metadata"]
    assert metadata["genome_schema_version"] == TREE_GENOME_VERSION
    assert metadata["environment_schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    assert metadata["context_channels"] == 0


def test_load_checkpoint_preserves_unversioned_app_checkpoints(tmp_path):
    source = NeuralCA2D(3, 4)
    path = tmp_path / "legacy.pt"
    torch.save({"model": source.state_dict(), "step": 3, "config": {"dimensions": 2}}, path)
    destination = NeuralCA2D(3, 4)

    payload = load_checkpoint(path, destination)

    assert payload["legacy_checkpoint"] is True
    assert payload["metadata"]["checkpoint_format_version"] == 0
    assert all(torch.equal(source.state_dict()[key], destination.state_dict()[key]) for key in source.state_dict())


def test_load_checkpoint_rejects_newer_formats_and_shape_mismatches_clearly(tmp_path):
    model = NeuralCA3D(4, hidden=6)
    path = tmp_path / "newer.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["checkpoint_format_version"] = 999
    payload["metadata"]["checkpoint_format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(CheckpointCompatibilityError, match="unsupported checkpoint format version 999"):
        load_checkpoint(path, NeuralCA3D(4, hidden=6))

    legacy_path = tmp_path / "specialist.pt"
    torch.save({"model": model.state_dict()}, legacy_path)
    family = NeuralCA3D(4, hidden=6, genome_size=TreeGenome.model_size())
    with pytest.raises(CheckpointCompatibilityError, match=r"update\.0\.weight.*convert_specialist_to_family"):
        load_checkpoint(legacy_path, family)


def test_load_checkpoint_rejects_incompatible_declared_schemas(tmp_path):
    model = NeuralCA3D(4, hidden=6, genome_size=TreeGenome.model_size())
    path = tmp_path / "family.pt"
    save_checkpoint(path, model, config={"model_kind": "tree_family"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metadata"]["genome_schema_version"] = TREE_GENOME_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(CheckpointCompatibilityError, match="unsupported tree genome schema version"):
        load_checkpoint(path, NeuralCA3D(4, hidden=6, genome_size=TreeGenome.model_size()))


def test_specialist_to_family_conversion_zeroes_added_inputs_and_preserves_behavior():
    torch.manual_seed(12)
    source = NeuralCA3D(5, hidden=7, fire_rate=1.0)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.2, 0.2)
    family = NeuralCA3D(
        5,
        hidden=7,
        genome_size=TreeGenome.model_size(),
        context_channels=len(ENVIRONMENT_CHANNELS),
        fire_rate=1.0,
    )

    returned = convert_specialist_to_family(source, family)

    assert returned is family
    source_first = source.update[0].weight
    family_first = family.update[0].weight
    perception_channels = source.channels * 5
    assert torch.equal(family_first[:, :perception_channels], source_first)
    assert torch.count_nonzero(family_first[:, perception_channels:]) == 0
    assert torch.equal(family.update[0].bias, source.update[0].bias)
    assert torch.equal(family.update[2].weight, source.update[2].weight)
    assert torch.equal(family.update[2].bias, source.update[2].bias)

    state = torch.zeros(2, 5, 7, 7, 7)
    state[:, 0, 3, 3, 3] = 1
    genome = torch.zeros(2, TreeGenome.model_size())
    context = torch.zeros(2, len(ENVIRONMENT_CHANNELS), 7, 7, 7)
    assert torch.allclose(source(state), family(state, genome, context))


def test_specialist_conversion_rejects_architecture_changes_other_than_inputs():
    source = NeuralCA3D(4, hidden=6)
    wrong_width = NeuralCA3D(4, hidden=7, genome_size=TreeGenome.model_size())
    with pytest.raises(CheckpointCompatibilityError, match="architectures differ"):
        convert_specialist_to_family(source, wrong_width)
