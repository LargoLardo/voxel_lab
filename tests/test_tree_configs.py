from pathlib import Path

from morphovoxel.config import load_config
from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec
from morphovoxel.genomes import TreeGenome


ROOT = Path(__file__).parents[1]


def test_tree_presets_form_a_parseable_checkpoint_pipeline():
    configs = {path.name: load_config(path) for path in (ROOT / "configs").glob("*.yaml")}
    assert all(isinstance(config, dict) for config in configs.values())

    assert configs["tree_family.yaml"]["initialize_from_specialist"] == "runs/tree_specialist/checkpoints/best.pt"
    assert configs["tree_regeneration.yaml"]["resume"] == "runs/tree_family/checkpoints/best.pt"
    assert configs["tree_environment.yaml"]["resume"] == "runs/tree_regeneration/checkpoints/best.pt"
    assert configs["tree_ecology.yaml"]["checkpoint"] == "runs/tree_environment/checkpoints/best.pt"

    shape_fields = ("materials", "hidden_channels", "model_width")
    model_presets = (
        "tree_specialist.yaml", "tree_family.yaml", "tree_regeneration.yaml",
        "tree_environment.yaml", "tree_ecology.yaml",
    )
    assert len({tuple(configs[name][field] for field in shape_fields) for name in model_presets}) == 1
    assert all(configs[name]["environment_conditioning"] for name in model_presets[1:-1])

    commands = configs["full_experiment.yaml"]["commands"]
    assert [Path(path).name for _, path in commands] == [
        "tree_specialist.yaml", "tree_family.yaml", "tree_regeneration.yaml",
        "tree_environment.yaml", "tree_ecology.yaml",
    ]
    assert all((ROOT / path).is_file() for _, path in commands)

    for name in ("tree_ecology.yaml", "smoke_tree_ecology.yaml"):
        config = configs[name]
        assert config["context_channels"] == len(ENVIRONMENT_CHANNELS)
        assert len(config["tree_genomes"]) == config["organisms"]
        assert all(TreeGenome.from_dict(value) for value in config["tree_genomes"])

    assert all(
        EnvironmentSpec.from_dict(value)
        for value in configs["tree_environment.yaml"]["validation_environment_specs"]
    )
    regeneration = configs["tree_regeneration.yaml"]
    assert regeneration["iterations"] == 8000
    assert regeneration["pool_size"] == 128
    assert regeneration["differentiable_step_limit"] == 96
    assert regeneration["persistence_steps"] == [32, 96]
    assert regeneration["persistence_weight"] == 2.0
    assert regeneration["validation_steps"] == 512
    assert regeneration["validation_recovery_steps"] == 128
    assert regeneration["loss_weights"]["occupancy_range"] == 2.0
    assert regeneration["loss_weights"]["magnitude"] == 0.01

    environment = configs["tree_environment.yaml"]
    assert environment["damage_probability"] == 0.35
    assert environment["damage_min_age"] == 64
    assert environment["differentiable_step_limit"] == 96
    assert environment["persistence_steps"] == [32, 96]
    assert environment["validation_steps"] == 512
    assert environment["validation_recovery_steps"] == 128
    assert "checkpoint" not in configs["smoke_tree_ecology.yaml"]
