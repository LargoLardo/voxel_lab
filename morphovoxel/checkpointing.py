"""Versioned, reproducible training checkpoints."""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .environment import ENVIRONMENT_SCHEMA_VERSION
from .genomes import TREE_GENOME_VERSION, TreeGenome
from .model_3d import NeuralCA3D
from .targets.targets_3d import TREE_TARGET_VERSION

CHECKPOINT_FORMAT = "morphovoxel"
CHECKPOINT_FORMAT_VERSION = 1
LEGACY_CHECKPOINT_FORMAT_VERSION = 0

_METADATA_FIELDS = (
    "checkpoint_format",
    "checkpoint_format_version",
    "model_kind",
    "genome_schema_version",
    "environment_schema_version",
    "context_channels",
    "target_generator",
    "target_generator_version",
    "training_genome_ranges",
    "validation_panel",
    "best_persistence_score",
)


class CheckpointCompatibilityError(ValueError):
    """A checkpoint is valid data but cannot be used by the requested model."""


def _model_kind(model: torch.nn.Module, config: Mapping[str, Any]) -> str:
    configured = config.get("model_kind")
    if configured:
        return str(configured)
    genome_size = int(getattr(model, "genome_size", 0))
    if genome_size == TreeGenome.model_size():
        return "tree_family"
    if genome_size:
        return "legacy_conditional"
    return "specialist"


def _validation_panel(config: Mapping[str, Any], validation: Mapping[str, Any]) -> Any:
    # A completed validation contains the exact expanded cases (including
    # genomes, environments, and fire seeds); prefer it to the generic recipe
    # stored in the training config. Fall back to the recipe before the first
    # validation has run.
    panel = validation.get("validation_panel", config.get("validation_panel"))
    if panel is not None:
        return copy.deepcopy(panel)
    per_genome = validation.get("per_genome")
    if isinstance(per_genome, Mapping):
        return {
            "genomes": list(per_genome),
            "steps": validation.get("validation_steps"),
            "seed": config.get("validation_seed"),
        }
    return None


def _checkpoint_metadata(
    model: torch.nn.Module,
    config: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    config, validation = config or {}, validation or {}
    kind = _model_kind(model, config)
    genome_size = int(getattr(model, "genome_size", 0))
    context_channels = int(getattr(model, "context_channels", 0))
    tree_model = kind in {"tree_specialist", "tree_family"}
    target_generator = config.get("target_generator")
    if isinstance(target_generator, Mapping):
        target_name = str(target_generator.get("name", "procedural_tree" if tree_model else "unspecified"))
        target_version = target_generator.get("version")
    else:
        target_name = str(target_generator or ("procedural_tree" if tree_model else "unspecified"))
        target_version = None
    metadata = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": kind,
        "genome_schema_version": config.get(
            "genome_schema_version",
            TREE_GENOME_VERSION if tree_model else (1 if genome_size else None),
        ),
        "environment_schema_version": config.get(
            "environment_schema_version", ENVIRONMENT_SCHEMA_VERSION if tree_model or context_channels else None,
        ),
        "context_channels": context_channels,
        "target_generator": target_name,
        "target_generator_version": config.get(
            "target_generator_version", target_version if target_version is not None else (TREE_TARGET_VERSION if tree_model else None),
        ),
        "training_genome_ranges": copy.deepcopy(config.get("training_genome_ranges")),
        "validation_panel": _validation_panel(config, validation),
        "best_persistence_score": validation.get(
            "best_persistence_score", validation.get("best_worst_genome_persistence_score"),
        ),
    }
    if overrides:
        unknown = set(overrides) - set(_METADATA_FIELDS)
        if unknown:
            raise ValueError(f"unknown checkpoint metadata fields: {', '.join(sorted(unknown))}")
        metadata.update(copy.deepcopy(dict(overrides)))
    _validate_metadata(metadata)
    return metadata


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    missing = [name for name in _METADATA_FIELDS if name not in metadata]
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint metadata is missing required fields: {', '.join(missing)}")
    if metadata["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format {metadata['checkpoint_format']!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    version = metadata["checkpoint_format_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise CheckpointCompatibilityError("checkpoint format version must be an integer")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format version {version}; this build supports version {CHECKPOINT_FORMAT_VERSION}"
        )
    if not isinstance(metadata["model_kind"], str) or not metadata["model_kind"]:
        raise CheckpointCompatibilityError("checkpoint model_kind must be a non-empty string")
    for field in ("genome_schema_version", "environment_schema_version", "target_generator_version"):
        value = metadata[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise CheckpointCompatibilityError(f"checkpoint {field} must be a non-negative integer or null")
    if not isinstance(metadata["target_generator"], str) or not metadata["target_generator"]:
        raise CheckpointCompatibilityError("checkpoint target_generator must be a non-empty string")
    context_channels = metadata["context_channels"]
    if isinstance(context_channels, bool) or not isinstance(context_channels, int) or context_channels < 0:
        raise CheckpointCompatibilityError("checkpoint context_channels must be a non-negative integer")


def _validate_metadata_for_model(metadata: Mapping[str, Any], model: torch.nn.Module) -> None:
    kind = metadata["model_kind"]
    genome_size = int(getattr(model, "genome_size", 0))
    context_channels = int(getattr(model, "context_channels", 0))
    genome_version = metadata["genome_schema_version"]
    environment_version = metadata["environment_schema_version"]
    if kind in {"tree_specialist", "tree_family"} and genome_version != TREE_GENOME_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported tree genome schema version {genome_version}; this build supports {TREE_GENOME_VERSION}"
        )
    if kind == "tree_family":
        if genome_size != TreeGenome.model_size():
            raise CheckpointCompatibilityError(
                f"tree-family checkpoint requires genome_size={TreeGenome.model_size()}, model has {genome_size}"
            )
    elif kind in {"specialist", "tree_specialist"} and genome_size:
        raise CheckpointCompatibilityError(
            f"{kind} checkpoint cannot load into a genome-conditioned model (genome_size={genome_size})"
        )
    elif kind == "legacy_conditional" and not genome_size:
        raise CheckpointCompatibilityError("legacy conditional checkpoint requires a genome-conditioned model")
    if environment_version is not None and environment_version != ENVIRONMENT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported environment schema version {environment_version}; this build supports {ENVIRONMENT_SCHEMA_VERSION}"
        )
    if int(metadata["context_channels"]) != context_channels:
        raise CheckpointCompatibilityError(
            f"checkpoint context_channels={metadata['context_channels']} but model has {context_channels}"
        )
    if metadata["target_generator"] == "procedural_tree" and metadata["target_generator_version"] != TREE_TARGET_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported procedural-tree target generator version {metadata['target_generator_version']}; "
            f"this build supports {TREE_TARGET_VERSION}"
        )


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    *,
    step: int = 0,
    scheduler=None,
    config=None,
    pool=None,
    genomes=None,
    validation=None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = _checkpoint_metadata(model, config, validation, metadata)
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": checkpoint_metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "config": config,
        "pool": pool.state_dict() if pool else None,
        "genomes": genomes,
        "validation": validation,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        },
    }
    torch.save(payload, path)


def _legacy_metadata(payload: Mapping[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    validation = payload.get("validation") if isinstance(payload.get("validation"), Mapping) else {}
    metadata = _checkpoint_metadata(model, config, validation, None)
    metadata["checkpoint_format_version"] = LEGACY_CHECKPOINT_FORMAT_VERSION
    metadata["checkpoint_format"] = f"{CHECKPOINT_FORMAT}-legacy"
    return metadata


def _payload_metadata(payload: dict[str, Any], model: torch.nn.Module) -> tuple[dict[str, Any], bool]:
    raw_metadata = payload.get("metadata")
    top_version = payload.get("checkpoint_format_version")
    if raw_metadata is None and top_version is None:
        return _legacy_metadata(payload, model), True
    if not isinstance(raw_metadata, Mapping):
        raise CheckpointCompatibilityError("versioned checkpoint metadata must be a mapping")
    metadata = dict(raw_metadata)
    _validate_metadata(metadata)
    if top_version != metadata["checkpoint_format_version"]:
        raise CheckpointCompatibilityError(
            "checkpoint format version disagrees between the payload and metadata"
        )
    top_format = payload.get("checkpoint_format")
    if top_format != metadata["checkpoint_format"]:
        raise CheckpointCompatibilityError("checkpoint format disagrees between the payload and metadata")
    return metadata, False


def _load_model_state(path: Path, model: torch.nn.Module, weights: Any) -> None:
    if not isinstance(weights, Mapping):
        raise CheckpointCompatibilityError(f"checkpoint {path} does not contain model weights")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    mismatched = []
    for name in expected.keys() & weights.keys():
        saved = weights[name]
        saved_shape = tuple(saved.shape) if isinstance(saved, torch.Tensor) else ("not a tensor",)
        if saved_shape != tuple(expected[name].shape):
            mismatched.append((name, saved_shape, tuple(expected[name].shape)))
    problems = []
    if missing:
        problems.append(f"missing keys: {', '.join(missing[:4])}")
    if unexpected:
        problems.append(f"unexpected keys: {', '.join(unexpected[:4])}")
    if mismatched:
        problems.append(
            "shape mismatches: "
            + "; ".join(f"{name} checkpoint={saved} model={wanted}" for name, saved, wanted in mismatched[:4])
        )
    if problems:
        hint = " Use convert_specialist_to_family() for a specialist-to-family NeuralCA3D migration." if any(
            name == "update.0.weight" for name, _, _ in mismatched
        ) else ""
        raise CheckpointCompatibilityError(
            f"checkpoint {path} is incompatible with {type(model).__name__}: {'; '.join(problems)}.{hint}"
        )
    try:
        model.load_state_dict(weights)
    except (RuntimeError, ValueError) as error:
        raise CheckpointCompatibilityError(
            f"checkpoint {path} model weights are incompatible with {type(model).__name__}: {error}"
        ) from error


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    *,
    expected_model_kind: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train the preset that creates it first, "
            "launch the Full experiment, or correct the checkpoint path."
        )
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError(f"checkpoint {path} payload must be a mapping")
    metadata, legacy = load_model_payload(
        payload, model, checkpoint_path=path, expected_model_kind=expected_model_kind,
    )
    if optimizer is not None and payload.get("optimizer"):
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except (KeyError, RuntimeError, ValueError) as error:
            raise CheckpointCompatibilityError(f"checkpoint {path} optimizer state is incompatible: {error}") from error
    if scheduler is not None and payload.get("scheduler"):
        try:
            scheduler.load_state_dict(payload["scheduler"])
        except (KeyError, RuntimeError, ValueError) as error:
            raise CheckpointCompatibilityError(f"checkpoint {path} scheduler state is incompatible: {error}") from error
    payload["metadata"] = metadata
    payload["legacy_checkpoint"] = legacy
    return payload


def load_model_payload(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    checkpoint_path: str | Path = "<memory>",
    expected_model_kind: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Validate and apply an already-deserialized checkpoint model payload.

    This lets trusted callers choose a safer deserialization policy while
    sharing the same format/schema/shape compatibility checks as training.
    """
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError(f"checkpoint {checkpoint_path} payload must be a mapping")
    metadata, legacy = _payload_metadata(payload, model)
    if expected_model_kind is not None and metadata["model_kind"] != expected_model_kind:
        raise CheckpointCompatibilityError(
            f"checkpoint {checkpoint_path} has model kind {metadata['model_kind']!r}, expected {expected_model_kind!r}"
        )
    _validate_metadata_for_model(metadata, model)
    _load_model_state(Path(checkpoint_path), model, payload.get("model"))
    return metadata, legacy


def _verify_specialist_conversion(source: NeuralCA3D, destination: NeuralCA3D) -> None:
    source_parameter = next(source.parameters())
    destination_parameter = next(destination.parameters())
    if source_parameter.device != destination_parameter.device or source_parameter.dtype != destination_parameter.dtype:
        raise CheckpointCompatibilityError("conversion verification requires both models on the same device and dtype")
    shape = (1, source.channels, 5, 5, 5)
    state = torch.zeros(shape, device=source_parameter.device, dtype=source_parameter.dtype)
    state[:, 0, 2, 2, 2] = 1
    if source.channels > 1:
        state[:, 1:, 2, 2, 2] = 0.25
    source_context = None
    if source.context_channels:
        source_context = torch.linspace(
            -0.5, 0.5, source.context_channels * 125,
            device=source_parameter.device, dtype=source_parameter.dtype,
        ).reshape(1, source.context_channels, 5, 5, 5)
    destination_context = None
    if destination.context_channels:
        destination_context = torch.zeros(
            1, destination.context_channels, 5, 5, 5,
            device=destination_parameter.device, dtype=destination_parameter.dtype,
        )
        if source_context is not None:
            destination_context[:, : source.context_channels] = source_context
    genome = torch.zeros(1, destination.genome_size, device=destination_parameter.device, dtype=destination_parameter.dtype)
    source_fire, destination_fire = source.fire_rate, destination.fire_rate
    try:
        source.fire_rate = destination.fire_rate = 1.0
        with torch.no_grad():
            expected = source(state, context=source_context)
            actual = destination(state, genome, destination_context)
    finally:
        source.fire_rate, destination.fire_rate = source_fire, destination_fire
    if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        difference = float((expected - actual).abs().max())
        raise CheckpointCompatibilityError(
            f"specialist-to-family conversion did not preserve zero-input behavior (maximum error {difference:.3g})"
        )


def convert_specialist_to_family(
    source: NeuralCA3D,
    destination: NeuralCA3D,
    *,
    verify: bool = True,
) -> NeuralCA3D:
    """Initialize a family NCA from a specialist without changing its zero-genome rule."""
    if not isinstance(source, NeuralCA3D) or not isinstance(destination, NeuralCA3D):
        raise CheckpointCompatibilityError("specialist-to-family conversion requires two NeuralCA3D models")
    if source.genome_size:
        raise CheckpointCompatibilityError("source model must be a specialist with genome_size=0")
    if destination.genome_size <= 0:
        raise CheckpointCompatibilityError("destination family model must have genome_size greater than zero")
    if source.channels != destination.channels:
        raise CheckpointCompatibilityError(
            f"specialist and family channel counts differ ({source.channels} != {destination.channels})"
        )
    if source.context_channels > destination.context_channels:
        raise CheckpointCompatibilityError("destination cannot remove specialist environment context channels")
    if source.fire_rate != destination.fire_rate:
        raise CheckpointCompatibilityError("specialist and family fire rates must match")

    source_state, destination_state = source.state_dict(), destination.state_dict()
    if set(source_state) != set(destination_state):
        raise CheckpointCompatibilityError("specialist and family update networks have different parameter keys")
    first_layer = "update.0.weight"
    incompatible = [
        name for name in source_state
        if name != first_layer and tuple(source_state[name].shape) != tuple(destination_state[name].shape)
    ]
    if incompatible:
        raise CheckpointCompatibilityError(
            f"specialist and family architectures differ at: {', '.join(incompatible)}"
        )
    source_first, destination_first = source_state[first_layer], destination_state[first_layer]
    if source_first.shape[0] != destination_first.shape[0] or source_first.shape[2:] != destination_first.shape[2:]:
        raise CheckpointCompatibilityError("specialist and family first update layers have incompatible shapes")

    converted = {name: tensor.detach().to(destination_state[name]).clone() for name, tensor in source_state.items()}
    expanded = torch.zeros_like(destination_first)
    perception_channels = source.channels * 5
    expanded[:, :perception_channels] = source_first[:, :perception_channels].to(expanded)
    if source.context_channels:
        source_start = perception_channels
        destination_start = perception_channels + destination.genome_size
        expanded[:, destination_start : destination_start + source.context_channels] = source_first[
            :, source_start : source_start + source.context_channels
        ].to(expanded)
    converted[first_layer] = expanded

    original = {name: tensor.detach().clone() for name, tensor in destination_state.items()}
    destination.load_state_dict(converted)
    if verify:
        try:
            _verify_specialist_conversion(source, destination)
        except Exception:
            destination.load_state_dict(original)
            raise
    return destination
