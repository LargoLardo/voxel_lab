"""Small, inspectable filesystem archive for validated tree variants."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import torch
from PIL import Image

from .environment import EnvironmentSpec
from .genomes import TreeGenome
from .rendering_3d import projection
from .targets import make_tree_target
from .validation import ValidationReport


ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_MIN_VALIDATION_STEPS = 512
ARCHIVE_MIN_RECOVERY_STEPS = 128
CreationMethod = Literal["random", "mutation", "interpolation", "manual"]
CREATION_METHODS = ("random", "mutation", "interpolation", "manual")
NOVELTY_DESCRIPTORS = (
    "height",
    "canopy_spread",
    "branch_material_fraction",
    "canopy_material_fraction",
    "asymmetry",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("archive metadata must contain finite JSON values") from error


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}-", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError("archived voxel arrays must be finite numeric values")
    return result


def _target_volume(value, name: str) -> np.ndarray:
    result = _array(value)
    while result.ndim > 3 and result.shape[0] == 1:
        result = result[0]
    if result.ndim != 3:
        raise ValueError(f"{name} must describe one [D,H,W] target")
    return result


def _final_occupancy(state: np.ndarray) -> np.ndarray:
    if state.ndim == 5 and state.shape[0] == 1:
        return state[0, 0]
    if state.ndim == 4:
        return state[0]
    if state.ndim == 3:
        return state
    raise ValueError("final state must be [1,C,D,H,W], [C,D,H,W], or [D,H,W]")


def _model_identity(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("model_identity must be a mapping")
    result = _json_value(dict(value))
    class_name = result.get("class_name", result.get("class"))
    architecture = result.get("architecture", result.get("config"))
    if not result.get("model_kind") or not class_name or not isinstance(architecture, dict):
        raise ValueError("model_identity requires model_kind, class/class_name, and config/architecture")
    return result


def _finite_mapping(value: Mapping[str, float] | None, fallback: Mapping[str, float]) -> dict[str, float]:
    source = fallback if value is None else value
    result = {str(name): float(number) for name, number in source.items()}
    if any(not math.isfinite(number) for number in result.values()):
        raise ValueError("archive metrics and descriptors must be finite")
    return result


def _descriptor_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    """Return a scale-aware morphology distance for quality-diversity admission."""
    differences = []
    for name in NOVELTY_DESCRIPTORS:
        if name not in left or name not in right:
            continue
        left_value, right_value = float(left[name]), float(right[name])
        scale = max(abs(left_value), abs(right_value), 1.0) if name in {"height", "canopy_spread"} else 1.0
        differences.append((left_value - right_value) / scale)
    return math.sqrt(sum(value * value for value in differences) / len(differences)) if differences else None


@dataclass(frozen=True)
class VariantRecord:
    variant_id: str
    directory: Path
    created_at: str
    genome: TreeGenome
    environment: EnvironmentSpec
    checkpoint_identity: Mapping[str, object]
    model_identity: Mapping[str, object]
    method: CreationMethod
    parents: tuple[str, ...]
    metrics: Mapping[str, float]
    descriptors: Mapping[str, float]
    validation: Mapping[str, object]
    artifacts: Mapping[str, Mapping[str, str]]

    @property
    def score(self) -> float:
        return float(self.validation["score"])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "variant_id": self.variant_id,
            "created_at": self.created_at,
            "genome": self.genome.to_dict(),
            "environment": self.environment.to_dict(),
            "checkpoint_identity": dict(self.checkpoint_identity),
            "model_identity": dict(self.model_identity),
            "creation": {"method": self.method, "parents": list(self.parents)},
            "metrics": dict(self.metrics),
            "descriptors": dict(self.descriptors),
            "validation": dict(self.validation),
            "artifacts": {name: dict(value) for name, value in self.artifacts.items()},
        }


class VariantArchive:
    """Directory-per-variant archive with manifest-last atomic admission."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _new_id(self) -> str:
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            value = f"var-{stamp}-{uuid.uuid4().hex[:10]}"
            if not (self.root / value).exists():
                return value

    @staticmethod
    def _validate_creation(method: str, parents: Iterable[str]) -> tuple[CreationMethod, tuple[str, ...]]:
        if method not in CREATION_METHODS:
            raise ValueError(f"creation method must be one of {CREATION_METHODS}")
        parent_ids = tuple(parents)
        if any(not _SAFE_ID.fullmatch(value) for value in parent_ids):
            raise ValueError("parent variant ids are invalid")
        if method == "mutation" and not parent_ids:
            raise ValueError("mutation variants require a parent")
        if method == "interpolation" and len(parent_ids) < 2:
            raise ValueError("interpolation variants require two parents")
        return method, parent_ids  # type: ignore[return-value]

    def save(
        self,
        *,
        genome: TreeGenome,
        environment: EnvironmentSpec,
        checkpoint: str | Path,
        model_identity: Mapping[str, object],
        method: CreationMethod,
        validation: ValidationReport,
        target_occupancy,
        target_materials,
        final_state,
        parents: Iterable[str] = (),
        metrics: Mapping[str, float] | None = None,
        descriptors: Mapping[str, float] | None = None,
        novelty_threshold: float = 0.02,
        expected_checkpoint_sha256: str | None = None,
    ) -> VariantRecord:
        """Admit one fully validated candidate and persist exact inputs/results."""
        if not isinstance(genome, TreeGenome) or not isinstance(environment, EnvironmentSpec):
            raise ValueError("archive entries require TreeGenome and EnvironmentSpec values")
        if not isinstance(validation, ValidationReport) or not validation.validated or not validation.accepted:
            raise ValueError("only candidates with a complete, accepted validation report may be archived")
        if any(
            trial.steps < ARCHIVE_MIN_VALIDATION_STEPS or trial.recovery_steps < ARCHIVE_MIN_RECOVERY_STEPS
            for trial in validation.trials
        ):
            raise ValueError(
                f"archive admission requires every trial to use at least {ARCHIVE_MIN_VALIDATION_STEPS} "
                f"validation steps and {ARCHIVE_MIN_RECOVERY_STEPS} recovery steps"
            )
        if any(trial.case.genome != genome for trial in validation.trials):
            raise ValueError("validation report contains a different genome than the archived candidate")
        if not any(trial.case.environment == environment for trial in validation.trials):
            raise ValueError("validation report does not include the archived environment")
        chosen_method, parent_ids = self._validate_creation(method, parents)
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("archive parent references must be distinct")
        parent_records = [self.load(parent_id) for parent_id in parent_ids]
        if any(record.genome.family != genome.family for record in parent_records):
            raise ValueError("archive parents must belong to the candidate's discrete tree family")
        identity = _model_identity(model_identity)
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        checkpoint_info = {
            "path": str(checkpoint_path.resolve()),
            "name": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "size": checkpoint_path.stat().st_size,
        }
        if expected_checkpoint_sha256 is not None and checkpoint_info["sha256"] != expected_checkpoint_sha256:
            raise ValueError("checkpoint changed after candidate validation; reload and revalidate before archiving")
        target = _target_volume(target_occupancy, "target occupancy").astype(np.float32, copy=False)
        materials = _target_volume(target_materials, "target materials").astype(np.int64, copy=False)
        state = _array(final_state).astype(np.float32, copy=False)
        final = _final_occupancy(state)
        if target.shape != materials.shape or target.shape != final.shape:
            raise ValueError("target occupancy, target materials, and final state must share a voxel shape")
        if target.min() < 0 or target.max() > 1:
            raise ValueError("target occupancy must be within [0, 1]")
        expected_target, expected_materials = make_tree_target(genome, target.shape[0], environment)
        if not np.array_equal(target, expected_target) or not np.array_equal(materials, expected_materials):
            raise ValueError("archived target does not match the exact genome and environment")
        saved_metrics = _finite_mapping(metrics, validation.mean_metrics("metrics"))
        saved_descriptors = _finite_mapping(descriptors, validation.mean_metrics("descriptors"))
        novelty_threshold = float(novelty_threshold)
        if not math.isfinite(novelty_threshold) or not 0 <= novelty_threshold <= 1:
            raise ValueError("archive novelty threshold must be finite and within [0, 1]")
        comparable = [
            record for record in self.list(family=genome.family, model_kind=str(identity["model_kind"]))
            if _descriptor_distance(saved_descriptors, record.descriptors) is not None
        ]
        novelty_distances = [
            _descriptor_distance(saved_descriptors, record.descriptors) for record in comparable
        ]
        novelty_distance = min((value for value in novelty_distances if value is not None), default=1.0)
        if novelty_distance < novelty_threshold:
            raise ValueError(
                f"candidate morphology novelty {novelty_distance:.4f} is below the archive threshold "
                f"{novelty_threshold:.4f}"
            )
        saved_metrics["novelty_distance"] = novelty_distance
        saved_metrics["novelty_threshold"] = novelty_threshold

        variant_id = self._new_id()
        staging = Path(tempfile.mkdtemp(prefix="variant-stage-", dir=self.root))
        final_directory = self.root / variant_id
        try:
            data_path = staging / "voxels.npz"
            np.savez_compressed(data_path, target_occupancy=target, target_materials=materials, final_state=state)
            target_preview = staging / "target.png"
            final_preview = staging / "final.png"
            Image.fromarray(projection(target)).save(target_preview)
            Image.fromarray(projection(final)).save(final_preview)
            artifacts = {
                "voxel_data": {"path": data_path.name, "sha256": _sha256(data_path)},
                "target_preview": {"path": target_preview.name, "sha256": _sha256(target_preview)},
                "final_preview": {"path": final_preview.name, "sha256": _sha256(final_preview)},
            }
            manifest = {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "variant_id": variant_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "genome": genome.to_dict(),
                "environment": environment.to_dict(),
                "checkpoint_identity": checkpoint_info,
                "model_identity": identity,
                "creation": {"method": chosen_method, "parents": list(parent_ids)},
                "metrics": saved_metrics,
                "descriptors": saved_descriptors,
                "validation": validation.to_dict(),
                "artifacts": artifacts,
            }
            _atomic_json(staging / "manifest.json", manifest)
            os.replace(staging, final_directory)
        except Exception:
            if staging.exists() and staging.parent.resolve() == self.root.resolve():
                shutil.rmtree(staging)
            raise
        return self.load(variant_id)

    def load(self, variant_id: str) -> VariantRecord:
        if not isinstance(variant_id, str) or not _SAFE_ID.fullmatch(variant_id):
            raise ValueError("invalid variant id")
        directory = self.root / variant_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"variant does not exist: {variant_id}")
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"variant manifest is unreadable: {variant_id}") from error
        if value.get("schema_version") != ARCHIVE_SCHEMA_VERSION or value.get("variant_id") != variant_id:
            raise ValueError(f"variant manifest identity/schema mismatch: {variant_id}")
        creation = value.get("creation", {})
        chosen_method, parent_ids = self._validate_creation(str(creation.get("method", "")), creation.get("parents", ()))
        genome = TreeGenome.from_dict(value.get("genome", {}))
        environment = EnvironmentSpec.from_dict(value.get("environment", {}))
        identity = _model_identity(value.get("model_identity", {}))
        validation = value.get("validation", {})
        if not isinstance(validation, dict) or not validation.get("validated") or not validation.get("accepted"):
            raise ValueError(f"variant manifest lacks accepted validation: {variant_id}")
        artifacts = value.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError(f"variant artifacts are invalid: {variant_id}")
        for artifact in artifacts.values():
            if not isinstance(artifact, dict) or not _SAFE_ID.fullmatch(Path(str(artifact.get("path", ""))).name.replace(".", "_")):
                raise ValueError(f"variant artifact path is invalid: {variant_id}")
            path = directory / str(artifact["path"])
            if path.parent != directory or not path.is_file() or _sha256(path) != artifact.get("sha256"):
                raise ValueError(f"variant artifact is missing or corrupt: {variant_id}")
        checkpoint_identity = value.get("checkpoint_identity", {})
        if not isinstance(checkpoint_identity, dict) or len(str(checkpoint_identity.get("sha256", ""))) != 64:
            raise ValueError(f"variant checkpoint identity is invalid: {variant_id}")
        return VariantRecord(
            variant_id,
            directory,
            str(value.get("created_at", "")),
            genome,
            environment,
            checkpoint_identity,
            identity,
            chosen_method,
            parent_ids,
            _finite_mapping(value.get("metrics", {}), {}),
            _finite_mapping(value.get("descriptors", {}), {}),
            validation,
            artifacts,
        )

    get = load

    def list(
        self,
        *,
        family: str | None = None,
        method: CreationMethod | None = None,
        model_kind: str | None = None,
        min_score: float | None = None,
        descriptor_ranges: Mapping[str, tuple[float | None, float | None]] | None = None,
    ) -> list[VariantRecord]:
        """List records newest-first, optionally filtering archive dimensions."""
        if method is not None and method not in CREATION_METHODS:
            raise ValueError(f"creation method must be one of {CREATION_METHODS}")
        records: list[VariantRecord] = []
        for directory in sorted(self.root.glob("var-*"), reverse=True):
            if not directory.is_dir():
                continue
            record = self.load(directory.name)
            if family is not None and record.genome.family != family:
                continue
            if method is not None and record.method != method:
                continue
            if model_kind is not None and record.model_identity.get("model_kind") != model_kind:
                continue
            if min_score is not None and record.score < min_score:
                continue
            if descriptor_ranges and any(
                name not in record.descriptors
                or (lower is not None and record.descriptors[name] < lower)
                or (upper is not None and record.descriptors[name] > upper)
                for name, (lower, upper) in descriptor_ranges.items()
            ):
                continue
            records.append(record)
        return records
