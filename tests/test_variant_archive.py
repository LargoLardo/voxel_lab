import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from morphovoxel.environment import EnvironmentSpec
from morphovoxel.genomes import TreeGenome
from morphovoxel.targets import make_tree_target
from morphovoxel.validation import ValidationCase, ValidationCriteria, ValidationReport, ValidationTrial
from morphovoxel.variant_archive import VariantArchive


def _report(genome, *, accepted=True, validated=True, height=8.0, canopy_spread=6.0, environment=None):
    case = ValidationCase("candidate-000-e00-f00", "candidate", genome, environment or EnvironmentSpec(), 0)
    reasons = () if accepted else ("late_drift_above_maximum",)
    trial = ValidationTrial(
        case,
        512,
        128,
        validated,
        accepted,
        0.9 if accepted else 0.1,
        reasons,
        {"target_iou": 0.9, "regeneration_score": 0.8},
        {"height": height, "canopy_spread": canopy_spread, "branch_material_fraction": 0.2},
    )
    return ValidationReport((trial,), ValidationCriteria(min_steps=512))


def _volumes(genome=None, environment=None):
    target, materials = make_tree_target(genome or TreeGenome(), 12, environment or EnvironmentSpec())
    final = np.zeros((1, 6, 12, 12, 12), dtype=np.float32)
    final[0, 0] = target
    return target, materials, final


def test_archive_atomically_saves_lists_filters_and_reloads_exact_variant(tmp_path):
    archive = VariantArchive(tmp_path / "variants")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"exact checkpoint bytes")
    genome = TreeGenome.random(7, family="conifer")
    environment = EnvironmentSpec(light_direction_x=1, seed=2)
    target, materials, final = _volumes(genome, environment)
    identity = {"model_kind": "tree_family", "class_name": "NeuralCA3D", "architecture": {"channels": 6}}

    record = archive.save(
        genome=genome,
        environment=environment,
        checkpoint=checkpoint,
        model_identity=identity,
        method="manual",
        validation=_report(genome, environment=environment),
        target_occupancy=target,
        target_materials=materials,
        final_state=final,
    )

    assert record.genome == genome and record.environment == environment
    assert record.checkpoint_identity["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record.validation["validated"] is True and record.validation["accepted"] is True
    assert json.loads((record.directory / "manifest.json").read_text())["variant_id"] == record.variant_id
    with np.load(record.directory / "voxels.npz", allow_pickle=False) as values:
        assert np.array_equal(values["target_occupancy"], target)
        assert np.array_equal(values["final_state"], final)
    assert Image.open(record.directory / "target.png").format == "PNG"
    assert Image.open(record.directory / "final.png").format == "PNG"
    assert archive.get(record.variant_id).genome == genome
    assert archive.list(family="conifer", method="manual", model_kind="tree_family", min_score=0.8) == [record]
    assert archive.list(descriptor_ranges={"height": (7, 9)}) == [record]
    assert archive.list(descriptor_ranges={"height": (9, None)}) == []
    assert not list((tmp_path / "variants").glob("variant-stage-*"))


def test_archive_rejects_failed_unvalidated_mismatched_or_parentless_candidates(tmp_path):
    archive = VariantArchive(tmp_path / "variants")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    genome = TreeGenome()
    target, materials, final = _volumes(genome)
    common = dict(
        genome=genome,
        environment=EnvironmentSpec(),
        checkpoint=checkpoint,
        model_identity={"model_kind": "tree_family", "class": "NeuralCA3D", "config": {}},
        method="manual",
        target_occupancy=target,
        target_materials=materials,
        final_state=final,
    )
    failed = _report(genome, accepted=False)
    with pytest.raises(ValueError, match="accepted validation"):
        archive.save(**common, validation=failed)
    unvalidated_trial = replace(failed.trials[0], validated=False)
    with pytest.raises(ValueError, match="accepted validation"):
        archive.save(**common, validation=ValidationReport((unvalidated_trial,), failed.criteria))
    with pytest.raises(ValueError, match="different genome"):
        archive.save(**common, validation=_report(TreeGenome(family="weeping")))
    with pytest.raises(ValueError, match="require a parent"):
        archive.save(**{**common, "method": "mutation"}, validation=_report(genome))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        archive.save(
            **{**common, "method": "mutation", "parents": ["var-missing-parent"]},
            validation=_report(genome),
        )
    short_trial = replace(_report(genome).trials[0], steps=128, recovery_steps=32)
    short_report = ValidationReport(
        (short_trial,), ValidationCriteria(min_steps=1, min_recovery_steps=1),
    )
    with pytest.raises(ValueError, match="at least 512 validation steps"):
        archive.save(**common, validation=short_report)
    wrong_target = target.copy()
    wrong_target[0, 0, 0] = 1 - wrong_target[0, 0, 0]
    with pytest.raises(ValueError, match="does not match the exact genome"):
        archive.save(**{**common, "target_occupancy": wrong_target}, validation=_report(genome))
    with pytest.raises(ValueError, match="checkpoint changed"):
        archive.save(
            **common,
            validation=_report(genome),
            expected_checkpoint_sha256="0" * 64,
        )


def test_archive_detects_artifact_corruption(tmp_path):
    archive = VariantArchive(tmp_path / "variants")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    genome = TreeGenome()
    target, materials, final = _volumes(genome)
    record = archive.save(
        genome=genome,
        environment=EnvironmentSpec(),
        checkpoint=checkpoint,
        model_identity={"model_kind": "tree_family", "class": "NeuralCA3D", "config": {}},
        method="random",
        validation=_report(genome),
        target_occupancy=target,
        target_materials=materials,
        final_state=final,
    )
    (record.directory / "final.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        archive.load(record.variant_id)


def test_archive_enforces_a_morphology_novelty_threshold(tmp_path):
    archive = VariantArchive(tmp_path / "variants")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    first = TreeGenome.random(1, family="branching")
    second = TreeGenome.random(2, family="branching")
    common = {
        "environment": EnvironmentSpec(),
        "checkpoint": checkpoint,
        "model_identity": {"model_kind": "tree_family", "class": "NeuralCA3D", "config": {}},
        "method": "manual",
    }
    first_target, first_materials, first_final = _volumes(first)
    second_target, second_materials, second_final = _volumes(second)
    archive.save(
        genome=first, validation=_report(first), target_occupancy=first_target,
        target_materials=first_materials, final_state=first_final, **common,
    )

    with pytest.raises(ValueError, match="novelty .* below"):
        archive.save(
            genome=second, validation=_report(second), target_occupancy=second_target,
            target_materials=second_materials, final_state=second_final, **common,
        )

    admitted = archive.save(
        genome=second,
        validation=_report(second, height=12.0, canopy_spread=3.0),
        target_occupancy=second_target, target_materials=second_materials,
        final_state=second_final, **common,
    )
    assert admitted.metrics["novelty_distance"] >= admitted.metrics["novelty_threshold"]
