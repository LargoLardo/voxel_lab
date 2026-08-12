import json
import hashlib
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec
from morphovoxel.genomes import TreeGenome
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.ui import CONFIGS, _inside, _launch, build_state, create_server
from morphovoxel.utils import steps_per_second, write_live_preview
from morphovoxel.validation import ValidationCase, ValidationCriteria, ValidationReport, ValidationTrial


def test_step_rate_uses_completed_updates(monkeypatch):
    monkeypatch.setattr("morphovoxel.utils.time.perf_counter", lambda: 12.0)
    assert steps_per_second(24, 10.0) == 12.0
    assert steps_per_second(0, 10.0) == 0.0


def test_live_preview_drops_locked_metadata_instead_of_crashing(tmp_path, monkeypatch):
    original_replace = Path.replace
    locked = tmp_path / "live.json"

    def replace(path, target):
        if Path(target) == locked:
            raise PermissionError("simulated Windows reader lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    write_live_preview(tmp_path / "live.png", np.zeros((2, 2), dtype=np.uint8), step=1)
    assert (tmp_path / "live.png").exists()
    assert not (tmp_path / ".live.tmp.json").exists()


def test_full_presets_precede_smoke_presets_and_missing_dependencies_are_blocked(tmp_path, monkeypatch):
    names = list(CONFIGS)
    assert names[0] == "full_experiment.yaml"
    assert names[1:6] == [
        "tree_specialist.yaml", "tree_family.yaml", "tree_regeneration.yaml",
        "tree_environment.yaml", "tree_ecology.yaml",
    ]
    assert all(name.startswith("smoke_") for name in names[-10:])
    assert names[-5:] == [
        "smoke_2d.yaml", "smoke_3d.yaml", "smoke_conditional.yaml",
        "smoke_regeneration.yaml", "smoke_ecology.yaml",
    ]

    server = create_server(tmp_path, port=0)
    try:
        with pytest.raises(ValueError, match="launch Full experiment"):
            _launch(server, {
                "config": "phase4_regeneration.yaml",
                "content": "checkpoints:\n  regeneration: runs/missing/checkpoints/latest.pt\n",
                "device": "cpu",
                "live_preview": True,
            })
        with pytest.raises(ValueError, match=r"tree_specialist.*best\.pt"):
            _launch(server, {
                "config": "tree_family.yaml",
                "content": (
                    "run_name: tree_family\nmodel_kind: tree_family\n"
                    "initialize_from_specialist: runs/tree_specialist/checkpoints/best.pt\n"
                ),
                "device": "cpu",
                "live_preview": True,
            })

        class Process:
            pid = 1

            @staticmethod
            def poll():
                return None

        monkeypatch.setattr("morphovoxel.ui.subprocess.Popen", lambda *args, **kwargs: Process())
        job = _launch(server, {
            "config": "phase1_2d.yaml",
            "content": "run_name: phase1_2d\ndimensions: 2\n",
            "device": "cpu",
            "live_preview": True,
        })
        assert job["run_name"] == "phase1_2d"
    finally:
        server.server_close()


def test_dashboard_serves_configs_runs_and_blocks_traversal(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "smoke_2d.yaml").write_text("run_name: demo\ndimensions: 2\n", encoding="utf-8")
    visual = tmp_path / "runs" / "demo" / "visualizations"
    visual.mkdir(parents=True)
    (visual / "growth.gif").write_bytes(b"GIF89a")
    (visual / ".live.tmp.png").write_bytes(b"temporary")

    state = build_state(tmp_path)
    assert state["configs"][0]["name"] == "smoke_2d.yaml"
    assert [item["name"] for item in state["runs"][0]["media"]] == ["growth.gif"]
    with pytest.raises(ValueError):
        _inside(tmp_path / "runs", "../pyproject.toml")

    server = create_server(tmp_path, port=0)
    payload = {"config": "smoke_2d.yaml", "content": "dimensions: 2\n", "device": "cpu", "live_preview": True}
    with pytest.raises(ValueError, match="already exists"):
        _launch(server, {**payload, "run_name": "demo"})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5).read().decode()
        payload = json.load(urlopen(f"http://127.0.0.1:{server.server_port}/api/state", timeout=5))
        assert "Experiment control room" in root
        assert "Overview is an information page" in root
        assert "Quick settings" in root
        assert "View Checkpoints" in root
        assert "Specialist" in root and "Tree Genome Lab" in root and "Environment Lab" in root
        assert "Variant Archive" in root and "Legacy" in root
        assert "Stability Evaluation" in root
        assert root.index('data-page="specialist"') < root.index('data-page="family"')
        assert root.index('data-page="family"') < root.index('data-page="regeneration"')
        assert root.index('data-page="regeneration"') < root.index('data-page="environment"')
        assert root.index('data-page="environment"') < root.index('data-page="ecology"')
        assert 'id="overviewPage"' in root and 'id="trainingPage"' in root
        assert 'id="evaluationPage"' in root and 'id="evaluationRunButton"' in root
        assert "/api/evaluate" in root and "1,024 and 2,048 steps" in root
        assert "Max channel magnitude" in root and "Regeneration" in root
        assert 'id="dependencyCheckpoint"' in root
        assert "'tree_family.yaml':{key:'initialize_from_specialist'" in root
        assert "'tree_regeneration.yaml':{key:'resume'" in root
        assert "'tree_environment.yaml':{key:'resume'" in root
        assert "'tree_ecology.yaml':{key:'checkpoint'" in root
        assert "No compatible checkpoints found" in root
        assert 'id="labCheckpoint"' in root
        assert 'id="labDeleteCheckpoint"' in root
        assert "/api/checkpoint/delete" in root
        assert "Permanently delete" in root
        assert "kind==='specialist'?null" in root
        assert ".field[hidden]{display:none}" in root
        assert 'id="openTreeGenomeWindow"' in root and 'id="openEnvironmentWindow"' in root
        assert "openUtilityWindow('genome')" in root and "openUtilityWindow('environment-lab')" in root
        assert "page==='environment-lab'" in root
        assert "environment:{kind:'environment',number:'04'" in root
        assert 'id="treeGeneControls"' in root and "data-tree-range" in root
        assert 'id="treeLiveRemodel"' in root and "Genome staged" in root
        assert 'id="treeStoreA"' in root and 'id="treeStoreB"' in root
        assert 'id="treeJsonFile"' in root and "Download JSON" in root
        assert 'id="environmentControls"' in root and "environmentDraft" in root
        assert "/api/lab/validate" in root and "/api/archive/save" in root
        assert 'id="labDisplay"' not in root
        assert 'id="labTargetCanvas"' in root and "Always shown for comparison" in root
        assert "/api/lab/voxels?source=target" in root
        assert "source=${encodeURIComponent(source)}" in root
        assert "Design lab" not in root
        assert root.index('<option value="voxels">3D voxels</option>') < root.index('<option value="slice">Z slice</option>')
        assert "meta.dimensions===3?'voxels':'slice'" in root
        assert "Double-click to place a seed" in root
        assert "3D voxels" in root
        assert "Drag to rotate" in root
        assert "const center=[x-(width-1)/2,(depth-1)/2-z,y-(height-1)/2]" in root
        assert "r.status===403&&data.token&&options.body" in root
        assert "End state every 10 iterations" not in root
        assert "Training previews show the completed rollout every 10 iterations" not in root
        assert "Tiny smoke runs can still be faster on CPU" not in root
        assert 'id="frameEvery"' not in root
        assert "steps/s" in root
        assert "Playback speed" in root
        assert "Steps per frame" not in root
        assert 'id="labSpeed" type="range" min="0" max="10" step="1" value="6"' in root
        assert "[[1/60,'1/60×']" in root
        assert "[5,'5×']" in root
        assert "labDeviceRate*speed/5" in root
        assert "1× uses one-fifth of measured device throughput" in root
        assert payload["runs"][0]["name"] == "demo"
        assert payload["runs"][0]["model_kind"] == ""
        assert payload["runs"][0]["context_channels"] == 0
        assert payload["hardware"]["auto_device"] in {"cpu", "cuda"}
    finally:
        server.shutdown()
        server.server_close()


def test_checkpoint_delete_route_is_scoped_and_closes_loaded_lab(tmp_path):
    run = tmp_path / "runs" / "tree"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "best.pt").write_bytes(b"best")
    (checkpoints / "latest.pt").write_bytes(b"latest")

    server = create_server(tmp_path, port=0)

    class Lab:
        run_name = "tree"
        checkpoint_name = "best.pt"

    server.lab = Lab()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def delete(checkpoint):
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/checkpoint/delete",
            data=json.dumps({"token": server.token, "run": "tree", "checkpoint": checkpoint}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.load(urlopen(request, timeout=5))

    try:
        result = delete("best.pt")
        assert result == {"deleted": "best.pt", "run": "tree", "closed_lab": True}
        assert not (checkpoints / "best.pt").exists()
        assert (checkpoints / "latest.pt").exists()
        assert server.lab is None

        with pytest.raises(HTTPError) as error:
            delete("../latest.pt")
        assert error.value.code == 400
        assert (checkpoints / "latest.pt").exists()

        class Process:
            @staticmethod
            def poll():
                return None

        server.jobs["active"] = {"run_name": "tree", "process": Process()}
        with pytest.raises(HTTPError) as error:
            delete("latest.pt")
        assert error.value.code == 400
        assert (checkpoints / "latest.pt").exists()
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_exposes_checkpoint_compatibility_metadata(tmp_path):
    run = tmp_path / "runs" / "specialist_a"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    (run / "config.yaml").write_text(
        "model_kind: tree_specialist\nenvironment_conditioning: false\n",
        encoding="utf-8",
    )
    (run / "metadata.json").write_text(
        json.dumps({"model_kind": "tree_specialist", "context_channels": 0}),
        encoding="utf-8",
    )

    item = build_state(tmp_path)["runs"][0]

    assert item["kind"] == "specialist"
    assert item["model_kind"] == "tree_specialist"
    assert item["context_channels"] == 0
    assert item["checkpoints"] == ["best.pt"]


def test_validation_route_bounds_inputs_and_does_not_hold_lab_lock(tmp_path):
    server = create_server(tmp_path, port=0)

    class Lab:
        def validate_tree_candidate(self, **value):
            assert server.lab_lock.acquire(blocking=False)
            server.lab_lock.release()
            return {"report": {"validated": True}, "arguments": value}

    server.lab = Lab()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(**value):
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/lab/validate",
            data=json.dumps({"token": server.token, **value}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.load(urlopen(request, timeout=10))

    try:
        result = post(steps=512, recovery_steps=128, fire_seeds=[3, 4])
        assert result["arguments"] == {"steps": 512, "recovery_steps": 128, "fire_seeds": [3, 4]}
        with pytest.raises(HTTPError) as error:
            post(steps=2049, recovery_steps=128, fire_seeds=[3])
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post(steps=512, recovery_steps=128, fire_seeds=list(range(9)))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_stability_evaluation_route_loads_checkpoint_without_replacing_lab(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "tree"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    calls = []

    class Session:
        model_kind = "tree_family"

        def validate_tree_candidate(self, **values):
            calls.append(values)
            return {
                "checkpoint": "best.pt",
                "scope": values["scope"],
                "report": {"accepted": True, "worst_score": 0.8, "trials": []},
            }

    monkeypatch.setattr("morphovoxel.ui.LabSession.from_run", lambda *args: Session())
    server = create_server(tmp_path, port=0)
    original_lab = object()
    server.lab = original_lab
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/evaluate",
            data=json.dumps({
                "token": server.token,
                "run": "tree",
                "checkpoint": "best.pt",
                "device": "cpu",
                "scope": "family",
                "horizons": [512, 1024],
                "recovery_steps": 256,
                "fire_seeds": [1, 2],
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        result = json.load(urlopen(request, timeout=10))
        assert result["accepted"] and result["worst_score"] == 0.8
        assert result["horizons"] == [512, 1024]
        assert [call["steps"] for call in calls] == [512, 1024]
        assert all(call["scope"] == "family" for call in calls)
        assert server.lab is original_lab
    finally:
        server.shutdown()
        server.server_close()


def test_variant_archive_http_save_filter_preview_and_reload(tmp_path):
    genome = TreeGenome(family="conifer", style_seed=7)
    environment = EnvironmentSpec(light_direction_x=0.5, seed=8)
    case = ValidationCase("candidate", "candidate", genome, environment, 3)
    trial = ValidationTrial(
        case, 512, 128, True, True, 0.9, (),
        {"target_iou": 0.9, "regeneration_score": 0.8},
        {"height": 8.0, "canopy_spread": 5.0},
    )
    report = ValidationReport((trial,), ValidationCriteria(min_steps=512, min_recovery_steps=128))
    run = tmp_path / "runs" / "tree_family"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"checkpoint")
    layout = StateLayout(4, 2)
    target_values, material_values = make_tree_target(genome, 12, environment)
    target = torch.from_numpy(target_values)
    materials = torch.from_numpy(material_values)

    class Model:
        context_channels = len(ENVIRONMENT_CHANNELS)
        genome_size = TreeGenome.model_size()

    class Lab:
        run_name = "tree_family"
        checkpoint_name = "best.pt"
        checkpoint_sha256 = hashlib.sha256(b"checkpoint").hexdigest()
        model_kind = "tree_family"
        dimensions = 3
        config = {"model_width": 4}
        model = Model()
        state = torch.zeros((1, layout.channels, 12, 12, 12))
        active_tree_genome = genome
        pending_tree_genome = genome
        environment_spec = environment
        last_validation_report = report
        steps = 5
        last_validation = {
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha256,
            "genome": genome.to_dict(),
            "environment": environment.to_dict(),
            "report": report.to_dict(),
        }

        def __init__(self):
            self.layout = layout

        @staticmethod
        def _target():
            return "tree", target, materials

        def set_tree_genome(self, value):
            self.pending_tree_genome = TreeGenome.from_dict(value)

        def set_environment(self, value):
            self.environment_spec = EnvironmentSpec.from_dict(value)

        def summary(self):
            return {
                "run": self.run_name, "checkpoint": self.checkpoint_name,
                "pending_tree_genome": self.pending_tree_genome.to_dict(),
                "environment": self.environment_spec.to_dict(),
            }

    lab = Lab()
    server = create_server(tmp_path, port=0)
    server.lab = lab
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, **value):
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps({"token": server.token, **value}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.load(urlopen(request, timeout=10))

    try:
        saved = post("/api/archive/save", method="manual", parents=[])
        variant_id = saved["variant_id"]
        listing = json.load(urlopen(
            f"http://127.0.0.1:{server.server_port}/api/archive?family=conifer&min_score=0.8", timeout=10,
        ))
        assert [item["variant_id"] for item in listing["variants"]] == [variant_id]
        preview = urlopen(
            f"http://127.0.0.1:{server.server_port}/api/archive/{variant_id}/preview/target", timeout=10,
        ).read()
        assert preview.startswith(b"\x89PNG")

        lab.pending_tree_genome = TreeGenome(family="weeping")
        with pytest.raises(HTTPError) as error:
            post("/api/archive/save", method="manual", parents=[])
        assert error.value.code == 400
        assert "apply the staged genome" in json.load(error.value)["error"]
        lab.pending_tree_genome = genome
        accepted_binding = lab.last_validation
        lab.last_validation = {**accepted_binding, "checkpoint": "latest.pt"}
        with pytest.raises(HTTPError) as error:
            post("/api/archive/save", method="manual", parents=[])
        assert error.value.code == 400
        lab.last_validation = accepted_binding
        lab.pending_tree_genome = TreeGenome(family="weeping")
        loaded = post("/api/archive/load", variant_id=variant_id)
        assert loaded["lab"]["pending_tree_genome"] == genome.to_dict()
        assert loaded["lab"]["environment"] == environment.to_dict()
    finally:
        server.shutdown()
        server.server_close()
