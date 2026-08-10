import json
import threading
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest

from morphovoxel.ui import CONFIGS, _inside, _launch, build_state, create_server
from morphovoxel.utils import steps_per_second, write_live_preview


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
        assert "Quick settings" in root
        assert "View Checkpoints" in root
        assert 'id="labDisplay"' in root
        assert "Training target" in root
        assert "source=${encodeURIComponent(source)}" in root
        assert "Design lab" not in root
        assert root.index('<option value="voxels">3D voxels</option>') < root.index('<option value="slice">Z slice</option>')
        assert "meta.dimensions===3?'voxels':'slice'" in root
        assert "Double-click to place a seed" in root
        assert "3D voxels" in root
        assert "Drag to rotate" in root
        assert "const center=[x-(width-1)/2,(depth-1)/2-z,y-(height-1)/2]" in root
        assert "r.status===403&&data.token&&options.body" in root
        assert "End state every 10 iterations" in root
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
        assert payload["hardware"]["auto_device"] in {"cpu", "cuda"}
    finally:
        server.shutdown()
        server.server_close()
