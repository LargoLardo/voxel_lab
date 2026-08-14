import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from morphovoxel.checkpointing import CheckpointCompatibilityError, save_checkpoint
from morphovoxel.config import save_config
from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec
from morphovoxel.genomes import MORPHOLOGIES, TREE_FAMILIES, TREE_GENOME_VERSION, TreeGenome
from morphovoxel.lab import LabSession, find_checkpoint
from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D, TreeFamilyNCA3D
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.ui import build_state, create_server


def _saved_run(root, dimensions: int, conditional: bool = False):
    run = root / "runs" / f"lab_{dimensions}d"
    (run / "checkpoints").mkdir(parents=True)
    materials = 3 if dimensions == 2 else 4
    config = {
        "run_name": run.name, "dimensions": dimensions, "conditional": conditional,
        "device": "cpu", "world_size": 12, "materials": materials,
        "hidden_channels": 2, "model_width": 4, "fire_rate": 1.0,
    }
    save_config(config, run / "config.yaml")
    layout = StateLayout(materials, 2)
    model_class = NeuralCA2D if dimensions == 2 else NeuralCA3D
    model = model_class(layout.channels, 4, len(MORPHOLOGIES) if conditional else 0, 1.0)
    save_checkpoint(run / "checkpoints" / "latest.pt", model, config=config)
    return run


def _saved_tree_family(root):
    run = root / "runs" / "tree_family"
    (run / "checkpoints").mkdir(parents=True)
    config = {
        "run_name": run.name, "dimensions": 3, "conditional": True,
        "model_kind": "tree_family", "environment_conditioning": True,
        "device": "cpu", "world_size": 16, "materials": 4,
        "hidden_channels": 2, "model_width": 4, "fire_rate": 1.0,
    }
    save_config(config, run / "config.yaml")
    layout = StateLayout(4, 2)
    model = TreeFamilyNCA3D(
        layout.channels, 4, TreeGenome.model_size(), 1.0, len(ENVIRONMENT_CHANNELS), len(TREE_FAMILIES),
    )
    save_checkpoint(run / "checkpoints" / "latest.pt", model, config=config)
    return run


def _saved_tree_specialist(root, *, checkpoint_kind="tree_specialist"):
    run = root / "runs" / f"tree_specialist_{checkpoint_kind}"
    (run / "checkpoints").mkdir(parents=True)
    genome = TreeGenome(family="weeping", style_seed=77)
    environment = EnvironmentSpec(water_direction_x=1, water_level=0.4, seed=9)
    config = {
        "run_name": run.name, "dimensions": 3, "conditional": False,
        "model_kind": "tree_specialist", "environment_conditioning": False,
        "world_size": 16, "materials": 4, "hidden_channels": 2,
        "model_width": 4, "fire_rate": 1.0,
        "tree_genome": genome.to_dict(), "environment": environment.to_dict(),
    }
    save_config(config, run / "config.yaml")
    model = NeuralCA3D(StateLayout(4, 2).channels, 4, 0, 1.0)
    save_checkpoint(
        run / "checkpoints" / "latest.pt", model,
        config={**config, "model_kind": checkpoint_kind},
    )
    return run, genome, environment


def test_lab_prefers_persistence_scored_checkpoint(tmp_path):
    run = _saved_run(tmp_path, 3, True)
    (run / "checkpoints" / "best.pt").write_bytes(b"best")

    assert find_checkpoint(run).name == "best.pt"


@pytest.mark.parametrize("dimensions,conditional", [(2, False), (3, True)])
def test_lab_session_seeds_erases_steps_and_renders(tmp_path, dimensions, conditional):
    lab = LabSession.from_run(_saved_run(tmp_path, dimensions, conditional), "cpu")
    lab.reset(clear=True)
    view, layer = ("plane", 0) if dimensions == 2 else ("slice", 3)
    lab.place_seed(view, 1, 2, layer)
    position = (1, 2) if dimensions == 2 else (3, 1, 2)
    assert lab.state[(0, 0, *position)] == 1
    assert lab.frame_png(view, layer).startswith(b"\x89PNG")
    assert lab.frame_png(view, layer, "target").startswith(b"\x89PNG")
    assert lab.summary()["target_cells"] > 1
    if dimensions == 3:
        voxels = lab.voxel_data()
        assert voxels["shape"] == [12, 12, 12]
        assert voxels["voxels"][0][:3] == [3, 1, 2]
        target_voxels = lab.voxel_data(source="target")
        assert target_voxels["shape"] == [12, 12, 12]
        assert len(target_voxels["voxels"]) > 1
    if conditional:
        assert lab.set_genome(2)["genome"] == 2
        assert lab.summary()["target_name"] == MORPHOLOGIES[2]
    lab.erase(view, 1, 2, layer, radius=1)
    assert not lab.state[(0, slice(None), *position)].any()
    assert lab.advance(2)["steps"] == 2


def test_design_lab_http_api(tmp_path):
    run = _saved_run(tmp_path, 2)
    run_3d = _saved_run(tmp_path, 3, True)
    stale_server = create_server(tmp_path, port=0)
    stale_token = stale_server.token
    stale_server.server_close()
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, token=None, **value):
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps({"token": server.token if token is None else token, **value}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.load(urlopen(request, timeout=10))

    try:
        assert build_state(tmp_path)["runs"][0]["lab_ready"]
        with pytest.raises(HTTPError) as error:
            post("/api/lab/load", run=run.name, device="cuda:0")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            post("/api/lab/load", token=stale_token, run=run.name, device="cpu")
        replacement = json.load(error.value)
        assert error.value.code == 403 and replacement["token"] == server.token
        assert post("/api/lab/load", token=replacement["token"], run=run.name, device="cpu")["dimensions"] == 2
        assert post("/api/lab/action", action="clear")["occupied_cells"] == 0
        assert post("/api/lab/action", action="seed", view="plane", row=1, column=2, layer=0)["occupied_cells"] == 1
        frame = urlopen(f"http://127.0.0.1:{server.server_port}/api/lab/frame?view=plane&layer=0", timeout=10).read()
        assert frame.startswith(b"\x89PNG")
        target_frame = urlopen(f"http://127.0.0.1:{server.server_port}/api/lab/frame?view=plane&layer=0&source=target", timeout=10).read()
        assert target_frame.startswith(b"\x89PNG")
        assert post("/api/lab/action", action="erase", view="plane", row=1, column=2, layer=0, radius=1)["occupied_cells"] == 0
        assert post("/api/lab/load", run=run_3d.name, device="cpu")["dimensions"] == 3
        voxels = json.load(urlopen(f"http://127.0.0.1:{server.server_port}/api/lab/voxels", timeout=10))
        assert voxels["shape"] == [12, 12, 12]
        assert len(voxels["voxels"]) == 1
        target_voxels = json.load(urlopen(f"http://127.0.0.1:{server.server_port}/api/lab/voxels?source=target", timeout=10))
        assert target_voxels["shape"] == [12, 12, 12]
        assert len(target_voxels["voxels"]) > 1
    finally:
        server.shutdown()
        server.server_close()


def test_tree_family_lab_keeps_genome_pending_until_reset_and_updates_environment(tmp_path):
    lab = LabSession.from_run(_saved_tree_family(tmp_path), "cpu")
    original = lab.summary()["active_tree_genome"]
    changed = TreeGenome.from_dict(original).with_values({"height": 1}).to_dict()
    pending = lab.set_tree_genome(changed)
    assert pending["genome_pending"]
    assert pending["active_tree_genome"] == original
    reset = lab.reset()
    assert not reset["genome_pending"] and reset["active_tree_genome"] == changed
    environment = {"light_direction_x": 1, "wind_direction_y": 1, "wind_strength": 0.5, "seed": 3}
    updated = lab.set_environment(environment)
    assert updated["environment"]["light_direction_x"] == 1
    assert lab.context.shape == (1, len(ENVIRONMENT_CHANNELS), 16, 16, 16)
    assert lab.advance(1)["steps"] == 1
    assert lab.voxel_data(source="target")["voxels"]
    assert lab.voxel_data(source="environment:light")["voxels"]
    validation = lab.validate_tree_candidate(steps=2, recovery_steps=1, fire_seeds=(0,))
    assert validation["report"]["validated"] is False
    assert validation["genome"] == lab.pending_tree_genome.to_dict()


def test_tree_specialist_lab_preserves_target_environment_and_checks_checkpoint_semantics(tmp_path):
    run, genome, environment = _saved_tree_specialist(tmp_path)
    lab = LabSession.from_run(run, "cpu")
    _, target, materials = lab._target()
    expected_target, expected_materials = make_tree_target(genome, 16, environment)
    assert target.numpy().tolist() == expected_target.tolist()
    assert materials.numpy().tolist() == expected_materials.tolist()
    assert len(lab.checkpoint_sha256) == 64

    mismatched, _, _ = _saved_tree_specialist(tmp_path, checkpoint_kind="specialist")
    with pytest.raises(CheckpointCompatibilityError, match="expected 'tree_specialist'"):
        LabSession.from_run(mismatched, "cpu")


def test_tree_genome_lab_http_schema_and_actions(tmp_path):
    run = _saved_tree_family(tmp_path)
    server = create_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(**value):
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/lab/" + ("load" if "run" in value else "action"),
            data=json.dumps({"token": server.token, **value}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.load(urlopen(request, timeout=10))

    try:
        schema = json.load(urlopen(f"http://127.0.0.1:{server.server_port}/api/tree/schema", timeout=10))
        assert len(schema["genes"]) == 9 and len(schema["environment_channels"]) == 12
        assert schema["genome_schema_version"] == TREE_GENOME_VERSION
        loaded = post(run=run.name, device="cpu")
        randomized = post(action="tree_random", seed=12, locked=["height"])
        assert randomized["genome_pending"] and randomized["pending_tree_genome"]["genes"]["height"] == loaded["active_tree_genome"]["genes"]["height"]
        applied = post(action="reset")
        assert not applied["genome_pending"]
        changed = post(action="environment", environment={"wind_strength": 0.5, "seed": 2})
        assert changed["environment"]["wind_strength"] == 0.5
    finally:
        server.shutdown()
        server.server_close()
