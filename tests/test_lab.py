import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from morphovoxel.checkpointing import save_checkpoint
from morphovoxel.config import save_config
from morphovoxel.genomes import MORPHOLOGIES
from morphovoxel.lab import LabSession
from morphovoxel.model_2d import NeuralCA2D
from morphovoxel.model_3d import NeuralCA3D
from morphovoxel.state import StateLayout
from morphovoxel.ui import build_state, create_server


def _saved_run(root, dimensions: int, conditional: bool = False):
    run = root / "runs" / f"lab_{dimensions}d"
    (run / "checkpoints").mkdir(parents=True)
    materials = 3 if dimensions == 2 else 4
    config = {
        "run_name": run.name, "dimensions": dimensions, "conditional": conditional,
        "device": "cpu", "world_size": 7, "materials": materials,
        "hidden_channels": 2, "model_width": 4, "fire_rate": 1.0,
    }
    save_config(config, run / "config.yaml")
    layout = StateLayout(materials, 2)
    model_class = NeuralCA2D if dimensions == 2 else NeuralCA3D
    model = model_class(layout.channels, 4, len(MORPHOLOGIES) if conditional else 0, 1.0)
    save_checkpoint(run / "checkpoints" / "latest.pt", model, config=config)
    return run


@pytest.mark.parametrize("dimensions,conditional", [(2, False), (3, True)])
def test_lab_session_seeds_erases_steps_and_renders(tmp_path, dimensions, conditional):
    lab = LabSession.from_run(_saved_run(tmp_path, dimensions, conditional), "cpu")
    lab.reset(clear=True)
    view, layer = ("plane", 0) if dimensions == 2 else ("slice", 3)
    lab.place_seed(view, 1, 2, layer)
    position = (1, 2) if dimensions == 2 else (3, 1, 2)
    assert lab.state[(0, 0, *position)] == 1
    assert lab.frame_png(view, layer).startswith(b"\x89PNG")
    if conditional:
        assert lab.set_genome(2)["genome"] == 2
    lab.erase(view, 1, 2, layer, radius=1)
    assert not lab.state[(0, slice(None), *position)].any()
    assert lab.advance(2)["steps"] == 2


def test_design_lab_http_api(tmp_path):
    run = _saved_run(tmp_path, 2)
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
        assert post("/api/lab/action", action="erase", view="plane", row=1, column=2, layer=0, radius=1)["occupied_cells"] == 0
    finally:
        server.shutdown()
        server.server_close()
