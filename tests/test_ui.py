import json
import hashlib
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from morphovoxel.environment import ENVIRONMENT_CHANNELS, EnvironmentSpec
from morphovoxel.genomes import TREE_GENOME_VERSION, TreeGenome
from morphovoxel.state import StateLayout
from morphovoxel.targets import make_tree_target
from morphovoxel.targets.targets_3d import TREE_TARGET_VERSION
from morphovoxel.ui import CONFIGS, HTML, DashboardHandler, _RecoveredProcess, _inside, _job_view, _launch, build_state, create_server
from morphovoxel.utils import steps_per_second, write_live_preview
from morphovoxel.validation import ValidationCase, ValidationCriteria, ValidationReport, ValidationTrial


def test_checkpoint_selector_accepts_previous_wind_target_version(tmp_path):
    for version in (2, 3, TREE_TARGET_VERSION, 999):
        run = tmp_path / "runs" / f"target_{version}"
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(json.dumps({
            "model_kind": "tree_family", "genome_schema_version": TREE_GENOME_VERSION,
            "target_generator_version": version, "context_channels": len(ENVIRONMENT_CHANNELS),
        }))
    supported = {run["name"] for run in build_state(tmp_path)["runs"] if run["tree_schema_compatible"]}
    assert supported == {"target_3", f"target_{TREE_TARGET_VERSION}"}


@pytest.mark.skipif(os.name == "nt", reason="Detached job recovery uses macOS/Linux process inspection")
@pytest.mark.parametrize("finish", ["stop", "exit"])
def test_restart_recovers_live_jobs_and_stop_controls(tmp_path, monkeypatch, finish):
    # Spaces exercise matching against the unquoted argv printed by ps.
    project = tmp_path / "project with spaces"
    project.mkdir()
    (project / "worker.py").write_text(
        "import pathlib, sys, time\n"
        "config = pathlib.Path(sys.argv[-1])\n"
        "print('training is alive', flush=True)\n"
        "while not config.with_suffix('.finish').exists(): time.sleep(.02)\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(CONFIGS, "smoke_2d.yaml", ("2d", "Test", "Test", "worker.py"))
    first = create_server(project, port=0)
    payload = {"config": "smoke_2d.yaml", "content": "dimensions: 2\n", "device": "cpu", "run_name": "recovered"}
    original = _launch(first, payload)
    process = first.jobs[original["id"]]["process"]
    first.server_close()
    second = create_server(project, port=0)
    thread = threading.Thread(target=second.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{second.server_port}"

    def post(route, body):
        request = Request(base + route, json.dumps({**body, "token": second.token}).encode(), {"Content-Type": "application/json"})
        return json.load(urlopen(request, timeout=5))

    try:
        for _ in range(100):
            view = json.load(urlopen(base + "/api/state", timeout=5))["jobs"][0]
            if "training is alive" in view["log"]:
                break
            time.sleep(.02)
        assert view["id"] == original["id"]
        assert view["run_name"] == "recovered" and view["status"] == "running"
        assert "training is alive" in view["log"]
        assert second.jobs[original["id"]]["process"].pid == process.pid
        with pytest.raises(ValueError, match="already active"):
            _launch(second, payload)
        run = project / "runs" / "recovered"
        run.mkdir()
        with pytest.raises(HTTPError) as error:
            post("/api/run/delete", {"run": "recovered"})
        assert error.value.code == 400 and run.exists()
        with pytest.raises(HTTPError) as error:
            post("/api/job/delete", {"job": original["id"]})
        assert error.value.code == 400
        if finish == "stop":
            post("/api/stop", {"job": original["id"]})
        else:
            second.jobs[original["id"]]["config_path"].with_suffix(".finish").touch()
        process.wait(timeout=5)
        view = json.load(urlopen(base + "/api/state", timeout=5))["jobs"][0]
        assert view["status"] == ("stopped" if finish == "stop" else "ended")
        assert view["return_code"] is None  # A different parent cannot recover the exit code.
        third = create_server(project, port=0)
        try:
            assert third.jobs == {}  # Old config files must not resurrect finished jobs.
        finally:
            third.server_close()
        assert post("/api/job/delete", {"job": original["id"]}) == {"deleted": original["id"]}
    finally:
        second.shutdown()
        second.server_close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


def test_recovered_job_does_not_follow_a_reused_pid(monkeypatch):
    process = _RecoveredProcess(123, ("original start time", "original command"))
    monkeypatch.setattr("morphovoxel.ui._process_snapshot", lambda pid: {123: ("new start time", "original command")})
    assert process.poll() is not None


def test_log_refresh_preserves_reading_position_and_only_follows_at_bottom():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard JavaScript regression check")
    function = re.search(r"function renderWithLogScroll\(.*?\n\}", HTML, re.DOTALL).group()
    script = """
const assert = require('node:assert/strict');
function log(key,height,top=0,left=0){
    const element={dataset:{logKey:key},scrollHeight:height,clientHeight:200,scrollLeft:left};
    Object.defineProperty(element,'scrollTop',{
        get:()=>top,
        set:value=>{top=Math.max(0,Math.min(value,Math.max(0,height-200)))}
    });
    return element;
}
const container={
    scrollTop:55,scrollLeft:12,
    logs:[log('job:reading',800,80,24),log('job:following',800,599.5),log('run:short',100)],
    querySelectorAll(){return this.logs},
    set innerHTML(value){
        this.logs=JSON.parse(value).map(([key,height])=>log(key,height));
        this.scrollTop=0;this.scrollLeft=0;
    }
};
""" + function + """
// Refreshed cards can change order or gain new output independently.
const update=JSON.stringify([['job:following',900],['job:reading',1100],['job:new',500],['run:short',600]]);
renderWithLogScroll(container,update);
assert.equal(container.scrollTop,55);
assert.equal(container.scrollLeft,12);
assert.deepEqual(container.logs.map(item=>item.scrollTop),[700,80,300,400]);
assert.equal(container.logs[1].scrollLeft,24);

// Scrolling to the top opts out of following, even during repeated refreshes.
container.logs[0].scrollTop=0;
renderWithLogScroll(container,update);
assert.deepEqual(container.logs.map(item=>item.scrollTop),[0,80,300,400]);
renderWithLogScroll(container,JSON.stringify([['run:different',1000]]));
assert.equal(container.logs[0].scrollTop,800);
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_checkpoint_selector_handles_empty_dashboard_and_preserves_selection():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard JavaScript regression check")
    function = next(line for line in HTML.splitlines() if line.startswith("function syncLabCheckpoints()"))
    script = """
const assert = require('node:assert/strict');
let state = {runs: []}, lab = null, labLoading = false;
const select = {dataset: {}, value: ''};
const elements = {
    '#labCheckpoint': select,
    '#labRun': {value: ''},
    '#labDeleteCheckpoint': {}
};
const $ = selector => elements[selector], esc = String;
""" + function + """
// Initial page load, before any run or checkpoint exists.
syncLabCheckpoints();
assert.equal(select.disabled, true);
assert.match(select.innerHTML, /No checkpoints/);
assert.equal(elements['#labDeleteCheckpoint'].disabled, true);

// A new training run has no checkpoint or open lab yet.
state.runs = [{name: 'tree', checkpoints: []}];
elements['#labRun'].value = 'tree';
syncLabCheckpoints();
assert.equal(select.disabled, true);

// Checkpoints appear, then an active lab is opened on the same run.
state.runs[0].checkpoints = ['best.pt', 'latest.pt'];
syncLabCheckpoints();
assert.equal(select.disabled, false);
assert.equal(elements['#labDeleteCheckpoint'].disabled, false);
lab = {run: 'tree', checkpoint: 'latest.pt'};
syncLabCheckpoints();
assert.equal(select.value, 'latest.pt');
select.value = 'best.pt';
syncLabCheckpoints();
assert.equal(select.value, 'best.pt');

// The final run is removed and the selector returns to its empty state.
state.runs = [];
lab = null;
syncLabCheckpoints();
assert.equal(select.disabled, true);
assert.equal(elements['#labDeleteCheckpoint'].disabled, true);
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_hidden_layer_control_roundtrips_yaml_and_rejects_invalid_widths():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to check the dashboard controls")
    functions = [
        next(line for line in HTML.splitlines() if line.startswith(f"function {name}("))
        for name in ("yamlNumber", "setYamlNumber")
    ]
    functions.extend(re.search(rf"function {name}\(.*?\n\}}", HTML, re.DOTALL).group()
                     for name in ("hiddenLayerText", "setHiddenLayers"))
    functions.insert(0, next(line for line in HTML.splitlines() if line.startswith("const hiddenLayersPattern=")))
    functions.append(next(line for line in HTML.splitlines() if line.startswith("async function launch()")))
    script = """
const assert=require('node:assert/strict');
const editor={value:'model_width: 64\\niterations: 10\\n'};
const elements={
  '#hiddenLayers':{value:'32',reportValidity:()=>true}, '#runName':{value:''},
  '#configSelect':{value:'tree_family.yaml'}, '#device':{value:'cpu'}, '#livePreview':{checked:true}
};
const $=selector=>selector==='#editor'?editor:elements[selector];
const TOKEN='test',toast=()=>{},refresh=async()=>{};
let posted;
const api=async(path,options)=>{posted=JSON.parse(options.body);return {run_name:'test'}};
""" + "\n".join(functions) + r"""
assert.equal(hiddenLayerText(),'64');
setHiddenLayers('32, 32');
assert.equal(hiddenLayerText(),'32, 32');
assert.match(editor.value,/hidden_layers: \[32, 32\]/);
assert.match(editor.value,/model_width: 32/);
assert.match(editor.value,/iterations: 10/);
editor.value='hidden_layers: # custom network\n  - 48\n\n  # second layer\n  - 24\niterations: 10\n';
assert.equal(hiddenLayerText(),'48, 24');
setHiddenLayers('16, 8, 4');
assert.equal(hiddenLayerText(),'16, 8, 4');
assert.doesNotMatch(editor.value,/48|24/);
assert.match(editor.value,/iterations: 10/);
const saved=editor.value;
for(const value of ['', '32,', '0', '-1, 32', '2.5', 'true', '32 32']){
  assert.throws(()=>setHiddenLayers(value),/positive neuron counts/);
  assert.equal(editor.value,saved);
}
// Unsupported or invalid YAML stays in the editor for the server to parse.
for(const value of ['hidden_layers: -32', 'hidden_layers: []', 'hidden_layers: *layers', 'base:\n  model_width: 64']){
  editor.value=value+'\n';
  assert.equal(hiddenLayerText(),null);
}
editor.value='hidden_layers: [\n  48,\n  24\n  ]\niterations: 10\n';
assert.equal(hiddenLayerText(),'48, 24');
setHiddenLayers('32, 32');
assert.equal(editor.value,'hidden_layers: [32, 32]\niterations: 10\n');
(async()=>{
  // Launch must not rewrite advanced YAML from a stale quick-control value.
  for(const content of ['hidden_layers: -32\n', 'hidden_layers: [48, 24]\n', 'model_width: 64\n']){
    editor.value=content;
    await launch();
    assert.equal(posted.content,content);
    assert.equal(editor.value,content);
  }
  elements['#hiddenLayers'].reportValidity=()=>false;
  posted=null;
  await launch();
  assert.equal(posted,null);
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("preset,section", [("full_experiment.yaml", "overrides"), ("ecology_experiments.yaml", "base")])
def test_launch_preserves_and_validates_nested_model_settings(tmp_path, monkeypatch, preset, section):
    from morphovoxel.config import load_config

    class Process:
        pid = 1

        def poll(self):
            return 0

    monkeypatch.setattr("morphovoxel.ui.subprocess.Popen", lambda *args, **kwargs: Process())
    server = create_server(tmp_path, port=0)
    try:
        payload = {"config": preset, "device": "cpu", "live_preview": False}
        job = _launch(server, {**payload, "content": f"{section}:\n  hidden_layers: [32, 32]\n  learning_rate: 0.002\n"})
        config = load_config(server.jobs[job["id"]]["config_path"])
        assert config[section] == {"hidden_layers": [32, 32], "learning_rate": .002, "device": "cpu", "live_preview": False}
        for content in (f"{section}: []\n", f"{section}:\n  hidden_layers: [-32]\n"):
            with pytest.raises(ValueError, match="mapping|positive integers"):
                _launch(server, {**payload, "content": content})
    finally:
        server.server_close()


def test_phase_two_dashboard_keeps_curriculum_and_checkpoint_selection_in_yaml():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard JavaScript regression check")
    functions = [re.search(r"function syncFamilyCurriculum\(.*?\n\}", HTML, re.DOTALL).group()]
    functions.append(re.search(r"const dependencySpecs=\{.*?\};", HTML, re.DOTALL).group())
    functions.extend(
        next(line for line in HTML.splitlines() if line.startswith(f"function {name}("))
        for name in ("yamlText", "yamlNumber", "setYamlText", "setDependencyCheckpoint", "syncDependencyCheckpoints")
    )
    script = """
const assert=require('node:assert/strict');
const elements={};
const $=id=>elements[id]||(elements[id]={value:'',textContent:'',innerHTML:'',hidden:false});
const esc=String;
let kind='family';
const state={runs:[
  {name:'specialist',kind:'specialist',model_kind:'tree_specialist',context_channels:0,tree_schema_compatible:true,checkpoints:['best.pt']},
  {name:'basic_families',kind:'family',model_kind:'tree_family',context_channels:12,tree_schema_compatible:true,checkpoints:['best.pt','basic_families.pt']},
  {name:'outdated',kind:'family',model_kind:'tree_family',context_channels:12,tree_schema_compatible:false,checkpoints:['best.pt']},
  {name:'legacy',kind:'conditional',model_kind:'legacy_conditional',context_channels:0,checkpoints:['best.pt']}
]};
$('#configSelect').value='tree_family.yaml';
""" + "\n".join(functions) + r"""
for(const mode of ['full','basics','variation']){
  $('#editor').value=`family_curriculum: ${mode}\ninitialize_from_checkpoint: runs/basic_families/checkpoints/best.pt\n`;
  syncDependencyCheckpoints();
  assert.equal($('#familyCurriculum').value,mode);
  assert.equal($('#familyCurriculumField').hidden,false);
  assert.equal($('#dependencyCheckpoint').value,'runs/basic_families/checkpoints/best.pt');
  assert.match($('#dependencyCheckpoint').innerHTML,/runs\/specialist\/checkpoints\/best.pt/);
  assert.match($('#dependencyCheckpoint').innerHTML,/basic_families.pt/);
  assert.doesNotMatch($('#dependencyCheckpoint').innerHTML,/outdated|legacy/);
  setDependencyCheckpoint('runs/specialist/checkpoints/best.pt');
  assert.equal(yamlText('initialize_from_checkpoint'),'runs/specialist/checkpoints/best.pt');
  assert.equal(yamlText('family_curriculum'),mode);
}
$('#editor').value+='resume: old.pt\ninitialize_from_specialist: old.pt\n';
setDependencyCheckpoint('runs/basic_families/checkpoints/best.pt');
assert.doesNotMatch($('#editor').value,/^resume:|^initialize_from_specialist:/m);
kind='specialist';$('#configSelect').value='tree_specialist.yaml';
syncDependencyCheckpoints();
assert.equal($('#familyCurriculumField').hidden,true);
assert.equal($('#dependencyCheckpointField').hidden,true);
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_step_rate_uses_completed_updates(monkeypatch):
    monkeypatch.setattr("morphovoxel.utils.time.perf_counter", lambda: 12.0)
    assert steps_per_second(24, 10.0) == 12.0
    assert steps_per_second(0, 10.0) == 0.0


def test_job_training_stats_work_without_previews_and_survive_log_tail_truncation(tmp_path, monkeypatch):
    class Process:
        returncode = None

        def poll(self):
            return self.returncode

    log = tmp_path / "job.log"
    process = Process()
    job = {
        "id": "job", "config": "tree_family.yaml", "run_name": "tree", "process": process,
        "command": "train", "started": "2026-09-29T12:00:00+00:00", "log_path": log,
    }
    now = 100.0
    monkeypatch.setattr("morphovoxel.ui.time.monotonic", lambda: now)
    assert _job_view(job)["progress"] is None
    # Resumed checkpoints start at a large step; count newly logged updates.
    content = "INFO step=12001 loss=2.0\n" + "validation detail\n" * 100
    log.write_text(content)
    first = _job_view(job)
    assert first["live"] is None and "loss=" not in first["log"]
    assert first["progress"] == {
        "iteration": 12001, "completed_iterations": 1, "average_loss": 2.0, "iterations_per_second": None,
    }
    now += 2
    log.write_text(content + "INFO step=12002 loss=4.0\nINFO step=12003 loss=6.0\nINFO step=12004 loss=")
    progress = _job_view(job)["progress"]
    assert progress["completed_iterations"] == 3
    assert progress["iterations_per_second"] == 1.0
    assert progress["average_loss"] == 4.0
    # Repeated polls do not duplicate losses, and validation contributes no updates.
    now += 2
    assert _job_view(job)["progress"]["iterations_per_second"] == .5
    for offset in range(6, 40, 2):
        now = 100 + offset
        progress = _job_view(job)["progress"]
    assert progress["iterations_per_second"] == 0
    assert progress["average_loss"] == 4.0
    # A new phase may restart its step counter; do not turn that into negative speed.
    now += 2
    with log.open("a") as stream:
        stream.write("\nINFO step=1 loss=8.0\nINFO step=2 loss=nan\n")
    assert _job_view(job)["progress"]["average_loss"] == 5.0
    assert _job_view(job)["progress"]["iterations_per_second"] > 0
    process.returncode = 0
    finished = _job_view(job)["progress"]
    now += 1000
    assert _job_view(job)["progress"] == finished
    # Dashboard recovery recomputes average loss and measures a fresh rate window.
    process.returncode = None
    recovered = _job_view({key: value for key, value in job.items() if not key.startswith("_")})
    assert recovered["progress"]["average_loss"] == 5.0
    assert recovered["progress"]["iterations_per_second"] is None
    # Older losses must leave the rolling window without capping the rate counter.
    log.write_text("".join(f"INFO step={step} loss={100 if step <= 80 else 2}\n" for step in range(1, 281)))
    now += 2
    progress = _job_view(job)["progress"]
    assert progress["average_loss"] == 2.0
    assert progress["completed_iterations"] == 280
    with log.open("a") as stream:
        stream.write("INFO step=281 loss=4\n")
    now += 2
    progress = _job_view(job)["progress"]
    assert progress["average_loss"] == pytest.approx(2.01)
    assert progress["completed_iterations"] == 281
    assert progress["iterations_per_second"] > 0


def test_running_job_renders_training_stats_without_an_image():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to check the dashboard controls")
    function = re.search(r"function trainingStats\(.*?\n\}", HTML, re.DOTALL).group()
    render = next(line for line in HTML.splitlines() if line.startswith("function jobs()"))
    script = """
const assert=require('node:assert/strict');
const count={}, $=()=>count, esc=String;
const state={runs:[],jobs:[{id:'test',run_name:'tree',status:'running',live:null,log:'training',
    progress:{iteration:12003,completed_iterations:260,iterations_per_second:1.25,average_loss:3.123456}}]};
""" + function + "\n" + render + r"""
let html=jobs();
assert.match(html,/1.25/);
assert.match(html,/iterations\/second/);
assert.match(html,/3.1235/);
assert.match(html,/average loss/);
assert.match(html,/last 200/);
assert.doesNotMatch(html,/<img/);
assert.match(html,/data-stop-job/);
state.jobs[0].progress.iterations_per_second=null;
assert.match(jobs(),/Measuring…/);
state.jobs[0].progress.completed_iterations=3;
assert.match(jobs(),/last 3/);
state.jobs[0].progress=null;
assert.doesNotMatch(jobs(),/NaN|average loss/);
"""
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


def test_response_ignores_a_browser_disconnect():
    class ClosedClient:
        send_response = send_header = end_headers = lambda *args: None

        class Stream:
            @staticmethod
            def write(_body):
                raise ConnectionAbortedError(10053, "client canceled request")

        wfile = Stream()

    DashboardHandler._send(ClosedClient(), b"stale poll", "application/json")


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


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_full_presets_precede_smoke_presets_and_missing_dependencies_are_blocked(tmp_path, monkeypatch, device):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
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
                    "initialize_from_checkpoint: runs/tree_specialist/checkpoints/best.pt\n"
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
            "device": device,
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
    (visual.parent / "config.yaml").write_text(
        "dashboard_preset: tree_family.yaml\nworld_size: 16\nbatch_size: 8\n",
        encoding="utf-8",
    )

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
        assert "saved runs" in root and "active jobs" in root
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
        assert "'tree_family.yaml':{key:'initialize_from_checkpoint'" in root
        assert 'id="familyCurriculum"' in root
        assert "Learn the basic families only" in root and "Learn variation only" in root
        assert "models:['tree_specialist','tree_family']" in root
        assert "'tree_regeneration.yaml':{key:'resume'" in root
        assert "'tree_environment.yaml':{key:'resume'" in root
        assert "'tree_ecology.yaml':{key:'checkpoint'" in root
        assert "No compatible checkpoints found" in root
        assert 'id="labCheckpoint"' in root
        assert 'id="labDeleteCheckpoint"' in root
        assert "/api/checkpoint/delete" in root
        assert "/api/run/delete" in root and "/api/job/delete" in root
        assert "data-run-preset" in root and "data-run-delete" in root and "Delete job" in root
        assert "font-size:30px" in root and "scaleX(2) rotate(180deg)" in root
        assert ".run-card [data-run-delete]{margin-left:auto;padding:4px 8px;font-size:11px" in root
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
        assert payload["runs"][0]["preset"] == "tree_family.yaml"
        assert payload["runs"][0]["settings"]["batch_size"] == 8
        assert payload["runs"][0]["model_kind"] == ""
        assert payload["runs"][0]["context_channels"] == 0
        assert payload["hardware"]["auto_device"] in {"cpu", "cuda", "mps"}
        assert payload["hardware"]["mps_available"] == torch.backends.mps.is_available()
        assert root.count('<option value="mps">Apple GPU (Metal)</option>') == 3
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


def test_run_and_completed_job_delete_are_scoped(tmp_path):
    run = tmp_path / "runs" / "finished"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"best")
    job_root = tmp_path / "runs" / ".ui_jobs"
    job_root.mkdir()
    log_path, config_path = job_root / "done.log", job_root / "done.yaml"
    log_path.write_text("done", encoding="utf-8")
    config_path.write_text("run_name: finished\n", encoding="utf-8")

    class Finished:
        @staticmethod
        def poll():
            return 0

    class Active:
        @staticmethod
        def poll():
            return None

    server = create_server(tmp_path, port=0)
    server.jobs["done"] = {
        "id": "done", "run_name": "finished", "process": Finished(),
        "log_path": log_path, "config_path": config_path,
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, payload):
        request = Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=json.dumps({"token": server.token, **payload}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        return json.load(urlopen(request, timeout=5))

    try:
        assert post("/api/job/delete", {"job": "done"}) == {"deleted": "done"}
        assert not log_path.exists() and not config_path.exists() and "done" not in server.jobs

        assert post("/api/run/delete", {"run": "finished"}) == {"deleted": "finished", "closed_lab": False}
        assert not run.exists()

        active_run = tmp_path / "runs" / "active"
        active_run.mkdir()
        server.jobs["active"] = {"id": "active", "run_name": "active", "process": Active()}
        with pytest.raises(HTTPError) as error:
            post("/api/run/delete", {"run": "active"})
        assert error.value.code == 400
        assert active_run.exists()
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
