"""Local browser dashboard for running and inspecting MorphoVoxel experiments."""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import torch

from .config import load_config, save_config
from .lab import LabSession, find_checkpoint
from .random_utils import resolve_device

CONFIGS: dict[str, tuple[str, str, str, str]] = {
    "full_experiment.yaml": ("experiments", "Full experiment", "Recommended · all phases in order", "scripts/run_full_experiment.py"),
    "phase1_2d.yaml": ("2d", "2D training", "Full growth", "morphovoxel.train_2d"),
    "phase2_3d.yaml": ("3d", "3D training", "Single morphology", "morphovoxel.train_3d"),
    "phase3_conditional.yaml": ("conditional", "Conditional training", "Shared 3D rule", "morphovoxel.train_conditional"),
    "phase4_regeneration_training.yaml": ("regeneration", "Damage training", "Creates the recovery checkpoint", "morphovoxel.train_conditional"),
    "phase4_regeneration.yaml": ("regeneration", "Regeneration evaluation", "Requires Phase 3 + damage training", "morphovoxel.evaluate_regeneration"),
    "phase5_ecology.yaml": ("ecology", "Ecology run", "Requires Genome lab training", "morphovoxel.run_ecology"),
    "ecology_experiments.yaml": ("ecology", "Ecology matrix", "Requires Genome lab training", "scripts/run_ecology_experiment.py"),
    "smoke_2d.yaml": ("2d", "2D smoke", "Fast pipeline check", "morphovoxel.train_2d"),
    "smoke_3d.yaml": ("3d", "3D smoke", "Fast voxel pipeline check", "morphovoxel.train_3d"),
    "smoke_conditional.yaml": ("conditional", "Conditional smoke", "Fast genome pipeline check", "morphovoxel.train_conditional"),
    "smoke_regeneration.yaml": ("regeneration", "Regeneration smoke", "Untrained pipeline check", "morphovoxel.evaluate_regeneration"),
    "smoke_ecology.yaml": ("ecology", "Ecology smoke", "Untrained pipeline check", "morphovoxel.run_ecology"),
}

MEDIA_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg", ".webp"}
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _inside(root: Path, relative: str) -> Path:
    """Resolve a relative path without allowing traversal outside root."""
    if Path(relative).is_absolute():
        raise ValueError("absolute paths are not allowed")
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("path leaves the allowed directory")
    return target


def _csv_preview(path: Path, limit: int = 20) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)[-limit:]
        return {"name": path.name, "columns": reader.fieldnames or [], "rows": rows}


def _run_kind(run: Path) -> str:
    config_path = run / "config.yaml"
    if not config_path.exists():
        return "experiments"
    try:
        config = load_config(config_path)
    except (OSError, ValueError):
        return "experiments"
    if "organisms" in config or "incident_light" in config:
        return "ecology"
    if "damage_probability" in config or ("damage_types" in config and "growth_steps" in config):
        return "regeneration"
    if config.get("conditional"):
        return "conditional"
    return "2d" if config.get("dimensions") == 2 else "3d"


def build_state(project_root: Path) -> dict[str, Any]:
    """Return dashboard data derived only from configs and run artifacts."""
    configs = []
    for filename, (kind, title, subtitle, _) in CONFIGS.items():
        path = project_root / "configs" / filename
        if path.exists():
            configs.append({
                "name": filename, "kind": kind, "title": title, "subtitle": subtitle,
                "content": path.read_text(encoding="utf-8"),
            })

    runs = []
    runs_root = project_root / "runs"
    for run in runs_root.iterdir() if runs_root.exists() else ():
        if not run.is_dir() or run.name.startswith("."):
            continue
        media = [
            {
                "name": path.name,
                "url": "/media/" + quote(path.relative_to(runs_root).as_posix(), safe="/") + f"?v={path.stat().st_mtime_ns}",
            }
            for path in sorted((run / "visualizations").glob("*"))
            if not path.name.startswith(".") and path.suffix.lower() in MEDIA_SUFFIXES
        ] if (run / "visualizations").exists() else []
        metrics = []
        if (run / "metrics").exists():
            for path in sorted((run / "metrics").glob("*.csv")):
                try:
                    metrics.append(_csv_preview(path))
                except (OSError, UnicodeError, csv.Error):
                    continue
        logs = ""
        if (run / "logs.csv").exists():
            logs = "\n".join((run / "logs.csv").read_text(encoding="utf-8").splitlines()[-12:])
        timestamps = [run.stat().st_mtime]
        for path in run.rglob("*"):
            if any(part.startswith(".") for part in path.relative_to(run).parts):
                continue
            try:
                if path.is_file():
                    timestamps.append(path.stat().st_mtime)
            except FileNotFoundError:
                continue
        updated = max(timestamps)
        runs.append({
            "name": run.name, "kind": _run_kind(run), "media": media, "metrics": metrics,
            "logs": logs, "updated": datetime.fromtimestamp(updated, timezone.utc).isoformat(),
            "lab_ready": (run / "config.yaml").is_file() and find_checkpoint(run) is not None,
        })
    runs.sort(key=lambda item: item["updated"], reverse=True)
    cuda_available = torch.cuda.is_available()
    return {
        "configs": configs, "runs": runs,
        "hardware": {
            "auto_device": str(resolve_device("auto")), "cuda_available": cuda_available,
            "cuda_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "torch_version": torch.__version__,
        },
    }


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    process = job["process"]
    return_code = process.poll()
    status = "running" if return_code is None else "complete" if return_code == 0 else "failed"
    try:
        log = "\n".join(job["log_path"].read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
    except OSError:
        log = ""
    live = None
    live_path = job.get("live_path")
    if live_path and live_path.is_file():
        try:
            progress = json.loads(live_path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = {}
        relative = live_path.relative_to(job["runs_root"]).as_posix()
        live = {
            **progress,
            "url": "/media/" + quote(relative, safe="/") + f"?v={live_path.stat().st_mtime_ns}",
        }
    return {
        "id": job["id"], "config": job["config"], "run_name": job["run_name"],
        "command": job["command"], "started": job["started"], "status": status,
        "return_code": return_code, "log": log, "live": live,
    }


def _launch(server: "DashboardServer", payload: dict[str, Any]) -> dict[str, Any]:
    filename = str(payload.get("config", ""))
    if filename not in CONFIGS:
        raise ValueError("unknown configuration")
    content = payload.get("content")
    if not isinstance(content, str) or len(content.encode("utf-8")) > 1_000_000:
        raise ValueError("configuration content must be text under 1 MB")
    import yaml
    config = yaml.safe_load(content) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    device = str(payload.get("device", "auto")).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    resolve_device(device)
    live_preview = payload.get("live_preview", True)
    if not isinstance(live_preview, bool):
        raise ValueError("live_preview must be true or false")
    overrides = {"device": device, "live_preview": live_preview}
    if filename == "ecology_experiments.yaml":
        base = config.get("base")
        if not isinstance(base, dict):
            raise ValueError("ecology experiment base must be a mapping")
        base.update(overrides)
    elif filename == "full_experiment.yaml":
        config["overrides"] = overrides
    else:
        config.update(overrides)

    if filename != "full_experiment.yaml":
        checkpoint_config = config.get("base", config) if filename == "ecology_experiments.yaml" else config
        checkpoint_group = checkpoint_config.get("checkpoints") or {}
        if not isinstance(checkpoint_group, dict):
            raise ValueError("checkpoints must be a mapping of labels to paths")
        required = [checkpoint_config.get("checkpoint")]
        required.extend(checkpoint_group.values())
        missing = []
        for value in filter(None, required):
            path = Path(str(value))
            candidate = path if path.is_absolute() else server.project_root / path
            if not candidate.is_file():
                missing.append(str(path))
        if missing:
            raise ValueError(
                f"Missing prerequisite checkpoint(s): {', '.join(missing)}. "
                "Run the required training presets first, or launch Full experiment to build every phase in order."
            )

    with server.jobs_lock:
        if sum(job["process"].poll() is None for job in server.jobs.values()) >= 2:
            raise ValueError("two jobs are already running")
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
        requested_name = str(payload.get("run_name", "")).strip()
        configured_name = "" if filename.startswith("smoke_") else str(config.get("run_name", "")).strip()
        run_name = requested_name or configured_name or f"ui_{Path(filename).stem}_{job_id[-13:]}"
        if not RUN_NAME.fullmatch(run_name):
            raise ValueError("run name may contain only letters, numbers, underscores, and hyphens")
        if any(job["run_name"] == run_name and job["process"].poll() is None for job in server.jobs.values()):
            raise ValueError("that run name is already active")

        kind, _, _, target = CONFIGS[filename]
        run_path = server.project_root / "runs" / run_name
        if kind != "experiments" and filename != "ecology_experiments.yaml" and run_path.exists():
            raise ValueError("that run name already exists; choose a new one")
        if kind not in {"experiments"} and filename != "ecology_experiments.yaml":
            config["run_name"] = run_name
            config["runs_root"] = "runs"
        job_root = server.project_root / "runs" / ".ui_jobs"
        job_root.mkdir(parents=True, exist_ok=True)
        config_path, log_path = job_root / f"{job_id}.yaml", job_root / f"{job_id}.log"
        save_config(config, config_path)
        command = [sys.executable]
        command += ["-m", target] if target.startswith("morphovoxel.") else [target]
        command += ["--config", str(config_path)]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=server.project_root, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                start_new_session=os.name != "nt",
            )
        # ponytail: in-memory job registry; use SQLite when restart persistence or multiple users matter.
        runs_root = server.project_root / "runs"
        server.jobs[job_id] = {
            "id": job_id, "process": process, "config": filename, "run_name": run_name,
            "command": " ".join(command), "started": datetime.now(timezone.utc).isoformat(), "log_path": log_path,
            "runs_root": runs_root,
            "live_path": None if kind == "experiments" or filename == "ecology_experiments.yaml" else runs_root / run_name / "visualizations" / "live.png",
        }
        return _job_view(server.jobs[job_id])


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], project_root: Path):
        super().__init__(address, DashboardHandler)
        self.project_root = project_root.resolve()
        self.token = secrets.token_urlsafe(24)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
        # ponytail: one local lab session; add per-user sessions only if this server becomes multi-user.
        self.lab: LabSession | None = None
        self.lab_lock = threading.Lock()


class InvalidRequestToken(PermissionError):
    pass


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request is too large")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        if value.get("token") != self.server.token:
            raise InvalidRequestToken("invalid request token")
        return value

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path == "/":
                self._send(HTML.replace("__TOKEN__", self.server.token).encode(), "text/html; charset=utf-8")
            elif route.path == "/api/state":
                state = build_state(self.server.project_root)
                with self.server.jobs_lock:
                    state["jobs"] = [_job_view(job) for job in reversed(self.server.jobs.values())]
                with self.server.lab_lock:
                    state["lab"] = self.server.lab.summary() if self.server.lab else None
                self._json(state)
            elif route.path == "/api/lab/frame":
                query = parse_qs(route.query)
                view = query.get("view", ["plane"])[0]
                layer = int(query.get("layer", ["0"])[0])
                source = query.get("source", ["organism"])[0]
                with self.server.lab_lock:
                    if self.server.lab is None:
                        raise ValueError("open a checkpoint-backed run in the lab first")
                    self._send(self.server.lab.frame_png(view, layer, source), "image/png")
            elif route.path == "/api/lab/voxels":
                source = parse_qs(route.query).get("source", ["organism"])[0]
                with self.server.lab_lock:
                    if self.server.lab is None:
                        raise ValueError("open a checkpoint-backed run in the lab first")
                    self._json(self.server.lab.voxel_data(source=source))
            elif route.path.startswith("/media/"):
                relative = unquote(route.path.removeprefix("/media/"))
                path = _inside(self.server.project_root / "runs", relative)
                if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                    raise FileNotFoundError
                self._send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (OSError, ValueError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._payload()
            if self.path == "/api/run":
                self._json(_launch(self.server, payload), HTTPStatus.ACCEPTED)
            elif self.path == "/api/lab/load":
                run_name = str(payload.get("run", ""))
                if not RUN_NAME.fullmatch(run_name):
                    raise ValueError("unknown run")
                device = str(payload.get("device", "auto")).lower()
                if device not in {"auto", "cpu", "cuda"}:
                    raise ValueError("device must be auto, cpu, or cuda")
                run = _inside(self.server.project_root / "runs", run_name)
                if not run.is_dir():
                    raise ValueError("unknown run")
                session = LabSession.from_run(run, device)
                with self.server.lab_lock:
                    self.server.lab = session
                    self._json(session.summary())
            elif self.path == "/api/lab/action":
                action = str(payload.get("action", ""))

                def integer(name: str, default: int | None = None) -> int:
                    value = payload.get(name, default)
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError(f"{name} must be an integer")
                    return value

                with self.server.lab_lock:
                    lab = self.server.lab
                    if lab is None:
                        raise ValueError("open a checkpoint-backed run in the lab first")
                    if action == "advance":
                        result = lab.advance(integer("steps", 1))
                    elif action == "reset":
                        result = lab.reset()
                    elif action == "clear":
                        result = lab.reset(clear=True)
                    elif action == "genome":
                        result = lab.set_genome(integer("genome"))
                    elif action in {"seed", "erase"}:
                        view = str(payload.get("view", "plane"))
                        row, column = integer("row"), integer("column")
                        layer = integer("layer", 0)
                        result = lab.place_seed(view, row, column, layer) if action == "seed" else lab.erase(
                            view, row, column, layer, integer("radius", 2),
                        )
                    else:
                        raise ValueError("unknown lab action")
                    self._json(result)
            elif self.path == "/api/stop":
                job_id = str(payload.get("job", ""))
                with self.server.jobs_lock:
                    job = self.server.jobs.get(job_id)
                    if not job:
                        raise ValueError("unknown job")
                    if job["process"].poll() is None:
                        if os.name == "nt":
                            subprocess.run(
                                ["taskkill", "/PID", str(job["process"].pid), "/T", "/F"],
                                capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            )
                        else:
                            os.killpg(job["process"].pid, signal.SIGTERM)
                    self._json(_job_view(job))
            else:
                raise FileNotFoundError
        except InvalidRequestToken as error:
            self._json({"error": str(error), "token": self.server.token}, HTTPStatus.FORBIDDEN)
        except PermissionError as error:
            self._json({"error": str(error)}, HTTPStatus.FORBIDDEN)
        except FileNotFoundError:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MorphoVoxel Lab</title><style>
:root{color-scheme:dark;--bg:#07110f;--panel:#0d1c18;--panel2:#11251f;--line:#24443a;--mint:#78f7bf;--lime:#c6ff67;--text:#eaf9f2;--muted:#8ca89d;--danger:#ff7b7b;--shadow:0 24px 70px #0008}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 70% -10%,#164634 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}button,input,select,textarea{font:inherit}.shell{display:grid;grid-template-columns:240px 1fr;min-height:100vh}.side{position:sticky;top:0;height:100vh;padding:26px 18px;border-right:1px solid var(--line);background:#081612e8;backdrop-filter:blur(16px)}.brand{display:flex;gap:11px;align-items:center;margin:0 8px 28px}.mark{width:36px;height:36px;border:1px solid var(--mint);border-radius:11px;box-shadow:inset 0 0 18px #78f7bf33;display:grid;place-items:center;color:var(--mint)}.brand b{font-size:16px;letter-spacing:.04em}.brand small{display:block;color:var(--muted)}.nav button{width:100%;border:0;background:transparent;color:var(--muted);padding:11px 12px;margin:2px 0;border-radius:10px;text-align:left;cursor:pointer}.nav button:hover,.nav button.active{background:#18372c;color:var(--text)}.nav button.active{box-shadow:inset 3px 0 var(--mint)}.side-foot{position:absolute;bottom:22px;left:26px;color:var(--muted);font-size:12px}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--mint);box-shadow:0 0 10px var(--mint);margin-right:7px}.main{padding:32px 38px 60px;min-width:0}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:26px}.eyebrow{color:var(--mint);text-transform:uppercase;letter-spacing:.16em;font-size:11px}.top h1{font-size:clamp(28px,4vw,48px);line-height:1;margin:7px 0 0;letter-spacing:-.045em}.counts{display:flex;gap:9px}.pill{border:1px solid var(--line);background:#11231e;padding:8px 12px;border-radius:99px;color:var(--muted)}.grid{display:grid;grid-template-columns:minmax(310px,.75fr) minmax(420px,1.25fr);gap:18px}.panel{background:linear-gradient(145deg,#10231d,#0b1915);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden}.panel-head{display:flex;align-items:center;justify-content:space-between;padding:18px 20px;border-bottom:1px solid var(--line)}.panel-head h2{font-size:15px;margin:0}.panel-body{padding:18px}.field{display:block;color:var(--muted);font-size:12px;margin-bottom:14px}.field span{display:block;margin-bottom:6px}.option-grid,.quick-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.quick-controls{margin:2px 0 16px;padding:13px;border:1px solid var(--line);border-radius:12px;background:#091713}.quick-title{display:flex;justify-content:space-between;margin-bottom:10px;color:var(--text);font-weight:700}.quick-title small{color:var(--muted);font-weight:400}.slider{color:var(--muted);font-size:11px}.slider-head{display:flex;justify-content:space-between;gap:8px}.slider output{color:var(--mint);font-variant-numeric:tabular-nums}.slider input{margin-top:7px}.check{display:flex;align-items:center;gap:9px;color:var(--text);margin:-2px 0 14px}.check input{width:auto}.hint{color:var(--muted);font-size:11px;margin:-7px 0 14px}select,input,textarea{width:100%;color:var(--text);background:#07120f;border:1px solid var(--line);border-radius:9px;padding:10px 12px;outline:none}input[type=range]{padding:0;border:0;background:transparent;accent-color:var(--mint)}select:focus,input:focus,textarea:focus{border-color:var(--mint);box-shadow:0 0 0 3px #78f7bf18}textarea{height:230px;resize:vertical;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.actions,.lab-toolbar{display:flex;gap:9px;flex-wrap:wrap}.btn{border:1px solid var(--line);background:#173328;color:var(--text);border-radius:10px;padding:10px 14px;cursor:pointer}.btn:hover{border-color:var(--mint)}.btn.primary{background:var(--lime);border-color:var(--lime);color:#10200d;font-weight:800}.btn.danger{color:var(--danger)}.warning{margin-top:12px;color:#e4cf8f;font-size:12px}.lab-panel{grid-column:1/-1}.lab-layout{display:grid;grid-template-columns:minmax(250px,.45fr) minmax(360px,1fr);gap:18px}.lab-controls{padding:2px}.lab-stage{display:grid;place-items:center;align-content:start;min-width:0}.lab-canvas-wrap{width:100%;padding:12px;border:1px solid var(--line);border-radius:14px;background:#030806;box-shadow:inset 0 0 40px #000}.lab-canvas-wrap canvas{display:block;width:100%;aspect-ratio:1;image-rendering:pixelated;background:#030806;cursor:crosshair;touch-action:none}.lab-canvas-wrap canvas.voxel-mode{cursor:grab}.lab-canvas-wrap canvas.voxel-mode:active{cursor:grabbing}.lab-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:12px 0}.lab-stat{padding:9px;border:1px solid var(--line);border-radius:9px;background:#091713;text-align:center}.lab-stat b{display:block;color:var(--mint);font-size:16px}.lab-stat small{color:var(--muted)}.lab-hidden{display:none!important}.lab-help{margin-top:10px;color:var(--muted);font-size:12px;text-align:center}.run-tools{display:flex;gap:10px;padding:14px 18px;border-bottom:1px solid var(--line)}.run-tools input{max-width:260px}.run-list{display:grid;gap:12px;padding:18px;max-height:calc(100vh - 230px);overflow:auto}.run-card{border:1px solid var(--line);border-radius:14px;background:#0a1713;padding:15px;cursor:pointer}.run-card:hover,.run-card.active{border-color:#4c8e76;background:#10251e}.run-card h3{margin:0 0 5px;font-size:15px}.run-card small{color:var(--muted)}.job{margin:0 0 12px;border:1px solid var(--line);border-radius:12px;padding:12px;background:#08130f}.job-row{display:flex;justify-content:space-between;gap:12px}.live-view{margin:10px 0 0}.live-view img{display:block;width:100%;height:220px;object-fit:contain;background:#030806}.live-view figcaption{padding:6px 0;color:var(--muted);font-size:11px}.status{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--mint)}.status.failed{color:var(--danger)}pre{white-space:pre-wrap;word-break:break-word;color:#a9c9bb;background:#050c0a;border-radius:9px;padding:10px;max-height:180px;overflow:auto;font-size:11px}.results{grid-column:1/-1;margin-top:2px}.result-head{display:flex;justify-content:space-between;align-items:center}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.media{margin:0;border:1px solid var(--line);border-radius:13px;overflow:hidden;background:#06100d}.media img{width:100%;height:260px;display:block;object-fit:contain;background:repeating-conic-gradient(#0c1d17 0 25%,#091712 0 50%) 50%/18px 18px}.media figcaption{padding:9px 12px;color:var(--muted)}.table-wrap{overflow:auto;margin-top:16px;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #1c372e;white-space:nowrap}th{color:var(--mint);background:#0d201a;position:sticky;top:0}.empty{color:var(--muted);padding:36px;text-align:center}.toast{position:fixed;right:24px;bottom:24px;background:#173328;border:1px solid var(--mint);padding:12px 16px;border-radius:10px;box-shadow:var(--shadow);display:none}.toast.error{border-color:var(--danger)}@media(max-width:900px){.shell{display:block}.side{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}.nav{display:flex;overflow:auto}.nav button{min-width:max-content}.side-foot{display:none}.main{padding:24px 16px}.grid{grid-template-columns:1fr}.lab-layout{grid-template-columns:1fr}.run-list{max-height:460px}.top{align-items:flex-start;flex-direction:column}.counts{flex-wrap:wrap}}@media(max-width:520px){.option-grid,.quick-grid{grid-template-columns:1fr}.lab-stats{grid-template-columns:1fr 1fr}}
</style></head><body><div class="shell"><aside class="side"><div class="brand"><div class="mark">⬡</div><div><b>MorphoVoxel</b><small>Local growth lab</small></div></div><nav class="nav" aria-label="Environment filters"><button class="active" data-kind="all">Overview</button><button id="labNav">View Checkpoints</button><button data-kind="2d">2D growth</button><button data-kind="3d">3D voxels</button><button data-kind="conditional">Genome lab</button><button data-kind="regeneration">Regeneration</button><button data-kind="ecology">Ecology</button><button data-kind="experiments">Full experiments</button></nav><div class="side-foot"><span class="dot"></span>Local server</div></aside><main class="main"><header class="top"><div><div class="eyebrow">Neural cellular automata</div><h1>Experiment control room</h1></div><div class="counts"><span class="pill" id="deviceStatus">Detecting hardware…</span><span class="pill" id="runCount">0 runs</span><span class="pill" id="jobCount">0 active</span></div></header><div class="grid"><section class="panel"><div class="panel-head"><h2>Configure environment</h2><span class="status" id="kindLabel">READY</span></div><div class="panel-body"><label class="field"><span>Preset</span><select id="configSelect" aria-label="Experiment preset"></select></label><label class="field"><span>Run name (optional)</span><input id="runName" maxlength="64" placeholder="Generated automatically"></label><div class="option-grid"><label class="field"><span>Compute device</span><select id="device" aria-label="Compute device"><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label><div class="field"><span>Training preview</span><strong>End state every 10 iterations</strong></div></div><label class="check" for="livePreview"><input id="livePreview" type="checkbox" checked> Show live organism preview</label><div class="hint" id="hardwareNote">Auto uses CUDA when PyTorch can access it. The live image is overwritten; the final GIF keeps sampled frames.</div><section class="quick-controls" aria-labelledby="quickTitle"><div class="quick-title" id="quickTitle">Quick settings <small>Updates YAML below</small></div><div class="quick-grid" id="quickControls"></div></section><label class="field"><span>Advanced YAML parameters</span><textarea id="editor" spellcheck="false" aria-label="YAML configuration"></textarea></label><div class="actions"><button class="btn" id="reset">Reset</button><button class="btn primary" id="launch">Launch run</button></div><div class="warning" id="warning"></div></div></section><section class="panel"><div class="panel-head"><h2>Runs & jobs</h2><button class="btn" id="refresh">Refresh</button></div><div class="run-tools"><input id="search" placeholder="Filter runs…" aria-label="Filter runs"></div><div class="run-list" id="runList"></div></section><section class="panel lab-panel" id="designLab"><div class="panel-head"><div><div class="eyebrow">INTERACTIVE INFERENCE</div><h2>View Checkpoints</h2></div><span class="status" id="labStatus" aria-live="polite">NO RUN</span></div><div class="panel-body"><div class="lab-layout"><div class="lab-controls"><label class="field"><span>Completed checkpoint run</span><select id="labRun" aria-label="Completed checkpoint run"></select></label><div class="option-grid"><label class="field"><span>Compute device</span><select id="labDevice"><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label><label class="field"><span>Mouse tool</span><select id="labTool"><option value="erase">Eraser</option><option value="seed">Seed</option></select></label></div><button class="btn primary" id="labLoad">Open checkpoint</button><div class="lab-stats" id="labStats"><div class="lab-stat"><b>—</b><small>step</small></div><div class="lab-stat"><b>—</b><small>device steps/s</small></div><div class="lab-stat"><b>—</b><small>cells</small></div><div class="lab-stat"><b>—</b><small>device</small></div></div><div class="lab-toolbar"><button class="btn primary" id="labPlay" disabled>Play</button><button class="btn" id="labStep" disabled>Step</button><button class="btn" id="labReset" disabled>Reset seed</button><button class="btn danger" id="labClear" disabled>Clear</button></div><label class="field slider"><span class="slider-head"><span>Playback speed</span><output id="labSpeedValue">1× · measuring</output></span><input id="labSpeed" type="range" min="0" max="10" step="1" value="6" aria-label="Playback speed"></label><div class="hint">1× uses one-fifth of measured device throughput; 5× removes the throttle.</div><label class="field slider"><span class="slider-head"><span>Eraser radius</span><output id="labRadiusValue">2</output></span><input id="labRadius" type="range" min="1" max="12" value="2"></label><label class="field lab-hidden" id="labGenomeField"><span>Genome</span><select id="labGenome"></select></label><div class="option-grid lab-hidden" id="lab3dControls"><label class="field"><span>3D view</span><select id="labView"><option value="voxels">3D voxels</option><option value="slice">Z slice</option><option value="top">Top projection</option><option value="front">Front projection</option><option value="side">Side projection</option></select></label><label class="field slider"><span class="slider-head"><span>Editing layer</span><output id="labLayerValue">0</output></span><input id="labLayer" type="range" min="0" max="0" value="0"></label></div></div><div class="lab-stage"><div class="lab-canvas-wrap"><canvas id="labCanvas" width="32" height="32" tabindex="0" aria-label="Interactive neural cellular automaton world"></canvas></div><div class="lab-help">Double-click to place a seed. Choose Eraser, then click or drag to remove every channel in that region. For 3D projections, edits occur on the selected layer.</div></div></div></div></section><section class="panel results"><div class="panel-head result-head"><div><div class="eyebrow" id="resultKind">RESULTS</div><h2 id="resultTitle">Select a run</h2></div><span class="pill" id="resultDate">—</span></div><div class="panel-body" id="resultBody"><div class="empty">Launch an environment or select an existing run.</div></div></section></div></main></div><div class="toast" role="status" aria-live="polite" id="toast"></div><script>
let TOKEN='__TOKEN__';let state={configs:[],runs:[],jobs:[],lab:null},kind='all',selectedRun=null,lab=null,labPlaying=false,labBusy=false,labLoading=false,labEditing=false,labMoved=false,labPlayGeneration=0,labLoopTimer=null,labDeviceRate=0,labClickTimer=null,labVoxelData=null,labYaw=-.7,labPitch=.55,labZoom=1,labOrbiting=false,labLastX=0,labLastY=0;const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('#labRun').closest('.field').insertAdjacentHTML('afterend','<label class="field"><span>Display</span><select id="labDisplay"><option value="organism">Organism</option><option value="target">Training target</option></select></label>');
async function api(url,options={},retry=true){const r=await fetch(url,options),data=await r.json();if(retry&&r.status===403&&data.token&&options.body){TOKEN=data.token;const body=JSON.parse(options.body);body.token=TOKEN;return api(url,{...options,body:JSON.stringify(body)},false)}if(!r.ok)throw new Error(data.error||r.statusText);return data}function toast(message,error=false){const t=$('#toast');t.textContent=message;t.className='toast'+(error?' error':'');t.style.display='block';setTimeout(()=>t.style.display='none',3200)}
function configs(force=false){const select=$('#configSelect'),previous=select.value,items=state.configs.filter(c=>kind==='all'||c.kind===kind);select.innerHTML=items.map(c=>`<option value="${esc(c.name)}">${esc(c.title)} · ${esc(c.subtitle)}</option>`).join('');if(items.some(c=>c.name===previous))select.value=previous;if(force||$('#editor').dataset.config!==select.value)loadConfig()}
const sliderSpecs=[['iterations','Training iterations',1,5000,1],['world_size','World size',8,64,2],['batch_size','Batch size',1,16,1],['model_width','Model width',4,128,4],['hidden_channels','Hidden channels',0,32,1],['fire_rate','Fire rate',.05,1,.05],['growth_steps','Growth steps',1,128,1],['recovery_steps','Recovery steps',1,128,1],['organisms','Organisms',1,8,1],['steps','Simulation steps',1,256,1]];
function yamlNumber(key){const match=$('#editor').value.match(new RegExp(`^(\\s*${key}:\\s*)([-+]?\\d*\\.?\\d+)(\\s*(?:#.*)?)$`,'m'));return match?Number(match[2]):null}
function setYamlNumber(key,value){const pattern=new RegExp(`^(\\s*${key}:\\s*)([-+]?\\d*\\.?\\d+)(\\s*(?:#.*)?)$`,'m');$('#editor').value=$('#editor').value.replace(pattern,(_,before,_old,after)=>before+value+after)}
function renderSliders(){const controls=sliderSpecs.flatMap(([key,label,min,max,step])=>{const value=yamlNumber(key);if(value===null)return[];return[`<label class="slider"><span class="slider-head"><span>${label}</span><output>${value}</output></span><input type="range" data-yaml-key="${key}" min="${Math.min(min,value)}" max="${Math.max(max,value)}" step="${step}" value="${value}" aria-label="${label}"></label>`]}).join('');$('#quickControls').innerHTML=controls||'<span class="hint">Use the YAML editor for this preset.</span>';document.querySelectorAll('[data-yaml-key]').forEach(input=>input.oninput=()=>{const value=Number(input.value);setYamlNumber(input.dataset.yamlKey,value);input.closest('.slider').querySelector('output').value=value})}
const presetNotes={'full_experiment.yaml':'Recommended default: trains the 2D, 3D, genome-conditioned, regeneration, and ecology phases in dependency order. This can run for hours.','phase1_2d.yaml':'Trains one rule to grow one flat 2D target.','phase2_3d.yaml':'Trains one rule to grow one 3D target.','phase3_conditional.yaml':'Trains one shared 3D rule whose one-hot genome selects among four target morphologies.','phase4_regeneration_training.yaml':'Trains the genome-conditioned 3D rule on damaged state-pool samples and creates the checkpoint used by regeneration evaluation.','phase4_regeneration.yaml':'Compares the Phase 3 checkpoint with the damage-trained checkpoint. Run both training presets first.','phase5_ecology.yaml':'Runs two genome-conditioned organisms with light, water, energy, and growth costs. Run Genome lab training first.','ecology_experiments.yaml':'Runs the full ecology scenario matrix. Run Genome lab training first.'};function loadConfig(){const c=state.configs.find(c=>c.name===$('#configSelect').value);if(!c){$('#editor').value='';return}$('#editor').value=c.content;$('#editor').dataset.config=c.name;$('#kindLabel').textContent=c.kind.toUpperCase();const smoke=c.name.startsWith('smoke_'),requested=(c.content.match(/^\s*device:\s*(auto|cpu|cuda)\s*$/m)||[])[1]||'auto';$('#device').value=requested;renderSliders();$('#warning').textContent=smoke?'Smoke presets only validate that the pipeline runs; they use tiny, untrained or barely trained models and are not meaningful scientific results.':presetNotes[c.name]||''}
function liveLabel(l){return [l.phase,l.genome,l.damage,l.severity!=null?`severity ${l.severity}`:null,l.step!=null&&l.total_steps!=null?`step ${l.step}/${l.total_steps}`:null,l.steps_per_second!=null?`${Number(l.steps_per_second).toFixed(1)} steps/s`:null,l.iteration!=null?`iteration ${l.iteration}`:null].filter(Boolean).join(' · ')}
function labPost(path,value={}){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:TOKEN,...value})})}
function syncLabRuns(){const select=$('#labRun'),previous=select.value,ready=state.runs.filter(r=>r.lab_ready);select.innerHTML=ready.map(r=>`<option value="${esc(r.name)}">${esc(r.name)} · ${esc(r.kind)}</option>`).join('');if(ready.some(r=>r.name===previous))select.value=previous;else if(lab&&ready.some(r=>r.name===lab.run))select.value=lab.run;$('#labLoad').disabled=labLoading||!ready.length;if(!ready.length)select.innerHTML='<option>No checkpoint-backed runs</option>'}
function labSource(){return $('#labDisplay').value}function labTargetMode(){return labSource()==='target'}function labView(){return lab?.dimensions===3?$('#labView').value:'plane'}function labVoxelMode(){return lab?.dimensions===3&&labView()==='voxels'}function labLayer(){return lab?.dimensions===3&&!labVoxelMode()?Number($('#labLayer').value):0}
function syncLabLayer(center=false){const readOnly=labVoxelMode()||labTargetMode();$('#labTool').disabled=readOnly;$('#labRadius').disabled=labTargetMode();if(!lab||lab.dimensions!==3)return;const input=$('#labLayer'),voxel=labVoxelMode();input.disabled=voxel;if(voxel){$('#labLayerValue').value='—';return}const count=lab.layers[labView()]||1;input.max=count-1;if(center||Number(input.value)>=count)input.value=Math.floor(count/2);$('#labLayerValue').value=input.value}
const cubeCorners=[[-.48,-.48,-.48],[.48,-.48,-.48],[.48,.48,-.48],[-.48,.48,-.48],[-.48,-.48,.48],[.48,-.48,.48],[.48,.48,.48],[-.48,.48,.48]],cubeFaces=[[[4,5,6,7],[0,0,1]],[[0,3,2,1],[0,0,-1]],[[1,2,6,5],[1,0,0]],[[0,4,7,3],[-1,0,0]],[[3,7,6,2],[0,1,0]],[[0,1,5,4],[0,-1,0]]],voxelHues=[155,78,205,35,330,190];
function rotateVoxel(x,y,z){const cy=Math.cos(labYaw),sy=Math.sin(labYaw),cp=Math.cos(labPitch),sp=Math.sin(labPitch),rx=x*cy-z*sy,rz=x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]}
function renderVoxels(){if(!labVoxelData)return;const canvas=$('#labCanvas'),context=canvas.getContext('2d'),ratio=Math.min(window.devicePixelRatio||1,2),side=Math.max(320,Math.round(canvas.getBoundingClientRect().width*ratio));if(canvas.width!==side||canvas.height!==side){canvas.width=side;canvas.height=side}context.clearRect(0,0,side,side);const[depth,height,width]=labVoxelData.shape,scale=side*.72/Math.max(depth,height,width)*labZoom,faces=[];for(const[z,y,x,alpha,material]of labVoxelData.voxels){const center=[x-(width-1)/2,(depth-1)/2-z,y-(height-1)/2];for(const[indices,normal]of cubeFaces){const rotatedNormal=rotateVoxel(...normal);if(rotatedNormal[2]<=0)continue;const points=indices.map(index=>{const corner=cubeCorners[index],rotated=rotateVoxel(center[0]+corner[0],center[1]+corner[1],center[2]+corner[2]);return[side/2+rotated[0]*scale,side/2-rotated[1]*scale,rotated[2]]});faces.push({points,depth:points.reduce((sum,p)=>sum+p[2],0)/4,alpha,material,light:Math.max(.25,Math.min(.8,.45-rotatedNormal[0]*.12+rotatedNormal[1]*.22+rotatedNormal[2]*.25))})}}faces.sort((a,b)=>a.depth-b.depth);for(const face of faces){context.beginPath();face.points.forEach((point,index)=>index?context.lineTo(point[0],point[1]):context.moveTo(point[0],point[1]));context.closePath();context.fillStyle=`hsla(${voxelHues[Math.trunc(face.material)%voxelHues.length]},70%,${Math.round(face.light*100)}%,${Math.max(.35,Math.min(1,face.alpha))})`;context.fill();context.strokeStyle='rgba(2,8,6,.55)';context.lineWidth=Math.max(1,ratio*.55);context.stroke()}if(!faces.length){context.fillStyle='#8ca89d';context.font=`${14*ratio}px system-ui`;context.textAlign='center';context.fillText('No occupied voxels yet',side/2,side/2)}}
async function drawLab(){if(!lab)return;const canvas=$('#labCanvas'),context=canvas.getContext('2d'),view=labView(),layer=labLayer(),source=labSource(),version=lab.frame_version;canvas.classList.toggle('voxel-mode',view==='voxels');$('.lab-help').textContent=source==='target'?(view==='voxels'?`Training target: ${lab.target_name} · drag to rotate · wheel to zoom.`:`Training target: ${lab.target_name} · reference only; switch to Organism to edit.`):(view==='voxels'?'Drag to rotate · wheel to zoom · switch to a slice or projection to edit.':'Double-click to place a seed. Choose Eraser, then click or drag to remove every channel in that region. For 3D projections, edits occur on the selected layer.');if(view==='voxels'){const data=await api(`/api/lab/voxels?source=${encodeURIComponent(source)}&v=${version}`);if(!lab||lab.frame_version!==version||labView()!=='voxels'||source!==labSource())return;labVoxelData=data;renderVoxels();return}labVoxelData=null;const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Could not render lab frame'));image.src=`/api/lab/frame?view=${encodeURIComponent(view)}&layer=${layer}&source=${encodeURIComponent(source)}&v=${version}`});if(!lab||lab.frame_version!==version||view!==labView()||layer!==labLayer()||source!==labSource())return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;context.imageSmoothingEnabled=false;context.drawImage(image,0,0)}
const labSpeedChoices=[[1/60,'1/60×'],[1/30,'1/30×'],[1/15,'1/15×'],[1/10,'1/10×'],[1/5,'1/5×'],[1/2,'1/2×'],[1,'1×'],[2,'2×'],[3,'3×'],[4,'4×'],[5,'5×']];function labSpeedChoice(){return labSpeedChoices[Number($('#labSpeed').value)]}function syncLabSpeed(){const[speed,label]=labSpeedChoice(),target=labDeviceRate*speed/5;$('#labSpeedValue').value=speed===5?`${label} · max`:labDeviceRate?`${label} · ~${target<10?target.toFixed(1):Math.round(target)} steps/s`:`${label} · measuring`}
function renderLabStats(){if(!lab)return;const target=labTargetMode(),cells=target?lab.target_cells:lab.occupied_cells;$('#labStats').innerHTML=`<div class="lab-stat"><b>${lab.steps}</b><small>step</small></div><div class="lab-stat"><b>${Number(lab.steps_per_second).toFixed(1)}</b><small>device steps/s</small></div><div class="lab-stat"><b>${cells}</b><small>${target?'target cells':'organism cells'}</small></div><div class="lab-stat"><b>${esc(lab.device)}</b><small>device</small></div>`}
async function setLab(meta){const changed=!lab||lab.run!==meta.run||lab.dimensions!==meta.dimensions;if(changed)labDeviceRate=0;lab=meta;if(Number(meta.steps_per_second)>0)labDeviceRate=Number(meta.steps_per_second);syncLabSpeed();$('#labStatus').textContent=`${meta.run} · ${meta.dimensions}D`;['#labPlay','#labStep','#labReset','#labClear'].forEach(id=>$(id).disabled=false);$('#labGenomeField').classList.toggle('lab-hidden',!meta.conditional);$('#lab3dControls').classList.toggle('lab-hidden',meta.dimensions!==3);$('#labDisplay option[value="target"]').textContent=`Training target · ${meta.target_name}`;if(changed){$('#labDisplay').value='organism';$('#labGenome').innerHTML=meta.genomes.map((name,index)=>`<option value="${index}">${esc(name)}</option>`).join('');$('#labGenome').value=meta.genome;$('#labView').value=meta.dimensions===3?'voxels':'slice';labYaw=-.7;labPitch=.55;labZoom=1;syncLabLayer(true)}else{$('#labGenome').value=meta.genome;syncLabLayer()}renderLabStats();await drawLab()}
function pauseLab(){labPlaying=false;labPlayGeneration++;clearTimeout(labLoopTimer);labLoopTimer=null;$('#labPlay').textContent='Play'}
async function openLab(runName=null){if(labLoading)return;labLoading=true;if(runName)$('#labRun').value=runName;pauseLab();syncLabRuns();while(labBusy)await new Promise(resolve=>setTimeout(resolve,10));$('#labStatus').textContent='LOADING';try{const meta=await labPost('/api/lab/load',{run:$('#labRun').value,device:$('#labDevice').value});await setLab(meta);$('#designLab').scrollIntoView({behavior:'smooth',block:'start'})}catch(error){$('#labStatus').textContent='ERROR';toast(error.message,true)}finally{labLoading=false;syncLabRuns()}}
async function labAction(action,extra={},wait=false){if(!lab||(labTargetMode()&&(action==='seed'||action==='erase')))return false;while(wait&&labBusy)await new Promise(resolve=>setTimeout(resolve,10));if(labBusy)return false;labBusy=true;try{await setLab(await labPost('/api/lab/action',{action,...extra}));return true}catch(error){pauseLab();toast(error.message,true);return false}finally{labBusy=false}}
async function labControl(action,extra={}){pauseLab();return labAction(action,extra,true)}
async function labLoop(generation){if(!labPlaying||generation!==labPlayGeneration)return;const[speed]=labSpeedChoice(),steps=speed>=1?Math.round(speed):1,started=performance.now(),advanced=await labAction('advance',{steps});if(!labPlaying||generation!==labPlayGeneration)return;const targetRate=Math.max(labDeviceRate*speed/5,.01),delay=advanced&&speed<5?Math.max(0,steps/targetRate*1000-(performance.now()-started)):advanced?0:10;labLoopTimer=setTimeout(()=>labLoop(generation),delay)}
function toggleLab(){if(labPlaying){pauseLab();return}labPlaying=true;const generation=++labPlayGeneration;$('#labPlay').textContent='Pause';labLoop(generation)}
function labPoint(event){const canvas=$('#labCanvas'),rect=canvas.getBoundingClientRect();return{row:Math.max(0,Math.min(canvas.height-1,Math.floor((event.clientY-rect.top)/rect.height*canvas.height))),column:Math.max(0,Math.min(canvas.width-1,Math.floor((event.clientX-rect.left)/rect.width*canvas.width)))}}
function labEditPayload(event){return{...labPoint(event),view:labView(),layer:labLayer(),radius:Number($('#labRadius').value)}}
async function editLab(payload,forceSeed=false,wait=true){if(!lab||labTargetMode())return;await labAction(forceSeed?'seed':$('#labTool').value,payload,wait)}
$('#labDisplay').onchange=()=>{syncLabLayer();renderLabStats();drawLab()};
function jobs(){const running=state.jobs.filter(j=>j.status==='running');$('#jobCount').textContent=`${running.length} active`;return state.jobs.map(j=>`<article class="job"><div class="job-row"><div><b>${esc(j.run_name)}</b><div class="status ${j.status}">${esc(j.status)}</div></div>${j.status==='running'?`<button class="btn danger" onclick="stopJob('${esc(j.id)}')">Stop</button>`:''}</div>${j.live?`<figure class="live-view"><img src="${esc(j.live.url)}" alt="Live organism state for ${esc(j.run_name)}"><figcaption>${esc(liveLabel(j.live))}</figcaption></figure>`:''}<pre>${esc(j.log||'Waiting for output…')}</pre></article>`).join('')}
function runs(){const q=$('#search').value.toLowerCase();const items=state.runs.filter(r=>(kind==='all'||r.kind===kind)&&r.name.toLowerCase().includes(q));$('#runCount').textContent=`${state.runs.length} runs`;$('#runList').innerHTML=jobs()+(items.length?items.map(r=>`<article class="run-card ${selectedRun===r.name?'active':''}" data-run="${esc(r.name)}"><h3>${esc(r.name)}</h3><small>${esc(r.kind)} · ${r.media.length} visuals · ${r.lab_ready?'lab ready':'artifacts only'}</small></article>`).join(''):'<div class="empty">No matching runs.</div>');document.querySelectorAll('[data-run]').forEach(el=>el.onclick=()=>showRun(el.dataset.run))}
function showRun(name){selectedRun=name;const r=state.runs.find(r=>r.name===name);if(!r)return;$('#resultTitle').textContent=r.name;$('#resultKind').textContent=r.kind.toUpperCase();$('#resultDate').textContent=new Date(r.updated).toLocaleString();let html=r.lab_ready?`<div class="actions" style="margin-bottom:14px"><button class="btn primary" data-open-lab="${esc(r.name)}">Open in View Checkpoints</button></div>`:'';html+=r.media.length?`<div class="gallery">${r.media.map(m=>`<figure class="media"><img src="${m.url}" alt="${esc(m.name)} visualization"><figcaption>${esc(m.name)}</figcaption></figure>`).join('')}</div>`:'<div class="empty">This run has no saved images yet.</div>';for(const metric of r.metrics){html+=`<div class="table-wrap"><table><caption style="padding:10px;text-align:left">${esc(metric.name)}</caption><thead><tr>${metric.columns.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${metric.rows.map(row=>`<tr>${metric.columns.map(c=>`<td>${esc(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}if(r.logs)html+=`<h3>Training log</h3><pre>${esc(r.logs)}</pre>`;$('#resultBody').innerHTML=html;document.querySelectorAll('[data-open-lab]').forEach(button=>button.onclick=()=>openLab(button.dataset.openLab));runs()}
async function refresh(){try{const first=!state.configs.length;state=await api('/api/state');const h=state.hardware||{};$('#deviceStatus').textContent=h.cuda_available?`GPU · ${h.cuda_name}`:`Auto · ${h.auto_device||'CPU'}`;for(const select of [$('#device'),$('#labDevice')])select.querySelector('option[value="cuda"]').disabled=!h.cuda_available;$('#hardwareNote').textContent=(h.cuda_available?`Auto will use ${h.cuda_name}. Tiny smoke runs can still be faster on CPU.`:`Auto is using CPU because this PyTorch ${h.torch_version||''} build cannot access CUDA.`)+' Training previews show the completed rollout every 10 iterations.';if(selectedRun&&!state.runs.some(r=>r.name===selectedRun))selectedRun=null;configs(first);syncLabRuns();runs();if(state.lab&&(!lab||state.lab.run!==lab.run||state.lab.frame_version!==lab.frame_version))setLab(state.lab);if(selectedRun)showRun(selectedRun)}catch(e){toast(e.message,true)}}
async function launch(){try{const job=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:TOKEN,config:$('#configSelect').value,content:$('#editor').value,run_name:$('#runName').value,device:$('#device').value,live_preview:$('#livePreview').checked})});toast(`Started ${job.run_name}`);$('#runName').value='';await refresh()}catch(e){toast(e.message,true)}}async function stopJob(id){try{await api('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:TOKEN,job:id})});toast('Stop requested');await refresh()}catch(e){toast(e.message,true)}}
document.querySelectorAll('.nav button[data-kind]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');kind=b.dataset.kind;configs(true);runs()});$('#labNav').onclick=()=>$('#designLab').scrollIntoView({behavior:'smooth',block:'start'});$('#configSelect').onchange=loadConfig;$('#editor').onchange=renderSliders;$('#reset').onclick=loadConfig;$('#launch').onclick=launch;$('#refresh').onclick=refresh;$('#search').oninput=runs;$('#labLoad').onclick=()=>openLab();$('#labPlay').onclick=toggleLab;$('#labStep').onclick=()=>labControl('advance',{steps:1});$('#labReset').onclick=()=>labControl('reset');$('#labClear').onclick=()=>labControl('clear');$('#labGenome').onchange=()=>labControl('genome',{genome:Number($('#labGenome').value)});$('#labView').onchange=()=>{syncLabLayer(true);drawLab()};$('#labLayer').oninput=()=>{$('#labLayerValue').value=$('#labLayer').value;drawLab()};$('#labSpeed').oninput=()=>{syncLabSpeed();if(labPlaying){const generation=++labPlayGeneration;clearTimeout(labLoopTimer);labLoopTimer=setTimeout(()=>labLoop(generation),labBusy?10:0)}};$('#labRadius').oninput=()=>$('#labRadiusValue').value=$('#labRadius').value;const labCanvas=$('#labCanvas');labCanvas.onpointerdown=event=>{if(event.button!==0||!lab)return;labCanvas.setPointerCapture(event.pointerId);if(labVoxelMode()){labOrbiting=true;labLastX=event.clientX;labLastY=event.clientY;return}labEditing=true;labMoved=false};labCanvas.onpointermove=event=>{if(labOrbiting){labYaw+=(event.clientX-labLastX)*.012;labPitch=Math.max(-1.45,Math.min(1.45,labPitch+(event.clientY-labLastY)*.012));labLastX=event.clientX;labLastY=event.clientY;renderVoxels();return}if(labEditing&&$('#labTool').value==='erase'){labMoved=true;clearTimeout(labClickTimer);editLab(labEditPayload(event),false,false)}};labCanvas.onpointerup=event=>{if(labOrbiting){labOrbiting=false;return}if(!labEditing)return;labEditing=false;const payload=labEditPayload(event);if(labMoved){labAction('erase',payload,true);return}if($('#labTool').value==='seed')editLab(payload);else{clearTimeout(labClickTimer);labClickTimer=setTimeout(()=>editLab(payload),450)}};labCanvas.onpointercancel=()=>{labEditing=false;labOrbiting=false};labCanvas.ondblclick=event=>{if(labVoxelMode())return;event.preventDefault();clearTimeout(labClickTimer);editLab(labEditPayload(event),true)};labCanvas.onwheel=event=>{if(!labVoxelMode())return;event.preventDefault();labZoom=Math.max(.5,Math.min(3,labZoom*Math.exp(-event.deltaY*.001)));renderVoxels()};labCanvas.oncontextmenu=event=>event.preventDefault();window.addEventListener('resize',()=>{if(labVoxelMode())renderVoxels()});refresh();setInterval(()=>{if(state.jobs.some(j=>j.status==='running'))refresh()},2000);
</script></body></html>'''


def create_server(project_root: Path, host: str = "127.0.0.1", port: int = 8765) -> DashboardServer:
    return DashboardServer((host, port), project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local MorphoVoxel dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    server = create_server(project_root, args.host, args.port)
    url = f"http://{args.host}:{server.server_port}"
    print(f"MorphoVoxel dashboard: {url}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: this local control panel has no authentication.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
