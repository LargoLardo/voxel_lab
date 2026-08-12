"""Local browser dashboard for running and inspecting MorphoVoxel experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from .environment import ENVIRONMENT_CHANNELS, ENVIRONMENT_PARAMETERS, EnvironmentSpec
from .genomes import TREE_FAMILIES, TREE_GENE_SPECS, TreeGenome
from .lab import LabSession, find_checkpoint, list_checkpoints
from .random_utils import resolve_device
from .variant_archive import VariantArchive

CONFIGS: dict[str, tuple[str, str, str, str]] = {
    "full_experiment.yaml": ("experiments", "Full tree workbench", "Specialist through ecology, in order", "scripts/run_full_experiment.py"),
    "tree_specialist.yaml": ("specialist", "Tree specialist", "Reliable single-tree baseline", "morphovoxel.train_3d"),
    "tree_family.yaml": ("family", "Tree genome family", "Continuous inherited tree controls", "morphovoxel.train_conditional"),
    "tree_regeneration.yaml": ("regeneration", "Tree regeneration", "Persistent family with damage training", "morphovoxel.train_conditional"),
    "tree_environment.yaml": ("environment", "Environment-conditioned trees", "Light, water, obstacles, and wind", "morphovoxel.train_conditional"),
    "tree_ecology.yaml": ("ecology", "Tree ecology", "Shared genomes growing together", "morphovoxel.run_ecology"),
    "phase1_2d.yaml": ("2d", "2D training", "Full growth", "morphovoxel.train_2d"),
    "phase2_3d.yaml": ("3d", "3D training", "Single morphology", "morphovoxel.train_3d"),
    "phase3_conditional.yaml": ("conditional", "Conditional training", "Shared 3D rule", "morphovoxel.train_conditional"),
    "phase4_regeneration_training.yaml": ("regeneration", "Damage training", "Creates the recovery checkpoint", "morphovoxel.train_conditional"),
    "phase4_regeneration.yaml": ("regeneration", "Regeneration evaluation", "Requires Phase 3 + damage training", "morphovoxel.evaluate_regeneration"),
    "phase5_ecology.yaml": ("ecology", "Ecology run", "Requires Genome lab training", "morphovoxel.run_ecology"),
    "ecology_experiments.yaml": ("ecology", "Ecology matrix", "Requires Genome lab training", "scripts/run_ecology_experiment.py"),
    "smoke_tree_specialist.yaml": ("specialist", "Tree specialist smoke", "Fast pipeline check", "morphovoxel.train_3d"),
    "smoke_tree_family.yaml": ("family", "Tree family smoke", "Fast continuous-genome check", "morphovoxel.train_conditional"),
    "smoke_tree_regeneration.yaml": ("regeneration", "Tree regeneration smoke", "Fast damage-pipeline check", "morphovoxel.train_conditional"),
    "smoke_tree_environment.yaml": ("environment", "Tree environment smoke", "Fast context-pipeline check", "morphovoxel.train_conditional"),
    "smoke_tree_ecology.yaml": ("ecology", "Tree ecology smoke", "Fast model-routing check", "morphovoxel.run_ecology"),
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
    model_kind = str(config.get("model_kind", ""))
    if model_kind == "tree_specialist":
        return "specialist"
    if model_kind == "tree_family":
        if config.get("damage_probability"):
            return "regeneration"
        if config.get("validation_environment_specs") or "environment" in run.name.lower():
            return "environment"
        return "family"
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
        try:
            run_config = load_config(run / "config.yaml") if (run / "config.yaml").is_file() else {}
        except (OSError, ValueError):
            run_config = {}
        try:
            run_metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            run_metadata = {}
        model_kind = str(run_metadata.get("model_kind") or run_config.get("model_kind") or "")
        context_channels = int(run_metadata.get(
            "context_channels",
            run_config.get("context_channels", len(ENVIRONMENT_CHANNELS) if run_config.get("environment_conditioning") else 0),
        ))
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
            "checkpoints": [path.name for path in list_checkpoints(run)],
            "model_kind": model_kind, "context_channels": context_channels,
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


def _variant_view(record: Any) -> dict[str, Any]:
    """Return archive metadata with browser-safe preview routes."""
    value = record.to_dict()
    identifier = quote(record.variant_id, safe="")
    value["preview_urls"] = {
        name.removesuffix("_preview"): f"/api/archive/{identifier}/preview/{name.removesuffix('_preview')}"
        for name in record.artifacts if name.endswith("_preview")
    }
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        required = [
            checkpoint_config.get("checkpoint"),
            checkpoint_config.get("resume"),
            checkpoint_config.get("initialize_from_specialist"),
        ]
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
        self.lab_load_lock = threading.Lock()
        self.validation_lock = threading.Lock()
        self.variant_archive = VariantArchive(self.project_root / "variant_archive")
        self.archive_lock = threading.Lock()


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
            elif route.path == "/api/tree/schema":
                self._json({
                    "genome_schema_version": 1,
                    "families": list(TREE_FAMILIES),
                    "genes": [spec.__dict__ for spec in TREE_GENE_SPECS],
                    "default_genome": TreeGenome().to_dict(),
                    "environment_schema_version": 1,
                    "environment_parameters": list(ENVIRONMENT_PARAMETERS),
                    "environment_channels": list(ENVIRONMENT_CHANNELS),
                    "default_environment": EnvironmentSpec().to_dict(),
                })
            elif route.path == "/api/archive":
                query = parse_qs(route.query)
                optional = lambda name: query.get(name, [None])[0] or None
                raw_score = optional("min_score")
                min_score = float(raw_score) if raw_score is not None else None
                if min_score is not None and (not math.isfinite(min_score) or not 0 <= min_score <= 1):
                    raise ValueError("min_score must be finite and within [0, 1]")
                with self.server.archive_lock:
                    records = self.server.variant_archive.list(
                        family=optional("family"), method=optional("method"),
                        model_kind=optional("model_kind"), min_score=min_score,
                    )
                self._json({"variants": [_variant_view(record) for record in records]})
            elif route.path.startswith("/api/archive/"):
                parts = route.path.removeprefix("/api/archive/").split("/")
                with self.server.archive_lock:
                    record = self.server.variant_archive.load(unquote(parts[0]))
                if len(parts) == 1:
                    self._json(_variant_view(record))
                elif len(parts) == 3 and parts[1] == "preview" and parts[2] in {"target", "final"}:
                    artifact = record.artifacts[f"{parts[2]}_preview"]
                    preview = record.directory / artifact["path"]
                    self._send(preview.read_bytes(), "image/png")
                else:
                    raise FileNotFoundError
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
                checkpoint = payload.get("checkpoint")
                if checkpoint is not None and not isinstance(checkpoint, str):
                    raise ValueError("checkpoint must be a filename")
                with self.server.lab_load_lock:
                    session = LabSession.from_run(run, device, checkpoint)
                with self.server.lab_lock:
                    self.server.lab = session
                    self._json(session.summary())
            elif self.path == "/api/lab/validate":
                def bounded_integer(name: str, default: int, maximum: int) -> int:
                    value = payload.get(name, default)
                    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
                    return value

                steps = bounded_integer("steps", 512, 2048)
                recovery_steps = bounded_integer("recovery_steps", 128, 1024)
                fire_seeds = payload.get("fire_seeds", [1001, 1002])
                if (
                    not isinstance(fire_seeds, list) or not 1 <= len(fire_seeds) <= 8
                    or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31 for seed in fire_seeds)
                ):
                    raise ValueError("fire_seeds must contain 1 to 8 integers from 0 through 2^31-1")
                with self.server.lab_lock:
                    lab = self.server.lab
                if lab is None:
                    raise ValueError("open a checkpoint-backed run in the lab first")
                if not self.server.validation_lock.acquire(blocking=False):
                    raise ValueError("candidate validation is already running")
                try:
                    result = lab.validate_tree_candidate(
                        steps=steps, recovery_steps=recovery_steps, fire_seeds=tuple(fire_seeds),
                    )
                finally:
                    self.server.validation_lock.release()
                self._json(result)
            elif self.path == "/api/archive/save":
                method = str(payload.get("method", "manual"))
                parents = payload.get("parents", [])
                if not isinstance(parents, list) or not all(isinstance(value, str) for value in parents):
                    raise ValueError("parents must be a list of variant ids")
                with self.server.lab_lock:
                    lab = self.server.lab
                    if lab is None:
                        raise ValueError("open a checkpoint-backed run in the lab first")
                    genome = lab.pending_tree_genome or lab.active_tree_genome
                    environment = lab.environment_spec or EnvironmentSpec()
                    bound = lab.last_validation
                    report = lab.last_validation_report
                    if genome is not None and lab.active_tree_genome != genome:
                        raise ValueError("reset or place a seed to apply the staged genome, then grow it before archiving")
                    if lab.steps <= 0:
                        raise ValueError("grow the applied candidate before archiving so its final preview is meaningful")
                    if genome is None or report is None or not report.validated or not report.accepted or not isinstance(bound, dict):
                        raise ValueError("validate this exact candidate successfully before archiving")
                    if (
                        bound.get("checkpoint") != lab.checkpoint_name
                        or bound.get("checkpoint_sha256") != lab.checkpoint_sha256
                        or bound.get("genome") != genome.to_dict()
                        or bound.get("environment") != environment.to_dict()
                    ):
                        raise ValueError("the accepted validation belongs to a different genome, environment, or checkpoint")
                    run = _inside(self.server.project_root / "runs", lab.run_name)
                    checkpoint = find_checkpoint(run, lab.checkpoint_name)
                    if checkpoint is None:
                        raise ValueError("the validated checkpoint is no longer available")
                    _, target, target_materials = lab._target()
                    final_state = lab.state.detach().cpu().clone()
                    target = target.detach().cpu().clone()
                    target_materials = target_materials.detach().cpu().clone()
                    model_identity = {
                        "model_kind": lab.model_kind,
                        "class_name": type(lab.model).__name__,
                        "architecture": {
                            "dimensions": lab.dimensions,
                            "state_channels": lab.layout.channels,
                            "model_width": int(lab.config.get("model_width", 32)),
                            "genome_size": int(getattr(lab.model, "genome_size", 0)),
                            "context_channels": int(getattr(lab.model, "context_channels", 0)),
                        },
                    }
                    novelty_threshold = float(lab.config.get("archive_novelty_threshold", 0.02))
                with self.server.archive_lock:
                    record = self.server.variant_archive.save(
                        genome=genome, environment=environment, checkpoint=checkpoint,
                        model_identity=model_identity, method=method, parents=parents,
                        validation=report, target_occupancy=target,
                        target_materials=target_materials, final_state=final_state,
                        novelty_threshold=novelty_threshold,
                        expected_checkpoint_sha256=lab.checkpoint_sha256,
                    )
                self._json(_variant_view(record), HTTPStatus.CREATED)
            elif self.path == "/api/archive/load":
                variant_id = payload.get("variant_id")
                if not isinstance(variant_id, str):
                    raise ValueError("variant_id must be a string")
                with self.server.archive_lock:
                    record = self.server.variant_archive.load(variant_id)
                with self.server.lab_lock:
                    lab = self.server.lab
                    if lab is None or lab.model_kind != "tree_family":
                        raise ValueError("open a compatible tree-family checkpoint before loading a variant")
                    if getattr(lab.model, "context_channels", 0) != len(ENVIRONMENT_CHANNELS):
                        raise ValueError("the open checkpoint cannot receive this variant's environment")
                    checkpoint_name = lab.checkpoint_name
                    run = _inside(self.server.project_root / "runs", lab.run_name)
                    checkpoint = find_checkpoint(run, checkpoint_name)
                if checkpoint is None:
                    raise ValueError("the open checkpoint is no longer available")
                current_sha256 = _sha256_file(checkpoint)
                archived_sha256 = str(record.checkpoint_identity.get("sha256", ""))
                warning = None if current_sha256 == archived_sha256 else (
                    "Checkpoint mismatch: the genome and environment were loaded for cross-checkpoint experimentation, "
                    "not exact replay. Revalidate this candidate before archiving it again."
                )
                with self.server.lab_lock:
                    if self.server.lab is not lab or lab.checkpoint_name != checkpoint_name:
                        raise ValueError("the open lab changed while loading the variant; try again")
                    lab.set_tree_genome(record.genome.to_dict())
                    lab.set_environment(record.environment.to_dict())
                    lab.last_validation = None
                    lab.last_validation_report = None
                    summary = lab.summary()
                self._json({"variant": _variant_view(record), "lab": summary, "warning": warning})
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
                    elif action == "tree_genome":
                        value = payload.get("genome")
                        live_remodel = payload.get("live_remodel", False)
                        if not isinstance(value, dict) or not isinstance(live_remodel, bool):
                            raise ValueError("genome must be an object and live_remodel a boolean")
                        result = lab.set_tree_genome(value, live_remodel=live_remodel)
                    elif action in {"tree_random", "tree_mutate"}:
                        locked = payload.get("locked", [])
                        if not isinstance(locked, list) or not all(isinstance(name, str) for name in locked):
                            raise ValueError("locked must be a list of gene names")
                        seed = integer("seed")
                        if action == "tree_random":
                            result = lab.randomize_tree_genome(seed, locked=locked)
                        else:
                            strength = payload.get("strength")
                            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                                raise ValueError("strength must be a number")
                            result = lab.mutate_tree_genome(float(strength), seed, locked=locked)
                    elif action == "tree_interpolate":
                        left, right, amount = payload.get("left"), payload.get("right"), payload.get("amount")
                        if not isinstance(left, dict) or not isinstance(right, dict) or isinstance(amount, bool) or not isinstance(amount, (int, float)):
                            raise ValueError("left/right must be genomes and amount must be a number")
                        result = lab.interpolate_tree_genome(left, right, float(amount))
                    elif action == "environment":
                        value = payload.get("environment")
                        if not isinstance(value, dict):
                            raise ValueError("environment must be an object")
                        result = lab.set_environment(value)
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
:root{color-scheme:dark;--bg:#07110f;--panel:#0d1c18;--panel2:#11251f;--line:#24443a;--mint:#78f7bf;--lime:#c6ff67;--text:#eaf9f2;--muted:#8ca89d;--danger:#ff7b7b;--shadow:0 24px 70px #0008}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 70% -10%,#164634 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}button,input,select,textarea{font:inherit}.shell{display:grid;grid-template-columns:240px 1fr;min-height:100vh}.side{position:sticky;top:0;height:100vh;padding:26px 18px;border-right:1px solid var(--line);background:#081612e8;backdrop-filter:blur(16px)}.brand{display:flex;gap:11px;align-items:center;margin:0 8px 28px}.mark{width:36px;height:36px;border:1px solid var(--mint);border-radius:11px;box-shadow:inset 0 0 18px #78f7bf33;display:grid;place-items:center;color:var(--mint)}.brand b{font-size:16px;letter-spacing:.04em}.brand small{display:block;color:var(--muted)}.nav button{width:100%;border:0;background:transparent;color:var(--muted);padding:11px 12px;margin:2px 0;border-radius:10px;text-align:left;cursor:pointer}.nav button:hover,.nav button.active{background:#18372c;color:var(--text)}.nav button.active{box-shadow:inset 3px 0 var(--mint)}.side-foot{position:absolute;bottom:22px;left:26px;color:var(--muted);font-size:12px}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--mint);box-shadow:0 0 10px var(--mint);margin-right:7px}.main{padding:32px 38px 60px;min-width:0}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:26px}.eyebrow{color:var(--mint);text-transform:uppercase;letter-spacing:.16em;font-size:11px}.top h1{font-size:clamp(28px,4vw,48px);line-height:1;margin:7px 0 0;letter-spacing:-.045em}.counts{display:flex;gap:9px}.pill{border:1px solid var(--line);background:#11231e;padding:8px 12px;border-radius:99px;color:var(--muted)}.grid{display:grid;grid-template-columns:minmax(310px,.75fr) minmax(420px,1.25fr);gap:18px}.panel{background:linear-gradient(145deg,#10231d,#0b1915);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden}.panel-head{display:flex;align-items:center;justify-content:space-between;padding:18px 20px;border-bottom:1px solid var(--line)}.panel-head h2{font-size:15px;margin:0}.panel-body{padding:18px}.field{display:block;color:var(--muted);font-size:12px;margin-bottom:14px}.field span{display:block;margin-bottom:6px}.option-grid,.quick-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.quick-controls{margin:2px 0 16px;padding:13px;border:1px solid var(--line);border-radius:12px;background:#091713}.quick-title{display:flex;justify-content:space-between;margin-bottom:10px;color:var(--text);font-weight:700}.quick-title small{color:var(--muted);font-weight:400}.slider{color:var(--muted);font-size:11px}.slider-head{display:flex;justify-content:space-between;gap:8px}.slider output{color:var(--mint);font-variant-numeric:tabular-nums}.slider input{margin-top:7px}.check{display:flex;align-items:center;gap:9px;color:var(--text);margin:-2px 0 14px}.check input{width:auto}.hint{color:var(--muted);font-size:11px;margin:-7px 0 14px}select,input,textarea{width:100%;color:var(--text);background:#07120f;border:1px solid var(--line);border-radius:9px;padding:10px 12px;outline:none}input[type=range]{padding:0;border:0;background:transparent;accent-color:var(--mint)}select:focus,input:focus,textarea:focus{border-color:var(--mint);box-shadow:0 0 0 3px #78f7bf18}textarea{height:230px;resize:vertical;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.actions,.lab-toolbar,.sub-actions{display:flex;gap:9px;flex-wrap:wrap}.btn{border:1px solid var(--line);background:#173328;color:var(--text);border-radius:10px;padding:10px 14px;cursor:pointer}.btn:hover{border-color:var(--mint)}.btn.primary{background:var(--lime);border-color:var(--lime);color:#10200d;font-weight:800}.btn.danger{color:var(--danger)}.btn:disabled{cursor:not-allowed;opacity:.45}.warning{margin-top:12px;color:#e4cf8f;font-size:12px}.lab-panel{grid-column:1/-1}.lab-layout{display:grid;grid-template-columns:minmax(330px,.65fr) minmax(360px,1fr);gap:18px}.lab-controls{padding:2px}.lab-stage{display:grid;place-items:center;align-content:start;min-width:0;position:sticky;top:20px}.lab-canvas-wrap{width:100%;padding:12px;border:1px solid var(--line);border-radius:14px;background:#030806;box-shadow:inset 0 0 40px #000}.lab-canvas-wrap canvas{display:block;width:100%;aspect-ratio:1;image-rendering:pixelated;background:#030806;cursor:crosshair;touch-action:none}.lab-canvas-wrap canvas.voxel-mode{cursor:grab}.lab-canvas-wrap canvas.voxel-mode:active{cursor:grabbing}.lab-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:12px 0}.lab-stat{padding:9px;border:1px solid var(--line);border-radius:9px;background:#091713;text-align:center}.lab-stat b{display:block;color:var(--mint);font-size:16px}.lab-stat small{color:var(--muted)}.lab-hidden{display:none!important}.lab-section{margin:14px 0;border:1px solid var(--line);border-radius:12px;background:#091713}.lab-section>summary{cursor:pointer;padding:12px 13px;font-weight:750;color:var(--text)}.lab-section-body{padding:0 13px 13px}.lab-note{padding:9px 11px;margin-bottom:12px;border-radius:9px;background:#10271f;color:var(--muted);font-size:11px}.lab-note.pending{color:#ffe39a;border:1px solid #7f6930}.lab-note.active{color:var(--mint);border:1px solid #2f765c}.model-summary{display:flex;justify-content:space-between;gap:8px;align-items:center;padding:9px 11px;margin:10px 0;border:1px solid var(--line);border-radius:9px;color:var(--muted)}.model-summary b{color:var(--text)}.gene-grid,.environment-grid{display:grid;gap:9px}.gene-row{display:grid;grid-template-columns:minmax(105px,.8fr) minmax(110px,1fr) 78px 38px;gap:8px;align-items:center}.gene-row>span{font-size:11px;color:var(--muted)}.gene-row input{padding:7px 8px}.gene-lock{display:grid;place-items:center;margin:0}.gene-lock input{width:auto}.compact-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.compact-grid .field{margin-bottom:9px}.section-label{margin:10px 0 7px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.slot-state{font-size:11px;color:var(--muted);margin:7px 0}.json-file{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}.lab-help{margin-top:10px;color:var(--muted);font-size:12px;text-align:center}.run-tools{display:flex;gap:10px;padding:14px 18px;border-bottom:1px solid var(--line)}.run-tools input{max-width:260px}.run-list{display:grid;gap:12px;padding:18px;max-height:calc(100vh - 230px);overflow:auto}.run-card{border:1px solid var(--line);border-radius:14px;background:#0a1713;padding:15px;cursor:pointer}.run-card:hover,.run-card.active{border-color:#4c8e76;background:#10251e}.run-card h3{margin:0 0 5px;font-size:15px}.run-card small{color:var(--muted)}.job{margin:0 0 12px;border:1px solid var(--line);border-radius:12px;padding:12px;background:#08130f}.job-row{display:flex;justify-content:space-between;gap:12px}.live-view{margin:10px 0 0}.live-view img{display:block;width:100%;height:220px;object-fit:contain;background:#030806}.live-view figcaption{padding:6px 0;color:var(--muted);font-size:11px}.status{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--mint)}.status.failed{color:var(--danger)}pre{white-space:pre-wrap;word-break:break-word;color:#a9c9bb;background:#050c0a;border-radius:9px;padding:10px;max-height:180px;overflow:auto;font-size:11px}.results{grid-column:1/-1;margin-top:2px}.result-head{display:flex;justify-content:space-between;align-items:center}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.media{margin:0;border:1px solid var(--line);border-radius:13px;overflow:hidden;background:#06100d}.media img{width:100%;height:260px;display:block;object-fit:contain;background:repeating-conic-gradient(#0c1d17 0 25%,#091712 0 50%) 50%/18px 18px}.media figcaption{padding:9px 12px;color:var(--muted)}.table-wrap{overflow:auto;margin-top:16px;border:1px solid var(--line);border-radius:12px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid #1c372e;white-space:nowrap}th{color:var(--mint);background:#0d201a;position:sticky;top:0}.empty{color:var(--muted);padding:36px;text-align:center}.toast{position:fixed;right:24px;bottom:24px;background:#173328;border:1px solid var(--mint);padding:12px 16px;border-radius:10px;box-shadow:var(--shadow);display:none}.toast.error{border-color:var(--danger)}@media(max-width:900px){.shell{display:block}.side{position:relative;height:auto;border-right:0;border-bottom:1px solid var(--line)}.nav{display:flex;overflow:auto}.nav button{min-width:max-content}.side-foot{display:none}.main{padding:24px 16px}.grid{grid-template-columns:1fr}.lab-layout{grid-template-columns:1fr}.lab-stage{position:static}.run-list{max-height:460px}.top{align-items:flex-start;flex-direction:column}.counts{flex-wrap:wrap}}@media(max-width:520px){.option-grid,.quick-grid,.compact-grid{grid-template-columns:1fr}.lab-stats{grid-template-columns:1fr 1fr}.gene-row{grid-template-columns:1fr 70px 34px}.gene-row>span{grid-column:1/-1}}
</style></head><body><div class="shell"><aside class="side"><div class="brand"><div class="mark">⬡</div><div><b>MorphoVoxel</b><small>Local growth lab</small></div></div><nav class="nav" aria-label="Environment filters"><button class="active" data-kind="all">Overview</button><button id="labNav">View Checkpoints</button><button data-kind="2d">2D growth</button><button data-kind="3d">3D voxels</button><button data-kind="conditional">Genome lab</button><button data-kind="regeneration">Regeneration</button><button data-kind="ecology">Ecology</button><button data-kind="experiments">Full experiments</button></nav><div class="side-foot"><span class="dot"></span>Local server</div></aside><main class="main"><header class="top"><div><div class="eyebrow">Neural cellular automata</div><h1>Experiment control room</h1></div><div class="counts"><span class="pill" id="deviceStatus">Detecting hardware…</span><span class="pill" id="runCount">0 runs</span><span class="pill" id="jobCount">0 active</span></div></header><div class="grid"><section class="panel"><div class="panel-head"><h2>Configure environment</h2><span class="status" id="kindLabel">READY</span></div><div class="panel-body"><label class="field"><span>Preset</span><select id="configSelect" aria-label="Experiment preset"></select></label><label class="field"><span>Run name (optional)</span><input id="runName" maxlength="64" placeholder="Generated automatically"></label><label class="field"><span>Compute device</span><select id="device" aria-label="Compute device"><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label><label class="check" for="livePreview"><input id="livePreview" type="checkbox" checked> Show live organism preview</label><section class="quick-controls" aria-labelledby="quickTitle"><div class="quick-title" id="quickTitle">Quick settings <small>Updates YAML below</small></div><div class="quick-grid" id="quickControls"></div></section><label class="field"><span>Advanced YAML parameters</span><textarea id="editor" spellcheck="false" aria-label="YAML configuration"></textarea></label><div class="actions"><button class="btn" id="reset">Reset</button><button class="btn primary" id="launch">Launch run</button></div><div class="warning" id="warning"></div></div></section><section class="panel"><div class="panel-head"><h2>Runs & jobs</h2><button class="btn" id="refresh">Refresh</button></div><div class="run-tools"><input id="search" placeholder="Filter runs…" aria-label="Filter runs"></div><div class="run-list" id="runList"></div></section><section class="panel lab-panel" id="designLab"><div class="panel-head"><div><div class="eyebrow">INTERACTIVE INFERENCE</div><h2>View Checkpoints</h2></div><span class="status" id="labStatus" aria-live="polite">NO RUN</span></div><div class="panel-body"><div class="lab-layout"><div class="lab-controls"><label class="field"><span>Completed checkpoint run</span><select id="labRun" aria-label="Completed checkpoint run"></select></label><div class="option-grid"><label class="field"><span>Compute device</span><select id="labDevice"><option value="auto">Auto</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label><label class="field"><span>Mouse tool</span><select id="labTool"><option value="erase">Eraser</option><option value="seed">Seed</option></select></label></div><button class="btn primary" id="labLoad">Open checkpoint</button><div class="lab-stats" id="labStats"><div class="lab-stat"><b>—</b><small>step</small></div><div class="lab-stat"><b>—</b><small>device steps/s</small></div><div class="lab-stat"><b>—</b><small>cells</small></div><div class="lab-stat"><b>—</b><small>device</small></div></div><div class="lab-toolbar"><button class="btn primary" id="labPlay" disabled>Play</button><button class="btn" id="labStep" disabled>Step</button><button class="btn" id="labReset" disabled>Reset seed</button><button class="btn danger" id="labClear" disabled>Clear</button></div><label class="field slider"><span class="slider-head"><span>Playback speed</span><output id="labSpeedValue">1× · measuring</output></span><input id="labSpeed" type="range" min="0" max="10" step="1" value="6" aria-label="Playback speed"></label><div class="hint">1× uses one-fifth of measured device throughput; 5× removes the throttle.</div><label class="field slider"><span class="slider-head"><span>Eraser radius</span><output id="labRadiusValue">2</output></span><input id="labRadius" type="range" min="1" max="12" value="2"></label><label class="field lab-hidden" id="labGenomeField"><span>Genome</span><select id="labGenome"></select></label><div class="option-grid lab-hidden" id="lab3dControls"><label class="field"><span>3D view</span><select id="labView"><option value="voxels">3D voxels</option><option value="slice">Z slice</option><option value="top">Top projection</option><option value="front">Front projection</option><option value="side">Side projection</option></select></label><label class="field slider"><span class="slider-head"><span>Editing layer</span><output id="labLayerValue">0</output></span><input id="labLayer" type="range" min="0" max="0" value="0"></label></div></div><div class="lab-stage"><div class="lab-canvas-wrap"><canvas id="labCanvas" width="32" height="32" tabindex="0" aria-label="Interactive neural cellular automaton world"></canvas></div><div class="lab-help">Double-click to place a seed. Choose Eraser, then click or drag to remove every channel in that region. For 3D projections, edits occur on the selected layer.</div></div></div></div></section><section class="panel results"><div class="panel-head result-head"><div><div class="eyebrow" id="resultKind">RESULTS</div><h2 id="resultTitle">Select a run</h2></div><span class="pill" id="resultDate">—</span></div><div class="panel-body" id="resultBody"><div class="empty">Launch an environment or select an existing run.</div></div></section></div></main></div><div class="toast" role="status" aria-live="polite" id="toast"></div><script>
let TOKEN='__TOKEN__';let state={configs:[],runs:[],jobs:[],lab:null},kind='all',selectedRun=null,lab=null,labPlaying=false,labBusy=false,labLoading=false,labEditing=false,labMoved=false,labPlayGeneration=0,labLoopTimer=null,labDeviceRate=0,labClickTimer=null,labVoxelData=null,labTargetVoxelData=null,labTargetKey='',labYaw=-.7,labPitch=.55,labZoom=1,labOrbiting=false,labLastX=0,labLastY=0,treeSchema=null,treeDraft=null,treeDirty=false,treeSlotA=null,treeSlotB=null,treeSlotAParent=null,treeSlotBParent=null,environmentDraft=null,environmentDirty=false,labValidating=false,archiveItems=[],loadedVariantId=null;const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('.brand small').textContent='Organism workbench';$('.top h1').textContent='Overview';const trainingPanel=document.querySelector('.grid>.panel');trainingPanel.id='trainingLab';trainingPanel.querySelector('h2').textContent='Train this stage';$('#configSelect').closest('.field').insertAdjacentHTML('afterend',`<label class="field" id="dependencyCheckpointField" hidden><span id="dependencyCheckpointLabel">Previous checkpoint</span><select id="dependencyCheckpoint" aria-label="Previous pipeline checkpoint"></select><small class="hint" id="dependencyCheckpointHint"></small></label>`);$('.nav').setAttribute('aria-label','Workbench pages');$('.nav').innerHTML=`<button class="active" data-page="overview">Overview</button><div class="nav-label">Training pipeline</div><button class="nav-step" data-page="specialist"><b>01</b> Specialist</button><button class="nav-step" data-page="family"><b>02</b> Tree family</button><button class="nav-step" data-page="regeneration"><b>03</b> Regeneration</button><button class="nav-step" data-page="environment"><b>04</b> Environment</button><button class="nav-step" data-page="ecology"><b>05</b> Ecology</button><div class="nav-label">Explore</div><button data-page="checkpoints">View Checkpoints</button><button data-page="archive">Variant Archive</button><div class="nav-label">Other</div><button data-page="legacy">Legacy & advanced</button>`;
$('.nav button[data-page="checkpoints"]').id='labNav';
$('#labRun').closest('.field').insertAdjacentHTML('afterend',`<label class="field"><span>Checkpoint</span><select id="labCheckpoint" aria-label="Checkpoint file"></select></label><div class="model-summary" id="labModelSummary"><span>Model</span><b>Open a checkpoint</b></div><div class="utility-launchers"><button class="btn" id="openTreeGenomeWindow" disabled>Open Tree Genome Lab ↗</button><button class="btn" id="openEnvironmentWindow" disabled>Open Environment Lab ↗</button></div><div class="hint">Genome and environment editors open in focused utility windows and use the checkpoint loaded here.</div>`);
$('#labCanvas').closest('.lab-canvas-wrap').insertAdjacentHTML('afterend',`<aside class="target-preview"><div class="target-preview-head"><span>Training target</span><b id="labTargetCaption">Loads with checkpoint</b></div><canvas id="labTargetCanvas" width="32" height="32" aria-label="Training target preview"></canvas><small>Always shown for comparison · 3D targets use voxels</small></aside>`);
$('#labGenomeField').insertAdjacentHTML('afterend',`<details class="lab-section lab-hidden" id="treeGenomePanel" open><summary>Tree Genome Lab</summary><div class="lab-section-body"><div class="lab-note active" id="treeGenomeStatus">Genome controls load with a tree-family checkpoint.</div><div class="compact-grid"><label class="field"><span>Discrete family</span><select id="treeFamily"></select></label><label class="field"><span>Inherited style seed</span><input id="treeStyleSeed" type="number" min="0" max="2147483647" step="1"></label></div><div class="section-label">Continuous genes · lock any value before randomizing or mutating</div><div class="gene-grid" id="treeGeneControls"></div><label class="check" for="treeLiveRemodel"><input id="treeLiveRemodel" type="checkbox"> Live remodel this mature organism</label><div class="hint">Off by default: Apply Genome stages it for the next Reset seed or placed seed. Environment and fire-mask randomness remain separate.</div><div class="sub-actions"><button class="btn primary" id="treeApply">Apply genome</button><button class="btn" id="treeRandomize">Randomize</button></div><div class="compact-grid"><label class="field slider"><span class="slider-head"><span>Mutation strength</span><output id="treeMutationValue">0.15</output></span><input id="treeMutation" type="range" min="0" max="1" step="0.01" value="0.15"></label><label class="field"><span>Reproducible operation seed</span><input id="treeOperationSeed" type="number" min="0" max="2147483647" step="1" value="42"></label></div><button class="btn" id="treeMutate">Mutate unlocked genes</button><div class="section-label">A/B interpolation (families must match)</div><div class="sub-actions"><button class="btn" id="treeStoreA">Store A</button><button class="btn" id="treeStoreB">Store B</button></div><div class="slot-state" id="treeSlotState">A: empty · B: empty</div><label class="field slider"><span class="slider-head"><span>A → B interpolation</span><output id="treeInterpolationValue">0.50</output></span><input id="treeInterpolation" type="range" min="0" max="1" step="0.01" value="0.5"></label><button class="btn" id="treeInterpolate" disabled>Stage interpolation</button><div class="section-label">Genome JSON</div><div class="sub-actions"><button class="btn" id="treeDownload">Download JSON</button><button class="btn" id="treeLoad">Load JSON</button><input class="json-file" id="treeJsonFile" type="file" accept="application/json,.json"><button class="btn" id="treeValidate">Validate candidate</button><button class="btn" id="treeArchive" disabled title="Variant Archive API is not available yet">Save to archive · unavailable</button></div><div class="lab-note" id="treeValidationStatus">Validation has not run for this candidate.</div></div></details><details class="lab-section lab-hidden" id="environmentPanel"><summary>Environment Lab</summary><div class="lab-section-body"><div class="lab-note">These are changing conditions sensed by cells, not inherited genome values. Apply explicitly to update local fields.</div><div class="environment-grid" id="environmentControls"></div><div class="sub-actions"><button class="btn primary" id="environmentApply">Apply environment</button><button class="btn" id="environmentReset">Restore loaded values</button></div></div></details>`);
$('#treeValidationStatus').insertAdjacentHTML('beforebegin',`<div class="compact-grid"><label class="field"><span>Validation steps</span><input id="treeValidationSteps" type="number" min="1" max="2048" step="1" value="512"></label><label class="field"><span>Recovery steps</span><input id="treeRecoverySteps" type="number" min="1" max="1024" step="1" value="128"></label></div><label class="field"><span>Fire-mask seeds (1–8, comma-separated)</span><input id="treeFireSeeds" value="1001, 1002"></label>`);
$('#treeValidationStatus').insertAdjacentHTML('afterend',`<div class="lab-stats" id="treeMorphologyStats"><div class="lab-stat"><b>—</b><small>persistence</small></div><div class="lab-stat"><b>—</b><small>height</small></div><div class="lab-stat"><b>—</b><small>canopy spread</small></div><div class="lab-stat"><b>—</b><small>occupied volume</small></div></div>`);
$('#treeDownload').closest('.sub-actions').insertAdjacentHTML('beforebegin',`<div class="compact-grid"><label class="field"><span>Archive creation method</span><select id="archiveCreationMethod"><option value="manual">Manual</option><option value="random">Random</option><option value="mutation">Mutation</option><option value="interpolation">Interpolation</option></select></label><label class="field"><span>Parent variant ids (comma-separated)</span><input id="archiveParents" placeholder="Optional except mutation/interpolation"></label></div>`);
$('#treeArchive').textContent='Save to archive';$('#treeArchive').title='Validate this exact candidate first';
$('#labTool option[value="erase"]').textContent='Damage / erase';
document.head.insertAdjacentHTML('beforeend',`<style>.archive-filters{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr)) auto;gap:10px;align-items:end}.archive-filters .field{margin:0}.archive-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:16px}.variant-card{border:1px solid var(--line);border-radius:13px;background:#091713;padding:12px}.variant-card h3{margin:0 0 3px}.variant-meta{color:var(--muted);font-size:11px;margin-bottom:9px}.variant-previews{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:10px}.variant-previews figure{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden}.variant-previews img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#030806;image-rendering:pixelated}.variant-previews figcaption{padding:4px 6px;color:var(--muted);font-size:10px}@media(max-width:900px){.archive-filters{grid-template-columns:1fr 1fr}}@media(max-width:520px){.archive-filters{grid-template-columns:1fr}}</style>`);
document.querySelector('.results').insertAdjacentHTML('beforebegin',`<section class="panel lab-panel" id="variantArchive"><div class="panel-head"><div><div class="eyebrow">VALIDATED ORGANISMS</div><h2>Variant Archive</h2></div><span class="status" id="archiveCount">0 VARIANTS</span></div><div class="panel-body"><div class="archive-filters"><label class="field"><span>Family</span><select id="archiveFamily"><option value="">All families</option></select></label><label class="field"><span>Creation method</span><select id="archiveMethod"><option value="">All methods</option><option value="manual">Manual</option><option value="random">Random</option><option value="mutation">Mutation</option><option value="interpolation">Interpolation</option></select></label><label class="field"><span>Model kind</span><select id="archiveModelKind"><option value="">All models</option><option value="tree_family">Tree family</option><option value="tree_specialist">Tree specialist</option></select></label><label class="field"><span>Minimum persistence score</span><input id="archiveMinScore" type="number" min="0" max="1" step="0.01" placeholder="Any"></label><button class="btn" id="archiveRefresh">Refresh archive</button></div><div class="archive-grid" id="archiveList"><div class="empty">No accepted variants have been archived yet.</div></div></div></section>`);
async function api(url,options={},retry=true){const r=await fetch(url,options),data=await r.json();if(retry&&r.status===403&&data.token&&options.body){TOKEN=data.token;const body=JSON.parse(options.body);body.token=TOKEN;return api(url,{...options,body:JSON.stringify(body)},false)}if(!r.ok)throw new Error(data.error||r.statusText);return data}function toast(message,error=false){const t=$('#toast');t.textContent=message;t.className='toast'+(error?' error':'');t.style.display='block';setTimeout(()=>t.style.display='none',3200)}
const stagePages={specialist:{kind:'specialist',number:'01',title:'Specialist',description:'Train one dependable branching tree. This establishes the basic local growth rule used to initialize the family model.'},family:{kind:'family',number:'02',title:'Tree Family',description:'Expand the specialist into one shared model controlled by continuous inherited tree genomes.'},regeneration:{kind:'regeneration',number:'03',title:'Regeneration',description:'Continue the family checkpoint while damaging mature organisms so they learn to repair their genome-specific form.'},environment:{kind:'environment',number:'04',title:'Environment',description:'Continue the regeneration checkpoint across changing light, water, obstacles, crowding, gravity, and wind.'},ecology:{kind:'ecology',number:'05',title:'Ecology',description:'Place trained shared-genome or specialist organisms together in one resource-limited voxel world.'},legacy:{kind:'legacy',number:'—',title:'Legacy & advanced',description:'Older 2D, 3D, one-hot, evaluation, and full orchestration presets retained for compatibility.'}};
document.head.insertAdjacentHTML('beforeend',`<style>
.side{overflow-y:auto}.side-foot{position:static;margin:28px 8px 0}.nav-label{margin:18px 12px 6px;color:#5f8073;font-size:10px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}.nav-step{display:flex!important;align-items:center;gap:10px}.nav-step b{display:inline-grid;place-items:center;min-width:28px;height:24px;border:1px solid var(--line);border-radius:7px;color:var(--mint);font-size:10px}.workspace-page{display:none}.workspace-page.active{display:block}.training-layout{display:grid;grid-template-columns:minmax(310px,.75fr) minmax(420px,1.25fr);gap:18px}.page-intro{margin-bottom:18px;padding:20px 22px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(135deg,#10251e,#0a1713)}.page-intro h2{margin:4px 0 6px;font-size:24px}.page-intro p{max-width:780px;margin:0;color:var(--muted)}.step-number{color:var(--mint);font-size:11px;font-weight:800;letter-spacing:.15em}.overview-hero{padding:clamp(24px,5vw,56px);border:1px solid var(--line);border-radius:20px;background:radial-gradient(circle at 90% 0,#225b4433,transparent 42%),linear-gradient(145deg,#10231d,#091512)}.overview-hero h2{max-width:760px;margin:8px 0 12px;font-size:clamp(28px,5vw,54px);line-height:1.03;letter-spacing:-.045em}.overview-hero>p{max-width:760px;color:var(--muted);font-size:16px}.pipeline-list{display:grid;gap:10px;margin-top:28px}.pipeline-card{display:grid;grid-template-columns:44px minmax(130px,.35fr) 1fr;gap:16px;align-items:center;padding:14px 16px;border:1px solid var(--line);border-radius:13px;background:#091713}.pipeline-card b:first-child{display:grid;place-items:center;width:36px;height:36px;border-radius:10px;background:#173328;color:var(--mint)}.pipeline-card span{color:var(--muted)}.overview-note{margin-top:18px;color:#c8dbd3}.utility-launchers{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:10px 0}.target-preview{width:100%;padding:11px;border:1px solid var(--line);border-radius:14px;background:#07120f}.target-preview-head{display:flex;justify-content:space-between;gap:8px;margin-bottom:8px;font-size:11px;color:var(--muted)}.target-preview-head b{color:var(--mint);font-weight:600;text-align:right}.target-preview canvas{display:block;width:100%;aspect-ratio:1;background:#030806;image-rendering:pixelated;border-radius:9px}.target-preview small{display:block;margin-top:7px;color:var(--muted);text-align:center}.lab-stage{grid-template-columns:minmax(0,1fr) minmax(170px,.34fr);gap:12px;align-items:start}.lab-help{grid-column:1/-1}.utility-window .side,.utility-window .top{display:none}.utility-window .shell{display:block}.utility-window .main{max-width:780px;margin:auto;padding:20px}.utility-window .workspace-page{display:none!important}.utility-window #utilityPage.active{display:block!important}.utility-window .lab-section{display:block!important;margin:0}.utility-window .lab-section>summary{display:none}.utility-window .lab-section-body{padding:16px}.utility-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:14px}.utility-head h2{margin:2px 0}.utility-head p{margin:4px 0 0;color:var(--muted)}
@media(max-width:1050px){.lab-stage{grid-template-columns:1fr}.target-preview{max-width:300px}.lab-help{grid-column:1}.training-layout{grid-template-columns:1fr}}@media(max-width:700px){.pipeline-card{grid-template-columns:40px 1fr}.pipeline-card span{grid-column:2}.utility-launchers{grid-template-columns:1fr}}
</style>`);
function setupWorkspace(){const workspace=$('.grid'),runsPanel=trainingPanel.nextElementSibling,designLab=$('#designLab'),archive=$('#variantArchive'),results=$('.results'),treePanel=$('#treeGenomePanel'),environmentPanel=$('#environmentPanel');for(const node of [trainingPanel,runsPanel,designLab,archive,results])node.remove();workspace.className='workspace';workspace.innerHTML=`<section class="workspace-page" id="overviewPage"><div class="overview-hero"><div class="eyebrow">How the workbench fits together</div><h2>Grow one reliable tree, then teach a family to vary, recover, and respond.</h2><p>Overview is an information page. Choose a numbered stage in the sidebar when you are ready to configure or launch training.</p><div class="pipeline-list"><div class="pipeline-card"><b>01</b><strong>Specialist</strong><span>Learn one stable default branching tree from scratch.</span></div><div class="pipeline-card"><b>02</b><strong>Tree Family</strong><span>Add continuous inherited genomes and widen the range of tree forms.</span></div><div class="pipeline-card"><b>03</b><strong>Regeneration</strong><span>Damage mature family members and train them to rebuild.</span></div><div class="pipeline-card"><b>04</b><strong>Environment</strong><span>Train local responses to light, water, obstacles, neighbors, and wind.</span></div><div class="pipeline-card"><b>05</b><strong>Ecology</strong><span>Place trained organisms together; shared space alone does not create learned behavior.</span></div></div><div class="overview-note">Checkpoints flow downward: each stage should use the persistence-scored <code>best.pt</code> from the stage above it.</div></div></section><section class="workspace-page" id="trainingPage"><div class="page-intro"><div class="step-number" id="stageNumber"></div><h2 id="stageTitle"></h2><p id="stageDescription"></p></div><div class="training-layout" id="trainingLayout"></div></section><section class="workspace-page" id="checkpointPage"><div class="page-intro"><div class="step-number">EXPLORE</div><h2>View Checkpoints</h2><p>Load a trained checkpoint, watch its organism grow, compare it with its target, seed it, or erase regions to test persistence.</p></div><div id="checkpointSlot"></div></section><section class="workspace-page" id="archivePage"><div class="page-intro"><div class="step-number">VALIDATED VARIANTS</div><h2>Variant Archive</h2><p>Browse only candidates that passed the configured persistence, connectedness, boundedness, and regeneration checks.</p></div><div id="archiveSlot"></div></section><section class="workspace-page" id="utilityPage"><div class="utility-head"><div><div class="eyebrow">CHECKPOINT UTILITY</div><h2 id="utilityTitle">Lab</h2><p id="utilityDescription"></p></div><button class="btn" id="utilityClose">Close window</button></div><div id="utilityContent"></div></section><div id="utilityParking" hidden></div>`;$('#trainingLayout').append(trainingPanel,runsPanel,results);$('#checkpointSlot').append(designLab);$('#archiveSlot').append(archive);$('#utilityParking').append(treePanel,environmentPanel);$('#utilityClose').onclick=()=>window.close();const requested=new URLSearchParams(location.search).get('page')||'overview';showPage(requested)}
function showPage(page){const utility=page==='genome'||page==='environment-lab',definition=stagePages[page];document.body.classList.toggle('utility-window',utility);document.querySelectorAll('.workspace-page').forEach(section=>section.classList.remove('active'));document.querySelectorAll('.nav button').forEach(button=>button.classList.toggle('active',button.dataset.page===page));if(utility){const genome=page==='genome',panel=$(genome?'#treeGenomePanel':'#environmentPanel');$('#utilityTitle').textContent=genome?'Tree Genome Lab':'Environment Lab';$('#utilityDescription').textContent=genome?'Edit inherited genes, randomize, mutate, interpolate, validate, and archive the loaded tree-family checkpoint.':'Edit changing local conditions sensed by the currently loaded checkpoint.';$('#utilityContent').append(panel);panel.open=true;panel.classList.remove('lab-hidden');$('#utilityPage').classList.add('active');document.title=`${$('#utilityTitle').textContent} · MorphoVoxel`;return}if(definition){kind=definition.kind;$('#stageNumber').textContent=`PIPELINE ${definition.number}`;$('#stageTitle').textContent=definition.title;$('#stageDescription').textContent=definition.description;$('#trainingPage').classList.add('active');$('.top h1').textContent=definition.title;configs(true);runs()}else if(page==='checkpoints'){$('#checkpointPage').classList.add('active');$('.top h1').textContent='View Checkpoints'}else if(page==='archive'){$('#archivePage').classList.add('active');$('.top h1').textContent='Variant Archive';refreshVariantArchive()}else{page='overview';$('#overviewPage').classList.add('active');$('.top h1').textContent='Overview';document.querySelector('.nav button[data-page="overview"]').classList.add('active')}const url=new URL(location.href);url.searchParams.set('page',page);history.replaceState(null,'',url)}
function openUtilityWindow(page){if(!lab){toast('Open a checkpoint first',true);return}const popup=window.open(`/?page=${page}`,`morphovoxel-${page}-lab`,'popup=yes,width=700,height=920,resizable=yes,scrollbars=yes');if(!popup)toast('Allow pop-up windows for this local dashboard',true)}
setupWorkspace();
function matchesArea(item){if(kind==='all')return true;if(kind==='legacy')return !item.name.startsWith('tree_')&&!item.name.startsWith('smoke_tree_')&&item.kind!=='specialist'&&item.kind!=='family'&&item.kind!=='environment';const pipelineConfigs={specialist:['tree_specialist.yaml','smoke_tree_specialist.yaml'],family:['tree_family.yaml','smoke_tree_family.yaml'],regeneration:['tree_regeneration.yaml','smoke_tree_regeneration.yaml'],environment:['tree_environment.yaml','smoke_tree_environment.yaml'],ecology:['tree_ecology.yaml','smoke_tree_ecology.yaml']};return Object.hasOwn(item,'content')&&pipelineConfigs[kind]?pipelineConfigs[kind].includes(item.name):item.kind===kind}
function configs(force=false){const select=$('#configSelect'),previous=select.value,items=state.configs.filter(matchesArea);select.innerHTML=items.map(c=>`<option value="${esc(c.name)}">${esc(c.title)} · ${esc(c.subtitle)}</option>`).join('');if(items.some(c=>c.name===previous))select.value=previous;if(force||$('#editor').dataset.config!==select.value)loadConfig()}
const dependencySpecs={
'tree_family.yaml':{key:'initialize_from_specialist',kinds:['specialist'],model:'tree_specialist',context:0,label:'Initialize from specialist checkpoint',hint:'Choose a completed tree specialist to expand into the continuous-genome family.'},
'tree_regeneration.yaml':{key:'resume',kinds:['family'],model:'tree_family',context:12,label:'Resume tree-family checkpoint',hint:'Choose the family model that should learn persistence and damage recovery.'},
'tree_environment.yaml':{key:'resume',kinds:['regeneration'],model:'tree_family',context:12,label:'Resume regeneration checkpoint',hint:'Choose the damage-trained family that should learn environmental adaptation.'},
'tree_ecology.yaml':{key:'checkpoint',kinds:['environment'],model:'tree_family',context:12,label:'Use environment-trained checkpoint',hint:'Choose the context-conditioned family that should populate the ecology world.'}};
function yamlText(key){const match=$('#editor').value.match(new RegExp(`^${key}:\\s*([^#\\n]+)`,'m'));return match?match[1].trim():''}
function setYamlText(key,value){const editor=$('#editor'),pattern=new RegExp(`^${key}:\\s*[^#\\n]*(\\s*(?:#.*)?)$`,'m');editor.value=pattern.test(editor.value)?editor.value.replace(pattern,`${key}: ${value}$1`):editor.value+(`${editor.value.endsWith('\n')?'':'\n'}${key}: ${value}\n`)}
function syncDependencyCheckpoints(){const spec=dependencySpecs[$('#configSelect').value],field=$('#dependencyCheckpointField'),select=$('#dependencyCheckpoint');field.hidden=!spec;if(!spec)return;$('#dependencyCheckpointLabel').textContent=spec.label;const candidates=state.runs.filter(run=>spec.kinds.includes(run.kind)&&run.model_kind===spec.model&&Number(run.context_channels)===spec.context).flatMap(run=>run.checkpoints.map(checkpoint=>({path:`runs/${run.name}/checkpoints/${checkpoint}`,label:`${run.name} · ${checkpoint}`})));const current=yamlText(spec.key);select.innerHTML=candidates.length?candidates.map(item=>`<option value="${esc(item.path)}">${esc(item.label)}</option>`).join(''):'<option value="">No compatible checkpoints found</option>';const chosen=candidates.some(item=>item.path===current)?current:candidates[0]?.path||'';select.value=chosen;if(chosen&&chosen!==current)setYamlText(spec.key,chosen);$('#dependencyCheckpointHint').textContent=candidates.length?spec.hint:`No compatible ${spec.kinds.join(' or ')} checkpoint exists yet. Complete the preceding pipeline stage first.`}
const sliderSpecs=[['iterations','Training iterations',1,5000,1],['world_size','World size',8,64,2],['batch_size','Batch size',1,16,1],['model_width','Model width',4,128,4],['hidden_channels','Hidden channels',0,32,1],['fire_rate','Fire rate',.05,1,.05],['growth_steps','Growth steps',1,128,1],['recovery_steps','Recovery steps',1,128,1],['organisms','Organisms',1,8,1],['steps','Simulation steps',1,256,1]];
function yamlNumber(key){const match=$('#editor').value.match(new RegExp(`^(\\s*${key}:\\s*)([-+]?\\d*\\.?\\d+)(\\s*(?:#.*)?)$`,'m'));return match?Number(match[2]):null}
function setYamlNumber(key,value){const pattern=new RegExp(`^(\\s*${key}:\\s*)([-+]?\\d*\\.?\\d+)(\\s*(?:#.*)?)$`,'m');$('#editor').value=$('#editor').value.replace(pattern,(_,before,_old,after)=>before+value+after)}
function renderSliders(){const controls=sliderSpecs.flatMap(([key,label,min,max,step])=>{const value=yamlNumber(key);if(value===null)return[];return[`<label class="slider"><span class="slider-head"><span>${label}</span><output>${value}</output></span><input type="range" data-yaml-key="${key}" min="${Math.min(min,value)}" max="${Math.max(max,value)}" step="${step}" value="${value}" aria-label="${label}"></label>`]}).join('');$('#quickControls').innerHTML=controls||'<span class="hint">Use the YAML editor for this preset.</span>';document.querySelectorAll('[data-yaml-key]').forEach(input=>input.oninput=()=>{const value=Number(input.value);setYamlNumber(input.dataset.yamlKey,value);input.closest('.slider').querySelector('output').value=value})}
const presetNotes={'full_experiment.yaml':'Recommended default: trains the 2D, 3D, genome-conditioned, regeneration, and ecology phases in dependency order. This can run for hours.','phase1_2d.yaml':'Trains one rule to grow one flat 2D target.','phase2_3d.yaml':'Trains one rule to grow one 3D target.','phase3_conditional.yaml':'Trains one shared 3D rule whose one-hot genome selects among four target morphologies.','phase4_regeneration_training.yaml':'Trains the genome-conditioned 3D rule on damaged state-pool samples and creates the checkpoint used by regeneration evaluation.','phase4_regeneration.yaml':'Compares the Phase 3 checkpoint with the damage-trained checkpoint. Run both training presets first.','phase5_ecology.yaml':'Runs two genome-conditioned organisms with light, water, energy, and growth costs. Run Genome lab training first.','ecology_experiments.yaml':'Runs the full ecology scenario matrix. Run Genome lab training first.'};function loadConfig(){const c=state.configs.find(c=>c.name===$('#configSelect').value);if(!c){$('#editor').value='';return}$('#editor').value=c.content;$('#editor').dataset.config=c.name;$('#kindLabel').textContent=c.kind.toUpperCase();const smoke=c.name.startsWith('smoke_'),requested=(c.content.match(/^\s*device:\s*(auto|cpu|cuda)\s*$/m)||[])[1]||'auto';$('#device').value=requested;renderSliders();syncDependencyCheckpoints();$('#warning').textContent=smoke?'Smoke presets only validate that the pipeline runs; they use tiny, untrained or barely trained models and are not meaningful scientific results.':presetNotes[c.name]||''}
function liveLabel(l){return [l.phase,l.genome,l.damage,l.severity!=null?`severity ${l.severity}`:null,l.step!=null&&l.total_steps!=null?`step ${l.step}/${l.total_steps}`:null,l.steps_per_second!=null?`${Number(l.steps_per_second).toFixed(1)} steps/s`:null,l.iteration!=null?`iteration ${l.iteration}`:null].filter(Boolean).join(' · ')}
function labPost(path,value={}){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:TOKEN,...value})})}
function syncLabCheckpoints(){const select=$('#labCheckpoint'),run=state.runs.find(item=>item.name===$('#labRun').value),sameRun=select.dataset.run===run?.name,previous=select.value,items=run?.checkpoints||[];select.innerHTML=items.map(name=>`<option value="${esc(name)}">${esc(name)}${name==='best.pt'?' · persistence-scored':''}</option>`).join('');if(sameRun&&items.includes(previous))select.value=previous;else if(lab?.run===run?.name&&items.includes(lab.checkpoint))select.value=lab.checkpoint;select.dataset.run=run?.name||'';select.disabled=!items.length;if(!items.length)select.innerHTML='<option>No checkpoints</option>'}
function syncLabRuns(){const select=$('#labRun'),previous=select.value,ready=state.runs.filter(r=>r.lab_ready);select.innerHTML=ready.map(r=>`<option value="${esc(r.name)}">${esc(r.name)} · ${esc(r.kind)}</option>`).join('');if(ready.some(r=>r.name===previous))select.value=previous;else if(lab&&ready.some(r=>r.name===lab.run))select.value=lab.run;$('#labLoad').disabled=labLoading||!ready.length;if(!ready.length)select.innerHTML='<option>No checkpoint-backed runs</option>';syncLabCheckpoints()}
const copyValue=value=>JSON.parse(JSON.stringify(value)),prettyName=name=>name.replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());
function environmentBounds(name){if(name==='obstacle_density')return[0,.3,.01];if(name.endsWith('_direction_x')||name.endsWith('_direction_y'))return[-1,1,.01];return[0,1,.01]}
async function loadTreeSchema(){if(treeSchema)return;treeSchema=await api('/api/tree/schema');$('#treeFamily').innerHTML=treeSchema.families.map(name=>`<option value="${esc(name)}">${esc(prettyName(name))}</option>`).join('');$('#treeGeneControls').innerHTML=treeSchema.genes.map(spec=>`<div class="gene-row"><span>${esc(spec.label)}</span><input type="range" min="${spec.minimum}" max="${spec.maximum}" step="0.01" data-tree-range="${esc(spec.name)}" aria-label="${esc(spec.label)} slider"><input type="number" min="${spec.minimum}" max="${spec.maximum}" step="0.01" data-tree-number="${esc(spec.name)}" aria-label="${esc(spec.label)} value"><label class="gene-lock" title="Lock ${esc(spec.label)}"><input type="checkbox" data-tree-lock="${esc(spec.name)}" aria-label="Lock ${esc(spec.label)}">L</label></div>`).join('');$('#environmentControls').innerHTML=treeSchema.environment_parameters.map(name=>{const[min,max,step]=environmentBounds(name),label=prettyName(name);return`<div class="gene-row"><span>${esc(label)}</span><input type="range" min="${min}" max="${max}" step="${step}" data-environment-range="${esc(name)}" aria-label="${esc(label)} slider"><input type="number" min="${min}" max="${max}" step="${step}" data-environment-number="${esc(name)}" aria-label="${esc(label)} value"><span></span></div>`}).join('')+'<label class="field"><span>Environment field seed</span><input id="environmentSeed" type="number" min="0" max="2147483647" step="1"></label>';document.querySelectorAll('[data-tree-range]').forEach(input=>input.oninput=()=>{document.querySelector(`[data-tree-number="${input.dataset.treeRange}"]`).value=input.value;markTreeDirty()});document.querySelectorAll('[data-tree-number]').forEach(input=>input.oninput=()=>{const range=document.querySelector(`[data-tree-range="${input.dataset.treeNumber}"]`);if(input.value!=='')range.value=input.value;markTreeDirty()});$('#treeFamily').onchange=markTreeDirty;$('#treeStyleSeed').oninput=markTreeDirty;document.querySelectorAll('[data-environment-range]').forEach(input=>input.oninput=()=>{document.querySelector(`[data-environment-number="${input.dataset.environmentRange}"]`).value=input.value;markEnvironmentDirty()});document.querySelectorAll('[data-environment-number]').forEach(input=>input.oninput=()=>{const range=document.querySelector(`[data-environment-range="${input.dataset.environmentNumber}"]`);if(input.value!=='')range.value=input.value;markEnvironmentDirty()});$('#environmentSeed').oninput=markEnvironmentDirty}
const loadTreeSchemaNow=loadTreeSchema;let treeSchemaPromise=null;loadTreeSchema=async function(){if(treeSchema)return;if(!treeSchemaPromise)treeSchemaPromise=loadTreeSchemaNow().finally(()=>treeSchemaPromise=null);await treeSchemaPromise};
function readTreeDraft(){if(!treeSchema)throw new Error('Tree schema is still loading');const genes={};for(const spec of treeSchema.genes){const value=Number(document.querySelector(`[data-tree-number="${spec.name}"]`).value);if(!Number.isFinite(value)||value<spec.minimum||value>spec.maximum)throw new Error(`${spec.label} must be from ${spec.minimum} to ${spec.maximum}`);genes[spec.name]=value}const styleSeed=Number($('#treeStyleSeed').value);if(!Number.isInteger(styleSeed)||styleSeed<0||styleSeed>=2**31)throw new Error('Style seed must be an integer from 0 through 2147483647');return{schema_version:treeSchema.genome_schema_version,family:$('#treeFamily').value,style_seed:styleSeed,genes}}
function setTreeDraft(value,dirty=false){if(!treeSchema||!value)return;const defaults=treeSchema.default_genome,genes={...defaults.genes,...(value.genes||{})};if(!treeSchema.families.includes(value.family))throw new Error('Genome JSON has an unknown tree family');for(const spec of treeSchema.genes){const number=document.querySelector(`[data-tree-number="${spec.name}"]`),range=document.querySelector(`[data-tree-range="${spec.name}"]`);number.value=genes[spec.name];range.value=genes[spec.name]}$('#treeFamily').value=value.family;$('#treeStyleSeed').value=value.style_seed;treeDraft={schema_version:value.schema_version??treeSchema.genome_schema_version,family:value.family,style_seed:Number(value.style_seed),genes};treeDirty=dirty;updateTreeStatus()}
function markTreeDirty(){if(!lab?.continuous_genome)return;treeDirty=true;$('#treeArchive').disabled=true;try{treeDraft=readTreeDraft()}catch(_error){}updateTreeStatus()}
function lockedTreeGenes(){return[...document.querySelectorAll('[data-tree-lock]:checked')].map(input=>input.dataset.treeLock)}
function updateTreeStatus(){const note=$('#treeGenomeStatus');if(!lab?.active_tree_genome){note.textContent='Genome controls load with a tree checkpoint.';note.className='lab-note';return}if(!lab.continuous_genome){note.textContent='Fixed specialist genome · inspect, download, and validate it here. Train or open a tree-family checkpoint to edit.';note.className='lab-note active';return}if(treeDirty){note.textContent='Edited locally · Apply Genome to stage these inherited values.';note.className='lab-note pending'}else if(lab.genome_pending){note.textContent='Genome staged · the mature organism still uses the active genome until Reset seed or a seed is placed.';note.className='lab-note pending'}else{note.textContent=$('#treeLiveRemodel').checked?'Genome is active · live-remodel mode was explicitly enabled.':'Genome is active · edits are isolated from the mature organism.';note.className='lab-note active'}}
function setTreeEditingEnabled(enabled){for(const element of [$('#treeFamily'),$('#treeStyleSeed'),$('#treeApply'),$('#treeRandomize'),$('#treeMutate'),$('#treeStoreA'),$('#treeStoreB'),$('#treeInterpolation'),$('#treeLoad'),$('#treeLiveRemodel'),...document.querySelectorAll('[data-tree-range],[data-tree-number],[data-tree-lock]')])element.disabled=!enabled;$('#treeInterpolate').disabled=!enabled||!treeSlotA||!treeSlotB}
function readEnvironmentDraft(){const value={schema_version:treeSchema.environment_schema_version};for(const name of treeSchema.environment_parameters){const number=Number(document.querySelector(`[data-environment-number="${name}"]`).value),[minimum,maximum]=environmentBounds(name);if(!Number.isFinite(number)||number<minimum||number>maximum)throw new Error(`${prettyName(name)} must be from ${minimum} to ${maximum}`);value[name]=number}const seed=Number($('#environmentSeed').value);if(!Number.isInteger(seed)||seed<0||seed>=2**31)throw new Error('Environment seed must be an integer from 0 through 2147483647');value.seed=seed;return value}
function setEnvironmentDraft(value,dirty=false){if(!treeSchema||!value)return;const merged={...treeSchema.default_environment,...value};for(const name of treeSchema.environment_parameters){document.querySelector(`[data-environment-number="${name}"]`).value=merged[name];document.querySelector(`[data-environment-range="${name}"]`).value=merged[name]}$('#environmentSeed').value=merged.seed;environmentDraft=merged;environmentDirty=dirty;$('#environmentApply').textContent=dirty?'Apply environment *':'Apply environment'}
function markEnvironmentDirty(){if(!lab?.environment)return;environmentDirty=true;$('#treeArchive').disabled=true;try{environmentDraft=readEnvironmentDraft()}catch(_error){}$('#environmentApply').textContent='Apply environment *'}
function validationBinding(value){return value?.report?value:value?.validation?.report?value.validation:null}
function sameJson(left,right){return JSON.stringify(left)===JSON.stringify(right)}
function renderMorphologyStats(report){const descriptors=report?.mean_descriptors||{},cell=(value,digits=1)=>Number.isFinite(Number(value))?Number(value).toFixed(digits):'—';$('#treeMorphologyStats').innerHTML=`<div class="lab-stat"><b>${cell(report?.score,3)}</b><small>persistence</small></div><div class="lab-stat"><b>${cell(descriptors.height)}</b><small>height</small></div><div class="lab-stat"><b>${cell(descriptors.canopy_spread)}</b><small>canopy spread</small></div><div class="lab-stat"><b>${cell(descriptors.occupied_volume,0)}</b><small>occupied volume</small></div>`}
function renderValidation(value){const bound=validationBinding(value),report=bound?.report||value?.validation;renderMorphologyStats(report);if(!report){$('#treeValidationStatus').textContent='Validation has not run for this candidate.';$('#treeArchive').disabled=true;return}const failures=Object.keys(report.failures||{}).length,status=report.accepted?'ARCHIVE-ELIGIBLE':report.validated?'REJECTED':'INCOMPLETE',matches=bound&&bound.checkpoint===lab?.checkpoint&&bound.checkpoint_sha256===lab?.checkpoint_sha256&&sameJson(bound.genome,lab?.pending_tree_genome||lab?.active_tree_genome)&&sameJson(bound.environment,lab?.environment||treeSchema?.default_environment);$('#treeValidationStatus').textContent=`${status} · score ${Number(report.score??0).toFixed(3)} · ${report.trials?.length||0} trials${failures?` · ${failures} failed cases`:''}${report.accepted&&!matches?' · candidate changed since validation':''}`;$('#treeArchive').disabled=!report.accepted||!matches;$('#treeArchive').title=$('#treeArchive').disabled?'Validate this exact genome, environment, and checkpoint first':'Save accepted candidate to the Variant Archive'}
function syncTreePanels(meta,changed){const hasTree=Boolean(meta.active_tree_genome),editable=Boolean(meta.continuous_genome);$('#treeGenomePanel').classList.toggle('lab-hidden',!hasTree);$('#environmentPanel').classList.toggle('lab-hidden',!meta.environment);$('#labModelSummary').innerHTML=`<span>${esc(prettyName(meta.model_kind||'specialist'))}</span><b>${esc(meta.checkpoint||'checkpoint')}</b>`;if(hasTree&&(changed||!treeDirty))setTreeDraft(meta.pending_tree_genome||meta.active_tree_genome,false);setTreeEditingEnabled(editable);updateTreeStatus();if(meta.environment&&(changed||!environmentDirty))setEnvironmentDraft(meta.environment,false);renderValidation(meta.validation)}
function labSource(){return'organism'}function labTargetMode(){return false}function labEditableSource(){return true}function labView(){return lab?.dimensions===3?$('#labView').value:'plane'}function labVoxelMode(){return lab?.dimensions===3&&labView()==='voxels'}function labLayer(){return lab?.dimensions===3&&!labVoxelMode()?Number($('#labLayer').value):0}
function syncLabLayer(center=false){const readOnly=labVoxelMode()||!labEditableSource();$('#labTool').disabled=readOnly;$('#labRadius').disabled=!labEditableSource();if(!lab||lab.dimensions!==3)return;const input=$('#labLayer'),voxel=labVoxelMode();input.disabled=voxel;if(voxel){$('#labLayerValue').value='—';return}const count=lab.layers[labView()]||1;input.max=count-1;if(center||Number(input.value)>=count)input.value=Math.floor(count/2);$('#labLayerValue').value=input.value}
const cubeCorners=[[-.48,-.48,-.48],[.48,-.48,-.48],[.48,.48,-.48],[-.48,.48,-.48],[-.48,-.48,.48],[.48,-.48,.48],[.48,.48,.48],[-.48,.48,.48]],cubeFaces=[[[4,5,6,7],[0,0,1]],[[0,3,2,1],[0,0,-1]],[[1,2,6,5],[1,0,0]],[[0,4,7,3],[-1,0,0]],[[3,7,6,2],[0,1,0]],[[0,1,5,4],[0,-1,0]]],voxelHues=[155,78,205,35,330,190];
function rotateVoxel(x,y,z){const cy=Math.cos(labYaw),sy=Math.sin(labYaw),cp=Math.cos(labPitch),sp=Math.sin(labPitch),rx=x*cy-z*sy,rz=x*sy+z*cy;return[rx,y*cp-rz*sp,y*sp+rz*cp]}
function renderVoxelCanvas(canvas,data,minimum=180){if(!data)return;const context=canvas.getContext('2d'),ratio=Math.min(window.devicePixelRatio||1,2),side=Math.max(minimum,Math.round(canvas.getBoundingClientRect().width*ratio));if(canvas.width!==side||canvas.height!==side){canvas.width=side;canvas.height=side}context.clearRect(0,0,side,side);const[depth,height,width]=data.shape,scale=side*.72/Math.max(depth,height,width)*labZoom,faces=[];for(const[z,y,x,alpha,material]of data.voxels){const center=[x-(width-1)/2,(depth-1)/2-z,y-(height-1)/2];for(const[indices,normal]of cubeFaces){const rotatedNormal=rotateVoxel(...normal);if(rotatedNormal[2]<=0)continue;const points=indices.map(index=>{const corner=cubeCorners[index],rotated=rotateVoxel(center[0]+corner[0],center[1]+corner[1],center[2]+corner[2]);return[side/2+rotated[0]*scale,side/2-rotated[1]*scale,rotated[2]]});faces.push({points,depth:points.reduce((sum,p)=>sum+p[2],0)/4,alpha,material,light:Math.max(.25,Math.min(.8,.45-rotatedNormal[0]*.12+rotatedNormal[1]*.22+rotatedNormal[2]*.25))})}}faces.sort((a,b)=>a.depth-b.depth);for(const face of faces){context.beginPath();face.points.forEach((point,index)=>index?context.lineTo(point[0],point[1]):context.moveTo(point[0],point[1]));context.closePath();context.fillStyle=`hsla(${voxelHues[Math.trunc(face.material)%voxelHues.length]},70%,${Math.round(face.light*100)}%,${Math.max(.35,Math.min(1,face.alpha))})`;context.fill();context.strokeStyle='rgba(2,8,6,.55)';context.lineWidth=Math.max(1,ratio*.55);context.stroke()}if(!faces.length){context.fillStyle='#8ca89d';context.font=`${14*ratio}px system-ui`;context.textAlign='center';context.fillText('No occupied voxels yet',side/2,side/2)}}
function renderVoxels(){renderVoxelCanvas($('#labCanvas'),labVoxelData,320)}function renderTargetVoxels(){renderVoxelCanvas($('#labTargetCanvas'),labTargetVoxelData,180)}
function targetIdentity(meta){return JSON.stringify([meta.run,meta.checkpoint,meta.dimensions,meta.genome,meta.active_tree_genome,meta.environment,meta.target_name])}
async function drawTargetPreview(force=false){if(!lab)return;const key=targetIdentity(lab);if(!force&&key===labTargetKey)return;labTargetKey=key;$('#labTargetCaption').textContent=lab.target_name||'Target';const canvas=$('#labTargetCanvas'),version=lab.frame_version;if(lab.dimensions===3){const data=await api(`/api/lab/voxels?source=target&v=${version}`);if(!lab||targetIdentity(lab)!==key)return;labTargetVoxelData=data;renderTargetVoxels();return}labTargetVoxelData=null;const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Could not render training target'));image.src=`/api/lab/frame?view=plane&layer=0&source=target&v=${version}`});if(!lab||targetIdentity(lab)!==key)return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;const context=canvas.getContext('2d');context.imageSmoothingEnabled=false;context.drawImage(image,0,0)}
async function drawLab(){if(!lab)return;const canvas=$('#labCanvas'),context=canvas.getContext('2d'),view=labView(),layer=labLayer(),source=labSource(),version=lab.frame_version;canvas.classList.toggle('voxel-mode',view==='voxels');$('.lab-help').textContent=source.startsWith('environment:')?`${prettyName(source.split(':')[1])} environment overlay · reference only; environment edits use the Environment Lab controls.`:source==='target'?(view==='voxels'?`Training target: ${lab.target_name} · drag to rotate · wheel to zoom.`:`Training target: ${lab.target_name} · reference only; switch to Organism to edit.`):(view==='voxels'?'Drag to rotate · wheel to zoom · switch to a slice or projection to damage or seed.':'Double-click to place a seed. Choose Damage / erase, then click or drag to remove every channel in that region. For 3D projections, edits occur on the selected layer.');if(view==='voxels'){const data=await api(`/api/lab/voxels?source=${encodeURIComponent(source)}&v=${version}`);if(!lab||lab.frame_version!==version||labView()!=='voxels'||source!==labSource())return;labVoxelData=data;renderVoxels();return}labVoxelData=null;const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error('Could not render lab frame'));image.src=`/api/lab/frame?view=${encodeURIComponent(view)}&layer=${layer}&source=${encodeURIComponent(source)}&v=${version}`});if(!lab||lab.frame_version!==version||view!==labView()||layer!==labLayer()||source!==labSource())return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;context.imageSmoothingEnabled=false;context.drawImage(image,0,0)}
const labSpeedChoices=[[1/60,'1/60×'],[1/30,'1/30×'],[1/15,'1/15×'],[1/10,'1/10×'],[1/5,'1/5×'],[1/2,'1/2×'],[1,'1×'],[2,'2×'],[3,'3×'],[4,'4×'],[5,'5×']];function labSpeedChoice(){return labSpeedChoices[Number($('#labSpeed').value)]}function syncLabSpeed(){const[speed,label]=labSpeedChoice(),target=labDeviceRate*speed/5;$('#labSpeedValue').value=speed===5?`${label} · max`:labDeviceRate?`${label} · ~${target<10?target.toFixed(1):Math.round(target)} steps/s`:`${label} · measuring`}
function renderLabStats(){if(!lab)return;const source=labSource(),target=source==='target',overlay=source.startsWith('environment:'),cells=overlay?'—':target?lab.target_cells:lab.occupied_cells,label=overlay?`${prettyName(source.split(':')[1])} overlay`:target?'target cells':'organism cells';$('#labStats').innerHTML=`<div class="lab-stat"><b>${lab.steps}</b><small>step</small></div><div class="lab-stat"><b>${Number(lab.steps_per_second).toFixed(1)}</b><small>device steps/s</small></div><div class="lab-stat"><b>${cells}</b><small>${esc(label)}</small></div><div class="lab-stat"><b>${esc(lab.device)}</b><small>device</small></div>`}
async function setLab(meta){await loadTreeSchema();const changed=!lab||lab.run!==meta.run||lab.checkpoint!==meta.checkpoint||lab.model_kind!==meta.model_kind||lab.dimensions!==meta.dimensions;if(changed){labDeviceRate=0;treeDirty=false;environmentDirty=false;labTargetKey=''}lab=meta;if(Number(meta.steps_per_second)>0)labDeviceRate=Number(meta.steps_per_second);syncLabSpeed();$('#labStatus').textContent=`${meta.run} · ${meta.dimensions}D`;['#labPlay','#labStep','#labReset','#labClear'].forEach(id=>$(id).disabled=false);$('#treeValidate').disabled=!meta.active_tree_genome||labValidating;$('#openTreeGenomeWindow').disabled=!meta.active_tree_genome;$('#openEnvironmentWindow').disabled=!meta.environment;$('#labGenomeField').classList.toggle('lab-hidden',!meta.conditional||meta.continuous_genome);$('#lab3dControls').classList.toggle('lab-hidden',meta.dimensions!==3);$('#labRun').value=meta.run;syncLabCheckpoints();$('#labCheckpoint').value=meta.checkpoint;if(changed){$('#labGenome').innerHTML=(meta.genomes||[]).map((name,index)=>`<option value="${index}">${esc(name)}</option>`).join('');$('#labGenome').value=meta.genome;$('#labView').value=meta.dimensions===3?'voxels':'slice';labYaw=-.7;labPitch=.55;labZoom=1;syncLabLayer(true)}else{$('#labGenome').value=meta.genome;syncLabLayer()}syncTreePanels(meta,changed);renderLabStats();await drawLab();try{await drawTargetPreview(changed)}catch(error){labTargetKey='';$('#labTargetCaption').textContent='Target unavailable';console.warn(error)}}
function pauseLab(){labPlaying=false;labPlayGeneration++;clearTimeout(labLoopTimer);labLoopTimer=null;$('#labPlay').textContent='Play'}
async function openLab(runName=null){if(labLoading)return;labLoading=true;if(runName){$('#labRun').value=runName;syncLabCheckpoints()}pauseLab();syncLabRuns();while(labBusy)await new Promise(resolve=>setTimeout(resolve,10));$('#labStatus').textContent='LOADING';try{const meta=await labPost('/api/lab/load',{run:$('#labRun').value,checkpoint:$('#labCheckpoint').value||undefined,device:$('#labDevice').value});await setLab(meta);if(!document.body.classList.contains('utility-window'))showPage('checkpoints');$('#designLab').scrollIntoView({behavior:'smooth',block:'start'})}catch(error){$('#labStatus').textContent='ERROR';toast(error.message,true)}finally{labLoading=false;syncLabRuns()}}
async function labAction(action,extra={},wait=false){if(!lab||(labTargetMode()&&(action==='seed'||action==='erase')))return false;while(wait&&labBusy)await new Promise(resolve=>setTimeout(resolve,10));if(labBusy)return false;labBusy=true;try{await setLab(await labPost('/api/lab/action',{action,...extra}));return true}catch(error){pauseLab();toast(error.message,true);return false}finally{labBusy=false}}
async function labControl(action,extra={}){pauseLab();return labAction(action,extra,true)}
function operationSeed(){const value=Number($('#treeOperationSeed').value);if(!Number.isInteger(value)||value<0||value>=2**31)throw new Error('Operation seed must be an integer from 0 through 2147483647');return value}
async function commitTreeDraft(){if(!lab?.continuous_genome)return false;let genome;try{genome=readTreeDraft()}catch(error){toast(error.message,true);return false}treeDirty=false;const completed=await labControl('tree_genome',{genome,live_remodel:$('#treeLiveRemodel').checked});if(!completed){treeDirty=true;updateTreeStatus()}else{loadedVariantId=null;$('#archiveCreationMethod').value='manual';$('#archiveParents').value=''}return completed}
async function randomizeTree(){try{if(treeDirty&&!await commitTreeDraft())return;treeDirty=false;if(await labControl('tree_random',{seed:operationSeed(),locked:lockedTreeGenes()})){loadedVariantId=null;$('#archiveCreationMethod').value='random';$('#archiveParents').value=''}}catch(error){toast(error.message,true)}}
async function mutateTree(){try{if(treeDirty&&!await commitTreeDraft())return;const parent=loadedVariantId;treeDirty=false;if(await labControl('tree_mutate',{strength:Number($('#treeMutation').value),seed:operationSeed(),locked:lockedTreeGenes()})){loadedVariantId=null;$('#archiveCreationMethod').value='mutation';$('#archiveParents').value=parent||''}}catch(error){toast(error.message,true)}}
function updateTreeSlots(){const label=value=>value?`${prettyName(value.family)} #${value.style_seed}`:'empty';$('#treeSlotState').textContent=`A: ${label(treeSlotA)} · B: ${label(treeSlotB)}`;$('#treeInterpolate').disabled=!lab?.continuous_genome||!treeSlotA||!treeSlotB}
function storeTreeSlot(which){try{const value=readTreeDraft();if(which==='A'){treeSlotA=copyValue(value);treeSlotAParent=loadedVariantId}else{treeSlotB=copyValue(value);treeSlotBParent=loadedVariantId}updateTreeSlots()}catch(error){toast(error.message,true)}}
async function interpolateTree(){if(!treeSlotA||!treeSlotB)return;if(treeSlotA.family!==treeSlotB.family){toast('A/B interpolation requires the same discrete tree family',true);return}treeDirty=false;if(await labControl('tree_interpolate',{left:treeSlotA,right:treeSlotB,amount:Number($('#treeInterpolation').value)})){loadedVariantId=null;$('#archiveCreationMethod').value='interpolation';const parents=[treeSlotAParent,treeSlotBParent].filter((value,index,values)=>value&&values.indexOf(value)===index);$('#archiveParents').value=parents.join(',');if(parents.length<2)toast('Interpolation is staged. Archive it after supplying two distinct archived parent ids.')}}
function downloadTreeGenome(){try{const genome=readTreeDraft(),blob=new Blob([JSON.stringify(genome,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`${genome.family}-tree-genome.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),0)}catch(error){toast(error.message,true)}}
async function loadTreeGenomeFile(file){try{const value=JSON.parse(await file.text());if(!value||typeof value!=='object'||Array.isArray(value)||!value.genes||typeof value.genes!=='object')throw new Error('Genome JSON must contain a genes object');setTreeDraft(value,true);toast('Genome JSON loaded locally · Apply Genome when ready')}catch(error){toast(error.message,true)}finally{$('#treeJsonFile').value=''}}
async function applyEnvironment(){let environment;try{environment=readEnvironmentDraft()}catch(error){toast(error.message,true);return false}environmentDirty=false;const completed=await labControl('environment',{environment});if(!completed){environmentDirty=true;$('#environmentApply').textContent='Apply environment *'}return completed}
async function validateTreeCandidate(){if(labValidating||!lab?.active_tree_genome)return;if(treeDirty&&!await commitTreeDraft())return;if(environmentDirty&&!await applyEnvironment())return;const candidate={run:lab.run,checkpoint:lab.checkpoint,genome:copyValue(lab.pending_tree_genome||lab.active_tree_genome),environment:copyValue(lab.environment)};let fireSeeds;try{fireSeeds=$('#treeFireSeeds').value.split(',').map(value=>value.trim()).filter(Boolean).map(Number);if(!fireSeeds.length||fireSeeds.length>8||fireSeeds.some(value=>!Number.isInteger(value)||value<0||value>=2**31))throw new Error('Enter 1 to 8 comma-separated integer fire-mask seeds');const steps=Number($('#treeValidationSteps').value),recoverySteps=Number($('#treeRecoverySteps').value);if(!Number.isInteger(steps)||steps<1||steps>2048||!Number.isInteger(recoverySteps)||recoverySteps<1||recoverySteps>1024)throw new Error('Validation and recovery steps are outside the allowed range');pauseLab();while(labBusy)await new Promise(resolve=>setTimeout(resolve,10));labValidating=true;$('#treeValidate').disabled=true;$('#treeValidationStatus').textContent=`Validating ${steps} steps plus damage recovery across ${fireSeeds.length} fire-mask seeds…`;const result=await labPost('/api/lab/validate',{steps,recovery_steps:recoverySteps,fire_seeds:fireSeeds}),stillCurrent=lab?.run===candidate.run&&lab?.checkpoint===candidate.checkpoint&&sameJson(lab?.pending_tree_genome||lab?.active_tree_genome,candidate.genome)&&sameJson(lab?.environment,candidate.environment);if(stillCurrent){lab.validation=result;renderValidation(result)}else toast(`Validation completed for ${candidate.run} / ${candidate.checkpoint}; the open lab has changed`,true)}catch(error){toast(error.message,true);$('#treeValidationStatus').textContent=`Validation failed to run · ${error.message}`}finally{labValidating=false;$('#treeValidate').disabled=!lab?.active_tree_genome}}
function variantCard(record){const preview=record.preview_urls||{},score=Number(record.validation?.score??0).toFixed(3),method=record.creation?.method||'manual',parents=record.creation?.parents||[];return`<article class="variant-card"><h3>${esc(prettyName(record.genome.family))}</h3><div class="variant-meta">${esc(record.variant_id)}<br>${esc(method)} · score ${score} · ${esc(record.model_identity?.model_kind||'model')}${parents.length?` · ${parents.length} parent${parents.length===1?'':'s'}`:''}</div><div class="variant-previews"><figure><img src="${esc(preview.target||'')}" alt="Target preview for ${esc(record.variant_id)}"><figcaption>Target</figcaption></figure><figure><img src="${esc(preview.final||'')}" alt="Final-state preview for ${esc(record.variant_id)}"><figcaption>Saved state</figcaption></figure></div><button class="btn primary" data-load-variant="${esc(record.variant_id)}">Load genome + environment</button></article>`}
async function refreshVariantArchive(){try{const params=new URLSearchParams(),filters=[['family',$('#archiveFamily').value],['method',$('#archiveMethod').value],['model_kind',$('#archiveModelKind').value],['min_score',$('#archiveMinScore').value]];for(const[name,value]of filters)if(value!=='')params.set(name,value);const result=await api('/api/archive'+(params.size?`?${params}`:''));archiveItems=result.variants||[];$('#archiveCount').textContent=`${archiveItems.length} VARIANT${archiveItems.length===1?'':'S'}`;$('#archiveList').innerHTML=archiveItems.length?archiveItems.map(variantCard).join(''):'<div class="empty">No accepted variants match these filters.</div>';document.querySelectorAll('[data-load-variant]').forEach(button=>button.onclick=()=>loadArchivedVariant(button.dataset.loadVariant))}catch(error){toast(error.message,true)}}
async function saveVariant(){if(treeDirty&&!await commitTreeDraft())return;if(environmentDirty&&!await applyEnvironment())return;try{const parents=$('#archiveParents').value.split(',').map(value=>value.trim()).filter(Boolean),record=await labPost('/api/archive/save',{method:$('#archiveCreationMethod').value,parents});loadedVariantId=record.variant_id;$('#archiveParents').value=record.variant_id;toast(`Archived ${record.variant_id}`);await refreshVariantArchive()}catch(error){toast(error.message,true)}}
async function loadArchivedVariant(variantId){try{pauseLab();while(labBusy)await new Promise(resolve=>setTimeout(resolve,10));const result=await labPost('/api/archive/load',{variant_id:variantId});loadedVariantId=variantId;$('#archiveCreationMethod').value='manual';$('#archiveParents').value=variantId;await setLab(result.lab);toast(result.warning||`Loaded ${variantId}; genome is pending until reset or seed`,Boolean(result.warning));if(result.warning)$('#treeValidationStatus').textContent=result.warning;if(!document.body.classList.contains('utility-window'))showPage('checkpoints')}catch(error){toast(error.message,true)}}
async function labLoop(generation){if(!labPlaying||generation!==labPlayGeneration)return;const[speed]=labSpeedChoice(),steps=speed>=1?Math.round(speed):1,started=performance.now(),advanced=await labAction('advance',{steps});if(!labPlaying||generation!==labPlayGeneration)return;const targetRate=Math.max(labDeviceRate*speed/5,.01),delay=advanced&&speed<5?Math.max(0,steps/targetRate*1000-(performance.now()-started)):advanced?0:10;labLoopTimer=setTimeout(()=>labLoop(generation),delay)}
function toggleLab(){if(labPlaying){pauseLab();return}labPlaying=true;const generation=++labPlayGeneration;$('#labPlay').textContent='Pause';labLoop(generation)}
function labPoint(event){const canvas=$('#labCanvas'),rect=canvas.getBoundingClientRect();return{row:Math.max(0,Math.min(canvas.height-1,Math.floor((event.clientY-rect.top)/rect.height*canvas.height))),column:Math.max(0,Math.min(canvas.width-1,Math.floor((event.clientX-rect.left)/rect.width*canvas.width)))}}
function labEditPayload(event){return{...labPoint(event),view:labView(),layer:labLayer(),radius:Number($('#labRadius').value)}}
async function editLab(payload,forceSeed=false,wait=true){if(!lab||!labEditableSource())return;await labAction(forceSeed?'seed':$('#labTool').value,payload,wait)}
function jobs(){const running=state.jobs.filter(j=>j.status==='running');$('#jobCount').textContent=`${running.length} active`;return state.jobs.map(j=>`<article class="job"><div class="job-row"><div><b>${esc(j.run_name)}</b><div class="status ${j.status}">${esc(j.status)}</div></div>${j.status==='running'?`<button class="btn danger" onclick="stopJob('${esc(j.id)}')">Stop</button>`:''}</div>${j.live?`<figure class="live-view"><img src="${esc(j.live.url)}" alt="Live organism state for ${esc(j.run_name)}"><figcaption>${esc(liveLabel(j.live))}</figcaption></figure>`:''}<pre>${esc(j.log||'Waiting for output…')}</pre></article>`).join('')}
function runs(){const q=$('#search').value.toLowerCase();const items=state.runs.filter(r=>matchesArea(r)&&r.name.toLowerCase().includes(q));$('#runCount').textContent=`${state.runs.length} runs`;$('#runList').innerHTML=jobs()+(items.length?items.map(r=>`<article class="run-card ${selectedRun===r.name?'active':''}" data-run="${esc(r.name)}"><h3>${esc(r.name)}</h3><small>${esc(r.kind)} · ${r.media.length} visuals · ${r.lab_ready?'lab ready':'artifacts only'}</small></article>`).join(''):'<div class="empty">No matching runs.</div>');document.querySelectorAll('[data-run]').forEach(el=>el.onclick=()=>showRun(el.dataset.run))}
function showRun(name){selectedRun=name;const r=state.runs.find(r=>r.name===name);if(!r)return;$('#resultTitle').textContent=r.name;$('#resultKind').textContent=r.kind.toUpperCase();$('#resultDate').textContent=new Date(r.updated).toLocaleString();let html=r.lab_ready?`<div class="actions" style="margin-bottom:14px"><button class="btn primary" data-open-lab="${esc(r.name)}">Open in View Checkpoints</button></div>`:'';html+=r.media.length?`<div class="gallery">${r.media.map(m=>`<figure class="media"><img src="${m.url}" alt="${esc(m.name)} visualization"><figcaption>${esc(m.name)}</figcaption></figure>`).join('')}</div>`:'<div class="empty">This run has no saved images yet.</div>';for(const metric of r.metrics){html+=`<div class="table-wrap"><table><caption style="padding:10px;text-align:left">${esc(metric.name)}</caption><thead><tr>${metric.columns.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${metric.rows.map(row=>`<tr>${metric.columns.map(c=>`<td>${esc(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}if(r.logs)html+=`<h3>Training log</h3><pre>${esc(r.logs)}</pre>`;$('#resultBody').innerHTML=html;document.querySelectorAll('[data-open-lab]').forEach(button=>button.onclick=()=>openLab(button.dataset.openLab));runs()}
async function refresh(){try{const first=!state.configs.length;state=await api('/api/state');await loadTreeSchema();if(!$('#archiveFamily').dataset.ready){$('#archiveFamily').dataset.ready='1';$('#archiveFamily').innerHTML='<option value="">All families</option>'+treeSchema.families.map(name=>`<option value="${esc(name)}">${esc(prettyName(name))}</option>`).join('');refreshVariantArchive()}const h=state.hardware||{};$('#deviceStatus').textContent=h.cuda_available?`GPU · ${h.cuda_name}`:`Auto · ${h.auto_device||'CPU'}`;for(const select of [$('#device'),$('#labDevice')])select.querySelector('option[value="cuda"]').disabled=!h.cuda_available;if(selectedRun&&!state.runs.some(r=>r.name===selectedRun))selectedRun=null;configs(first);syncDependencyCheckpoints();syncLabRuns();runs();if(state.lab&&(!lab||state.lab.run!==lab.run||state.lab.frame_version!==lab.frame_version))await setLab(state.lab);if(selectedRun)showRun(selectedRun)}catch(e){toast(e.message,true)}}
async function launch(){try{const job=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:TOKEN,config:$('#configSelect').value,content:$('#editor').value,run_name:$('#runName').value,device:$('#device').value,live_preview:$('#livePreview').checked})});toast(`Started ${job.run_name}`);$('#runName').value='';await refresh()}catch(e){toast(e.message,true)}}async function stopJob(id){try{await api('/api/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:TOKEN,job:id})});toast('Stop requested');await refresh()}catch(e){toast(e.message,true)}}
$('#labRun').onchange=syncLabCheckpoints;$('#treeApply').onclick=commitTreeDraft;$('#treeRandomize').onclick=randomizeTree;$('#treeMutate').onclick=mutateTree;$('#treeMutation').oninput=()=>$('#treeMutationValue').value=Number($('#treeMutation').value).toFixed(2);$('#treeStoreA').onclick=()=>storeTreeSlot('A');$('#treeStoreB').onclick=()=>storeTreeSlot('B');$('#treeInterpolation').oninput=()=>$('#treeInterpolationValue').value=Number($('#treeInterpolation').value).toFixed(2);$('#treeInterpolate').onclick=interpolateTree;$('#treeDownload').onclick=downloadTreeGenome;$('#treeLoad').onclick=()=>$('#treeJsonFile').click();$('#treeJsonFile').onchange=event=>{if(event.target.files[0])loadTreeGenomeFile(event.target.files[0])};$('#treeValidate').onclick=validateTreeCandidate;$('#treeArchive').onclick=saveVariant;$('#environmentApply').onclick=applyEnvironment;$('#environmentReset').onclick=()=>setEnvironmentDraft(lab?.environment||treeSchema?.default_environment,false);$('#archiveRefresh').onclick=refreshVariantArchive;for(const id of ['#archiveFamily','#archiveMethod','#archiveModelKind'])$(id).onchange=refreshVariantArchive;$('#archiveMinScore').onchange=refreshVariantArchive;loadTreeSchema().then(()=>{$('#archiveFamily').innerHTML='<option value="">All families</option>'+treeSchema.families.map(name=>`<option value="${esc(name)}">${esc(prettyName(name))}</option>`).join('');refreshVariantArchive()}).catch(error=>toast(error.message,true));
document.querySelectorAll('.nav button[data-kind]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));b.classList.add('active');kind=b.dataset.kind;configs(true);runs()});$('#labNav').onclick=()=>$('#designLab').scrollIntoView({behavior:'smooth',block:'start'});$('#configSelect').onchange=loadConfig;$('#dependencyCheckpoint').onchange=()=>setYamlText(dependencySpecs[$('#configSelect').value].key,$('#dependencyCheckpoint').value);$('#editor').onchange=()=>{renderSliders();syncDependencyCheckpoints()};$('#reset').onclick=loadConfig;$('#launch').onclick=launch;$('#refresh').onclick=refresh;$('#search').oninput=runs;$('#labLoad').onclick=()=>openLab();$('#labPlay').onclick=toggleLab;$('#labStep').onclick=()=>labControl('advance',{steps:1});$('#labReset').onclick=()=>labControl('reset');$('#labClear').onclick=()=>labControl('clear');$('#labGenome').onchange=()=>labControl('genome',{genome:Number($('#labGenome').value)});$('#labView').onchange=()=>{syncLabLayer(true);drawLab()};$('#labLayer').oninput=()=>{$('#labLayerValue').value=$('#labLayer').value;drawLab()};$('#labSpeed').oninput=()=>{syncLabSpeed();if(labPlaying){const generation=++labPlayGeneration;clearTimeout(labLoopTimer);labLoopTimer=setTimeout(()=>labLoop(generation),labBusy?10:0)}};$('#labRadius').oninput=()=>$('#labRadiusValue').value=$('#labRadius').value;const labCanvas=$('#labCanvas');labCanvas.onpointerdown=event=>{if(event.button!==0||!lab)return;labCanvas.setPointerCapture(event.pointerId);if(labVoxelMode()){labOrbiting=true;labLastX=event.clientX;labLastY=event.clientY;return}labEditing=true;labMoved=false};labCanvas.onpointermove=event=>{if(labOrbiting){labYaw+=(event.clientX-labLastX)*.012;labPitch=Math.max(-1.45,Math.min(1.45,labPitch+(event.clientY-labLastY)*.012));labLastX=event.clientX;labLastY=event.clientY;renderVoxels();return}if(labEditing&&$('#labTool').value==='erase'){labMoved=true;clearTimeout(labClickTimer);editLab(labEditPayload(event),false,false)}};labCanvas.onpointerup=event=>{if(labOrbiting){labOrbiting=false;return}if(!labEditing)return;labEditing=false;const payload=labEditPayload(event);if(labMoved){labAction('erase',payload,true);return}if($('#labTool').value==='seed')editLab(payload);else{clearTimeout(labClickTimer);labClickTimer=setTimeout(()=>editLab(payload),450)}};labCanvas.onpointercancel=()=>{labEditing=false;labOrbiting=false};labCanvas.ondblclick=event=>{if(labVoxelMode())return;event.preventDefault();clearTimeout(labClickTimer);editLab(labEditPayload(event),true)};labCanvas.onwheel=event=>{if(!labVoxelMode())return;event.preventDefault();labZoom=Math.max(.5,Math.min(3,labZoom*Math.exp(-event.deltaY*.001)));renderVoxels()};labCanvas.oncontextmenu=event=>event.preventDefault();window.addEventListener('resize',()=>{if(labVoxelMode())renderVoxels()});refresh();setInterval(()=>{if(state.jobs.some(j=>j.status==='running'))refresh()},2000);
document.querySelectorAll('.nav button[data-page]').forEach(button=>button.onclick=()=>showPage(button.dataset.page));$('#openTreeGenomeWindow').onclick=()=>openUtilityWindow('genome');$('#openEnvironmentWindow').onclick=()=>openUtilityWindow('environment-lab');labCanvas.addEventListener('pointermove',()=>{if(labOrbiting)renderTargetVoxels()});labCanvas.addEventListener('wheel',renderTargetVoxels);window.addEventListener('resize',renderTargetVoxels);setInterval(()=>{if((lab||state.lab)&&!state.jobs.some(job=>job.status==='running'))refresh()},2000);
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
