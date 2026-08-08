"""Run-directory and metadata helpers."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def create_run_directory(name: str, root: str | Path = "runs") -> Path:
    run = Path(root) / name
    for child in ("checkpoints", "rollouts", "metrics", "visualizations", "targets"):
        (run / child).mkdir(parents=True, exist_ok=True)
    return run


def metadata(seed: int, model, device: torch.device) -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "pytorch_version": torch.__version__, "parameter_count": sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0,
        "random_seeds": [seed], "git_commit": commit, "command": " ".join(sys.argv), "checkpoint_identifier": "latest.pt",
    }


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def steps_per_second(steps: int, started: float) -> float:
    """Return average completed cellular updates per wall-clock second."""
    return round(steps / max(time.perf_counter() - started, 1e-9), 1) if steps else 0.0


def write_live_preview(path: str | Path, image, **progress: object) -> None:
    """Atomically replace the dashboard preview image and its progress metadata."""
    def replace(temporary: Path, destination: Path) -> bool:
        for attempt in range(20):
            try:
                temporary.replace(destination)
                return True
            except PermissionError:
                if attempt == 19:
                    temporary.unlink(missing_ok=True)
                    return False
                time.sleep(0.005)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    Image.fromarray(image).save(temporary)
    if not replace(temporary, target):
        return
    progress_path = target.with_suffix(".json")
    progress_temporary = progress_path.with_name(f".{progress_path.stem}.tmp{progress_path.suffix}")
    progress_temporary.write_text(json.dumps(progress, allow_nan=False), encoding="utf-8")
    replace(progress_temporary, progress_path)
