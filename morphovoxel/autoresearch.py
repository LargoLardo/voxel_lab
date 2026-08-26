"""Isolated, frozen-score autoresearch campaigns for tree-family training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import load_checkpoint
from .config import load_config, save_config
from .environment import ENVIRONMENT_CHANNELS, EnvironmentSpec, environment_context_batch
from .genomes import TREE_FAMILIES, TreeGenome, tree_genome_tensor
from .metrics import threshold_iou
from .model_3d import NeuralCA3D, TreeFamilyNCA3D
from .random_utils import resolve_device
from .rollout import rollout
from .seeding import seed_state
from .state import StateLayout
from .targets import make_tree_target
from .validation import ValidationCriteria, build_validation_panel, validate_panel


CANDIDATE = Path("configs/autoresearch_candidate.yaml")
PROGRAM = Path("autoresearch/program.md")
SCOPES = {
    "config": (CANDIDATE,),
    "expanded": (
        CANDIDATE,
        Path("morphovoxel/model_3d.py"),
        Path("morphovoxel/training/losses.py"),
        Path("morphovoxel/training/family.py"),
    ),
}
FROZEN_FILES = (
    Path("morphovoxel/autoresearch.py"),
    Path("morphovoxel/validation.py"),
    Path("morphovoxel/metrics.py"),
    Path("morphovoxel/targets/targets_3d.py"),
    PROGRAM,
)
BENCHMARK_SEED = 90210
FIRE_SEEDS = (7701, 7702)
PROXY_WORLD_SIZE = 16
PROXY_STEPS = 400
VALIDATION_STEPS = 256
RECOVERY_STEPS = 64


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(command: list[str], *, root: Path, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=root, text=True, **kwargs)


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    result = _run(["git", *arguments], root=root, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git command failed")
    return result.stdout.strip()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_hashes(root: Path) -> dict[str, str]:
    return {path.as_posix(): _hash(root / path) for path in FROZEN_FILES}


def _changed_paths(root: Path) -> set[Path]:
    lines = _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    paths = set()
    for line in lines:
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.add(Path(value.replace("\\", "/")))
    return paths


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _append_record(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _campaign_path(root: Path) -> Path:
    configured = os.environ.get("MORPHOVOXEL_AUTORESEARCH_CAMPAIGN")
    return Path(configured).resolve() if configured else root / "runs" / "autoresearch" / "active.json"


def _load_campaign(root: Path) -> tuple[Path, dict[str, Any]]:
    path = _campaign_path(root)
    if not path.is_file():
        raise RuntimeError("no active autoresearch campaign; start one with --hours before running --trial")
    state = json.loads(path.read_text(encoding="utf-8"))
    if Path(state["root"]).resolve() != root:
        raise RuntimeError("active campaign belongs to a different repository")
    return path, state


def _verify_campaign(root: Path, state: dict[str, Any], *, allow_clean: bool) -> set[Path]:
    branch = _git(root, "branch", "--show-current")
    if branch != state["branch"]:
        raise RuntimeError(f"campaign belongs to branch {state['branch']!r}, current branch is {branch!r}")
    changed = _changed_paths(root)
    allowed = {Path(value) for value in state["editable_paths"]}
    unexpected = changed - allowed
    if unexpected:
        raise RuntimeError("immutable-file guard rejected changes to: " + ", ".join(sorted(map(str, unexpected))))
    if not allow_clean and CANDIDATE not in changed and state["scope"] == "config":
        raise RuntimeError("a config-only trial must change configs/autoresearch_candidate.yaml")
    current_hashes = _frozen_hashes(root)
    changed_frozen = [name for name, digest in state["frozen_hashes"].items() if current_hashes.get(name) != digest]
    if changed_frozen:
        raise RuntimeError("frozen evaluator changed: " + ", ".join(changed_frozen))
    baseline = Path(state["baseline_checkpoint"])
    if not baseline.is_file() or _hash(baseline) != state["baseline_sha256"]:
        raise RuntimeError("the frozen specialist checkpoint is missing or changed")
    return changed


def _trial_config(root: Path, state: dict[str, Any], trial_id: str, trial_dir: Path) -> dict[str, Any]:
    config = load_config(root / CANDIDATE)
    config.update({
        "run_name": f"autoresearch_{trial_id}",
        "runs_root": str((trial_dir / "runs").resolve()),
        "dimensions": 3,
        "model_kind": "tree_family",
        "conditional": True,
        "environment_conditioning": True,
        "initialize_from_specialist": state["baseline_checkpoint"],
        "resume": None,
        "seed": BENCHMARK_SEED,
        "device": "auto",
        "world_size": PROXY_WORLD_SIZE,
        "iterations": int(state["proxy_steps"]),
        "validation_steps": 0,
        "live_preview": False,
    })
    config.pop("resume", None)
    save_config(config, trial_dir / "candidate.yaml")
    return config


def _worker(config_path: Path, output_path: Path) -> int:
    from .training import train

    config = load_config(config_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        run = train(config, dimensions=3, conditional=True)
        result = {
            "run_dir": str(run.resolve()),
            "training_seconds": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
        }
        _write_json(output_path, result)
        return 0
    except BaseException as error:
        _write_json(output_path, {
            "error": f"{type(error).__name__}: {error}",
            "training_seconds": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
        })
        raise


def _load_model(config: dict[str, Any], checkpoint: Path):
    device = resolve_device("auto")
    layout = StateLayout(int(config.get("materials", 4)), int(config.get("hidden_channels", 8)))
    model = TreeFamilyNCA3D(
        layout.channels,
        int(config.get("model_width", 64)),
        TreeGenome.model_size(),
        float(config.get("fire_rate", 0.5)),
        len(ENVIRONMENT_CHANNELS),
        len(TREE_FAMILIES),
    ).to(device)
    load_checkpoint(checkpoint, model, map_location=device, expected_model_kind="tree_family")
    return model, layout, device


def _genome_sensitivity(model, layout: StateLayout, device: torch.device) -> dict[str, float]:
    default = TreeGenome(family="branching")
    genomes = [
        default.with_values({"height": -0.8, "branch_density": -0.8, "canopy_spread": -0.8}),
        default.with_values({"height": 0.8, "branch_density": -0.8, "canopy_spread": 0.8}),
        default.with_values({"height": -0.8, "branch_density": 0.8, "branch_length": 0.8}),
        default.with_values({"height": 0.8, "branch_density": 0.8, "branch_length": -0.8}),
    ]
    environment = EnvironmentSpec()
    targets = [torch.as_tensor(make_tree_target(genome, PROXY_WORLD_SIZE, environment)[0], device=device) for genome in genomes]
    predictions = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            for genome in genomes:
                torch.manual_seed(FIRE_SEEDS[0])
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(FIRE_SEEDS[0])
                state = seed_state(1, PROXY_WORLD_SIZE, layout, dimensions=3, random_seed=FIRE_SEEDS[0], device=device)
                context = environment_context_batch((environment,), PROXY_WORLD_SIZE, device=device)
                final, _ = rollout(model, state, VALIDATION_STEPS, tree_genome_tensor((genome,), device=device), context=context)
                predictions.append(final[0, layout.occupancy])
    finally:
        model.train(was_training)
    matrix = np.asarray([[threshold_iou(prediction, target) for target in targets] for prediction in predictions])
    own = np.diag(matrix)
    best_cross = np.asarray([np.max(np.delete(row, index)) for index, row in enumerate(matrix)])
    margins = own - best_cross
    return {
        "genome_own_target_iou": float(own.mean()),
        "genome_cross_target_iou": float(best_cross.mean()),
        "genome_sensitivity": float(margins.mean()),
        "worst_genome_sensitivity": float(margins.min()),
    }


def _evaluate(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    model, layout, device = _load_model(config, checkpoint)
    panel = build_validation_panel(
        seed=BENCHMARK_SEED,
        boundary_genes=("height", "branch_density", "branch_length", "canopy_spread"),
        random_count=0,
        interpolation_steps=0,
        mutation_count=0,
        fire_seeds=FIRE_SEEDS,
        environments=(EnvironmentSpec(),),
    )
    report = validate_panel(
        model,
        panel,
        layout=layout,
        world_size=PROXY_WORLD_SIZE,
        steps=VALIDATION_STEPS,
        recovery_steps=RECOVERY_STEPS,
        device=device,
        criteria=ValidationCriteria(),
    )
    trials = report.trials
    metrics = {
        "validation_cases": len(trials),
        "accepted_cases": sum(trial.accepted for trial in trials),
        "finite_cases": sum(bool(trial.metrics["finite_state"]) for trial in trials),
        "bounded_cases": sum(bool(trial.metrics["bounded_state"]) for trial in trials),
        "worst_iou": min(trial.metrics["target_iou"] for trial in trials),
        "worst_branch_dice": min(trial.metrics["branch_dice"] for trial in trials),
        "worst_leaf_dice": min(trial.metrics["leaf_dice"] for trial in trials),
        "max_occupancy_violation": max(trial.metrics["occupancy_range_violation_fraction"] for trial in trials),
        "max_magnitude": max(trial.metrics["max_channel_magnitude"] for trial in trials),
        "max_late_drift": max(trial.metrics["late_drift"] for trial in trials),
        "worst_regeneration": min(trial.metrics["regeneration_score"] for trial in trials),
        "strict_accepted": report.accepted,
        **_genome_sensitivity(model, layout, device),
    }
    return metrics


def _ranking(metrics: dict[str, Any], training_seconds: float, peak_vram_mb: float) -> list[float]:
    """Immutable graded order; strict archive acceptance remains unchanged."""
    return [
        1.0,
        float(metrics["finite_cases"]),
        float(metrics["bounded_cases"]),
        -float(metrics["max_occupancy_violation"]),
        float(metrics["worst_iou"]),
        float(metrics["worst_branch_dice"]),
        float(metrics["worst_leaf_dice"]),
        float(metrics["worst_genome_sensitivity"]),
        -float(metrics["max_late_drift"]),
        float(metrics["worst_regeneration"]),
        -float(training_seconds),
        -float(peak_vram_mb),
    ]


def _discard_candidate(root: Path, paths: set[Path]) -> None:
    tracked = [path.as_posix() for path in paths if _git(root, "ls-files", "--error-unmatch", path.as_posix(), check=False)]
    if tracked:
        _git(root, "restore", "--staged", "--worktree", "--", *tracked)


def _trial(root: Path, hypothesis: str, *, allow_clean: bool = False) -> dict[str, Any]:
    campaign_path, state = _load_campaign(root)
    changed = _verify_campaign(root, state, allow_clean=allow_clean)
    records_path = Path(state["records_path"])
    sequence = len(_read_records(records_path))
    trial_id = f"{sequence:04d}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    trial_dir = Path(state["campaign_dir"]) / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / CANDIDATE, trial_dir / CANDIDATE.name)
    parent_commit = _git(root, "rev-parse", "HEAD")
    config = _trial_config(root, state, trial_id, trial_dir)
    record: dict[str, Any] = {
        "trial_id": trial_id,
        "parent_commit": parent_commit,
        "hypothesis": hypothesis,
        "configuration": config,
        "status": "crash",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        smoke = _run(
            [sys.executable, "-m", "pytest", "tests/test_genome_redesign.py", "-q", "-p", "no:cacheprovider"],
            root=root,
            capture_output=True,
            timeout=float(state["smoke_timeout_seconds"]),
        )
        (trial_dir / "smoke.log").write_text(smoke.stdout + smoke.stderr, encoding="utf-8")
        if smoke.returncode:
            raise RuntimeError("frozen smoke test failed")
        worker_result = trial_dir / "worker_result.json"
        with (trial_dir / "training.log").open("w", encoding="utf-8") as log:
            worker = subprocess.run(
                [sys.executable, "-m", "morphovoxel.autoresearch", "--worker", str(trial_dir / "candidate.yaml"), str(worker_result)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=float(state["trial_timeout_seconds"]),
            )
        worker_data = json.loads(worker_result.read_text(encoding="utf-8")) if worker_result.is_file() else {}
        record.update({
            "training_seconds": float(worker_data.get("training_seconds", state["trial_timeout_seconds"])),
            "peak_vram_mb": float(worker_data.get("peak_vram_mb", 0.0)),
        })
        if worker.returncode:
            raise RuntimeError(worker_data.get("error", "training worker failed"))
        run_dir = Path(worker_data["run_dir"])
        checkpoint = run_dir / "checkpoints" / "latest.pt"
        metrics = _evaluate(config, checkpoint)
        score = _ranking(metrics, record["training_seconds"], record["peak_vram_mb"])
        record.update(metrics)
        record["research_score"] = score
        best = state.get("best_score")
        keep = best is None or tuple(score) > tuple(best)
        if keep:
            if changed:
                _git(root, "add", "--", *(path.as_posix() for path in sorted(changed)))
                message = " ".join(hypothesis.split())[:72] or "candidate improvement"
                _git(root, "commit", "-m", f"autoresearch: {message}")
            record["status"] = "keep"
            record["commit"] = _git(root, "rev-parse", "HEAD")
            state["best_score"] = score
            state["best_trial_id"] = trial_id
        else:
            _discard_candidate(root, changed)
            record["status"] = "discard"
            record["commit"] = parent_commit
    except subprocess.TimeoutExpired:
        _discard_candidate(root, changed)
        record.update({"status": "timeout", "error": "trial exceeded its frozen wall-clock limit", "commit": parent_commit})
    except Exception as error:
        _discard_candidate(root, changed)
        record.update({"status": "crash", "error": f"{type(error).__name__}: {error}", "commit": parent_commit})
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _append_record(records_path, record)
    state["trial_count"] = sequence + 1
    _write_json(campaign_path, state)
    _write_json(trial_dir / "record.json", record)
    return record


def _new_campaign(root: Path, args) -> tuple[Path, dict[str, Any]]:
    branch = _git(root, "branch", "--show-current")
    if branch in {"main", "master"}:
        raise RuntimeError("autoresearch must run on a dedicated branch, not main/master")
    if _changed_paths(root):
        raise RuntimeError("commit or discard current changes before starting an autoresearch campaign")
    candidate = load_config(root / CANDIDATE)
    baseline = _resolve_baseline(root, candidate, args.baseline_checkpoint)
    campaign_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = root / "runs" / "autoresearch" / campaign_id
    state = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_dir": str(campaign_dir.resolve()),
        "records_path": str((campaign_dir / "trials.jsonl").resolve()),
        "root": str(root),
        "branch": branch,
        "scope": args.scope,
        "editable_paths": [path.as_posix() for path in SCOPES[args.scope]],
        "frozen_hashes": _frozen_hashes(root),
        "baseline_checkpoint": str(baseline),
        "baseline_sha256": _hash(baseline),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "deadline_epoch": time.time() + args.hours * 3600,
        "proxy_steps": args.proxy_steps,
        "trial_timeout_seconds": args.trial_minutes * 60,
        "smoke_timeout_seconds": args.smoke_minutes * 60,
        "trial_count": 0,
        "best_score": None,
        "best_trial_id": None,
    }
    campaign_path = root / "runs" / "autoresearch" / "active.json"
    if not args.dry_run:
        _write_json(campaign_path, state)
        os.environ["MORPHOVOXEL_AUTORESEARCH_CAMPAIGN"] = str(campaign_path)
    return campaign_path, state


def _resolve_baseline(root: Path, candidate: dict[str, Any], explicit: str | None) -> Path:
    layout = StateLayout(int(candidate.get("materials", 4)), int(candidate.get("hidden_channels", 8)))

    def compatible(path: Path) -> tuple[bool, int]:
        if not path.is_file():
            return False, 0
        model = NeuralCA3D(
            layout.channels,
            int(candidate.get("model_width", 64)),
            0,
            float(candidate.get("fire_rate", 0.5)),
            int(candidate.get("specialist_context_channels", 0)),
        )
        try:
            payload = load_checkpoint(path, model, map_location="cpu", expected_model_kind="tree_specialist")
            return True, int((payload.get("config") or {}).get("world_size", 0))
        except (OSError, RuntimeError, ValueError):
            return False, 0

    configured = Path(explicit or candidate.get("initialize_from_specialist", ""))
    configured = configured if configured.is_absolute() else root / configured
    valid, _ = compatible(configured)
    if valid:
        return configured.resolve()
    if explicit:
        raise ValueError(f"explicit specialist baseline is missing or incompatible: {configured}")
    candidates = sorted(
        (path for path in (root / "runs").glob("*/checkpoints/best.pt") if path != configured),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    matches = []
    for path in candidates:
        valid, world_size = compatible(path)
        if valid:
            matches.append((path, world_size))
    if matches:
        matches.sort(key=lambda item: item[1] == PROXY_WORLD_SIZE, reverse=True)
        return matches[0][0].resolve()
    raise FileNotFoundError(
        f"no compatible tree-specialist best.pt was found; configured path was {configured}. "
        "Train the current tree_specialist preset or pass --baseline-checkpoint."
    )


def _agent_prompt(root: Path, state: dict[str, Any]) -> str:
    records = _read_records(Path(state["records_path"]))
    recent = records[-5:]
    context = {
        "remaining_minutes": max(0, round((state["deadline_epoch"] - time.time()) / 60, 1)),
        "editable_paths": state["editable_paths"],
        "best_trial_id": state.get("best_trial_id"),
        "best_score": state.get("best_score"),
        "recent_trials": [
            {key: row.get(key) for key in ("trial_id", "hypothesis", "status", "research_score", "error")}
            for row in recent
        ],
    }
    return (root / PROGRAM).read_text(encoding="utf-8") + "\n\n## Campaign context\n\n```json\n" + json.dumps(context, indent=2) + "\n```\n"


def _campaign(root: Path, args) -> int:
    campaign_path, state = _new_campaign(root, args)
    if args.dry_run:
        print(json.dumps({
            "campaign": str(campaign_path),
            "branch": state["branch"],
            "scope": state["scope"],
            "editable_paths": state["editable_paths"],
            "baseline_checkpoint": state["baseline_checkpoint"],
            "proxy_steps": state["proxy_steps"],
            "trial_timeout_minutes": args.trial_minutes,
        }, indent=2))
        return 0
    baseline = _trial(root, "Frozen baseline", allow_clean=True)
    print(json.dumps(baseline, indent=2))
    state = json.loads(campaign_path.read_text(encoding="utf-8"))
    if args.manual:
        print(f"Campaign initialized. Use {PROGRAM} in Goal mode; records: {state['records_path']}")
        return 0
    while time.time() < state["deadline_epoch"]:
        prompt = _agent_prompt(root, state)
        remaining = max(60, int(state["deadline_epoch"] - time.time()))
        codex = shutil.which("codex.cmd" if os.name == "nt" else "codex") or shutil.which("codex")
        if not codex:
            raise RuntimeError("Codex CLI was not found; rerun with --manual and use autoresearch/program.md in Goal mode")
        result = _run(
            [codex, "exec", "--ephemeral", "-C", str(root), "-s", "workspace-write", "-a", "never", "-"],
            root=root,
            input=prompt,
            timeout=remaining,
        )
        state = json.loads(campaign_path.read_text(encoding="utf-8"))
        _verify_campaign(root, state, allow_clean=True)
        if result.returncode:
            print(f"Codex agent exited with status {result.returncode}; stopping campaign.", file=sys.stderr)
            return result.returncode
    print(f"Campaign time budget complete. Records: {state['records_path']}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-score MorphoVoxel autoresearch")
    parser.add_argument("--hours", type=float, default=8.0, help="total campaign wall-clock budget")
    parser.add_argument("--scope", choices=tuple(SCOPES), default="config", help="candidate edit allowlist")
    parser.add_argument("--proxy-steps", type=int, default=PROXY_STEPS, help="frozen optimizer updates per proxy trial")
    parser.add_argument("--trial-minutes", type=float, default=20.0, help="hard training timeout per trial")
    parser.add_argument("--smoke-minutes", type=float, default=2.0, help="hard smoke-test timeout")
    parser.add_argument("--baseline-checkpoint", help="fixed specialist best.pt; defaults to candidate config")
    parser.add_argument("--manual", action="store_true", help="initialize after the baseline for an existing Goal-mode agent")
    parser.add_argument("--dry-run", action="store_true", help="validate and print a campaign plan without training")
    parser.add_argument("--trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hypothesis", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker", nargs=2, metavar=("CONFIG", "RESULT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.hours <= 0 or args.proxy_steps < 1 or args.trial_minutes <= 0 or args.smoke_minutes <= 0:
        parser.error("hours, proxy steps, and timeouts must be positive")
    root = _root()
    if args.worker:
        raise SystemExit(_worker(Path(args.worker[0]).resolve(), Path(args.worker[1]).resolve()))
    if args.trial:
        if not args.hypothesis.strip():
            parser.error("--trial requires a non-empty --hypothesis")
        print(json.dumps(_trial(root, args.hypothesis.strip()), indent=2))
        return
    raise SystemExit(_campaign(root, args))


if __name__ == "__main__":
    main()
