"""Preview or remove disposable development artifacts from the repository."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT_NAMES = {
    ".hypothesis",
    ".matplotlib",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".verification-runs",
    "build",
    "dist",
    "htmlcov",
}
ROOT_PREFIXES = (".test-", ".tmp-")
RECURSIVE_DIR_NAMES = {".hypothesis", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
PROTECTED_TOP_LEVEL = {".git", ".venv", "graphify-out", "runs", "variant_archive"}
DISPOSABLE_FILES = {".coverage", ".DS_Store", "Thumbs.db", "coverage.xml"}


def find_disposable(root: Path) -> list[Path]:
    """Return a non-overlapping, deterministic list of safe cleanup targets."""
    root = root.resolve()
    targets: set[Path] = set()

    for child in root.iterdir():
        if (
            child.name in ROOT_NAMES
            or child.name.startswith(ROOT_PREFIXES)
            or child.name.endswith(".egg-info")
        ):
            targets.add(child)

    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            directories[:] = [
                name for name in directories
                if name not in PROTECTED_TOP_LEVEL and root / name not in targets
            ]

        disposable_dirs = [name for name in directories if name in RECURSIVE_DIR_NAMES]
        for name in disposable_dirs:
            targets.add(current_path / name)
        directories[:] = [name for name in directories if name not in disposable_dirs]

        for name in files:
            if name in DISPOSABLE_FILES or name.startswith(".coverage.") or Path(name).suffix in {".pyc", ".pyo"}:
                targets.add(current_path / name)

    return sorted(targets, key=lambda path: path.relative_to(root).as_posix().lower())


def remove_disposable(root: Path, *, apply: bool = False) -> tuple[list[Path], list[tuple[Path, str]]]:
    targets = find_disposable(root)
    failures: list[tuple[Path, str]] = []
    if not apply:
        return targets, failures

    for path in targets:
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        except OSError as error:
            failures.append((path, str(error)))
    return targets, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove disposable MorphoVoxel development artifacts")
    parser.add_argument("--apply", action="store_true", help="delete the listed paths; the default is a dry run")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "morphovoxel").is_dir():
        parser.error("run this command from the MorphoVoxel repository root")

    targets, failures = remove_disposable(root, apply=args.apply)
    action = "Removed" if args.apply else "Would remove"
    for path in targets:
        print(f"{action}: {path.relative_to(root)}")
    print(f"{action} {len(targets)} disposable path(s).")

    if not args.apply and targets:
        print(r"Run again with --apply to delete them.")
    if failures:
        for path, error in failures:
            print(f"Could not remove {path.relative_to(root)}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
