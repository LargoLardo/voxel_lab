"""Run a matrix of matched ecological scenarios."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        matrix = yaml.safe_load(stream)
    for name, overrides in matrix["scenarios"].items():
        config = {**matrix["base"], **overrides, "run_name": f"ecology_{name}"}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as stream:
            yaml.safe_dump(config, stream)
            path = Path(stream.name)
        try:
            subprocess.run([sys.executable, "-m", "morphovoxel.run_ecology", "--config", str(path)], check=True)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
