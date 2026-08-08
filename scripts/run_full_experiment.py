"""Run the configured experiment commands in order and stop on failure."""
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
        config = yaml.safe_load(stream)
    overrides = config.get("overrides", {})
    for module, module_config in config["commands"]:
        temporary = None
        if overrides:
            with open(module_config, encoding="utf-8") as stream:
                module_values = yaml.safe_load(stream) or {}
            module_values.update(overrides)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as stream:
                yaml.safe_dump(module_values, stream)
                temporary = Path(stream.name)
            module_config = str(temporary)
        try:
            subprocess.run([sys.executable, "-m", module, "--config", module_config], check=True)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
