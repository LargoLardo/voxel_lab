"""Populate the research report only from saved run files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SECTIONS = ["Normal-growth results", "Conditional-growth results", "Regeneration results", "Scaling results", "Ecology results"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="reports/experiment_report.md")
    args = parser.parse_args()
    run, output = Path(args.run_dir), Path(args.output)
    summary = run / "metrics" / "summary.csv"
    result = f"```csv\n{pd.read_csv(summary).to_csv(index=False).strip()}\n```" if summary.exists() else "The experiment has not yet been run."
    text = (Path("reports/experiment_report.md").read_text(encoding="utf-8") if Path("reports/experiment_report.md").exists() else "# MorphoVoxel Experiment Report\n")
    text += f"\n\n## Run: {run.name}\n\n{result}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
