"""Aggregate run summaries without inventing missing results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output", default="runs/summary.csv")
    args = parser.parse_args()
    frames = []
    for path in Path(args.runs_root).glob("*/metrics/summary.csv"):
        frame = pd.read_csv(path)
        frame.insert(0, "run", path.parents[1].name)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no run summary files found")
    pd.concat(frames, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

