"""Metric plotting."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(csv_path: str | Path, output: str | Path) -> None:
    data = pd.read_csv(csv_path)
    numeric = [column for column in data.columns if column != "step" and pd.api.types.is_numeric_dtype(data[column])]
    axis = data.plot(x="step", y=numeric, figsize=(7, 4))
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=140)
    plt.close(axis.figure)
