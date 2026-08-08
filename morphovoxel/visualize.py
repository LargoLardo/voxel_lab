"""Re-render a saved run without retraining."""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    frames = np.load(run / "rollouts" / "states.npz")["states"]
    if frames.ndim == 5:
        from .rendering_2d import save_gif
    elif frames.ndim == 6:
        from .rendering_3d import save_gif
    else:
        raise ValueError("saved states have an unsupported shape")
    save_gif(frames, run / "visualizations" / "growth.gif")
    print(run / "visualizations" / "growth.gif")


if __name__ == "__main__":
    main()

