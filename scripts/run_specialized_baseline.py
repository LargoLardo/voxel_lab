"""Train one independent model per target morphology."""
from __future__ import annotations

import argparse

from morphovoxel.config import load_config
from morphovoxel.genomes import MORPHOLOGIES
from morphovoxel.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    base = load_config(args.config)
    for kind in MORPHOLOGIES:
        config = {**base, "run_name": f"{base.get('run_name', 'specialized')}_{kind}", "target_kind": kind}
        train(config, dimensions=3, conditional=False)


if __name__ == "__main__":
    main()

