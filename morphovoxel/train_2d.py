"""Phase 1 command."""
import argparse
import logging

from .config import load_config
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(train(load_config(args.config), dimensions=2))


if __name__ == "__main__":
    main()

