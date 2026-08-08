"""Phase 3 command."""
import argparse
import logging

from .config import load_config
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    print(train(config, dimensions=int(config.get("dimensions", 3)), conditional=True))


if __name__ == "__main__":
    main()

