from __future__ import annotations

import argparse

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and primary credit-risk models.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    load_config(args.config)
    raise SystemExit("Milestone 4 not implemented yet: modeling is blocked until feature gates pass.")


if __name__ == "__main__":
    main()
