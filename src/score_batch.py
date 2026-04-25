from __future__ import annotations

import argparse

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Score applicants in batch and write DuckDB score outputs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    load_config(args.config)
    raise SystemExit("Milestone 8 not implemented yet: scoring requires trained model artifacts.")


if __name__ == "__main__":
    main()
