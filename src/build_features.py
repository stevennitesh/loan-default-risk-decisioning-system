from __future__ import annotations

import argparse

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQL feature tables and the final feature mart.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    load_config(args.config)
    raise SystemExit("Milestone 2 not implemented yet: feature building is blocked until ingestion exists.")


if __name__ == "__main__":
    main()
