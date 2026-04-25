from __future__ import annotations

import argparse

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance and reason-code-style outputs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    load_config(args.config)
    raise SystemExit("Milestone 9 not implemented yet: explainability requires trained model artifacts.")


if __name__ == "__main__":
    main()
