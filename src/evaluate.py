from __future__ import annotations

import argparse

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model metrics and export reporting tables.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    parser.add_argument("--export-dashboard-data", action="store_true", help="Export Power BI-ready dashboard data.")
    args = parser.parse_args()

    load_config(args.config)
    if args.export_dashboard_data:
        raise SystemExit("Milestone 10 not implemented yet: dashboard exports require evaluation outputs.")
    raise SystemExit("Milestone 6 not implemented yet: evaluation requires trained model artifacts.")


if __name__ == "__main__":
    main()
