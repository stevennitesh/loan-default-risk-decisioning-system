from __future__ import annotations

import argparse
from typing import NoReturn

from src.config import DEFAULT_CONFIG_PATH

CONFIG_HELP = "Path to the project config file."


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared --config option to a command parser."""
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=CONFIG_HELP)


def format_int_csv(values: tuple[int, ...]) -> str:
    """Format integer CLI defaults as comma-separated values."""
    return ",".join(str(value) for value in values)


def parse_int_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer CLI argument."""
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def exit_with_error(error: Exception) -> NoReturn:
    """Exit a CLI command with the upstream validation error message."""
    raise SystemExit(str(error)) from error
