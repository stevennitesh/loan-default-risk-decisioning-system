from __future__ import annotations

import argparse
from typing import NoReturn

from src.config import DEFAULT_CONFIG_PATH

CONFIG_HELP = "Path to the project config file."


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=CONFIG_HELP)


def format_int_csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def exit_with_error(error: Exception) -> NoReturn:
    raise SystemExit(str(error)) from error
