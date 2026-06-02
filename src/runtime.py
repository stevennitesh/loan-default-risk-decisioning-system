from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_config_path(config: dict[str, Any], path_key: str) -> Path:
    return resolve_project_path(config["paths"][path_key])


def require_existing_path(path: Path, label: str, error_cls: type[Exception]) -> None:
    if not path.exists():
        raise error_cls(f"{label} not found: {path}")


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def created_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_utc_datetime() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    return features.where(pd.notna(features), np.nan)


def replace_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    replace_duckdb_table_from_frame(connection, table_name, frame)


def replace_duckdb_table_from_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    connection.register("output_frame", frame)
    try:
        connection.execute(f"CREATE OR REPLACE TABLE {sql_identifier(table_name)} AS SELECT * FROM output_frame")
    finally:
        connection.unregister("output_frame")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, fieldnames: list[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if fieldnames is not None and reader.fieldnames != fieldnames:
            raise ValueError(f"Unexpected CSV columns for {path}: {reader.fieldnames}")
        return list(reader)


def sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


def sql_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) + chr(39))}'"
