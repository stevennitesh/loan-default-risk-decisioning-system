from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from src.config import SUPPORTED_SOURCE_FILES
from src.config import load_config
from src.report_contracts import INGESTION_SUMMARY_COLUMNS
from src.runtime import REPO_ROOT
from src.runtime import created_at_utc
from src.runtime import resolve_project_path
from src.runtime import sql_identifier
from src.runtime import write_csv

STAGING_TABLES = {
    "application_train": "stg_application_train",
    "application_test": "stg_application_test",
    "bureau": "stg_bureau",
    "bureau_balance": "stg_bureau_balance",
    "pos_cash_balance": "stg_pos_cash_balance",
    "credit_card_balance": "stg_credit_card_balance",
    "previous_application": "stg_previous_application",
    "installments_payments": "stg_installments_payments",
}

class IngestionError(RuntimeError):
    """Raised when ingestion cannot satisfy the Milestone 1 contract."""


def run_ingestion(config_path: str | Path = "configs/base.yaml") -> list[dict[str, Any]]:
    config = load_config(config_path)
    paths = config["paths"]

    raw_dir = resolve_project_path(paths["raw_dir"])
    parquet_dir = resolve_project_path(paths["parquet_dir"])
    duckdb_path = resolve_project_path(paths["duckdb_path"])
    report_dir = resolve_project_path(paths["report_dir"])

    raw_files = {
        source_name: raw_dir / source_file
        for source_name, source_file in config["source_files"].items()
    }
    missing_files = [path.name for path in raw_files.values() if not path.exists()]
    if missing_files:
        missing_display = ", ".join(sorted(missing_files))
        raise IngestionError(f"Missing required raw CSV files: {missing_display}")

    parquet_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ingestion_created_at = created_at_utc()
    summary_rows: list[dict[str, Any]] = []

    with duckdb.connect(str(duckdb_path)) as connection:
        for source_name, source_file in config["source_files"].items():
            _validate_source_name(source_name)

            raw_path = raw_files[source_name]
            parquet_path = parquet_dir / f"{source_name}.parquet"
            staging_table = STAGING_TABLES[source_name]

            csv_rows = _fetch_count(connection, f"SELECT COUNT(*) FROM read_csv_auto({_sql_path(raw_path)})")
            if parquet_path.exists():
                parquet_path.unlink()
            connection.execute(
                f"COPY (SELECT * FROM read_csv_auto({_sql_path(raw_path)})) "
                f"TO {_sql_path(parquet_path)} (FORMAT PARQUET)"
            )
            parquet_rows = _fetch_count(
                connection,
                f"SELECT COUNT(*) FROM read_parquet({_sql_path(parquet_path)})",
            )
            connection.execute(
                f"CREATE OR REPLACE TABLE {sql_identifier(staging_table)} AS "
                f"SELECT * FROM read_parquet({_sql_path(parquet_path)})"
            )
            duckdb_rows = _fetch_count(
                connection,
                f"SELECT COUNT(*) FROM {sql_identifier(staging_table)}",
            )

            summary_rows.append(
                {
                    "source_name": source_name,
                    "source_file": source_file,
                    "raw_path": _display_path(raw_path),
                    "parquet_path": _display_path(parquet_path),
                    "staging_table": staging_table,
                    "csv_rows": csv_rows,
                    "parquet_rows": parquet_rows,
                    "duckdb_rows": duckdb_rows,
                    "created_at_utc": ingestion_created_at,
                }
            )

    write_csv(report_dir / "ingestion_summary.csv", INGESTION_SUMMARY_COLUMNS, summary_rows)
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CSV files to Parquet and DuckDB staging tables.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    try:
        run_ingestion(args.config)
    except IngestionError as error:
        raise SystemExit(str(error)) from error


def _validate_source_name(source_name: str) -> None:
    if source_name not in SUPPORTED_SOURCE_FILES:
        raise IngestionError(f"Unsupported source file key: {source_name}")
    if source_name not in STAGING_TABLES:
        raise IngestionError(f"No staging table configured for source file key: {source_name}")


def _sql_path(path: Path) -> str:
    escaped = path.resolve().as_posix().replace("'", "''")
    return f"'{escaped}'"


def _fetch_count(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    result = connection.execute(sql).fetchone()
    if result is None:
        raise IngestionError(f"Count query returned no rows: {sql}")
    return int(result[0])


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
