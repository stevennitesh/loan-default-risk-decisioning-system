from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from src.cli import add_config_argument, exit_with_error
from src.config import DEFAULT_CONFIG_PATH, SUPPORTED_SOURCE_FILES, load_config
from src.mart_access import fetch_count
from src.report_contracts import INGESTION_SUMMARY_COLUMNS
from src.runtime import (
    REPO_ROOT,
    created_at_utc,
    ensure_directories,
    resolve_config_path,
    sql_identifier,
    write_csv,
)

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


def run_ingestion(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Convert configured raw CSV files to Parquet and DuckDB staging tables."""
    config = load_config(config_path)

    raw_dir = resolve_config_path(config, "raw_dir")
    parquet_dir = resolve_config_path(config, "parquet_dir")
    duckdb_path = resolve_config_path(config, "duckdb_path")
    report_dir = resolve_config_path(config, "report_dir")

    raw_files = {
        source_name: raw_dir / source_file
        for source_name, source_file in config["source_files"].items()
    }
    missing_files = [path.name for path in raw_files.values() if not path.exists()]
    if missing_files:
        missing_display = ", ".join(sorted(missing_files))
        raise IngestionError(f"Missing required raw CSV files: {missing_display}")

    ensure_directories(parquet_dir, duckdb_path.parent, report_dir)

    ingestion_created_at = created_at_utc()
    summary_rows: list[dict[str, Any]] = []

    with duckdb.connect(str(duckdb_path)) as connection:
        for source_name, source_file in config["source_files"].items():
            _validate_source_name(source_name)

            raw_path = raw_files[source_name]
            parquet_path = parquet_dir / f"{source_name}.parquet"
            staging_table = STAGING_TABLES[source_name]

            csv_rows = fetch_count(
                connection,
                f"SELECT COUNT(*) FROM read_csv_auto({_sql_path(raw_path)})",
                IngestionError,
            )
            if parquet_path.exists():
                parquet_path.unlink()
            connection.execute(
                f"COPY (SELECT * FROM read_csv_auto({_sql_path(raw_path)})) "
                f"TO {_sql_path(parquet_path)} (FORMAT PARQUET)"
            )
            parquet_rows = fetch_count(
                connection,
                f"SELECT COUNT(*) FROM read_parquet({_sql_path(parquet_path)})",
                IngestionError,
            )
            connection.execute(
                f"CREATE OR REPLACE TABLE {sql_identifier(staging_table)} AS "
                f"SELECT * FROM read_parquet({_sql_path(parquet_path)})"
            )
            duckdb_rows = fetch_count(
                connection,
                f"SELECT COUNT(*) FROM {sql_identifier(staging_table)}",
                IngestionError,
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

    write_csv(
        report_dir / "ingestion_summary.csv", INGESTION_SUMMARY_COLUMNS, summary_rows
    )
    return summary_rows


def _validate_source_name(source_name: str) -> None:
    """Validate that a configured source name has a supported staging table."""
    if source_name not in SUPPORTED_SOURCE_FILES:
        raise IngestionError(f"Unsupported source file key: {source_name}")
    if source_name not in STAGING_TABLES:
        raise IngestionError(
            f"No staging table configured for source file key: {source_name}"
        )


def _sql_path(path: Path) -> str:
    """Return a SQL-safe absolute path literal for DuckDB file functions."""
    escaped = path.resolve().as_posix().replace("'", "''")
    return f"'{escaped}'"


def _display_path(path: Path) -> str:
    """Return a repo-relative path when possible for inventory reports."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    """Run the ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Convert raw CSV files to Parquet and DuckDB staging tables."
    )
    add_config_argument(parser)
    args = parser.parse_args()

    try:
        run_ingestion(args.config)
    except IngestionError as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
