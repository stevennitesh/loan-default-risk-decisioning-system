from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from src.config import is_post_v1_scope
from src.config import load_config
from src.ingest import STAGING_TABLES
from src.mart_access import existing_tables
from src.mart_access import table_columns
from src.report_contracts import FEATURE_PROFILE_COLUMNS
from src.data_contracts import DataContractError
from src.data_contracts import build_data_inventory
from src.data_contracts import build_feature_inventory
from src.data_contracts import validate_data_contracts
from src.data_contracts import write_contract_reports
from src.runtime import REPO_ROOT
from src.runtime import created_at_utc
from src.runtime import resolve_project_path
from src.runtime import write_csv

V1_FEATURE_SQL_FILES = [
    "sql/02_feature_applicant.sql",
    "sql/03_feature_bureau.sql",
    "sql/04_feature_previous_applications.sql",
    "sql/05_feature_installments.sql",
    "sql/06_build_feature_mart_v1.sql",
]

POST_V1_FEATURE_SQL_FILES = [
    "sql/02_feature_applicant.sql",
    "sql/03_feature_bureau.sql",
    "sql/03b_feature_bureau_balance.sql",
    "sql/04_feature_previous_applications.sql",
    "sql/04b_feature_pos_cash.sql",
    "sql/04c_feature_credit_card.sql",
    "sql/05_feature_installments.sql",
    "sql/05b_feature_risk_pressure.sql",
    "sql/05c_feature_recency_deterioration.sql",
    "sql/05d_feature_last_k_temporal.sql",
    "sql/06_build_feature_mart.sql",
]
FEATURE_SQL_FILES = POST_V1_FEATURE_SQL_FILES

V1_PROFILE_TABLES = [
    "f_applicant_static",
    "segment_diagnostics",
    "f_bureau_agg",
    "f_previous_application_agg",
    "f_installments_agg",
    "mart_credit_risk_features",
]

POST_V1_PROFILE_TABLES = [
    "f_applicant_static",
    "segment_diagnostics",
    "f_bureau_agg",
    "f_bureau_balance_agg",
    "f_pos_cash_agg",
    "f_credit_card_agg",
    "f_previous_application_agg",
    "f_installments_agg",
    "f_risk_pressure_features",
    "f_recency_deterioration_features",
    "f_last_k_temporal_features",
    "mart_credit_risk_features",
]
PROFILE_TABLES = POST_V1_PROFILE_TABLES

class FeatureBuildError(RuntimeError):
    """Raised when feature building cannot satisfy the Milestone 2 contract."""


def run_feature_build(config_path: str | Path = "configs/base.yaml") -> list[dict[str, Any]]:
    config = load_config(config_path)
    duckdb_path = resolve_project_path(config["paths"]["duckdb_path"])
    report_dir = resolve_project_path(config["paths"]["report_dir"])

    report_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(duckdb_path)) as connection:
        _ensure_staging_tables(connection, config)
        for sql_file in _feature_sql_files(config):
            sql_path = REPO_ROOT / sql_file
            connection.execute(sql_path.read_text(encoding="utf-8"))

        profile_rows = _profile_feature_tables(connection, config)
        _write_profile(report_dir / "feature_mart_profile.csv", profile_rows)
        validate_data_contracts(connection, config)
        write_contract_reports(
            report_dir,
            build_data_inventory(connection, config),
            build_feature_inventory(connection, config),
        )

    return profile_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQL feature tables and the final feature mart.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    try:
        run_feature_build(args.config)
    except (FeatureBuildError, DataContractError) as error:
        raise SystemExit(str(error)) from error


def _feature_sql_files(config: dict[str, Any]) -> list[str]:
    return POST_V1_FEATURE_SQL_FILES if is_post_v1_scope(config) else V1_FEATURE_SQL_FILES


def _profile_tables(config: dict[str, Any]) -> list[str]:
    return POST_V1_PROFILE_TABLES if is_post_v1_scope(config) else V1_PROFILE_TABLES


def _required_staging_tables(config: dict[str, Any]) -> list[str]:
    return [STAGING_TABLES[source_name] for source_name in config["source_files"]]


def _ensure_staging_tables(connection: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> None:
    available_tables = existing_tables(connection)
    missing_tables = sorted(set(_required_staging_tables(config)).difference(available_tables))
    if missing_tables:
        raise FeatureBuildError(f"Missing required staging tables: {', '.join(missing_tables)}")


def _profile_feature_tables(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    profile_created_at = created_at_utc()
    rows: list[dict[str, Any]] = []
    available_tables = existing_tables(connection)
    profile_tables = _profile_tables(config)

    missing_feature_tables = sorted(set(profile_tables).difference(available_tables))
    if missing_feature_tables:
        raise FeatureBuildError(f"Missing feature output tables: {', '.join(missing_feature_tables)}")

    for table_name in profile_tables:
        columns = list(table_columns(connection, table_name))
        if "SK_ID_CURR" not in columns:
            raise FeatureBuildError(f"Feature table is missing SK_ID_CURR: {table_name}")

        rows.append(
            {
                "table_name": table_name,
                "row_count": _fetch_count(connection, f'SELECT COUNT(*) FROM "{table_name}"'),
                "distinct_applicant_count": _fetch_count(
                    connection,
                    f'SELECT COUNT(DISTINCT SK_ID_CURR) FROM "{table_name}"',
                ),
                "duplicate_key_count": _duplicate_key_count(connection, table_name, columns),
                "column_count": len(columns),
                "created_at_utc": profile_created_at,
            }
        )
    return rows


def _duplicate_key_count(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: list[str],
) -> int:
    if "source_population" in columns:
        key_columns = "SK_ID_CURR, source_population"
    else:
        key_columns = "SK_ID_CURR"
    return _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {key_columns}
            FROM "{table_name}"
            GROUP BY {key_columns}
            HAVING COUNT(*) > 1
        )
        """,
    )


def _fetch_count(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    result = connection.execute(sql).fetchone()
    if result is None:
        raise FeatureBuildError(f"Count query returned no rows: {sql}")
    return int(result[0])


def _write_profile(profile_path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(profile_path, FEATURE_PROFILE_COLUMNS, rows)


if __name__ == "__main__":
    main()
