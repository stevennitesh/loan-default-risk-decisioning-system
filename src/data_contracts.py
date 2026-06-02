from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from src.config import is_post_v1_scope
from src.mart_access import (
    duplicate_key_count,
    existing_tables,
    fetch_count,
    table_columns,
)
from src.report_contracts import DATA_INVENTORY_COLUMNS, FEATURE_INVENTORY_COLUMNS
from src.runtime import created_at_utc as current_created_at_utc
from src.runtime import ensure_directories, sql_identifier, write_csv

ALLOWED_SOURCE_POPULATIONS = {"application_train", "application_test"}
MART_TABLE = "mart_credit_risk_features"
DIAGNOSTIC_TABLE = "segment_diagnostics"
DIAGNOSTIC_ONLY_COLUMNS = {
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "applicant_age_years",
    "applicant_age_band",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}
ALWAYS_EXCLUDED_COLUMNS = {
    "source_population": "metadata",
}


@dataclass(frozen=True)
class TableContract:
    table_name: str
    layer: str
    grain_columns: tuple[str, ...]

    @property
    def grain_key(self) -> str:
        return ",".join(self.grain_columns)


V1_REQUIRED_TABLES = [
    TableContract("stg_application_train", "staging", ("SK_ID_CURR",)),
    TableContract("stg_application_test", "staging", ("SK_ID_CURR",)),
    TableContract("stg_bureau", "staging", ("SK_ID_BUREAU",)),
    TableContract("stg_previous_application", "staging", ("SK_ID_PREV",)),
    TableContract("stg_installments_payments", "staging", ("SK_ID_CURR",)),
    TableContract("f_applicant_static", "feature", ("SK_ID_CURR", "source_population")),
    TableContract(DIAGNOSTIC_TABLE, "diagnostic", ("SK_ID_CURR", "source_population")),
    TableContract("f_bureau_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_previous_application_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_installments_agg", "feature", ("SK_ID_CURR",)),
    TableContract(MART_TABLE, "mart", ("SK_ID_CURR", "source_population")),
]

POST_V1_EXTRA_REQUIRED_TABLES = [
    TableContract("stg_bureau_balance", "staging", ("SK_ID_BUREAU", "MONTHS_BALANCE")),
    TableContract("stg_pos_cash_balance", "staging", ("SK_ID_PREV", "MONTHS_BALANCE")),
    TableContract(
        "stg_credit_card_balance", "staging", ("SK_ID_PREV", "MONTHS_BALANCE")
    ),
    TableContract("f_bureau_balance_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_pos_cash_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_credit_card_agg", "feature", ("SK_ID_CURR",)),
]

REQUIRED_TABLES = [*V1_REQUIRED_TABLES, *POST_V1_EXTRA_REQUIRED_TABLES]

V1_FEATURE_INVENTORY_TABLES = [
    "f_applicant_static",
    DIAGNOSTIC_TABLE,
    "f_bureau_agg",
    "f_previous_application_agg",
    "f_installments_agg",
    MART_TABLE,
]

POST_V1_EXTRA_FEATURE_INVENTORY_TABLES = [
    "f_bureau_balance_agg",
    "f_pos_cash_agg",
    "f_credit_card_agg",
]

FEATURE_INVENTORY_TABLES = [
    *V1_FEATURE_INVENTORY_TABLES[:-1],
    *POST_V1_EXTRA_FEATURE_INVENTORY_TABLES,
    MART_TABLE,
]

V1_AGGREGATE_TABLES = [
    "f_bureau_agg",
    "f_previous_application_agg",
    "f_installments_agg",
]

POST_V1_EXTRA_AGGREGATE_TABLES = [
    "f_bureau_balance_agg",
    "f_pos_cash_agg",
    "f_credit_card_agg",
]

AGGREGATE_TABLES = [
    *V1_AGGREGATE_TABLES[:1],
    *POST_V1_EXTRA_AGGREGATE_TABLES,
    *V1_AGGREGATE_TABLES[1:],
]


class DataContractError(RuntimeError):
    """Raised when the feature mart violates the pre-model data contract."""


def validate_data_contracts(
    connection: duckdb.DuckDBPyConnection, config: dict[str, Any]
) -> None:
    available_tables = existing_tables(connection)
    missing_tables = sorted(
        table.table_name
        for table in _required_tables(config)
        if table.table_name not in available_tables
    )
    if missing_tables:
        raise DataContractError(
            f"Missing required DuckDB tables: {', '.join(missing_tables)}"
        )

    # Collect all contract failures before raising so one pipeline run gives a useful fix list.
    errors: list[str] = []
    _validate_mart_contract(connection, errors)
    _validate_applicant_row_reconciliation(connection, errors)
    _validate_aggregate_keys(connection, config, errors)
    _validate_diagnostic_separation(connection, config, errors)
    _validate_model_feature_quality(connection, config, errors)

    if errors:
        raise DataContractError("Data contract validation failed: " + "; ".join(errors))


def get_model_feature_columns(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
) -> list[str]:
    exclusion_groups = _exclusion_group_map(config)
    # The mart is the only modeling surface; diagnostics stay separate even when available in DuckDB.
    return [
        column_name
        for column_name in table_columns(connection, MART_TABLE)
        if column_name not in exclusion_groups
    ]


def build_data_inventory(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = created_at_utc or current_created_at_utc()
    rows: list[dict[str, Any]] = []

    for table in _required_tables(config):
        columns = table_columns(connection, table.table_name)
        has_target_column = "TARGET" in columns
        rows.append(
            {
                "table_name": table.table_name,
                "layer": table.layer,
                "grain_key": table.grain_key,
                "row_count": _fetch_count(
                    connection,
                    f"SELECT COUNT(*) FROM {sql_identifier(table.table_name)}",
                ),
                "distinct_applicant_count": _distinct_applicant_count(
                    connection,
                    table.table_name,
                    columns,
                ),
                "duplicate_grain_key_count": duplicate_key_count(
                    connection,
                    table.table_name,
                    table.grain_columns,
                    DataContractError,
                ),
                "has_target_column": has_target_column,
                "target_non_null_count": _target_count(
                    connection, table.table_name, "IS NOT NULL"
                )
                if has_target_column
                else 0,
                "target_null_count": _target_count(
                    connection, table.table_name, "IS NULL"
                )
                if has_target_column
                else 0,
                "created_at_utc": timestamp,
            }
        )
    return rows


def build_feature_inventory(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    created_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = created_at_utc or current_created_at_utc()
    exclusion_groups = _exclusion_group_map(config)
    model_features = set(get_model_feature_columns(connection, config))
    rows: list[dict[str, Any]] = []

    for table_name in _feature_inventory_tables(config):
        row_count = _fetch_count(
            connection, f"SELECT COUNT(*) FROM {sql_identifier(table_name)}"
        )
        for column_name, duckdb_type in table_columns(connection, table_name).items():
            missing_count = _fetch_count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM {sql_identifier(table_name)}
                WHERE {sql_identifier(column_name)} IS NULL
                """,
            )
            rows.append(
                {
                    "table_name": table_name,
                    "column_name": column_name,
                    "duckdb_type": duckdb_type,
                    "is_model_feature": table_name == MART_TABLE
                    and column_name in model_features,
                    "exclusion_group": exclusion_groups.get(column_name, ""),
                    "missing_count": missing_count,
                    "missing_rate": missing_count / row_count if row_count else None,
                    "distinct_value_count": _fetch_count(
                        connection,
                        f"""
                        SELECT COUNT(DISTINCT {sql_identifier(column_name)})
                        FROM {sql_identifier(table_name)}
                        """,
                    ),
                    "created_at_utc": timestamp,
                }
            )
    return rows


def write_contract_reports(
    report_dir: str | Path,
    data_inventory_rows: list[dict[str, Any]],
    feature_inventory_rows: list[dict[str, Any]],
) -> None:
    report_path = Path(report_dir)
    ensure_directories(report_path)
    write_csv(
        report_path / "data_inventory.csv", DATA_INVENTORY_COLUMNS, data_inventory_rows
    )
    write_csv(
        report_path / "feature_inventory.csv",
        FEATURE_INVENTORY_COLUMNS,
        feature_inventory_rows,
    )


def _distinct_applicant_count(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: dict[str, str],
) -> int | None:
    if "SK_ID_CURR" not in columns:
        return None
    return _fetch_count(
        connection,
        f"SELECT COUNT(DISTINCT SK_ID_CURR) FROM {sql_identifier(table_name)}",
    )


def _validate_mart_contract(
    connection: duckdb.DuckDBPyConnection, errors: list[str]
) -> None:
    mart_columns = set(table_columns(connection, MART_TABLE))
    missing_columns = {"SK_ID_CURR", "source_population", "TARGET"}.difference(
        mart_columns
    )
    if missing_columns:
        errors.append(
            f"mart_credit_risk_features is missing required columns: {sorted(missing_columns)}"
        )
        return

    source_populations = {
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT source_population FROM {sql_identifier(MART_TABLE)}"
        ).fetchall()
    }
    unexpected_populations = sorted(
        source_populations.difference(ALLOWED_SOURCE_POPULATIONS)
    )
    if unexpected_populations:
        errors.append(f"Unexpected source_population values: {unexpected_populations}")

    duplicate_keys = duplicate_key_count(
        connection,
        MART_TABLE,
        ("SK_ID_CURR", "source_population"),
        DataContractError,
    )
    if duplicate_keys:
        errors.append(f"Duplicate mart grain keys: {duplicate_keys}")

    train_null_targets = _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {sql_identifier(MART_TABLE)}
        WHERE source_population = 'application_train'
          AND TARGET IS NULL
        """,
    )
    if train_null_targets:
        errors.append(
            f"application_train rows must have non-null TARGET: {train_null_targets}"
        )

    train_non_binary_targets = _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {sql_identifier(MART_TABLE)}
        WHERE source_population = 'application_train'
          AND TARGET IS NOT NULL
          AND TARGET NOT IN (0, 1)
        """,
    )
    if train_non_binary_targets:
        errors.append(
            f"application_train TARGET values must be binary: {train_non_binary_targets}"
        )

    test_non_null_targets = _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {sql_identifier(MART_TABLE)}
        WHERE source_population = 'application_test'
          AND TARGET IS NOT NULL
        """,
    )
    if test_non_null_targets:
        errors.append(
            f"application_test rows must have NULL TARGET: {test_non_null_targets}"
        )


def _validate_applicant_row_reconciliation(
    connection: duckdb.DuckDBPyConnection,
    errors: list[str],
) -> None:
    for source_population, staging_table in [
        ("application_train", "stg_application_train"),
        ("application_test", "stg_application_test"),
    ]:
        staging_rows = _fetch_count(
            connection,
            f"SELECT COUNT(*) FROM {sql_identifier(staging_table)}",
        )
        mart_rows = _fetch_count(
            connection,
            f"""
            SELECT COUNT(*)
            FROM {sql_identifier(MART_TABLE)}
            WHERE source_population = '{source_population}'
            """,
        )
        if staging_rows != mart_rows:
            errors.append(
                f"{source_population} staging rows do not reconcile to mart rows: "
                f"staging={staging_rows}, mart={mart_rows}"
            )


def _required_tables(config: dict[str, Any] | None) -> list[TableContract]:
    # v1 remains reproducible after post-v1 tables were added, so table requirements are scope-aware.
    if config is not None and not is_post_v1_scope(config):
        return V1_REQUIRED_TABLES
    return REQUIRED_TABLES


def _feature_inventory_tables(config: dict[str, Any]) -> list[str]:
    if not is_post_v1_scope(config):
        return V1_FEATURE_INVENTORY_TABLES
    return FEATURE_INVENTORY_TABLES


def _aggregate_tables(config: dict[str, Any]) -> list[str]:
    if not is_post_v1_scope(config):
        return V1_AGGREGATE_TABLES
    return AGGREGATE_TABLES


def _validate_aggregate_keys(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    for table_name in _aggregate_tables(config):
        duplicate_keys = duplicate_key_count(
            connection,
            table_name,
            ("SK_ID_CURR",),
            DataContractError,
        )
        if duplicate_keys:
            errors.append(
                f"{table_name} has duplicate SK_ID_CURR keys: {duplicate_keys}"
            )


def _validate_diagnostic_separation(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    mart_columns = set(table_columns(connection, MART_TABLE))
    diagnostic_columns = set(table_columns(connection, DIAGNOSTIC_TABLE))

    sensitive_columns = set(
        config["excluded_features"]["sensitive_or_protected_status_like"]
    )
    sensitive_columns_in_mart = sorted(sensitive_columns.intersection(mart_columns))
    if sensitive_columns_in_mart:
        errors.append(
            f"Diagnostic-only columns are present in the model mart: {sensitive_columns_in_mart}"
        )

    missing_diagnostic_columns = sorted(
        DIAGNOSTIC_ONLY_COLUMNS.difference(diagnostic_columns)
    )
    if missing_diagnostic_columns:
        errors.append(
            f"segment_diagnostics is missing diagnostic columns: {missing_diagnostic_columns}"
        )


def _validate_model_feature_quality(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    all_missing_features = []
    row_count = _fetch_count(
        connection, f"SELECT COUNT(*) FROM {sql_identifier(MART_TABLE)}"
    )
    if row_count == 0:
        errors.append("mart_credit_risk_features must not be empty")
        return

    for column_name in get_model_feature_columns(connection, config):
        non_null_count = _fetch_count(
            connection,
            f"""
            SELECT COUNT(*)
            FROM {sql_identifier(MART_TABLE)}
            WHERE {sql_identifier(column_name)} IS NOT NULL
            """,
        )
        if non_null_count == 0:
            all_missing_features.append(column_name)

    if all_missing_features:
        errors.append(
            f"Model feature columns are 100% missing: {sorted(all_missing_features)}"
        )


def _exclusion_group_map(config: dict[str, Any]) -> dict[str, str]:
    exclusions = dict(ALWAYS_EXCLUDED_COLUMNS)
    for group_name, column_names in config["excluded_features"].items():
        for column_name in column_names:
            exclusions[column_name] = group_name
    return exclusions


def _target_count(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    null_predicate: str,
) -> int:
    return _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {sql_identifier(table_name)}
        WHERE TARGET {null_predicate}
        """,
    )


def _fetch_count(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    return fetch_count(connection, sql, DataContractError)
