from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


DATA_INVENTORY_COLUMNS = [
    "table_name",
    "layer",
    "grain_key",
    "row_count",
    "distinct_applicant_count",
    "duplicate_grain_key_count",
    "has_target_column",
    "target_non_null_count",
    "target_null_count",
    "created_at_utc",
]

FEATURE_INVENTORY_COLUMNS = [
    "table_name",
    "column_name",
    "duckdb_type",
    "is_model_feature",
    "exclusion_group",
    "missing_count",
    "missing_rate",
    "distinct_value_count",
    "created_at_utc",
]

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


REQUIRED_TABLES = [
    TableContract("stg_application_train", "staging", ("SK_ID_CURR",)),
    TableContract("stg_application_test", "staging", ("SK_ID_CURR",)),
    TableContract("stg_bureau", "staging", ("SK_ID_BUREAU",)),
    TableContract("stg_bureau_balance", "staging", ("SK_ID_BUREAU", "MONTHS_BALANCE")),
    TableContract("stg_pos_cash_balance", "staging", ("SK_ID_PREV", "MONTHS_BALANCE")),
    TableContract("stg_credit_card_balance", "staging", ("SK_ID_PREV", "MONTHS_BALANCE")),
    TableContract("stg_previous_application", "staging", ("SK_ID_PREV",)),
    TableContract("stg_installments_payments", "staging", ("SK_ID_CURR",)),
    TableContract("f_applicant_static", "feature", ("SK_ID_CURR", "source_population")),
    TableContract(DIAGNOSTIC_TABLE, "diagnostic", ("SK_ID_CURR", "source_population")),
    TableContract("f_bureau_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_bureau_balance_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_pos_cash_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_credit_card_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_previous_application_agg", "feature", ("SK_ID_CURR",)),
    TableContract("f_installments_agg", "feature", ("SK_ID_CURR",)),
    TableContract(MART_TABLE, "mart", ("SK_ID_CURR", "source_population")),
]

FEATURE_INVENTORY_TABLES = [
    "f_applicant_static",
    DIAGNOSTIC_TABLE,
    "f_bureau_agg",
    "f_bureau_balance_agg",
    "f_pos_cash_agg",
    "f_credit_card_agg",
    "f_previous_application_agg",
    "f_installments_agg",
    MART_TABLE,
]

AGGREGATE_TABLES = [
    "f_bureau_agg",
    "f_bureau_balance_agg",
    "f_pos_cash_agg",
    "f_credit_card_agg",
    "f_previous_application_agg",
    "f_installments_agg",
]


class DataContractError(RuntimeError):
    """Raised when the feature mart violates the pre-model data contract."""


def validate_data_contracts(connection: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> None:
    existing_tables = _existing_tables(connection)
    missing_tables = sorted(
        table.table_name for table in REQUIRED_TABLES if table.table_name not in existing_tables
    )
    if missing_tables:
        raise DataContractError(f"Missing required DuckDB tables: {', '.join(missing_tables)}")

    errors: list[str] = []
    _validate_mart_contract(connection, errors)
    _validate_applicant_row_reconciliation(connection, errors)
    _validate_aggregate_keys(connection, errors)
    _validate_diagnostic_separation(connection, config, errors)
    _validate_model_feature_quality(connection, config, errors)

    if errors:
        raise DataContractError("Data contract validation failed: " + "; ".join(errors))


def get_model_feature_columns(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
) -> list[str]:
    exclusion_groups = _exclusion_group_map(config)
    return [
        column_name
        for column_name in _table_columns(connection, MART_TABLE)
        if column_name not in exclusion_groups
    ]


def build_data_inventory(
    connection: duckdb.DuckDBPyConnection,
    created_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = created_at_utc or _created_at_utc()
    rows: list[dict[str, Any]] = []

    for table in REQUIRED_TABLES:
        columns = _table_columns(connection, table.table_name)
        has_target_column = "TARGET" in columns
        rows.append(
            {
                "table_name": table.table_name,
                "layer": table.layer,
                "grain_key": table.grain_key,
                "row_count": _fetch_count(
                    connection,
                    f"SELECT COUNT(*) FROM {_sql_identifier(table.table_name)}",
                ),
                "distinct_applicant_count": _distinct_applicant_count(
                    connection,
                    table.table_name,
                    columns,
                ),
                "duplicate_grain_key_count": _duplicate_key_count(
                    connection,
                    table.table_name,
                    table.grain_columns,
                ),
                "has_target_column": has_target_column,
                "target_non_null_count": _target_count(connection, table.table_name, "IS NOT NULL")
                if has_target_column
                else 0,
                "target_null_count": _target_count(connection, table.table_name, "IS NULL")
                if has_target_column
                else 0,
                "created_at_utc": timestamp,
            }
        )
    return rows


def _distinct_applicant_count(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: dict[str, str],
) -> int | None:
    if "SK_ID_CURR" not in columns:
        return None
    return _fetch_count(
        connection,
        f"SELECT COUNT(DISTINCT SK_ID_CURR) FROM {_sql_identifier(table_name)}",
    )


def build_feature_inventory(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    created_at_utc: str | None = None,
) -> list[dict[str, Any]]:
    timestamp = created_at_utc or _created_at_utc()
    exclusion_groups = _exclusion_group_map(config)
    model_features = set(get_model_feature_columns(connection, config))
    rows: list[dict[str, Any]] = []

    for table_name in FEATURE_INVENTORY_TABLES:
        row_count = _fetch_count(connection, f"SELECT COUNT(*) FROM {_sql_identifier(table_name)}")
        for column_name, duckdb_type in _table_columns(connection, table_name).items():
            missing_count = _fetch_count(
                connection,
                f"""
                SELECT COUNT(*)
                FROM {_sql_identifier(table_name)}
                WHERE {_sql_identifier(column_name)} IS NULL
                """,
            )
            rows.append(
                {
                    "table_name": table_name,
                    "column_name": column_name,
                    "duckdb_type": duckdb_type,
                    "is_model_feature": table_name == MART_TABLE and column_name in model_features,
                    "exclusion_group": exclusion_groups.get(column_name, ""),
                    "missing_count": missing_count,
                    "missing_rate": missing_count / row_count if row_count else None,
                    "distinct_value_count": _fetch_count(
                        connection,
                        f"""
                        SELECT COUNT(DISTINCT {_sql_identifier(column_name)})
                        FROM {_sql_identifier(table_name)}
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
    report_path.mkdir(parents=True, exist_ok=True)
    _write_csv(report_path / "data_inventory.csv", DATA_INVENTORY_COLUMNS, data_inventory_rows)
    _write_csv(report_path / "feature_inventory.csv", FEATURE_INVENTORY_COLUMNS, feature_inventory_rows)


def _validate_mart_contract(connection: duckdb.DuckDBPyConnection, errors: list[str]) -> None:
    mart_columns = set(_table_columns(connection, MART_TABLE))
    missing_columns = {"SK_ID_CURR", "source_population", "TARGET"}.difference(mart_columns)
    if missing_columns:
        errors.append(f"mart_credit_risk_features is missing required columns: {sorted(missing_columns)}")
        return

    source_populations = {
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT source_population FROM {_sql_identifier(MART_TABLE)}"
        ).fetchall()
    }
    unexpected_populations = sorted(source_populations.difference(ALLOWED_SOURCE_POPULATIONS))
    if unexpected_populations:
        errors.append(f"Unexpected source_population values: {unexpected_populations}")

    duplicate_keys = _duplicate_key_count(connection, MART_TABLE, ("SK_ID_CURR", "source_population"))
    if duplicate_keys:
        errors.append(f"Duplicate mart grain keys: {duplicate_keys}")

    train_null_targets = _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {_sql_identifier(MART_TABLE)}
        WHERE source_population = 'application_train'
          AND TARGET IS NULL
        """,
    )
    if train_null_targets:
        errors.append(f"application_train rows must have non-null TARGET: {train_null_targets}")

    train_non_binary_targets = _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {_sql_identifier(MART_TABLE)}
        WHERE source_population = 'application_train'
          AND TARGET IS NOT NULL
          AND TARGET NOT IN (0, 1)
        """,
    )
    if train_non_binary_targets:
        errors.append(f"application_train TARGET values must be binary: {train_non_binary_targets}")

    test_non_null_targets = _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {_sql_identifier(MART_TABLE)}
        WHERE source_population = 'application_test'
          AND TARGET IS NOT NULL
        """,
    )
    if test_non_null_targets:
        errors.append(f"application_test rows must have NULL TARGET: {test_non_null_targets}")


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
            f"SELECT COUNT(*) FROM {_sql_identifier(staging_table)}",
        )
        mart_rows = _fetch_count(
            connection,
            f"""
            SELECT COUNT(*)
            FROM {_sql_identifier(MART_TABLE)}
            WHERE source_population = '{source_population}'
            """,
        )
        if staging_rows != mart_rows:
            errors.append(
                f"{source_population} staging rows do not reconcile to mart rows: "
                f"staging={staging_rows}, mart={mart_rows}"
            )


def _validate_aggregate_keys(connection: duckdb.DuckDBPyConnection, errors: list[str]) -> None:
    for table_name in AGGREGATE_TABLES:
        duplicate_keys = _duplicate_key_count(connection, table_name, ("SK_ID_CURR",))
        if duplicate_keys:
            errors.append(f"{table_name} has duplicate SK_ID_CURR keys: {duplicate_keys}")


def _validate_diagnostic_separation(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    mart_columns = set(_table_columns(connection, MART_TABLE))
    diagnostic_columns = set(_table_columns(connection, DIAGNOSTIC_TABLE))

    sensitive_columns = set(config["excluded_features"]["sensitive_or_protected_status_like"])
    sensitive_columns_in_mart = sorted(sensitive_columns.intersection(mart_columns))
    if sensitive_columns_in_mart:
        errors.append(f"Diagnostic-only columns are present in the model mart: {sensitive_columns_in_mart}")

    missing_diagnostic_columns = sorted(DIAGNOSTIC_ONLY_COLUMNS.difference(diagnostic_columns))
    if missing_diagnostic_columns:
        errors.append(f"segment_diagnostics is missing diagnostic columns: {missing_diagnostic_columns}")


def _validate_model_feature_quality(
    connection: duckdb.DuckDBPyConnection,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    all_missing_features = []
    row_count = _fetch_count(connection, f"SELECT COUNT(*) FROM {_sql_identifier(MART_TABLE)}")
    if row_count == 0:
        errors.append("mart_credit_risk_features must not be empty")
        return

    for column_name in get_model_feature_columns(connection, config):
        non_null_count = _fetch_count(
            connection,
            f"""
            SELECT COUNT(*)
            FROM {_sql_identifier(MART_TABLE)}
            WHERE {_sql_identifier(column_name)} IS NOT NULL
            """,
        )
        if non_null_count == 0:
            all_missing_features.append(column_name)

    if all_missing_features:
        errors.append(f"Model feature columns are 100% missing: {sorted(all_missing_features)}")


def _existing_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def _table_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in connection.execute(f"PRAGMA table_info({_sql_literal(table_name)})").fetchall()
    }


def _exclusion_group_map(config: dict[str, Any]) -> dict[str, str]:
    exclusions = dict(ALWAYS_EXCLUDED_COLUMNS)
    for group_name, column_names in config["excluded_features"].items():
        for column_name in column_names:
            exclusions[column_name] = group_name
    return exclusions


def _duplicate_key_count(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    key_columns: tuple[str, ...],
) -> int:
    key_select = ", ".join(_sql_identifier(column_name) for column_name in key_columns)
    return _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT {key_select}
            FROM {_sql_identifier(table_name)}
            GROUP BY {key_select}
            HAVING COUNT(*) > 1
        )
        """,
    )


def _target_count(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    null_predicate: str,
) -> int:
    return _fetch_count(
        connection,
        f"""
        SELECT COUNT(*)
        FROM {_sql_identifier(table_name)}
        WHERE TARGET {null_predicate}
        """,
    )


def _fetch_count(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    result = connection.execute(sql).fetchone()
    if result is None:
        raise DataContractError(f"Count query returned no rows: {sql}")
    return int(result[0])


def _created_at_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


def _sql_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) + chr(39))}'"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
