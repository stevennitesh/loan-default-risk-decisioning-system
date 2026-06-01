from __future__ import annotations

import duckdb
import pandas as pd

from src.runtime import sql_identifier
from src.runtime import sql_literal


def load_labeled_split_frames(
    connection: duckdb.DuckDBPyConnection,
    split_applicant_ids: dict[str, list[int]],
    feature_columns: list[str],
    error_cls: type[Exception] = ValueError,
) -> dict[str, pd.DataFrame]:
    split_frames = {}

    for split_name, applicant_ids in split_applicant_ids.items():
        split_frames[split_name] = load_labeled_split_frame(
            connection,
            applicant_ids,
            feature_columns,
            split_name,
            error_cls=error_cls,
            require_both_target_classes=True,
        )

    return split_frames


def load_labeled_split_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
    split_name: str,
    error_cls: type[Exception] = ValueError,
    require_both_target_classes: bool = False,
    missing_context: str = "split",
) -> pd.DataFrame:
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
    connection.register("split_ids", ids_frame)
    try:
        frame = connection.execute(
            f"""
            SELECT {", ".join(sql_identifier(column) for column in selected_columns)}
            FROM mart_credit_risk_features
            INNER JOIN split_ids USING (SK_ID_CURR)
            WHERE source_population = 'application_train'
            ORDER BY SK_ID_CURR
            """
        ).fetch_df()
    finally:
        connection.unregister("split_ids")

    if len(frame) != len(applicant_ids):
        found_ids = set(frame["SK_ID_CURR"].astype(int).tolist()) if not frame.empty else set()
        missing_ids = sorted(set(applicant_ids).difference(found_ids))
        raise error_cls(
            f"Saved {missing_context} IDs no longer reconcile to mart_credit_risk_features for "
            f"{split_name}: missing {missing_ids[:10]}"
        )
    if frame["TARGET"].isna().any():
        raise error_cls(f"{split_name} rows must have observed TARGET values")
    if require_both_target_classes:
        targets = set(frame["TARGET"].astype(int).unique())
        if targets != {0, 1}:
            raise error_cls(f"{split_name} split must contain binary TARGET classes, got {sorted(targets)}")
    return frame.reset_index(drop=True)


def load_application_test_frame(
    connection: duckdb.DuckDBPyConnection,
    feature_columns: list[str],
    error_cls: type[Exception] = ValueError,
) -> pd.DataFrame:
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    frame = connection.execute(
        f"""
        SELECT {", ".join(sql_identifier(column) for column in selected_columns)}
        FROM mart_credit_risk_features
        WHERE source_population = 'application_test'
        ORDER BY SK_ID_CURR
        """
    ).fetch_df()
    if frame.empty:
        raise error_cls("No application_test rows are available for kaggle_test scoring")
    if frame["TARGET"].notna().any():
        raise error_cls("kaggle_test rows must have NULL TARGET values")
    return frame.reset_index(drop=True)


def load_labeled_segment_split_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
    segment_columns: list[str] | tuple[str, ...],
    split_name: str,
    error_cls: type[Exception] = ValueError,
) -> pd.DataFrame:
    mart_columns = set(table_columns(connection, "mart_credit_risk_features"))
    diagnostic_columns = set(table_columns(connection, "segment_diagnostics"))
    missing_feature_columns = sorted(set(feature_columns).difference(mart_columns))
    missing_segment_columns = sorted(set(segment_columns).difference(diagnostic_columns))
    if missing_feature_columns:
        raise error_cls(
            f"mart_credit_risk_features is missing selected model feature columns: {missing_feature_columns}"
        )
    if missing_segment_columns:
        raise error_cls(f"segment_diagnostics is missing segment columns: {missing_segment_columns}")

    ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
    connection.register("split_ids", ids_frame)
    try:
        feature_select = ", ".join(f"m.{sql_identifier(column)}" for column in feature_columns)
        segment_select = ", ".join(f"d.{sql_identifier(column)}" for column in segment_columns)
        frame = connection.execute(
            f"""
            SELECT
                m.SK_ID_CURR,
                m.TARGET,
                {feature_select},
                {segment_select}
            FROM mart_credit_risk_features AS m
            INNER JOIN split_ids USING (SK_ID_CURR)
            INNER JOIN segment_diagnostics AS d
                ON d.SK_ID_CURR = m.SK_ID_CURR
               AND d.source_population = m.source_population
            WHERE m.source_population = 'application_train'
            ORDER BY m.SK_ID_CURR
            """
        ).fetch_df()
    finally:
        connection.unregister("split_ids")

    if len(frame) != len(applicant_ids):
        found_ids = set(frame["SK_ID_CURR"].astype(int).tolist()) if not frame.empty else set()
        missing_ids = sorted(set(applicant_ids).difference(found_ids))
        raise error_cls(
            f"Saved split IDs no longer reconcile for {split_name} dashboard export: missing {missing_ids[:10]}"
        )
    target_values = set(frame["TARGET"].dropna().astype(int).unique())
    if target_values != {0, 1}:
        raise error_cls(f"{split_name} dashboard segment rows must contain both target classes")
    return frame.reset_index(drop=True)


def existing_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def require_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    error_cls: type[Exception] = ValueError,
) -> None:
    if table_name not in existing_tables(connection):
        raise error_cls(f"Missing required DuckDB table: {table_name}")


def require_tables(
    connection: duckdb.DuckDBPyConnection,
    table_names: list[str] | tuple[str, ...],
    error_cls: type[Exception] = ValueError,
) -> None:
    missing_tables = sorted(set(table_names).difference(existing_tables(connection)))
    if missing_tables:
        raise error_cls(f"Missing required DuckDB tables: {', '.join(missing_tables)}")


def table_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in connection.execute(f"PRAGMA table_info({sql_literal(table_name)})").fetchall()
    }


def require_table_columns(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    expected_columns: list[str] | tuple[str, ...] | set[str],
    error_cls: type[Exception] = ValueError,
) -> None:
    existing_columns = set(table_columns(connection, table_name))
    missing_columns = sorted(set(expected_columns).difference(existing_columns))
    if missing_columns:
        raise error_cls(f"{table_name} is missing required columns: {missing_columns}")
