from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.evaluate import MODEL_CALIBRATION_BINS_COLUMNS
from src.evaluate import MODEL_CONFUSION_MATRIX_COLUMNS
from src.evaluate import MODEL_LIFT_BY_DECILE_COLUMNS
from src.explain import MODEL_FEATURE_IMPORTANCE_COLUMNS
from src.score_batch import CREDIT_RISK_SCORE_COLUMNS
from src.thresholding import MODEL_THRESHOLD_METRICS_COLUMNS
from src.train import BASELINE_MODEL_ARTIFACT_NAME
from src.train import BASELINE_MODEL_TYPE
from src.train import BASELINE_MODEL_VERSION
from src.train import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.train import LIGHTGBM_MODEL_TYPE
from src.train import LIGHTGBM_MODEL_VERSION
from src.train import MODEL_METRICS_SUMMARY_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[1]

REPORTING_SPLITS = ("validation", "test")
SEGMENT_DIMENSIONS = [
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "applicant_age_band",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]

SEGMENT_PERFORMANCE_SUMMARY_COLUMNS = [
    "model_version",
    "split",
    "segment_name",
    "segment_value",
    "applicant_count",
    "observed_default_rate",
    "average_score",
    "roc_auc",
    "pr_auc",
    "brier_score",
]

DASHBOARD_EXPORT_TABLES = [
    "credit_risk_scores",
    "model_metrics_summary",
    "model_threshold_metrics",
    "model_lift_by_decile",
    "model_calibration_bins",
    "model_confusion_matrix",
    "model_feature_importance",
    "segment_performance_summary",
]

REQUIRED_SOURCE_TABLES = [
    "credit_risk_scores",
    "model_metrics_summary",
    "model_threshold_metrics",
    "model_lift_by_decile",
    "model_calibration_bins",
    "model_confusion_matrix",
    "model_feature_importance",
    "model_comparison_summary",
    "mart_credit_risk_features",
    "segment_diagnostics",
]

EXPORT_TABLE_COLUMNS = {
    "credit_risk_scores": CREDIT_RISK_SCORE_COLUMNS,
    "model_metrics_summary": MODEL_METRICS_SUMMARY_COLUMNS,
    "model_threshold_metrics": MODEL_THRESHOLD_METRICS_COLUMNS,
    "model_lift_by_decile": MODEL_LIFT_BY_DECILE_COLUMNS,
    "model_calibration_bins": MODEL_CALIBRATION_BINS_COLUMNS,
    "model_confusion_matrix": MODEL_CONFUSION_MATRIX_COLUMNS,
    "model_feature_importance": MODEL_FEATURE_IMPORTANCE_COLUMNS,
    "segment_performance_summary": SEGMENT_PERFORMANCE_SUMMARY_COLUMNS,
}

MODEL_ARTIFACTS = {
    BASELINE_MODEL_TYPE: (BASELINE_MODEL_VERSION, BASELINE_MODEL_ARTIFACT_NAME),
    LIGHTGBM_MODEL_TYPE: (LIGHTGBM_MODEL_VERSION, LIGHTGBM_MODEL_ARTIFACT_NAME),
}


class DashboardExportError(RuntimeError):
    """Raised when Power BI dashboard exports cannot satisfy the Milestone 10 contract."""


def run_dashboard_export(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])
    export_dir = _resolve_project_path(config["paths"]["dashboard_export_dir"])

    if not duckdb_path.exists():
        raise DashboardExportError(f"DuckDB database not found: {duckdb_path}")

    with duckdb.connect(str(duckdb_path)) as connection:
        _require_tables(connection, REQUIRED_SOURCE_TABLES)
        _validate_export_source_columns(connection)
        selected_model_type = _load_selected_model_type(connection)
        artifact = _load_selected_artifact(model_dir, selected_model_type)
        segment_rows = _build_segment_performance_rows(connection, artifact)
        _replace_duckdb_table(connection, "segment_performance_summary", segment_rows)
        _validate_export_source_columns(connection)

        export_dir.mkdir(parents=True, exist_ok=True)
        row_counts = {}
        for table_name in DASHBOARD_EXPORT_TABLES:
            row_counts[table_name] = _export_table(
                connection,
                table_name,
                export_dir / f"{table_name}.csv",
            )

    return {
        "export_dir": export_dir,
        "exported_tables": DASHBOARD_EXPORT_TABLES,
        "row_counts": row_counts,
        "selected_model_type": selected_model_type,
        "selected_model_version": artifact["model_version"],
    }


def _load_selected_model_type(connection: duckdb.DuckDBPyConnection) -> str:
    selected_values = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT selected_model_type FROM model_comparison_summary"
        ).fetchall()
    }
    if len(selected_values) != 1:
        raise DashboardExportError(
            f"model_comparison_summary must contain exactly one selected_model_type, got {sorted(selected_values)}"
        )
    selected_model_type = str(next(iter(selected_values)))
    if selected_model_type not in MODEL_ARTIFACTS:
        raise DashboardExportError(f"Unsupported selected_model_type: {selected_model_type}")
    return selected_model_type


def _load_selected_artifact(model_dir: Path, selected_model_type: str) -> dict[str, Any]:
    expected_model_version, artifact_name = MODEL_ARTIFACTS[selected_model_type]
    artifact_path = model_dir / artifact_name
    if not artifact_path.exists():
        raise DashboardExportError(f"Missing selected model artifact: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise DashboardExportError(f"Selected model artifact must be a dict: {artifact_path}")

    required_keys = {
        "pipeline",
        "model_version",
        "model_type",
        "feature_columns",
        "split_applicant_ids",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise DashboardExportError(f"Selected model artifact is missing required keys: {missing_keys}")
    if artifact["model_type"] != selected_model_type:
        raise DashboardExportError(
            f"Selected artifact model_type={artifact['model_type']}, expected {selected_model_type}"
        )
    if artifact["model_version"] != expected_model_version:
        raise DashboardExportError(
            f"Selected artifact model_version={artifact['model_version']}, expected {expected_model_version}"
        )
    if not artifact["feature_columns"]:
        raise DashboardExportError("Selected model artifact does not contain feature_columns")
    return artifact


def _build_segment_performance_rows(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    feature_columns = list(artifact["feature_columns"])
    split_applicant_ids = _normalize_split_ids(artifact["split_applicant_ids"])
    rows: list[dict[str, Any]] = []

    for split_name in REPORTING_SPLITS:
        split_frame = _load_split_segment_frame(
            connection,
            split_applicant_ids[split_name],
            feature_columns,
            split_name,
        )
        probabilities = artifact["pipeline"].predict_proba(
            _feature_frame(split_frame, feature_columns)
        )[:, 1]
        _validate_probabilities(probabilities, split_name)
        split_frame = split_frame.copy()
        split_frame["probability"] = probabilities.astype(float)
        target_values = split_frame["TARGET"].astype(int)

        for segment_name in SEGMENT_DIMENSIONS:
            for segment_value, segment_frame in split_frame.groupby(segment_name, dropna=False, sort=True):
                segment_targets = segment_frame["TARGET"].astype(int)
                segment_probabilities = segment_frame["probability"].to_numpy(dtype=float)
                rows.append(
                    {
                        "model_version": artifact["model_version"],
                        "split": split_name,
                        "segment_name": segment_name,
                        "segment_value": _segment_value(segment_value),
                        "applicant_count": len(segment_frame),
                        "observed_default_rate": float(segment_targets.mean()),
                        "average_score": float(segment_probabilities.mean()),
                        "roc_auc": _roc_auc_or_none(segment_targets, segment_probabilities),
                        "pr_auc": _pr_auc_or_none(segment_targets, segment_probabilities),
                        "brier_score": float(brier_score_loss(segment_targets, segment_probabilities)),
                    }
                )

        if len(split_frame) != len(target_values):
            raise DashboardExportError(f"{split_name} segment frame changed size while building summaries")

    if not rows:
        raise DashboardExportError("segment_performance_summary must not be empty")
    return rows


def _load_split_segment_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    mart_columns = set(_table_columns(connection, "mart_credit_risk_features"))
    diagnostic_columns = set(_table_columns(connection, "segment_diagnostics"))
    missing_feature_columns = sorted(set(feature_columns).difference(mart_columns))
    missing_segment_columns = sorted(set(SEGMENT_DIMENSIONS).difference(diagnostic_columns))
    if missing_feature_columns:
        raise DashboardExportError(
            f"mart_credit_risk_features is missing selected model feature columns: {missing_feature_columns}"
        )
    if missing_segment_columns:
        raise DashboardExportError(f"segment_diagnostics is missing segment columns: {missing_segment_columns}")

    ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
    connection.register("split_ids", ids_frame)
    try:
        feature_select = ", ".join(f"m.{_sql_identifier(column)}" for column in feature_columns)
        segment_select = ", ".join(f"d.{_sql_identifier(column)}" for column in SEGMENT_DIMENSIONS)
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
        raise DashboardExportError(
            f"Saved split IDs no longer reconcile for {split_name} dashboard export: missing {missing_ids[:10]}"
        )
    target_values = set(frame["TARGET"].dropna().astype(int).unique())
    if target_values != {0, 1}:
        raise DashboardExportError(f"{split_name} dashboard segment rows must contain both target classes")
    return frame.reset_index(drop=True)


def _normalize_split_ids(raw_split_ids: Any) -> dict[str, list[int]]:
    if not isinstance(raw_split_ids, dict):
        raise DashboardExportError("split_applicant_ids must be a mapping")
    split_ids = {}
    missing_splits = [split for split in REPORTING_SPLITS if split not in raw_split_ids]
    if missing_splits:
        raise DashboardExportError(f"split_applicant_ids is missing splits: {missing_splits}")
    for split_name in REPORTING_SPLITS:
        ids = [int(value) for value in raw_split_ids[split_name]]
        if not ids:
            raise DashboardExportError(f"split_applicant_ids[{split_name}] must not be empty")
        if len(ids) != len(set(ids)):
            raise DashboardExportError(f"split_applicant_ids[{split_name}] contains duplicate applicants")
        split_ids[split_name] = ids
    return split_ids


def _export_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    export_path: Path,
) -> int:
    columns = EXPORT_TABLE_COLUMNS[table_name]
    frame = connection.execute(
        f"""
        SELECT {", ".join(_sql_identifier(column) for column in columns)}
        FROM {_sql_identifier(table_name)}
        """
    ).fetch_df()
    frame.to_csv(export_path, index=False)
    return len(frame)


def _validate_export_source_columns(connection: duckdb.DuckDBPyConnection) -> None:
    existing_tables = _existing_tables(connection)
    missing_tables = sorted(table for table in DASHBOARD_EXPORT_TABLES if table not in existing_tables)
    missing_tables = [
        table for table in missing_tables if table != "segment_performance_summary"
    ]
    if missing_tables:
        raise DashboardExportError(f"Missing required DuckDB tables: {', '.join(missing_tables)}")

    for table_name, expected_columns in EXPORT_TABLE_COLUMNS.items():
        if table_name not in existing_tables:
            continue
        columns = _table_columns(connection, table_name)
        missing_columns = sorted(set(expected_columns).difference(columns))
        if missing_columns:
            raise DashboardExportError(f"{table_name} is missing required columns: {missing_columns}")


def _require_tables(
    connection: duckdb.DuckDBPyConnection,
    table_names: list[str],
) -> None:
    existing_tables = _existing_tables(connection)
    missing_tables = sorted(set(table_names).difference(existing_tables))
    if missing_tables:
        raise DashboardExportError(f"Missing required DuckDB tables: {', '.join(missing_tables)}")


def _validate_probabilities(probabilities: np.ndarray, split_name: str) -> None:
    if probabilities.ndim != 1:
        raise DashboardExportError(f"{split_name} probabilities must be one-dimensional")
    if not np.isfinite(probabilities).all():
        raise DashboardExportError(f"{split_name} probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise DashboardExportError(f"{split_name} probabilities must be in [0, 1]")


def _roc_auc_or_none(targets: pd.Series, probabilities: np.ndarray) -> float | None:
    if set(targets.astype(int).unique()) != {0, 1}:
        return None
    return float(roc_auc_score(targets, probabilities))


def _pr_auc_or_none(targets: pd.Series, probabilities: np.ndarray) -> float | None:
    if set(targets.astype(int).unique()) != {0, 1}:
        return None
    return float(average_precision_score(targets, probabilities))


def _segment_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    return features.where(pd.notna(features), np.nan)


def _replace_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(rows, columns=SEGMENT_PERFORMANCE_SUMMARY_COLUMNS)
    connection.register("output_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {_sql_identifier(table_name)} AS SELECT * FROM output_frame")
    connection.unregister("output_frame")


def _existing_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def _table_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in connection.execute(f"PRAGMA table_info({_sql_literal(table_name)})").fetchall()
    }


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


def _sql_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) + chr(39))}'"
