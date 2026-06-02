from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.calibrate import CALIBRATION_ARTIFACT_NAME
from src.config import load_config
from src.dashboard_probability_quality import build_probability_quality_overrides
from src.dashboard_segments import build_segment_performance_rows
from src.mart_access import existing_tables
from src.mart_access import require_tables
from src.mart_access import table_columns
from src.model_contracts import LIGHTGBM_MODEL_TYPE
from src.model_contracts import MODEL_ARTIFACTS
from src.model_artifacts import load_calibration_artifact
from src.model_artifacts import load_selected_model_artifact
from src.model_artifacts import load_selected_model_type
from src.report_contracts import CREDIT_RISK_SCORE_COLUMNS
from src.report_contracts import MODEL_CALIBRATION_BINS_COLUMNS
from src.report_contracts import MODEL_CONFUSION_MATRIX_COLUMNS
from src.report_contracts import MODEL_FEATURE_IMPORTANCE_COLUMNS
from src.report_contracts import MODEL_LIFT_BY_DECILE_COLUMNS
from src.report_contracts import MODEL_METRICS_SUMMARY_COLUMNS
from src.report_contracts import MODEL_THRESHOLD_METRICS_COLUMNS
from src.report_contracts import SEGMENT_PERFORMANCE_SUMMARY_COLUMNS
from src.runtime import ensure_directories
from src.runtime import replace_duckdb_table
from src.runtime import require_existing_path
from src.runtime import resolve_config_path
from src.runtime import resolve_project_path
from src.runtime import sql_identifier


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

POST_V1_DASHBOARD_MODEL_VERSION = "lightgbm_credit_risk_post_v1"


class DashboardExportError(RuntimeError):
    """Raised when Power BI dashboard exports cannot satisfy the Milestone 10 contract."""


def run_dashboard_export(
    config_path: str | Path = "configs/base.yaml",
    export_dir: str | Path | None = None,
    use_calibrated_probability_quality: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    resolved_export_dir = (
        resolve_project_path(str(export_dir))
        if export_dir is not None
        else resolve_config_path(config, "dashboard_export_dir")
    )

    require_existing_path(duckdb_path, "DuckDB database", DashboardExportError)

    with duckdb.connect(str(duckdb_path)) as connection:
        require_tables(connection, REQUIRED_SOURCE_TABLES, error_cls=DashboardExportError)
        _validate_export_source_columns(connection)
        selected_model_type = load_selected_model_type(
            connection,
            set(MODEL_ARTIFACTS),
            error_cls=DashboardExportError,
        )
        artifact = load_selected_model_artifact(
            model_dir,
            selected_model_type,
            MODEL_ARTIFACTS,
            error_cls=DashboardExportError,
        )
        calibration_artifact = (
            load_calibration_artifact(
                model_dir,
                artifact,
                CALIBRATION_ARTIFACT_NAME,
                LIGHTGBM_MODEL_TYPE,
                error_cls=DashboardExportError,
            )
            if use_calibrated_probability_quality
            else {"selected_method": "uncalibrated", "calibrators": {}}
        )
        source_model_version = str(artifact["model_version"])
        dashboard_model_version = (
            POST_V1_DASHBOARD_MODEL_VERSION
            if use_calibrated_probability_quality
            else source_model_version
        )
        model_version_relabel = (
            (source_model_version, dashboard_model_version)
            if dashboard_model_version != source_model_version
            else None
        )
        dashboard_table_overrides = build_probability_quality_overrides(
            connection,
            artifact,
            calibration_artifact,
            config,
            dashboard_model_version,
            DashboardExportError,
        )
        segment_rows = build_segment_performance_rows(
            connection,
            artifact,
            calibration_artifact,
            dashboard_model_version,
            DashboardExportError,
        )
        replace_duckdb_table(
            connection,
            "segment_performance_summary",
            segment_rows,
            SEGMENT_PERFORMANCE_SUMMARY_COLUMNS,
        )
        _validate_export_source_columns(connection)

        ensure_directories(resolved_export_dir)
        row_counts = {}
        for table_name in DASHBOARD_EXPORT_TABLES:
            export_path = resolved_export_dir / f"{table_name}.csv"
            if table_name in dashboard_table_overrides:
                row_counts[table_name] = _export_frame(dashboard_table_overrides[table_name], export_path)
            else:
                row_counts[table_name] = _export_table(
                    connection,
                    table_name,
                    export_path,
                    model_version_relabel,
                )

    return {
        "export_dir": resolved_export_dir,
        "exported_tables": DASHBOARD_EXPORT_TABLES,
        "row_counts": row_counts,
        "selected_model_type": selected_model_type,
        "selected_model_source_version": source_model_version,
        "selected_model_version": dashboard_model_version,
        "use_calibrated_probability_quality": use_calibrated_probability_quality,
    }


def _export_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    export_path: Path,
    model_version_relabel: tuple[str, str] | None = None,
) -> int:
    columns = EXPORT_TABLE_COLUMNS[table_name]
    frame = connection.execute(
        f"""
        SELECT {", ".join(sql_identifier(column) for column in columns)}
        FROM {sql_identifier(table_name)}
        """
    ).fetch_df()
    frame = _relabel_model_version(frame, model_version_relabel)
    return _export_frame(frame, export_path)


def _export_frame(frame: pd.DataFrame, export_path: Path) -> int:
    frame.to_csv(export_path, index=False)
    return len(frame)


def _relabel_model_version(
    frame: pd.DataFrame,
    model_version_relabel: tuple[str, str] | None,
) -> pd.DataFrame:
    if model_version_relabel is None or "model_version" not in frame.columns:
        return frame
    source_model_version, dashboard_model_version = model_version_relabel
    if source_model_version == dashboard_model_version:
        return frame
    output_frame = frame.copy()
    output_frame["model_version"] = output_frame["model_version"].replace(
        {source_model_version: dashboard_model_version}
    )
    return output_frame


def _validate_export_source_columns(connection: duckdb.DuckDBPyConnection) -> None:
    available_tables = existing_tables(connection)
    missing_tables = sorted(table for table in DASHBOARD_EXPORT_TABLES if table not in available_tables)
    missing_tables = [
        table for table in missing_tables if table != "segment_performance_summary"
    ]
    if missing_tables:
        raise DashboardExportError(f"Missing required DuckDB tables: {', '.join(missing_tables)}")

    for table_name, expected_columns in EXPORT_TABLE_COLUMNS.items():
        if table_name not in available_tables:
            continue
        columns = table_columns(connection, table_name)
        missing_columns = sorted(set(expected_columns).difference(columns))
        if missing_columns:
            raise DashboardExportError(f"{table_name} is missing required columns: {missing_columns}")
