from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from src.calibrate import CALIBRATION_ARTIFACT_NAME
from src.calibration import apply_calibration_to_probabilities
from src.config import load_config
from src.metrics import build_calibration_bin_rows
from src.metrics import build_probability_metric_rows
from src.metrics import validate_probabilities
from src.mart_access import existing_tables
from src.mart_access import load_labeled_segment_split_frame
from src.mart_access import load_labeled_split_frame
from src.mart_access import require_table_columns
from src.mart_access import require_tables
from src.mart_access import table_columns
from src.model_contracts import EVALUATION_SPLITS
from src.model_contracts import LIGHTGBM_MODEL_TYPE
from src.model_contracts import MODEL_ARTIFACTS
from src.model_contracts import REPORTING_SPLITS
from src.model_artifacts import load_calibration_artifact
from src.model_artifacts import load_selected_model_artifact
from src.model_artifacts import load_selected_model_type
from src.model_artifacts import normalize_split_ids
from src.report_contracts import CREDIT_RISK_SCORE_COLUMNS
from src.report_contracts import MODEL_CALIBRATION_BINS_COLUMNS
from src.report_contracts import MODEL_CONFUSION_MATRIX_COLUMNS
from src.report_contracts import MODEL_FEATURE_IMPORTANCE_COLUMNS
from src.report_contracts import MODEL_LIFT_BY_DECILE_COLUMNS
from src.report_contracts import MODEL_METRICS_SUMMARY_COLUMNS
from src.report_contracts import MODEL_THRESHOLD_METRICS_COLUMNS
from src.report_contracts import SEGMENT_PERFORMANCE_SUMMARY_COLUMNS
from src.runtime import created_at_utc
from src.runtime import feature_frame
from src.runtime import replace_duckdb_table
from src.runtime import resolve_project_path
from src.runtime import sql_identifier


SEGMENT_DIMENSIONS = [
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "applicant_age_band",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
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

POST_V1_DASHBOARD_MODEL_VERSION = "lightgbm_credit_risk_post_v1"


class DashboardExportError(RuntimeError):
    """Raised when Power BI dashboard exports cannot satisfy the Milestone 10 contract."""


def run_dashboard_export(
    config_path: str | Path = "configs/base.yaml",
    export_dir: str | Path | None = None,
    use_calibrated_probability_quality: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = resolve_project_path(config["paths"]["model_dir"])
    resolved_export_dir = (
        resolve_project_path(str(export_dir))
        if export_dir is not None
        else resolve_project_path(config["paths"]["dashboard_export_dir"])
    )

    if not duckdb_path.exists():
        raise DashboardExportError(f"DuckDB database not found: {duckdb_path}")

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
        dashboard_table_overrides = _build_dashboard_table_overrides(
            connection,
            artifact,
            calibration_artifact,
            config,
            dashboard_model_version,
        )
        segment_rows = _build_segment_performance_rows(
            connection,
            artifact,
            calibration_artifact,
            dashboard_model_version,
        )
        replace_duckdb_table(
            connection,
            "segment_performance_summary",
            segment_rows,
            SEGMENT_PERFORMANCE_SUMMARY_COLUMNS,
        )
        _validate_export_source_columns(connection)

        resolved_export_dir.mkdir(parents=True, exist_ok=True)
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


def _build_dashboard_table_overrides(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
    config: dict[str, Any],
    dashboard_model_version: str,
) -> dict[str, pd.DataFrame]:
    if calibration_artifact["selected_method"] == "uncalibrated":
        return {}

    prediction_frames = _build_calibrated_prediction_frames(connection, artifact, calibration_artifact)
    return {
        "model_metrics_summary": _metrics_frame_with_calibrated_selected_model(
            connection,
            artifact,
            prediction_frames,
            config,
            dashboard_model_version,
        ),
        "model_calibration_bins": pd.DataFrame(
            _build_calibrated_bin_rows(dashboard_model_version, prediction_frames),
            columns=MODEL_CALIBRATION_BINS_COLUMNS,
        ),
    }


def _build_calibrated_prediction_frames(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    feature_columns = list(artifact["feature_columns"])
    split_applicant_ids = _normalize_evaluation_split_ids(artifact["split_applicant_ids"])
    prediction_frames = {}

    for split_name in EVALUATION_SPLITS:
        split_frame = _load_split_feature_frame(
            connection,
            split_applicant_ids[split_name],
            feature_columns,
            split_name,
        )
        raw_probabilities = artifact["pipeline"].predict_proba(
            feature_frame(split_frame, feature_columns)
        )[:, 1]
        validate_probabilities(raw_probabilities, split_name, error_cls=DashboardExportError)
        calibrated_probabilities = _calibrated_probabilities(raw_probabilities, calibration_artifact)
        validate_probabilities(
            calibrated_probabilities,
            f"{split_name} calibrated",
            error_cls=DashboardExportError,
        )
        prediction_frames[split_name] = pd.DataFrame(
            {
                "SK_ID_CURR": split_frame["SK_ID_CURR"].astype(int),
                "target": split_frame["TARGET"].astype(int),
                "probability": calibrated_probabilities.astype(float),
            }
        )

    return prediction_frames


def _metrics_frame_with_calibrated_selected_model(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    prediction_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    dashboard_model_version: str,
) -> pd.DataFrame:
    existing_frame = connection.execute(
        f"""
        SELECT {", ".join(sql_identifier(column) for column in MODEL_METRICS_SUMMARY_COLUMNS)}
        FROM model_metrics_summary
        """
    ).fetch_df()
    model_version = str(artifact["model_version"])
    retained_frame = existing_frame.loc[existing_frame["model_version"] != model_version].copy()
    created_at = _existing_metric_created_at(existing_frame, model_version)
    calibrated_frame = pd.DataFrame(
        _build_calibrated_metric_rows(
            dashboard_model_version,
            prediction_frames,
            created_at,
            float(config["business_assumptions"]["manual_review_capacity_rate"]),
        ),
        columns=MODEL_METRICS_SUMMARY_COLUMNS,
    )
    return pd.concat([retained_frame, calibrated_frame], ignore_index=True)[MODEL_METRICS_SUMMARY_COLUMNS]


def _existing_metric_created_at(existing_frame: pd.DataFrame, model_version: str) -> str:
    matching_rows = existing_frame.loc[existing_frame["model_version"] == model_version]
    if matching_rows.empty:
        return created_at_utc()
    return str(matching_rows["created_at"].iloc[0])


def _build_calibrated_metric_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    created_at: str,
    manual_review_capacity_rate: float,
) -> list[dict[str, Any]]:
    return build_probability_metric_rows(
        model_version,
        prediction_frames,
        created_at,
        manual_review_capacity_rate,
        error_cls=DashboardExportError,
    )


def _build_calibrated_bin_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    return build_calibration_bin_rows(model_version, prediction_frames, REPORTING_SPLITS)


def _build_segment_performance_rows(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
    dashboard_model_version: str,
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
            feature_frame(split_frame, feature_columns)
        )[:, 1]
        validate_probabilities(probabilities, split_name, error_cls=DashboardExportError)
        probabilities = _calibrated_probabilities(probabilities, calibration_artifact)
        validate_probabilities(probabilities, f"{split_name} calibrated", error_cls=DashboardExportError)
        split_frame = split_frame.copy()
        split_frame["probability"] = probabilities.astype(float)
        target_values = split_frame["TARGET"].astype(int)

        for segment_name in SEGMENT_DIMENSIONS:
            for segment_value, segment_frame in split_frame.groupby(segment_name, dropna=False, sort=True):
                segment_targets = segment_frame["TARGET"].astype(int)
                segment_probabilities = segment_frame["probability"].to_numpy(dtype=float)
                rows.append(
                    {
                        "model_version": dashboard_model_version,
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


def _load_split_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    require_table_columns(
        connection,
        "mart_credit_risk_features",
        feature_columns,
        error_cls=DashboardExportError,
    )
    return load_labeled_split_frame(
        connection,
        applicant_ids,
        feature_columns,
        split_name,
        error_cls=DashboardExportError,
        require_both_target_classes=True,
        missing_context="split",
    )


def _load_split_segment_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    return load_labeled_segment_split_frame(
        connection,
        applicant_ids,
        feature_columns,
        SEGMENT_DIMENSIONS,
        split_name,
        error_cls=DashboardExportError,
    )


def _normalize_evaluation_split_ids(raw_split_ids: Any) -> dict[str, list[int]]:
    return normalize_split_ids(raw_split_ids, EVALUATION_SPLITS, error_cls=DashboardExportError)


def _normalize_split_ids(raw_split_ids: Any) -> dict[str, list[int]]:
    return normalize_split_ids(raw_split_ids, REPORTING_SPLITS, error_cls=DashboardExportError)


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
    frame.to_csv(export_path, index=False)
    return len(frame)


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


def _calibrated_probabilities(
    raw_probabilities: np.ndarray,
    calibration_artifact: dict[str, Any],
) -> np.ndarray:
    return apply_calibration_to_probabilities(
        str(calibration_artifact["selected_method"]),
        calibration_artifact["calibrators"],
        raw_probabilities,
        error_cls=DashboardExportError,
        label="dashboard calibration",
    ).astype(float)


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
