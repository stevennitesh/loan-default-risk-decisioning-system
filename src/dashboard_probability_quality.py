from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from src.calibration import apply_saved_calibration_artifact
from src.config import manual_review_capacity_rate
from src.metrics import build_calibration_bin_rows
from src.metrics import build_probability_metric_rows
from src.metrics import validate_probabilities
from src.mart_access import load_labeled_split_frame
from src.mart_access import require_table_columns
from src.model_artifacts import normalize_split_ids
from src.model_contracts import EVALUATION_SPLITS
from src.model_contracts import REPORTING_SPLITS
from src.modeling import predict_probabilities
from src.modeling import prediction_frame
from src.report_contracts import MODEL_CALIBRATION_BINS_COLUMNS
from src.report_contracts import MODEL_METRICS_SUMMARY_COLUMNS
from src.runtime import created_at_utc
from src.runtime import sql_identifier


def build_probability_quality_overrides(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
    config: dict[str, Any],
    dashboard_model_version: str,
    error_cls: type[Exception] = ValueError,
) -> dict[str, pd.DataFrame]:
    if calibration_artifact["selected_method"] == "uncalibrated":
        return {}

    prediction_frames = _build_calibrated_prediction_frames(
        connection,
        artifact,
        calibration_artifact,
        error_cls,
    )
    return {
        "model_metrics_summary": _metrics_frame_with_calibrated_selected_model(
            connection,
            artifact,
            prediction_frames,
            config,
            dashboard_model_version,
            error_cls,
        ),
        "model_calibration_bins": pd.DataFrame(
            build_calibration_bin_rows(dashboard_model_version, prediction_frames, REPORTING_SPLITS),
            columns=MODEL_CALIBRATION_BINS_COLUMNS,
        ),
    }


def _build_calibrated_prediction_frames(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
    error_cls: type[Exception],
) -> dict[str, pd.DataFrame]:
    feature_columns = list(artifact["feature_columns"])
    split_applicant_ids = normalize_split_ids(
        artifact["split_applicant_ids"],
        EVALUATION_SPLITS,
        error_cls=error_cls,
    )
    prediction_frames = {}

    for split_name in EVALUATION_SPLITS:
        split_frame = _load_split_feature_frame(
            connection,
            split_applicant_ids[split_name],
            feature_columns,
            split_name,
            error_cls,
        )
        raw_probabilities = predict_probabilities(
            artifact,
            split_frame,
            feature_columns,
            split_name,
            error_cls,
        )
        adjusted_probabilities = apply_saved_calibration_artifact(
            raw_probabilities,
            calibration_artifact,
            error_cls=error_cls,
            label="dashboard calibration",
        )
        validate_probabilities(
            adjusted_probabilities,
            f"{split_name} calibrated",
            error_cls=error_cls,
        )
        prediction_frames[split_name] = prediction_frame(split_frame, adjusted_probabilities)

    return prediction_frames


def _metrics_frame_with_calibrated_selected_model(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    prediction_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
    dashboard_model_version: str,
    error_cls: type[Exception],
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
        build_probability_metric_rows(
            dashboard_model_version,
            prediction_frames,
            created_at,
            manual_review_capacity_rate(config),
            error_cls=error_cls,
        ),
        columns=MODEL_METRICS_SUMMARY_COLUMNS,
    )
    return pd.concat([retained_frame, calibrated_frame], ignore_index=True)[MODEL_METRICS_SUMMARY_COLUMNS]


def _existing_metric_created_at(existing_frame: pd.DataFrame, model_version: str) -> str:
    matching_rows = existing_frame.loc[existing_frame["model_version"] == model_version]
    if matching_rows.empty:
        return created_at_utc()
    return str(matching_rows["created_at"].iloc[0])


def _load_split_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
    split_name: str,
    error_cls: type[Exception],
) -> pd.DataFrame:
    require_table_columns(
        connection,
        "mart_credit_risk_features",
        feature_columns,
        error_cls=error_cls,
    )
    return load_labeled_split_frame(
        connection,
        applicant_ids,
        feature_columns,
        split_name,
        error_cls=error_cls,
        require_both_target_classes=True,
        missing_context="split",
    )
