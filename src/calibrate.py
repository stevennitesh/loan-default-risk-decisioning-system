from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.calibration import (
    CALIBRATION_FIT_SPLIT,
    CALIBRATION_METHODS,
    apply_calibration_method,
    fit_calibrators,
    select_calibration_method,
)
from src.cli import add_config_argument, exit_with_error
from src.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    manual_review_capacity_rate,
    project_random_seed,
)
from src.mart_access import load_labeled_split_frames
from src.metrics import build_calibration_bin_rows, probability_metrics
from src.model_artifacts import load_model_artifact, normalize_split_ids
from src.model_contracts import (
    EVALUATION_SPLITS,
    LIGHTGBM_MODEL_ARTIFACT_NAME,
    LIGHTGBM_MODEL_TYPE,
    LIGHTGBM_MODEL_VERSION,
    REPORTING_SPLITS,
)
from src.modeling import predict_probabilities, prediction_frame
from src.report_contracts import (
    MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS,
    MODEL_CALIBRATION_COMPARISON_COLUMNS,
)
from src.runtime import (
    created_at_utc,
    ensure_directories,
    replace_duckdb_table,
    require_existing_path,
    resolve_config_path,
    write_csv,
)

CALIBRATION_ARTIFACT_NAME = "lightgbm_credit_risk_calibration.joblib"


class CalibrationError(RuntimeError):
    """Raised when the post-v1 calibration experiment cannot run safely."""


def run_calibration_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Run the LightGBM probability calibration comparison and save its artifact."""
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    require_existing_path(duckdb_path, "DuckDB database", CalibrationError)

    artifact = load_model_artifact(
        model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME,
        expected_model_type=LIGHTGBM_MODEL_TYPE,
        expected_model_version=LIGHTGBM_MODEL_VERSION,
        error_cls=CalibrationError,
        artifact_label="LightGBM artifact",
        missing_label="LightGBM model artifact",
        require_predict_proba=True,
    )

    feature_columns = list(artifact["feature_columns"])
    split_applicant_ids = normalize_split_ids(
        artifact["split_applicant_ids"],
        EVALUATION_SPLITS,
        error_cls=CalibrationError,
    )
    created_at = created_at_utc()
    review_capacity_rate = manual_review_capacity_rate(config)

    with duckdb.connect(str(duckdb_path)) as connection:
        split_frames = load_labeled_split_frames(
            connection,
            split_applicant_ids,
            feature_columns,
            error_cls=CalibrationError,
        )
        uncalibrated_predictions = _build_uncalibrated_predictions(
            artifact,
            split_frames,
            feature_columns,
        )
        calibrators = fit_calibrators(
            uncalibrated_predictions[CALIBRATION_FIT_SPLIT]["probability"].to_numpy(),
            uncalibrated_predictions[CALIBRATION_FIT_SPLIT]["target"].to_numpy(),
            project_random_seed(config),
            error_cls=CalibrationError,
        )
        calibrated_predictions = {
            method: apply_calibration_method(
                method,
                calibrators,
                uncalibrated_predictions,
                error_cls=CalibrationError,
            )
            for method in CALIBRATION_METHODS
        }
        comparison_rows, bin_rows = _build_comparison_outputs(
            calibrated_predictions,
            review_capacity_rate,
            created_at,
        )
        selected_method = select_calibration_method(
            comparison_rows, error_cls=CalibrationError
        )

        ensure_directories(report_dir, model_dir)
        write_csv(
            report_dir / "model_calibration_comparison.csv",
            MODEL_CALIBRATION_COMPARISON_COLUMNS,
            comparison_rows,
        )
        write_csv(
            report_dir / "model_calibration_bins_comparison.csv",
            MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS,
            bin_rows,
        )
        replace_duckdb_table(
            connection, "model_calibration_comparison", comparison_rows
        )
        replace_duckdb_table(connection, "model_calibration_bins_comparison", bin_rows)

    calibration_artifact = {
        "base_model_version": LIGHTGBM_MODEL_VERSION,
        "base_model_type": LIGHTGBM_MODEL_TYPE,
        "calibration_fit_split": CALIBRATION_FIT_SPLIT,
        "selected_method": selected_method,
        "selection_rule": (
            "Require at least 0.0005 validation Brier improvement over uncalibrated scores; "
            "prefer sigmoid when it is within 0.0005 Brier of isotonic because it is simpler and "
            "rank-preserving. Test metrics are held out for reporting only."
        ),
        "calibrators": calibrators,
        "fit_applicant_ids": split_applicant_ids[CALIBRATION_FIT_SPLIT],
        "split_applicant_ids": split_applicant_ids,
        "feature_columns": feature_columns,
        "comparison_rows": comparison_rows,
        "created_at": created_at,
    }
    joblib.dump(calibration_artifact, model_dir / CALIBRATION_ARTIFACT_NAME)

    return {
        "selected_method": selected_method,
        "comparison_rows": comparison_rows,
        "bin_rows": bin_rows,
        "artifact": model_dir / CALIBRATION_ARTIFACT_NAME,
    }


def _build_uncalibrated_predictions(
    artifact: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Predict raw LightGBM probabilities for all evaluation splits."""
    prediction_frames = {}
    for split_name, frame in split_frames.items():
        probabilities = predict_probabilities(
            artifact,
            frame,
            feature_columns,
            f"{LIGHTGBM_MODEL_VERSION}_{split_name}",
            CalibrationError,
        )
        prediction_frames[split_name] = prediction_frame(frame, probabilities)
    return prediction_frames


def _build_comparison_outputs(
    predictions_by_method: dict[str, dict[str, pd.DataFrame]],
    manual_review_capacity_rate: float,
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build calibration comparison metric rows and bin rows."""
    comparison_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    for method, split_predictions in predictions_by_method.items():
        model_version = f"{LIGHTGBM_MODEL_VERSION}_{method}"
        bin_rows = _build_bin_rows(model_version, method, split_predictions, created_at)
        all_bin_rows.extend(bin_rows)
        bin_errors = _bin_error_summary(bin_rows)
        for split_name in REPORTING_SPLITS:
            frame = split_predictions[split_name]
            probabilities = frame["probability"].to_numpy()
            y_true = frame["target"]
            metrics = probability_metrics(
                y_true,
                probabilities,
                manual_review_capacity_rate,
                CalibrationError,
            )
            split_bin_errors = bin_errors[split_name]
            comparison_rows.append(
                {
                    "model_version": model_version,
                    "base_model_version": LIGHTGBM_MODEL_VERSION,
                    "calibration_method": method,
                    "split": split_name,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "brier_score": metrics["brier_score"],
                    "min_predicted_probability": metrics["min_predicted_probability"],
                    "max_predicted_probability": metrics["max_predicted_probability"],
                    "top_decile_lift": metrics["top_decile_lift"],
                    "precision_at_top_decile": metrics["precision_at_top_decile"],
                    "recall_at_manual_review_capacity": metrics[
                        "recall_at_manual_review_capacity"
                    ],
                    "mean_absolute_bin_error": split_bin_errors[
                        "mean_absolute_bin_error"
                    ],
                    "weighted_calibration_error": split_bin_errors[
                        "weighted_calibration_error"
                    ],
                    "max_absolute_bin_error": split_bin_errors[
                        "max_absolute_bin_error"
                    ],
                    "created_at": created_at,
                }
            )
    return comparison_rows, all_bin_rows


def _build_bin_rows(
    model_version: str,
    method: str,
    prediction_frames: dict[str, pd.DataFrame],
    created_at: str,
) -> list[dict[str, Any]]:
    """Build calibration-bin rows annotated with calibration method metadata."""
    return [
        {
            **row,
            "base_model_version": LIGHTGBM_MODEL_VERSION,
            "calibration_method": method,
            "created_at": created_at,
        }
        for row in build_calibration_bin_rows(
            model_version, prediction_frames, REPORTING_SPLITS
        )
    ]


def _bin_error_summary(bin_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize absolute calibration-bin errors for each reporting split."""
    summaries: dict[str, dict[str, float]] = {}
    for split_name in REPORTING_SPLITS:
        split_rows = [
            row
            for row in bin_rows
            if row["split"] == split_name and row["calibration_error"] is not None
        ]
        total_count = sum(int(row["applicant_count"]) for row in split_rows)
        absolute_errors = [abs(float(row["calibration_error"])) for row in split_rows]
        weighted_error = (
            sum(
                abs(float(row["calibration_error"])) * int(row["applicant_count"])
                for row in split_rows
            )
            / total_count
            if total_count
            else 0.0
        )
        summaries[split_name] = {
            "mean_absolute_bin_error": float(np.mean(absolute_errors))
            if absolute_errors
            else 0.0,
            "weighted_calibration_error": float(weighted_error),
            "max_absolute_bin_error": float(np.max(absolute_errors))
            if absolute_errors
            else 0.0,
        }
    return summaries


def main() -> None:
    """Run the calibration experiment CLI."""
    parser = argparse.ArgumentParser(
        description="Run post-v1 probability calibration comparison for the LightGBM model.",
    )
    add_config_argument(parser)
    args = parser.parse_args()

    try:
        run_calibration_experiment(args.config)
    except CalibrationError as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
