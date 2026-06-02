from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.config import business_assumptions
from src.config import load_config
from src.config import manual_review_capacity_rate
from src.config import threshold_policy
from src.config import threshold_version
from src.evaluation_reports import write_business_value_report
from src.evaluation_reports import write_figures
from src.evaluation_reports import write_validation_report
from src.metrics import build_calibration_bin_rows
from src.metrics import build_probability_metric_rows
from src.metrics import nullable_mean
from src.metrics import with_probability_rank_bin
from src.mart_access import load_labeled_split_frames
from src.model_contracts import BASELINE_MODEL_TYPE
from src.model_contracts import BASELINE_MODEL_VERSION
from src.model_contracts import EVALUATION_SPLITS
from src.model_contracts import LIGHTGBM_MODEL_TYPE
from src.model_contracts import LIGHTGBM_MODEL_VERSION
from src.model_contracts import MODEL_ARTIFACTS
from src.model_contracts import REPORTING_SPLITS
from src.model_contracts import select_model_type_by_validation_pr_auc
from src.model_artifacts import load_model_artifact
from src.model_artifacts import normalize_split_ids
from src.model_artifacts import selected_model_types
from src.modeling import predict_probabilities
from src.modeling import prediction_frame
from src.report_contracts import MODEL_CALIBRATION_BINS_COLUMNS
from src.report_contracts import MODEL_CONFUSION_MATRIX_COLUMNS
from src.report_contracts import MODEL_LIFT_BY_DECILE_COLUMNS
from src.report_contracts import MODEL_METRICS_SUMMARY_COLUMNS
from src.report_contracts import MODEL_THRESHOLD_METRICS_COLUMNS
from src.runtime import created_at_utc
from src.runtime import replace_duckdb_table
from src.runtime import resolve_config_path
from src.runtime import write_csv
from src.thresholding import ThresholdingError
from src.thresholding import build_confusion_matrix_rows
from src.thresholding import build_threshold_metric_rows
from src.thresholding import resolve_scenario_thresholds


class EvaluationError(RuntimeError):
    """Raised when model evaluation cannot satisfy the Milestone 6 contract."""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def run_evaluation(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    if not duckdb_path.exists():
        raise EvaluationError(f"DuckDB database not found: {duckdb_path}")

    artifacts = {
        model_type: load_model_artifact(
            model_dir / artifact_name,
            expected_model_type=model_type,
            expected_model_version=model_version,
            error_cls=EvaluationError,
            artifact_label=f"Model artifact {artifact_name}",
            missing_label="model artifact",
        )
        for model_type, (model_version, artifact_name) in MODEL_ARTIFACTS.items()
    }
    # Evaluation compares model families only when they were trained on the same features and split IDs.
    feature_columns, split_applicant_ids = _validate_artifacts(artifacts)

    created_at = created_at_utc()
    with duckdb.connect(str(duckdb_path)) as connection:
        split_frames = load_labeled_split_frames(
            connection,
            split_applicant_ids,
            feature_columns,
            error_cls=EvaluationError,
        )
        prediction_frames = {
            model_type: _build_prediction_frames(artifact, split_frames, feature_columns)
            for model_type, artifact in artifacts.items()
        }
        metric_rows = _build_metric_rows(prediction_frames, created_at, config)
        selected_model_type = _select_model_type(metric_rows)
        _verify_saved_model_selection(connection, selected_model_type)

        selected_artifact = artifacts[selected_model_type]
        selected_model_version = str(selected_artifact["model_version"])
        selected_predictions = prediction_frames[selected_model_type]
        try:
            # Thresholds are selected from validation predictions and then applied unchanged to test.
            scenario_thresholds = resolve_scenario_thresholds(
                threshold_policy(config),
                selected_predictions["validation"]["probability"].to_numpy(),
            )
            threshold_rows = build_threshold_metric_rows(
                selected_model_version,
                threshold_version(config),
                selected_predictions,
                scenario_thresholds,
                business_assumptions(config),
                created_at,
            )
            confusion_rows = build_confusion_matrix_rows(
                selected_model_version,
                selected_predictions,
                scenario_thresholds,
            )
        except ThresholdingError as error:
            raise EvaluationError(f"Threshold policy validation failed: {error}") from error
        lift_rows = _build_lift_rows(selected_model_version, selected_predictions)
        calibration_rows = build_calibration_bin_rows(
            selected_model_version,
            selected_predictions,
            REPORTING_SPLITS,
        )

        report_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        write_csv(report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS, metric_rows)
        write_csv(report_dir / "model_lift_by_decile.csv", MODEL_LIFT_BY_DECILE_COLUMNS, lift_rows)
        write_csv(
            report_dir / "model_calibration_bins.csv",
            MODEL_CALIBRATION_BINS_COLUMNS,
            calibration_rows,
        )
        write_csv(
            report_dir / "model_confusion_matrix.csv",
            MODEL_CONFUSION_MATRIX_COLUMNS,
            confusion_rows,
        )
        write_csv(
            report_dir / "model_threshold_metrics.csv",
            MODEL_THRESHOLD_METRICS_COLUMNS,
            threshold_rows,
        )
        write_validation_report(
            report_dir / "validation_report.md",
            selected_model_type,
            selected_model_version,
            metric_rows,
            selected_artifact,
            scenario_thresholds,
            threshold_rows,
            business_assumptions(config),
        )
        write_business_value_report(
            report_dir / "business_value_analysis.md",
            selected_model_type,
            selected_model_version,
            threshold_rows,
            business_assumptions(config),
        )
        write_figures(figures_dir, selected_model_version, selected_predictions, lift_rows, calibration_rows)

        replace_duckdb_table(connection, "model_metrics_summary", metric_rows)
        replace_duckdb_table(connection, "model_lift_by_decile", lift_rows)
        replace_duckdb_table(connection, "model_calibration_bins", calibration_rows)
        replace_duckdb_table(connection, "model_confusion_matrix", confusion_rows)
        replace_duckdb_table(connection, "model_threshold_metrics", threshold_rows)

    return {
        "selected_model_type": selected_model_type,
        "selected_model_version": selected_model_version,
        "scenario_thresholds": scenario_thresholds,
        "metric_rows": metric_rows,
        "lift_rows": lift_rows,
        "calibration_rows": calibration_rows,
        "confusion_rows": confusion_rows,
        "threshold_rows": threshold_rows,
    }


def _validate_artifacts(
    artifacts: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, list[int]]]:
    baseline = artifacts[BASELINE_MODEL_TYPE]
    lightgbm = artifacts[LIGHTGBM_MODEL_TYPE]

    feature_columns = list(baseline["feature_columns"])
    if feature_columns != list(lightgbm["feature_columns"]):
        raise EvaluationError("Model artifacts must use the same feature_columns")
    if not feature_columns:
        raise EvaluationError("Model artifacts do not contain any feature_columns")

    split_applicant_ids = normalize_split_ids(
        baseline["split_applicant_ids"],
        EVALUATION_SPLITS,
        error_cls=EvaluationError,
    )
    lightgbm_split_applicant_ids = normalize_split_ids(
        lightgbm["split_applicant_ids"],
        EVALUATION_SPLITS,
        error_cls=EvaluationError,
    )
    if split_applicant_ids != lightgbm_split_applicant_ids:
        raise EvaluationError("Model artifacts must use the same split_applicant_ids")
    return feature_columns, split_applicant_ids


def _build_prediction_frames(
    artifact: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    prediction_frames = {}
    for split_name, frame in split_frames.items():
        probabilities = predict_probabilities(
            artifact,
            frame,
            feature_columns,
            f"{artifact['model_version']} {split_name}",
            EvaluationError,
        )
        prediction_frames[split_name] = prediction_frame(frame, probabilities)
    return prediction_frames


def _build_metric_rows(
    prediction_frames: dict[str, dict[str, pd.DataFrame]],
    created_at: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    model_versions = {
        BASELINE_MODEL_TYPE: BASELINE_MODEL_VERSION,
        LIGHTGBM_MODEL_TYPE: LIGHTGBM_MODEL_VERSION,
    }
    rows: list[dict[str, Any]] = []
    review_capacity_rate = manual_review_capacity_rate(config)

    for model_type, split_predictions in prediction_frames.items():
        model_version = model_versions[model_type]
        rows.extend(
            build_probability_metric_rows(
                model_version,
                split_predictions,
                created_at,
                review_capacity_rate,
                error_cls=EvaluationError,
            )
        )
    return rows


def _build_lift_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in REPORTING_SPLITS:
        frame = with_probability_rank_bin(prediction_frames[split_name], "decile", descending=True)
        total_defaults = int(frame["target"].sum())
        portfolio_default_rate = float(frame["target"].mean())
        cumulative_defaults = 0

        for decile in range(1, 11):
            decile_frame = frame.loc[frame["decile"] == decile]
            applicant_count = len(decile_frame)
            observed_default_rate = nullable_mean(decile_frame["target"])
            if applicant_count:
                cumulative_defaults += int(decile_frame["target"].sum())
            rows.append(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "decile": decile,
                    "applicant_count": applicant_count,
                    "average_score": nullable_mean(decile_frame["probability"]),
                    "observed_default_rate": observed_default_rate,
                    "portfolio_default_rate": portfolio_default_rate,
                    "lift": observed_default_rate / portfolio_default_rate
                    if observed_default_rate is not None and portfolio_default_rate
                    else None,
                    "cumulative_default_capture_rate": cumulative_defaults / total_defaults
                    if total_defaults
                    else None,
                }
            )
    return rows


def _select_model_type(metric_rows: list[dict[str, Any]]) -> str:
    validation_metrics = {
        row["model_version"]: float(row["metric_value"])
        for row in metric_rows
        if row["split"] == "validation" and row["metric_name"] == "pr_auc"
    }
    return select_model_type_by_validation_pr_auc(
        validation_metrics[BASELINE_MODEL_VERSION],
        validation_metrics[LIGHTGBM_MODEL_VERSION],
    )


def _verify_saved_model_selection(
    connection: duckdb.DuckDBPyConnection,
    selected_model_type: str,
) -> None:
    saved_selections = selected_model_types(connection)
    if saved_selections and saved_selections != {selected_model_type}:
        raise EvaluationError(
            "Saved model_comparison_summary selection does not match recomputed validation PR-AUC "
            f"selection: saved={sorted(saved_selections)}, recomputed={selected_model_type}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model metrics and export reporting tables.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    parser.add_argument("--export-dashboard-data", action="store_true", help="Export Power BI-ready dashboard data.")
    parser.add_argument(
        "--dashboard-export-dir",
        default=None,
        help="Optional override for the Power BI CSV export directory.",
    )
    parser.add_argument(
        "--use-calibrated-dashboard-metrics",
        action="store_true",
        help="Apply the selected calibration artifact to Power BI probability-quality tables.",
    )
    args = parser.parse_args()

    if args.export_dashboard_data:
        # Keep dashboard export imports local so normal evaluation does not depend on export helpers.
        from src.dashboard_exports import DashboardExportError
        from src.dashboard_exports import run_dashboard_export

        try:
            run_dashboard_export(
                args.config,
                export_dir=args.dashboard_export_dir,
                use_calibrated_probability_quality=args.use_calibrated_dashboard_metrics,
            )
        except DashboardExportError as error:
            raise SystemExit(str(error)) from error
        return

    try:
        run_evaluation(args.config)
    except EvaluationError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
