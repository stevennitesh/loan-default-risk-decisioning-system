from __future__ import annotations

import argparse
import csv
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from src.config import load_config
from src.train import BASELINE_MODEL_ARTIFACT_NAME
from src.train import BASELINE_MODEL_TYPE
from src.train import BASELINE_MODEL_VERSION
from src.train import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.train import LIGHTGBM_MODEL_TYPE
from src.train import LIGHTGBM_MODEL_VERSION
from src.train import MODEL_METRICS_SUMMARY_COLUMNS
from src.thresholding import MODEL_CONFUSION_MATRIX_COLUMNS
from src.thresholding import MODEL_THRESHOLD_METRICS_COLUMNS
from src.thresholding import ThresholdingError
from src.thresholding import build_confusion_matrix_rows
from src.thresholding import build_threshold_metric_rows
from src.thresholding import resolve_scenario_thresholds


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_LIFT_BY_DECILE_COLUMNS = [
    "model_version",
    "split",
    "decile",
    "applicant_count",
    "average_score",
    "observed_default_rate",
    "portfolio_default_rate",
    "lift",
    "cumulative_default_capture_rate",
]

MODEL_CALIBRATION_BINS_COLUMNS = [
    "model_version",
    "split",
    "bin_id",
    "applicant_count",
    "average_predicted_score",
    "observed_default_rate",
    "calibration_error",
]

EVALUATION_SPLITS = ("train", "validation", "test")
REPORTING_SPLITS = ("validation", "test")
MODEL_ARTIFACTS = {
    BASELINE_MODEL_TYPE: (BASELINE_MODEL_VERSION, BASELINE_MODEL_ARTIFACT_NAME),
    LIGHTGBM_MODEL_TYPE: (LIGHTGBM_MODEL_VERSION, LIGHTGBM_MODEL_ARTIFACT_NAME),
}


class EvaluationError(RuntimeError):
    """Raised when model evaluation cannot satisfy the Milestone 6 contract."""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def run_evaluation(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])
    report_dir = _resolve_project_path(config["paths"]["report_dir"])

    if not duckdb_path.exists():
        raise EvaluationError(f"DuckDB database not found: {duckdb_path}")

    artifacts = {
        model_type: _load_model_artifact(model_dir / artifact_name, model_type, model_version)
        for model_type, (model_version, artifact_name) in MODEL_ARTIFACTS.items()
    }
    feature_columns, split_applicant_ids = _validate_artifacts(artifacts)

    created_at = _created_at()
    with duckdb.connect(str(duckdb_path)) as connection:
        split_frames = _load_split_frames(connection, split_applicant_ids, feature_columns)
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
            scenario_thresholds = resolve_scenario_thresholds(
                config["threshold_policy"],
                selected_predictions["validation"]["probability"].to_numpy(),
            )
            threshold_rows = build_threshold_metric_rows(
                selected_model_version,
                str(config["threshold_policy"]["threshold_version"]),
                selected_predictions,
                scenario_thresholds,
                config["business_assumptions"],
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
        calibration_rows = _build_calibration_rows(selected_model_version, selected_predictions)

        report_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        _write_csv(report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS, metric_rows)
        _write_csv(report_dir / "model_lift_by_decile.csv", MODEL_LIFT_BY_DECILE_COLUMNS, lift_rows)
        _write_csv(
            report_dir / "model_calibration_bins.csv",
            MODEL_CALIBRATION_BINS_COLUMNS,
            calibration_rows,
        )
        _write_csv(
            report_dir / "model_confusion_matrix.csv",
            MODEL_CONFUSION_MATRIX_COLUMNS,
            confusion_rows,
        )
        _write_csv(
            report_dir / "model_threshold_metrics.csv",
            MODEL_THRESHOLD_METRICS_COLUMNS,
            threshold_rows,
        )
        _write_validation_report(
            report_dir / "validation_report.md",
            selected_model_type,
            selected_model_version,
            metric_rows,
            selected_artifact,
            scenario_thresholds,
            threshold_rows,
            config["business_assumptions"],
        )
        _write_business_value_report(
            report_dir / "business_value_analysis.md",
            selected_model_type,
            selected_model_version,
            threshold_rows,
            config["business_assumptions"],
        )
        _write_figures(figures_dir, selected_model_version, selected_predictions, lift_rows, calibration_rows)

        _replace_duckdb_table(connection, "model_metrics_summary", metric_rows)
        _replace_duckdb_table(connection, "model_lift_by_decile", lift_rows)
        _replace_duckdb_table(connection, "model_calibration_bins", calibration_rows)
        _replace_duckdb_table(connection, "model_confusion_matrix", confusion_rows)
        _replace_duckdb_table(connection, "model_threshold_metrics", threshold_rows)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model metrics and export reporting tables.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    parser.add_argument("--export-dashboard-data", action="store_true", help="Export Power BI-ready dashboard data.")
    args = parser.parse_args()

    if args.export_dashboard_data:
        raise SystemExit("Milestone 10 not implemented yet: dashboard exports require evaluation outputs.")

    try:
        run_evaluation(args.config)
    except EvaluationError as error:
        raise SystemExit(str(error)) from error


def _load_model_artifact(path: Path, model_type: str, model_version: str) -> dict[str, Any]:
    if not path.exists():
        raise EvaluationError(f"Missing model artifact: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise EvaluationError(f"Model artifact must be a dict: {path}")

    required_keys = {
        "pipeline",
        "model_version",
        "model_type",
        "feature_columns",
        "split_applicant_ids",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise EvaluationError(f"Model artifact {path.name} is missing required keys: {missing_keys}")
    if artifact["model_type"] != model_type:
        raise EvaluationError(
            f"Model artifact {path.name} has model_type={artifact['model_type']}, expected {model_type}"
        )
    if artifact["model_version"] != model_version:
        raise EvaluationError(
            f"Model artifact {path.name} has model_version={artifact['model_version']}, expected {model_version}"
        )
    return artifact


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

    split_applicant_ids = _normalize_split_ids(baseline["split_applicant_ids"])
    if split_applicant_ids != _normalize_split_ids(lightgbm["split_applicant_ids"]):
        raise EvaluationError("Model artifacts must use the same split_applicant_ids")
    return feature_columns, split_applicant_ids


def _normalize_split_ids(raw_split_ids: Any) -> dict[str, list[int]]:
    if not isinstance(raw_split_ids, dict):
        raise EvaluationError("split_applicant_ids must be a mapping")
    missing_splits = [split for split in EVALUATION_SPLITS if split not in raw_split_ids]
    if missing_splits:
        raise EvaluationError(f"split_applicant_ids is missing splits: {missing_splits}")

    split_ids: dict[str, list[int]] = {}
    for split_name in EVALUATION_SPLITS:
        ids = [int(value) for value in raw_split_ids[split_name]]
        if not ids:
            raise EvaluationError(f"split_applicant_ids[{split_name}] must not be empty")
        if len(ids) != len(set(ids)):
            raise EvaluationError(f"split_applicant_ids[{split_name}] contains duplicate applicants")
        split_ids[split_name] = ids
    return split_ids


def _load_split_frames(
    connection: duckdb.DuckDBPyConnection,
    split_applicant_ids: dict[str, list[int]],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    _require_table(connection, "mart_credit_risk_features")
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    split_frames = {}

    for split_name, applicant_ids in split_applicant_ids.items():
        ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
        connection.register("split_ids", ids_frame)
        try:
            frame = connection.execute(
                f"""
                SELECT {", ".join(_sql_identifier(column) for column in selected_columns)}
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
            raise EvaluationError(
                f"Saved split IDs no longer reconcile to mart_credit_risk_features for "
                f"{split_name}: missing {missing_ids[:10]}"
            )
        targets = set(frame["TARGET"].astype(int).unique())
        if targets != {0, 1}:
            raise EvaluationError(f"{split_name} split must contain binary TARGET classes, got {sorted(targets)}")
        split_frames[split_name] = frame.reset_index(drop=True)

    return split_frames


def _build_prediction_frames(
    artifact: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise EvaluationError(f"Model {artifact['model_version']} does not expose predict_proba")

    prediction_frames = {}
    for split_name, frame in split_frames.items():
        probabilities = pipeline.predict_proba(_feature_frame(frame, feature_columns))[:, 1]
        _validate_probabilities(probabilities, artifact["model_version"], split_name)
        prediction_frames[split_name] = pd.DataFrame(
            {
                "SK_ID_CURR": frame["SK_ID_CURR"].astype(int),
                "target": frame["TARGET"].astype(int),
                "probability": probabilities.astype(float),
            }
        )
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
    manual_review_capacity_rate = float(config["business_assumptions"]["manual_review_capacity_rate"])

    for model_type, split_predictions in prediction_frames.items():
        model_version = model_versions[model_type]
        for split_name, frame in split_predictions.items():
            y_true = frame["target"]
            probabilities = frame["probability"].to_numpy()
            metrics = {
                "roc_auc": roc_auc_score(y_true, probabilities),
                "pr_auc": average_precision_score(y_true, probabilities),
                "brier_score": brier_score_loss(y_true, probabilities),
                "min_predicted_probability": float(np.min(probabilities)),
                "max_predicted_probability": float(np.max(probabilities)),
                "top_decile_lift": _top_decile_lift(y_true, probabilities),
                "precision_at_top_decile": _precision_at_rate(y_true, probabilities, 0.10),
                "recall_at_manual_review_capacity": _recall_at_rate(
                    y_true,
                    probabilities,
                    manual_review_capacity_rate,
                ),
            }
            rows.extend(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "created_at": created_at,
                }
                for metric_name, metric_value in metrics.items()
            )
    return rows


def _build_lift_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in REPORTING_SPLITS:
        frame = _with_rank_bin(prediction_frames[split_name], "decile", descending=True)
        total_defaults = int(frame["target"].sum())
        portfolio_default_rate = float(frame["target"].mean())
        cumulative_defaults = 0

        for decile in range(1, 11):
            decile_frame = frame.loc[frame["decile"] == decile]
            applicant_count = len(decile_frame)
            observed_default_rate = _nullable_mean(decile_frame["target"])
            if applicant_count:
                cumulative_defaults += int(decile_frame["target"].sum())
            rows.append(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "decile": decile,
                    "applicant_count": applicant_count,
                    "average_score": _nullable_mean(decile_frame["probability"]),
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


def _build_calibration_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in REPORTING_SPLITS:
        frame = _with_rank_bin(prediction_frames[split_name], "bin_id", descending=False)
        for bin_id in range(1, 11):
            bin_frame = frame.loc[frame["bin_id"] == bin_id]
            average_predicted_score = _nullable_mean(bin_frame["probability"])
            observed_default_rate = _nullable_mean(bin_frame["target"])
            rows.append(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "bin_id": bin_id,
                    "applicant_count": len(bin_frame),
                    "average_predicted_score": average_predicted_score,
                    "observed_default_rate": observed_default_rate,
                    "calibration_error": observed_default_rate - average_predicted_score
                    if observed_default_rate is not None and average_predicted_score is not None
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
    return (
        LIGHTGBM_MODEL_TYPE
        if validation_metrics[LIGHTGBM_MODEL_VERSION] >= validation_metrics[BASELINE_MODEL_VERSION]
        else BASELINE_MODEL_TYPE
    )


def _verify_saved_model_selection(
    connection: duckdb.DuckDBPyConnection,
    selected_model_type: str,
) -> None:
    if "model_comparison_summary" not in _existing_tables(connection):
        return
    saved_selections = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT selected_model_type FROM model_comparison_summary"
        ).fetchall()
    }
    if saved_selections and saved_selections != {selected_model_type}:
        raise EvaluationError(
            "Saved model_comparison_summary selection does not match recomputed validation PR-AUC "
            f"selection: saved={sorted(saved_selections)}, recomputed={selected_model_type}"
        )


def _write_validation_report(
    path: Path,
    selected_model_type: str,
    selected_model_version: str,
    metric_rows: list[dict[str, Any]],
    selected_artifact: dict[str, Any],
    scenario_thresholds: dict[str, dict[str, float]],
    threshold_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
) -> None:
    metrics = {
        (row["model_version"], row["split"], row["metric_name"]): float(row["metric_value"])
        for row in metric_rows
    }
    split_summary_rows = selected_artifact.get("split_summary", [])
    scenario_lines = "\n".join(
        f"- {scenario}: threshold_low={thresholds['threshold_low']:.6f}, "
        f"threshold_high={thresholds['threshold_high']:.6f}"
        for scenario, thresholds in scenario_thresholds.items()
    )
    split_lines = "\n".join(
        f"- {row['split']}: {row['row_count']} rows, positive_rate={float(row['positive_rate']):.4f}"
        for row in split_summary_rows
    )
    metric_lines = "\n".join(
        f"- {split}: PR-AUC={metrics[(selected_model_version, split, 'pr_auc')]:.6f}, "
        f"ROC-AUC={metrics[(selected_model_version, split, 'roc_auc')]:.6f}, "
        f"Brier={metrics[(selected_model_version, split, 'brier_score')]:.6f}, "
        f"top-decile lift={metrics[(selected_model_version, split, 'top_decile_lift')]:.6f}"
        for split in REPORTING_SPLITS
    )
    balanced_rows = [
        row
        for row in threshold_rows
        if row["scenario_name"] == "balanced" and row["split"] in REPORTING_SPLITS
    ]
    balanced_lines = "\n".join(
        f"- {row['split']}: approval_rate={row['approval_rate']:.4f}, "
        f"manual_review_rate={row['manual_review_rate']:.4f}, "
        f"high_risk_rate={row['high_risk_rate']:.4f}, "
        f"expected_value_per_applicant={row['expected_value_per_applicant']:.2f}"
        for row in balanced_rows
    )
    text = f"""# Validation Report

## Executive Summary

Selected model: `{selected_model_type}` (`{selected_model_version}`), using validation PR-AUC from the saved Milestone 5 split.

Kaggle application_test rows are not used for evaluation metrics. The test results below refer only to the held-out labeled split from `application_train`.

## Split Strategy

{split_lines}

## Selected Model Results

{metric_lines}

## Calibration Analysis

Calibration is evaluated with Brier score and calibration bins. No Platt or isotonic calibration model is fitted in Milestone 6.

## Lift and Decile Analysis

`model_lift_by_decile` reports validation and held-out test deciles with decile 1 representing the highest-risk applicants.

## Threshold Scenario Analysis

The following validation-derived thresholds are used only to produce confusion matrices in Milestone 6:

{scenario_lines}

Manual-review handling is explicit: the confusion matrix treats only the high-risk action as the positive prediction.

## Business-Value Analysis

Threshold expected-value analysis is produced in `model_threshold_metrics` and `reports/business_value_analysis.md`.

Business assumptions:

- Expected margin per good approved loan: {assumptions['expected_margin_per_good_loan']}
- Expected loss per bad approved loan: {assumptions['expected_loss_per_bad_loan']}
- Manual review cost: {assumptions['manual_review_cost']}

Balanced scenario summary:

{balanced_lines}

## Limitations

This is a portfolio decision-support simulation, not a production credit-decisioning system. Metrics describe labeled holdout behavior and should not be interpreted as production underwriting readiness.
"""
    path.write_text(text, encoding="utf-8")


def _write_business_value_report(
    path: Path,
    selected_model_type: str,
    selected_model_version: str,
    threshold_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
) -> None:
    rows = sorted(
        threshold_rows,
        key=lambda row: (row["split"], row["scenario_name"]),
    )
    table_lines = "\n".join(
        "| {split} | {scenario_name} | {approval_rate:.4f} | {manual_review_rate:.4f} | "
        "{high_risk_rate:.4f} | {default_rate_approved:.4f} | "
        "{high_risk_default_capture_rate:.4f} | {expected_value:.2f} | "
        "{expected_value_per_applicant:.2f} |".format(**row)
        for row in rows
    )
    text = f"""# Business Value Analysis

Selected model: `{selected_model_type}` (`{selected_model_version}`).

Thresholds are selected from validation scores and applied unchanged to the held-out labeled test split. Kaggle `application_test` rows are not included.

## Assumptions

- Expected margin per good approved loan: {assumptions['expected_margin_per_good_loan']}
- Expected loss per bad approved loan: {assumptions['expected_loss_per_bad_loan']}
- Manual review cost: {assumptions['manual_review_cost']}
- Manual review capacity rate: {assumptions['manual_review_capacity_rate']}

## Scenario Metrics

| Split | Scenario | Approval Rate | Review Rate | High-Risk Rate | Approved Default Rate | High-Risk Default Capture | Expected Value | EV / Applicant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table_lines}

## Notes

The expected-value formula is:

`approved_good_count * expected_margin_per_good_loan - approved_bad_count * expected_loss_per_bad_loan - manual_review_count * manual_review_cost`.

High-risk applicants contribute no approved-loan margin or loss in this simulation.
"""
    path.write_text(text, encoding="utf-8")


def _write_figures(
    figures_dir: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    lift_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> None:
    _write_roc_curve(figures_dir / "roc_curve.png", model_version, prediction_frames)
    _write_pr_curve(figures_dir / "pr_curve.png", model_version, prediction_frames)
    _write_calibration_curve(figures_dir / "calibration_curve.png", model_version, calibration_rows)
    _write_lift_chart(figures_dir / "lift_chart.png", model_version, lift_rows)


def _write_roc_curve(
    path: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        frame = prediction_frames[split_name]
        fpr, tpr, _ = roc_curve(frame["target"], frame["probability"])
        auc = roc_auc_score(frame["target"], frame["probability"])
        axis.plot(fpr, tpr, label=f"{split_name} AUC={auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    axis.set_title(f"ROC Curve - {model_version}")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_pr_curve(
    path: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        frame = prediction_frames[split_name]
        precision, recall, _ = precision_recall_curve(frame["target"], frame["probability"])
        pr_auc = average_precision_score(frame["target"], frame["probability"])
        axis.plot(recall, precision, label=f"{split_name} PR-AUC={pr_auc:.3f}")
    axis.set_title(f"Precision-Recall Curve - {model_version}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_calibration_curve(
    path: Path,
    model_version: str,
    calibration_rows: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        rows = [row for row in calibration_rows if row["split"] == split_name and row["applicant_count"]]
        axis.plot(
            [row["average_predicted_score"] for row in rows],
            [row["observed_default_rate"] for row in rows],
            marker="o",
            label=split_name,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    axis.set_title(f"Calibration Curve - {model_version}")
    axis.set_xlabel("Average predicted score")
    axis.set_ylabel("Observed default rate")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_lift_chart(
    path: Path,
    model_version: str,
    lift_rows: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        rows = [row for row in lift_rows if row["split"] == split_name and row["applicant_count"]]
        axis.plot(
            [row["decile"] for row in rows],
            [row["lift"] for row in rows],
            marker="o",
            label=split_name,
        )
    axis.set_title(f"Lift by Decile - {model_version}")
    axis.set_xlabel("Risk decile (1 = highest risk)")
    axis.set_ylabel("Lift")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _top_decile_lift(y_true: pd.Series, probabilities: np.ndarray) -> float:
    portfolio_positive_rate = float(y_true.mean())
    top_precision = _precision_at_rate(y_true, probabilities, 0.10)
    return float(top_precision / portfolio_positive_rate)


def _precision_at_rate(y_true: pd.Series, probabilities: np.ndarray, rate: float) -> float:
    top_count = _top_count(len(y_true), rate)
    frame = pd.DataFrame({"target": y_true.to_numpy(), "probability": probabilities})
    return float(frame.sort_values("probability", ascending=False).head(top_count)["target"].mean())


def _recall_at_rate(y_true: pd.Series, probabilities: np.ndarray, rate: float) -> float:
    top_count = _top_count(len(y_true), rate)
    frame = pd.DataFrame({"target": y_true.to_numpy(), "probability": probabilities})
    positives_in_top = int(frame.sort_values("probability", ascending=False).head(top_count)["target"].sum())
    total_positives = int(frame["target"].sum())
    return float(positives_in_top / total_positives) if total_positives else 0.0


def _top_count(row_count: int, rate: float) -> int:
    if rate <= 0 or rate > 1:
        raise EvaluationError(f"Selection rate must be in (0, 1], got {rate}")
    return max(1, int(np.ceil(row_count * rate)))


def _with_rank_bin(frame: pd.DataFrame, column_name: str, descending: bool) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["probability", "SK_ID_CURR"],
        ascending=[not descending, True],
    ).reset_index(drop=True)
    ranked[column_name] = np.ceil((np.arange(len(ranked)) + 1) * 10 / len(ranked)).astype(int)
    ranked[column_name] = ranked[column_name].clip(1, 10)
    return ranked


def _nullable_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.mean())


def _feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    return features.where(pd.notna(features), np.nan)


def _validate_probabilities(probabilities: np.ndarray, model_version: str, split_name: str) -> None:
    if probabilities.ndim != 1:
        raise EvaluationError(f"{model_version} {split_name} probabilities must be one-dimensional")
    if not np.isfinite(probabilities).all():
        raise EvaluationError(f"{model_version} {split_name} probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise EvaluationError(f"{model_version} {split_name} probabilities must be in [0, 1]")


def _replace_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(rows)
    connection.register("output_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {_sql_identifier(table_name)} AS SELECT * FROM output_frame")
    connection.unregister("output_frame")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    if table_name not in _existing_tables(connection):
        raise EvaluationError(f"Missing required DuckDB table: {table_name}")


def _existing_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _created_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


if __name__ == "__main__":
    main()
