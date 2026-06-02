from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.calibration import SIGMOID_METHOD, UNCALIBRATED_METHOD
from src.model_contracts import REPORTING_SPLITS
from src.runtime import read_csv
from src.thresholding import BALANCED_SCENARIO

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_validation_report(
    path: Path,
    selected_model_type: str,
    selected_model_version: str,
    metric_rows: list[dict[str, Any]],
    selected_artifact: dict[str, Any],
    scenario_thresholds: dict[str, dict[str, float]],
    threshold_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
) -> None:
    """Write the markdown validation report from evaluated metric rows."""
    metrics = {
        (row["model_version"], row["split"], row["metric_name"]): float(
            row["metric_value"]
        )
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
        if row["scenario_name"] == BALANCED_SCENARIO
        and row["split"] in REPORTING_SPLITS
    ]
    balanced_lines = "\n".join(
        f"- {row['split']}: approval_rate={row['approval_rate']:.4f}, "
        f"manual_review_rate={row['manual_review_rate']:.4f}, "
        f"high_risk_rate={row['high_risk_rate']:.4f}, "
        f"expected_value_per_applicant={row['expected_value_per_applicant']:.2f}"
        for row in balanced_rows
    )
    calibration_experiment_section = _build_post_v1_calibration_section(path.parent)
    text = f"""# Validation Report

## Executive Summary

Selected model: `{selected_model_type}` (`{selected_model_version}`), using validation PR-AUC from the saved Milestone 5 split.

Kaggle application_test rows are not used for evaluation metrics. The test results below refer only to the held-out labeled split from `application_train`.

## Split Strategy

{split_lines}

## Selected Model Results

{metric_lines}

## Calibration Analysis

Calibration is evaluated with Brier score and calibration bins. No calibration layer is fitted by the Milestone 6 evaluation path; post-v1 calibration experiments are reported separately.

{calibration_experiment_section}

## Lift and Decile Analysis

`model_lift_by_decile` reports validation and held-out test deciles with decile 1 representing the highest-risk applicants.

## Threshold Scenario Analysis

The following validation-derived thresholds are used only to produce confusion matrices in Milestone 6:

{scenario_lines}

Manual-review handling is explicit: the confusion matrix treats only the high-risk action as the positive prediction.

## Business-Value Analysis

Threshold expected-value analysis is produced in `model_threshold_metrics` and `reports/business_value_analysis.md`.

Business assumptions:

- Expected margin per good approved loan: {assumptions["expected_margin_per_good_loan"]}
- Expected loss per bad approved loan: {assumptions["expected_loss_per_bad_loan"]}
- Manual review cost: {assumptions["manual_review_cost"]}

Balanced scenario summary:

{balanced_lines}

## Limitations

This is a portfolio decision-support simulation, not a production credit-decisioning system. Metrics describe labeled holdout behavior and should not be interpreted as production underwriting readiness.
"""
    path.write_text(text, encoding="utf-8")


def write_business_value_report(
    path: Path,
    selected_model_type: str,
    selected_model_version: str,
    threshold_rows: list[dict[str, Any]],
    assumptions: dict[str, Any],
) -> None:
    """Write the markdown business-value report from threshold metric rows."""
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
    calibration_note = _build_business_value_calibration_note(path.parent)
    text = f"""# Business Value Analysis

Selected model: `{selected_model_type}` (`{selected_model_version}`).

Thresholds are selected from validation scores and applied unchanged to the held-out labeled test split. Kaggle `application_test` rows are not included.

## Assumptions

- Expected margin per good approved loan: {assumptions["expected_margin_per_good_loan"]}
- Expected loss per bad approved loan: {assumptions["expected_loss_per_bad_loan"]}
- Manual review cost: {assumptions["manual_review_cost"]}
- Manual review capacity rate: {assumptions["manual_review_capacity_rate"]}

These values are scenario-comparison utility weights, not calibrated Home Credit economics. The good-loan margin and bad-loan loss encode a simple penalty ratio so approval, review, and high-risk policies can be compared in v1. A production value model would scale margin and loss by exposure, term, pricing, funding cost, recovery, and loss-given-default assumptions.

## Scenario Metrics

| Split | Scenario | Approval Rate | Review Rate | High-Risk Rate | Approved Default Rate | High-Risk Default Capture | Expected Value | EV / Applicant |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table_lines}

## Notes

The expected-value formula is:

`approved_good_count * expected_margin_per_good_loan - approved_bad_count * expected_loss_per_bad_loan - manual_review_count * manual_review_cost`.

High-risk applicants contribute no approved-loan margin or loss in this simulation.

{calibration_note}
"""
    path.write_text(text, encoding="utf-8")


def write_figures(
    figures_dir: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    lift_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> None:
    """Write evaluation ROC, PR, calibration, and lift figures."""
    _write_roc_curve(figures_dir / "roc_curve.png", model_version, prediction_frames)
    _write_pr_curve(figures_dir / "pr_curve.png", model_version, prediction_frames)
    _write_calibration_curve(
        figures_dir / "calibration_curve.png", model_version, calibration_rows
    )
    _write_lift_chart(figures_dir / "lift_chart.png", model_version, lift_rows)


def _build_post_v1_calibration_section(report_dir: Path) -> str:
    """Build optional validation-report text from calibration comparison output."""
    row_lookup = _calibration_comparison_lookup(report_dir)
    if not row_lookup:
        return ""
    required_keys = [
        (UNCALIBRATED_METHOD, "validation"),
        (UNCALIBRATED_METHOD, "test"),
        (SIGMOID_METHOD, "validation"),
        (SIGMOID_METHOD, "test"),
    ]
    if any(key not in row_lookup for key in required_keys):
        return ""

    validation_uncalibrated = row_lookup[(UNCALIBRATED_METHOD, "validation")]
    validation_sigmoid = row_lookup[(SIGMOID_METHOD, "validation")]
    test_uncalibrated = row_lookup[(UNCALIBRATED_METHOD, "test")]
    test_sigmoid = row_lookup[(SIGMOID_METHOD, "test")]
    return f"""Post-v1 Experiment 004 adds a separate sigmoid calibration layer for the Experiment 003 LightGBM model. This is a large improvement in probability quality: held-out test Brier score improves from {float(test_uncalibrated["brier_score"]):.6f} to {float(test_sigmoid["brier_score"]):.6f}, and held-out test weighted calibration-bin error improves from {float(test_uncalibrated["weighted_calibration_error"]):.6f} to {float(test_sigmoid["weighted_calibration_error"]):.6f}. PR-AUC, ROC-AUC, top-decile lift, precision at top decile, recall at review capacity, and expected value are unchanged because sigmoid calibration is monotonic.

| Split | Uncalibrated Brier | Sigmoid Brier | Uncalibrated weighted bin error | Sigmoid weighted bin error |
|---|---:|---:|---:|---:|
| validation | {float(validation_uncalibrated["brier_score"]):.6f} | {float(validation_sigmoid["brier_score"]):.6f} | {float(validation_uncalibrated["weighted_calibration_error"]):.6f} | {float(validation_sigmoid["weighted_calibration_error"]):.6f} |
| test | {float(test_uncalibrated["brier_score"]):.6f} | {float(test_sigmoid["brier_score"]):.6f} | {float(test_uncalibrated["weighted_calibration_error"]):.6f} | {float(test_sigmoid["weighted_calibration_error"]):.6f} |"""


def _build_business_value_calibration_note(report_dir: Path) -> str:
    """Build optional business-value text for the saved calibration experiment."""
    row_lookup = _calibration_comparison_lookup(report_dir)
    if not row_lookup:
        return ""
    if (UNCALIBRATED_METHOD, "test") not in row_lookup or (
        SIGMOID_METHOD,
        "test",
    ) not in row_lookup:
        return ""

    test_uncalibrated = row_lookup[(UNCALIBRATED_METHOD, "test")]
    test_sigmoid = row_lookup[(SIGMOID_METHOD, "test")]
    return f"""## Calibration Note

Post-v1 Experiment 004 materially improves probability quality with a separate sigmoid calibration layer. Because the current threshold and expected-value workflow is rank-based, sigmoid calibration does not change the action ordering, scenario thresholds, or expected-value metrics shown above. It does improve the interpretability of the model score scale: held-out test Brier score improves from {float(test_uncalibrated["brier_score"]):.6f} to {float(test_sigmoid["brier_score"]):.6f}, and held-out test weighted calibration-bin error improves from {float(test_uncalibrated["weighted_calibration_error"]):.6f} to {float(test_sigmoid["weighted_calibration_error"]):.6f}."""


def _calibration_comparison_lookup(
    report_dir: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    """Load calibration comparison rows keyed by method and split."""
    comparison_path = report_dir / "model_calibration_comparison.csv"
    if not comparison_path.exists():
        return {}

    return {
        (row["calibration_method"], row["split"]): row
        for row in read_csv(comparison_path)
    }


def _write_roc_curve(
    path: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    """Write ROC curve figure for reporting splits."""
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
    _save_figure(path, figure)


def _write_pr_curve(
    path: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    """Write precision-recall curve figure for reporting splits."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        frame = prediction_frames[split_name]
        precision, recall, _ = precision_recall_curve(
            frame["target"], frame["probability"]
        )
        pr_auc = average_precision_score(frame["target"], frame["probability"])
        axis.plot(recall, precision, label=f"{split_name} PR-AUC={pr_auc:.3f}")
    axis.set_title(f"Precision-Recall Curve - {model_version}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(path, figure)


def _write_calibration_curve(
    path: Path,
    model_version: str,
    calibration_rows: list[dict[str, Any]],
) -> None:
    """Write observed-vs-predicted calibration curve figure."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        rows = [
            row
            for row in calibration_rows
            if row["split"] == split_name and row["applicant_count"]
        ]
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
    _save_figure(path, figure)


def _write_lift_chart(
    path: Path,
    model_version: str,
    lift_rows: list[dict[str, Any]],
) -> None:
    """Write lift-by-decile figure for reporting splits."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for split_name in REPORTING_SPLITS:
        rows = [
            row
            for row in lift_rows
            if row["split"] == split_name and row["applicant_count"]
        ]
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
    _save_figure(path, figure)


def _save_figure(path: Path, figure: Any) -> None:
    """Persist and close a matplotlib figure."""
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
