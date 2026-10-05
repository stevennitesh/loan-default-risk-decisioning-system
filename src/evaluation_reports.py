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

from src.model_contracts import REPORTING_SPLITS
from src.presentation import model_type_label, scenario_label, split_label
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
        f"- {scenario_label(scenario)}: lower score cutoff={thresholds['threshold_low']:.6f}, "
        f"upper score cutoff={thresholds['threshold_high']:.6f}"
        for scenario, thresholds in scenario_thresholds.items()
    )
    split_lines = "\n".join(
        f"- {split_label(row['split'])}: {row['row_count']:,} applicants, repayment-difficulty rate {float(row['positive_rate']):.1%}"
        for row in split_summary_rows
    )
    metric_lines = "\n".join(
        f"- {split_label(split)}: Average precision={metrics[(selected_model_version, split, 'pr_auc')]:.3f}, "
        f"ROC AUC={metrics[(selected_model_version, split, 'roc_auc')]:.3f}, "
        f"Brier score={metrics[(selected_model_version, split, 'brier_score')]:.3f}, "
        f"top-decile lift={metrics[(selected_model_version, split, 'top_decile_lift')]:.2f}"
        for split in REPORTING_SPLITS
    )
    balanced_rows = [
        row
        for row in threshold_rows
        if row["scenario_name"] == BALANCED_SCENARIO
        and row["split"] in REPORTING_SPLITS
    ]
    balanced_lines = "\n".join(
        f"- {split_label(row['split'])}: simulated approval={row['approval_rate']:.1%}, "
        f"manual review={row['manual_review_rate']:.1%}, "
        f"simulated decline={row['high_risk_rate']:.1%}, "
        f"illustrative utility={row['expected_value_per_applicant']:.2f} units per applicant"
        for row in balanced_rows
    )
    text = f"""# Validation Report

## Executive Summary

This report checks how well a credit-risk model ranks observed repayment difficulty and describes its raw probabilities and simulated action bands. {model_type_label(selected_model_type)} was selected using model-selection average precision (a ranking measure, not accuracy).

Only labeled applicants contribute outcome metrics. Unlabeled Kaggle applications are a batch-scoring demonstration. Model-selection results support choosing the model; historical-comparison results describe reused applicants and do not establish an untouched final test.

## Selected model results

{metric_lines}

Higher average precision and ROC AUC indicate stronger ranking. Lower Brier score indicates smaller probability errors. Top-decile lift compares the difficulty rate in the highest-risk 10% with the overall rate; it is separate from middle-band manual review.

## Applicant groups

{split_lines}

## Probability quality

These metrics and bins use raw model probabilities. The application-and-history workflow fits probability adjustments (calibrators) on reserved calibration rows and assesses methods on separate selection-validation rows in `model_calibration_comparison.csv` and the calibration artifact. Evaluation does not import that separate report, which may belong to an earlier fitted run.

## Concentration in highest-risk groups

`model_lift_by_decile` reports model-selection and historical-comparison groups with decile 1 representing the highest-risk applicants.

## Threshold Scenario Analysis

The following model-selection-derived score cutoffs define simulated approval, manual review and simulated decline. The confusion matrix summarizes the upper-cutoff flag:

{scenario_lines}

Manual-review handling is explicit: the confusion matrix treats only the high-risk action as the positive prediction.

## Illustrative action utility

Threshold expected-value analysis is produced in `model_threshold_metrics` and `reports/business_value_analysis.md`.

Illustrative weights (units, not dollars or measured profit):

- Expected margin per good approved loan: {assumptions["expected_margin_per_good_loan"]}
- Expected loss per bad approved loan: {assumptions["expected_loss_per_bad_loan"]}
- Manual review cost: {assumptions["manual_review_cost"]}

Balanced display scenario (a reference, not an optimal policy or hard queue cap):

{balanced_lines}

## Limitations

This is a portfolio decision-support simulation, not a production credit-decisioning system. Validation metrics are selection evidence; test metrics are historical comparison evidence. Neither establishes production underwriting readiness or independent assessment after prior dataset exploration.

## Technical run details

- Model type: `{selected_model_type}`
- Model version: `{selected_model_version}`
- Fitted run: `{selected_artifact["run_id"]}`
- Feature build: `{selected_artifact["feature_build_id"]}`
- Methodology: `{selected_artifact["methodology_version"]}`

Exact numeric metrics remain in `model_metrics_summary.csv`; report numbers use display rounding. Kaggle application_test rows are not used for evaluation metrics. The labeled historical comparison comes from `application_train`.
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
        "| {split_display} | {scenario_display} | {approval_rate:.1%} | {manual_review_rate:.1%} | "
        "{high_risk_rate:.1%} | {default_rate_approved:.1%} | "
        "{high_risk_default_capture_rate:.1%} | {expected_value:.2f} | "
        "{expected_value_per_applicant:.2f} |".format(
            **row,
            split_display=split_label(row["split"]),
            scenario_display=scenario_label(row["scenario_name"]),
        )
        for row in rows
    )
    text = f"""# Illustrative decision scenarios

This report shows how score cutoffs change simulated approvals, manual reviews and declines under explicit utility assumptions. It uses the {model_type_label(selected_model_type)} selected with development applicants.

Cutoffs are selected from raw model-selection scores and applied unchanged to historical-comparison applicants. Only labeled applicants contribute outcomes; unlabeled Kaggle applications are a scoring demonstration. Prior use of the comparison applicants prevents independent final-test claims.

## Assumptions

- Expected margin per good approved loan: {assumptions["expected_margin_per_good_loan"]}
- Expected loss per bad approved loan: {assumptions["expected_loss_per_bad_loan"]}
- Manual review cost: {assumptions["manual_review_cost"]}
- Review-rate scenario reference (not an enforced cap): {assumptions["manual_review_capacity_rate"]:.0%}

These values are scenario-comparison utility weights, not calibrated Home Credit economics. The good-loan margin and bad-loan loss encode a simple penalty ratio so simulated approval, review and decline scenarios can be compared. They do not estimate lender economics or reviewer effectiveness.

## Scenario Metrics

| Applicant group | Scenario | Simulated approval rate | Manual-review rate | Simulated-decline rate | Difficulty rate among approvals | Difficulty cases in decline band | Total utility (units) | Utility per applicant (units) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{table_lines}

## Notes

Utility adds the assumed margin for approvals without recorded difficulty, subtracts assumed loss for approvals with difficulty, and subtracts the middle-band review cost. The units are illustrative, not dollars or measured profit.

High-risk applicants are simulated declines, contributing no issued-loan margin, loss, or review cost. Only the middle band incurs review cost; reviewer effectiveness is not estimated. Utility uses raw scores regardless of the separate probability-adjustment method.

## Technical run details

- Model type: `{selected_model_type}`
- Model version: `{selected_model_version}`
- Exact scenario values: `model_threshold_metrics.csv`

The preserved calculation is `approved_good_count * expected_margin_per_good_loan - approved_bad_count * expected_loss_per_bad_loan - manual_review_count * manual_review_cost`.
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


def _write_roc_curve(
    path: Path,
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    """Write ROC curve figure for reporting splits."""
    figure, axis = plt.subplots(figsize=(9, 6))
    for split_name in REPORTING_SPLITS:
        frame = prediction_frames[split_name]
        fpr, tpr, _ = roc_curve(frame["target"], frame["probability"])
        auc = roc_auc_score(frame["target"], frame["probability"])
        axis.plot(fpr, tpr, label=f"{split_label(split_name)} · AUC {auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    axis.set_title("Risk ranking: ROC curve")
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
    figure, axis = plt.subplots(figsize=(9, 6))
    for split_name in REPORTING_SPLITS:
        frame = prediction_frames[split_name]
        precision, recall, _ = precision_recall_curve(
            frame["target"], frame["probability"]
        )
        pr_auc = average_precision_score(frame["target"], frame["probability"])
        axis.plot(
            recall,
            precision,
            label=f"{split_label(split_name)} · average precision {pr_auc:.3f}",
        )
    axis.set_title("Risk ranking: precision and case capture")
    axis.set_xlabel("Share of repayment-difficulty cases captured (recall)")
    axis.set_ylabel("Repayment-difficulty rate among flagged applicants (precision)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(path, figure)


def _write_calibration_curve(
    path: Path,
    model_version: str,
    calibration_rows: list[dict[str, Any]],
) -> None:
    """Write observed-vs-predicted calibration curve figure."""
    figure, axis = plt.subplots(figsize=(9, 6))
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
            label=split_label(split_name),
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    axis.set_title("Raw probability reliability")
    axis.set_xlabel("Mean predicted repayment-difficulty probability")
    axis.set_ylabel("Observed repayment-difficulty rate")
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(path, figure)


def _write_lift_chart(
    path: Path,
    model_version: str,
    lift_rows: list[dict[str, Any]],
) -> None:
    """Write lift-by-decile figure for reporting splits."""
    figure, axis = plt.subplots(figsize=(9, 6))
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
            label=split_label(split_name),
        )
    axis.set_title("Repayment difficulty concentrated in highest-risk groups")
    axis.set_xlabel("Risk decile (1 = highest risk)")
    axis.set_ylabel("Outcome rate / overall rate (lift)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(path, figure)


def _save_figure(path: Path, figure: Any) -> None:
    """Persist and close a matplotlib figure."""
    figure.text(
        0.5,
        0.01,
        "Labeled model-selection / reused historical comparison applicants; unlabeled scoring excluded.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)
