from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import pytest

from src.evaluate import EvaluationError, run_evaluation
from src.report_contracts import (
    MODEL_CALIBRATION_BINS_COLUMNS,
    MODEL_CONFUSION_MATRIX_COLUMNS,
    MODEL_LIFT_BY_DECILE_COLUMNS,
    MODEL_METRICS_SUMMARY_COLUMNS,
    MODEL_THRESHOLD_METRICS_COLUMNS,
)
from src.thresholding import SCENARIO_NAMES
from src.train import run_training
from tests.helpers import (
    create_training_database,
    read_csv_rows,
    table_exists,
    table_row_count,
)

REQUIRED_EVALUATION_METRICS = {
    "roc_auc",
    "pr_auc",
    "brier_score",
    "min_predicted_probability",
    "max_predicted_probability",
    "top_decile_lift",
    "precision_at_top_decile",
    "recall_at_manual_review_capacity",
}

SCENARIOS = set(SCENARIO_NAMES)

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def test_evaluation_fails_clearly_without_model_artifacts(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)

    with pytest.raises(EvaluationError) as error:
        run_evaluation(project_config_path)

    assert "Missing model artifact" in str(error.value)
    assert not (scratch_path / "reports" / "model_lift_by_decile.csv").exists()


def test_evaluation_fails_when_saved_split_ids_are_missing(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    run_training(project_config_path)
    artifact_path = scratch_path / "models" / "lightgbm_credit_risk.joblib"
    artifact = joblib.load(artifact_path)
    artifact.pop("split_applicant_ids")
    joblib.dump(artifact, artifact_path)

    with pytest.raises(EvaluationError) as error:
        run_evaluation(project_config_path)

    assert "split_applicant_ids" in str(error.value)


def test_run_evaluation_creates_metrics_reports_figures_and_duckdb_tables(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    train_rows = 80
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=train_rows)
    run_training(project_config_path)

    result = run_evaluation(project_config_path)

    report_dir = scratch_path / "reports"
    metrics_rows = read_csv_rows(report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS)
    lift_rows = read_csv_rows(report_dir / "model_lift_by_decile.csv", MODEL_LIFT_BY_DECILE_COLUMNS)
    calibration_rows = read_csv_rows(
        report_dir / "model_calibration_bins.csv",
        MODEL_CALIBRATION_BINS_COLUMNS,
    )
    threshold_rows = read_csv_rows(
        report_dir / "model_threshold_metrics.csv",
        MODEL_THRESHOLD_METRICS_COLUMNS,
    )
    confusion_rows = read_csv_rows(
        report_dir / "model_confusion_matrix.csv",
        MODEL_CONFUSION_MATRIX_COLUMNS,
    )

    assert result["selected_model_type"] in {"logistic_regression", "lightgbm"}
    assert set(result["scenario_thresholds"]) == SCENARIOS
    for thresholds in result["scenario_thresholds"].values():
        assert 0 <= thresholds["threshold_low"] < thresholds["threshold_high"] <= 1

    assert len(metrics_rows) == 2 * 3 * len(REQUIRED_EVALUATION_METRICS)
    for model_version in {"logistic_regression_baseline_v1", "lightgbm_credit_risk_v1"}:
        for split in {"train", "validation", "test"}:
            split_metrics = {
                row["metric_name"]: float(row["metric_value"])
                for row in metrics_rows
                if row["model_version"] == model_version and row["split"] == split
            }
            assert set(split_metrics) == REQUIRED_EVALUATION_METRICS
            assert 0 <= split_metrics["min_predicted_probability"] <= 1
            assert 0 <= split_metrics["max_predicted_probability"] <= 1
            assert split_metrics["top_decile_lift"] >= 0
            assert 0 <= split_metrics["precision_at_top_decile"] <= 1
            assert 0 <= split_metrics["recall_at_manual_review_capacity"] <= 1

    split_sizes = {
        split: len(ids)
        for split, ids in joblib.load(
            scratch_path / "models" / "lightgbm_credit_risk.joblib"
        )["split_applicant_ids"].items()
    }

    assert {row["split"] for row in lift_rows} == {"validation", "test"}
    for split in {"validation", "test"}:
        rows = [row for row in lift_rows if row["split"] == split]
        assert {int(row["decile"]) for row in rows} == set(range(1, 11))
        assert sum(int(row["applicant_count"]) for row in rows) == split_sizes[split]
        decile_scores = {int(row["decile"]): float(row["average_score"]) for row in rows}
        assert decile_scores[1] >= decile_scores[10]
        assert all(float(row["lift"]) >= 0 for row in rows)
        assert all(0 <= float(row["cumulative_default_capture_rate"]) <= 1 for row in rows)

    assert {row["split"] for row in calibration_rows} == {"validation", "test"}
    for split in {"validation", "test"}:
        rows = [row for row in calibration_rows if row["split"] == split]
        assert {int(row["bin_id"]) for row in rows} == set(range(1, 11))
        assert sum(int(row["applicant_count"]) for row in rows) == split_sizes[split]
        for row in rows:
            assert 0 <= float(row["average_predicted_score"]) <= 1
            assert 0 <= float(row["observed_default_rate"]) <= 1
            assert -1 <= float(row["calibration_error"]) <= 1

    assert {row["split"] for row in confusion_rows} == {"validation", "test"}
    assert {row["scenario_name"] for row in confusion_rows} == SCENARIOS
    assert {row["split"] for row in threshold_rows} == {"validation", "test"}
    assert {row["scenario_name"] for row in threshold_rows} == SCENARIOS
    assert {row["threshold_version"] for row in threshold_rows} == {"threshold_v1"}
    assert {row["model_version"] for row in threshold_rows} == {result["selected_model_version"]}
    assert len(threshold_rows) == 2 * len(SCENARIOS)
    validation_thresholds = {
        row["scenario_name"]: (row["threshold_low"], row["threshold_high"])
        for row in threshold_rows
        if row["split"] == "validation"
    }
    test_thresholds = {
        row["scenario_name"]: (row["threshold_low"], row["threshold_high"])
        for row in threshold_rows
        if row["split"] == "test"
    }
    assert validation_thresholds == test_thresholds

    for split in {"validation", "test"}:
        for scenario_name in SCENARIOS:
            rows = [
                row
                for row in confusion_rows
                if row["split"] == split and row["scenario_name"] == scenario_name
            ]
            assert len(rows) == 4
            assert sum(int(row["count"]) for row in rows) == split_sizes[split]
            assert {int(row["true_label"]) for row in rows} == {0, 1}
            assert {int(row["predicted_label"]) for row in rows} == {0, 1}
            threshold_row = next(
                row
                for row in threshold_rows
                if row["split"] == split and row["scenario_name"] == scenario_name
            )
            applicant_count = int(threshold_row["applicant_count"])
            approved_good_count = int(threshold_row["approved_good_count"])
            approved_bad_count = int(threshold_row["approved_bad_count"])
            manual_review_count = int(threshold_row["manual_review_count"])
            high_risk_count = int(threshold_row["high_risk_count"])
            high_risk_default_count = sum(
                int(row["count"])
                for row in rows
                if row["true_label"] == "1" and row["predicted_label"] == "1"
            )
            total_default_count = sum(
                int(row["count"])
                for row in rows
                if row["true_label"] == "1"
            )

            assert applicant_count == split_sizes[split]
            assert approved_good_count + approved_bad_count + manual_review_count + high_risk_count == applicant_count
            assert float(threshold_row["approval_rate"]) == pytest.approx(
                (approved_good_count + approved_bad_count) / applicant_count
            )
            assert float(threshold_row["manual_review_rate"]) == pytest.approx(
                manual_review_count / applicant_count
            )
            assert float(threshold_row["high_risk_rate"]) == pytest.approx(
                high_risk_count / applicant_count
            )
            assert high_risk_count == sum(
                int(row["count"])
                for row in rows
                if row["predicted_label"] == "1"
            )
            assert float(threshold_row["high_risk_default_capture_rate"]) == pytest.approx(
                high_risk_default_count / total_default_count
            )

    for figure_name in [
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "lift_chart.png",
    ]:
        assert (report_dir / "figures" / figure_name).stat().st_size > 0

    validation_report = (report_dir / "validation_report.md").read_text(encoding="utf-8")
    business_value_report = (report_dir / "business_value_analysis.md").read_text(encoding="utf-8")
    assert "Expected-value analysis is pending Milestone 7" not in validation_report
    assert "Threshold expected-value analysis" in validation_report
    assert "Expected margin per good approved loan: 1000" in business_value_report
    assert "Expected loss per bad approved loan: 5000" in business_value_report
    assert "Manual review cost: 50" in business_value_report
    assert "Kaggle application_test rows are not used for evaluation metrics" in validation_report

    with duckdb.connect(str(scratch_path / "db" / "credit_risk.duckdb"), read_only=True) as connection:
        assert table_row_count(connection, "model_metrics_summary") == len(metrics_rows)
        assert table_row_count(connection, "model_lift_by_decile") == len(lift_rows)
        assert table_row_count(connection, "model_calibration_bins") == len(calibration_rows)
        assert table_row_count(connection, "model_confusion_matrix") == len(confusion_rows)
        assert table_row_count(connection, "model_threshold_metrics") == len(threshold_rows)
        assert table_exists(connection, "model_threshold_metrics")
