from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import numpy as np
import pytest
import yaml

from src.calibrate import run_calibration_experiment
from src.config import load_config
from src.dashboard_exports import run_dashboard_export
from src.evaluate import EvaluationError, _validate_artifacts, run_evaluation
from src.model_contracts import MODEL_ARTIFACTS
from src.report_contracts import (
    MODEL_CALIBRATION_BINS_COLUMNS,
    MODEL_CONFUSION_MATRIX_COLUMNS,
    MODEL_LIFT_BY_DECILE_COLUMNS,
    MODEL_METRICS_SUMMARY_COLUMNS,
    MODEL_THRESHOLD_METRICS_COLUMNS,
)
from src.score_batch import run_scoring
from src.thresholding import SCENARIO_NAMES
from src.train import run_training
from tests.helpers import (
    create_training_database,
    read_csv_rows,
    table_exists,
    table_row_count,
)

REQUIRED_EVALUATION_METRICS = {
    "log_loss",
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

pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names.*:UserWarning"
)


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
    create_training_database(
        scratch_path / "db" / "credit_risk.duckdb", train_rows=train_rows
    )
    run_training(project_config_path)

    result = run_evaluation(project_config_path)

    report_dir = scratch_path / "reports"
    metrics_rows = read_csv_rows(
        report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS
    )
    lift_rows = read_csv_rows(
        report_dir / "model_lift_by_decile.csv", MODEL_LIFT_BY_DECILE_COLUMNS
    )
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
    for model_version in ("logistic_regression_baseline_v1", "lightgbm_credit_risk_v1"):
        for split in ("train", "validation", "test"):
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
    for split in ("validation", "test"):
        rows = [row for row in lift_rows if row["split"] == split]
        assert {int(row["decile"]) for row in rows} == set(range(1, 11))
        assert sum(int(row["applicant_count"]) for row in rows) == split_sizes[split]
        decile_scores = {
            int(row["decile"]): float(row["average_score"])
            for row in rows
            if row["average_score"]
        }
        assert decile_scores[min(decile_scores)] >= decile_scores[max(decile_scores)]
        assert all(float(row["lift"]) >= 0 for row in rows if row["lift"])
        assert all(
            0 <= float(row["cumulative_default_capture_rate"]) <= 1 for row in rows
        )

    assert {row["split"] for row in calibration_rows} == {"validation", "test"}
    for split in ("validation", "test"):
        rows = [row for row in calibration_rows if row["split"] == split]
        assert {int(row["bin_id"]) for row in rows} == set(range(1, 11))
        assert sum(int(row["applicant_count"]) for row in rows) == split_sizes[split]
        for row in rows:
            if int(row["applicant_count"]) == 0:
                assert row["average_predicted_score"] == ""
                assert row["observed_default_rate"] == ""
                assert row["calibration_error"] == ""
                continue
            assert 0 <= float(row["average_predicted_score"]) <= 1
            assert 0 <= float(row["observed_default_rate"]) <= 1
            assert -1 <= float(row["calibration_error"]) <= 1

    assert {row["split"] for row in confusion_rows} == {"validation", "test"}
    assert {row["scenario_name"] for row in confusion_rows} == SCENARIOS
    assert {row["split"] for row in threshold_rows} == {"validation", "test"}
    assert {row["scenario_name"] for row in threshold_rows} == SCENARIOS
    assert {row["threshold_version"] for row in threshold_rows} == {"threshold_v1"}
    assert {row["model_version"] for row in threshold_rows} == {
        result["selected_model_version"]
    }
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

    for split in ("validation", "test"):
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
                int(row["count"]) for row in rows if row["true_label"] == "1"
            )

            assert applicant_count == split_sizes[split]
            assert (
                approved_good_count
                + approved_bad_count
                + manual_review_count
                + high_risk_count
                == applicant_count
            )
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
                int(row["count"]) for row in rows if row["predicted_label"] == "1"
            )
            assert float(
                threshold_row["high_risk_default_capture_rate"]
            ) == pytest.approx(high_risk_default_count / total_default_count)

    for figure_name in [
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "lift_chart.png",
    ]:
        assert (report_dir / "figures" / figure_name).stat().st_size > 0

    validation_report = (report_dir / "validation_report.md").read_text(
        encoding="utf-8"
    )
    business_value_report = (report_dir / "business_value_analysis.md").read_text(
        encoding="utf-8"
    )
    for report in (validation_report, business_value_report):
        narrative, run_details = report.split("## Technical run details")
        assert result["selected_model_version"] not in narrative
        assert result["selected_model_version"] in run_details
    selected_artifact = joblib.load(
        scratch_path / "models" / MODEL_ARTIFACTS[result["selected_model_type"]][1]
    )
    for identity in (
        selected_artifact["run_id"],
        selected_artifact["feature_build_id"],
        selected_artifact["methodology_version"],
    ):
        narrative, run_details = validation_report.split("## Technical run details")
        assert identity not in narrative
        assert identity in run_details
    for split in ("validation", "test"):
        value = next(
            float(row["metric_value"])
            for row in metrics_rows
            if row["model_version"] == result["selected_model_version"]
            and row["split"] == split
            and row["metric_name"] == "pr_auc"
        )
        assert f"Average precision={value:.3f}" in validation_report
    assert "Expected-value analysis is pending Milestone 7" not in validation_report
    assert "Threshold expected-value analysis" in validation_report
    assert "Expected margin per good approved loan: 1000" in business_value_report
    assert "Expected loss per bad approved loan: 5000" in business_value_report
    assert "Manual review cost: 50" in business_value_report
    assert (
        "Kaggle application_test rows are not used for evaluation metrics"
        in validation_report
    )

    with duckdb.connect(
        str(scratch_path / "db" / "credit_risk.duckdb"), read_only=True
    ) as connection:
        assert table_row_count(connection, "model_metrics_summary") == len(metrics_rows)
        assert table_row_count(connection, "model_lift_by_decile") == len(lift_rows)
        assert table_row_count(connection, "model_calibration_bins") == len(
            calibration_rows
        )
        assert table_row_count(connection, "model_confusion_matrix") == len(
            confusion_rows
        )
        assert table_row_count(connection, "model_threshold_metrics") == len(
            threshold_rows
        )
        assert table_exists(connection, "model_threshold_metrics")


@pytest.mark.parametrize("limit", [40, 80])
def test_joint_subset_training_evaluates_each_fitted_surface(
    scratch_path,
    project_config_path,
    monkeypatch,
    limit,
):
    from src import evaluate, tuning

    config = load_config(project_config_path)
    config["resources"] = {"model_threads": 1}
    config["model"]["lightgbm_tuning"].update(
        mode="bounded_inner_cv",
        max_candidates=1,
        inner_folds=3,
        max_rounds=20,
        stopping_rounds=3,
        sensitivity_seeds=[101],
    )
    project_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    database = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database, train_rows=500)
    with duckdb.connect(str(database)) as con:
        for index in range(70):
            con.execute(
                f"ALTER TABLE mart_credit_risk_features ADD COLUMN fixture_extra_{index} DOUBLE DEFAULT {index}"
            )
    original_specs = tuning.candidate_specs
    # Force a declared top-N winner while retaining real CV preprocessing,
    # ranking, stopping, fixed-round fitting and ordinary artifact persistence.
    monkeypatch.setattr(
        tuning,
        "candidate_specs",
        lambda cfg, surfaces: original_specs(cfg, [f"top_{limit}"]),
    )
    trained = run_training(project_config_path)
    baseline = joblib.load(trained["artifacts"]["logistic_regression"])
    lightgbm = joblib.load(trained["artifacts"]["lightgbm"])
    assert len(baseline["feature_columns"]) == 85
    assert len(lightgbm["feature_columns"]) == limit
    assert (
        lightgbm["lightgbm_tuning"]["selected_candidate"]["feature_set"]
        == f"top_{limit}"
    )
    union, ids = _validate_artifacts(
        {"logistic_regression": baseline, "lightgbm": lightgbm}
    )
    assert set(union) == set(baseline["feature_columns"])
    assert set(ids) == {"train", "validation", "test"}
    calls = []
    original_predict = evaluate.predict_probabilities

    def observed_predict(artifact, frame, columns, *args):
        assert columns == artifact["feature_columns"]
        calls.append((artifact["model_type"], len(columns)))
        return original_predict(artifact, frame, columns, *args)

    monkeypatch.setattr(evaluate, "predict_probabilities", observed_predict)
    result = run_evaluation(project_config_path)
    assert calls == [("logistic_regression", 85)] * 3 + [("lightgbm", limit)] * 3
    expected = {
        (r["model_version"], r["split"], r["metric_name"]): r["metric_value"]
        for r in trained["metric_rows"]
    }
    for row in result["metric_rows"]:
        assert row["metric_value"] == pytest.approx(
            expected[(row["model_version"], row["split"], row["metric_name"])]
        )
    # Downstream consumers load the chosen artifact's own fields.
    run_calibration_experiment(project_config_path)
    run_scoring(project_config_path)
    assert result["selected_model_type"] == "lightgbm"
    from src.explain import run_explain

    explained = run_explain(project_config_path)
    assert explained
    exported = run_dashboard_export(project_config_path)
    assert exported


def test_legacy_equal_surface_artifacts_still_evaluate(
    scratch_path, project_config_path
):
    database = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database, train_rows=80)
    trained = run_training(project_config_path)  # Explicit pre-v3 config contract.
    artifacts = {name: joblib.load(path) for name, path in trained["artifacts"].items()}
    assert artifacts["lightgbm"]["methodology_version"] == "disjoint_calibration_v2"
    assert (
        artifacts["logistic_regression"]["feature_columns"]
        == artifacts["lightgbm"]["feature_columns"]
    )
    # Old equal-surface files did not require an eligibility-search manifest.
    for name, artifact in artifacts.items():
        artifact.pop("eligible_feature_columns", None)
        joblib.dump(artifact, trained["artifacts"][name])
    evaluated = run_evaluation(project_config_path)
    assert np.allclose(
        [row["metric_value"] for row in evaluated["metric_rows"]],
        [row["metric_value"] for row in trained["metric_rows"]],
    )


@pytest.mark.parametrize("mutation", ["ineligible", "build", "calibration_roles"])
def test_evaluation_keeps_artifact_identity_guards(
    scratch_path, project_config_path, mutation
):
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=100)
    trained = run_training(project_config_path)
    path = trained["artifacts"]["lightgbm"]
    artifact = joblib.load(path)
    if mutation == "ineligible":
        artifact["eligible_feature_columns"] = artifact["feature_columns"][1:]
    elif mutation == "build":
        artifact["feature_build_id"] = "another_build"
    else:
        ids = artifact["split_applicant_ids"]
        ids["calibration"][0], ids["train"][0] = ids["train"][0], ids["calibration"][0]
    joblib.dump(artifact, path)
    with pytest.raises(EvaluationError):
        run_evaluation(project_config_path)
