"""Diagnostic and reconciliation paths honor non-default capture rates."""

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from src import assessment_diagnostics, correctness_summary
from src.dashboard_exports import EXPORT_TABLE_COLUMNS
from src.metrics import probability_metrics


@pytest.mark.parametrize("capacity,expected_recall", [(0.1, 0.5), (0.2, 1.0)])
def test_frozen_segment_diagnostics_use_configured_capture_rate(
    scratch_path, capacity, expected_recall
):
    database = scratch_path / "features.duckdb"
    ids = list(range(1, 11))
    scores = np.linspace(0.95, 0.05, 10)
    targets = [1, 1] + [0] * 8
    with duckdb.connect(str(database)) as connection:
        connection.register("applicants", pd.DataFrame({"SK_ID_CURR": ids}))
        connection.execute(
            "CREATE TABLE mart_credit_risk_features AS SELECT SK_ID_CURR, 'application_train' AS source_population, 1 AS installment_obligation_count, 0 AS installment_ambiguous_obligation_count, 0 AS installment_unknown_payment_obligation_count FROM applicants"
        )
    manifest = {
        "status": "complete",
        "run_id": "capture-regression",
        "config": {
            "paths": {"duckdb_path": str(database)},
            "business_assumptions": {"manual_review_capacity_rate": capacity},
        },
        "feature_columns": ["feature"],
        "selected_workflows": [
            {
                "workflow": "history_selected",
                "split_seed": 42,
                "outer_fold": 1,
                "calibration_method": "sigmoid",
                "feature_columns": ["feature"],
                "candidate": {"candidate_name": "fixed"},
            }
        ],
    }
    (scratch_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "workflow": "history_selected",
            "split_seed": 42,
            "outer_fold": 1,
            "target": targets,
            "raw_score": scores,
            "calibrated_score": scores,
        }
    ).to_csv(scratch_path / "assessment_predictions.csv", index=False)

    assessment_diagnostics.run(scratch_path)

    rows = pd.read_csv(scratch_path / "history_segment_metrics.csv")
    assert set(rows.score_kind) == {"raw", "calibrated"}
    assert rows.recall_at_manual_review_capacity.tolist() == pytest.approx(
        [expected_recall, expected_recall]
    )


@pytest.mark.parametrize("capacity,expected_recall", [(0.1, 0.5), (0.2, 1.0)])
def test_dashboard_reconciliation_uses_configured_capture_rate(
    scratch_path, monkeypatch, capacity, expected_recall
):
    ids = list(range(1, 11))
    scores = np.linspace(0.95, 0.05, 10)
    targets = [1, 1] + [0] * 8
    alias = "selected-capture-regression"
    folder = scratch_path / "dashboard"
    folder.mkdir()
    models = scratch_path / "models"
    models.mkdir()
    config = {
        "paths": {"dashboard_export_dir": str(folder), "model_dir": str(models)},
        "business_assumptions": {
            "manual_review_capacity_rate": capacity,
            "expected_margin_per_good_loan": 1000,
            "expected_loss_per_bad_loan": 5000,
            "manual_review_cost": 50,
        },
    }
    tables = {
        name: pd.DataFrame(columns=columns)
        for name, columns in EXPORT_TABLE_COLUMNS.items()
    }
    tables["credit_risk_scores"] = pd.DataFrame(
        {
            "applicant_id": ids,
            "scoring_population": "holdout_test",
            "observed_target": targets,
            "model_version": alias,
            "score": scores,
            "raw_risk_score": scores,
            "calibrated_risk_score": scores,
            "recommended_action": "approve",
        },
        columns=EXPORT_TABLE_COLUMNS["credit_risk_scores"],
    )
    expected_metrics = probability_metrics(pd.Series(targets), scores, capacity)
    tables["model_metrics_summary"] = pd.DataFrame(
        [
            {
                "model_version": alias,
                "split": "test",
                "metric_name": name,
                "metric_value": value,
            }
            for name, value in expected_metrics.items()
        ],
        columns=EXPORT_TABLE_COLUMNS["model_metrics_summary"],
    )
    tables["model_threshold_metrics"] = pd.DataFrame(
        [
            {
                "model_version": alias,
                "split": "test",
                "scenario_name": "balanced",
                "threshold_low": 1,
                "threshold_high": 1,
                "expected_value_per_applicant": -200,
            }
        ],
        columns=EXPORT_TABLE_COLUMNS["model_threshold_metrics"],
    )
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    artifact = {
        "run_id": "frozen-model",
        "feature_build_id": "frozen-build",
        "split_applicant_ids": {"test": ids},
    }
    (models / "historical_split_reference.json").write_text(
        json.dumps({"split_applicant_ids": {"test": ids}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        correctness_summary.joblib,
        "load",
        lambda path: (
            {"calibration_run_id": "frozen-calibrator"}
            if path.name.endswith("_calibration.joblib")
            else artifact
        ),
    )
    monkeypatch.setattr(correctness_summary, "ORIGINAL_ASSETS", {})

    result, _ = correctness_summary.reconcile_dashboard(config)

    assert result["schemas_checked"] == 8
    assert result["test_probability_metrics"][
        "recall_at_manual_review_capacity"
    ] == pytest.approx(expected_recall)
