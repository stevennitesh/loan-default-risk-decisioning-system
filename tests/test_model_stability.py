from __future__ import annotations

from pathlib import Path

import pytest

from src.model_stability import MODEL_STABILITY_AGGREGATE_COLUMNS
from src.model_stability import MODEL_STABILITY_RUN_COLUMNS
from src.model_stability import _aggregate_stability_rows
from src.model_stability import _select_stability_feature_set
from src.model_stability import run_model_stability_experiment
from src.train import run_training
from tests.test_feature_selection import _write_feature_importance
from tests.test_train import create_training_database
from tests.test_train import read_csv_rows


pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def test_stability_selection_uses_validation_aggregate_not_test_edge() -> None:
    rows = [
        _stability_run("top_100", 17, validation_pr_auc=0.270, test_pr_auc=0.266),
        _stability_run("top_100", 29, validation_pr_auc=0.272, test_pr_auc=0.267),
        _stability_run("full", 17, validation_pr_auc=0.268, test_pr_auc=0.270, feature_count=140),
        _stability_run("full", 29, validation_pr_auc=0.269, test_pr_auc=0.271, feature_count=140),
    ]

    aggregate_rows = _aggregate_stability_rows(rows, created_at="2026-04-30T00:00:00Z")
    selected_feature_set = _select_stability_feature_set(aggregate_rows)

    assert selected_feature_set == "top_100"
    selected_row = next(row for row in aggregate_rows if row["feature_set"] == selected_feature_set)
    full_row = next(row for row in aggregate_rows if row["feature_set"] == "full")
    assert selected_row["validation_win_count"] == 2
    assert selected_row["validation_win_rate"] == pytest.approx(1.0)
    assert full_row["test_pr_auc_mean"] > selected_row["test_pr_auc_mean"]


def test_model_stability_experiment_writes_seed_and_aggregate_reports(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    training_result = run_training(project_config_path)
    report_dir = scratch_path / "reports"
    _write_feature_importance(
        report_dir / "model_feature_importance.csv",
        training_result["feature_columns"],
    )

    result = run_model_stability_experiment(
        project_config_path,
        seeds=(17, 29),
        feature_limits=(3, 5),
        include_full=True,
    )

    run_rows = read_csv_rows(
        report_dir / "model_stability_seed_runs.csv",
        MODEL_STABILITY_RUN_COLUMNS,
    )
    aggregate_rows = read_csv_rows(
        report_dir / "model_stability_summary.csv",
        MODEL_STABILITY_AGGREGATE_COLUMNS,
    )
    assert (report_dir / "experiments" / "006_model_stability.md").exists()
    report_text = (report_dir / "experiments" / "006_model_stability.md").read_text(encoding="utf-8")
    assert "validation-only aggregate rule" in report_text
    assert "Held-out test is reported after selection" in report_text
    assert "## Interpretation" in report_text
    assert result["selected_feature_set"] in {"top_3", "top_5", "full"}
    assert len(run_rows) == 6
    assert {row["feature_set"] for row in run_rows} == {"top_3", "top_5", "full"}
    assert {row["seed"] for row in run_rows} == {"17", "29"}
    assert {row["feature_set"] for row in aggregate_rows} == {"top_3", "top_5", "full"}
    assert sum(row["selected"] == "True" for row in aggregate_rows) == 1
    for row in aggregate_rows:
        assert int(row["seed_count"]) == 2
        assert 0 <= float(row["validation_win_rate"]) <= 1
        assert float(row["validation_pr_auc_mean"]) >= 0
        assert float(row["validation_pr_auc_std"]) >= 0
        assert float(row["test_pr_auc_mean"]) >= 0


def test_model_stability_experiment_can_write_named_outputs(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    training_result = run_training(project_config_path)
    report_dir = scratch_path / "reports"
    _write_feature_importance(
        report_dir / "model_feature_importance.csv",
        training_result["feature_columns"],
    )

    result = run_model_stability_experiment(
        project_config_path,
        seeds=(17,),
        feature_limits=(),
        include_full=True,
        seed_runs_name="010_model_stability_seed_runs.csv",
        summary_name="010_model_stability_summary.csv",
        report_name="010_recency_model_stability.md",
    )

    assert result["seed_runs_path"] == report_dir / "010_model_stability_seed_runs.csv"
    assert result["summary_path"] == report_dir / "010_model_stability_summary.csv"
    assert result["report_path"] == report_dir / "experiments" / "010_recency_model_stability.md"
    assert result["seed_runs_path"].exists()
    assert result["summary_path"].exists()
    assert result["report_path"].exists()
    assert not (report_dir / "experiments" / "006_model_stability.md").exists()


def _stability_run(
    feature_set: str,
    seed: int,
    *,
    validation_pr_auc: float,
    test_pr_auc: float,
    feature_count: int = 100,
) -> dict[str, object]:
    return {
        "seed": seed,
        "feature_set": feature_set,
        "seed_validation_winner": False,
        "feature_count": feature_count,
        "feature_limit": feature_count,
        "selected_calibration_method": "sigmoid",
        "selected_candidate_name": "feature_subsample_regularized",
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": validation_pr_auc + 0.50,
        "validation_brier_score": 0.066,
        "validation_top_decile_lift": 3.6,
        "validation_precision_at_top_decile": 0.29,
        "validation_recall_at_review_capacity": 0.36,
        "validation_weighted_calibration_error": 0.003,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_pr_auc + 0.50,
        "test_brier_score": 0.067,
        "test_top_decile_lift": 3.5,
        "test_precision_at_top_decile": 0.28,
        "test_recall_at_review_capacity": 0.35,
        "test_weighted_calibration_error": 0.004,
        "validation_balanced_ev_per_applicant": 576.0,
        "test_balanced_ev_per_applicant": 581.0,
        "created_at": "2026-04-30T00:00:00Z",
    }
