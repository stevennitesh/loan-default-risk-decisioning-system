from __future__ import annotations

from pathlib import Path

import pytest

from src.feature_selection import ranked_raw_features
from src.feature_selection import run_feature_selection_experiment
from src.report_contracts import FEATURE_SELECTION_COMPARISON_COLUMNS
from src.train import run_training
from tests.helpers import create_training_database
from tests.helpers import read_csv_rows
from tests.helpers import write_feature_importance


pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def test_ranked_raw_features_maps_readable_shap_labels_to_model_columns() -> None:
    feature_columns = [
        "EXT_SOURCE_MEAN",
        "NAME_EDUCATION_TYPE",
        "AMT_CREDIT",
        "credit_card_avg_credit_utilization",
    ]
    importance_rows = [
        {"feature_name": "Name education type: Higher education", "rank": "1"},
        {"feature_name": "Ext source mean", "rank": "2"},
        {"feature_name": "Credit card avg credit utilization", "rank": "3"},
        {"feature_name": "Name education type: Secondary / secondary special", "rank": "4"},
        {"feature_name": "Amt credit", "rank": "5"},
    ]

    assert ranked_raw_features(importance_rows, feature_columns) == [
        "NAME_EDUCATION_TYPE",
        "EXT_SOURCE_MEAN",
        "credit_card_avg_credit_utilization",
        "AMT_CREDIT",
    ]


def test_feature_selection_experiment_writes_comparison_and_report(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    training_result = run_training(project_config_path)
    report_dir = scratch_path / "reports"
    write_feature_importance(
        report_dir / "model_feature_importance.csv",
        training_result["feature_columns"],
    )

    result = run_feature_selection_experiment(
        project_config_path,
        feature_limits=(3, 5),
        include_full=True,
    )

    rows = read_csv_rows(
        report_dir / "feature_selection_comparison.csv",
        FEATURE_SELECTION_COMPARISON_COLUMNS,
    )
    assert (report_dir / "experiments" / "005_feature_selection.md").exists()
    assert (report_dir / "experiments" / "005_selected_features.csv").exists()
    report_text = (report_dir / "experiments" / "005_feature_selection.md").read_text(encoding="utf-8")
    assert "## Interpretation" in report_text
    assert "validation-only rule" in report_text
    assert "not the optimization target" in report_text
    assert result["selected_feature_set"] in {"top_3", "top_5", "full"}
    assert {row["feature_set"] for row in rows} == {"top_3", "top_5", "full"}
    assert {int(row["feature_count"]) for row in rows}.issuperset({3, 5})
    assert sum(row["selected"] == "True" for row in rows) == 1
    for row in rows:
        assert row["selected_calibration_method"] in {"uncalibrated", "sigmoid", "isotonic"}
        assert float(row["validation_pr_auc"]) >= 0
        assert float(row["validation_brier_score"]) >= 0
        assert float(row["test_brier_score"]) >= 0
        assert float(row["validation_balanced_ev_per_applicant"]) != 0


def test_feature_selection_experiment_can_write_named_outputs(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    training_result = run_training(project_config_path)
    report_dir = scratch_path / "reports"
    write_feature_importance(
        report_dir / "model_feature_importance.csv",
        training_result["feature_columns"],
    )

    result = run_feature_selection_experiment(
        project_config_path,
        feature_limits=(3,),
        include_full=True,
        comparison_name="013_feature_cleanup_comparison.csv",
        selected_features_name="013_selected_features.csv",
        report_name="013_feature_cleanup.md",
    )

    assert result["comparison_path"] == report_dir / "013_feature_cleanup_comparison.csv"
    assert result["selected_features_path"] == report_dir / "experiments" / "013_selected_features.csv"
    assert result["report_path"] == report_dir / "experiments" / "013_feature_cleanup.md"
    assert result["comparison_path"].exists()
    assert result["selected_features_path"].exists()
    assert result["report_path"].exists()
    report_text = result["report_path"].read_text(encoding="utf-8")
    assert "reports/experiments/013_selected_features.csv" in report_text
    assert not (report_dir / "experiments" / "005_feature_selection.md").exists()
