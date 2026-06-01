from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import pytest

from src.calibrate import run_calibration_experiment
from src.calibration import select_calibration_method
from src.report_contracts import MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS
from src.report_contracts import MODEL_CALIBRATION_COMPARISON_COLUMNS
from src.train import run_training
from tests.helpers import create_training_database
from tests.helpers import read_csv_rows


CALIBRATION_METHODS = {"uncalibrated", "sigmoid", "isotonic"}
REPORTING_SPLITS = {"validation", "test"}

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def test_calibration_experiment_fits_on_validation_and_exports_comparison_artifacts(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80)
    run_training(project_config_path)

    result = run_calibration_experiment(project_config_path)

    report_dir = scratch_path / "reports"
    model_dir = scratch_path / "models"
    artifact_path = model_dir / "lightgbm_credit_risk_calibration.joblib"
    comparison_rows = read_csv_rows(
        report_dir / "model_calibration_comparison.csv",
        MODEL_CALIBRATION_COMPARISON_COLUMNS,
    )
    bin_rows = read_csv_rows(
        report_dir / "model_calibration_bins_comparison.csv",
        MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS,
    )
    artifact = joblib.load(artifact_path)

    assert artifact_path.exists()
    assert result["selected_method"] in CALIBRATION_METHODS
    assert artifact["base_model_version"] == "lightgbm_credit_risk_v1"
    assert artifact["calibration_fit_split"] == "validation"
    assert artifact["selected_method"] == result["selected_method"]
    assert set(artifact["calibrators"]) == {"sigmoid", "isotonic"}
    assert artifact["fit_applicant_ids"] == artifact["split_applicant_ids"]["validation"]

    assert len(comparison_rows) == len(CALIBRATION_METHODS) * len(REPORTING_SPLITS)
    assert {
        (row["calibration_method"], row["split"]) for row in comparison_rows
    } == {
        (method, split)
        for method in CALIBRATION_METHODS
        for split in REPORTING_SPLITS
    }
    for row in comparison_rows:
        assert row["base_model_version"] == "lightgbm_credit_risk_v1"
        assert row["model_version"] == f"lightgbm_credit_risk_v1_{row['calibration_method']}"
        assert 0 <= float(row["min_predicted_probability"]) <= 1
        assert 0 <= float(row["max_predicted_probability"]) <= 1
        assert float(row["brier_score"]) >= 0
        assert float(row["mean_absolute_bin_error"]) >= 0
        assert float(row["weighted_calibration_error"]) >= 0
        assert float(row["max_absolute_bin_error"]) >= 0

    assert len(bin_rows) == len(CALIBRATION_METHODS) * len(REPORTING_SPLITS) * 10
    for method in CALIBRATION_METHODS:
        for split in REPORTING_SPLITS:
            rows = [
                row
                for row in bin_rows
                if row["calibration_method"] == method and row["split"] == split
            ]
            assert {int(row["bin_id"]) for row in rows} == set(range(1, 11))
            assert sum(int(row["applicant_count"]) for row in rows) > 0
            assert all(0 <= float(row["average_predicted_score"]) <= 1 for row in rows)
            assert all(0 <= float(row["observed_default_rate"]) <= 1 for row in rows)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_calibration_comparison").fetchone()[0] == len(
            comparison_rows
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM model_calibration_bins_comparison"
        ).fetchone()[0] == len(bin_rows)


def test_calibration_selection_prefers_sigmoid_when_isotonic_brier_gain_is_tiny() -> None:
    rows = [
        {
            "calibration_method": "uncalibrated",
            "split": "validation",
            "brier_score": 0.175,
        },
        {
            "calibration_method": "sigmoid",
            "split": "validation",
            "brier_score": 0.0665,
        },
        {
            "calibration_method": "isotonic",
            "split": "validation",
            "brier_score": 0.0663,
        },
    ]

    assert select_calibration_method(rows) == "sigmoid"
