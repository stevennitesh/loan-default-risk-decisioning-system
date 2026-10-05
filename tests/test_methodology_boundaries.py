import itertools
import subprocess
import sys
from pathlib import Path

import duckdb
import joblib
import pytest
import yaml

from src.calibrate import run_calibration_experiment
from src.calibration import select_calibration_method
from src.dashboard_exports import DashboardExportError, run_dashboard_export
from src.evaluate import EvaluationError, run_evaluation
from src.explain import run_explain
from src.feature_selection import run_feature_selection_experiment
from src.score_batch import ScoringError, run_scoring
from src.train import run_training
from tests.helpers import create_training_database


def test_evaluation_does_not_import_stale_calibration_reports(
    scratch_path: Path, project_config_path: Path
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    run_training(project_config_path)
    artifact = joblib.load(scratch_path / "models" / "lightgbm_credit_risk.joblib")
    report_dir = scratch_path / "reports"
    (report_dir / "model_calibration_comparison.csv").write_text(
        "calibration_method,split,brier_score,weighted_calibration_error\n"
        "uncalibrated,validation,0.9,0.9\n"
        "uncalibrated,test,0.9,0.9\n"
        "sigmoid,validation,0.000001,0.000001\n"
        "sigmoid,test,0.000001,0.000001\n",
        encoding="utf-8",
    )
    run_evaluation(project_config_path)
    validation_report = (report_dir / "validation_report.md").read_text(
        encoding="utf-8"
    )
    business_report = (report_dir / "business_value_analysis.md").read_text(
        encoding="utf-8"
    )
    assert artifact["run_id"] in validation_report
    assert artifact["feature_build_id"] in validation_report
    for report in (validation_report, business_report):
        assert "Experiment 004" not in report
        assert "0.000001" not in report
        assert "large improvement" not in report


def test_training_seed_changes_preserve_comparison_ids_and_disjoint_roles(
    scratch_path: Path, project_config_path: Path
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    run_training(project_config_path)
    model_path = scratch_path / "models" / "lightgbm_credit_risk.joblib"
    original = joblib.load(model_path)["split_applicant_ids"]
    run_training(project_config_path)
    assert joblib.load(model_path)["split_applicant_ids"] == original
    config = yaml.safe_load(project_config_path.read_text())
    config["project"]["random_seed"] = 17
    project_config_path.write_text(yaml.safe_dump(config))
    run_training(project_config_path)
    changed = joblib.load(model_path)["split_applicant_ids"]
    assert changed["test"] == original["test"]
    for left, right in itertools.combinations(changed.values(), 2):
        assert not set(left).intersection(right)


def test_non_calibration_labels_cannot_change_fitted_calibrators(
    scratch_path: Path, project_config_path: Path
) -> None:
    database = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database, train_rows=80)
    run_training(project_config_path)
    first = run_calibration_experiment(project_config_path)
    artifact = joblib.load(first["artifact"])
    fitted_hash = joblib.hash(artifact["calibrators"])
    fit_ids = artifact["fit_applicant_ids"]
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE mart_credit_risk_features SET TARGET=1-TARGET WHERE source_population='application_train' AND SK_ID_CURR NOT IN (SELECT unnest(?))",
            [fit_ids],
        )
    second = run_calibration_experiment(project_config_path)
    assert joblib.hash(joblib.load(second["artifact"])["calibrators"]) == fitted_hash


def test_feature_ranking_ignores_reporting_shap_and_non_training_labels(
    scratch_path: Path, project_config_path: Path
) -> None:
    database = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database, train_rows=80)
    run_training(project_config_path)
    first = run_feature_selection_experiment(
        project_config_path, feature_limits=(3,), include_full=False
    )
    chosen = first["selected_features_path"].read_text()
    (scratch_path / "reports" / "model_feature_importance.csv").write_text(
        "invalid reporting input"
    )
    artifact = joblib.load(scratch_path / "models" / "lightgbm_credit_risk.joblib")
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE mart_credit_risk_features SET TARGET=1-TARGET WHERE source_population='application_train' AND SK_ID_CURR NOT IN (SELECT unnest(?))",
            [artifact["split_applicant_ids"]["train"]],
        )
    second = run_feature_selection_experiment(
        project_config_path, feature_limits=(3,), include_full=False
    )
    assert second["selected_features_path"].read_text() == chosen


def test_scoring_rejects_thresholds_from_an_earlier_fit(
    scratch_path: Path, project_config_path: Path
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    run_training(project_config_path)
    run_evaluation(project_config_path)
    run_training(project_config_path)
    with pytest.raises(ScoringError, match="evaluation_run_identity.*match"):
        run_scoring(project_config_path)


def test_training_cli_runs_the_corrected_protocol(
    scratch_path: Path, project_config_path: Path
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(project_config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = joblib.load(scratch_path / "models" / "lightgbm_credit_risk.joblib")
    assert artifact["methodology_version"] == "disjoint_calibration_v2"


def test_rebuilt_features_reject_preexisting_models(
    scratch_path: Path, project_config_path: Path
) -> None:
    database = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database, train_rows=80)
    run_training(project_config_path)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE feature_build_metadata SET feature_build_id='new_build'"
        )
    with pytest.raises(EvaluationError, match="current feature build"):
        run_evaluation(project_config_path)


def test_regenerated_calibration_requires_rescoring_before_export(
    scratch_path: Path, project_config_path: Path
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    run_training(project_config_path)
    run_evaluation(project_config_path)
    run_calibration_experiment(project_config_path)
    run_scoring(project_config_path)
    run_explain(project_config_path)
    run_calibration_experiment(project_config_path)
    with pytest.raises(DashboardExportError, match="Scored calibration.*rerun scoring"):
        run_dashboard_export(
            project_config_path, use_calibrated_probability_quality=True
        )


@pytest.mark.parametrize("bad_brier", [float("nan"), float("inf"), -0.1, 1.1])
def test_calibration_selector_rejects_invalid_metrics(bad_brier: float) -> None:
    rows = [
        {"split": "validation", "calibration_method": method, "brier_score": score}
        for method, score in [
            ("uncalibrated", 0.2),
            ("sigmoid", bad_brier),
            ("isotonic", 0.19),
        ]
    ]
    with pytest.raises(ValueError, match="finite"):
        select_calibration_method(rows)
