import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.feature_experiments import training_feature_ranking
from src.nested_assessment import (
    NestedAssessmentError,
    assess_workflow,
    assessment_settings,
    fit_fold_workflow,
    make_fold_plan,
    run_nested_assessment,
)
from src.report_contracts import (
    NESTED_CANDIDATE_COLUMNS,
    NESTED_METRIC_COLUMNS,
    NESTED_PREDICTION_COLUMNS,
    NESTED_SUMMARY_COLUMNS,
)
from src.train import run_training
from tests.helpers import create_training_database, read_csv_rows


def _config(path: Path) -> dict:
    config = load_config(path)
    config["project"]["split_seed"] = 42
    config["project"]["model_seed"] = 7
    config["feature_selection"] = {"ranking_seeds": [11, 23]}
    config["assessment"] = {
        "outer_folds": 2,
        "split_seeds": [42],
        "feature_limits": [3],
        "include_full": True,
    }
    config["model"]["lightgbm_tuning"] = {"enabled": False, "max_candidates": 1}
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config


def test_fold_plan_partitions_development_once_per_repeat_and_never_comparison() -> (
    None
):
    frame = pd.DataFrame({"SK_ID_CURR": range(200), "TARGET": [0, 1] * 100})
    settings = assessment_settings(
        {
            "project": {"random_seed": 42},
            "assessment": {"outer_folds": 3, "split_seeds": [42, 17]},
        }
    )
    plans = make_fold_plan(frame, [300, 301], settings)
    assert plans == make_fold_plan(frame.iloc[::-1], [300, 301], settings)
    for seed in settings["split_seeds"]:
        assessment_counts = Counter()
        for plan in [value for value in plans if value["split_seed"] == seed]:
            roles = [set(ids) for ids in plan["applicant_ids"].values()]
            assert set.union(*roles) == set(frame["SK_ID_CURR"])
            for left, right in itertools.combinations(roles + [{300, 301}], 2):
                assert not left & right
            assessment_counts.update(plan["applicant_ids"]["assessment"])
        assert assessment_counts == Counter(
            {applicant: 1 for applicant in frame["SK_ID_CURR"]}
        )


def test_training_rankings_repeat_only_fixed_fitting_rows_and_average_ranks(
    monkeypatch,
) -> None:
    frame = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]})
    calls = []
    order = {101: ["a", "b", "c"], 211: ["b", "a", "c"], 307: ["b", "c", "a"]}

    def rank(training, columns, config, seed, error_cls):
        calls.append((list(training["SK_ID_CURR"]), seed))
        return order[seed]

    monkeypatch.setattr("src.feature_experiments._single_training_ranking", rank)
    ranking = training_feature_ranking(frame, ["a", "b", "c"], {})
    assert calls == [([1, 2], 101), ([1, 2], 211), ([1, 2], 307)]
    assert [row["feature_name"] for row in ranking] == ["b", "a", "c"]
    assert ranking[0]["mean_rank"] == pytest.approx(4 / 3)
    assert ranking[0]["ranking_seed_count"] == 3


def test_workflow_fitting_rejects_an_assessment_argument() -> None:
    with pytest.raises(NestedAssessmentError, match="must not receive"):
        fit_fold_workflow({}, {"assessment": pd.DataFrame()}, [], {})


@pytest.mark.parametrize(
    "override",
    [
        {"outer_folds": 1},
        {"split_seeds": [True]},
        {"split_seeds": [42, 42]},
        {"fit_fraction": 0.8},
        {"feature_limits": [], "include_full": False},
    ],
)
def test_invalid_assessment_settings_fail_before_fitting(override) -> None:
    with pytest.raises(NestedAssessmentError):
        assessment_settings({"project": {"random_seed": 42}, "assessment": override})


def test_nested_cli_assesses_selected_workflows_and_preserves_dashboard_artifacts(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    config = _config(project_config_path)
    database = scratch_path / "db/credit_risk.duckdb"
    create_training_database(database, train_rows=240)
    run_training(project_config_path)
    model_path = scratch_path / "models/lightgbm_credit_risk.joblib"
    original_bytes = model_path.read_bytes()
    saved = joblib.load(model_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.nested_assessment",
            "--config",
            str(project_config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Nested assessment written to" in completed.stdout
    (output,) = (scratch_path / "reports/nested_assessment").iterdir()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["model_family"] == "lightgbm"
    assert manifest["model_seed"] == 7
    assert manifest["ranking_seeds"] == [11, 23]
    predictions = read_csv_rows(
        output / "assessment_predictions.csv", NESTED_PREDICTION_COLUMNS
    )
    metrics = read_csv_rows(output / "fold_metrics.csv", NESTED_METRIC_COLUMNS)
    choices = read_csv_rows(output / "inner_selection.csv", NESTED_CANDIDATE_COLUMNS)
    summary = read_csv_rows(output / "summary.csv", NESTED_SUMMARY_COLUMNS)
    expected = set.union(
        *(
            set(saved["split_applicant_ids"][name])
            for name in ("train", "calibration", "validation")
        )
    )
    assert {row["workflow"] for row in predictions} == {
        "history_selected",
        "training_prevalence",
        "logistic_fixed",
        "application_only",
    }
    for name in {row["workflow"] for row in predictions}:
        assert Counter(
            int(row["SK_ID_CURR"]) for row in predictions if row["workflow"] == name
        ) == Counter({applicant: 1 for applicant in expected})
    assert not any(
        row["workflow"] == "training_prevalence"
        and row["metric_name"] == "balanced_utility_per_applicant"
        for row in metrics
    )
    assert all(
        float(row["metric_value"]) == pytest.approx(1.0)
        for row in metrics
        if row["workflow"] == "training_prevalence"
        and row["metric_name"] == "top_decile_lift"
    )
    assert not expected & set(saved["split_applicant_ids"]["test"])
    assert len(choices) == 4
    assert all(
        row["calibration_method"] == row["selected_calibration_method"]
        for row in choices
    )
    assert sum(row["selected"] == "True" for row in choices) == 2
    assert {row["score_kind"] for row in metrics} == {"raw", "calibrated"}
    assert all(row["fold_count"] == "2" for row in summary)
    assert model_path.read_bytes() == original_bytes
    with duckdb.connect(str(database)) as connection:
        assert (
            connection.execute(
                "SELECT feature_build_id FROM feature_build_metadata"
            ).fetchone()[0]
            == saved["feature_build_id"]
        )
        frame = connection.execute(
            "SELECT * FROM mart_credit_risk_features WHERE source_population='application_train'"
        ).fetch_df()
    fold = manifest["folds"][0]
    assessment = frame.loc[
        frame["SK_ID_CURR"].isin(fold["applicant_ids"]["assessment"])
    ].reset_index(drop=True)
    frozen = joblib.load(output / "fold_42_1.joblib")
    original_hash = joblib.hash(frozen)
    context = frozen["assessment_context"]
    _rows, first = assess_workflow(frozen, assessment, config, context)
    _rows, changed = assess_workflow(
        frozen, assessment.assign(TARGET=1 - assessment["TARGET"]), config, context
    )
    assert [row["raw_score"] for row in first] == [row["raw_score"] for row in changed]
    assert [row["calibrated_score"] for row in first] == [
        row["calibrated_score"] for row in changed
    ]
    assert [row["target"] for row in first] != [row["target"] for row in changed]
    assert joblib.hash(frozen) == original_hash
    # Outer covariates may change scores, but cannot change the fitted choices
    # or training-only preprocessor statistics.
    changed_features = assessment.copy()
    for feature in frozen["feature_columns"]:
        if pd.api.types.is_numeric_dtype(changed_features[feature]):
            changed_features[feature] = 1000000.0
    assess_workflow(frozen, changed_features, config, context)
    assert joblib.hash(frozen) == original_hash
    fitting_applicant = frozen["fit_applicant_ids"][0]
    with pytest.raises(NestedAssessmentError, match="overlap"):
        assess_workflow(
            frozen, assessment.assign(SK_ID_CURR=fitting_applicant), config, context
        )
    report = (output / "assessment_report.md").read_text(encoding="utf-8")
    assert "not a confidence interval" in report
    assert "Mean across applicant test groups" in report
    assert "five test groups" not in report  # This fixture assesses two groups.
    narrative, run_details = report.split("## Technical run details")
    assert manifest["run_id"] not in narrative
    assert manifest["run_id"] in run_details


def test_changing_model_seed_does_not_change_training_membership(
    scratch_path: Path, project_config_path: Path
) -> None:
    config = _config(project_config_path)
    create_training_database(scratch_path / "db/credit_risk.duckdb", train_rows=120)
    run_training(project_config_path)
    path = scratch_path / "models/lightgbm_credit_risk.joblib"
    original = joblib.load(path)
    config["project"]["model_seed"] = 19
    project_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_training(project_config_path)
    changed = joblib.load(path)
    assert original["split_applicant_ids"] == changed["split_applicant_ids"]
    assert original["model_seed"] == 7 and changed["model_seed"] == 19
    assert original["split_seed"] == changed["split_seed"] == 42


def test_failed_nested_run_is_identified_without_publishing_complete_metrics(
    scratch_path: Path,
    project_config_path: Path,
    monkeypatch,
) -> None:
    _config(project_config_path)
    create_training_database(scratch_path / "db/credit_risk.duckdb", train_rows=120)
    run_training(project_config_path)

    def fail(*args, **kwargs):
        raise NestedAssessmentError("fixture fitting failure")

    monkeypatch.setattr("src.nested_assessment.fit_fold_workflow", fail)
    with pytest.raises(NestedAssessmentError, match="fixture fitting failure"):
        run_nested_assessment(project_config_path)
    (output,) = (scratch_path / "reports/nested_assessment").iterdir()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (output / "summary.csv").exists()
