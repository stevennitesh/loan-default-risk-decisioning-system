import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import yaml

from src.config import load_config
from src.feature_experiments import load_split_frames, run_single_feature_set
from src.feature_selection import run_feature_selection_experiment
from src.model_stability import run_model_stability_experiment
from src.nested_assessment import run_nested_assessment
from src.report_contracts import (
    NESTED_PROBABILITY_ACCEPTANCE_COLUMNS,
    NESTED_SEED_SENSITIVITY_COLUMNS,
)
from src.train import run_training
from tests.helpers import create_training_database


def test_current_callers_share_one_joint_budget_and_frozen_roles(project_config_path):
    config = load_config(project_config_path)
    config["resources"] = {"model_threads": 1}
    config["feature_selection"] = {"ranking_seeds": [101, 211, 307]}
    config["model"]["lightgbm_tuning"].update(
        mode="bounded_inner_cv",
        max_candidates=9,
        inner_folds=3,
        max_rounds=8,
        stopping_rounds=2,
        sensitivity_seeds=[101],
    )
    config["assessment"] = {
        "outer_folds": 2,
        "split_seeds": [42],
        "feature_limits": [3],
        "include_full": True,
        "workflows": [
            "history_selected",
            "application_only",
            "logistic_tuned",
            "training_prevalence",
        ],
    }
    project_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    create_training_database(Path(config["paths"]["duckdb_path"]), train_rows=1000)
    trained = run_training(project_config_path)
    artifact = joblib.load(trained["artifacts"]["lightgbm"])
    assert artifact["methodology_version"] == "nested_inner_cv_v3"
    baseline = joblib.load(trained["artifacts"]["logistic_regression"])
    assert baseline["methodology_version"] == "nested_inner_cv_v3"
    assert baseline["baseline_selection"]["candidate_name"].startswith("tuned_C")
    assert baseline["pipeline"].named_steps["classifier"].class_weight is None
    with duckdb.connect(config["paths"]["duckdb_path"], read_only=True) as con:
        splits = load_split_frames(
            con, artifact["split_applicant_ids"], artifact["eligible_feature_columns"]
        )
    direct = run_single_feature_set(
        config,
        "input_alias",
        artifact["eligible_feature_columns"],
        None,
        splits,
        0.1,
        "fixture",
    )
    assert direct["feature_set"] == "full"
    assert direct["feature_count"] == len(artifact["eligible_feature_columns"])
    assert artifact["lightgbm_tuning"]["candidate_count"] == 9
    assert set(artifact["feature_columns"]) <= set(artifact["eligible_feature_columns"])
    selected = run_feature_selection_experiment(
        project_config_path, feature_limits=(3,)
    )
    assert len(selected["comparison_rows"]) == 1
    assert "tuning/feature_selection" in selected["report_path"].as_posix()
    assert "reports/experiments/" not in selected["report_path"].read_text(
        encoding="utf-8"
    )
    run_model_stability_experiment(
        project_config_path, seeds=(17, 29), feature_limits=(3,)
    )
    assessed = run_nested_assessment(project_config_path)
    output = assessed["output_dir"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["protocol_version"] == "nested_inner_cv_v3"
    predictions = pd.read_csv(output / "assessment_predictions.csv")
    assert not predictions.duplicated(["workflow", "split_seed", "SK_ID_CURR"]).any()
    assert not set(predictions["SK_ID_CURR"]) & set(
        manifest["historical_comparison_ids"]
    )
    assert set(predictions["workflow"]) == set(config["assessment"]["workflows"])
    for plan in manifest["folds"]:
        base = set(plan["applicant_ids"]["train"])
        for workflow in ("history_selected", "application_only"):
            evidence = json.loads(
                (output / f"search_42_{plan['outer_fold']}_{workflow}.json").read_text()
            )
            assert len(evidence["candidate_specs"]) == 9
            assert len(evidence["cv_results"]) == 27
            for inner in evidence["inner_memberships"]:
                assert (
                    set(inner["fitting"] + inner["stopping"] + inner["scoring"]) == base
                )
                assert not set(inner["fitting"]) & set(inner["stopping"])
                assert not set(inner["scoring"]) & set(inner["stopping"])
    seed_rows = pd.read_csv(output / "model_seed_sensitivity.csv")
    assert seed_rows.columns.tolist() == NESTED_SEED_SENSITIVITY_COLUMNS
    assert (
        pd.read_csv(output / "probability_acceptance.csv").columns.tolist()
        == NESTED_PROBABILITY_ACCEPTANCE_COLUMNS
    )
    assert set(seed_rows["model_seed"]) == {101}
    assert not seed_rows["promotes_seed"].any()
