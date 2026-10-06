from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.model_contracts import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.model_stability import (
    aggregate_stability_rows,
    run_model_stability_experiment,
    select_stability_feature_set,
)
from src.report_contracts import (
    MODEL_STABILITY_AGGREGATE_COLUMNS,
    MODEL_STABILITY_RUN_COLUMNS,
    MODEL_STABILITY_SELECTED_AGGREGATE_COLUMNS,
)
from src.train import run_training
from tests.helpers import (
    create_training_database,
    read_csv_rows,
    write_feature_importance,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names.*:UserWarning"
)


def test_stability_selection_uses_validation_aggregate_not_test_edge() -> None:
    rows = [
        _stability_run("top_100", 17, validation_pr_auc=0.270, test_pr_auc=0.266),
        _stability_run("top_100", 29, validation_pr_auc=0.272, test_pr_auc=0.267),
        _stability_run(
            "full", 17, validation_pr_auc=0.268, test_pr_auc=0.270, feature_count=140
        ),
        _stability_run(
            "full", 29, validation_pr_auc=0.269, test_pr_auc=0.271, feature_count=140
        ),
    ]

    aggregate_rows = aggregate_stability_rows(rows, created_at="2026-04-30T00:00:00Z")
    selected_feature_set = select_stability_feature_set(aggregate_rows)

    assert selected_feature_set == "top_100"
    selected_row = next(
        row for row in aggregate_rows if row["feature_set"] == selected_feature_set
    )
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
    write_feature_importance(
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
    report_text = (report_dir / "experiments" / "006_model_stability.md").read_text(
        encoding="utf-8"
    )
    assert "validation-only aggregate rule" in report_text
    assert "Historical comparison is reported after selection" in report_text
    assert "## Interpretation" in report_text
    assert result["selected_feature_set"] in {"top_3", "top_5", "full"}
    assert len(run_rows) == 6
    assert {row["feature_set"] for row in run_rows} == {"top_3", "top_5", "full"}
    assert {row["seed"] for row in run_rows} == {"17", "29"}
    assert {row["split_seed"] for row in run_rows} == {"17", "29"}
    assert {row["model_seed"] for row in run_rows} == {"42"}
    assert {row["ranking_seeds"] for row in run_rows} == {"101,211,307"}
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
    write_feature_importance(
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
    assert (
        result["report_path"]
        == report_dir / "experiments" / "010_recency_model_stability.md"
    )
    assert result["seed_runs_path"].exists()
    assert result["summary_path"].exists()
    assert result["report_path"].exists()
    assert not (report_dir / "experiments" / "006_model_stability.md").exists()


def test_stability_seeds_keep_saved_test_applicants_out_of_development(
    scratch_path: Path,
    project_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_training_database(scratch_path / "db" / "credit_risk.duckdb", train_rows=80)
    run_training(project_config_path)
    artifact = joblib.load(scratch_path / "models" / LIGHTGBM_MODEL_ARTIFACT_NAME)
    saved_ids = artifact["split_applicant_ids"]
    captured_splits: list[dict[str, set[int]]] = []

    def capture_candidate(
        config: dict,
        feature_set_name: str,
        feature_columns: list[str],
        feature_limit: int | None,
        split_frames: dict[str, pd.DataFrame],
        review_capacity_rate: float,
        created_at: str,
        *,
        random_seed: int,
        error_cls: type[Exception],
    ) -> dict[str, object]:
        captured_splits.append(
            {name: set(frame["SK_ID_CURR"]) for name, frame in split_frames.items()}
        )
        return _stability_run(
            feature_set_name,
            random_seed,
            validation_pr_auc=0.27,
            test_pr_auc=0.26,
            feature_count=len(feature_columns),
        )

    monkeypatch.setattr("src.model_stability.run_single_feature_set", capture_candidate)
    run_model_stability_experiment(
        project_config_path, seeds=(17, 29, 43), feature_limits=(), include_full=True
    )

    test_ids = set(saved_ids["test"])
    development_ids = (
        set(saved_ids["train"])
        | set(saved_ids["validation"])
        | set(saved_ids["calibration"])
    )
    assert len(captured_splits) == 3
    for split_ids in captured_splits:
        assert split_ids["test"] == test_ids
        assert (
            split_ids["train"] | split_ids["validation"] | split_ids["calibration"]
            == development_ids
        )
        assert not split_ids["train"] & split_ids["validation"]
        assert not test_ids & (split_ids["train"] | split_ids["validation"])
        assert not split_ids["calibration"] & (
            test_ids | split_ids["train"] | split_ids["validation"]
        )
    assert captured_splits[0]["validation"] != captured_splits[1]["validation"]


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


def test_current_winner_only_frequency_and_conditional_support(
    scratch_path,
    project_config_path,
    monkeypatch,
):
    database = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database, train_rows=200)
    with duckdb.connect(str(database)) as con:
        for index in range(70):
            con.execute(
                f"ALTER TABLE mart_credit_risk_features ADD COLUMN fixture_extra_{index} DOUBLE DEFAULT {index}"
            )
    run_training(project_config_path)
    config = load_config(project_config_path)
    config["model"]["lightgbm_tuning"]["mode"] = "bounded_inner_cv"
    project_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outcomes = iter([("top_40", 17, 0.90), ("full", 29, 0.50), ("full", 43, 0.70)])

    def chosen_only(
        config, frames, columns, limits, include_full, rate, created_at, error_cls
    ):
        name, seed, ap = next(outcomes)
        row = _stability_run(
            name,
            seed,
            validation_pr_auc=ap,
            test_pr_auc=ap - 0.01,
            feature_count=40 if name == "top_40" else len(columns),
        )
        row.pop("seed")
        row.pop("seed_validation_winner")
        return [row], {name: columns[:40] if name == "top_40" else columns}

    monkeypatch.setattr("src.model_stability.run_joint_feature_sets", chosen_only)
    result = run_model_stability_experiment(
        project_config_path, seeds=(17, 29, 43), feature_limits=(40, 80)
    )
    rows = {row["feature_set"]: row for row in result["aggregate_rows"]}
    assert len(result["run_rows"]) == 3
    assert all(row["seed_validation_winner"] for row in result["run_rows"])
    assert rows["top_40"]["validation_win_rate"] == pytest.approx(1 / 3)
    assert rows["full"]["validation_win_rate"] == pytest.approx(2 / 3)
    assert rows["top_80"]["validation_win_rate"] == 0
    assert rows["top_40"]["seed_count"] == 1
    assert rows["full"]["seed_count"] == 2
    assert rows["top_80"]["seed_count"] == 0
    assert rows["top_40"]["validation_pr_auc_mean"] == 0.90
    assert rows["full"]["validation_pr_auc_mean"] == pytest.approx(0.60)
    assert rows["full"]["validation_pr_auc_std"] == pytest.approx(
        pd.Series([0.50, 0.70]).std(ddof=1)
    )
    assert rows["top_40"]["validation_pr_auc_std"] is None
    assert rows["top_80"]["validation_pr_auc_mean"] is None
    assert rows["top_80"]["validation_pr_auc_std"] is None
    assert {row["completed_split_count"] for row in rows.values()} == {3}
    assert {row["metric_scope"] for row in rows.values()} == {
        "conditional_on_selection"
    }
    assert result["selected_feature_set"] is None
    assert not any(row["selected"] for row in rows.values())
    persisted = {
        r["feature_set"]: r
        for r in read_csv_rows(
            result["summary_path"], MODEL_STABILITY_SELECTED_AGGREGATE_COLUMNS
        )
    }
    assert persisted["top_40"]["validation_pr_auc_std"] == ""
    assert persisted["top_80"]["validation_pr_auc_mean"] == ""
    text = result["report_path"].read_text(encoding="utf-8")
    assert "conditional on" in text and "unavailable standard deviation" in text
    assert "all 3" in text and "no aggregate feature-set selection" in text
    assert "Selected Setup" not in text and "supports promoting" not in text
