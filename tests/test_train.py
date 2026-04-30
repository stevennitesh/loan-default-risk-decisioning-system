import csv
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import pytest

from src.train import (
    MODEL_COMPARISON_SUMMARY_COLUMNS,
    MODEL_METRICS_SUMMARY_COLUMNS,
    MODEL_RUN_SUMMARY_COLUMNS,
    SPLIT_SUMMARY_COLUMNS,
    TrainingError,
    run_training,
)


REQUIRED_METRICS = {
    "roc_auc",
    "pr_auc",
    "brier_score",
    "min_predicted_probability",
    "max_predicted_probability",
    "top_decile_lift",
    "precision_at_top_decile",
    "recall_at_manual_review_capacity",
}

LIGHTGBM_TUNING_SUMMARY_COLUMNS = [
    "candidate_rank",
    "selected",
    "candidate_name",
    "candidate_source",
    "validation_selection_score",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_manual_review_capacity",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
    "created_at",
]

FORBIDDEN_FEATURES = {
    "SK_ID_CURR",
    "TARGET",
    "source_population",
    "SK_ID_PREV",
    "SK_ID_BUREAU",
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "DAYS_BIRTH",
    "applicant_age_years",
    "applicant_age_band",
    "employment_to_age_ratio",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def read_csv_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == expected_columns
        return list(reader)


def create_training_database(database_path: Path, train_rows: int = 40, test_rows: int = 6) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    train_records = [
        _mart_record(
            applicant_id=100000 + index,
            source_population="application_train",
            target=index % 2,
            index=index,
        )
        for index in range(train_rows)
    ]
    test_records = [
        _mart_record(
            applicant_id=200000 + index,
            source_population="application_test",
            target=None,
            index=train_rows + index,
        )
        for index in range(test_rows)
    ]
    mart = pd.DataFrame(train_records + test_records)
    staging_train = mart.loc[mart["source_population"] == "application_train", ["SK_ID_CURR", "TARGET"]]
    staging_test = mart.loc[mart["source_population"] == "application_test", ["SK_ID_CURR"]]
    diagnostics = mart[["SK_ID_CURR", "source_population", "TARGET"]].copy()
    diagnostics["CODE_GENDER"] = ["F" if row % 2 == 0 else "M" for row in range(len(diagnostics))]
    diagnostics["NAME_FAMILY_STATUS"] = "Married"
    diagnostics["applicant_age_years"] = 35
    diagnostics["applicant_age_band"] = "30_to_44"
    diagnostics["CNT_CHILDREN"] = 1
    diagnostics["CNT_FAM_MEMBERS"] = 3

    with duckdb.connect(str(database_path)) as connection:
        _create_table_from_frame(connection, "stg_application_train", staging_train)
        _create_table_from_frame(connection, "stg_application_test", staging_test)
        _create_table_from_frame(
            connection,
            "stg_bureau",
            pd.DataFrame(
                {
                    "SK_ID_BUREAU": range(1, len(mart) + 1),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_bureau_balance",
            pd.DataFrame(
                {
                    "SK_ID_BUREAU": range(1, len(mart) + 1),
                    "MONTHS_BALANCE": [0 for _ in range(len(mart))],
                    "STATUS": ["1" if row % 4 == 0 else "0" for row in range(len(mart))],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_previous_application",
            pd.DataFrame(
                {
                    "SK_ID_PREV": range(1000, 1000 + len(mart)),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_pos_cash_balance",
            pd.DataFrame(
                {
                    "SK_ID_PREV": range(1000, 1000 + len(mart)),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                    "MONTHS_BALANCE": [0 for _ in range(len(mart))],
                    "CNT_INSTALMENT": [12.0 for _ in range(len(mart))],
                    "CNT_INSTALMENT_FUTURE": [6.0 for _ in range(len(mart))],
                    "NAME_CONTRACT_STATUS": [
                        "Active" if row % 2 == 0 else "Completed" for row in range(len(mart))
                    ],
                    "SK_DPD": [1 if row % 5 == 0 else 0 for row in range(len(mart))],
                    "SK_DPD_DEF": [1 if row % 7 == 0 else 0 for row in range(len(mart))],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_credit_card_balance",
            pd.DataFrame(
                {
                    "SK_ID_PREV": range(2000, 2000 + len(mart)),
                    "SK_ID_CURR": mart["SK_ID_CURR"],
                    "MONTHS_BALANCE": [0 for _ in range(len(mart))],
                    "AMT_BALANCE": [100.0 + row for row in range(len(mart))],
                    "AMT_CREDIT_LIMIT_ACTUAL": [1000.0 for _ in range(len(mart))],
                    "AMT_DRAWINGS_CURRENT": [10.0 if row % 3 == 0 else 0.0 for row in range(len(mart))],
                    "AMT_INST_MIN_REGULARITY": [20.0 for _ in range(len(mart))],
                    "AMT_PAYMENT_CURRENT": [20.0 for _ in range(len(mart))],
                    "AMT_PAYMENT_TOTAL_CURRENT": [20.0 for _ in range(len(mart))],
                    "AMT_TOTAL_RECEIVABLE": [100.0 + row for row in range(len(mart))],
                    "CNT_DRAWINGS_CURRENT": [1.0 if row % 3 == 0 else 0.0 for row in range(len(mart))],
                    "NAME_CONTRACT_STATUS": [
                        "Active" if row % 2 == 0 else "Completed" for row in range(len(mart))
                    ],
                    "SK_DPD": [1 if row % 6 == 0 else 0 for row in range(len(mart))],
                    "SK_DPD_DEF": [1 if row % 8 == 0 else 0 for row in range(len(mart))],
                }
            ),
        )
        _create_table_from_frame(
            connection,
            "stg_installments_payments",
            pd.DataFrame({"SK_ID_CURR": mart["SK_ID_CURR"]}),
        )
        _create_table_from_frame(
            connection,
            "f_applicant_static",
            mart[
                [
                    "SK_ID_CURR",
                    "source_population",
                    "TARGET",
                    "credit_to_income_ratio",
                    "category_feature",
                ]
            ],
        )
        _create_table_from_frame(connection, "segment_diagnostics", diagnostics)
        _create_table_from_frame(
            connection,
            "f_bureau_agg",
            mart[["SK_ID_CURR", "bureau_credit_count"]],
        )
        _create_table_from_frame(
            connection,
            "f_bureau_balance_agg",
            mart[
                [
                    "SK_ID_CURR",
                    "bureau_balance_month_count",
                    "bureau_balance_dpd_1plus_rate",
                    "bureau_balance_recent_dpd_1plus_rate",
                ]
            ],
        )
        _create_table_from_frame(
            connection,
            "f_pos_cash_agg",
            mart[
                [
                    "SK_ID_CURR",
                    "pos_cash_month_count",
                    "pos_cash_dpd_month_rate",
                    "pos_cash_recent_dpd_month_rate",
                ]
            ],
        )
        _create_table_from_frame(
            connection,
            "f_credit_card_agg",
            mart[
                [
                    "SK_ID_CURR",
                    "credit_card_month_count",
                    "credit_card_avg_credit_utilization",
                    "credit_card_dpd_month_rate",
                ]
            ],
        )
        _create_table_from_frame(
            connection,
            "f_previous_application_agg",
            mart[["SK_ID_CURR", "previous_application_count"]],
        )
        _create_table_from_frame(
            connection,
            "f_installments_agg",
            mart[["SK_ID_CURR", "payment_amount_ratio"]],
        )
        _create_table_from_frame(connection, "mart_credit_risk_features", mart)


def _mart_record(
    applicant_id: int,
    source_population: str,
    target: int | None,
    index: int,
) -> dict[str, object]:
    return {
        "SK_ID_CURR": applicant_id,
        "source_population": source_population,
        "TARGET": target,
        "credit_to_income_ratio": 1.0 + index / 100.0,
        "bureau_credit_count": index % 5 + 1,
        "bureau_balance_month_count": index % 6 + 1,
        "bureau_balance_dpd_1plus_rate": (index % 4) / 10.0,
        "bureau_balance_recent_dpd_1plus_rate": (index % 3) / 10.0,
        "pos_cash_month_count": index % 8 + 1,
        "pos_cash_dpd_month_rate": (index % 5) / 10.0,
        "pos_cash_recent_dpd_month_rate": (index % 4) / 10.0,
        "credit_card_month_count": index % 7 + 1,
        "credit_card_avg_credit_utilization": (index % 6) / 10.0,
        "credit_card_dpd_month_rate": (index % 4) / 10.0,
        "payment_amount_ratio": 0.75 + (index % 7) / 20.0,
        "previous_application_count": index % 3 + 1,
        "category_feature": ["low", "medium", "high"][index % 3],
        "optional_numeric_feature": None if index % 6 == 0 else index / 10.0,
    }


def _create_table_from_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    connection.register("table_frame", frame)
    connection.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM table_frame')
    connection.unregister("table_frame")


def test_training_fails_clearly_without_duckdb_database(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    with pytest.raises(TrainingError) as error:
        run_training(project_config_path)

    assert "DuckDB database not found" in str(error.value)
    assert not (scratch_path / "models" / "logistic_regression_baseline.joblib").exists()


def test_run_training_creates_model_artifacts_reports_and_duckdb_tables(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path)

    result = run_training(project_config_path)

    baseline_artifact_path = scratch_path / "models" / "logistic_regression_baseline.joblib"
    lightgbm_artifact_path = scratch_path / "models" / "lightgbm_credit_risk.joblib"
    assert baseline_artifact_path.exists()
    assert lightgbm_artifact_path.exists()
    baseline_artifact = joblib.load(baseline_artifact_path)
    lightgbm_artifact = joblib.load(lightgbm_artifact_path)
    assert baseline_artifact["model_type"] == "logistic_regression"
    assert baseline_artifact["model_version"] == "logistic_regression_baseline_v1"
    assert lightgbm_artifact["model_type"] == "lightgbm"
    assert lightgbm_artifact["model_version"] == "lightgbm_credit_risk_v1"
    assert set(result["artifacts"]) == {"logistic_regression", "lightgbm"}
    assert baseline_artifact["feature_columns"] == lightgbm_artifact["feature_columns"]
    assert baseline_artifact["feature_columns"] == result["feature_columns"]
    assert baseline_artifact["split_applicant_ids"] == lightgbm_artifact["split_applicant_ids"]
    assert baseline_artifact["numeric_feature_columns"]
    assert baseline_artifact["categorical_feature_columns"] == ["category_feature"]
    assert lightgbm_artifact["categorical_feature_columns"] == ["category_feature"]
    assert lightgbm_artifact["lightgbm_params"]["scale_pos_weight"] == pytest.approx(1.0)
    assert lightgbm_artifact["lightgbm_tuning"]["selection_metric_order"] == [
        "nonconstant_score_distribution",
        "pr_auc",
        "top_decile_lift",
        "recall_at_manual_review_capacity",
        "roc_auc",
        "brier_score",
    ]
    assert lightgbm_artifact["lightgbm_tuning"]["candidate_count"] == 4
    assert (
        lightgbm_artifact["lightgbm_tuning"]["selected_candidate"]["params"]
        == lightgbm_artifact["lightgbm_params"]
    )
    selected_validation_metrics = lightgbm_artifact["lightgbm_tuning"]["selected_candidate"][
        "validation_metrics"
    ]
    assert selected_validation_metrics["max_predicted_probability"] > selected_validation_metrics[
        "min_predicted_probability"
    ]

    assert {
        "credit_to_income_ratio",
        "bureau_credit_count",
        "payment_amount_ratio",
    }.issubset(baseline_artifact["feature_columns"])
    assert not FORBIDDEN_FEATURES.intersection(baseline_artifact["feature_columns"])
    assert not FORBIDDEN_FEATURES.intersection(lightgbm_artifact["feature_columns"])

    split_ids = {split: set(ids) for split, ids in baseline_artifact["split_applicant_ids"].items()}
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert {split: len(ids) for split, ids in split_ids.items()} == {
        "train": 28,
        "validation": 6,
        "test": 6,
    }

    split_summary = baseline_artifact["split_summary"]
    assert {row["split"] for row in split_summary} == {"train", "validation", "test"}
    assert all(row["positive_count"] > 0 and row["negative_count"] > 0 for row in split_summary)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        labeled_frame = connection.execute(
            """
            SELECT *
            FROM mart_credit_risk_features
            WHERE source_population = 'application_train'
            ORDER BY SK_ID_CURR
            """
        ).fetch_df()
        for artifact in [baseline_artifact, lightgbm_artifact]:
            probabilities = artifact["pipeline"].predict_proba(labeled_frame[artifact["feature_columns"]])[:, 1]
            assert len(probabilities) == len(labeled_frame)
            assert probabilities.min() >= 0
            assert probabilities.max() <= 1

        assert connection.execute("SELECT COUNT(*) FROM model_run_summary").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM split_summary").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM model_metrics_summary").fetchone()[0] == 48
        assert connection.execute("SELECT COUNT(*) FROM model_comparison_summary").fetchone()[0] == 8
        assert connection.execute("SELECT COUNT(*) FROM lightgbm_tuning_summary").fetchone()[0] == 4

    run_rows = read_csv_rows(scratch_path / "reports" / "model_run_summary.csv", MODEL_RUN_SUMMARY_COLUMNS)
    metrics_rows = read_csv_rows(
        scratch_path / "reports" / "model_metrics_summary.csv",
        MODEL_METRICS_SUMMARY_COLUMNS,
    )
    split_rows = read_csv_rows(scratch_path / "reports" / "split_summary.csv", SPLIT_SUMMARY_COLUMNS)
    comparison_rows = read_csv_rows(
        scratch_path / "reports" / "model_comparison_summary.csv",
        MODEL_COMPARISON_SUMMARY_COLUMNS,
    )
    tuning_rows = read_csv_rows(
        scratch_path / "reports" / "lightgbm_tuning_summary.csv",
        LIGHTGBM_TUNING_SUMMARY_COLUMNS,
    )

    assert {row["model_type"] for row in run_rows} == {"logistic_regression", "lightgbm"}
    for row in run_rows:
        assert row["train_rows"] == "28"
        assert row["validation_rows"] == "6"
        assert row["test_rows"] == "6"
        assert int(row["feature_count"]) == len(baseline_artifact["feature_columns"])
    assert {row["split"] for row in split_rows} == {"train", "validation", "test"}

    metrics_by_split = {
        (model_version, split): {
            row["metric_name"]: float(row["metric_value"])
            for row in metrics_rows
            if row["model_version"] == model_version and row["split"] == split
        }
        for model_version in {"logistic_regression_baseline_v1", "lightgbm_credit_risk_v1"}
        for split in {"train", "validation", "test"}
    }
    for split_metrics in metrics_by_split.values():
        assert REQUIRED_METRICS.issubset(split_metrics)
        assert 0 <= split_metrics["min_predicted_probability"] <= 1
        assert 0 <= split_metrics["max_predicted_probability"] <= 1
        assert split_metrics["top_decile_lift"] >= 0

    assert {row["metric_name"] for row in comparison_rows} == REQUIRED_METRICS
    assert {row["selected_model_type"] for row in comparison_rows}.issubset(
        {"logistic_regression", "lightgbm"}
    )
    pr_auc_comparison = next(row for row in comparison_rows if row["metric_name"] == "pr_auc")
    expected_selection = (
        "lightgbm"
        if float(pr_auc_comparison["lightgbm_metric_value"])
        >= float(pr_auc_comparison["baseline_metric_value"])
        else "logistic_regression"
    )
    assert pr_auc_comparison["selected_model_type"] == expected_selection
    assert len(tuning_rows) == 4
    assert {row["candidate_name"] for row in tuning_rows}.issuperset({"baseline_current"})
    selected_tuning_rows = [row for row in tuning_rows if row["selected"] == "True"]
    assert len(selected_tuning_rows) == 1
    assert selected_tuning_rows[0]["validation_pr_auc"] != ""
    assert selected_tuning_rows[0]["validation_recall_at_manual_review_capacity"] != ""


def test_training_wraps_data_contract_failures(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("CREATE TABLE mart_credit_risk_features (SK_ID_CURR BIGINT)")

    with pytest.raises(TrainingError) as error:
        run_training(project_config_path)

    assert "Data contract validation failed before training" in str(error.value)
    assert not (scratch_path / "models" / "logistic_regression_baseline.joblib").exists()
