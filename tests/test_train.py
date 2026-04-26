import csv
from pathlib import Path

import duckdb
import joblib
import pandas as pd
import pytest

from src.train import (
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
}

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


def test_run_training_creates_baseline_artifact_reports_and_duckdb_tables(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path)

    result = run_training(project_config_path)

    artifact_path = scratch_path / "models" / "logistic_regression_baseline.joblib"
    assert artifact_path.exists()
    artifact = joblib.load(artifact_path)
    assert artifact["model_type"] == "logistic_regression"
    assert artifact["model_version"] == "logistic_regression_baseline_v1"
    assert artifact["feature_columns"] == result["feature_columns"]
    assert artifact["numeric_feature_columns"]
    assert artifact["categorical_feature_columns"] == ["category_feature"]

    assert {
        "credit_to_income_ratio",
        "bureau_credit_count",
        "payment_amount_ratio",
    }.issubset(artifact["feature_columns"])
    assert not FORBIDDEN_FEATURES.intersection(artifact["feature_columns"])

    split_ids = {split: set(ids) for split, ids in artifact["split_applicant_ids"].items()}
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert {split: len(ids) for split, ids in split_ids.items()} == {
        "train": 28,
        "validation": 6,
        "test": 6,
    }

    split_summary = artifact["split_summary"]
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
        probabilities = artifact["pipeline"].predict_proba(labeled_frame[artifact["feature_columns"]])[:, 1]
        assert len(probabilities) == len(labeled_frame)
        assert probabilities.min() >= 0
        assert probabilities.max() <= 1

        assert connection.execute("SELECT COUNT(*) FROM model_run_summary").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM split_summary").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM model_metrics_summary").fetchone()[0] == 15

    run_rows = read_csv_rows(scratch_path / "reports" / "model_run_summary.csv", MODEL_RUN_SUMMARY_COLUMNS)
    metrics_rows = read_csv_rows(
        scratch_path / "reports" / "model_metrics_summary.csv",
        MODEL_METRICS_SUMMARY_COLUMNS,
    )
    split_rows = read_csv_rows(scratch_path / "reports" / "split_summary.csv", SPLIT_SUMMARY_COLUMNS)

    assert run_rows[0]["model_type"] == "logistic_regression"
    assert run_rows[0]["train_rows"] == "28"
    assert run_rows[0]["validation_rows"] == "6"
    assert run_rows[0]["test_rows"] == "6"
    assert int(run_rows[0]["feature_count"]) == len(artifact["feature_columns"])
    assert {row["split"] for row in split_rows} == {"train", "validation", "test"}

    metrics_by_split = {
        split: {
            row["metric_name"]: float(row["metric_value"])
            for row in metrics_rows
            if row["split"] == split
        }
        for split in {"train", "validation", "test"}
    }
    for split_metrics in metrics_by_split.values():
        assert REQUIRED_METRICS.issubset(split_metrics)
        assert 0 <= split_metrics["min_predicted_probability"] <= 1
        assert 0 <= split_metrics["max_predicted_probability"] <= 1


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
