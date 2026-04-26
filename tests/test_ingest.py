import csv
import shutil
from pathlib import Path

import duckdb
import pytest
import yaml

from src.ingest import IngestionError, run_ingestion


ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}

EXPECTED_STAGING_TABLES = {
    "application_train": "stg_application_train",
    "application_test": "stg_application_test",
    "bureau": "stg_bureau",
    "previous_application": "stg_previous_application",
    "installments_payments": "stg_installments_payments",
}

EXPECTED_SUMMARY_COLUMNS = [
    "source_name",
    "source_file",
    "raw_path",
    "parquet_path",
    "staging_table",
    "csv_rows",
    "parquet_rows",
    "duckdb_rows",
    "created_at_utc",
]


@pytest.fixture()
def scratch_path(request: pytest.FixtureRequest) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in request.node.name
    )
    path = ROOT / ".tmp" / "tests" / "ingest" / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def write_config(tmp_path: Path) -> Path:
    config = {
        "project": {
            "name": "loan-default-decisioning",
            "random_seed": 42,
            "data_scope_version": "v1",
        },
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "parquet_dir": str(tmp_path / "parquet"),
            "duckdb_path": str(tmp_path / "db" / "credit_risk.duckdb"),
            "model_dir": str(tmp_path / "models"),
            "report_dir": str(tmp_path / "reports"),
            "dashboard_export_dir": str(tmp_path / "reports" / "dashboard_data"),
        },
        "source_files": SOURCE_FILES,
        "split": {
            "train_size": 0.70,
            "validation_size": 0.15,
            "test_size": 0.15,
            "stratify": True,
        },
        "model": {
            "primary_model": "lightgbm",
            "baseline_model": "logistic_regression",
            "use_class_weighting": True,
            "calibrate_probabilities": True,
        },
        "excluded_features": {
            "identifiers": ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"],
            "target": ["TARGET"],
            "sensitive_or_protected_status_like": [
                "CODE_GENDER",
                "NAME_FAMILY_STATUS",
                "DAYS_BIRTH",
                "applicant_age_years",
                "applicant_age_band",
                "employment_to_age_ratio",
            ],
        },
        "business_assumptions": {
            "expected_margin_per_good_loan": 1000,
            "expected_loss_per_bad_loan": 5000,
            "manual_review_cost": 50,
            "manual_review_capacity_rate": 0.10,
        },
        "threshold_policy": {
            "threshold_version": "threshold_v1",
            "scenarios": {
                "growth_oriented": {"threshold_low": None, "threshold_high": None},
                "balanced": {"threshold_low": None, "threshold_high": None},
                "risk_averse": {"threshold_low": None, "threshold_high": None},
            },
        },
    }
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def write_required_csvs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_contents = {
        "application_train.csv": (
            "SK_ID_CURR,TARGET,AMT_INCOME_TOTAL\n"
            "100001,0,100000\n"
            "100002,1,75000\n"
        ),
        "application_test.csv": (
            "SK_ID_CURR,AMT_INCOME_TOTAL\n"
            "200001,120000\n"
            "200002,64000\n"
        ),
        "bureau.csv": (
            "SK_ID_CURR,SK_ID_BUREAU,CREDIT_ACTIVE\n"
            "100001,500001,Active\n"
            "100001,500002,Closed\n"
            "200001,500003,Active\n"
        ),
        "previous_application.csv": (
            "SK_ID_CURR,SK_ID_PREV,NAME_CONTRACT_STATUS\n"
            "100001,700001,Approved\n"
            "100002,700002,Refused\n"
        ),
        "installments_payments.csv": (
            "SK_ID_CURR,SK_ID_PREV,NUM_INSTALMENT_NUMBER,AMT_PAYMENT\n"
            "100001,700001,1,250.00\n"
            "100001,700001,2,250.00\n"
            "100002,700002,1,125.00\n"
        ),
    }
    for filename, content in csv_contents.items():
        (raw_dir / filename).write_text(content, encoding="utf-8")


def read_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        reader = csv.DictReader(summary_file)
        assert reader.fieldnames == EXPECTED_SUMMARY_COLUMNS
        return list(reader)


def table_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()}


def test_ingestion_fails_before_conversion_when_required_raw_files_are_missing(
    scratch_path: Path,
) -> None:
    config_path = write_config(scratch_path)
    raw_dir = scratch_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET\n100001,0\n",
        encoding="utf-8",
    )

    with pytest.raises(IngestionError) as error:
        run_ingestion(config_path)

    message = str(error.value)
    assert "Missing required raw CSV files" in message
    assert "application_test.csv" in message
    assert "bureau.csv" in message
    assert not (scratch_path / "parquet" / "application_train.parquet").exists()
    assert not (scratch_path / "db" / "credit_risk.duckdb").exists()


def test_ingestion_converts_required_csvs_and_loads_duckdb_staging_without_optional_docs(
    scratch_path: Path,
) -> None:
    config_path = write_config(scratch_path)
    write_required_csvs(scratch_path / "raw")

    summary = run_ingestion(config_path)
    second_summary = run_ingestion(config_path)

    assert {row["source_name"] for row in summary} == set(SOURCE_FILES)
    assert {row["source_name"] for row in second_summary} == set(SOURCE_FILES)

    for source_name in SOURCE_FILES:
        assert (scratch_path / "parquet" / f"{source_name}.parquet").exists()

    summary_rows = read_summary(scratch_path / "reports" / "ingestion_summary.csv")
    assert len(summary_rows) == len(SOURCE_FILES)
    assert {row["source_name"] for row in summary_rows} == set(SOURCE_FILES)

    with duckdb.connect(str(scratch_path / "db" / "credit_risk.duckdb"), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        assert set(EXPECTED_STAGING_TABLES.values()).issubset(tables)

        for row in summary_rows:
            assert row["staging_table"] == EXPECTED_STAGING_TABLES[row["source_name"]]
            assert row["csv_rows"] == row["parquet_rows"] == row["duckdb_rows"]
            duckdb_rows = connection.execute(
                f"SELECT COUNT(*) FROM {row['staging_table']}"
            ).fetchone()[0]
            assert duckdb_rows == int(row["duckdb_rows"])

        assert "TARGET" in table_columns(connection, "stg_application_train")
        assert "TARGET" not in table_columns(connection, "stg_application_test")
