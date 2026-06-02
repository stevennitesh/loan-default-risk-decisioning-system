from pathlib import Path

import duckdb
import pytest

from src.ingest import IngestionError, run_ingestion
from src.report_contracts import INGESTION_SUMMARY_COLUMNS
from src.runtime import ensure_directories
from tests.helpers import (
    read_csv_rows,
    read_table_columns,
    table_names,
    table_row_count,
)

SOURCE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}

EXPECTED_STAGING_TABLES = {
    "application_train": "stg_application_train",
    "application_test": "stg_application_test",
    "bureau": "stg_bureau",
    "bureau_balance": "stg_bureau_balance",
    "pos_cash_balance": "stg_pos_cash_balance",
    "credit_card_balance": "stg_credit_card_balance",
    "previous_application": "stg_previous_application",
    "installments_payments": "stg_installments_payments",
}


def test_ingestion_fails_before_conversion_when_required_raw_files_are_missing(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    raw_dir = scratch_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "application_train.csv").write_text(
        "SK_ID_CURR,TARGET\n100001,0\n",
        encoding="utf-8",
    )

    with pytest.raises(IngestionError) as error:
        run_ingestion(project_config_path)

    message = str(error.value)
    assert "Missing required raw CSV files" in message
    assert "application_test.csv" in message
    assert "bureau.csv" in message
    assert not (scratch_path / "parquet" / "application_train.parquet").exists()
    assert not (scratch_path / "db" / "credit_risk.duckdb").exists()


def test_ingestion_converts_required_csvs_and_loads_duckdb_staging_without_optional_docs(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    _write_required_csvs(scratch_path / "raw")

    summary = run_ingestion(project_config_path)
    second_summary = run_ingestion(project_config_path)

    assert {row["source_name"] for row in summary} == set(SOURCE_FILES)
    assert {row["source_name"] for row in second_summary} == set(SOURCE_FILES)

    for source_name in SOURCE_FILES:
        assert (scratch_path / "parquet" / f"{source_name}.parquet").exists()

    summary_rows = read_csv_rows(
        scratch_path / "reports" / "ingestion_summary.csv",
        INGESTION_SUMMARY_COLUMNS,
    )
    assert len(summary_rows) == len(SOURCE_FILES)
    assert {row["source_name"] for row in summary_rows} == set(SOURCE_FILES)

    with duckdb.connect(
        str(scratch_path / "db" / "credit_risk.duckdb"), read_only=True
    ) as connection:
        tables = table_names(connection)
        assert set(EXPECTED_STAGING_TABLES.values()).issubset(tables)

        for row in summary_rows:
            assert row["staging_table"] == EXPECTED_STAGING_TABLES[row["source_name"]]
            assert row["csv_rows"] == row["parquet_rows"] == row["duckdb_rows"]
            assert table_row_count(connection, row["staging_table"]) == int(
                row["duckdb_rows"]
            )

        assert "TARGET" in read_table_columns(connection, "stg_application_train")
        assert "TARGET" not in read_table_columns(connection, "stg_application_test")


def _write_required_csvs(raw_dir: Path) -> None:
    ensure_directories(raw_dir)
    csv_contents = {
        "application_train.csv": (
            "SK_ID_CURR,TARGET,AMT_INCOME_TOTAL\n100001,0,100000\n100002,1,75000\n"
        ),
        "application_test.csv": (
            "SK_ID_CURR,AMT_INCOME_TOTAL\n200001,120000\n200002,64000\n"
        ),
        "bureau.csv": (
            "SK_ID_CURR,SK_ID_BUREAU,CREDIT_ACTIVE\n"
            "100001,500001,Active\n"
            "100001,500002,Closed\n"
            "200001,500003,Active\n"
        ),
        "bureau_balance.csv": (
            "SK_ID_BUREAU,MONTHS_BALANCE,STATUS\n"
            "500001,0,0\n"
            "500001,-1,1\n"
            "500002,-2,C\n"
            "500003,-3,X\n"
        ),
        "POS_CASH_balance.csv": (
            "SK_ID_PREV,SK_ID_CURR,MONTHS_BALANCE,CNT_INSTALMENT,CNT_INSTALMENT_FUTURE,"
            "NAME_CONTRACT_STATUS,SK_DPD,SK_DPD_DEF\n"
            "700001,100001,0,12,10,Active,0,0\n"
            "700001,100001,-1,12,9,Active,3,1\n"
            "700002,100002,-2,6,4,Demand,7,2\n"
            "700003,200001,-3,10,8,Completed,0,0\n"
        ),
        "credit_card_balance.csv": (
            "SK_ID_PREV,SK_ID_CURR,MONTHS_BALANCE,AMT_BALANCE,AMT_CREDIT_LIMIT_ACTUAL,"
            "AMT_DRAWINGS_CURRENT,AMT_INST_MIN_REGULARITY,AMT_PAYMENT_CURRENT,"
            "AMT_PAYMENT_TOTAL_CURRENT,AMT_TOTAL_RECEIVABLE,CNT_DRAWINGS_CURRENT,"
            "NAME_CONTRACT_STATUS,SK_DPD,SK_DPD_DEF\n"
            "800001,100001,0,100,1000,50,20,25,25,100,1,Active,0,0\n"
            "800001,100001,-1,500,1000,100,50,25,25,500,2,Active,5,2\n"
            "800002,100002,-2,300,1500,0,30,10,10,300,0,Demand,4,1\n"
            "800003,200001,-3,200,2000,40,10,20,20,200,1,Completed,0,0\n"
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
