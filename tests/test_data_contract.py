import duckdb
import pytest

from src.build_features import run_feature_build
from src.config import load_config
from src.data_contracts import (
    DataContractError,
    build_data_inventory,
    build_feature_inventory,
    get_model_feature_columns,
    validate_data_contracts,
)
from src.report_contracts import DATA_INVENTORY_COLUMNS
from src.report_contracts import FEATURE_INVENTORY_COLUMNS
from tests.helpers import read_csv_rows


def test_data_contract_fails_when_feature_tables_are_missing(staged_feature_fixture) -> None:
    config = load_config(staged_feature_fixture.config_path)

    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection, pytest.raises(
        DataContractError
    ) as error:
        validate_data_contracts(connection, config)

    message = str(error.value)
    assert "Missing required DuckDB tables" in message
    assert "mart_credit_risk_features" in message
    assert not (staged_feature_fixture.scratch_path / "reports" / "data_inventory.csv").exists()
    assert not (staged_feature_fixture.scratch_path / "reports" / "feature_inventory.csv").exists()


def test_data_contract_accepts_valid_mart_and_writes_inventory_reports(
    staged_feature_fixture,
) -> None:
    run_feature_build(staged_feature_fixture.config_path)
    config = load_config(staged_feature_fixture.config_path)

    with duckdb.connect(str(staged_feature_fixture.database_path), read_only=True) as connection:
        validate_data_contracts(connection, config)
        model_features = get_model_feature_columns(connection, config)
        data_inventory = build_data_inventory(connection)
        feature_inventory = build_feature_inventory(connection, config)

    assert {
        "credit_to_income_ratio",
        "bureau_credit_count",
        "bureau_balance_dpd_1plus_rate",
        "pos_cash_dpd_month_rate",
        "credit_card_avg_credit_utilization",
        "payment_amount_ratio",
    }.issubset(model_features)
    assert not {
        "SK_ID_CURR",
        "TARGET",
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
    }.intersection(model_features)

    mart_inventory = next(row for row in data_inventory if row["table_name"] == "mart_credit_risk_features")
    assert mart_inventory["row_count"] == 3
    assert mart_inventory["distinct_applicant_count"] == 3
    assert mart_inventory["duplicate_grain_key_count"] == 0
    assert mart_inventory["has_target_column"] is True
    assert mart_inventory["target_non_null_count"] == 2
    assert mart_inventory["target_null_count"] == 1

    credit_ratio_inventory = next(
        row
        for row in feature_inventory
        if row["table_name"] == "mart_credit_risk_features"
        and row["column_name"] == "credit_to_income_ratio"
    )
    assert credit_ratio_inventory["is_model_feature"] is True
    assert credit_ratio_inventory["exclusion_group"] == ""
    assert credit_ratio_inventory["missing_count"] == 1

    identifier_inventory = next(
        row
        for row in feature_inventory
        if row["table_name"] == "mart_credit_risk_features"
        and row["column_name"] == "SK_ID_CURR"
    )
    assert identifier_inventory["is_model_feature"] is False
    assert identifier_inventory["exclusion_group"] == "identifiers"

    data_report = read_csv_rows(
        staged_feature_fixture.scratch_path / "reports" / "data_inventory.csv",
        DATA_INVENTORY_COLUMNS,
    )
    feature_report = read_csv_rows(
        staged_feature_fixture.scratch_path / "reports" / "feature_inventory.csv",
        FEATURE_INVENTORY_COLUMNS,
    )
    assert len(data_report) == 17
    assert any(
        row["table_name"] == "mart_credit_risk_features"
        and row["duplicate_grain_key_count"] == "0"
        for row in data_report
    )
    assert any(
        row["table_name"] == "segment_diagnostics"
        and row["column_name"] == "CODE_GENDER"
        and row["exclusion_group"] == "sensitive_or_protected_status_like"
        for row in feature_report
    )


def test_data_contract_detects_duplicate_mart_keys(staged_feature_fixture) -> None:
    run_feature_build(staged_feature_fixture.config_path)
    config = load_config(staged_feature_fixture.config_path)

    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            """
            INSERT INTO mart_credit_risk_features
            SELECT *
            FROM mart_credit_risk_features
            WHERE SK_ID_CURR = 100001
            """
        )

        with pytest.raises(DataContractError) as error:
            validate_data_contracts(connection, config)

    assert "Duplicate mart grain keys" in str(error.value)


def test_data_contract_detects_target_population_violations(staged_feature_fixture) -> None:
    run_feature_build(staged_feature_fixture.config_path)
    config = load_config(staged_feature_fixture.config_path)

    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            """
            UPDATE mart_credit_risk_features
            SET TARGET = 1
            WHERE source_population = 'application_test'
            """
        )

        with pytest.raises(DataContractError) as error:
            validate_data_contracts(connection, config)

    assert "application_test rows must have NULL TARGET" in str(error.value)


def test_data_contract_detects_applicant_row_reconciliation_errors(
    staged_feature_fixture,
) -> None:
    run_feature_build(staged_feature_fixture.config_path)
    config = load_config(staged_feature_fixture.config_path)

    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            """
            DELETE FROM mart_credit_risk_features
            WHERE source_population = 'application_test'
            """
        )

        with pytest.raises(DataContractError) as error:
            validate_data_contracts(connection, config)

    assert "application_test staging rows do not reconcile to mart rows" in str(error.value)
