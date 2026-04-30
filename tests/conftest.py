from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FeatureFixture:
    scratch_path: Path
    config_path: Path
    database_path: Path


@pytest.fixture()
def scratch_path(request: pytest.FixtureRequest) -> Path:
    module_name = request.module.__name__.replace(".", "_")
    safe_name = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in request.node.name
    )
    path = ROOT / ".tmp" / "tests" / module_name / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def project_config_path(scratch_path: Path) -> Path:
    return write_config(scratch_path)


@pytest.fixture()
def staged_feature_fixture(scratch_path: Path) -> FeatureFixture:
    config_path = write_config(scratch_path)
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_staging_tables(database_path)
    return FeatureFixture(
        scratch_path=scratch_path,
        config_path=config_path,
        database_path=database_path,
    )


def write_config(scratch_path: Path) -> Path:
    config = {
        "project": {
            "name": "loan-default-decisioning",
            "random_seed": 42,
            "data_scope_version": "post_v1_test",
        },
        "paths": {
            "raw_dir": str(scratch_path / "raw"),
            "parquet_dir": str(scratch_path / "parquet"),
            "duckdb_path": str(scratch_path / "db" / "credit_risk.duckdb"),
            "model_dir": str(scratch_path / "models"),
            "report_dir": str(scratch_path / "reports"),
            "dashboard_export_dir": str(scratch_path / "reports" / "dashboard_data"),
        },
        "source_files": {
            "application_train": "application_train.csv",
            "application_test": "application_test.csv",
            "bureau": "bureau.csv",
            "bureau_balance": "bureau_balance.csv",
            "pos_cash_balance": "POS_CASH_balance.csv",
            "credit_card_balance": "credit_card_balance.csv",
            "previous_application": "previous_application.csv",
            "installments_payments": "installments_payments.csv",
        },
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
            "calibrate_probabilities": False,
            "lightgbm_tuning": {
                "enabled": True,
                "max_candidates": 4,
            },
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
                "CNT_CHILDREN",
                "CNT_FAM_MEMBERS",
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
    config_path = scratch_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def create_staging_tables(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE stg_application_train (
                SK_ID_CURR BIGINT,
                TARGET BIGINT,
                NAME_CONTRACT_TYPE VARCHAR,
                CODE_GENDER VARCHAR,
                FLAG_OWN_CAR VARCHAR,
                FLAG_OWN_REALTY VARCHAR,
                CNT_CHILDREN BIGINT,
                AMT_INCOME_TOTAL DOUBLE,
                AMT_CREDIT DOUBLE,
                AMT_ANNUITY DOUBLE,
                AMT_GOODS_PRICE DOUBLE,
                NAME_INCOME_TYPE VARCHAR,
                NAME_EDUCATION_TYPE VARCHAR,
                NAME_FAMILY_STATUS VARCHAR,
                NAME_HOUSING_TYPE VARCHAR,
                REGION_POPULATION_RELATIVE DOUBLE,
                DAYS_BIRTH BIGINT,
                DAYS_EMPLOYED BIGINT,
                DAYS_REGISTRATION DOUBLE,
                DAYS_ID_PUBLISH BIGINT,
                OWN_CAR_AGE DOUBLE,
                OCCUPATION_TYPE VARCHAR,
                CNT_FAM_MEMBERS DOUBLE,
                REGION_RATING_CLIENT BIGINT,
                REGION_RATING_CLIENT_W_CITY BIGINT,
                HOUR_APPR_PROCESS_START BIGINT,
                ORGANIZATION_TYPE VARCHAR,
                EXT_SOURCE_1 DOUBLE,
                EXT_SOURCE_2 DOUBLE,
                EXT_SOURCE_3 DOUBLE,
                DAYS_LAST_PHONE_CHANGE DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_application_train VALUES
            (100001, 0, 'Cash loans', 'F', 'N', 'Y', 0, 100000, 200000, 10000, 150000,
             'Working', 'Higher education', 'Single / not married', 'House / apartment',
             0.02, -12000, -1000, -3000, -2500, NULL, 'Laborers', 1, 2, 2, 10,
             'Business Entity Type 3', 0.10, 0.20, NULL, -500),
            (100002, 1, 'Revolving loans', 'M', 'Y', 'N', 2, 0, 50000, 5000, 40000,
             'Commercial associate', 'Secondary / secondary special', 'Married', 'Rented apartment',
             0.03, -16000, 365243, -1000, -1500, 5, 'Managers', 4, 3, 3, 11,
             'Self-employed', NULL, NULL, NULL, -100)
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_application_test AS
            SELECT
                SK_ID_CURR,
                NAME_CONTRACT_TYPE,
                CODE_GENDER,
                FLAG_OWN_CAR,
                FLAG_OWN_REALTY,
                CNT_CHILDREN,
                AMT_INCOME_TOTAL,
                AMT_CREDIT,
                AMT_ANNUITY,
                AMT_GOODS_PRICE,
                NAME_INCOME_TYPE,
                NAME_EDUCATION_TYPE,
                NAME_FAMILY_STATUS,
                NAME_HOUSING_TYPE,
                REGION_POPULATION_RELATIVE,
                DAYS_BIRTH,
                DAYS_EMPLOYED,
                DAYS_REGISTRATION,
                DAYS_ID_PUBLISH,
                OWN_CAR_AGE,
                OCCUPATION_TYPE,
                CNT_FAM_MEMBERS,
                REGION_RATING_CLIENT,
                REGION_RATING_CLIENT_W_CITY,
                HOUR_APPR_PROCESS_START,
                ORGANIZATION_TYPE,
                EXT_SOURCE_1,
                EXT_SOURCE_2,
                EXT_SOURCE_3,
                DAYS_LAST_PHONE_CHANGE
            FROM stg_application_train
            WHERE false
            """
        )
        connection.execute(
            """
            INSERT INTO stg_application_test VALUES
            (200001, 'Cash loans', 'F', 'N', 'Y', 1, 120000, 180000, 12000, 160000,
             'Pensioner', 'Higher education', 'Widow', 'House / apartment', 0.04,
             -20000, -3000, -500, -4500, NULL, NULL, 2, 1, 1, 9, 'XNA',
             0.30, 0.40, 0.50, -30)
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_bureau (
                SK_ID_CURR BIGINT,
                SK_ID_BUREAU BIGINT,
                CREDIT_ACTIVE VARCHAR,
                DAYS_CREDIT BIGINT,
                CREDIT_DAY_OVERDUE BIGINT,
                DAYS_CREDIT_ENDDATE DOUBLE,
                DAYS_ENDDATE_FACT DOUBLE,
                AMT_CREDIT_SUM DOUBLE,
                AMT_CREDIT_SUM_DEBT DOUBLE,
                AMT_CREDIT_SUM_LIMIT DOUBLE,
                AMT_CREDIT_SUM_OVERDUE DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_bureau VALUES
            (100001, 1, 'Active', -100, 0, 300, NULL, 1000, 250, 100, 0),
            (100001, 2, 'Closed', -400, 5, -10, -20, 2000, 0, 0, 50),
            (200001, 3, 'Active', -50, 0, 500, NULL, 500, 100, 30, 0)
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_bureau_balance (
                SK_ID_BUREAU BIGINT,
                MONTHS_BALANCE BIGINT,
                STATUS VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_bureau_balance VALUES
            (1, 0, '0'),
            (1, -1, '1'),
            (1, -13, '2'),
            (2, -2, 'C'),
            (2, -3, 'X'),
            (3, 0, '5')
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_previous_application (
                SK_ID_PREV BIGINT,
                SK_ID_CURR BIGINT,
                AMT_APPLICATION DOUBLE,
                AMT_CREDIT DOUBLE,
                NAME_CONTRACT_STATUS VARCHAR,
                DAYS_DECISION BIGINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_previous_application VALUES
            (10, 100001, 1000, 800, 'Approved', -20),
            (11, 100001, 2000, 2500, 'Refused', -200),
            (12, 100002, 0, 100, 'Canceled', -30)
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_pos_cash_balance (
                SK_ID_PREV BIGINT,
                SK_ID_CURR BIGINT,
                MONTHS_BALANCE BIGINT,
                CNT_INSTALMENT DOUBLE,
                CNT_INSTALMENT_FUTURE DOUBLE,
                NAME_CONTRACT_STATUS VARCHAR,
                SK_DPD BIGINT,
                SK_DPD_DEF BIGINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_pos_cash_balance VALUES
            (10, 100001, 0, 12, 10, 'Active', 0, 0),
            (10, 100001, -1, 12, 9, 'Active', 3, 1),
            (11, 100001, -13, 24, 0, 'Completed', 0, 0),
            (11, 100001, -2, 24, 0, 'Completed', 0, 0),
            (12, 100002, -4, 6, 4, 'Demand', 7, 2),
            (20, 200001, 0, 10, 8, 'Active', 0, 0)
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_credit_card_balance (
                SK_ID_PREV BIGINT,
                SK_ID_CURR BIGINT,
                MONTHS_BALANCE BIGINT,
                AMT_BALANCE DOUBLE,
                AMT_CREDIT_LIMIT_ACTUAL DOUBLE,
                AMT_DRAWINGS_CURRENT DOUBLE,
                AMT_INST_MIN_REGULARITY DOUBLE,
                AMT_PAYMENT_CURRENT DOUBLE,
                AMT_PAYMENT_TOTAL_CURRENT DOUBLE,
                AMT_TOTAL_RECEIVABLE DOUBLE,
                CNT_DRAWINGS_CURRENT DOUBLE,
                NAME_CONTRACT_STATUS VARCHAR,
                SK_DPD BIGINT,
                SK_DPD_DEF BIGINT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_credit_card_balance VALUES
            (30, 100001, 0, 100, 1000, 50, 20, 25, 25, 100, 1, 'Active', 0, 0),
            (30, 100001, -1, 500, 1000, 100, 50, 25, 25, 500, 2, 'Active', 5, 2),
            (31, 100001, -13, 0, 2000, 0, 0, 0, 0, 0, 0, 'Completed', 0, 0),
            (31, 100001, -2, 200, 2000, 40, 10, 20, 20, 200, 1, 'Completed', 0, 0),
            (32, 100002, -4, 300, 1500, 0, 30, 10, 10, 300, 0, 'Demand', 4, 1),
            (40, 200001, 0, 250, 1000, 80, 25, 30, 30, 250, 1, 'Active', 0, 0)
            """
        )
        connection.execute(
            """
            CREATE TABLE stg_installments_payments (
                SK_ID_PREV BIGINT,
                SK_ID_CURR BIGINT,
                NUM_INSTALMENT_NUMBER BIGINT,
                DAYS_INSTALMENT DOUBLE,
                DAYS_ENTRY_PAYMENT DOUBLE,
                AMT_INSTALMENT DOUBLE,
                AMT_PAYMENT DOUBLE
            )
            """
        )
        connection.execute(
            """
            INSERT INTO stg_installments_payments VALUES
            (10, 100001, 1, -10, -8, 100, 90),
            (10, 100001, 2, -5, -7, 100, 100),
            (12, 100002, 1, -4, -1, 50, 25)
            """
        )
