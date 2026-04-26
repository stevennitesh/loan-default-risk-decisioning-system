import csv
import shutil
from pathlib import Path

import duckdb
import pytest
import yaml

from src.build_features import FeatureBuildError, run_feature_build


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MART_COLUMNS = {
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "DAYS_BIRTH",
    "applicant_age_years",
    "applicant_age_band",
    "employment_to_age_ratio",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}
PROFILE_COLUMNS = [
    "table_name",
    "row_count",
    "distinct_applicant_count",
    "duplicate_key_count",
    "column_count",
    "created_at_utc",
]


@pytest.fixture()
def scratch_path(request: pytest.FixtureRequest) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in request.node.name
    )
    path = ROOT / ".tmp" / "tests" / "feature_sql" / safe_name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def write_config(scratch_path: Path) -> Path:
    config = {
        "project": {
            "name": "loan-default-decisioning",
            "random_seed": 42,
            "data_scope_version": "v1",
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


def table_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()}


def read_profile(profile_path: Path) -> list[dict[str, str]]:
    with profile_path.open(newline="", encoding="utf-8") as profile_file:
        reader = csv.DictReader(profile_file)
        assert reader.fieldnames == PROFILE_COLUMNS
        return list(reader)


def test_feature_build_fails_clearly_when_staging_tables_are_missing(scratch_path: Path) -> None:
    config_path = write_config(scratch_path)

    with pytest.raises(FeatureBuildError) as error:
        run_feature_build(config_path)

    message = str(error.value)
    assert "Missing required staging tables" in message
    assert "stg_application_train" in message
    assert not (scratch_path / "reports" / "feature_mart_profile.csv").exists()


def test_feature_build_creates_feature_tables_mart_diagnostics_and_profile(
    scratch_path: Path,
) -> None:
    config_path = write_config(scratch_path)
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_staging_tables(database_path)

    profile_rows = run_feature_build(config_path)

    assert {row["table_name"] for row in profile_rows} == {
        "f_applicant_static",
        "segment_diagnostics",
        "f_bureau_agg",
        "f_previous_application_agg",
        "f_installments_agg",
        "mart_credit_risk_features",
    }

    with duckdb.connect(str(database_path), read_only=True) as connection:
        mart_columns = table_columns(connection, "mart_credit_risk_features")
        diagnostic_columns = table_columns(connection, "segment_diagnostics")

        assert not FORBIDDEN_MART_COLUMNS.intersection(mart_columns)
        assert {
            "CODE_GENDER",
            "NAME_FAMILY_STATUS",
            "applicant_age_years",
            "applicant_age_band",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
        }.issubset(diagnostic_columns)

        row = connection.execute(
            """
            SELECT
                credit_to_income_ratio,
                annuity_to_income_ratio,
                ext_source_mean,
                ext_source_missing_count,
                bureau_credit_count,
                active_credit_count,
                overdue_credit_count,
                total_credit_sum,
                previous_application_count,
                approved_application_count,
                refused_application_count,
                approval_rate,
                avg_credit_to_application_ratio,
                installment_payment_count,
                late_payment_count,
                max_payment_delay_days,
                underpayment_count,
                payment_amount_ratio
            FROM mart_credit_risk_features
            WHERE SK_ID_CURR = 100001
            """
        ).fetchone()

        assert row == pytest.approx(
            (
                2.0,
                0.1,
                0.15,
                1,
                2,
                1,
                1,
                3000.0,
                2,
                1,
                1,
                0.5,
                1.025,
                2,
                1,
                2.0,
                1,
                0.95,
            )
        )

        zero_income_ratios = connection.execute(
            """
            SELECT credit_to_income_ratio, annuity_to_income_ratio, goods_price_to_income_ratio
            FROM mart_credit_risk_features
            WHERE SK_ID_CURR = 100002
            """
        ).fetchone()
        assert zero_income_ratios == (None, None, None)

        population_counts = dict(
            connection.execute(
                """
                SELECT source_population, COUNT(*)
                FROM mart_credit_risk_features
                GROUP BY source_population
                """
            ).fetchall()
        )
        assert population_counts == {"application_train": 2, "application_test": 1}

        target_counts = connection.execute(
            """
            SELECT
                SUM(CASE WHEN source_population = 'application_train' AND TARGET IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN source_population = 'application_test' AND TARGET IS NULL THEN 1 ELSE 0 END)
            FROM mart_credit_risk_features
            """
        ).fetchone()
        assert target_counts == (2, 1)

        duplicate_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT SK_ID_CURR, source_population, COUNT(*) AS row_count
                FROM mart_credit_risk_features
                GROUP BY SK_ID_CURR, source_population
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        assert duplicate_count == 0

    saved_profile = read_profile(scratch_path / "reports" / "feature_mart_profile.csv")
    mart_profile = next(row for row in saved_profile if row["table_name"] == "mart_credit_risk_features")
    assert mart_profile["row_count"] == "3"
    assert mart_profile["distinct_applicant_count"] == "3"
    assert mart_profile["duplicate_key_count"] == "0"
