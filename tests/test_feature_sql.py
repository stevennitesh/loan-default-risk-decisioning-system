from pathlib import Path

import duckdb
import pytest
import yaml

from src.build_features import FeatureBuildError, run_feature_build
from src.report_contracts import FEATURE_PROFILE_COLUMNS
from tests.helpers import query_value, read_csv_rows, read_table_columns, table_names

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
V1_SOURCE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}
V1_PROFILE_TABLES = {
    "f_applicant_static",
    "segment_diagnostics",
    "f_bureau_agg",
    "f_previous_application_agg",
    "f_installments_agg",
    "mart_credit_risk_features",
}
POST_V1_TABLES = {
    "f_bureau_balance_agg",
    "f_pos_cash_agg",
    "f_credit_card_agg",
    "f_recency_deterioration_features",
    "f_risk_pressure_features",
    "f_last_k_temporal_features",
}


def test_feature_build_fails_clearly_when_staging_tables_are_missing(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    with pytest.raises(FeatureBuildError) as error:
        run_feature_build(project_config_path)

    message = str(error.value)
    assert "Missing required staging tables" in message
    assert "stg_application_train" in message
    assert not (scratch_path / "reports" / "feature_mart_profile.csv").exists()


def test_feature_build_creates_feature_tables_mart_diagnostics_and_profile(
    staged_feature_fixture,
) -> None:
    profile_rows = run_feature_build(staged_feature_fixture.config_path)

    assert {row["table_name"] for row in profile_rows} == {
        "f_applicant_static",
        "segment_diagnostics",
        "f_bureau_agg",
        "f_bureau_balance_agg",
        "f_pos_cash_agg",
        "f_credit_card_agg",
        "f_recency_deterioration_features",
        "f_risk_pressure_features",
        "f_previous_application_agg",
        "f_installments_agg",
        "f_last_k_temporal_features",
        "mart_credit_risk_features",
    }

    with duckdb.connect(
        str(staged_feature_fixture.database_path), read_only=True
    ) as connection:
        mart_columns = read_table_columns(connection, "mart_credit_risk_features")
        pressure_columns = read_table_columns(connection, "f_risk_pressure_features")
        diagnostic_columns = read_table_columns(connection, "segment_diagnostics")

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
                bureau_balance_month_count,
                bureau_balance_bureau_count,
                bureau_balance_dpd_1plus_count,
                bureau_balance_dpd_2plus_count,
                bureau_balance_max_status,
                bureau_balance_dpd_1plus_rate,
                bureau_balance_recent_dpd_1plus_rate,
                pos_cash_month_count,
                pos_cash_contract_count,
                pos_cash_active_month_count,
                pos_cash_completed_month_count,
                pos_cash_dpd_month_count,
                pos_cash_dpd_def_month_count,
                pos_cash_max_dpd,
                pos_cash_dpd_month_rate,
                pos_cash_recent_dpd_month_rate,
                credit_card_month_count,
                credit_card_contract_count,
                credit_card_active_month_count,
                credit_card_completed_month_count,
                credit_card_dpd_month_count,
                credit_card_dpd_def_month_count,
                credit_card_max_dpd,
                credit_card_avg_balance,
                credit_card_max_balance,
                credit_card_avg_credit_limit,
                credit_card_avg_credit_utilization,
                credit_card_payment_to_min_ratio,
                credit_card_recent_dpd_month_rate,
                bureau_balance_recent_dpd_rate_delta,
                bureau_balance_recent_status_delta,
                pos_cash_recent_dpd_rate_delta,
                pos_cash_remaining_installment_ratio_delta,
                credit_card_recent_utilization_delta,
                credit_card_recent_balance_ratio_delta,
                credit_card_recent_drawings_ratio_delta,
                credit_card_recent_dpd_rate_delta,
                installments_last_3_late_payment_rate,
                installments_last_3_underpayment_rate,
                installments_last_3_avg_payment_delay_days,
                installments_last_3_payment_amount_ratio,
                installments_last_payment_delay_days,
                installments_last_payment_ratio,
                pos_cash_last_3_dpd_rate,
                pos_cash_last_3_dpd_def_rate,
                pos_cash_last_3_future_installment_ratio,
                pos_cash_last_3_dpd_rate_delta,
                pos_cash_last_loan_dpd_rate,
                credit_card_last_3_credit_utilization,
                credit_card_last_3_payment_to_min_ratio,
                credit_card_last_3_drawing_count,
                credit_card_last_3_dpd_rate,
                credit_card_last_3_utilization_delta,
                external_score_credit_pressure,
                external_score_annuity_pressure,
                bureau_debt_to_income_ratio,
                payment_shortfall_ratio,
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
                5,
                2,
                2,
                1,
                2,
                0.4,
                1 / 3,
                4,
                2,
                2,
                2,
                1,
                1,
                3,
                0.25,
                1 / 3,
                4,
                2,
                2,
                2,
                1,
                1,
                5,
                200,
                500,
                1500,
                0.175,
                0.875,
                1 / 3,
                -1 / 15,
                -0.5,
                1 / 12,
                19 / 144,
                7 / 120,
                1 / 3,
                1 / 3,
                1 / 12,
                1 / 2,
                1 / 2,
                1,
                0.95,
                0,
                1,
                1 / 3,
                1 / 3,
                19 / 48,
                1 / 12,
                1 / 2,
                7 / 30,
                0.875,
                4 / 3,
                1 / 3,
                7 / 120,
                1.7,
                0.085,
                0.0025,
                0.05,
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
            SELECT
                credit_to_income_ratio,
                annuity_to_income_ratio,
                goods_price_to_income_ratio,
                bureau_debt_to_income_ratio
            FROM mart_credit_risk_features
            WHERE SK_ID_CURR = 100002
            """
        ).fetchone()
        assert zero_income_ratios == (None, None, None, None)

        assert not {
            "total_credit_exposure_to_income_ratio",
            "monthly_delinquency_pressure",
            "revolving_utilization_delinquency_pressure",
            "prior_refusal_delay_pressure",
        }.intersection(mart_columns)
        assert not {
            "total_credit_exposure_to_income_ratio",
            "monthly_delinquency_pressure",
            "revolving_utilization_delinquency_pressure",
            "prior_refusal_delay_pressure",
        }.intersection(pressure_columns)

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

        duplicate_count = query_value(
            connection,
            """
            SELECT COUNT(*)
            FROM (
                SELECT SK_ID_CURR, source_population, COUNT(*) AS row_count
                FROM mart_credit_risk_features
                GROUP BY SK_ID_CURR, source_population
                HAVING COUNT(*) > 1
            )
            """,
        )
        assert duplicate_count == 0

    saved_profile = read_csv_rows(
        staged_feature_fixture.scratch_path / "reports" / "feature_mart_profile.csv",
        FEATURE_PROFILE_COLUMNS,
    )
    mart_profile = next(
        row for row in saved_profile if row["table_name"] == "mart_credit_risk_features"
    )
    assert mart_profile["row_count"] == "3"
    assert mart_profile["distinct_applicant_count"] == "3"
    assert mart_profile["duplicate_key_count"] == "0"


def test_feature_build_supports_v1_scope_without_post_v1_staging_tables(
    staged_feature_fixture,
) -> None:
    config = yaml.safe_load(
        staged_feature_fixture.config_path.read_text(encoding="utf-8")
    )
    config["project"]["data_scope_version"] = "v1"
    config["source_files"] = V1_SOURCE_FILES
    v1_config_path = staged_feature_fixture.scratch_path / "v1.yaml"
    v1_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        for table_name in [
            "stg_bureau_balance",
            "stg_pos_cash_balance",
            "stg_credit_card_balance",
        ]:
            connection.execute(f'DROP TABLE "{table_name}"')

    profile_rows = run_feature_build(v1_config_path)

    assert {row["table_name"] for row in profile_rows} == V1_PROFILE_TABLES

    with duckdb.connect(
        str(staged_feature_fixture.database_path), read_only=True
    ) as connection:
        tables = table_names(connection)
        mart_columns = read_table_columns(connection, "mart_credit_risk_features")

    assert not POST_V1_TABLES.intersection(tables)
    assert {
        "bureau_credit_count",
        "previous_application_count",
        "payment_amount_ratio",
    }.issubset(mart_columns)
    assert not {
        "bureau_balance_dpd_1plus_rate",
        "pos_cash_dpd_month_rate",
        "credit_card_avg_credit_utilization",
        "external_score_credit_pressure",
        "credit_card_last_3_credit_utilization",
    }.intersection(mart_columns)
