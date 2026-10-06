import duckdb
import pytest

from src.build_features import FeatureBuildError, run_feature_build
from src.data_contracts import DataContractError


def test_identical_payment_records_remain_cashflows_without_unique_event_ids(
    staged_feature_fixture,
):
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            "DELETE FROM stg_installments_payments WHERE SK_ID_CURR=100001"
        )
        connection.execute(
            "INSERT INTO stg_installments_payments VALUES (10,100001,1,1,-10,-12,100,50),(10,100001,1,1,-10,-12,100,50)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT installment_payment_count,installment_obligation_count,total_instalment_amount,total_payment_amount,underpayment_count,late_payment_count FROM f_installments_agg WHERE SK_ID_CURR=100001"
        ).fetchone()
    assert row == pytest.approx((2, 1, 100, 100, 0, 0))


def test_split_payments_use_one_obligation_and_ignore_future_cashflows(
    staged_feature_fixture,
) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            "DELETE FROM stg_installments_payments WHERE SK_ID_CURR=100001"
        )
        connection.execute(
            "INSERT INTO stg_installments_payments VALUES "
            "(10,100001,1,1,-10,-12,100,40),"
            "(10,100001,1,1,-10,-9,100,60),"
            "(10,100001,1,1,-10,2,100,1000)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT total_instalment_amount,total_payment_amount,payment_amount_ratio,"
            "underpayment_count,late_payment_count,avg_payment_delay_days,"
            "installments_last_3_payment_amount_ratio,payment_shortfall_ratio "
            "FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
        ).fetchone()
    assert row == pytest.approx((100, 100, 1, 0, 1, 1, 1, 0))


def test_unknown_payments_and_competing_versions_are_not_treated_as_paid(
    staged_feature_fixture,
) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            "DELETE FROM stg_installments_payments WHERE SK_ID_CURR=100001"
        )
        connection.execute(
            "INSERT INTO stg_installments_payments VALUES "
            "(10,100001,1,1,-10,-12,100,100),"
            "(10,100001,2,1,-5,NULL,200,NULL),"
            "(10,100001,3,1,-4,-6,100,100),"
            "(10,100001,3,2,-3,-5,200,200)"
        )
        connection.execute(
            "INSERT INTO stg_installments_payments VALUES (12,100002,2,1,-2,-2,50,50)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT total_instalment_amount,payment_amount_ratio,"
            "installment_ambiguous_obligation_count,installment_unknown_payment_obligation_count "
            "FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
        ).fetchone()
    assert row == pytest.approx((100, 1, 2, 1))


def test_last_three_pos_months_include_every_account_in_each_month(
    staged_feature_fixture,
) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute("DELETE FROM stg_pos_cash_balance WHERE SK_ID_CURR=100001")
        connection.execute(
            "INSERT INTO stg_pos_cash_balance VALUES "
            "(10,100001,-1,12,6,'Active',0,0),"
            "(11,100001,-1,12,6,'Active',0,0),"
            "(12,100001,-1,12,6,'Active',0,0),"
            "(10,100001,-2,12,6,'Active',4,0),"
            "(10,100001,-3,12,6,'Active',4,0),"
            "(10,100001,0,12,6,'Active',9,9)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        value = connection.execute(
            "SELECT pos_cash_last_3_dpd_rate FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
        ).fetchone()[0]
    assert value == pytest.approx(2 / 3)


def test_failed_feature_rebuild_preserves_the_previous_mart_and_build_identity(
    staged_feature_fixture,
) -> None:
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        previous_id = connection.execute(
            "SELECT feature_build_id FROM feature_build_metadata"
        ).fetchone()[0]
        previous_amount = connection.execute(
            "SELECT payment_amount_ratio FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
        ).fetchone()[0]
        connection.execute("UPDATE stg_installments_payments SET AMT_PAYMENT=NULL")
    with pytest.raises(DataContractError, match="100% missing"):
        run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        assert (
            connection.execute(
                "SELECT feature_build_id FROM feature_build_metadata"
            ).fetchone()[0]
            == previous_id
        )
        assert (
            connection.execute(
                "SELECT payment_amount_ratio FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
            ).fetchone()[0]
            == previous_amount
        )


def test_pos_burden_ratios_use_matched_known_amounts(staged_feature_fixture) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute("DELETE FROM stg_pos_cash_balance WHERE SK_ID_CURR=100001")
        connection.execute(
            "INSERT INTO stg_pos_cash_balance VALUES "
            "(10,100001,-1,12,6,'Active',0,0),"
            "(10,100001,-2,12,NULL,'Active',0,0),"
            "(10,100001,-13,12,0,'Completed',0,0)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT pos_cash_remaining_installment_ratio_delta,"
            "pos_cash_last_3_future_installment_ratio "
            "FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
        ).fetchone()
    assert row == pytest.approx((0.25, 0.25))


def test_duplicate_account_months_are_rejected(staged_feature_fixture) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            "INSERT INTO stg_pos_cash_balance SELECT * FROM stg_pos_cash_balance "
            "WHERE SK_ID_CURR=100001 AND MONTHS_BALANCE=-1"
        )
    with pytest.raises(FeatureBuildError, match="Duplicate"):
        run_feature_build(staged_feature_fixture.config_path)


def test_unknown_monthly_delinquency_is_not_counted_as_on_time(
    staged_feature_fixture,
) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute("DELETE FROM stg_pos_cash_balance WHERE SK_ID_CURR=100001")
        connection.execute(
            "INSERT INTO stg_pos_cash_balance VALUES "
            "(10,100001,-1,12,6,'Active',0,0),"
            "(11,100001,-1,12,6,'Active',NULL,NULL),"
            "(10,100001,-2,12,6,'Active',4,0),"
            "(10,100001,-3,12,6,'Active',0,0)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT pos_cash_dpd_month_rate,pos_cash_last_3_dpd_rate "
            "FROM mart_credit_risk_features WHERE SK_ID_CURR=100001"
        ).fetchone()
    assert row == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize("future_version", [1, 2])
def test_future_schedule_rows_cannot_conceal_ambiguous_obligations(
    staged_feature_fixture, future_version: int
) -> None:
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        connection.execute(
            "DELETE FROM stg_installments_payments WHERE SK_ID_CURR=100001"
        )
        connection.execute(
            "INSERT INTO stg_installments_payments VALUES "
            "(10,100001,1,1,-10,-9,100,100),"
            "(10,100001,2,1,-5,-6,100,100),"
            f"(10,100001,2,{future_version},2,-1,200,100),"
            "(10,100001,3,1,2,-1,100,100)"
        )
    run_feature_build(staged_feature_fixture.config_path)
    with duckdb.connect(str(staged_feature_fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT installment_obligation_count,installment_ambiguous_obligation_count,"
            "total_instalment_amount,payment_amount_ratio FROM mart_credit_risk_features "
            "WHERE SK_ID_CURR=100001"
        ).fetchone()
    assert row == pytest.approx((2, 1, 100, 1))
