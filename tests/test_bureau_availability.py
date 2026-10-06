from pathlib import Path

import duckdb
import pytest


def test_bureau_children_cannot_restore_unproven_origins_and_future_maturity_stays():
    root = Path(__file__).resolve().parents[1]
    with duckdb.connect() as con:
        con.execute("""CREATE TABLE stg_bureau AS SELECT 1 AS SK_ID_CURR, 10 AS SK_ID_BUREAU,
            -30.0 AS DAYS_CREDIT, 'Active' AS CREDIT_ACTIVE, 0 AS CREDIT_DAY_OVERDUE,
            100.0::DOUBLE AS AMT_CREDIT_SUM, 40.0::DOUBLE AS AMT_CREDIT_SUM_DEBT, 100.0::DOUBLE AS AMT_CREDIT_SUM_LIMIT,
            0.0::DOUBLE AS AMT_CREDIT_SUM_OVERDUE, 365.0 AS DAYS_CREDIT_ENDDATE, NULL::DOUBLE AS DAYS_ENDDATE_FACT""")
        for bureau, origin in ((11, 0.0), (12, 1.0), (13, None)):
            con.execute(
                "INSERT INTO stg_bureau SELECT 1, ?, ?, 'Active', 100, 999, 999, 999,999,365,NULL",
                [bureau, origin],
            )
        con.execute(
            "CREATE TABLE stg_bureau_balance AS SELECT 10 AS SK_ID_BUREAU,-1 AS MONTHS_BALANCE,'0' AS STATUS"
        )
        con.execute(
            "INSERT INTO stg_bureau_balance VALUES (11,-1,'5'),(12,-1,'5'),(13,-1,'5')"
        )
        for name in ("03_feature_bureau.sql", "03b_feature_bureau_balance.sql"):
            con.execute((root / "sql" / name).read_text())
        assert con.execute("SELECT COUNT(*) FROM n_eligible_bureau").fetchone()[0] == 1
        assert con.execute(
            "SELECT bureau_credit_count,total_credit_sum,avg_days_credit_enddate FROM f_bureau_agg"
        ).fetchone() == pytest.approx((1, 100, 365))
        assert con.execute(
            "SELECT bureau_balance_month_count,bureau_balance_dpd_1plus_rate FROM f_bureau_balance_agg"
        ).fetchone() == pytest.approx((1, 0))
        assert con.execute(
            "SELECT eligible_loan_count,day_zero_loan_count,positive_origin_loan_count,unknown_origin_loan_count FROM bureau_origin_coverage"
        ).fetchone() == (1, 1, 1, 1)
        # Recency must join the exact normalized owner too; production SQL tests
        # exercise its complete feature dependencies.
        assert (
            "INNER JOIN n_eligible_bureau AS bureau"
            in (root / "sql/05c_feature_recency_deterioration.sql").read_text()
        )
