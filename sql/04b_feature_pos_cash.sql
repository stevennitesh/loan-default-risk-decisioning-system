-- POS cash balance is monthly per previous loan. This table summarizes status,
-- delinquency, and remaining-installment pressure at applicant grain.
CREATE OR REPLACE TABLE f_pos_cash_agg AS
WITH pos_cash AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        CNT_INSTALMENT,
        CNT_INSTALMENT_FUTURE,
        LOWER(NAME_CONTRACT_STATUS) AS contract_status,
        SK_DPD,
        SK_DPD_DEF
    FROM stg_pos_cash_balance
)
SELECT
    SK_ID_CURR,
    COUNT(*) AS pos_cash_month_count,
    COUNT(DISTINCT SK_ID_PREV) AS pos_cash_contract_count,
    SUM(CASE WHEN contract_status = 'active' THEN 1 ELSE 0 END) AS pos_cash_active_month_count,
    SUM(CASE WHEN contract_status = 'completed' THEN 1 ELSE 0 END) AS pos_cash_completed_month_count,
    SUM(
        CASE
            WHEN contract_status IN ('demand', 'returned to the store', 'amortized debt') THEN 1
            ELSE 0
        END
    ) AS pos_cash_problem_status_month_count,
    SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) AS pos_cash_dpd_month_count,
    SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END) AS pos_cash_dpd_def_month_count,
    MAX(SK_DPD) AS pos_cash_max_dpd,
    MAX(SK_DPD_DEF) AS pos_cash_max_dpd_def,
    AVG(SK_DPD) AS pos_cash_avg_dpd,
    AVG(SK_DPD_DEF) AS pos_cash_avg_dpd_def,
    AVG(CNT_INSTALMENT) AS pos_cash_avg_instalment_count,
    AVG(CNT_INSTALMENT_FUTURE) AS pos_cash_avg_future_installments,
    MAX(MONTHS_BALANCE) AS pos_cash_latest_month,
    MIN(MONTHS_BALANCE) AS pos_cash_earliest_month,
    SUM(CASE WHEN MONTHS_BALANCE >= -12 THEN 1 ELSE 0 END) AS pos_cash_recent_month_count,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND SK_DPD > 0 THEN 1
            ELSE 0
        END
    ) AS pos_cash_recent_dpd_month_count,
    SUM(CASE WHEN contract_status = 'completed' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS pos_cash_completed_month_rate,
    SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS pos_cash_dpd_month_rate,
    SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS pos_cash_dpd_def_month_rate,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND SK_DPD > 0 THEN 1
            ELSE 0
        END
    ) / NULLIF(SUM(CASE WHEN MONTHS_BALANCE >= -12 THEN 1 ELSE 0 END), 0)
        AS pos_cash_recent_dpd_month_rate
FROM pos_cash
GROUP BY SK_ID_CURR;
