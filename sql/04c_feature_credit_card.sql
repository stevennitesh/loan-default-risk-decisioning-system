-- Credit-card balance is monthly per previous credit-card account. This table
-- summarizes utilization, payment, drawing, and delinquency signals per applicant.
CREATE OR REPLACE TABLE f_credit_card_agg AS
WITH credit_card AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        AMT_BALANCE,
        AMT_CREDIT_LIMIT_ACTUAL,
        AMT_DRAWINGS_CURRENT,
        AMT_INST_MIN_REGULARITY,
        AMT_PAYMENT_CURRENT,
        AMT_PAYMENT_TOTAL_CURRENT,
        AMT_TOTAL_RECEIVABLE,
        CNT_DRAWINGS_CURRENT,
        LOWER(NAME_CONTRACT_STATUS) AS contract_status,
        SK_DPD,
        SK_DPD_DEF,
        -- Use NULLIF so zero-limit records do not create infinite utilization.
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS credit_utilization
    FROM stg_credit_card_balance
)
SELECT
    SK_ID_CURR,
    COUNT(*) AS credit_card_month_count,
    COUNT(DISTINCT SK_ID_PREV) AS credit_card_contract_count,
    SUM(CASE WHEN contract_status = 'active' THEN 1 ELSE 0 END) AS credit_card_active_month_count,
    SUM(CASE WHEN contract_status = 'completed' THEN 1 ELSE 0 END) AS credit_card_completed_month_count,
    SUM(
        CASE
            WHEN contract_status IN ('demand', 'signed', 'sent proposal', 'refused') THEN 1
            ELSE 0
        END
    ) AS credit_card_problem_status_month_count,
    SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) AS credit_card_dpd_month_count,
    SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END) AS credit_card_dpd_def_month_count,
    MAX(SK_DPD) AS credit_card_max_dpd,
    MAX(SK_DPD_DEF) AS credit_card_max_dpd_def,
    AVG(SK_DPD) AS credit_card_avg_dpd,
    AVG(SK_DPD_DEF) AS credit_card_avg_dpd_def,
    AVG(AMT_BALANCE) AS credit_card_avg_balance,
    MAX(AMT_BALANCE) AS credit_card_max_balance,
    AVG(AMT_CREDIT_LIMIT_ACTUAL) AS credit_card_avg_credit_limit,
    MAX(AMT_CREDIT_LIMIT_ACTUAL) AS credit_card_max_credit_limit,
    SUM(AMT_BALANCE) / NULLIF(SUM(AMT_CREDIT_LIMIT_ACTUAL), 0)
        AS credit_card_balance_to_limit_ratio,
    AVG(credit_utilization) AS credit_card_avg_credit_utilization,
    AVG(AMT_DRAWINGS_CURRENT) AS credit_card_avg_drawings_current,
    SUM(AMT_DRAWINGS_CURRENT) AS credit_card_total_drawings_current,
    AVG(CNT_DRAWINGS_CURRENT) AS credit_card_avg_drawing_count,
    SUM(CNT_DRAWINGS_CURRENT) AS credit_card_total_drawing_count,
    AVG(AMT_TOTAL_RECEIVABLE) AS credit_card_avg_total_receivable,
    SUM(AMT_TOTAL_RECEIVABLE) AS credit_card_total_receivable,
    SUM(AMT_PAYMENT_CURRENT) / NULLIF(SUM(AMT_INST_MIN_REGULARITY), 0)
        AS credit_card_payment_to_min_ratio,
    SUM(AMT_PAYMENT_TOTAL_CURRENT) / NULLIF(SUM(AMT_INST_MIN_REGULARITY), 0)
        AS credit_card_total_payment_to_min_ratio,
    MAX(MONTHS_BALANCE) AS credit_card_latest_month,
    MIN(MONTHS_BALANCE) AS credit_card_earliest_month,
    SUM(CASE WHEN MONTHS_BALANCE >= -12 THEN 1 ELSE 0 END) AS credit_card_recent_month_count,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND SK_DPD > 0 THEN 1
            ELSE 0
        END
    ) AS credit_card_recent_dpd_month_count,
    SUM(CASE WHEN contract_status = 'completed' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS credit_card_completed_month_rate,
    SUM(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS credit_card_dpd_month_rate,
    SUM(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS credit_card_dpd_def_month_rate,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND SK_DPD > 0 THEN 1
            ELSE 0
        END
    ) / NULLIF(SUM(CASE WHEN MONTHS_BALANCE >= -12 THEN 1 ELSE 0 END), 0)
        AS credit_card_recent_dpd_month_rate
FROM credit_card
GROUP BY SK_ID_CURR;
