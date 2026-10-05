-- Last-k temporal features isolate the most recent repayment and account
-- behavior without changing the final applicant population grain.
CREATE OR REPLACE TABLE f_last_k_temporal_features AS
WITH installment_ranked AS (
    -- DAYS_INSTALMENT is negative relative to application date; DESC selects the
    -- least-negative, most recent installment first.
    SELECT
        SK_ID_CURR,
        scheduled_amount AS AMT_INSTALMENT,
        paid_amount AS AMT_PAYMENT,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY due_day DESC, SK_ID_PREV DESC,
                NUM_INSTALMENT_NUMBER DESC, NUM_INSTALMENT_VERSION DESC
        ) AS payment_recency_rank,
        CASE WHEN completed_delay_days IS NOT NULL
            THEN GREATEST(completed_delay_days, 0) END AS payment_delay_days,
        paid_amount / NULLIF(scheduled_amount, 0) AS payment_ratio,
        late AS paid_late,
        underpaid
    FROM n_installment_obligations
    WHERE obligation_known
),
installment_last_3 AS (
    -- Last three due obligations capture near-term repayment behavior.
    SELECT
        SK_ID_CURR,
        AVG(paid_late) AS installments_last_3_late_payment_rate,
        AVG(underpaid) AS installments_last_3_underpayment_rate,
        AVG(payment_delay_days) AS installments_last_3_avg_payment_delay_days,
        SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT) FILTER (WHERE AMT_PAYMENT IS NOT NULL), 0)
            AS installments_last_3_payment_amount_ratio
    FROM installment_ranked
    WHERE payment_recency_rank <= 3
    GROUP BY SK_ID_CURR
),
installment_last_payment AS (
    SELECT
        SK_ID_CURR,
        payment_delay_days AS installments_last_payment_delay_days,
        payment_ratio AS installments_last_payment_ratio
    FROM installment_ranked
    WHERE payment_recency_rank = 1
),
pos_cash_ranked AS (
    -- MONTHS_BALANCE is negative relative to application date; DESC selects the
    -- most recent POS cash month first.
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        SUM(CNT_INSTALMENT) FILTER (
            WHERE CNT_INSTALMENT > 0 AND CNT_INSTALMENT_FUTURE >= 0
        ) AS CNT_INSTALMENT,
        SUM(CNT_INSTALMENT_FUTURE) FILTER (
            WHERE CNT_INSTALMENT > 0 AND CNT_INSTALMENT_FUTURE >= 0
        ) AS CNT_INSTALMENT_FUTURE,
        CASE WHEN MAX(SK_DPD) > 0 THEN 1
            WHEN COUNT(SK_DPD) = COUNT(*) AND MIN(SK_DPD) >= 0 THEN 0 END AS SK_DPD,
        CASE WHEN MAX(SK_DPD_DEF) > 0 THEN 1
            WHEN COUNT(SK_DPD_DEF) = COUNT(*) AND MIN(SK_DPD_DEF) >= 0 THEN 0 END AS SK_DPD_DEF,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY MONTHS_BALANCE DESC
        ) AS pos_month_recency_rank
    FROM stg_pos_cash_balance
    WHERE MONTHS_BALANCE <= -1
    GROUP BY SK_ID_CURR, MONTHS_BALANCE
),
pos_cash_last_3 AS (
    SELECT
        SK_ID_CURR,
        AVG(CASE WHEN SK_DPD IS NOT NULL THEN CAST(SK_DPD > 0 AS INTEGER) END)
            AS pos_cash_last_3_dpd_rate,
        AVG(CASE WHEN SK_DPD_DEF IS NOT NULL THEN CAST(SK_DPD_DEF > 0 AS INTEGER) END)
            AS pos_cash_last_3_dpd_def_rate,
        SUM(CNT_INSTALMENT_FUTURE) / NULLIF(SUM(CNT_INSTALMENT), 0)
            AS pos_cash_last_3_future_installment_ratio
    FROM pos_cash_ranked
    WHERE pos_month_recency_rank <= 3
    GROUP BY SK_ID_CURR
),
pos_cash_latest_loan AS (
    -- Identify the most recently observed POS cash loan before summarizing all
    -- monthly rows for that loan.
    SELECT
        SK_ID_CURR,
        SK_ID_PREV
    FROM (
        SELECT
            SK_ID_CURR,
            SK_ID_PREV,
            MAX(MONTHS_BALANCE) AS latest_month,
            ROW_NUMBER() OVER (
                PARTITION BY SK_ID_CURR
                ORDER BY MAX(MONTHS_BALANCE) DESC, SK_ID_PREV DESC
            ) AS loan_recency_rank
        FROM stg_pos_cash_balance
        WHERE MONTHS_BALANCE <= -1
        GROUP BY SK_ID_CURR, SK_ID_PREV
    )
    WHERE loan_recency_rank = 1
),
pos_cash_last_loan AS (
    SELECT
        pos_cash.SK_ID_CURR,
        AVG(CASE WHEN pos_cash.SK_DPD >= 0 THEN CAST(pos_cash.SK_DPD > 0 AS INTEGER) END)
            AS pos_cash_last_loan_dpd_rate
    FROM stg_pos_cash_balance AS pos_cash
    INNER JOIN pos_cash_latest_loan AS latest_loan
        ON pos_cash.SK_ID_CURR = latest_loan.SK_ID_CURR
        AND pos_cash.SK_ID_PREV = latest_loan.SK_ID_PREV
    WHERE pos_cash.MONTHS_BALANCE <= -1
    GROUP BY pos_cash.SK_ID_CURR
),
credit_card_ranked AS (
    -- Rank credit-card months per applicant so the last-3 window can be compared
    -- with lifetime credit-card aggregates.
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        SUM(AMT_BALANCE) FILTER (WHERE AMT_BALANCE >= 0 AND AMT_CREDIT_LIMIT_ACTUAL > 0) AS AMT_BALANCE,
        SUM(AMT_CREDIT_LIMIT_ACTUAL) FILTER (WHERE AMT_BALANCE >= 0 AND AMT_CREDIT_LIMIT_ACTUAL > 0) AS AMT_CREDIT_LIMIT_ACTUAL,
        SUM(AMT_INST_MIN_REGULARITY) FILTER (WHERE AMT_PAYMENT_CURRENT >= 0 AND AMT_INST_MIN_REGULARITY > 0) AS AMT_INST_MIN_REGULARITY,
        SUM(AMT_PAYMENT_CURRENT) FILTER (WHERE AMT_PAYMENT_CURRENT >= 0 AND AMT_INST_MIN_REGULARITY > 0) AS AMT_PAYMENT_CURRENT,
        CASE WHEN COUNT(CNT_DRAWINGS_CURRENT) = COUNT(*) AND MIN(CNT_DRAWINGS_CURRENT) >= 0
            THEN SUM(CNT_DRAWINGS_CURRENT) END AS CNT_DRAWINGS_CURRENT,
        CASE WHEN MAX(SK_DPD) > 0 THEN 1
            WHEN COUNT(SK_DPD) = COUNT(*) AND MIN(SK_DPD) >= 0 THEN 0 END AS SK_DPD,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY MONTHS_BALANCE DESC
        ) AS card_month_recency_rank
    FROM stg_credit_card_balance
    WHERE MONTHS_BALANCE <= -1
    GROUP BY SK_ID_CURR, MONTHS_BALANCE
),
credit_card_last_3 AS (
    SELECT
        SK_ID_CURR,
        SUM(AMT_BALANCE) / NULLIF(SUM(AMT_CREDIT_LIMIT_ACTUAL), 0) AS credit_card_last_3_credit_utilization,
        SUM(AMT_PAYMENT_CURRENT) / NULLIF(SUM(AMT_INST_MIN_REGULARITY), 0)
            AS credit_card_last_3_payment_to_min_ratio,
        AVG(CNT_DRAWINGS_CURRENT) AS credit_card_last_3_drawing_count,
        AVG(CASE WHEN SK_DPD IS NOT NULL THEN CAST(SK_DPD > 0 AS INTEGER) END)
            AS credit_card_last_3_dpd_rate
    FROM credit_card_ranked
    WHERE card_month_recency_rank <= 3
    GROUP BY SK_ID_CURR
)
SELECT
    applicant.SK_ID_CURR,
    applicant.source_population,
    installment_last_3.installments_last_3_late_payment_rate,
    installment_last_3.installments_last_3_underpayment_rate,
    installment_last_3.installments_last_3_avg_payment_delay_days,
    installment_last_3.installments_last_3_payment_amount_ratio,
    installment_last_payment.installments_last_payment_delay_days,
    installment_last_payment.installments_last_payment_ratio,
    pos_cash_last_3.pos_cash_last_3_dpd_rate,
    pos_cash_last_3.pos_cash_last_3_dpd_def_rate,
    pos_cash_last_3.pos_cash_last_3_future_installment_ratio,
    pos_cash_last_3.pos_cash_last_3_dpd_rate - pos_cash.pos_cash_dpd_month_rate
        AS pos_cash_last_3_dpd_rate_delta,
    pos_cash_last_loan.pos_cash_last_loan_dpd_rate,
    credit_card_last_3.credit_card_last_3_credit_utilization,
    credit_card_last_3.credit_card_last_3_payment_to_min_ratio,
    credit_card_last_3.credit_card_last_3_drawing_count,
    credit_card_last_3.credit_card_last_3_dpd_rate,
    credit_card_last_3.credit_card_last_3_credit_utilization
        - credit_card.credit_card_avg_credit_utilization
        AS credit_card_last_3_utilization_delta
FROM f_applicant_static AS applicant
-- Preserve the applicant/static population grain; missing histories remain NULL
-- feature values for downstream imputers instead of dropping applicants.
LEFT JOIN installment_last_3
    ON applicant.SK_ID_CURR = installment_last_3.SK_ID_CURR
LEFT JOIN installment_last_payment
    ON applicant.SK_ID_CURR = installment_last_payment.SK_ID_CURR
LEFT JOIN pos_cash_last_3
    ON applicant.SK_ID_CURR = pos_cash_last_3.SK_ID_CURR
LEFT JOIN pos_cash_last_loan
    ON applicant.SK_ID_CURR = pos_cash_last_loan.SK_ID_CURR
LEFT JOIN f_pos_cash_agg AS pos_cash
    ON applicant.SK_ID_CURR = pos_cash.SK_ID_CURR
LEFT JOIN credit_card_last_3
    ON applicant.SK_ID_CURR = credit_card_last_3.SK_ID_CURR
LEFT JOIN f_credit_card_agg AS credit_card
    ON applicant.SK_ID_CURR = credit_card.SK_ID_CURR;
