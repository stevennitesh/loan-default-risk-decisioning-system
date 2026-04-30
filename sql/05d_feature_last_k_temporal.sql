CREATE OR REPLACE TABLE f_last_k_temporal_features AS
WITH installment_ranked AS (
    SELECT
        SK_ID_CURR,
        DAYS_INSTALMENT,
        AMT_INSTALMENT,
        AMT_PAYMENT,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY DAYS_INSTALMENT DESC, SK_ID_PREV DESC, NUM_INSTALMENT_NUMBER DESC
        ) AS payment_recency_rank,
        GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0) AS payment_delay_days,
        AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0) AS payment_ratio,
        CASE
            WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1
            ELSE 0
        END AS paid_late,
        CASE
            WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1
            ELSE 0
        END AS underpaid
    FROM stg_installments_payments
),
installment_last_3 AS (
    SELECT
        SK_ID_CURR,
        AVG(paid_late) AS installments_last_3_late_payment_rate,
        AVG(underpaid) AS installments_last_3_underpayment_rate,
        AVG(payment_delay_days) AS installments_last_3_avg_payment_delay_days,
        SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT), 0)
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
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        MONTHS_BALANCE,
        CNT_INSTALMENT,
        CNT_INSTALMENT_FUTURE,
        SK_DPD,
        SK_DPD_DEF,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY MONTHS_BALANCE DESC, SK_ID_PREV DESC
        ) AS pos_month_recency_rank
    FROM stg_pos_cash_balance
),
pos_cash_last_3 AS (
    SELECT
        SK_ID_CURR,
        AVG(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)
            AS pos_cash_last_3_dpd_rate,
        AVG(CASE WHEN SK_DPD_DEF > 0 THEN 1 ELSE 0 END)
            AS pos_cash_last_3_dpd_def_rate,
        SUM(CNT_INSTALMENT_FUTURE) / NULLIF(SUM(CNT_INSTALMENT), 0)
            AS pos_cash_last_3_future_installment_ratio
    FROM pos_cash_ranked
    WHERE pos_month_recency_rank <= 3
    GROUP BY SK_ID_CURR
),
pos_cash_latest_loan AS (
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
        GROUP BY SK_ID_CURR, SK_ID_PREV
    )
    WHERE loan_recency_rank = 1
),
pos_cash_last_loan AS (
    SELECT
        pos_cash.SK_ID_CURR,
        AVG(CASE WHEN pos_cash.SK_DPD > 0 THEN 1 ELSE 0 END)
            AS pos_cash_last_loan_dpd_rate
    FROM stg_pos_cash_balance AS pos_cash
    INNER JOIN pos_cash_latest_loan AS latest_loan
        ON pos_cash.SK_ID_CURR = latest_loan.SK_ID_CURR
        AND pos_cash.SK_ID_PREV = latest_loan.SK_ID_PREV
    GROUP BY pos_cash.SK_ID_CURR
),
credit_card_ranked AS (
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        AMT_BALANCE,
        AMT_CREDIT_LIMIT_ACTUAL,
        AMT_DRAWINGS_CURRENT,
        AMT_INST_MIN_REGULARITY,
        AMT_PAYMENT_CURRENT,
        CNT_DRAWINGS_CURRENT,
        SK_DPD,
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS credit_utilization,
        ROW_NUMBER() OVER (
            PARTITION BY SK_ID_CURR
            ORDER BY MONTHS_BALANCE DESC, SK_ID_PREV DESC
        ) AS card_month_recency_rank
    FROM stg_credit_card_balance
),
credit_card_last_3 AS (
    SELECT
        SK_ID_CURR,
        AVG(credit_utilization) AS credit_card_last_3_credit_utilization,
        SUM(AMT_PAYMENT_CURRENT) / NULLIF(SUM(AMT_INST_MIN_REGULARITY), 0)
            AS credit_card_last_3_payment_to_min_ratio,
        AVG(CNT_DRAWINGS_CURRENT) AS credit_card_last_3_drawing_count,
        AVG(CASE WHEN SK_DPD > 0 THEN 1 ELSE 0 END)
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
