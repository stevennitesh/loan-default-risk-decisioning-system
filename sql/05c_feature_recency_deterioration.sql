CREATE OR REPLACE TABLE f_recency_deterioration_features AS
WITH bureau_status_delta AS (
    SELECT
        bureau.SK_ID_CURR,
        AVG(
            CASE
                WHEN balance.MONTHS_BALANCE >= -12
                THEN CASE
                    WHEN UPPER(CAST(balance.STATUS AS VARCHAR)) IN ('0', '1', '2', '3', '4', '5')
                    THEN CAST(balance.STATUS AS INTEGER)
                END
            END
        ) - AVG(
            CASE
                WHEN UPPER(CAST(balance.STATUS AS VARCHAR)) IN ('0', '1', '2', '3', '4', '5')
                THEN CAST(balance.STATUS AS INTEGER)
            END
        ) AS bureau_balance_recent_status_delta
    FROM stg_bureau_balance AS balance
    INNER JOIN stg_bureau AS bureau
        ON balance.SK_ID_BUREAU = bureau.SK_ID_BUREAU
    GROUP BY bureau.SK_ID_CURR
),
pos_cash_installment_delta AS (
    SELECT
        SK_ID_CURR,
        (
            SUM(
                CASE
                    WHEN MONTHS_BALANCE >= -12 THEN CNT_INSTALMENT_FUTURE
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN MONTHS_BALANCE >= -12 THEN CNT_INSTALMENT
                        ELSE 0
                    END
                ),
                0
            )
        ) - (
            SUM(CNT_INSTALMENT_FUTURE) / NULLIF(SUM(CNT_INSTALMENT), 0)
        ) AS pos_cash_remaining_installment_ratio_delta
    FROM stg_pos_cash_balance
    GROUP BY SK_ID_CURR
),
credit_card_months AS (
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        AMT_BALANCE,
        AMT_DRAWINGS_CURRENT,
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS credit_utilization
    FROM stg_credit_card_balance
),
credit_card_delta AS (
    SELECT
        SK_ID_CURR,
        AVG(CASE WHEN MONTHS_BALANCE >= -12 THEN credit_utilization END)
            - AVG(credit_utilization) AS credit_card_recent_utilization_delta,
        AVG(CASE WHEN MONTHS_BALANCE >= -12 THEN AMT_BALANCE END)
            / NULLIF(AVG(AMT_BALANCE), 0)
            - 1 AS credit_card_recent_balance_ratio_delta,
        AVG(CASE WHEN MONTHS_BALANCE >= -12 THEN AMT_DRAWINGS_CURRENT END)
            / NULLIF(AVG(AMT_DRAWINGS_CURRENT), 0)
            - 1 AS credit_card_recent_drawings_ratio_delta
    FROM credit_card_months
    GROUP BY SK_ID_CURR
)
SELECT
    applicant.SK_ID_CURR,
    applicant.source_population,
    bureau_balance.bureau_balance_recent_dpd_1plus_rate
        - bureau_balance.bureau_balance_dpd_1plus_rate
        AS bureau_balance_recent_dpd_rate_delta,
    bureau_status_delta.bureau_balance_recent_status_delta,
    pos_cash.pos_cash_recent_dpd_month_rate
        - pos_cash.pos_cash_dpd_month_rate
        AS pos_cash_recent_dpd_rate_delta,
    pos_cash_installment_delta.pos_cash_remaining_installment_ratio_delta,
    credit_card_delta.credit_card_recent_utilization_delta,
    credit_card_delta.credit_card_recent_balance_ratio_delta,
    credit_card_delta.credit_card_recent_drawings_ratio_delta,
    credit_card.credit_card_recent_dpd_month_rate
        - credit_card.credit_card_dpd_month_rate
        AS credit_card_recent_dpd_rate_delta
FROM f_applicant_static AS applicant
LEFT JOIN f_bureau_balance_agg AS bureau_balance
    ON applicant.SK_ID_CURR = bureau_balance.SK_ID_CURR
LEFT JOIN bureau_status_delta
    ON applicant.SK_ID_CURR = bureau_status_delta.SK_ID_CURR
LEFT JOIN f_pos_cash_agg AS pos_cash
    ON applicant.SK_ID_CURR = pos_cash.SK_ID_CURR
LEFT JOIN pos_cash_installment_delta
    ON applicant.SK_ID_CURR = pos_cash_installment_delta.SK_ID_CURR
LEFT JOIN f_credit_card_agg AS credit_card
    ON applicant.SK_ID_CURR = credit_card.SK_ID_CURR
LEFT JOIN credit_card_delta
    ON applicant.SK_ID_CURR = credit_card_delta.SK_ID_CURR;
