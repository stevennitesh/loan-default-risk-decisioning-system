-- Recent deterioration features compare recent behavior against lifetime
-- behavior. Positive deltas generally mean the recent window looks riskier.
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
    INNER JOIN n_eligible_bureau AS bureau
        ON balance.SK_ID_BUREAU = bureau.SK_ID_BUREAU
    WHERE balance.MONTHS_BALANCE <= -1
    GROUP BY bureau.SK_ID_CURR
),
pos_cash_pairs AS (
    SELECT SK_ID_CURR, MONTHS_BALANCE,
        CASE WHEN CNT_INSTALMENT > 0 AND CNT_INSTALMENT_FUTURE >= 0
            THEN CNT_INSTALMENT END AS installment_count,
        CASE WHEN CNT_INSTALMENT > 0 AND CNT_INSTALMENT_FUTURE >= 0
            THEN CNT_INSTALMENT_FUTURE END AS future_count
    FROM stg_pos_cash_balance
    WHERE MONTHS_BALANCE <= -1
),
pos_cash_installment_delta AS (
    -- Compare recent remaining-installment burden with the applicant's full POS history.
    SELECT
        SK_ID_CURR,
        (
            SUM(
                CASE
                    WHEN MONTHS_BALANCE >= -12 THEN future_count
                    ELSE 0
                END
            ) / NULLIF(
                SUM(
                    CASE
                        WHEN MONTHS_BALANCE >= -12 THEN installment_count
                        ELSE 0
                    END
                ),
                0
            )
        ) - (
            SUM(future_count) / NULLIF(SUM(installment_count), 0)
        ) AS pos_cash_remaining_installment_ratio_delta
    FROM pos_cash_pairs
    GROUP BY SK_ID_CURR
),
credit_card_months AS (
    -- Normalize utilization once so recent and lifetime windows use the same definition.
    SELECT
        SK_ID_CURR,
        MONTHS_BALANCE,
        AMT_BALANCE,
        AMT_DRAWINGS_CURRENT,
        CASE WHEN AMT_BALANCE>=0 AND AMT_CREDIT_LIMIT_ACTUAL>0 THEN AMT_CREDIT_LIMIT_ACTUAL END AS eligible_limit,
        CASE WHEN AMT_BALANCE>=0 AND AMT_CREDIT_LIMIT_ACTUAL>0 THEN AMT_BALANCE END AS eligible_balance,
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS credit_utilization
    FROM stg_credit_card_balance
    WHERE MONTHS_BALANCE <= -1
),
credit_card_delta AS (
    SELECT
        SK_ID_CURR,
        SUM(eligible_balance) FILTER (WHERE MONTHS_BALANCE>=-12)
            / NULLIF(SUM(eligible_limit) FILTER (WHERE MONTHS_BALANCE>=-12), 0)
            - SUM(eligible_balance)/NULLIF(SUM(eligible_limit), 0)
            AS credit_card_recent_utilization_delta,
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
-- Start from f_applicant_static so train and scoring populations both receive
-- one row even when a downstream history table has no records.
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
