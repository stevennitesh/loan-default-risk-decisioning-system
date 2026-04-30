CREATE OR REPLACE TABLE f_risk_pressure_features AS
SELECT
    applicant.SK_ID_CURR,
    applicant.source_population,
    CASE
        WHEN applicant.ext_source_mean IS NOT NULL
             AND applicant.credit_to_income_ratio IS NOT NULL
        THEN (1 - applicant.ext_source_mean) * applicant.credit_to_income_ratio
    END AS external_score_credit_pressure,
    CASE
        WHEN applicant.ext_source_mean IS NOT NULL
             AND applicant.annuity_to_income_ratio IS NOT NULL
        THEN (1 - applicant.ext_source_mean) * applicant.annuity_to_income_ratio
    END AS external_score_annuity_pressure,
    CASE
        WHEN applicant.AMT_INCOME_TOTAL IS NOT NULL
             AND applicant.AMT_INCOME_TOTAL <> 0
        THEN (applicant.AMT_CREDIT + COALESCE(bureau.total_credit_debt, 0)) / applicant.AMT_INCOME_TOTAL
    END AS total_credit_exposure_to_income_ratio,
    CASE
        WHEN applicant.AMT_INCOME_TOTAL IS NOT NULL
             AND applicant.AMT_INCOME_TOTAL <> 0
             AND bureau.total_credit_debt IS NOT NULL
        THEN bureau.total_credit_debt / applicant.AMT_INCOME_TOTAL
    END AS bureau_debt_to_income_ratio,
    CASE
        WHEN (
            CASE WHEN bureau_balance.bureau_balance_recent_dpd_1plus_rate IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN pos_cash.pos_cash_recent_dpd_month_rate IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN credit_card.credit_card_recent_dpd_month_rate IS NOT NULL THEN 1 ELSE 0 END
        ) > 0
        THEN (
            COALESCE(bureau_balance.bureau_balance_recent_dpd_1plus_rate, 0)
          + COALESCE(pos_cash.pos_cash_recent_dpd_month_rate, 0)
          + COALESCE(credit_card.credit_card_recent_dpd_month_rate, 0)
        ) / (
            CASE WHEN bureau_balance.bureau_balance_recent_dpd_1plus_rate IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN pos_cash.pos_cash_recent_dpd_month_rate IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN credit_card.credit_card_recent_dpd_month_rate IS NOT NULL THEN 1 ELSE 0 END
        )
    END AS monthly_delinquency_pressure,
    CASE
        WHEN credit_card.credit_card_avg_credit_utilization IS NOT NULL
             AND credit_card.credit_card_recent_dpd_month_rate IS NOT NULL
        THEN credit_card.credit_card_avg_credit_utilization * credit_card.credit_card_recent_dpd_month_rate
    END AS revolving_utilization_delinquency_pressure,
    CASE
        WHEN previous.refusal_rate IS NOT NULL
             AND installments.max_payment_delay_days IS NOT NULL
        THEN previous.refusal_rate * GREATEST(installments.max_payment_delay_days, 0)
    END AS prior_refusal_delay_pressure,
    CASE
        WHEN installments.payment_amount_ratio IS NOT NULL
        THEN GREATEST(1 - installments.payment_amount_ratio, 0)
    END AS payment_shortfall_ratio
FROM f_applicant_static AS applicant
LEFT JOIN f_bureau_agg AS bureau
    ON applicant.SK_ID_CURR = bureau.SK_ID_CURR
LEFT JOIN f_bureau_balance_agg AS bureau_balance
    ON applicant.SK_ID_CURR = bureau_balance.SK_ID_CURR
LEFT JOIN f_pos_cash_agg AS pos_cash
    ON applicant.SK_ID_CURR = pos_cash.SK_ID_CURR
LEFT JOIN f_credit_card_agg AS credit_card
    ON applicant.SK_ID_CURR = credit_card.SK_ID_CURR
LEFT JOIN f_previous_application_agg AS previous
    ON applicant.SK_ID_CURR = previous.SK_ID_CURR
LEFT JOIN f_installments_agg AS installments
    ON applicant.SK_ID_CURR = installments.SK_ID_CURR;
