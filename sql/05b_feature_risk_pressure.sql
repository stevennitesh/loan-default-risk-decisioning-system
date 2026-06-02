-- Risk-pressure features combine applicant affordability with external scores
-- and historical repayment/debt signals. Output stays at applicant population grain.
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
             AND bureau.total_credit_debt IS NOT NULL
        THEN bureau.total_credit_debt / applicant.AMT_INCOME_TOTAL
    END AS bureau_debt_to_income_ratio,
    CASE
        WHEN installments.payment_amount_ratio IS NOT NULL
        THEN GREATEST(1 - installments.payment_amount_ratio, 0)
    END AS payment_shortfall_ratio
FROM f_applicant_static AS applicant
LEFT JOIN f_bureau_agg AS bureau
    ON applicant.SK_ID_CURR = bureau.SK_ID_CURR
LEFT JOIN f_installments_agg AS installments
    ON applicant.SK_ID_CURR = installments.SK_ID_CURR;
