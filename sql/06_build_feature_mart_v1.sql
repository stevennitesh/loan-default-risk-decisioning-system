-- v1 mart joins only applicant-grain feature tables, preserving one row per
-- (SK_ID_CURR, source_population). Diagnostic segment fields stay outside.
CREATE OR REPLACE TABLE mart_credit_risk_features AS
SELECT
    applicant.*,
    bureau.bureau_credit_count,
    bureau.active_credit_count,
    bureau.closed_credit_count,
    bureau.overdue_credit_count,
    bureau.max_credit_day_overdue,
    bureau.total_credit_sum,
    bureau.avg_credit_sum,
    bureau.total_credit_debt,
    bureau.total_credit_limit,
    bureau.total_credit_overdue,
    bureau.avg_days_credit,
    bureau.earliest_days_credit,
    bureau.latest_days_credit,
    bureau.avg_days_credit_enddate,
    bureau.avg_days_enddate_fact,
    previous.previous_application_count,
    previous.approved_application_count,
    previous.refused_application_count,
    previous.canceled_application_count,
    previous.approval_rate,
    previous.refusal_rate,
    previous.avg_application_amount,
    previous.avg_previous_credit_amount,
    previous.total_previous_credit_amount,
    previous.avg_credit_to_application_ratio,
    previous.avg_days_decision,
    previous.earliest_days_decision,
    previous.latest_days_decision,
    installments.installment_payment_count,
    installments.late_payment_count,
    installments.avg_payment_delay_days,
    installments.max_payment_delay_days,
    installments.underpayment_count,
    installments.total_instalment_amount,
    installments.total_payment_amount,
    installments.payment_amount_ratio,
    installments.avg_payment_to_instalment_ratio
FROM f_applicant_static AS applicant
-- LEFT JOINs keep applicants without a given history source in the mart.
LEFT JOIN f_bureau_agg AS bureau
    ON applicant.SK_ID_CURR = bureau.SK_ID_CURR
LEFT JOIN f_previous_application_agg AS previous
    ON applicant.SK_ID_CURR = previous.SK_ID_CURR
LEFT JOIN f_installments_agg AS installments
    ON applicant.SK_ID_CURR = installments.SK_ID_CURR;
