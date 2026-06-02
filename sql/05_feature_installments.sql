-- Installment payment rows are collapsed to applicant-level repayment behavior.
CREATE OR REPLACE TABLE f_installments_agg AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS installment_payment_count,
    SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) AS late_payment_count,
    AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS avg_payment_delay_days,
    MAX(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS max_payment_delay_days,
    SUM(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1 ELSE 0 END) AS underpayment_count,
    SUM(AMT_INSTALMENT) AS total_instalment_amount,
    SUM(AMT_PAYMENT) AS total_payment_amount,
    CASE WHEN SUM(AMT_INSTALMENT) IS NOT NULL AND SUM(AMT_INSTALMENT) <> 0 THEN SUM(AMT_PAYMENT) / SUM(AMT_INSTALMENT) END AS payment_amount_ratio,
    AVG(CASE WHEN AMT_INSTALMENT IS NOT NULL AND AMT_INSTALMENT <> 0 THEN AMT_PAYMENT / AMT_INSTALMENT END) AS avg_payment_to_instalment_ratio
FROM stg_installments_payments
GROUP BY SK_ID_CURR;
