CREATE OR REPLACE TABLE f_previous_application_agg AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS previous_application_count,
    SUM(CASE WHEN LOWER(NAME_CONTRACT_STATUS) = 'approved' THEN 1 ELSE 0 END) AS approved_application_count,
    SUM(CASE WHEN LOWER(NAME_CONTRACT_STATUS) = 'refused' THEN 1 ELSE 0 END) AS refused_application_count,
    SUM(CASE WHEN LOWER(NAME_CONTRACT_STATUS) = 'canceled' THEN 1 ELSE 0 END) AS canceled_application_count,
    SUM(CASE WHEN LOWER(NAME_CONTRACT_STATUS) = 'approved' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS approval_rate,
    SUM(CASE WHEN LOWER(NAME_CONTRACT_STATUS) = 'refused' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS refusal_rate,
    AVG(AMT_APPLICATION) AS avg_application_amount,
    AVG(AMT_CREDIT) AS avg_previous_credit_amount,
    SUM(AMT_CREDIT) AS total_previous_credit_amount,
    AVG(CASE WHEN AMT_APPLICATION IS NOT NULL AND AMT_APPLICATION <> 0 THEN AMT_CREDIT / AMT_APPLICATION END) AS avg_credit_to_application_ratio,
    AVG(DAYS_DECISION) AS avg_days_decision,
    MIN(DAYS_DECISION) AS earliest_days_decision,
    MAX(DAYS_DECISION) AS latest_days_decision
FROM stg_previous_application
GROUP BY SK_ID_CURR;
