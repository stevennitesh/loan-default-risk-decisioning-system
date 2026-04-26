CREATE OR REPLACE TABLE f_bureau_agg AS
SELECT
    SK_ID_CURR,
    COUNT(*) AS bureau_credit_count,
    SUM(CASE WHEN LOWER(CREDIT_ACTIVE) = 'active' THEN 1 ELSE 0 END) AS active_credit_count,
    SUM(CASE WHEN LOWER(CREDIT_ACTIVE) = 'closed' THEN 1 ELSE 0 END) AS closed_credit_count,
    SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 OR AMT_CREDIT_SUM_OVERDUE > 0 THEN 1 ELSE 0 END) AS overdue_credit_count,
    MAX(CREDIT_DAY_OVERDUE) AS max_credit_day_overdue,
    SUM(AMT_CREDIT_SUM) AS total_credit_sum,
    AVG(AMT_CREDIT_SUM) AS avg_credit_sum,
    SUM(AMT_CREDIT_SUM_DEBT) AS total_credit_debt,
    SUM(AMT_CREDIT_SUM_LIMIT) AS total_credit_limit,
    SUM(AMT_CREDIT_SUM_OVERDUE) AS total_credit_overdue,
    AVG(DAYS_CREDIT) AS avg_days_credit,
    MIN(DAYS_CREDIT) AS earliest_days_credit,
    MAX(DAYS_CREDIT) AS latest_days_credit,
    AVG(DAYS_CREDIT_ENDDATE) AS avg_days_credit_enddate,
    AVG(DAYS_ENDDATE_FACT) AS avg_days_enddate_fact
FROM stg_bureau
GROUP BY SK_ID_CURR;
