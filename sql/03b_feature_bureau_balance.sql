CREATE OR REPLACE TABLE f_bureau_balance_agg AS
WITH balance_status AS (
    SELECT
        bureau.SK_ID_CURR,
        balance.SK_ID_BUREAU,
        balance.MONTHS_BALANCE,
        UPPER(CAST(balance.STATUS AS VARCHAR)) AS status_code,
        CASE
            WHEN UPPER(CAST(balance.STATUS AS VARCHAR)) IN ('0', '1', '2', '3', '4', '5')
            THEN CAST(balance.STATUS AS INTEGER)
            ELSE NULL
        END AS numeric_status
    FROM stg_bureau_balance AS balance
    INNER JOIN stg_bureau AS bureau
        ON balance.SK_ID_BUREAU = bureau.SK_ID_BUREAU
)
SELECT
    SK_ID_CURR,
    COUNT(*) AS bureau_balance_month_count,
    COUNT(DISTINCT SK_ID_BUREAU) AS bureau_balance_bureau_count,
    SUM(CASE WHEN status_code = 'C' THEN 1 ELSE 0 END) AS bureau_balance_closed_month_count,
    SUM(CASE WHEN status_code = 'X' THEN 1 ELSE 0 END) AS bureau_balance_unknown_month_count,
    SUM(CASE WHEN numeric_status = 0 THEN 1 ELSE 0 END) AS bureau_balance_dpd_0_count,
    SUM(CASE WHEN numeric_status >= 1 THEN 1 ELSE 0 END) AS bureau_balance_dpd_1plus_count,
    SUM(CASE WHEN numeric_status >= 2 THEN 1 ELSE 0 END) AS bureau_balance_dpd_2plus_count,
    SUM(CASE WHEN numeric_status >= 5 THEN 1 ELSE 0 END) AS bureau_balance_dpd_5plus_count,
    MAX(numeric_status) AS bureau_balance_max_status,
    AVG(numeric_status) AS bureau_balance_avg_numeric_status,
    MAX(MONTHS_BALANCE) AS bureau_balance_latest_month,
    MIN(MONTHS_BALANCE) AS bureau_balance_earliest_month,
    SUM(CASE WHEN MONTHS_BALANCE >= -12 THEN 1 ELSE 0 END) AS bureau_balance_recent_month_count,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND status_code <> 'X' THEN 1
            ELSE 0
        END
    ) AS bureau_balance_recent_known_month_count,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND numeric_status >= 1 THEN 1
            ELSE 0
        END
    ) AS bureau_balance_recent_dpd_1plus_count,
    SUM(CASE WHEN numeric_status >= 1 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS bureau_balance_dpd_1plus_rate,
    SUM(CASE WHEN numeric_status >= 2 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0)
        AS bureau_balance_dpd_2plus_rate,
    SUM(
        CASE
            WHEN MONTHS_BALANCE >= -12 AND numeric_status >= 1 THEN 1
            ELSE 0
        END
    ) / NULLIF(
        SUM(
            CASE
                WHEN MONTHS_BALANCE >= -12 AND status_code <> 'X' THEN 1
                ELSE 0
            END
        ),
        0
    ) AS bureau_balance_recent_dpd_1plus_rate
FROM balance_status
GROUP BY SK_ID_CURR;
