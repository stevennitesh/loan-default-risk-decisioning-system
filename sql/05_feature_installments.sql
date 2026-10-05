-- Payment events are not scheduled obligations. Competing calendar versions,
-- conflicting schedules, and unknown cashflows remain explicit unknown history.
CREATE OR REPLACE TABLE n_installment_obligations AS
WITH source_installments AS (
    SELECT * FROM stg_installments_payments
), versions AS (
    SELECT SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER,
        COUNT(DISTINCT NUM_INSTALMENT_VERSION) AS version_count
    FROM source_installments
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER
), schedules AS (
    SELECT SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER,
        MIN(DAYS_INSTALMENT) AS due_day,
        MIN(AMT_INSTALMENT) AS scheduled_amount,
        COUNT(*) FILTER (WHERE DAYS_ENTRY_PAYMENT < 0) AS payment_row_count,
        COUNT(DISTINCT DAYS_INSTALMENT) = 1
            AND COUNT(DAYS_INSTALMENT) = COUNT(*)
            AND COUNT(DISTINCT AMT_INSTALMENT) = 1
            AND COUNT(AMT_INSTALMENT) = COUNT(*)
            AND BOOL_AND(ISFINITE(AMT_INSTALMENT) AND AMT_INSTALMENT > 0)
            AND BOOL_AND(ISFINITE(DAYS_INSTALMENT) AND DAYS_INSTALMENT < 0)
            AND SK_ID_PREV IS NOT NULL
            AND NUM_INSTALMENT_VERSION IS NOT NULL
            AND NUM_INSTALMENT_NUMBER IS NOT NULL AS schedule_known,
        BOOL_OR(
            DAYS_ENTRY_PAYMENT IS NULL OR NOT ISFINITE(DAYS_ENTRY_PAYMENT)
            OR (DAYS_ENTRY_PAYMENT < 0 AND (
                AMT_PAYMENT IS NULL OR NOT ISFINITE(AMT_PAYMENT) OR AMT_PAYMENT < 0
            ))
        ) AS payment_unknown
    FROM source_installments
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER
    -- Select obligations after checking the complete schedule. Filtering rows
    -- first would conceal contradictory future due days or competing versions.
    HAVING BOOL_OR(DAYS_INSTALMENT < 0 OR DAYS_INSTALMENT IS NULL)
), payment_days AS (
    -- Same-day partial cashflows are accumulated together. Identical records are
    -- retained: the source provides no unique cashflow ID proving duplication.
    SELECT SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER,
        DAYS_ENTRY_PAYMENT AS payment_day, SUM(AMT_PAYMENT) AS daily_paid
    FROM source_installments
    WHERE DAYS_ENTRY_PAYMENT < 0 AND ISFINITE(DAYS_ENTRY_PAYMENT)
        AND AMT_PAYMENT >= 0 AND ISFINITE(AMT_PAYMENT)
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION,
        NUM_INSTALMENT_NUMBER, DAYS_ENTRY_PAYMENT
), cumulative_payments AS (
    SELECT *, SUM(daily_paid) OVER (
        PARTITION BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER
        ORDER BY payment_day ROWS UNBOUNDED PRECEDING
    ) AS cumulative_paid FROM payment_days
), repayment AS (
    SELECT s.*, v.version_count,
        COALESCE(SUM(p.daily_paid), 0) AS observed_paid,
        MIN(p.payment_day) FILTER (WHERE p.cumulative_paid >= s.scheduled_amount)
            AS completion_day
    FROM schedules s
    LEFT JOIN versions v
        ON s.SK_ID_CURR IS NOT DISTINCT FROM v.SK_ID_CURR
        AND s.SK_ID_PREV IS NOT DISTINCT FROM v.SK_ID_PREV
        AND s.NUM_INSTALMENT_NUMBER IS NOT DISTINCT FROM v.NUM_INSTALMENT_NUMBER
    LEFT JOIN cumulative_payments p
        ON s.SK_ID_CURR IS NOT DISTINCT FROM p.SK_ID_CURR
        AND s.SK_ID_PREV IS NOT DISTINCT FROM p.SK_ID_PREV
        AND s.NUM_INSTALMENT_VERSION IS NOT DISTINCT FROM p.NUM_INSTALMENT_VERSION
        AND s.NUM_INSTALMENT_NUMBER IS NOT DISTINCT FROM p.NUM_INSTALMENT_NUMBER
    GROUP BY ALL
), status AS (
    SELECT *, COALESCE(schedule_known AND version_count = 1, FALSE) AS obligation_known
    FROM repayment
)
SELECT SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER,
    due_day, scheduled_amount, payment_row_count,
    obligation_known, payment_unknown,
    CASE WHEN obligation_known AND NOT payment_unknown THEN observed_paid END AS paid_amount,
    CASE WHEN obligation_known AND NOT payment_unknown
        THEN CAST(observed_paid < scheduled_amount AS INTEGER) END AS underpaid,
    CASE WHEN obligation_known AND NOT payment_unknown
        THEN CAST(completion_day IS NULL OR completion_day > due_day AS INTEGER) END AS late,
    CASE WHEN obligation_known AND NOT payment_unknown
        THEN completion_day - due_day END AS completed_delay_days,
    CASE WHEN obligation_known AND NOT payment_unknown AND completion_day IS NULL
        THEN -due_day END AS arrears_age_days
FROM status;

CREATE OR REPLACE TABLE f_installments_agg AS
SELECT SK_ID_CURR,
    SUM(payment_row_count) AS installment_payment_count,
    COUNT(*) AS installment_obligation_count,
    COUNT(*) FILTER (WHERE NOT obligation_known) AS installment_ambiguous_obligation_count,
    COUNT(*) FILTER (WHERE obligation_known AND payment_unknown)
        AS installment_unknown_payment_obligation_count,
    COUNT(*) FILTER (WHERE paid_amount IS NOT NULL) AS installment_known_payment_obligation_count,
    COUNT(arrears_age_days) AS installment_ongoing_arrears_count,
    AVG(arrears_age_days) AS installment_avg_arrears_age_days,
    SUM(late) AS late_payment_count,
    AVG(completed_delay_days) AS avg_payment_delay_days,
    MAX(completed_delay_days) AS max_payment_delay_days,
    SUM(underpaid) AS underpayment_count,
    SUM(scheduled_amount) FILTER (WHERE paid_amount IS NOT NULL) AS total_instalment_amount,
    SUM(paid_amount) AS total_payment_amount,
    SUM(paid_amount) / NULLIF(SUM(scheduled_amount) FILTER (WHERE paid_amount IS NOT NULL), 0)
        AS payment_amount_ratio,
    AVG(paid_amount / NULLIF(scheduled_amount, 0)) AS avg_payment_to_instalment_ratio
FROM n_installment_obligations GROUP BY SK_ID_CURR;
