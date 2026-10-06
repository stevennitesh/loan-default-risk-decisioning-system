-- Legacy score-table DDL for manual DuckDB bootstrap. Runtime scoring replaces
-- credit_risk_scores from Python using the current report_contracts column set.
CREATE TABLE IF NOT EXISTS credit_risk_scores (
    applicant_id BIGINT,
    scoring_population VARCHAR,
    observed_target INTEGER,
    score DOUBLE,
    score_decile INTEGER,
    risk_band VARCHAR,
    recommended_action VARCHAR,
    threshold_version VARCHAR,
    model_version VARCHAR,
    top_reason_1 VARCHAR,
    top_reason_2 VARCHAR,
    top_reason_3 VARCHAR,
    scored_at TIMESTAMP
);
