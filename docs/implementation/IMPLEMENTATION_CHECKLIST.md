# Loan Default Risk Decisioning System — Implementation Checklist

## 0. Repo setup

- [ ] Create GitHub repository: `loan-default-decisioning`
- [ ] Add `.gitignore` for raw data, models, DuckDB files, cache files, and secrets
- [ ] Add `README.md`
- [ ] Add `docs/spec/PROJECT_SPEC.md`
- [ ] Add `requirements.txt` or `pyproject.toml`
- [ ] Add `Makefile`
- [ ] Add `Dockerfile`
- [ ] Add `configs/base.yaml`

## 1. Data ingestion

- [ ] Download Home Credit Default Risk data from Kaggle
- [ ] Place raw CSV files in `data/raw/`
- [ ] Write `src/ingest.py`
- [ ] Convert v1 CSV files to Parquet
- [ ] Save Parquet files to `data/parquet/`
- [ ] Add `make ingest`

v1 files:

- [ ] `application_train.csv`
- [ ] `application_test.csv`
- [ ] `bureau.csv`
- [ ] `previous_application.csv`
- [ ] `installments_payments.csv`

## 2. DuckDB staging

- [ ] Create DuckDB database in `data/db/credit_risk.duckdb`
- [ ] Write SQL for staging tables
- [ ] Load Parquet into staging tables
- [ ] Validate row counts against raw files
- [ ] Add data contract checks

## 3. SQL feature engineering

- [ ] Build `f_applicant_static`
- [ ] Build `f_bureau_agg`
- [ ] Build `f_previous_application_agg`
- [ ] Build `f_installments_agg`
- [ ] Build `mart_credit_risk_features`
- [ ] Ensure one row per `SK_ID_CURR`
- [ ] Ensure `TARGET` exists only for labeled training rows
- [ ] Add `make features`

## 4. Tests before modeling

- [ ] Add `test_data_contract.py`
- [ ] Add `test_feature_sql.py`
- [ ] Test no duplicate applicant IDs
- [ ] Test score-feature table grain
- [ ] Test required columns exist
- [ ] Test representative feature calculations with small sample data

## 5. Baseline model

- [ ] Write `src/train.py`
- [ ] Create stratified train/validation/test split
- [ ] Build preprocessing pipeline
- [ ] Train logistic regression baseline
- [ ] Save baseline metrics
- [ ] Save model artifact

## 6. LightGBM model

- [ ] Train LightGBM model
- [ ] Compare against logistic regression baseline
- [ ] Tune main hyperparameters lightly
- [ ] Save model artifact
- [ ] Save model metadata

## 7. Evaluation

- [ ] Write `src/evaluate.py`
- [ ] Compute ROC-AUC
- [ ] Compute PR-AUC
- [ ] Compute Brier score
- [ ] Build lift-by-decile table
- [ ] Build calibration bins
- [ ] Build confusion matrix by threshold
- [ ] Build recall-at-review-capacity metric
- [ ] Export `model_metrics_summary`
- [ ] Add `make evaluate`

## 8. Threshold and business-value analysis

- [ ] Write `src/thresholding.py`
- [ ] Define growth-oriented, balanced, and risk-averse scenarios
- [ ] Implement expected-value formula
- [ ] Add configurable assumptions in `configs/base.yaml`
- [ ] Export `model_threshold_metrics`
- [ ] Add `test_threshold_policy.py`
- [ ] Add `test_expected_value.py`

## 9. Batch scoring

- [ ] Write `src/score_batch.py`
- [ ] Create `credit_risk_scores` table
- [ ] Score labeled holdout population
- [ ] Score unlabeled Kaggle test population
- [ ] Assign risk bands
- [ ] Assign recommended simulated actions
- [ ] Add model version and threshold version
- [ ] Add scoring timestamp
- [ ] Add `make score`
- [ ] Add `test_scoring_schema.py`

## 10. Explainability

- [ ] Write `src/explain.py`
- [ ] Generate SHAP global feature importance
- [ ] Generate top applicant-level reason-code-style outputs
- [ ] Export `model_feature_importance`
- [ ] Add SHAP plot(s) to `reports/figures/`

## 11. Power BI data exports

- [ ] Export `credit_risk_scores`
- [ ] Export `model_metrics_summary`
- [ ] Export `model_threshold_metrics`
- [ ] Export `model_lift_by_decile`
- [ ] Export `model_calibration_bins`
- [ ] Export `model_feature_importance`
- [ ] Add `make dashboard-data`

## 12. Power BI dashboard

- [ ] Build page 1: Decisioning Overview
- [ ] Add KPI cards
- [ ] Add score distribution
- [ ] Add risk band counts
- [ ] Add threshold scenario comparison
- [ ] Add confusion matrix
- [ ] Add lift chart
- [ ] Add expected-value by threshold visual
- [ ] Add top model drivers
- [ ] Save dashboard screenshot to `powerbi/screenshots/`

## 13. Documentation polish

- [ ] Update README with final metrics
- [ ] Add architecture diagram
- [ ] Add dashboard screenshot
- [ ] Add threshold scenario table
- [ ] Add limitations/model-risk section
- [ ] Add run instructions
- [ ] Add project narrative to top of README

## 14. Stretch goals

- [ ] Add v1.1 tables: `bureau_balance`, `POS_CASH_balance`, `credit_card_balance`
- [ ] Add model validation appendix page in Power BI
- [ ] Add simple FastAPI scoring endpoint
- [ ] Add optional Postgres support
- [ ] Add more robust segment diagnostics
