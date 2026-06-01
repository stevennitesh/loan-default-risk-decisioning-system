# Loan Default Risk Decisioning System — Implementation Checklist

**Version:** 0.2.1 aligned to `PROJECT_SPEC_v0.3.1.md`  
**Status:** Build checklist  
**Last updated:** 2026-04-25

---

## 0. Repo setup

- [ ] Create GitHub repository: `loan-default-decisioning`
- [ ] Add `.gitignore` for raw data, models, DuckDB files, cache files, and secrets
- [ ] Add `README.md`
- [ ] Add `docs/spec/PROJECT_SPEC.md`
- [ ] Add `requirements.txt` or `pyproject.toml`
- [ ] Add `Makefile`
- [ ] Add `Dockerfile`
- [ ] Add `configs/base.yaml`
- [ ] Add empty package file: `src/__init__.py`
- [ ] Add initial `reports/`, `models/`, `powerbi/screenshots/`, and `data/sample/` placeholders

## 1. Config contract

- [ ] Define project name and random seed
- [ ] Define all local paths
- [ ] Define v1 source files
- [ ] Define train/validation/test split sizes
- [ ] Define baseline and primary model settings
- [ ] Define excluded feature groups: target, identifiers, direct demographic fields, protected-status-like fields, and direct age-derived fields
- [ ] Define business assumptions
- [ ] Define threshold policy scenarios with placeholder thresholds

## 2. Data ingestion

- [ ] Download Home Credit Default Risk data from Kaggle
- [ ] Place raw CSV files in `data/raw/`
- [ ] Write `src/ingest.py`
- [ ] Convert v1 CSV files to Parquet
- [ ] Save Parquet files to `data/parquet/`
- [ ] Create DuckDB staging tables from Parquet
- [ ] Validate row counts against raw files
- [ ] Add `make ingest`

v1 files:

- [ ] `application_train.csv`
- [ ] `application_test.csv`
- [ ] `bureau.csv`
- [ ] `previous_application.csv`
- [ ] `installments_payments.csv`

## 3. DuckDB staging

- [ ] Create DuckDB database in `data/db/credit_risk.duckdb`
- [ ] Write SQL for staging tables
- [ ] Load Parquet into staging tables
- [ ] Validate required columns exist
- [ ] Validate row counts
- [ ] Add data contract checks

## 4. SQL feature engineering

- [ ] Build `f_applicant_static`
- [ ] Build `f_bureau_agg`
- [ ] Build `f_previous_application_agg`
- [ ] Build `f_installments_agg`
- [ ] Build `mart_credit_risk_features`
- [ ] Add `source_population` column
- [ ] Ensure one row per `SK_ID_CURR` per `source_population`
- [ ] Ensure `TARGET` exists only for labeled training rows
- [ ] Exclude target, identifiers, direct demographic/protected-status-like fields, marital/family-status fields, and direct age-derived model predictors from the model feature set
- [ ] Keep any diagnostic-only demographic/protected-status-like fields separate from the model feature matrix
- [ ] Save final model feature list
- [ ] Add `make features`

## 5. Tests before modeling

- [ ] Add `test_data_contract.py`
- [ ] Add `test_feature_sql.py`
- [ ] Test no duplicate applicant IDs per source population
- [ ] Test feature table grain
- [ ] Test required columns exist
- [ ] Test training rows have non-null `TARGET`
- [ ] Test Kaggle test rows have null `TARGET`
- [ ] Test forbidden and diagnostic-only columns are excluded from model feature list
- [ ] Test representative feature calculations with small sample data

## 6. Baseline model

- [ ] Write `src/train.py`
- [ ] Create stratified train/validation/test split from labeled rows only
- [ ] Build preprocessing pipeline
- [ ] Train logistic regression baseline
- [ ] Save baseline metrics
- [ ] Save model artifact
- [ ] Save feature list and run metadata

## 7. LightGBM model

- [ ] Train LightGBM model
- [ ] Use class weighting or `scale_pos_weight` as the initial imbalance strategy
- [ ] Compare against logistic regression baseline
- [ ] Tune main hyperparameters lightly
- [ ] Save model artifact
- [ ] Save model metadata

## 8. Evaluation

- [ ] Write `src/evaluate.py`
- [ ] Compute ROC-AUC
- [ ] Compute PR-AUC
- [ ] Compute Brier score
- [ ] Build lift-by-decile table
- [ ] Build calibration bins
- [ ] Build confusion matrix by threshold scenario
- [ ] Build recall-at-review-capacity metric
- [ ] Export `model_run_summary`
- [ ] Export `model_metrics_summary`
- [ ] Export `model_lift_by_decile`
- [ ] Export `model_calibration_bins`
- [ ] Export `model_confusion_matrix`
- [ ] Add `make evaluate`

## 9. Threshold and business-value analysis

- [ ] Write `src/thresholding.py`
- [ ] Define growth-oriented, balanced, and risk-averse scenarios
- [ ] Implement expected-value formula
- [ ] Pull business assumptions from `configs/base.yaml`
- [ ] Select thresholds using validation data only
- [ ] Freeze threshold values before final test reporting
- [ ] Export `model_threshold_metrics`
- [ ] Add `test_threshold_policy.py`
- [ ] Add `test_expected_value.py`

## 10. Batch scoring

- [ ] Write `src/score_batch.py`
- [ ] Create `credit_risk_scores` table
- [ ] Score labeled holdout population
- [ ] Score unlabeled Kaggle test population
- [ ] Populate `scoring_population`
- [ ] Populate nullable `observed_target`
- [ ] Assign score deciles
- [ ] Assign risk bands
- [ ] Assign recommended simulated actions
- [ ] Add model version and threshold version
- [ ] Add scoring timestamp
- [ ] Add `make score`
- [ ] Add `test_scoring_schema.py`

## 11. Explainability

- [ ] Write `src/explain.py`
- [ ] Generate SHAP global feature importance
- [ ] Export `model_feature_importance`
- [ ] Generate top applicant-level reason-code-style outputs
- [ ] Confirm SHAP/reason-code outputs do not surface excluded diagnostic-only fields
- [ ] Add top reason fields to `credit_risk_scores`
- [ ] Add SHAP plot(s) to `reports/figures/`

## 12. Power BI data exports

- [ ] Export `credit_risk_scores`
- [ ] Export `model_run_summary`
- [ ] Export `model_metrics_summary`
- [ ] Export `model_threshold_metrics`
- [ ] Export `model_lift_by_decile`
- [ ] Export `model_calibration_bins`
- [ ] Export `model_confusion_matrix`
- [ ] Export `model_feature_importance`
- [ ] Export `segment_performance_summary`
- [ ] Add `make dashboard-data`

## 13. Power BI dashboard

- [ ] Build page 1: Decisioning Overview
- [ ] Add KPI cards
- [ ] Add score distribution
- [ ] Add risk band counts
- [ ] Add threshold scenario comparison
- [ ] Add confusion matrix
- [ ] Add lift chart
- [ ] Add expected-value by threshold visual
- [ ] Add approval/default tradeoff visual
- [ ] Add top model drivers
- [ ] Save dashboard screenshot to `powerbi/screenshots/`

## 14. Model card

- [ ] Create `reports/model_card.md`
- [ ] Add intended use
- [ ] Add not-intended-for section
- [ ] Add dataset and target definition
- [ ] Add training split summary
- [ ] Add feature summary and excluded features
- [ ] Add model type and metrics
- [ ] Add threshold policy
- [ ] Add expected-value assumptions
- [ ] Add explainability notes
- [ ] Add limitations and monitoring considerations

## 15. Documentation polish

- [ ] Update README with final metrics
- [ ] Add architecture diagram
- [ ] Add dashboard screenshot
- [ ] Add threshold scenario table
- [ ] Add lift chart or decile table
- [ ] Add limitations/model-risk section
- [ ] Add run instructions
- [ ] Add project narrative to top of README
- [ ] Clearly separate labeled evaluation results from unlabeled scoring outputs

## 16. Milestone 1 gate

No modeling should start until this passes:

- [ ] Raw v1 CSV files convert to Parquet
- [ ] DuckDB staging tables exist
- [ ] Final feature mart exists
- [ ] Feature mart has one row per `SK_ID_CURR` per source population
- [ ] Row-count checks pass
- [ ] Duplicate checks pass
- [ ] Forbidden model feature checks pass

## 17. Stretch goals

- [ ] Add v1.1 tables: `bureau_balance`, `POS_CASH_balance`, `credit_card_balance`
- [ ] Add model validation appendix page in Power BI
- [ ] Add simple FastAPI scoring endpoint
- [ ] Add optional Postgres support
- [ ] Add more robust segment diagnostics with diagnostic-only sensitive fields kept separate from model features
- [ ] Add MLflow only after the pipeline is already clean
