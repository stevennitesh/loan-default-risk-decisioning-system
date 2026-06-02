# Loan Default Risk Decisioning System — Implementation Plan

**Version:** 0.1  
**Status:** Implemented build plan with frozen v1 and post-v1 comparison
**Owner:** Steven  
**Last updated:** 2026-06-01
**Aligned spec:** `docs/spec/PROJECT_SPEC.md` / v0.3.1 final portfolio contract

---

## 1. Purpose

This document turns the project specification into an executable build plan. The goal is to produce a recruiter-facing financial ML project that demonstrates:

- SQL-first feature engineering;
- reproducible Python training and evaluation;
- LightGBM modeling with a logistic regression baseline;
- threshold and expected-value analysis;
- batch scoring back to DuckDB;
- Power BI-ready reporting outputs;
- testing and validation discipline.

The project should read as an applied financial decisioning system, not a notebook-only Kaggle exercise.

---

## 2. Locked v1 Scope

| Area | v1 decision |
|---|---|
| Dataset | Home Credit Default Risk |
| Database | DuckDB |
| Storage | Parquet |
| Primary model | LightGBM |
| Baseline model | Logistic regression |
| Modeling grain | One row per `SK_ID_CURR` |
| v1 source tables | `application_train`, `application_test`, `bureau`, `previous_application`, `installments_payments` |
| Post-v1 comparison source tables | `bureau_balance`, `POS_CASH_balance`, `credit_card_balance` |
| Dashboard scope | One polished executive page first |
| API | Deferred unless core project is complete |
| Model framing | Decision-support simulation, not automated underwriting |

---

## 3. Implementation Principles

1. **No modeling until the data pipeline works.**  
   First prove: CSV → Parquet → DuckDB staging → SQL feature mart.

2. **SQL owns feature extraction.**  
   Python should orchestrate, train, evaluate, score, and export. SQL should build the applicant-level feature mart.

3. **The feature mart must have one row per applicant.**  
   Any join that expands `SK_ID_CURR` is a build failure.

4. **Evaluation and scoring populations stay separate.**  
   Model metrics come only from labeled splits of `application_train`. Kaggle `application_test` is used only for production-like scoring demonstration.

5. **No direct sensitive/protected-status-like fields in model features.**  
   Excluded fields may exist only in a separate diagnostic layer, if used at all.

6. **Power BI data contracts are defined before dashboarding.**  
   The dashboard should consume clean exported tables, not ad hoc notebook outputs.

7. **Every major step has an artifact.**  
   If a step cannot produce a file, table, metric, or test, it is probably not implementation-ready.

---

## 4. Build Milestones

## Milestone 0 — Repo Skeleton and Configuration

### Objective
Create the project structure and reproducibility interface before writing modeling code.

### Tasks

- Create repository: `loan-default-decisioning`.
- Add `.gitignore` for:
  - raw data;
  - DuckDB files;
  - Parquet files;
  - model artifacts;
  - cache files;
  - environment files;
  - local Power BI temporary files.
- Add folder structure:

```text
configs/
data/raw/
data/parquet/
data/db/
sql/
src/
tests/
reports/figures/
powerbi/screenshots/
models/
```

- Add:
  - `README.md`
  - `docs/spec/PROJECT_SPEC.md`
  - `docs/implementation/IMPLEMENTATION_PLAN.md`
  - `docs/testing/TESTING_PLAN.md`
  - `docs/validation/VALIDATION_PLAN.md`
  - `Makefile`
  - `Dockerfile`
  - `requirements.txt` or `pyproject.toml`
  - `configs/base.yaml`

### Required command targets

```bash
make setup
make ingest
make features
make train
make evaluate
make score
make dashboard-data
make test
```

### Output artifacts

```text
Makefile
Dockerfile
configs/base.yaml
requirements.txt or pyproject.toml
```

### Gate

- `make setup` works in a clean local environment.
- Folder structure exists.
- Config file parses successfully.

---

## Milestone 1 — Data Ingestion and Parquet Conversion

### Objective
Convert raw Kaggle CSVs into Parquet files and load them into DuckDB staging tables.

### Tasks

- Download Home Credit files manually from Kaggle.
- Place v1 raw files in `data/raw/`:

```text
application_train.csv
application_test.csv
bureau.csv
previous_application.csv
installments_payments.csv
HomeCredit_columns_description.csv
```

- Write `src/ingest.py`.
- Convert each v1 CSV file to Parquet.
- Create DuckDB database at:

```text
data/db/credit_risk.duckdb
```

- Load Parquet files into staging tables:

```text
stg_application_train
stg_application_test
stg_bureau
stg_previous_application
stg_installments_payments
```

- Store ingestion metadata:

```text
reports/ingestion_summary.csv
```

### Output artifacts

```text
data/parquet/application_train.parquet
data/parquet/application_test.parquet
data/parquet/bureau.parquet
data/parquet/previous_application.parquet
data/parquet/installments_payments.parquet
data/db/credit_risk.duckdb
reports/ingestion_summary.csv
```

### Gate

- Row counts in DuckDB match row counts from source CSVs.
- No source file required by v1 is missing.
- Staging tables are queryable from DuckDB.

---

## Milestone 2 — SQL Feature Engineering

### Objective
Build applicant-level SQL feature tables and the final feature mart.

### SQL files

```text
sql/02_feature_applicant.sql
sql/03_feature_bureau.sql
sql/03b_feature_bureau_balance.sql
sql/04_feature_previous_applications.sql
sql/04b_feature_pos_cash.sql
sql/04c_feature_credit_card.sql
sql/05_feature_installments.sql
sql/05b_feature_risk_pressure.sql
sql/05c_feature_recency_deterioration.sql
sql/05d_feature_last_k_temporal.sql
sql/06_build_feature_mart.sql
sql/06_build_feature_mart_v1.sql
sql/07_create_score_tables.sql
```

Staging table creation and Parquet loading are owned by `src/ingest.py`; these SQL files own
feature-table and score-table contracts.

### Feature tables

| Table | Purpose |
|---|---|
| `f_applicant_static` | Current application features, affordability ratios, external scores |
| `f_bureau_agg` | External credit history aggregates |
| `f_previous_application_agg` | Prior application/refusal/approval behavior |
| `f_installments_agg` | Payment timing and late-payment behavior |
| `mart_credit_risk_features` | Final one-row-per-applicant modeling table |
| `segment_diagnostics` | Optional diagnostic-only fields not used in training |

### Feature examples

| Feature group | Examples |
|---|---|
| Affordability | `credit_to_income_ratio`, `annuity_to_income_ratio`, `goods_price_to_income_ratio` |
| External scores | `ext_source_mean`, `ext_source_min`, `ext_source_max`, `ext_source_missing_count` |
| Bureau | active credit count, closed credit count, overdue amount summary, credit duration summary |
| Previous applications | previous application count, approval rate, refusal count, amount ratio summary |
| Installments | average payment delay, max payment delay, late payment count, payment ratio |

### Output artifacts

```text
mart_credit_risk_features
segment_diagnostics
reports/feature_mart_profile.csv
```

### Gate

- `mart_credit_risk_features` has exactly one row per `SK_ID_CURR`.
- Training rows include `TARGET`.
- Unlabeled scoring rows from `application_test` do not have a target.
- Direct sensitive/protected-status-like fields are excluded from model features.
- All feature SQL can be rerun from scratch.

---

## Milestone 3 — Data Contract and Feature Tests

### Objective
Prevent modeling from starting until the feature mart is structurally correct.

### Tasks

- Add `tests/test_data_contract.py`.
- Add `tests/test_feature_sql.py`.
- Create small synthetic fixtures through pytest helpers.
- Validate:
  - required columns exist;
  - no duplicate applicant IDs;
  - feature mart grain is correct;
  - no forbidden fields are used for model features;
  - representative SQL aggregates are calculated correctly;
  - staging row counts match expected values.

### Output artifacts

```text
tests/test_data_contract.py
tests/test_feature_sql.py
tests/conftest.py
```

### Gate

```bash
make test
```

passes all contract and feature tests.

---

## Milestone 4 — Baseline Model

### Objective
Train a simple logistic regression baseline to validate preprocessing and provide a benchmark.

### Tasks

- Write `src/train.py`.
- Load `mart_credit_risk_features` from DuckDB.
- Create stratified train/validation/test split from labeled `application_train` rows:

```text
Train: 70%
Validation: 15%
Test: 15%
```

- Fit preprocessing only on the training split.
- Exclude:
  - `TARGET`;
  - identifiers;
  - direct demographic fields;
  - protected-status-like fields;
  - direct age-derived fields;
  - diagnostic-only fields.
- Train logistic regression baseline.
- Save baseline artifact and metrics.

### Output artifacts

```text
models/logistic_regression_baseline.joblib
reports/model_run_summary.csv
reports/model_metrics_summary.csv
```

### Gate

- Baseline model trains without leakage-prone fields.
- Predicted scores are between 0 and 1.
- Baseline metrics are exported.
- Preprocessing pipeline is reproducible.

---

## Milestone 5 — LightGBM Model

### Objective
Train the primary boosted-tree model and compare it against the baseline.

### Tasks

- Train LightGBM classifier.
- Use controlled, lightweight hyperparameter tuning.
- Consider class weighting or `scale_pos_weight`.
- Save model artifact and metadata.
- Compare against logistic regression on validation data.

### Output artifacts

```text
models/lightgbm_credit_risk.joblib
reports/model_run_summary.csv
reports/model_metrics_summary.csv
```

### Gate

- LightGBM trains successfully.
- LightGBM is compared to baseline using PR-AUC, ROC-AUC, Brier score, and lift.
- If LightGBM does not outperform the baseline, document the result rather than hiding it.

---

## Milestone 6 — Evaluation Pipeline

### Objective
Generate model evaluation outputs suitable for README, validation report, and Power BI.

### Tasks

- Write `src/evaluate.py`.
- Compute:
  - ROC-AUC;
  - PR-AUC;
  - Brier score;
  - precision at top decile;
  - recall at manual review capacity;
  - lift by decile;
  - calibration bins;
  - confusion matrix by selected threshold scenarios.
- Export dashboard-ready tables.

### Output tables/files

```text
model_metrics_summary
model_lift_by_decile
model_calibration_bins
model_confusion_matrix
reports/validation_report.md
reports/figures/roc_curve.png
reports/figures/pr_curve.png
reports/figures/calibration_curve.png
reports/figures/lift_chart.png
```

### Gate

- Evaluation runs from saved model artifacts.
- Final test-set results are produced only after model/threshold choices are fixed on validation data.
- Metrics are reproducible with the configured seed.

---

## Milestone 7 — Threshold and Expected-Value Analysis

### Objective
Convert model scores into business policy scenarios.

### Tasks

- Write `src/thresholding.py`.
- Define policy scenarios:

| Scenario | Intent |
|---|---|
| Growth-oriented | Higher approval rate, higher credit risk |
| Balanced | Balanced approval, default capture, and review cost |
| Risk-averse | Lower approval rate, stronger risk control |

- Use configurable assumptions:

```yaml
business_assumptions:
  expected_margin_per_good_loan: 1000
  expected_loss_per_bad_loan: 5000
  manual_review_cost: 50
  manual_review_capacity_rate: 0.10
```

- Export threshold metrics:

```text
model_threshold_metrics
```

### Gate

- `T_low < T_high` for all scenarios.
- Every applicant maps to exactly one risk band.
- Expected-value calculations reconcile to assumptions.
- Selected threshold scenarios are chosen from validation results before final test reporting.

---

## Milestone 8 — Batch Scoring

### Objective
Create the implementation-facing scoring step that writes predictions back to DuckDB.

### Tasks

- Write `src/score_batch.py`.
- Score two populations:

| Population | Purpose |
|---|---|
| Labeled holdout test split | Evaluation and dashboard validation |
| Kaggle `application_test` | Production-like unlabeled scoring demo |

- Create/replace `credit_risk_scores` table.
- Add:
  - applicant ID;
  - score;
  - risk band;
  - recommended action;
  - threshold version;
  - model version;
  - scoring population;
  - timestamp;
  - reason-code-style fields when available.

### Output table

```text
credit_risk_scores
```

### Gate

- Scores are between 0 and 1.
- No duplicate `applicant_id` values within a scoring population.
- All rows receive a risk band and action.
- Holdout scoring and Kaggle test scoring are distinguishable.

---

## Milestone 9 — Explainability Outputs

### Objective
Generate global feature importance and local reason-code-style explanations.

### Tasks

- Write `src/explain.py`.
- Compute SHAP values for the selected LightGBM model.
- Export global feature importance.
- Generate top local reason-code-style drivers for scored applicants where feasible.
- Exclude diagnostic-only fields from all SHAP/reason-code outputs.

### Output artifacts

```text
model_feature_importance
reports/figures/shap_summary.png
credit_risk_scores.top_reason_1
credit_risk_scores.top_reason_2
credit_risk_scores.top_reason_3
```

### Gate

- SHAP outputs contain only model-eligible features.
- Reason-code-style outputs are clearly labeled as interpretation artifacts, not adverse-action notices.

---

## Milestone 10 — Power BI Data Exports

### Objective
Export clean tables for dashboarding.

### Tasks

- Add `make dashboard-data`.
- Export:

```text
credit_risk_scores
model_metrics_summary
model_threshold_metrics
model_lift_by_decile
model_calibration_bins
model_confusion_matrix
model_feature_importance
segment_performance_summary
```

- Save exports to:

```text
reports/dashboard_data/
```

### Gate

- Power BI can load exported tables without manual transformation.
- Column names are stable and readable.
- Tables include model version and scenario identifiers where needed.

---

## Milestone 11 — Power BI Dashboard

### Objective
Build the recruiter-facing dashboard screenshot.

### Page 1: Decisioning Overview

Required visuals:

- KPI cards: ROC-AUC, PR-AUC, Brier score, top-decile lift, selected expected value;
- score distribution;
- risk band counts;
- threshold scenario selector;
- confusion matrix;
- lift chart;
- expected value by threshold/scenario;
- approval/default tradeoff;
- top model drivers.

### Optional Page 2: Validation Appendix

- ROC curve;
- precision-recall curve;
- calibration curve;
- decile table;
- segment diagnostics;
- missingness summary.

### Output artifacts

```text
powerbi/dashboard.pbix
powerbi/dashboard_post_v1.pbix
powerbi/screenshots/decisioning_overview.png
powerbi/screenshots/model_validation_appendix.png
```

### Gate

- README can explain the dashboard screenshot in less than two minutes.
- The main dashboard does not look overcrowded.

---

## Milestone 12 — Documentation and Recruiter Polish

### Objective
Package the project for a recruiter or hiring manager.

### Tasks

- Finalize `README.md`.
- Add:
  - architecture diagram;
  - metrics table;
  - threshold scenario table;
  - dashboard screenshot;
  - model-risk limitations;
  - run instructions;
  - project outputs;
  - next steps.
- Finalize:
  - `reports/model_card.md`;
  - `reports/validation_report.md`;
  - `reports/business_value_analysis.md`.

### Gate

The README answers these questions quickly:

1. What business problem does this solve?
2. What stack was used?
3. What makes this more than a notebook?
4. How was the model evaluated?
5. How are scores converted into business actions?
6. What does the dashboard show?
7. What are the limitations?
8. How can someone run or inspect it?

---

## 5. Command-to-Artifact Map

| Command | Primary outputs |
|---|---|
| `make setup` | environment setup |
| `make ingest` | Parquet files, DuckDB staging tables, ingestion summary |
| `make features` | SQL feature tables, `mart_credit_risk_features`, feature profile |
| `make train` | model artifacts, model run summary |
| `make evaluate` | metrics, lift, calibration, threshold tables, validation figures |
| `make score` | `credit_risk_scores` |
| `make calibrate` | calibration comparison tables and selected calibration artifact |
| `make explain` | SHAP feature importance and reason-code-style outputs |
| `make dashboard-data` | Power BI-ready exports |
| `make dashboard-data-post-v1` | calibrated post-v1 Power BI-ready exports |
| `make pipeline-v1` | frozen v1 end-to-end rebuild |
| `make pipeline-post-v1` | post-v1 calibrated comparison rebuild |
| `make test` | passing pytest suite |

---

## 6. Database Output Contract

Minimum v1 tables/views:

```text
stg_application_train
stg_application_test
stg_bureau
stg_previous_application
stg_installments_payments
f_applicant_static
f_bureau_agg
f_previous_application_agg
f_installments_agg
mart_credit_risk_features
segment_diagnostics
credit_risk_scores
model_run_summary
model_metrics_summary
model_threshold_metrics
model_lift_by_decile
model_calibration_bins
model_confusion_matrix
model_feature_importance
segment_performance_summary
```

---

## 7. Definition of Done for v1

v1 is complete when:

- raw v1 Kaggle CSV files convert to Parquet;
- DuckDB staging tables load successfully;
- SQL builds a one-row-per-applicant feature mart;
- data contract and feature tests pass;
- logistic regression baseline trains;
- LightGBM model trains;
- evaluation exports PR-AUC, ROC-AUC, Brier score, lift, calibration, and threshold metrics;
- expected-value analysis compares at least three scenarios;
- batch scoring writes `credit_risk_scores`;
- SHAP feature importance and reason-code-style outputs are generated;
- Power BI dashboard consumes exported tables;
- README includes final metrics, screenshots, limitations, and run instructions;
- the project can be run from a clean environment using documented commands.

---

## 8. Scope Controls

The v1 pipeline is complete and the post-v1 comparison now includes the monthly-history
feature families listed below. Keep the following production-style extensions out of scope
unless the project brief changes:

- FastAPI endpoint;
- Postgres support;
- MLflow;
- Spark;
- deep learning;
- heavy hyperparameter search;
- complex fairness tooling;
- deployment readiness claims.

Completed post-v1 additions:

- `bureau_balance`;
- `POS_CASH_balance`;
- `credit_card_balance`;
- richer segment diagnostics;
- validation appendix dashboard page.

---

## 9. Immediate Next Actions

1. Keep the recruiter-facing README aligned with the current runnable pipeline and curated artifacts.
2. Preserve the v1 and post-v1 command contracts in the Makefile as the main review interface.
3. Use focused tests and validation reports to guard feature grain, leakage controls, scoring schema, and dashboard exports.
4. Defer optional production extensions until the portfolio decision-support workflow remains easy to rerun and review.
