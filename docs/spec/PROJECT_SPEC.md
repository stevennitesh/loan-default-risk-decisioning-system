# Loan Default Risk Decisioning System

**Version:** 0.3.1 final portfolio contract
**Status:** Implemented v1 contract with post-v1 comparison
**Owner:** Steven  
**Last updated:** 2026-06-01

---

## 1. Executive Summary

This project builds an end-to-end financial decisioning pipeline that predicts loan default risk, assigns applicants to risk-based action bands, writes batch predictions back to a database table, and visualizes threshold tradeoffs in Power BI.

The goal is **not** to build a production underwriting system. The goal is to demonstrate applied financial-services machine learning: SQL feature engineering, reproducible Python modeling, rigorous evaluation, explainability, business-value analysis, batch scoring, testing, and recruiter-readable documentation.

**One-line portfolio summary:**

> Built a reproducible loan default risk decisioning pipeline using public credit application data, SQL feature engineering, LightGBM modeling, SHAP explainability, batch scoring, and Power BI threshold analysis.

**Recruiter-facing claim:**

> This project simulates a credit-risk decision-support workflow. It converts public loan-application data into applicant-level risk scores, evaluates the model with imbalanced-class and business metrics, and shows how threshold choices affect approval volume, default capture, manual review load, and expected value.

---

## 2. Locked v1 Scope

These decisions are fixed for v1. Post-v1 work is limited to the comparison scope documented below; production extensions remain stretch goals.

| Area | v1 decision | Build implication |
|---|---|---|
| Dataset | Home Credit Default Risk | Public, credit-risk focused, relational tables support SQL feature engineering |
| Database | DuckDB | Local, fast, reviewer-friendly; no server setup |
| Storage | Parquet | Clean raw-to-processed boundary and efficient local analytics |
| Primary model | LightGBM | Strong tabular model, fast training, good missing-value handling, SHAP-compatible |
| Baseline model | Logistic regression | Simple benchmark, preprocessing validation, calibration reference |
| v1 data scope | `application_train`, `application_test`, `bureau`, `previous_application`, `installments_payments` | Enough relational depth without overwhelming the first build |
| Post-v1 comparison data scope | `bureau_balance`, `POS_CASH_balance`, `credit_card_balance` | Adds richer monthly history for comparison against frozen v1 |
| Split strategy | Stratified train/validation/test split from `application_train` | Metrics come only from labeled data |
| Scoring demo population | `application_test` plus labeled holdout test split | Separates model evaluation from production-like scoring |
| Primary metrics | PR-AUC, ROC-AUC, Brier score, top-decile lift, expected value | Better than accuracy for imbalanced financial outcomes |
| Threshold policy | Approve / Review / High-risk action bands | Makes the model operational rather than notebook-only |
| Dashboard scope | One polished executive page first; validation appendix second | Prioritizes a clean recruiter screenshot |
| Sensitive-variable posture | Exclude direct demographic and protected-status-like fields from v1 model features, including `CODE_GENDER`, direct age-derived fields, and marital/family-status fields; retain them only in a separate diagnostic layer if inspected | Avoids casual use of legally or ethically sensitive variables in a credit-related model while preserving limited model-risk diagnostics. |
| Production framing | Decision-support simulation, not automated underwriting | Avoids overclaiming compliance readiness |
| API | Out of scope for v1 | Batch scoring is the implementation priority |
| MLflow | Out of scope for v1 | Keep reproducibility simple with config, artifacts, and run summaries |
| Spark | Out of scope | Data size does not justify it |

---

## 3. Business Problem

Lenders need to decide which applicants should be approved, manually reviewed, or treated as high risk while balancing portfolio growth against credit losses. A raw model score is not enough; the score must be translated into operational decisions under constraints such as expected loss, approval volume, and manual review capacity.

### Primary business question

> Which applicants are most likely to experience repayment difficulty, and how should the business set risk thresholds to balance approval rate, default capture, manual review volume, and expected portfolio value?

### Decision-support workflow

```text
Applicant data
   ↓
Risk model score
   ↓
Threshold policy
   ↓
Risk band
   ↓
Simulated business action
   ↓
Power BI threshold and value analysis
```

---

## 4. Project Objectives

### Core objectives

1. Ingest public Home Credit CSV files into a reproducible local data layout.
2. Convert raw CSV files to Parquet.
3. Load Parquet files into DuckDB staging tables.
4. Use SQL to create applicant-level feature tables and a final feature mart.
5. Train a logistic regression baseline and a LightGBM model.
6. Evaluate ranking quality, class-imbalance behavior, calibration, lift, threshold outcomes, and expected business value.
7. Choose decision thresholds using validation data and explicit business assumptions.
8. Score applicants in batch and write predictions to DuckDB.
9. Generate SHAP-based global feature importance and applicant-level reason-code-style outputs.
10. Export Power BI-ready tables and build a recruiter-readable dashboard.
11. Add tests for data contracts, feature logic, scoring schema, threshold logic, and expected-value math.
12. Publish a README and model card that explain the system, results, limitations, and run instructions.

### Non-goals

This project will not:

- claim legal or regulatory readiness for credit approval;
- automate final lending decisions;
- optimize for Kaggle leaderboard rank;
- use deep learning;
- use Spark;
- add MLflow to this portfolio scope;
- build a real-time API in v1;
- commit raw Kaggle data to GitHub.

---

## 5. Dataset and Population Definitions

### Primary dataset

**Dataset:** Home Credit Default Risk  
**Source:** Kaggle public competition dataset  
**Problem type:** Supervised binary classification  
**Prediction target:** Probability that an applicant experiences repayment difficulty.

### v1 source files

| File | Role |
|---|---|
| `application_train.csv` | Labeled applications used for train/validation/test splits |
| `application_test.csv` | Unlabeled production-like scoring population |
| `bureau.csv` | External credit-history aggregates |
| `previous_application.csv` | Prior Home Credit application behavior |
| `installments_payments.csv` | Prior repayment behavior and payment-delay features |
| `HomeCredit_columns_description.csv` | Documentation reference only |

### Post-v1 comparison source files

| File | Role |
|---|---|
| `bureau_balance.csv` | Monthly bureau delinquency/status history |
| `POS_CASH_balance.csv` | Monthly POS/cash loan status history |
| `credit_card_balance.csv` | Credit-card balance, utilization, and delinquency patterns |

### Entity grain

The modeling table uses one row per current loan application, keyed by:

```text
SK_ID_CURR
```

### Target definition

```text
TARGET = 1: applicant experienced repayment difficulty
TARGET = 0: applicant did not experience observed repayment difficulty
```

The positive class is expected to be relatively rare. Accuracy will not be used as the headline metric.

### Population separation

| Population | Source | Has `TARGET`? | Purpose |
|---|---|---:|---|
| Training split | `application_train` feature mart rows | Yes | Fit preprocessing and model |
| Validation split | `application_train` feature mart rows | Yes | Tune thresholds, calibration, and model choices |
| Test split | `application_train` feature mart rows | Yes | Final labeled performance report |
| Kaggle test scoring population | `application_test` feature mart rows | No | Production-like batch scoring demo only |

**Rule:** Do not report model performance on `application_test.csv` because it has no labels. It can be used only for score distribution, risk-band volume, and production-like scoring demonstration.

---

## 6. Technical Architecture

```text
Kaggle CSV files
   ↓
data/raw
   ↓
CSV-to-Parquet conversion
   ↓
data/parquet
   ↓
DuckDB staging tables
   ↓
SQL feature tables
   ↓
mart_credit_risk_features
   ↓
Python training/evaluation pipeline
   ↓
model artifact + metrics outputs
   ↓
batch scoring script
   ↓
credit_risk_scores + dashboard tables
   ↓
Power BI dashboard
```

### Database choice

v1 uses **DuckDB only**. Postgres can be added later as an optional extension if the project needs a server-backed database demonstration.

---

## 7. Stack Mapping

| Tool | Role in project |
|---|---|
| Python | Pipeline orchestration, modeling, evaluation, batch scoring |
| DuckDB | Local analytical database and SQL feature extraction |
| Parquet | Efficient storage for raw and processed datasets |
| pandas | Data validation, small reporting tables, exported artifacts |
| scikit-learn | Baseline model, preprocessing, metrics, calibration |
| LightGBM | Main gradient-boosted model |
| imbalanced-learn | Optional imbalance experiments after baseline results |
| SHAP | Global feature attribution and local explanation artifacts |
| pytest | Tests for data contracts, scoring schema, threshold policy, expected value |
| Docker | Reproducible runtime environment |
| Power BI | Decisioning dashboard and model-performance visuals |

---

## 8. Data Pipeline Specification

### 8.1 Data layout

```text
data/
├── raw/                  # original Kaggle CSVs; ignored by git
├── parquet/              # converted Parquet files; ignored by git
├── db/                   # DuckDB database file; ignored by git
└── sample/               # tiny synthetic/sample data for tests or docs
```

### 8.2 Staging tables

Each source file is loaded into DuckDB with minimal transformation.

```text
stg_application_train
stg_application_test
stg_bureau
stg_previous_application
stg_installments_payments
```

Post-v1 comparison adds:

```text
stg_bureau_balance
stg_pos_cash_balance
stg_credit_card_balance
```

### 8.3 Feature tables

| Feature table | Description |
|---|---|
| `f_applicant_static` | Current application features, affordability ratios, external scores, non-sensitive profile variables |
| `f_bureau_agg` | External credit-history aggregates by applicant |
| `f_previous_application_agg` | Prior application counts, approval/refusal patterns, amount ratios |
| `f_installments_agg` | Payment timing, late-payment behavior, payment ratios |
| `mart_credit_risk_features` | Final one-row-per-applicant modeling table |

Post-v1 comparison adds:

| Feature table | Description |
|---|---|
| `f_bureau_balance_agg` | Monthly external-credit delinquency/status summaries |
| `f_pos_cash_agg` | POS/cash status and delinquency aggregates |
| `f_credit_card_agg` | Utilization, balance, drawdown, and delinquency aggregates |

### 8.4 Feature categories

| Category | Example features |
|---|---|
| Affordability | credit-to-income ratio, annuity-to-income ratio, goods-price-to-income ratio |
| Employment stability | days employed, employment length indicators |
| Application profile | contract type, income type, education type, housing type, and other non-excluded application fields |
| Bureau history | prior credit count, active/closed count, overdue amount summaries |
| Prior applications | prior approval rate, refusal count, average requested credit amount |
| Installments | average payment delay, max payment delay, late count, payment ratio |
| External scores | anonymized external score fields and summary statistics |

### 8.5 Feature and column exclusions

The following columns must not be used as model predictors:

| Exclusion type | Columns / examples | Reason |
|---|---|---|
| Target | `TARGET` | Label leakage |
| Identifiers | `SK_ID_CURR`, `SK_ID_BUREAU`, prior-application IDs | IDs are not behavioral predictors |
| Direct demographic and protected-status-like fields | `CODE_GENDER`, `NAME_FAMILY_STATUS` | Avoid direct use of sensitive or legally risky fields in a credit-related project |
| Direct age-derived predictors | `DAYS_BIRTH`, applicant age transforms, age bands, and age-derived ratios | Use age only for optional diagnostics, not v1 model training |
| Post-outcome artifacts | Any column derived from `TARGET` or validation/test outcomes | Leakage |

Excluded demographic and protected-status-like fields may be retained only in a separate diagnostic layer for limited segment-performance checks. This is not a claim of fair-lending compliance. It is a conservative portfolio-project modeling posture: model features and diagnostic fields must be separated so sensitive or protected-status-like variables cannot accidentally become predictors. SHAP and reason-code-style outputs must not expose excluded diagnostic-only fields as model drivers.

---

## 9. Data Leakage and Validation Controls

The project will enforce these controls:

- `TARGET` is never used as a feature.
- `SK_ID_CURR` is used only as an identifier, not as a model feature.
- Train/validation/test split is created before fitting encoders, imputers, scalers, calibrators, or models.
- Imputation, encoding, scaling, model fitting, and calibration are fit only on the appropriate training or validation data.
- Feature SQL must produce one row per `SK_ID_CURR` per population.
- Historical tables are aggregated before joining back to the application grain.
- Joins are checked for duplicate-row expansion.
- Batch scoring uses the same feature columns and preprocessing transformations as training.
- Labeled holdout predictions are used for evaluation.
- Unlabeled Kaggle test predictions are used only for production-like scoring demonstration.
- All threshold choices are made on validation data before final test reporting.

---

## 10. Feature Mart Contract

### 10.1 Required feature mart table

```text
mart_credit_risk_features
```

### 10.2 Required columns

| Column | Type | Required? | Notes |
|---|---|---:|---|
| `SK_ID_CURR` | integer | Yes | Applicant identifier |
| `TARGET` | integer/null | Yes | Present for labeled rows; null for unlabeled Kaggle test rows |
| `source_population` | string | Yes | `application_train` or `application_test` |
| engineered feature columns | numeric/categorical | Yes | Final feature set used by pipeline |

### 10.3 Required checks

- one row per `SK_ID_CURR` within each `source_population`;
- no duplicate feature names;
- no forbidden predictor columns in model feature list;
- all training rows have non-null `TARGET`;
- all Kaggle test rows have null `TARGET`;
- no columns with 100% missing values in the final model matrix unless explicitly whitelisted;
- feature list saved with the model artifact.

---

## 11. Modeling Plan

### 11.1 Split strategy

Initial split from labeled `application_train` rows:

```text
Training: 70%
Validation: 15%
Test: 15%
```

The split will be stratified by `TARGET`.

### 11.2 Baseline model

```text
Logistic Regression
```

Purpose:

- establish a simple benchmark;
- validate preprocessing;
- provide a calibration reference;
- show incremental value from LightGBM.

### 11.3 Main model

```text
LightGBM Classifier
```

Selection criteria:

- validation PR-AUC;
- validation ROC-AUC;
- top-decile lift;
- calibration quality;
- inference speed;
- implementation simplicity;
- SHAP compatibility.

### 11.4 Imbalance handling

Initial imbalance strategy:

1. Use LightGBM class weighting or `scale_pos_weight` as the first option.
2. Tune thresholds separately from model fitting.
3. Evaluate PR-AUC, lift, and recall at review capacity.
4. Add SMOTE/undersampling only as an experiment, not as the default.

SMOTE will be retained only if it improves validation results and does not degrade calibration or business-value analysis.

### 11.5 Calibration

Calibration candidates for future or post-v1 experiments:

- uncalibrated LightGBM risk scores, the v1 default;
- Platt scaling;
- isotonic regression.

For v1, calibration is evaluated with Brier score and calibration curves on validation/test data. Platt or isotonic calibration requires a separate implemented experiment before it can be described as fitted calibration.

---

## 12. Evaluation Plan

### 12.1 Model metrics

| Metric | Why it matters |
|---|---|
| ROC-AUC | General ranking quality |
| PR-AUC | Better for imbalanced binary outcomes |
| Brier score | Probability calibration quality |
| Precision at top decile | Risk concentration in highest-score group |
| Recall at review capacity | Operational usefulness when review resources are limited |
| Lift by decile | Business-friendly ranking evaluation |
| Confusion matrix by threshold | Decision impact at selected cutoffs |
| Expected business value | Connects model output to financial tradeoffs |

Accuracy may be reported in an appendix but will not be the headline result.

### 12.2 Lift analysis

Applicants are ranked by predicted risk score and grouped into deciles.

For each decile:

- applicant count;
- observed default rate;
- average predicted score;
- cumulative default capture rate;
- lift versus portfolio average;
- action distribution under selected thresholds.

### 12.3 Threshold analysis

Thresholds are evaluated on validation data and finalized before test reporting.

The threshold grid will produce:

- approval rate;
- review rate;
- high-risk action rate;
- default rate among approved applicants;
- default capture rate in review/high-risk bands;
- confusion matrix counts;
- expected value;
- manual review volume.

### 12.4 Test-set reporting

The final README reports test-set results only after threshold/model choices are fixed on training/validation data.

### 12.5 Reporting rule

The README should distinguish between:

- **model validation results** from labeled train/validation/test splits;
- **batch scoring outputs** from unlabeled `application_test` rows.

Do not mix these in one metric table.

---

## 13. Decision Policy and Expected-Value Analysis

### 13.1 Risk bands

| Score range | Risk band | Simulated action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `T_low` to `< T_high` | Medium risk | Manual review |
| `>= T_high` | High risk | Decline or high-priority review |

The selected `T_low` and `T_high` values will come from validation-set threshold analysis.

### 13.2 Starting business assumptions

These assumptions are illustrative and configurable in `configs/base.yaml`.

| Assumption | Starting value |
|---|---:|
| Expected margin per good approved loan | `$1,000` |
| Expected loss per bad approved loan | `$5,000` |
| Manual review cost | `$50` |
| Manual review capacity | `10%` of applicants |

### 13.3 Expected-value formula

```text
Expected Value =
    approved_good_loans * expected_margin_per_good_loan
  - approved_bad_loans * expected_loss_per_bad_loan
  - manual_reviews * manual_review_cost
```

### 13.4 Policy scenarios

| Scenario | Intent |
|---|---|
| Growth-oriented | Higher approval volume, more credit risk |
| Balanced | Compromise between growth, risk, and review capacity |
| Risk-averse | Lower approval volume, stronger default capture |

---

## 14. Configuration Contract

The project should centralize tunable assumptions and paths in:

```text
configs/base.yaml
```

Minimum configuration shape:

```yaml
project:
  name: loan-default-decisioning
  random_seed: 42
  data_scope_version: v1

paths:
  raw_dir: data/raw
  parquet_dir: data/parquet
  duckdb_path: data/db/credit_risk.duckdb
  model_dir: models
  report_dir: reports
  dashboard_export_dir: reports/dashboard_exports

source_files:
  application_train: application_train.csv
  application_test: application_test.csv
  bureau: bureau.csv
  previous_application: previous_application.csv
  installments_payments: installments_payments.csv

split:
  train_size: 0.70
  validation_size: 0.15
  test_size: 0.15
  stratify: true

model:
  primary_model: lightgbm
  baseline_model: logistic_regression
  use_class_weighting: true
  calibrate_probabilities: false

excluded_features:
  identifiers:
    - SK_ID_CURR
  target:
    - TARGET
  sensitive_or_protected_status_like:
    - CODE_GENDER
    - NAME_FAMILY_STATUS
    - DAYS_BIRTH
    - applicant_age_years
    - applicant_age_band
    - employment_to_age_ratio

business_assumptions:
  expected_margin_per_good_loan: 1000
  expected_loss_per_bad_loan: 5000
  manual_review_cost: 50
  manual_review_capacity_rate: 0.10

threshold_policy:
  threshold_version: threshold_v1
  scenarios:
    growth_oriented:
      threshold_low: null
      threshold_high: null
    balanced:
      threshold_low: null
      threshold_high: null
    risk_averse:
      threshold_low: null
      threshold_high: null
```

Threshold values are initially null and filled after validation-set threshold analysis.

---

## 15. Batch Scoring Specification

### 15.1 Scoring populations

v1 will score two populations:

| Population | Has target? | Purpose |
|---|---:|---|
| Holdout test split from `application_train` | Yes | Evaluation, confusion matrix, lift, expected value |
| Kaggle `application_test` | No | Production-like unlabeled scoring demonstration |

### 15.2 Scoring command

```bash
make score
```

Equivalent module command:

```bash
python -m src.score_batch --config configs/base.yaml
```

### 15.3 Prediction table

```sql
CREATE TABLE credit_risk_scores (
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
```

Notes:

- `observed_target` is populated for labeled holdout scoring and null for Kaggle test scoring.
- `scoring_population` must distinguish at least `holdout_test` and `kaggle_test`.
- `score` must be in `[0, 1]`.
- `score_decile` is calculated separately within the relevant scoring population.

---

## 16. Dashboard Output Table Contracts

Power BI should read from DuckDB exports or CSV/Parquet files generated by `make dashboard-data`.

### 16.1 `model_run_summary`

| Column | Purpose |
|---|---|
| `model_version` | Model artifact/version identifier |
| `run_id` | Unique run identifier |
| `model_type` | `logistic_regression` or `lightgbm` |
| `data_scope_version` | `v1` or a `post_v1...` comparison version |
| `train_rows` | Training row count |
| `validation_rows` | Validation row count |
| `test_rows` | Test row count |
| `feature_count` | Number of model features |
| `positive_rate_train` | Training default/repayment-difficulty rate |
| `random_seed` | Reproducibility seed |
| `created_at` | Run timestamp |

### 16.2 `model_metrics_summary`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `split` | `train`, `validation`, or `test` |
| `metric_name` | Metric name |
| `metric_value` | Metric value |
| `created_at` | Timestamp |

### 16.3 `model_threshold_metrics`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `scenario_name` | `growth_oriented`, `balanced`, or `risk_averse` |
| `threshold_low` | Low-to-review cutoff |
| `threshold_high` | Review-to-high-risk cutoff |
| `approval_rate` | Share assigned low-risk/approve |
| `manual_review_rate` | Share assigned review |
| `high_risk_rate` | Share assigned high-risk action |
| `approved_good_count` | Approved applicants with `TARGET = 0` |
| `approved_bad_count` | Approved applicants with `TARGET = 1` |
| `manual_review_count` | Manual review count |
| `high_risk_default_capture_rate` | Share of defaults captured in high-risk group |
| `expected_value` | Total expected value under scenario |
| `expected_value_per_applicant` | Expected value normalized per applicant |

### 16.4 `model_lift_by_decile`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `split` | `validation` or `test` |
| `decile` | Score decile, with 1 = highest risk |
| `applicant_count` | Applicants in decile |
| `average_score` | Average predicted score |
| `observed_default_rate` | Observed `TARGET = 1` rate |
| `portfolio_default_rate` | Overall default rate for split |
| `lift` | Decile default rate divided by portfolio default rate |
| `cumulative_default_capture_rate` | Cumulative defaults captured through decile |

### 16.5 `model_calibration_bins`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `split` | `validation` or `test` |
| `bin_id` | Calibration bin |
| `applicant_count` | Applicants in bin |
| `average_predicted_score` | Mean predicted score |
| `observed_default_rate` | Actual default rate |
| `calibration_error` | Observed minus predicted rate |

### 16.6 `model_confusion_matrix`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `split` | `validation` or `test` |
| `scenario_name` | Threshold scenario |
| `true_label` | Observed target label |
| `predicted_label` | Binary prediction used for confusion matrix |
| `count` | Count of rows |

For confusion-matrix display, the high-risk action can be treated as the positive prediction. Manual-review handling should be explicit in the evaluation report.

### 16.7 `model_feature_importance`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `feature_name` | Feature name |
| `importance_type` | `mean_abs_shap`, `gain`, etc. |
| `importance_value` | Numeric importance value |
| `rank` | Feature rank |

### 16.8 `segment_performance_summary`

| Column | Purpose |
|---|---|
| `model_version` | Model identifier |
| `split` | `validation` or `test` |
| `segment_name` | Segment dimension, e.g. income band |
| `segment_value` | Segment bucket |
| `applicant_count` | Rows in segment |
| `observed_default_rate` | Segment default rate |
| `average_score` | Average predicted score |
| `roc_auc` | Segment ROC-AUC where calculable |
| `pr_auc` | Segment PR-AUC where calculable |
| `brier_score` | Segment Brier score |

---

## 17. Explainability Plan

SHAP will be used for global and local model explanation.

### Global outputs

- top features by mean absolute SHAP value;
- SHAP summary plot;
- selected feature dependence plots;
- exported `model_feature_importance` table.

### Local outputs

For scored applicants, the pipeline will generate top reason-code-style fields.

Example:

```text
High credit-to-income ratio
Recent payment delays
Low external risk score
```

These are explanatory artifacts, not legally compliant adverse-action notices.

---

## 18. Power BI Dashboard Specification

### 18.1 Page 1: Decisioning Overview

This is the recruiter-facing page.

| Visual | Purpose |
|---|---|
| KPI cards | ROC-AUC, PR-AUC, top-decile lift, selected expected value |
| Score distribution | Shows model score spread |
| Risk band counts | Shows operational volume |
| Threshold scenario selector | Growth, balanced, risk-averse |
| Confusion matrix | Shows classification tradeoffs |
| Lift chart | Shows risk concentration by decile |
| Expected value by threshold | Shows business-optimal region |
| Approval/default tradeoff | Shows risk-growth balance |
| Top model drivers | Shows explainability |

### 18.2 Page 2: Model Validation Appendix

Optional but recommended after page 1 is polished.

| Visual | Purpose |
|---|---|
| ROC curve | General ranking performance |
| Precision-recall curve | Imbalanced-outcome performance |
| Calibration curve | Probability reliability |
| Decile table | Business-readable rank ordering |
| Segment performance table | Diagnostic performance variation |
| Missingness summary | Data-quality transparency |

### 18.3 Dashboard design rule

The main dashboard should be understandable from a screenshot. Avoid making recruiters click through multiple slicers to understand the project.

---

## 19. Segment and Model-Risk Diagnostics

This project should acknowledge credit-model risk without pretending to complete a regulatory review.

v1 diagnostics:

- performance by income band;
- performance by loan amount band;
- performance by application/contract type;
- missingness by major feature group;
- optional age-band, gender, and marital/family-status diagnostics if clearly framed as diagnostic-only and excluded from model training.

Sensitive or legally risky fields should not be used casually as model drivers. If demographic or protected-status-like fields are inspected, they should live in a separate diagnostic layer, not in the model feature matrix. The README must frame any such analysis as a diagnostic limitation check, not a deployment approval or fair-lending certification.

---

## 20. Testing Strategy

| Test file | Purpose |
|---|---|
| `test_data_contract.py` | Validate expected columns, primary keys, and no duplicate applicant IDs |
| `test_feature_sql.py` | Validate representative feature calculations on small sample data |
| `test_threshold_policy.py` | Validate approve/review/high-risk action assignment |
| `test_expected_value.py` | Validate expected-value math |
| `test_scoring_schema.py` | Validate prediction table columns, score ranges, and risk bands |

Required test expectations:

- scores are between 0 and 1;
- `T_low < T_high`;
- every scored applicant gets exactly one risk band;
- no duplicate `applicant_id` values within a scoring population;
- expected-value calculations reconcile to assumptions;
- feature mart contains `SK_ID_CURR` and `TARGET` for labeled training rows;
- feature mart has one row per applicant per source population;
- model feature list excludes identifiers, target, and v1 demographic/protected-status-like exclusions.

---

## 21. Repository Structure

```text
loan-default-decisioning/
│
├── README.md
├── Makefile
├── Dockerfile
├── requirements.txt
│
├── configs/
│   └── base.yaml
│
├── docs/
│   ├── spec/
│   │   └── PROJECT_SPEC.md
│   ├── implementation/
│   │   └── IMPLEMENTATION_PLAN.md
│   ├── testing/
│   │   └── TESTING_PLAN.md
│   └── validation/
│       └── VALIDATION_PLAN.md
│
├── data/
│   ├── raw/                 # ignored by git
│   ├── parquet/             # ignored by git
│   ├── db/                  # ignored by git
│   └── sample/
│
├── sql/
│   ├── 02_feature_applicant.sql
│   ├── 03_feature_bureau.sql
│   ├── 03b_feature_bureau_balance.sql
│   ├── 04_feature_previous_applications.sql
│   ├── 04b_feature_pos_cash.sql
│   ├── 04c_feature_credit_card.sql
│   ├── 05_feature_installments.sql
│   ├── 05b_feature_risk_pressure.sql
│   ├── 05c_feature_recency_deterioration.sql
│   ├── 05d_feature_last_k_temporal.sql
│   ├── 06_build_feature_mart.sql
│   ├── 06_build_feature_mart_v1.sql
│   └── 07_create_score_tables.sql
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── score_batch.py
│   ├── explain.py
│   └── thresholding.py
│
├── tests/
│   ├── test_data_contract.py
│   ├── test_feature_sql.py
│   ├── test_threshold_policy.py
│   ├── test_expected_value.py
│   └── test_scoring_schema.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   ├── figures/
│   ├── model_card.md
│   ├── business_value_analysis.md
│   └── validation_report.md
│
├── powerbi/
│   ├── dashboard.pbix
│   ├── dashboard_post_v1.pbix
│   └── screenshots/
│
└── models/
    └── .gitkeep
```

---

## 22. Reproducibility Interface

Required commands:

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

Expected behavior:

| Command | Output |
|---|---|
| `make ingest` | Parquet files and DuckDB staging tables |
| `make features` | `mart_credit_risk_features` table |
| `make train` | model artifact, feature list, and training metadata |
| `make evaluate` | metrics, lift, calibration, threshold tables |
| `make score` | `credit_risk_scores` table |
| `make dashboard-data` | exported Power BI-ready CSV/Parquet tables |
| `make test` | passing pytest suite |

---

## 23. Model Card Deliverable

The repo should include:

```text
reports/model_card.md
```

Required sections:

```text
Intended Use
Not Intended For
Dataset
Target Definition
Training Data and Splits
Feature Summary
Excluded Features
Model Type
Metrics
Threshold Policy
Expected-Value Assumptions
Explainability
Limitations
Monitoring Considerations
```

The model card should be brief but concrete. It should make clear that this is a portfolio decision-support simulation, not a production credit model.

---

## 24. Recruiter-Facing README Requirements

The README should answer these questions within two minutes:

1. What business problem does this solve?
2. What data and stack were used?
3. What makes this more than a notebook?
4. How was the model evaluated?
5. How are scores converted into decisions?
6. What does the Power BI dashboard show?
7. What are the limitations?
8. How can someone run or inspect the project?

Required README sections:

```text
Overview
Business Problem
Architecture
Dataset
Modeling Approach
Evaluation Results
Threshold and Business Value Analysis
Power BI Dashboard
How to Run
Repository Structure
Limitations
Next Steps
```

Required README artifacts:

- architecture diagram;
- dashboard screenshot;
- metrics table;
- threshold scenario table;
- lift chart or decile table;
- model-risk/limitations section.

---

## 25. Deliverables

| Deliverable | Description |
|---|---|
| `docs/spec/PROJECT_SPEC.md` | Technical plan and build contract |
| `README.md` | Recruiter-facing summary and run instructions |
| `configs/base.yaml` | Paths, split parameters, model settings, business assumptions |
| SQL feature pipeline | Reproducible SQL scripts for feature extraction |
| Training pipeline | Logistic regression baseline and LightGBM model |
| Evaluation report | Metrics, lift, calibration, threshold analysis |
| Batch scoring script | Writes scored applicants back to DuckDB |
| Prediction table | Score, band, action, model version, reason codes |
| SHAP outputs | Global and local explanation artifacts |
| Power BI dashboard | Business-facing threshold and model-performance dashboard |
| `reports/model_card.md` | Model purpose, metrics, thresholds, limitations |
| Tests | Unit tests for feature, scoring, and business logic |
| Dockerfile | Reproducible runtime environment |

---

## 26. Implemented Acceptance Criteria

The frozen v1 portfolio contract is complete when these remain true:

- [x] `make ingest` converts raw Kaggle CSV files to Parquet and creates DuckDB staging tables.
- [x] `make features` builds a one-row-per-applicant `mart_credit_risk_features` table.
- [x] Feature mart includes application, bureau, previous-application, and installment features.
- [x] Feature mart and model feature list exclude target, identifiers, and v1 demographic/protected-status-like exclusions.
- [x] `make train` trains a logistic regression baseline and a LightGBM model.
- [x] Model artifact includes feature list, preprocessing details, model version, and run metadata.
- [x] `make evaluate` exports ROC-AUC, PR-AUC, Brier score, lift by decile, calibration bins, and confusion matrix by threshold.
- [x] Threshold analysis compares growth-oriented, balanced, and risk-averse scenarios.
- [x] Expected-value simulation uses explicit configurable assumptions from `configs/base.yaml`.
- [x] `make score` writes predictions to `credit_risk_scores`.
- [x] Scoring distinguishes labeled holdout scoring from unlabeled Kaggle test scoring.
- [x] SHAP outputs identify global drivers and applicant-level reason-code-style explanations.
- [x] `make dashboard-data` exports Power BI-ready tables.
- [x] Power BI page 1 reads scored/evaluation outputs and has a saved screenshot in `powerbi/screenshots/`.
- [x] `make test` passes tests for data contracts, scoring, thresholding, and expected-value logic.
- [x] README includes final metrics, architecture diagram, dashboard screenshot, limitations, and run instructions.
- [x] `reports/model_card.md` exists and clearly states intended use, non-use, metrics, thresholds, and limitations.
- [x] The repo can be run from a clean environment using documented commands.

---

## 27. Model Risk, Ethics, and Limitations

Required limitations section:

- This is a portfolio demonstration, not an automated underwriting system.
- The dataset is historical and anonymized.
- Public data may not reflect current lending populations, products, or policies.
- The target is a proxy for repayment difficulty, not a complete loss/default framework.
- Model outputs are insufficient for real credit decisions.
- Production lending systems require fair-lending review, monitoring, governance, explainability controls, adverse-action processes, and legal/compliance approval.
- SHAP explanations are useful for debugging and interpretation but are not automatically compliant adverse-action reason codes.
- Business-value assumptions are illustrative and configurable.
- Direct demographic and protected-status-like variables are excluded from v1 model features and may only be used, if inspected, in a separate diagnostic layer.
- Segment diagnostics are model-risk awareness checks, not evidence of deployment approval or fair-lending compliance.

---

## 28. Build Sequence

Completed build sequence:

1. Create repo skeleton, config, Makefile, Dockerfile, `.gitignore`, and baseline docs.
2. Add data ingestion and CSV-to-Parquet conversion.
3. Load Parquet files into DuckDB staging tables.
4. Build application-level SQL features.
5. Build bureau, previous-application, and installment aggregate features.
6. Build final feature mart.
7. Add data contract tests.
8. Train logistic regression baseline.
9. Train LightGBM model.
10. Add model evaluation metrics.
11. Add lift, calibration, and threshold tables.
12. Add expected-value analysis.
13. Add batch scoring table and scoring script.
14. Add SHAP global and local outputs.
15. Export Power BI-ready tables.
16. Build dashboard page 1.
17. Add model card.
18. Add README screenshots and final results.
19. Add validation appendix.

### First build milestone

**Milestone 1: working data-to-feature pipeline**

Done when:

```text
CSV → Parquet → DuckDB staging → SQL feature mart
```

Acceptance for milestone 1:

- raw v1 files convert to Parquet;
- DuckDB staging tables exist;
- final feature mart exists;
- feature mart has one row per `SK_ID_CURR` per source population;
- row counts and duplicate checks pass;
- no modeling starts until this milestone passes.

---

## 29. Success Standard

This project succeeds if a recruiter or hiring manager can quickly see that the work demonstrates:

- financial-services ML framing;
- SQL-based feature engineering;
- reproducible Python modeling;
- careful imbalanced classification evaluation;
- calibration and threshold analysis;
- business-value thinking;
- explainability;
- batch implementation;
- testing and documentation;
- mature limitations around credit-model usage.

The project should read as an applied financial ML system, not a notebook-only Kaggle exercise.
