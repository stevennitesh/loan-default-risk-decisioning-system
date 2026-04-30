# Loan Default Risk Decisioning System

**Version:** 0.2 build-ready draft  
**Status:** Implementation specification  
**Owner:** Steven  
**Last updated:** 2026-04-25

---

## 1. Executive Summary

This project builds an end-to-end financial decisioning pipeline that predicts loan default risk, assigns applicants to risk-based action bands, writes batch predictions back to a database table, and visualizes threshold tradeoffs in Power BI.

The goal is not to build a production underwriting system. The goal is to demonstrate applied financial-services machine learning: SQL feature engineering, reproducible Python modeling, rigorous evaluation, explainability, business-value analysis, batch scoring, testing, and recruiter-readable documentation.

**One-line portfolio summary:**

> Built a reproducible loan default risk decisioning pipeline using public credit application data, SQL feature engineering, LightGBM modeling, SHAP explainability, batch scoring, and Power BI threshold analysis.

---

## 2. Locked v1 Decisions

These decisions are fixed for v1 so implementation can begin without scope drift.

| Area | v1 decision | Rationale |
|---|---|---|
| Dataset | Home Credit Default Risk | Strong financial decisioning fit; relational tables support SQL feature engineering |
| Database | DuckDB | Fast local analytics; easy for reviewers to run; no server setup |
| Storage | Parquet | Efficient local file format; clean raw-to-processed boundary |
| Primary model | LightGBM | Strong tabular baseline, fast training, handles missing values well |
| Baseline model | Logistic regression | Simple benchmark and calibration reference |
| v1 data scope | `application_train`, `application_test`, `bureau`, `previous_application`, `installments_payments` | Enough relational depth without overloading the first build |
| v1.1 data scope | Add `bureau_balance`, `POS_CASH_balance`, `credit_card_balance` | Adds richer history after the first pipeline works |
| Split strategy | Stratified train/validation/test split | Good first pass because the dataset does not provide a clean production timestamp |
| Primary metrics | PR-AUC, ROC-AUC, Brier score, top-decile lift, expected value | Better than accuracy for imbalanced financial outcomes |
| Threshold policy | Approve / Review / High-risk action bands | Makes the model operational rather than notebook-only |
| Dashboard scope | One polished executive page first; validation appendix second | Recruiters need a clear screenshot fast |
| Sensitive variables | Exclude clearly sensitive demographic fields from model features; use only for diagnostic segment analysis if used at all | Better model-risk posture for a credit-related project |
| Production framing | Decision-support simulation, not automated underwriting | Avoids overclaiming compliance readiness |

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
11. Add tests for feature contracts, scoring schema, threshold logic, and expected-value math.

### Non-goals

This project will not:

- claim legal or regulatory readiness for credit approval;
- automate final lending decisions;
- optimize for Kaggle leaderboard rank;
- use deep learning;
- use Spark;
- add MLflow before the core pipeline is complete;
- commit raw Kaggle data to GitHub.

---

## 5. Dataset

### Primary dataset

**Dataset:** Home Credit Default Risk  
**Source:** Kaggle public competition dataset  
**Problem type:** Supervised binary classification  
**Prediction target:** Probability that an applicant experiences repayment difficulty.

### v1 tables

| Table | v1 role |
|---|---|
| `application_train.csv` | Training applications with `TARGET` labels |
| `application_test.csv` | Unlabeled scoring population for production-like batch scoring |
| `bureau.csv` | External credit history aggregates |
| `previous_application.csv` | Prior Home Credit application behavior |
| `installments_payments.csv` | Prior repayment behavior and payment delay features |
| `HomeCredit_columns_description.csv` | Documentation reference |

### v1.1 tables

| Table | v1.1 role |
|---|---|
| `bureau_balance.csv` | Monthly bureau delinquency/status history |
| `POS_CASH_balance.csv` | Monthly POS/cash loan status history |
| `credit_card_balance.csv` | Credit card balance, utilization, and delinquency patterns |

### Entity grain

The modeling table will use one row per current loan application, keyed by:

```text
SK_ID_CURR
```

### Target definition

```text
TARGET = 1: applicant experienced repayment difficulty
TARGET = 0: applicant did not experience observed repayment difficulty
```

The positive class is expected to be relatively rare. Accuracy will not be used as the headline metric.

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

v1 uses DuckDB only. Postgres can be added later as an optional extension if the project needs a server-backed database demonstration.

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

v1.1 adds:

```text
stg_bureau_balance
stg_pos_cash_balance
stg_credit_card_balance
```

### 8.3 Feature tables

| Feature table | Description |
|---|---|
| `f_applicant_static` | Current application features, affordability ratios, external scores |
| `f_bureau_agg` | External credit history aggregates by applicant |
| `f_previous_application_agg` | Prior application counts, approval/refusal patterns, amount ratios |
| `f_installments_agg` | Payment timing, late-payment behavior, payment ratios |
| `mart_credit_risk_features` | Final one-row-per-applicant modeling table |

v1.1 adds:

| Feature table | Description |
|---|---|
| `f_bureau_balance_agg` | Monthly external-credit delinquency/status summaries |
| `f_pos_cash_agg` | POS/cash status and delinquency aggregates |
| `f_credit_card_agg` | Utilization, balance, drawdown, and delinquency aggregates |

### 8.4 Feature categories

| Category | Example features |
|---|---|
| Affordability | credit-to-income ratio, annuity-to-income ratio, goods-price-to-income ratio |
| Employment stability | days employed, employment-to-age ratio |
| Application profile | contract type, income type, education type, family status, housing type |
| Bureau history | prior credit count, active/closed count, overdue amount summaries |
| Prior applications | prior approval rate, refusal count, average requested credit amount |
| Installments | average payment delay, max payment delay, late count, payment ratio |
| External scores | anonymized external score fields and summary statistics |

### 8.5 Data leakage controls

The project will enforce these controls:

- `TARGET` is never used as a feature.
- `SK_ID_CURR` is used only as an identifier, not as a model feature.
- Train/validation/test split is created before fitting encoders, imputers, calibrators, or models.
- Any imputation, encoding, scaling, or calibration is fit only on training or validation data as appropriate.
- Feature SQL must produce one row per `SK_ID_CURR`.
- Batch scoring uses the same feature columns and transformations as training.
- Labeled holdout predictions are used for evaluation; unlabeled Kaggle test predictions are used for production-like scoring demonstration.

---

## 9. Modeling Plan

### 9.1 Split strategy

Initial split:

```text
Training: 70%
Validation: 15%
Test: 15%
```

The split will be stratified by `TARGET`.

### 9.2 Baseline model

```text
Logistic Regression
```

Purpose:

- establish a simple benchmark;
- validate preprocessing;
- provide a calibration reference;
- show incremental value from LightGBM.

### 9.3 Main model

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

### 9.4 Imbalance handling

Initial imbalance strategy:

1. Use appropriate model class weighting or scale-positive weighting.
2. Tune thresholds separately from model fitting.
3. Evaluate PR-AUC, lift, and recall at review capacity.
4. Add SMOTE/undersampling only as an experiment, not as the default.

SMOTE will be retained only if it improves validation results and does not degrade calibration or business-value analysis.

### 9.5 Calibration

Calibration candidates for future or post-v1 experiments:

- uncalibrated LightGBM risk scores, the v1 default;
- Platt scaling;
- isotonic regression.

For v1, calibration is evaluated with Brier score and calibration curves on validation/test data. Platt or isotonic calibration requires a separate implemented experiment before it can be described as fitted calibration.

---

## 10. Evaluation Plan

### 10.1 Model metrics

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

### 10.2 Lift analysis

Applicants are ranked by predicted risk score and grouped into deciles.

For each decile:

- applicant count;
- observed default rate;
- cumulative default capture rate;
- lift versus portfolio average;
- action distribution under selected thresholds.

### 10.3 Threshold analysis

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

### 10.4 Test-set reporting

The final README reports test-set results only after threshold/model choices are fixed on training/validation data.

---

## 11. Decision Policy and Expected-Value Analysis

### 11.1 Risk bands

| Score range | Risk band | Simulated action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `T_low` to `< T_high` | Medium risk | Manual review |
| `>= T_high` | High risk | Decline or high-priority review |

The selected `T_low` and `T_high` values will come from validation-set threshold analysis.

### 11.2 Starting business assumptions

These assumptions are illustrative and configurable in `configs/base.yaml`.

| Assumption | Starting value |
|---|---:|
| Expected margin per good approved loan | `$1,000` |
| Expected loss per bad approved loan | `$5,000` |
| Manual review cost | `$50` |
| Manual review capacity | `10%` of applicants |

### 11.3 Expected-value formula

```text
Expected Value =
    approved_good_loans * expected_margin_per_good_loan
  - approved_bad_loans * expected_loss_per_bad_loan
  - manual_reviews * manual_review_cost
```

### 11.4 Policy scenarios

| Scenario | Intent |
|---|---|
| Growth-oriented | Higher approval volume, more credit risk |
| Balanced | Compromise between growth, risk, and review capacity |
| Risk-averse | Lower approval volume, stronger default capture |

---

## 12. Batch Scoring Specification

### 12.1 Scoring populations

v1 will score two populations:

| Population | Has target? | Purpose |
|---|---:|---|
| Holdout test split from `application_train` | Yes | Evaluation, confusion matrix, lift, expected value |
| Kaggle `application_test` | No | Production-like unlabeled scoring demonstration |

### 12.2 Scoring command

```bash
make score
```

Equivalent module command:

```bash
python -m src.score_batch --config configs/base.yaml
```

### 12.3 Prediction table

```sql
CREATE TABLE credit_risk_scores (
    applicant_id BIGINT,
    score DOUBLE,
    risk_band VARCHAR,
    recommended_action VARCHAR,
    threshold_version VARCHAR,
    top_reason_1 VARCHAR,
    top_reason_2 VARCHAR,
    top_reason_3 VARCHAR,
    model_version VARCHAR,
    scoring_population VARCHAR,
    scored_at TIMESTAMP
);
```

### 12.4 Dashboard tables

```text
credit_risk_scores
model_metrics_summary
model_threshold_metrics
model_lift_by_decile
model_calibration_bins
model_feature_importance
segment_performance_summary
```

---

## 13. Explainability Plan

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

## 14. Power BI Dashboard Specification

### 14.1 Page 1: Decisioning Overview

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

### 14.2 Page 2: Model Validation Appendix

Optional but recommended after page 1 is polished.

| Visual | Purpose |
|---|---|
| ROC curve | General ranking performance |
| Precision-recall curve | Imbalanced-outcome performance |
| Calibration curve | Probability reliability |
| Decile table | Business-readable rank ordering |
| Segment performance table | Diagnostic performance variation |
| Missingness summary | Data-quality transparency |

---

## 15. Segment and Model-Risk Diagnostics

This project should acknowledge credit-model risk without pretending to complete a regulatory review.

v1 diagnostics:

- performance by income band;
- performance by age band;
- performance by loan amount band;
- performance by application type or contract type;
- missingness by major feature group.

Sensitive or legally risky fields should not be used casually as model drivers. If demographic fields are inspected, the README must frame the analysis as a diagnostic limitation check, not a deployment approval.

---

## 16. Testing Strategy

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
- no duplicate `applicant_id` values in a scoring population;
- expected-value calculations reconcile to assumptions;
- feature mart contains `SK_ID_CURR` and `TARGET` for training rows;
- feature mart has one row per applicant.

---

## 17. Repository Structure

```text
loan-default-decisioning/
│
├── README.md
├── PROJECT_SPEC.md
├── Makefile
├── Dockerfile
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   └── base.yaml
│
├── data/
│   ├── raw/                 # ignored by git
│   ├── parquet/             # ignored by git
│   ├── db/                  # ignored by git
│   └── sample/
│
├── sql/
│   ├── 00_create_tables.sql
│   ├── 01_load_staging.sql
│   ├── 02_feature_applicant.sql
│   ├── 03_feature_bureau.sql
│   ├── 04_feature_previous_applications.sql
│   ├── 05_feature_installments.sql
│   ├── 06_build_feature_mart.sql
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
│   └── screenshots/
│
└── models/
    └── .gitkeep
```

---

## 18. Reproducibility Interface

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
| `make train` | model artifact and training metadata |
| `make evaluate` | metrics, lift, calibration, threshold tables |
| `make score` | `credit_risk_scores` table |
| `make dashboard-data` | exported Power BI-ready CSV/Parquet tables |
| `make test` | passing pytest suite |

---

## 19. Recruiter-Facing README Requirements

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

## 20. Deliverables

| Deliverable | Description |
|---|---|
| `PROJECT_SPEC.md` | Technical plan and build contract |
| `README.md` | Recruiter-facing summary and run instructions |
| SQL feature pipeline | Reproducible SQL scripts for feature extraction |
| Training pipeline | Logistic regression baseline and LightGBM model |
| Evaluation report | Metrics, lift, calibration, threshold analysis |
| Batch scoring script | Writes scored applicants back to DuckDB |
| Prediction table | Score, band, action, model version, reason codes |
| SHAP outputs | Global and local explanation artifacts |
| Power BI dashboard | Business-facing threshold and model-performance dashboard |
| Tests | Unit tests for feature, scoring, and business logic |
| Dockerfile | Reproducible runtime environment |

---

## 21. Acceptance Criteria

A strong v1 is complete when:

- [ ] Raw Kaggle CSV files can be converted to Parquet.
- [ ] DuckDB database can be created from Parquet files.
- [ ] SQL scripts build a one-row-per-applicant feature mart.
- [ ] Feature mart includes application, bureau, previous-application, and installment features.
- [ ] Training pipeline produces a logistic regression baseline.
- [ ] Training pipeline produces a LightGBM model.
- [ ] Evaluation includes ROC-AUC, PR-AUC, Brier score, lift by decile, and confusion matrix by threshold.
- [ ] Threshold analysis compares growth-oriented, balanced, and risk-averse scenarios.
- [ ] Expected-value simulation uses explicit configurable assumptions.
- [ ] Batch scoring writes predictions to `credit_risk_scores`.
- [ ] Scoring distinguishes labeled holdout scoring from unlabeled Kaggle test scoring.
- [ ] SHAP outputs identify global drivers and applicant-level reason-code-style explanations.
- [ ] Power BI reads scored/evaluation outputs.
- [ ] `pytest` tests pass for data contracts, scoring, thresholding, and expected-value logic.
- [ ] README includes final metrics, architecture diagram, dashboard screenshots, limitations, and run instructions.
- [ ] The repo can be run from a clean environment using documented commands.

---

## 22. Model Risk, Ethics, and Limitations

Required limitations section:

- This is a portfolio demonstration, not an automated underwriting system.
- The dataset is historical and anonymized.
- Public data may not reflect current lending populations, products, or policies.
- The target is a proxy for repayment difficulty, not a complete loss/default framework.
- Model outputs are insufficient for real credit decisions.
- Production lending systems require fair-lending review, monitoring, governance, explainability controls, adverse-action processes, and legal/compliance approval.
- SHAP explanations are useful for debugging and interpretation but are not automatically compliant adverse-action reason codes.
- Business-value assumptions are illustrative and configurable.
- Sensitive demographic variables should not be used as casual model features.

---

## 23. v1 Build Sequence

1. Create repo skeleton, config, Makefile, Dockerfile, and `.gitignore`.
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
17. Add README screenshots and final results.
18. Add validation appendix if time allows.

---

## 24. Success Standard

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
