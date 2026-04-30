# Loan Default Risk Decisioning System

**Version:** 0.1 draft  
**Status:** Working specification  
**Owner:** Steven  
**Last updated:** 2026-04-24

---

## 1. Executive Summary

This project builds an end-to-end financial decisioning pipeline that predicts loan default risk, assigns applicants to risk-based action bands, writes batch predictions back to a database table, and visualizes threshold tradeoffs in Power BI.

The goal is not to create a production-ready underwriting system. The goal is to demonstrate applied financial-services machine learning: clean data engineering, SQL-based feature extraction, reproducible model training, rigorous evaluation, explainability, business-value analysis, and implementation-oriented batch scoring.

**One-line portfolio summary:**

> Built a reproducible loan default risk decisioning pipeline using public credit application data, SQL feature engineering, gradient-boosted modeling, SHAP explainability, batch scoring, and Power BI threshold analysis.

---

## 2. Business Problem

Lenders need to decide which applicants should be approved, manually reviewed, or treated as high risk while balancing portfolio growth against credit losses. A raw probability score is not enough; the score must be translated into operational decisions under business constraints such as expected loss, approval volume, and manual review capacity.

This project simulates a decision-support workflow where a machine learning model estimates repayment difficulty risk and a policy layer converts that score into business actions.

### Primary business question

> Which applicants are most likely to experience repayment difficulty, and how should the business set score thresholds to balance approval rate, default risk, review volume, and expected portfolio value?

---

## 3. Project Objectives

### Core objectives

1. Ingest public credit-risk data and store raw files in a reproducible local data layout.
2. Convert raw CSV files to Parquet for efficient analytics.
3. Load data into DuckDB or Postgres staging tables.
4. Use SQL to create applicant-level feature tables and a final feature mart.
5. Train and compare a baseline model and a boosted-tree model.
6. Evaluate model performance using ranking, classification, calibration, and business metrics.
7. Select decision thresholds using expected-value and operational tradeoff analysis.
8. Write batch predictions back to a database table.
9. Generate model explainability outputs using SHAP.
10. Build a Power BI dashboard showing threshold tradeoffs, lift, confusion matrix, risk bands, and expected business value.

### Non-goals

This project will **not** attempt to:

- build a legally compliant credit approval system;
- automate final lending decisions;
- claim production regulatory readiness;
- optimize for Kaggle leaderboard rank;
- use deep learning unless a clear business need emerges;
- use Spark unless the data volume justifies it;
- use MLflow unless the core pipeline is already complete.

---

## 4. Dataset

### Primary dataset

**Dataset:** Home Credit Default Risk  
**Source:** Kaggle public competition dataset  
**Problem type:** Supervised binary classification  
**Prediction target:** Probability that an applicant experiences repayment difficulty.

The dataset includes a main application table split into train and test files. The training file includes the `TARGET` column, while the test file does not. Related tables include historical bureau records, previous applications, repayment behavior, monthly balances, and credit card behavior.

### Important data tables

| Table | Intended use |
|---|---|
| `application_train.csv` | Current loan applications with target labels |
| `application_test.csv` | Current loan applications without target labels |
| `bureau.csv` | Previous credits reported by other financial institutions |
| `bureau_balance.csv` | Monthly status history for bureau credits |
| `previous_application.csv` | Prior Home Credit loan applications |
| `installments_payments.csv` | Repayment history for previous credits |
| `POS_CASH_balance.csv` | Monthly balance snapshots for POS/cash loans |
| `credit_card_balance.csv` | Monthly credit card balance snapshots |
| `HomeCredit_columns_description.csv` | Column descriptions |

### Entity grain

The final modeling table will use one row per current loan application, keyed by:

```text
SK_ID_CURR
```

### Target definition

The model predicts:

```text
TARGET = 1: applicant experienced repayment difficulty
TARGET = 0: applicant did not experience observed repayment difficulty
```

The positive class is expected to be relatively rare, so accuracy will not be used as the primary success metric. Evaluation will emphasize ranking quality, precision-recall behavior, lift, calibration, and threshold tradeoffs.

### Dataset limitations

- The dataset is historical and anonymized.
- The target is a proxy for repayment difficulty, not a complete credit-risk outcome framework.
- The dataset does not represent a production lending environment.
- Fair lending, adverse action, policy rules, and compliance controls are outside the scope of this portfolio project but will be acknowledged as limitations.
- Raw data will not be committed to the repository. Users must download it from Kaggle separately.

---

## 5. Technical Architecture

```text
Kaggle CSV files
   ↓
Raw local data directory
   ↓
Parquet conversion
   ↓
DuckDB/Postgres staging tables
   ↓
SQL feature extraction
   ↓
Applicant-level feature mart
   ↓
Python model training and evaluation
   ↓
Model artifact + metrics tables
   ↓
Batch scoring script
   ↓
Database prediction table
   ↓
Power BI dashboard
```

### Recommended database choice

The initial implementation will use **DuckDB** because it is lightweight, local, fast, and easy for recruiters or reviewers to run. The SQL structure should remain portable enough to migrate to Postgres later.

---

## 6. Stack Mapping

| Tool | Role in project |
|---|---|
| Python | Pipeline orchestration, modeling, evaluation, batch scoring |
| DuckDB | Local analytical database and SQL feature extraction |
| Parquet | Efficient storage for raw and processed datasets |
| pandas | Data validation, joins for small artifacts, reporting outputs |
| scikit-learn | Baseline model, preprocessing, metrics, calibration |
| LightGBM or XGBoost | Main gradient-boosted model |
| imbalanced-learn | Optional imbalance experiments if justified by validation results |
| SHAP | Feature attribution and reason-code generation |
| pytest | Tests for feature logic, scoring schema, and threshold policy |
| Docker | Reproducible execution environment |
| Power BI | Decisioning dashboard and business tradeoff visualization |

---

## 7. Data Pipeline Specification

### 7.1 Raw data layout

```text
data/
├── raw/                  # original Kaggle CSVs; ignored by git
├── parquet/              # converted Parquet files; ignored by git
├── db/                   # DuckDB database file; ignored by git
└── sample/               # optional tiny sample for tests or docs
```

### 7.2 Staging tables

Each raw dataset will be loaded into a staging table with minimal transformations.

Example staging tables:

```text
stg_application_train
stg_application_test
stg_bureau
stg_bureau_balance
stg_previous_application
stg_installments_payments
stg_pos_cash_balance
stg_credit_card_balance
```

### 7.3 Feature tables

Feature extraction will be done primarily in SQL.

Planned feature tables:

| Feature table | Description |
|---|---|
| `f_applicant_static` | Current application, income, annuity, employment, family, ownership, and external-score features |
| `f_bureau_agg` | Aggregated external credit history by applicant |
| `f_bureau_balance_agg` | Delinquency/status summaries from bureau monthly balances |
| `f_previous_application_agg` | Prior application counts, approval/refusal patterns, amount ratios |
| `f_installments_agg` | Repayment timing, missed/late payment behavior, payment ratios |
| `f_pos_cash_agg` | POS/cash loan status and delinquency aggregates |
| `f_credit_card_agg` | Credit card utilization, balance, drawdown, and delinquency patterns |
| `mart_credit_risk_features` | Final applicant-level modeling table |

### 7.4 Example feature categories

| Category | Example features |
|---|---|
| Affordability | credit-to-income ratio, annuity-to-income ratio, goods-price-to-income ratio |
| Employment stability | days employed, employment-to-age ratio |
| Application profile | contract type, income type, education type, family status, housing type |
| Bureau history | count of previous bureau credits, active/closed credit counts, overdue amount summaries |
| Prior application behavior | prior approval rate, refusal count, average requested credit amount |
| Installment behavior | average payment delay, max payment delay, late payment count, payment completion ratio |
| Credit card behavior | utilization ratio, balance trends, drawing behavior, delinquency counts |
| External scores | provided anonymized external risk score fields and summary statistics |

---

## 8. Modeling Plan

### 8.1 Train/validation/test strategy

Initial split strategy:

```text
Training set: 70%
Validation set: 15%
Test set: 15%
```

The split will be stratified by `TARGET`. If a reliable time variable can be identified and justified, a time-aware split may be considered as a later improvement.

### 8.2 Baseline model

The baseline model will be:

```text
Logistic Regression
```

Purpose:

- establish a simple benchmark;
- support calibration comparison;
- provide an interpretable reference model.

### 8.3 Main model

The primary model will be one of:

```text
LightGBM Classifier
XGBoost Classifier
```

Selection criteria:

- validation PR-AUC;
- validation ROC-AUC;
- top-decile lift;
- calibration quality;
- inference speed;
- explainability stability;
- implementation simplicity.

### 8.4 Imbalance handling

The positive class is expected to be uncommon. The project will test imbalance strategies carefully rather than assume oversampling is beneficial.

Candidate approaches:

- model class weights;
- threshold tuning;
- probability calibration;
- optional SMOTE or undersampling experiments using `imbalanced-learn`.

SMOTE will only be retained if it improves validation performance and does not distort calibration or business-value analysis.

---

## 9. Evaluation Plan

### 9.1 Model performance metrics

| Metric | Why it matters |
|---|---|
| ROC-AUC | Measures overall ranking quality |
| PR-AUC | More informative for imbalanced binary outcomes |
| Brier score | Measures probability calibration quality |
| Precision at top decile | Shows concentration of high-risk applicants |
| Recall at review capacity | Shows operational usefulness under limited review resources |
| Lift by decile | Business-friendly ranking evaluation |
| Confusion matrix by threshold | Shows decision impact at selected cutoffs |
| Expected business value | Connects model output to financial tradeoffs |

Accuracy will be reported only as a secondary metric, if at all.

### 9.2 Lift analysis

Applicants will be ranked by predicted risk score and grouped into deciles.

For each decile:

- applicant count;
- observed default rate;
- cumulative default capture rate;
- lift versus portfolio average;
- approval/review/decline action distribution.

### 9.3 Calibration analysis

Calibration will be evaluated using:

- Brier score;
- calibration curve;
- predicted versus observed default rate by score bucket.

Calibration may be improved in a future or post-v1 experiment using:

- Platt scaling;
- isotonic regression.

For v1, calibration is evaluated on validation/test data using Brier score, calibration curves, and predicted-versus-observed score buckets. Platt or isotonic calibration requires a separate implemented experiment before it can be described as fitted calibration.

---

## 10. Decision Policy and Threshold Analysis

The model will output a risk score. A decision policy will convert scores into simulated business actions.

### 10.1 Risk bands

| Score range | Risk band | Simulated action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `T_low` to `< T_high` | Medium risk | Manual review |
| `>= T_high` | High risk | Decline or high-priority review |

The exact thresholds will be chosen using validation results and business-value assumptions.

### 10.2 Business-value formula

A simple expected-value framework will be used:

```text
Expected Value =
    approved_good_loans * expected_margin_per_good_loan
  - approved_bad_loans * expected_loss_per_bad_loan
  - manual_reviews * manual_review_cost
```

Initial editable assumptions:

| Assumption | Starting value | Notes |
|---|---:|---|
| Expected margin per good approved loan | $1,000 | Placeholder; configurable |
| Expected loss per bad approved loan | $5,000 | Placeholder; configurable |
| Manual review cost | $50 | Placeholder; configurable |
| Manual review capacity | configurable | Used for operating-point analysis |

These values are not claimed to be real Home Credit economics. They are scenario assumptions used to demonstrate threshold optimization.

### 10.3 Threshold scenarios

The dashboard and evaluation report will compare at least three policy scenarios:

| Scenario | Intent |
|---|---|
| Growth-oriented | Higher approval volume, greater credit risk |
| Balanced | Compromise between approval volume and loss control |
| Risk-averse | Lower approval volume, stronger default capture |

---

## 11. Batch Scoring Specification

The project will include a batch scoring script that loads a trained model, scores a feature mart, assigns risk bands, generates reason-code fields, and writes predictions to a database table.

### 11.1 Scoring command

Target command:

```bash
make score
```

or:

```bash
python -m src.score_batch --config configs/base.yaml
```

### 11.2 Prediction table

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
    scored_at TIMESTAMP
);
```

### 11.3 Required scoring outputs

Each scored row should include:

- applicant ID;
- predicted default-risk score;
- risk band;
- recommended simulated action;
- threshold version;
- model version;
- top SHAP-based reason codes where feasible;
- timestamp.

---

## 12. Explainability Plan

SHAP will be used to generate global and local explanations.

### Global outputs

- top features by mean absolute SHAP value;
- SHAP summary plot;
- feature dependence plots for selected variables.

### Local outputs

For batch-scored applicants, the scoring process should generate the top risk-contributing features.

Example reason-code format:

```text
High credit-to-income ratio
Recent payment delays
Low external risk score
```

Reason codes will be treated as explanatory artifacts, not legally compliant adverse-action notices.

---

## 13. Power BI Dashboard Specification

### 13.1 Main page: Decisioning Overview

| Visual | Purpose |
|---|---|
| Score distribution | Shows model score spread |
| Risk band counts | Shows operational volume by action group |
| Confusion matrix | Shows classification tradeoffs at selected threshold |
| Lift chart | Shows risk concentration by decile |
| Expected value by threshold | Shows business-optimal threshold region |
| Approval/default tradeoff | Shows growth-risk balance |
| Top model drivers | Shows global explainability |

### 13.2 Optional second page: Model Validation

| Visual | Purpose |
|---|---|
| ROC curve | General ranking quality |
| Precision-recall curve | Imbalanced-class performance |
| Calibration curve | Probability reliability |
| Decile table | Business-readable rank ordering |
| Segment performance table | Checks for performance variation across major segments |
| Missingness summary | Data-quality transparency |

### 13.3 Dashboard data sources

Power BI should read from exported database tables or CSV/Parquet extracts such as:

```text
credit_risk_scores
model_threshold_metrics
model_lift_by_decile
model_calibration_bins
model_feature_importance
```

---

## 14. Testing Strategy

The project will include `pytest` tests for core logic.

### Planned test areas

| Test file | Purpose |
|---|---|
| `test_feature_sql.py` | Validate key feature calculations on small sample data |
| `test_scoring_schema.py` | Ensure scoring output has required columns and valid ranges |
| `test_threshold_policy.py` | Validate approve/review/decline assignment logic |
| `test_expected_value.py` | Validate business-value calculations |
| `test_data_contract.py` | Validate expected columns and primary keys in feature mart |

### Example test expectations

- scores must be between 0 and 1;
- every scored applicant must receive exactly one risk band;
- `T_low` must be lower than `T_high`;
- no duplicate applicant IDs in scoring output;
- feature mart must contain `SK_ID_CURR` and `TARGET` for training rows;
- expected-value calculations must reconcile to scenario assumptions.

---

## 15. Repository Structure

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
│   ├── 06_feature_credit_card.sql
│   ├── 07_build_feature_mart.sql
│   └── 08_create_score_tables.sql
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
│   ├── test_feature_sql.py
│   ├── test_scoring_schema.py
│   ├── test_threshold_policy.py
│   ├── test_expected_value.py
│   └── test_data_contract.py
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

## 16. Reproducibility Interface

The project should support these commands:

```bash
make setup
make ingest
make features
make train
make evaluate
make score
make test
```

Stretch command:

```bash
make dashboard-data
```

This should export clean dashboard-ready tables for Power BI.

---

## 17. Deliverables

| Deliverable | Description |
|---|---|
| `PROJECT_SPEC.md` | Project plan and technical contract |
| `README.md` | Recruiter-facing project summary and instructions |
| SQL feature pipeline | Reproducible SQL scripts for feature engineering |
| Training pipeline | Baseline and boosted model training scripts |
| Evaluation report | Metrics, lift, calibration, and threshold analysis |
| Batch scoring script | Writes scored applicants back to database |
| Prediction table | Applicant-level score, risk band, action, reason codes |
| SHAP outputs | Global and local explanation artifacts |
| Power BI dashboard | Business-facing threshold and model-performance dashboard |
| Tests | Unit tests for feature, scoring, and business logic |
| Dockerfile | Reproducible runtime environment |

---

## 18. Acceptance Criteria

A strong v1 is complete when:

- [ ] Raw Kaggle CSV files can be converted to Parquet.
- [ ] DuckDB database can be created from Parquet files.
- [ ] SQL scripts build a one-row-per-applicant feature mart.
- [ ] Training pipeline produces a logistic regression baseline.
- [ ] Training pipeline produces a LightGBM or XGBoost model.
- [ ] Evaluation includes ROC-AUC, PR-AUC, Brier score, lift by decile, and confusion matrix by threshold.
- [ ] Threshold analysis compares at least three business scenarios.
- [ ] Expected-value simulation uses explicit, configurable assumptions.
- [ ] Batch scoring writes predictions to `credit_risk_scores`.
- [ ] SHAP outputs identify global drivers and applicant-level reason codes.
- [ ] Power BI dashboard reads scored/evaluation outputs.
- [ ] `pytest` tests pass for scoring, thresholding, and expected-value logic.
- [ ] README includes final metrics, architecture diagram, dashboard screenshots, limitations, and run instructions.
- [ ] The repo can be run from a clean environment using documented commands.

---

## 19. Model Risk, Ethics, and Limitations

This project is a portfolio demonstration and should be framed as decision support, not automated underwriting.

Required limitations section in final README:

- The dataset is historical and anonymized.
- Public data may not reflect current lending populations or policies.
- The target definition is a proxy for repayment difficulty.
- Model outputs are not sufficient for real credit decisions.
- Production lending systems require fair-lending review, explainability controls, monitoring, governance, adverse-action processes, and legal/compliance approval.
- SHAP explanations are useful for debugging and interpretation but are not automatically compliant adverse-action reason codes.
- Business-value assumptions are illustrative and configurable.

---

## 20. Open Questions

These should be resolved before implementation begins:

1. Use DuckDB only, or include optional Postgres support?
2. Use LightGBM or XGBoost as the primary boosted model?
3. Should the first version use all related tables or begin with application + bureau + previous applications?
4. How much fairness/segment analysis should be included in v1?
5. Should the batch scoring population use `application_test.csv`, a holdout set from training, or both?
6. What assumptions should be used for expected margin, expected loss, and review cost?
7. Should the final dashboard have one polished page or one executive page plus one validation appendix?

---

## 21. Initial Build Sequence

Recommended order of execution:

1. Create repo skeleton, config, Makefile, and Dockerfile.
2. Add data ingestion and CSV-to-Parquet conversion.
3. Load Parquet files into DuckDB staging tables.
4. Build applicant-level feature mart with SQL.
5. Train logistic regression baseline.
6. Train LightGBM/XGBoost model.
7. Add evaluation metrics and lift analysis.
8. Add calibration and threshold analysis.
9. Add batch scoring table and script.
10. Add SHAP explainability outputs.
11. Export Power BI-ready tables.
12. Build dashboard.
13. Add tests.
14. Polish README with screenshots and final results.

---

## 22. Success Standard

This project succeeds if a recruiter or hiring manager can understand, within two minutes, that the work demonstrates:

- financial-services ML framing;
- SQL-based feature engineering;
- reproducible Python modeling;
- careful imbalanced classification evaluation;
- business threshold analysis;
- explainability;
- batch implementation thinking;
- clean testing and documentation.

The project should read as an applied financial ML system, not a notebook-only Kaggle exercise.
