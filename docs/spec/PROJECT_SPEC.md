# Loan Default Risk Decisioning System

**Version:** 0.6.0 (matched correctness assessment completed 2026-10-04; historical evidence preserved)
**Status:** Verified local pipeline and matched assessment; source/temporal/native Power BI limits documented
**Owner:** Steven  
**Last updated:** 2026-10-04

---

## 1. Executive Summary

This resume portfolio presents an end-to-end financial ML workflow to recruiters and hiring managers: it ranks observed repayment-difficulty risk, assigns simulated action bands, writes batch predictions to DuckDB, and presents current matched results in an offline report and exportable charts, alongside historical Power BI demonstrations.

The goal is to demonstrate SQL feature engineering, configured Python modeling, imbalanced-outcome evaluation, interpretation, scenario utility, batch scoring, testing, and clear documentation. This is a decision-support simulation, not a production underwriting system.

### Contract and evidence status

This document owns scope and implemented public behavior. [Current evidence status](../validation/VALIDATION_PLAN.md#current-evidence-status) records the corrected features/roles, nested assessment, action semantics, and local artifact binding. The [assessment methodology](../validation/ASSESSMENT_METHODOLOGY.md) owns the explanation of fold roles, seed separation, and estimands. Historical numbers precede these corrections. Prior exploration, illustrative utility, same-host reproduction limits, and unrefreshed native dashboards still limit claims. The [remediation plan](../implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) retains historical proposals. Current explicitly authorized tuning implements inner CV as described in the methodology; capacity policy and broader lineage remain outside scope.

The [current case study](../../reports/portfolio/case_study.md) owns the reader journey: application-only versus application-and-loan-history ranking, probability checks and restrained simulated-action conclusions. Numerical assessment and contract owners remain authoritative.

**One-line portfolio summary:**

> Built a reproducible loan default risk decisioning pipeline using public credit application data, SQL feature engineering, LightGBM modeling, SHAP explainability, batch scoring, and matched applicant-group assessment and readable reporting.

**Portfolio claim:**

> This project simulates a credit-risk decision-support workflow. It converts public loan-application data into applicant-level risk scores, evaluates the model with imbalanced-class and business metrics, and shows how threshold choices affect approval volume, repayment-difficulty case capture, manual review volume, and illustrative utility.

---

## 2. Locked v1 Scope

These decisions define the local portfolio scope. Post-v1 work uses the comparison sources below; production extensions require an explicit scope change.

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
| Dashboard scope | One polished executive page first; validation appendix second | Prioritizes a clean dashboard screenshot |
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
7. Derive decision thresholds from validation-score quantiles and review-capacity
   scenarios, then evaluate those scenarios under explicit business assumptions.
8. Score applicants in batch and write predictions to DuckDB.
9. Generate SHAP-based global feature importance and applicant-level reason-code-style outputs.
10. Export Power BI-ready tables and build a business-readable dashboard.
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
(SK_ID_CURR, source_population)
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
| Calibration split (post-v1) | `application_train` feature mart rows | Yes | Fit sigmoid/isotonic transforms only; disjoint from training and selection |
| Validation split | `application_train` feature mart rows | Yes | Derive threshold scenarios and select calibration and model choices |
| Test split | `application_train` feature mart rows | Yes | Within-run reporting; historical comparison across reused experiments |
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

The local portfolio uses **DuckDB only**. Postgres or any server-backed demonstration requires an explicit change to the project scope.

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
└── db/                   # DuckDB database file; ignored by git
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
| `f_risk_pressure_features` | Configured cross-source pressure interactions |
| `f_recency_deterioration_features` | Recent-versus-older record behavior summaries |
| `f_last_k_temporal_features` | Latest three due obligations; latest three distinct applicant POS/card months; last-loan summaries |

#### Bureau historical-origin availability (0.6.0)

SQL `n_eligible_bureau` admits only finite `DAYS_CREDIT < 0`. Bureau aggregates, bureau-balance children and recency children consume this same owner, so rejected origins cannot re-enter through a child join. Separate `bureau_origin_coverage` preserves day-zero, positive and unknown-origin counts outside model features. Planned future maturity/enddate fields remain known contract attributes on eligible prior loans; positive planned dates alone are not excluded. The real source contains 25 day-zero origins, no positive/unknown origins; day zero is not proven future leakage, but lacks intraday prior-history proof. Relative cutoffs do not establish source ingestion timestamps.

#### Repayment and monthly feature semantics (0.4.0)

`n_installment_obligations` groups rows by applicant, previous loan, installment number, and schedule version. An obligation is eligible only when its key, positive owed amount, and pre-application due day are known and consistent, with no competing version for the same loan/installment number. Ambiguous schedules are counted and excluded from repayment statistics; version numbers do not establish amendment chronology.

Known dated payments strictly before application are accumulated within each obligation. Owed amount is counted once; the first cumulative full-payment day determines completed delay. Incomplete obligations have ongoing arrears age rather than an invented completion delay. Unknown payment evidence yields null repayment statistics, not zero delay or full payment. Future-due obligations and application-day/future cashflows are excluded. Identical payment rows are retained because no unique payment-event ID supports safe deduplication. Repayment totals and ratios use the same known obligation support.

Support fields expose total, ambiguous, unknown-payment, known-payment, and ongoing-arrears obligation counts plus average arrears age. Legacy `late_payment_count`, underpayment, and delay field names now summarize obligations, not payment rows; latest-installment fields refer to the latest due obligation. `installment_payment_count` still counts eligible dated payment records.

POS/card last-three windows contain distinct applicant months, including all accounts within each month. Only `MONTHS_BALANCE <= -1` is eligible, as a conservative snapshot cutoff. A month is delinquent if any account has known positive delinquency; it is on-time only if every account has known nonnegative zero delinquency. Otherwise its status is unknown and omitted from rate denominators. Bureau `X`/unknown status is also excluded from delinquency-rate denominators. Duplicate account/month rows fail rather than silently multiplying support. Monetary/count ratios use matched known operands; card utilization is pooled eligible balance divided by eligible limit. Legacy lifetime/recent `*_month_count` fields still count account-month records; applicant-month delinquency rates and last-three windows have the distinct grain above.

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

These are required controls. Current per-run pipeline checks implement several of them; cross-run selection and persisted-artifact gaps remain open as described in the validation owner:

- `TARGET` is never used as a feature.
- `SK_ID_CURR` is used only as an identifier, not as a model feature.
- Train/validation/test split is created before fitting encoders, imputers, scalers, calibrators, or models.
- Imputation, encoding, scaling, model fitting, and selection importance use only training rows; post-v1 calibrators fit only their reserved calibration role.
- Feature SQL must produce one row per `SK_ID_CURR` per population.
- Historical tables are aggregated before joining back to the application grain.
- Joins are checked for duplicate-row expansion.
- Batch scoring uses the same feature columns and preprocessing transformations as training.
- Labeled holdout predictions are used for evaluation.
- Unlabeled Kaggle test predictions are used only for production-like scoring demonstration.
- All threshold scenarios are derived from validation-score quantiles before
  held-out test reporting.

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

v1 uses these three stratified roles. Post-v1 divides the configured 15% validation budget equally: training 70%, calibration fitting 7.5%, selection validation 7.5%, and historical comparison test 15%. Counts can differ by rounding. Every role must contain both classes; insufficient support fails rather than reusing another role. All preprocessing and feature ranking fit only training rows. Calibration fits only its reserved role; validation selects models/calibration and derives scenarios. The split summary records all four post-v1 roles, while model-performance reports retain train/validation/test rows.

If a model artifact already exists, retraining preserves its test applicant IDs across seed changes. Stability seeds redivide only the saved development roles and recompute training-only feature rankings. This is a local reuse boundary, not an untouched lockbox or a separately persisted assessment quarantine. Deleting the model artifact removes that membership reference.

### 11.1a Nested assessment

The ordinary pipeline fits the configured dashboard comparison. The separate
`make assess-post-v1` command evaluates a declared LightGBM selection procedure
using five stratified outer folds over development applicants. Its inner
70%/15%/15% fitting/calibration/selection roles are fractions of outer training,
not of the full dataset. Current v3 uses three inner CV folds within base fitting, with separate stopping subsets; r2 used a reserved selection holdout. Outer metrics do not select or promote the workflow and do not replace the
dashboard model's metrics. Matched independent controls assess training prevalence, tuned logistic regression (fixed C=1 in r2) and application-only LightGBM on the exact same folds/roles. Outer results never select a model family or promote a workflow.

Feature rankings average raw-column ranks across independently configured model
seeds, using only fitting rows. `project.split_seed` and `project.model_seed` split
ordinary partition/model randomness; `assessment.split_seeds` defines outer-fold
repeats. Exact settings and local role IDs are recorded before candidate fitting.
See the [methodology owner](../validation/ASSESSMENT_METHODOLOGY.md) for the workflow,
artifacts, and interpretation. The named [correctness assessment](../../reports/correctness_20261004/assessment_report.md) identifies completed real-data evidence separately from historical snapshots.

### 11.1b Evaluation ties and diagnostics

Top-rate precision/recall/lift use uniform fractional membership across a boundary score tie, with selected mass `ceil(n*rate)`. Equal scores have equal expected membership independent of labels and row order. This is an evaluation expectation, not a queue policy. Reliability bins keep score groups together by empirical midrank; ten nominal bins may be empty and counts must be shown. Target-blind score/ID display rank bins remain separate. Log loss accompanies Brier, average precision and ROC-AUC. Prespecified utility sensitivities use frozen raw thresholds; history-quality segments and matched fold differences are descriptive, never selection inputs. Training prevalence has no meaningful quantile policy threshold.

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

Implemented family selection compares validation average precision (legacy PR-AUC). Current LightGBM joint tuning first checks mean inner Brier/log loss against fitting-prevalence tolerances, then sorts by inner average precision, top-decile lift, top-score capture, ROC-AUC, and lower Brier score. The all-failed fallback retains ranking and records failed probability acceptance; legacy preset tuning filters degenerate scores before its ranking. Expected value, inference speed, simplicity, and SHAP compatibility are review considerations, not additional automatic selection criteria. Historical fits left row bagging disabled; the current builder sets `subsample_freq=1` so configured `subsample` takes effect. New corrected comparisons are recorded separately in the named correctness assessment; they do not rewrite historical metrics.

### 11.4 Imbalance handling

Current imbalance strategy:

1. Jointly search unweighted, lighter and balanced LightGBM `scale_pos_weight` fractions using fitting labels alone.
2. Derive threshold scenarios separately from model fitting.
3. Evaluate PR-AUC, lift, and recall at review capacity.
4. Resampling is not implemented as part of the selected pipeline; a future experiment would need separate development-only evidence.

### 11.5 Calibration

Implemented calibration options:

- uncalibrated LightGBM risk scores, the v1 default;
- Platt scaling;
- isotonic regression.

v1 does not fit a calibration layer. Post-v1 fits sigmoid/isotonic on reserved calibration rows and compares them with raw scores on disjoint selection-validation rows. Each candidate must improve validation Brier by at least 0.0005 before selection; sigmoid is preferred among eligible methods within 0.0005 of the best method. Historical experiments shared fit/selection rows and selected sigmoid; their recorded gains are exploratory. Current validation gains remain selection evidence. Named outer assessment and saved-model comparison evidence are separate, with no untouched-lockbox claim.

Feature-subset experiments aggregate encoded LightGBM gain to raw fields, rank each model-seed repeat, and use mean ranks with deterministic name-based tie breaking. They do not read reporting SHAP rankings. Probability-quality metrics use the selected calibration; threshold scenario utility uses raw scores, matching batch scoring. Ranking seeds are independent of partition/model seeds.

---

## 12. Evaluation Plan

### 12.1 Model metrics

| Metric | Why it matters |
|---|---|
| ROC-AUC | General ranking quality |
| PR-AUC (average precision) | Ranking quality for the imbalanced outcome; not trapezoidal PR-curve area |
| Brier score | Overall probability quality; inspect calibration bins separately |
| Precision at top decile | Risk concentration in highest-score group |
| Recall at review capacity (legacy name) | Capture among the highest scores at a reference rate, not actual middle-review capture |
| Lift by decile | Business-friendly ranking evaluation |
| Confusion matrix by threshold | Decision impact at selected cutoffs |
| Expected business value (legacy name) | Retrospective scenario utility under explicit weights, not actual profit |

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

Thresholds are derived from validation-score quantiles that encode approval,
reference review-rate, and high-risk volume scenarios. Scenario outcomes are
then evaluated on validation data under the illustrative business assumptions
before held-out test reporting. Expected value is an evaluation output, not the
threshold optimization rule.

Fixed validation quantiles do not guarantee a hard review limit on a new population or under ties. Current high-band action labels and review costing also differ; do not claim an operationally constrained policy until they agree.

The threshold scenario analysis will produce:

- approval rate;
- review rate;
- high-risk action rate;
- default rate among approved applicants;
- high-risk default capture rate; actual middle-review capture is not the legacy top-score capture metric;
- confusion matrix counts;
- expected value;
- manual review volume.

### 12.4 Test-set reporting

Within a run, test metrics are reported after validation-based choices. Across the historical experiments the original test applicants were reused in fitting and reporting-population SHAP fed selection, so the README must describe these values as historical comparisons, not an independent final test.

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
| `>= T_high` | High risk | Simulated decline (`simulated_decline`) |

The selected `T_low` and `T_high` values come from validation-score quantiles
for the configured capacity scenarios. Expected value and other business
outcomes are used to interpret the resulting scenarios, not to optimize the
cutoffs.

The high band issues no simulated loan and receives zero modeled value/cost. Only the middle band incurs manual-review cost; no reviewer success rate or subsequent lending outcome is assumed. These illustrative actions align with the existing utility formula and do not form an operational underwriting policy.

### 13.2 Starting business assumptions

These assumptions are illustrative and configurable in `configs/base.yaml`.

| Assumption | Starting value |
|---|---:|
| Expected margin per good approved loan | `1000` utility units |
| Expected loss per bad approved loan | `5000` utility units |
| Manual review cost | `50` utility units |
| Review-rate scenario reference | `10%` of applicants; not an enforced cap |

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

The YAML below is an illustrative v1-shaped example, not the authoritative runnable config. Use [configs/v1.yaml](../../configs/v1.yaml) or [configs/post_v1.yaml](../../configs/post_v1.yaml) for scoped runs. [configs/base.yaml](../../configs/base.yaml) defaults to post-v1 feature scope with separate output paths. Do not copy this abbreviated example over an active config.

The project should centralize tunable assumptions and paths in:

```text
configs/base.yaml
```

Minimum configuration shape:

```yaml
project:
  name: loan-default-decisioning
  random_seed: 42
  split_seed: 42
  model_seed: 42
  data_scope_version: post_v1_011_last_k_temporal

paths:
  raw_dir: data/raw
  parquet_dir: data/parquet
  duckdb_path: data/db/credit_risk.duckdb
  model_dir: models
  report_dir: reports
  dashboard_export_dir: reports/dashboard_data

source_files:
  application_train: application_train.csv
  application_test: application_test.csv
  bureau: bureau.csv
  bureau_balance: bureau_balance.csv
  pos_cash_balance: POS_CASH_balance.csv
  credit_card_balance: credit_card_balance.csv
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
  lightgbm_tuning:
    enabled: true
    mode: bounded_inner_cv
    max_candidates: 24
    inner_folds: 3

excluded_features:
  identifiers:
    - SK_ID_CURR
    - SK_ID_PREV
    - SK_ID_BUREAU
  target:
    - TARGET
  sensitive_or_protected_status_like:
    - CODE_GENDER
    - NAME_FAMILY_STATUS
    - DAYS_BIRTH
    - applicant_age_years
    - applicant_age_band
    - employment_to_age_ratio
    - CNT_CHILDREN
    - CNT_FAM_MEMBERS

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

Threshold values are initially null and filled from validation-score quantiles
for the configured capacity scenarios. `project.random_seed` is the legacy
fallback when a separate split/model seed is absent. Scoped post-v1 and base
configs also define `feature_selection.ranking_seeds` and `assessment` settings;
their meaning and defaults belong to the
[assessment methodology](../validation/ASSESSMENT_METHODOLOGY.md#reserved-calibration-seeds-and-interpretation).

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
make score CONFIG=configs/v1.yaml
```

Equivalent module command:

```bash
python -m src.score_batch --config configs/v1.yaml
```

### 15.3 Prediction table

```sql
CREATE TABLE credit_risk_scores (
    applicant_id BIGINT,
    scoring_population VARCHAR,
    observed_target INTEGER,
    score DOUBLE,
    raw_risk_score DOUBLE,
    calibrated_risk_score DOUBLE,
    calibration_method VARCHAR,
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
- `score`/`raw_risk_score` drive the existing rank policy. `calibrated_risk_score` is a separate score view, with `calibration_method` identifying its treatment. Exact column order is owned by [src/report_contracts.py](../../src/report_contracts.py), with SQL in [07_create_score_tables.sql](../../sql/07_create_score_tables.sql).

---

## 16. Dashboard Output Table Contracts

Power BI should read from the CSV export directories generated by `make dashboard-data` and `make dashboard-data-post-v1`.

The tables below describe field meaning, not a duplicate exhaustive schema. [src/report_contracts.py](../../src/report_contracts.py) owns exact columns; [src/dashboard_exports.py](../../src/dashboard_exports.py) owns the eight exported tables. `model_run_summary` is a database/runtime report and is not part of that eight-file dashboard bundle. Post-v1 export recomputes calibrated selected-model probability metrics and segment diagnostics; raw evaluation rows and calibrated dashboard rows are different score views.

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
| `random_seed` | Legacy summary column for the model-fitting seed; fitted artifacts record `split_seed` and `model_seed` separately |
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
| `segment_name` | Implemented diagnostic dimension, e.g. `applicant_age_band` |
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
- feature dependence plots are an optional extension, not a current generated output;
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

This is the executive overview page.

| Visual | Purpose |
|---|---|
| KPI cards | ROC-AUC, PR-AUC, top-decile lift, selected expected value |
| Score distribution | Shows model score spread |
| Risk band counts | Shows operational volume |
| Threshold scenario selector | Growth, balanced, risk-averse |
| Confusion matrix | Shows classification tradeoffs |
| Lift chart | Shows risk concentration by decile |
| Expected value by threshold scenario | Shows illustrative business tradeoffs across the validation-derived scenarios |
| Approval/default tradeoff | Shows risk-growth balance |
| Top model drivers | Shows explainability |

### 18.2 Page 2: Model Validation Appendix

Both saved PBIX reports include this page. The visual tables describe design intent; they are not proof that every listed chart is present or numerically reconciled. Existing screenshots show the historical post-v1 report.

| Visual | Purpose |
|---|---|
| ROC curve | General ranking performance |
| Precision-recall curve | Imbalanced-outcome performance |
| Calibration curve | Probability reliability |
| Decile table | Business-readable rank ordering |
| Segment performance table | Diagnostic performance variation |
| Missingness summary | Data-quality transparency |

### 18.3 Dashboard design rule

The main dashboard should be understandable from a screenshot. Avoid making readers click through multiple slicers to understand the project.

---

## 19. Segment and Model-Risk Diagnostics

This project should acknowledge credit-model risk without pretending to complete a regulatory review.

Implemented dashboard segment diagnostics are `CODE_GENDER`, `NAME_FAMILY_STATUS`, `applicant_age_band`, `CNT_CHILDREN`, and `CNT_FAM_MEMBERS`, as owned by [src/dashboard_segments.py](../../src/dashboard_segments.py). Income, loan-amount, contract-type, and missingness-group segment exports are possible extensions, not current outputs. Missingness is also recorded separately in feature inventory reports.

Sensitive or legally risky fields should not be used casually as model drivers. If demographic or protected-status-like fields are inspected, they should live in a separate diagnostic layer, not in the model feature matrix. The README must frame any such analysis as a diagnostic limitation check, not a deployment approval or fair-lending certification.

---

## 20. Testing Strategy

| Test file | Purpose |
|---|---|
| `test_config.py` | Validate config shape and reproducibility profiles |
| `test_ingest.py` | Validate raw-to-staging ingestion contracts |
| `test_data_contract.py` | Validate expected columns, primary keys, and no duplicate applicant IDs |
| `test_feature_sql.py` | Validate representative feature calculations on synthetic fixtures |
| `test_train.py` | Validate model training artifacts, split summaries, and metrics |
| `test_evaluate.py` | Validate evaluation tables, lift, calibration, and selected model checks |
| `test_threshold_policy.py` | Validate approve/review/high-risk action assignment |
| `test_expected_value.py` | Validate expected-value math |
| `test_scoring_schema.py` | Validate prediction table columns, score ranges, and risk bands |
| `test_calibrate.py` | Validate calibration comparison outputs and artifact selection |
| `test_explain.py` | Validate SHAP/reason-code outputs exclude diagnostic-only fields |
| `test_dashboard_exports.py` | Validate Power BI export schemas and dashboard-ready summaries |
| `test_powerbi_artifacts.py` | Validate committed Power BI report artifacts |

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

Use [the README repository guide](../../README.md#repository-guide) for the maintained layout. The main ownership boundaries are:

- `configs/v1.yaml` and `configs/post_v1.yaml`: explicit feature/output scopes; `configs/base.yaml` supplies separate defaults.
- `sql/`: source aggregation and mart construction; `src/`: orchestration, modeling, metrics, scoring, interpretation, and exports.
- `tests/`: synthetic fixture checks; no downloaded sample dataset is required.
- `docs/spec/`, `docs/implementation/`, `docs/testing/`, and `docs/validation/`: scope, build history/proposals, checks, and evidence requirements.
- `data/`, `models/`, and scoped runtime reports: local generated artifacts, ignored by Git.
- `reports/experiments/` and `reports/model_card.md`: curated historical evidence and its interpretation.
- `powerbi/`: saved report binaries and historical screenshots.

The active agent reading path is [AGENTS.md](../../AGENTS.md). Domain and accepted architecture decisions remain at their existing owners; there is no mandatory separate engineering contract, domain template, ADR tree, or bootstrap record.

---

## 22. Reproducibility Interface

The [Makefile](../../Makefile) owns the interface. See [the README run guide](../../README.md#how-to-run) for explicit scopes and Windows `PYTHON=python`. `make setup` installs into the selected interpreter; it does not create an environment. Scoped pipelines regenerate local artifacts from raw Kaggle data; they do not guarantee exact historical metrics or refresh PBIX visuals. Export targets use existing artifacts without retraining and can recompute probability-quality/segment views. `requirements.lock` pins the dependency closure used for the named correctness run and same-host clean reproduction.

Required commands:

```bash
make setup
make lint
make format-check
make ingest
make features
make train
make evaluate
make score
make calibrate
make explain
make dashboard-data
make dashboard-data-post-v1
make pipeline-v1
make pipeline-post-v1
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
| `make calibrate` | calibration comparison tables and selected calibration artifact |
| `make explain` | SHAP feature importance and applicant reason-code-style outputs |
| `make dashboard-data` | exported v1 Power BI-ready CSV tables |
| `make dashboard-data-post-v1` | exported calibrated post-v1 Power BI-ready CSV tables |
| `make pipeline-v1` | frozen v1 end-to-end rebuild |
| `make pipeline-post-v1` | post-v1 calibrated comparison rebuild |
| `make test` | passing pytest suite |

---

## 23. Model Card Deliverable

The repo should include:

```text
reports/model_card.md
```

Required information (headings may vary):

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

## 24. Portfolio README Requirements

The README should answer these questions within two minutes:

1. What business problem does this solve?
2. What data and stack were used?
3. What makes this more than a notebook?
4. How was the model evaluated?
5. How are scores converted into decisions?
6. What does the Power BI dashboard show?
7. What are the limitations?
8. How can someone run or inspect the project?

Required README information (headings may vary):

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
| `README.md` | Portfolio summary and run instructions |
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
| Dockerfile | Python 3.12 test container; defaults to `make test`, dependencies are not locked |

---

## 26. Implemented Acceptance Criteria

This is a historical implementation inventory, not current validation sign-off. Checked items record implemented paths and curated artifacts. Current acceptance also requires the pending correctness gates in the validation owner:

- [x] `make ingest` converts raw Kaggle CSV files to Parquet and creates DuckDB staging tables.
- [x] `make features` builds a one-row-per-applicant `mart_credit_risk_features` table.
- [x] Feature mart includes application, bureau, previous-application, and installment features.
- [x] Model feature list excludes target, identifiers, and demographic/protected-status-like exclusions. The mart retains its identifier, population, and labeled target for joins/evaluation; diagnostics remain separate.
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
- [x] The named assessment records a clean locked same-host fold reproduction and CSV bundle reconciliation. Native PBIX refresh and opaque DAX/relationship certification remain unverified.

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

This project succeeds if a reader can quickly see that the work demonstrates:

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

## Current tuning protocol, 2026-10-04

Current ordinary/post-v1 and assessment callers share `src/tuning.py` and
`nested_inner_cv_v3`. The methodology owns the exact 24 joint-candidate budget,
parameter bounds, probability acceptance, fixed rounds and seed roles. Final
base/calibration/selection roles remain disjoint; outer quality never promotes a
recipe. Tuned logistic and fixed historical logistic have distinct workflow
identities. Static public offsets/IDs do not identify calendar application,
feature availability or label maturity; no chronological splitter is justified.
