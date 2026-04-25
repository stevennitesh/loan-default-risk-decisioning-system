# Loan Default Risk Decisioning System — Testing Plan

**Version:** 0.1  
**Status:** Pre-build testing plan  
**Owner:** Steven  
**Aligned spec:** `PROJECT_SPEC.md` v0.3.1  
**Last updated:** 2026-04-25

---

## 1. Purpose

This testing plan defines how the project will verify that the data pipeline, feature engineering, model scoring, threshold policy, expected-value calculations, and dashboard exports behave correctly.

Testing is not the same as model validation. This plan checks whether the system was implemented correctly. The validation plan checks whether the model and decisioning outputs are credible.

---

## 2. Testing Principles

1. **Test business-critical logic, not every line.** Focus on transformations, contracts, thresholds, scoring, and value calculations.
2. **Use small synthetic fixtures.** Unit tests should not require the full Kaggle dataset.
3. **Keep raw data out of Git.** Tests should run without proprietary or downloaded Kaggle files.
4. **Make failures specific.** A failed test should clearly identify the broken contract.
5. **Test before modeling.** Feature and data-contract tests must pass before training is trusted.
6. **Protect against silent leakage.** Excluded fields must be tested explicitly.
7. **Test exported artifacts.** Power BI-facing tables need schema and reconciliation checks.

---

## 3. Test Pyramid

```text
High volume
┌──────────────────────────────────────────┐
│ Unit tests                               │
│ - thresholding                           │
│ - expected value                         │
│ - feature calculations                   │
│ - config parsing                         │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Data contract and integration tests      │
│ - DuckDB staging                         │
│ - feature mart                           │
│ - scoring output                         │
│ - dashboard exports                      │
└──────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Pipeline smoke tests                     │
│ - make ingest/features/train/evaluate    │
│ - synthetic or tiny sample data          │
└──────────────────────────────────────────┘
Low volume
```

---

## 4. Test Files

| Test file | Primary purpose |
|---|---|
| `tests/test_config.py` | Config can be parsed; required sections exist |
| `tests/test_data_contract.py` | Required tables, columns, keys, row grain, excluded fields |
| `tests/test_feature_sql.py` | Representative SQL feature calculations on sample data |
| `tests/test_split_preprocessing.py` | Split integrity and preprocessing leakage checks |
| `tests/test_threshold_policy.py` | Risk-band and action assignment logic |
| `tests/test_expected_value.py` | Business-value formula and scenario calculations |
| `tests/test_scoring_schema.py` | Score output schema, ranges, uniqueness, scoring population labels |
| `tests/test_explainability.py` | Reason-code outputs exclude diagnostic-only fields |
| `tests/test_dashboard_exports.py` | Dashboard files exist and match required schema |
| `tests/test_cli_smoke.py` | Make/module commands run on tiny sample data where feasible |

The initial version can start with the five core files already defined in the spec, then add the remaining tests as implementation matures.

---

## 5. Fixtures and Test Data

### 5.1 Fixture strategy

Use small synthetic dataframes or tiny CSV/Parquet files under:

```text
tests/fixtures/
data/sample/
```

The fixture data should mimic the key columns and relationships of the Home Credit tables without including the full dataset.

### 5.2 Minimum fixture tables

```text
sample_application_train
sample_application_test
sample_bureau
sample_previous_application
sample_installments_payments
```

### 5.3 Fixture requirements

The fixture set should include:

- at least one applicant with multiple bureau records;
- at least one applicant with no bureau records;
- at least one applicant with multiple previous applications;
- at least one applicant with installment payments on time;
- at least one applicant with late installment payments;
- at least one labeled positive target and one labeled negative target;
- categorical missingness;
- numeric missingness;
- a diagnostic-only field such as `CODE_GENDER` or age-band to test exclusion.

---

## 6. Data and Ingestion Tests

### 6.1 Raw file presence

**Purpose:** Fail early if required raw files are missing.

Test expectations:

- required v1 files are configured;
- script reports missing files clearly;
- optional v1.1 files are not required for v1.

### 6.2 CSV-to-Parquet conversion

Test expectations:

- each input CSV creates one output Parquet file;
- row counts match;
- required key columns remain present;
- output paths follow config values;
- conversion does not mutate source files.

### 6.3 DuckDB staging

Test expectations:

- staging tables exist;
- staging row counts match Parquet row counts;
- key columns exist;
- target exists only in `stg_application_train`;
- application train/test grains are one row per `SK_ID_CURR`.

---

## 7. SQL Feature Tests

### 7.1 Feature mart grain

Test expectations:

- `mart_credit_risk_features` has one row per `SK_ID_CURR`;
- no duplicate applicant IDs;
- labeled rows retain `TARGET`;
- unlabeled rows do not invent target values;
- joins do not multiply application rows.

### 7.2 Applicant static features

Example expectations:

- `credit_to_income_ratio = AMT_CREDIT / AMT_INCOME_TOTAL`;
- `annuity_to_income_ratio = AMT_ANNUITY / AMT_INCOME_TOTAL`;
- divide-by-zero protection returns `NULL` or a configured safe value;
- external-score summary features handle missing values consistently.

### 7.3 Bureau aggregate features

Example expectations:

- applicant-level bureau credit count matches fixture records;
- active/closed credit counts reconcile to source rows;
- overdue amount summaries aggregate correctly;
- applicants with no bureau records receive expected null/default aggregate values.

### 7.4 Previous application features

Example expectations:

- prior application count matches fixture rows;
- approval/refusal rates are calculated correctly;
- amount-ratio features handle nulls and zeros.

### 7.5 Installment features

Example expectations:

- payment delay calculation is correct;
- late payment count matches fixture cases;
- max/average delay aggregates reconcile;
- payment ratio calculations handle missing or zero denominators.

---

## 8. Leakage and Feature-Exclusion Tests

These tests are mandatory for a credit-risk project.

### 8.1 Forbidden model feature list

The model feature set must exclude:

```text
TARGET
SK_ID_CURR
SK_ID_PREV
SK_ID_BUREAU
CODE_GENDER
DAYS_BIRTH
applicant_age_years
applicant_age_band
NAME_FAMILY_STATUS
CNT_CHILDREN if classified as diagnostic-only
CNT_FAM_MEMBERS if classified as diagnostic-only
```

The exact list should come from `configs/base.yaml`.

### 8.2 Diagnostic-only separation

Test expectations:

- diagnostic-only fields may exist in a diagnostics table;
- diagnostic-only fields do not enter the training feature list;
- SHAP/reason-code outputs cannot surface excluded fields;
- dashboard segment diagnostics are clearly separated from model features.

### 8.3 Split leakage

Test expectations:

- train/validation/test splits are disjoint by `SK_ID_CURR`;
- encoders, imputers, scalers, and calibrators are fit only on the appropriate split;
- validation/test labels are not used in training transformations;
- thresholds are selected on validation, not final test.

---

## 9. Model Pipeline Tests

### 9.1 Training smoke test

On a small fixture dataset or reduced sample:

- training command runs without error;
- model artifact is created;
- model metadata is created;
- model feature list is saved;
- prediction probabilities have correct shape and range.

### 9.2 Metric export test

Test expectations:

- `model_metrics_summary` exists;
- required metric names are present;
- metrics are numeric where expected;
- model version exists;
- split labels are present.

### 9.3 Calibration output test

Test expectations:

- `model_calibration_bins` exists;
- predicted and observed rates are bounded between 0 and 1;
- bin counts are nonnegative;
- total bin counts reconcile to evaluation population.

---

## 10. Threshold Policy Tests

### 10.1 Risk-band assignment

Given `T_low` and `T_high`:

| Score | Expected band | Expected action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `= T_low` | Medium risk | Manual review |
| between thresholds | Medium risk | Manual review |
| `= T_high` | High risk | Decline or high-priority review |
| `> T_high` | High risk | Decline or high-priority review |

Test expectations:

- `T_low < T_high`;
- every score receives exactly one band;
- null scores fail explicitly or are assigned to a configured error band;
- action labels match config.

### 10.2 Scenario tests

Test expectations:

- growth-oriented, balanced, and risk-averse scenarios exist;
- thresholds are valid for each scenario;
- scenario names are exported cleanly;
- scenario outputs reconcile with confusion matrix and expected-value calculations.

---

## 11. Expected-Value Tests

### 11.1 Formula test

Expected formula:

```text
Expected Value =
    approved_good_loans * expected_margin_per_good_loan
  - approved_bad_loans * expected_loss_per_bad_loan
  - manual_reviews * manual_review_cost
```

Test expectations:

- calculation matches hand-computed fixture examples;
- changing assumptions changes expected value predictably;
- manual review cost is subtracted only for manual-review cases;
- approved bad loans incur expected loss;
- high-risk/declined applicants are not counted as approved profit or approved loss unless a scenario explicitly defines otherwise.

### 11.2 Reconciliation test

For each threshold scenario:

- approved count + manual review count + high-risk count = total evaluated applicants;
- approved good + approved bad = approved count;
- confusion matrix counts reconcile to the selected threshold/action definition;
- expected-value output uses the same counts.

---

## 12. Scoring Output Tests

### 12.1 `credit_risk_scores` schema

Required columns:

```text
applicant_id
score
risk_band
recommended_action
threshold_version
top_reason_1
top_reason_2
top_reason_3
model_version
scoring_population
scored_at
```

Test expectations:

- all required columns exist;
- `score` is between 0 and 1;
- no duplicate `(applicant_id, scoring_population, model_version, threshold_version)` rows;
- risk bands are valid labels;
- recommended actions are valid labels;
- `scoring_population` distinguishes labeled holdout from Kaggle test scoring;
- scored timestamp is non-null.

### 12.2 Batch scoring reproducibility

Test expectations:

- scoring uses saved model artifact;
- scoring uses saved model feature list;
- scoring fails if required feature columns are missing;
- scoring does not require `TARGET` for unlabeled application test rows.

---

## 13. Explainability Tests

### 13.1 Global importance

Test expectations:

- `model_feature_importance` exists;
- feature importance rows include model version;
- feature names are not null;
- excluded diagnostic-only fields are absent.

### 13.2 Local reason-code-style fields

Test expectations:

- reason-code fields are strings or null when intentionally unavailable;
- excluded fields cannot appear in reason-code fields;
- reason-code mapping produces readable labels, not raw cryptic feature names only;
- top reasons are tied to positive risk contribution where feasible.

---

## 14. Dashboard Export Tests

Required exported tables/files:

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

Test expectations:

- all required export files exist after `make dashboard-data`;
- exported files are readable;
- required columns exist;
- scenario names match across threshold, confusion matrix, and expected-value outputs;
- decile counts reconcile to evaluation population;
- metric values match evaluation outputs.

---

## 15. CLI and Pipeline Smoke Tests

### 15.1 Local smoke commands

At minimum, these commands should run without uncaught errors once implemented:

```bash
make test
make ingest
make features
make train
make evaluate
make score
make dashboard-data
```

For CI, use synthetic or tiny sample data instead of the full Kaggle dataset.

### 15.2 Suggested CI behavior

A lightweight GitHub Actions workflow can run:

```bash
python -m pip install -r requirements.txt
pytest -q
```

Optional later:

```bash
make sample-pipeline
```

Where `sample-pipeline` runs ingestion, features, train, evaluate, and score on tiny fixture data.

---

## 16. Test Execution Order

Recommended order during development:

1. Config tests.
2. Ingestion tests.
3. DuckDB staging/data-contract tests.
4. SQL feature tests.
5. Leakage/exclusion tests.
6. Threshold and expected-value unit tests.
7. Scoring schema tests.
8. Model pipeline smoke tests.
9. Explainability tests.
10. Dashboard export tests.
11. Full `make test` before README polish.

---

## 17. Minimum Test Suite for v1 Release

v1 should not be considered complete unless these tests pass:

- [ ] config parses and required sections exist;
- [ ] feature mart has one row per applicant;
- [ ] no duplicate applicant IDs in scoring output;
- [ ] required feature and scoring columns exist;
- [ ] model feature list excludes forbidden fields;
- [ ] threshold policy assigns exactly one band to each score;
- [ ] expected-value calculations match fixture examples;
- [ ] scores are bounded between 0 and 1;
- [ ] dashboard export files exist and are readable;
- [ ] SHAP/reason-code outputs exclude diagnostic-only fields.

---

## 18. Definition of Tested

The project is considered adequately tested for v1 when:

- all core `pytest` tests pass;
- sample or fixture-based tests can run without the full Kaggle dataset;
- data contracts pass on the real feature mart;
- scoring schema tests pass on both labeled holdout and unlabeled scoring populations;
- threshold and expected-value outputs reconcile;
- excluded fields are blocked from model training and reason-code output;
- dashboard exports load cleanly.

This is a portfolio-grade testing standard, not production bank-grade QA. The README should not claim production readiness.
