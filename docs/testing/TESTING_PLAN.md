# Loan Default Risk Decisioning System — Testing Plan

## Current presentation verification

`tests/test_assessment_model_inputs.py` checks target-blind deterministic sampling, one-hot ownership despite category/input prefix collisions, cumulative absolute grouping and separate signed additivity failure. The portfolio tests reject mismatched model-input identities, applicant-level columns, changed summaries, inconsistent segment populations and incomplete utility grids. Static-bundle tests reject unexpected files and changed hashes; all eight charts and local downloads render without source data or models. These are synthetic/anonymous verification checks, distinct from the completed bounded real-data interpretation.

`tests/test_class_weighting_ablation.py` checks that the diagnostic changes only class weighting and rejects incompatible or overlapping applicant roles. `tests/test_class_weighting_report.py` independently reconciles anonymous fold/paired summaries and rejects stale run identities, bad reference reproduction and applicant-level input. The portfolio test also checks the controlled-comparison chart and explanation. The completed real-data diagnostic is separate evidence; these tests require only synthetic structures and curated anonymous aggregates.

`tests/test_portfolio_report.py` verifies anonymous aggregate inputs, numerical fold-summary reconciliation, matched applicant counts, preserved metric keys and standalone offline rendering without raw data/models. `make portfolio` regenerates final presentation only. Fixture workflows exercise runtime reports/charts and preserve export schemas. These checks do not establish future-cohort performance or native Power BI refresh.

Presentation checks also cover portable case-study/method links and the distinction between renamed exported provenance and unchanged scientific source evidence. Inspect the rendered desktop and phone report for legible chart labels, stacked portrait panels on phones, in-report enlargement and keyboard return, expandable evidence/direct section links, and the default reading path. Both chart layouts must stay embedded for offline reading and preserve plotted values, axis limits and units. A build or HTML string assertion does not replace this browser inspection.


**Version:** 0.1  
**Status:** Existing fixture coverage and required checks; missing correctness regressions identified
**Owner:** Steven  
**Aligned spec:** [PROJECT_SPEC.md](../spec/PROJECT_SPEC.md)
**Last updated:** 2026-10-04

---

## 1. Purpose

This testing plan defines how the project will verify that the data pipeline, feature engineering, model scoring, threshold policy, expected-value calculations, and dashboard exports behave correctly.

Testing is not the same as model validation. This plan checks whether the system was implemented correctly. The validation plan checks whether the model and decisioning outputs are credible.

This document records current coverage and intended test requirements, not proof that every requirement below has a regression test. Passing fixtures do not certify scientific correctness. See [current evidence status](../validation/VALIDATION_PLAN.md#current-evidence-status) and the missing regression cases below before describing the portfolio as validated.

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
│ - synthetic fixture data                 │
└──────────────────────────────────────────┘
Low volume
```

---

## 4. Test Files

| Test file | Primary purpose |
|---|---|
| `tests/test_config.py` | Config can be parsed; required sections exist |
| `tests/test_ingest.py` | Raw CSVs are converted to Parquet and DuckDB staging tables |
| `tests/test_data_contract.py` | Required tables, columns, keys, row grain, excluded fields |
| `tests/test_feature_sql.py` | Representative SQL feature calculations on synthetic fixtures |
| `tests/test_train.py` | Training artifacts, split summaries, and model comparison outputs |
| `tests/test_evaluate.py` | Evaluation metrics, lift, calibration, and selected-model checks |
| `tests/test_threshold_policy.py` | Risk-band and action assignment logic |
| `tests/test_expected_value.py` | Business-value formula and scenario calculations |
| `tests/test_scoring_schema.py` | Score output schema, ranges, uniqueness, scoring population labels |
| `tests/test_calibrate.py` | Calibration experiment outputs and selected-method artifact |
| `tests/test_explain.py` | Reason-code outputs exclude diagnostic-only fields |
| `tests/test_feature_selection.py` | Feature selection experiment reporting and named outputs |
| `tests/test_model_stability.py` | Seed-stability experiment outputs and selection logic |
| `tests/test_dashboard_exports.py` | Dashboard export files and schemas |
| `tests/test_powerbi_artifacts.py` | Committed Power BI report artifacts |
| `tests/test_repo_contract.py` | Repository command, path, SQL, and ignore contracts |
| `tests/test_methodology_boundaries.py` | Reserved calibration, training-only selection, fixed comparison membership, CLI execution, and stale-output rejection |
| `tests/test_nested_assessment.py` | Outer-fold coverage, candidate/assessment isolation, mean-rank repeats, seed separation, frozen predictions under changed assessment labels, CLI artifacts, and failed-run status |
| `tests/test_repayment_methodology.py` | Split payments, ambiguous schedules, unknown payments, month grain, matched ratios, duplicates, and transactional rebuilds |
| `tests/test_model_artifacts.py` | Split manifests and parent/child calibration contracts |
| `tests/test_metrics.py` | Strict label validation before integer conversion |

The current suite uses pytest fixtures and small in-memory/staged datasets so it can run without the full Kaggle dataset.

---

## 5. Fixtures and Test Data

### 5.1 Fixture strategy

Use small synthetic dataframes and staging tables built by pytest fixtures:

```text
tests/conftest.py
tests/helpers.py
```

The fixture data should mimic the key columns and relationships of the Home Credit tables without including the full dataset.

### 5.2 Minimum fixture tables

```text
sample_application_train
sample_application_test
sample_bureau
sample_previous_application
sample_installments_payments
sample_bureau_balance
sample_pos_cash_balance
sample_credit_card_balance
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
- post-v1 comparison files are not required for the frozen v1 config.

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

- `mart_credit_risk_features` has one row per `(SK_ID_CURR, source_population)`;
- no duplicate keys within a source population;
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

For a scoped run, read the exclusion groups from that run's config (`configs/v1.yaml` or `configs/post_v1.yaml`); do not assume a separate default config owns its fitted feature list.

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

Saved-artifact loading rejects malformed/overlapping split IDs; retraining and stability seeds preserve saved test membership. Feature selection uses current training rows, including per-seed ranking. Metamorphic tests change labels outside training and verify unchanged selected features; changing labels outside reserved calibration leaves fitted transforms unchanged. These repairs do not establish an independent historical assessment population or nested assessment.

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
| `= T_high` | High risk | High-priority review |
| `> T_high` | High risk | High-priority review |

Test expectations:

- `T_low < T_high`;
- every score receives exactly one band;
- null scores fail explicitly or are assigned to a configured error band;
- action labels match the implemented `ACTION_LABELS` in `src/score_batch.py`.

### 10.2 Scenario tests

Test expectations:

- growth-oriented, balanced, and risk-averse scenarios exist;
- thresholds are valid for each scenario;
- scenario names are exported cleanly;
- scenario outputs reconcile with confusion matrix and expected-value calculations.

These tests cover legacy quantile scenarios. They do not establish hard queue capacity under tied scores or distribution shift, or equivalence between top-score ranking capture and middle-review capture.

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

Exact order and required columns are owned by `CREDIT_RISK_SCORE_COLUMNS` in [src/report_contracts.py](../../src/report_contracts.py):

```text
applicant_id
scoring_population
observed_target
score
raw_risk_score
calibrated_risk_score
calibration_method
score_decile
risk_band
recommended_action
threshold_version
model_version
top_reason_1
top_reason_2
top_reason_3
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

For post-v1, distinguish calibrated dashboard probability metrics from raw runtime evaluation metrics rather than requiring them to be numerically identical. Export schemas are owned by `src/report_contracts.py`. PBIX file/page checks do not verify displayed numbers or a Power BI refresh.

---

## 15. CLI and Pipeline Smoke Tests

### 15.1 Local smoke commands

At minimum, these repo commands are the documented local smoke path:

```bash
make test
make ingest
make features
make train
make evaluate
make score
make dashboard-data
```

Full-data commands require downloaded Kaggle files. CI runs pytest's synthetic integration paths, not the full-data Make pipeline. Use explicit scoped configs as documented in the README; bare step commands use the separate base-config paths.

### 15.2 Implemented CI behavior

The checked-in GitHub Actions workflow uses Python 3.12 and runs:

```bash
make setup
make lint
make format-check
make test
```

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

Existing fixture coverage includes the following checks. These are implementation checks, not a model-validation sign-off:

- [x] config parses and required sections exist;
- [x] feature mart has one row per applicant;
- [x] no duplicate applicant IDs in scoring output;
- [x] required feature and scoring columns exist;
- [x] model feature list excludes forbidden fields;
- [x] threshold policy assigns exactly one band to each score;
- [x] expected-value calculations match fixture examples;
- [x] scores are bounded between 0 and 1;
- [x] dashboard export files exist and are readable;
- [x] SHAP/reason-code outputs exclude diagnostic-only fields.

Correctness-repair regressions and remaining requirements:

- [x] split overlap and invalid identifier manifests fail on artifact load;
- [x] fractional, non-binary, and missing target labels fail before integer conversion in shared label validation;
- [x] stability seeds preserve saved test membership and exclude it from train/validation;
- [x] full-only feature experiments run without a SHAP ranking, and an empty feature-set request fails clearly;
- [x] feature-experiment utility uses raw scores even when calibrated scores are flat;
- [x] nested outer-assessment IDs remain outside that fold's adaptive fitting/selection paths, with exactly one prediction per development applicant per declared workflow per repeat;
- [x] model seed changes leave partition membership unchanged; ranking repeats see only the current fitting rows;
- [x] changed assessment labels leave the frozen workflow and scores unchanged, and failed assessment runs retain a failed manifest;
- [x] selection importance excludes reporting populations;
- [x] a fully paid obligation split across payment rows is not underpaid or double-counted; competing versions and unknown amounts are explicit;
- [x] last-k windows use distinct applicant months and due obligations; current/future rows are excluded and burden ratios use matched known operands;
- [x] duplicate account-month records fail and a failed feature rebuild preserves the previous mart/build identity;
- [x] calibration candidates meet their own minimum gain before sigmoid preference;
- [x] calibration candidates are assessed outside calibration-fitting rows;
- [x] tie-aware evaluation covers boundary, flat, reordered and mixed scores; reliability keeps score ties together; no hard-capacity queue is claimed or implemented;
- [x] action disposition and review costing reconcile: high band is simulated decline, middle band alone incurs review cost;
- [x] training runs receive distinct IDs even with the same timestamp, and calibration rejects a mismatched fitted parent;
- [x] thresholds and exports reject a mismatched fitted parent; replacing the feature build or calibrator also invalidates dependent outputs;
- [x] evaluation reports ignore stale calibration CSVs and identify the fitted model/feature build;
- [x] corrected CSV raw/calibrated metrics, actions, model/split/scenario and counts reconcile to one identified bundle; native PBIX DAX/filter propagation remains unverified;
- [x] LightGBM classifier parameters enable the configured row bagging; named corrected full-data fits remain separate from historical snapshots.

---

## 18. Definition of Tested

The following are required checks for the existing local pipeline. Completing them alone does not close the missing regressions or pending scientific gates above:

- all core `pytest` tests pass;
- sample or fixture-based tests can run without the full Kaggle dataset;
- data contracts pass on the real feature mart;
- scoring schema tests pass on both labeled holdout and unlabeled scoring populations;
- threshold and expected-value outputs reconcile;
- excluded fields are blocked from model training and reason-code output;
- dashboard exports load cleanly.

This is a portfolio-grade testing standard, not production bank-grade QA. The README should not claim production readiness.

## Named real-data verification

The [2026-10-04 correctness assessment](../../reports/correctness_20261004/assessment_report.md) records full corrected raw-to-dashboard v1/post-v1 runs, an independent Python actual-source oracle, matched full-data outer assessment, the declared 20,000-row shuffled-label diagnostic and a second locked same-host fold refit. Runtime manifests/logs and exact role IDs remain local; curated evidence contains no applicant IDs. Future and identical-payment source examples are absent in the real files, so distinguishing synthetic regressions explicitly cover their inclusion/retention behavior. Native PBIX numeric refresh remains unverified because required tooling is unavailable; CSV and newly generated anonymous static evidence are checked separately. Use `make lint format-check test PYTHON=.tmp/assessment-env/Scripts/python.exe` in the verified locked environment.

## Current tuning verification

The [v3 tuning evidence](../../reports/tuning_20261004/assessment_report.md) records the completed matched five-fold original-label assessment and complete sampled-20k shuffled-label control. `tests/test_tuning.py` verifies deterministic budgets beyond eight, unweighted/lighter cases, inner fitting/stopping/scoring boundaries, fitting-only preprocessing/raw rankings, actual best-iteration use and seed roles. `tests/test_tuning_integration.py` covers shared current train, feature-selection, split-stability and nested callers, eligible/selected field identities, named artifact schemas and historical-ID exclusion. Curating verifies complete CV evidence, fixed rounds, prior membership equality and preserved files. Independent recomputation checks coverage and paired metrics; all frozen scores and a first-fold locked-environment refit reproduce. These are local scientific checks, with native Power BI and chronology limitations unchanged.

Review regressions force genuine CV-selected top-40/top-80 training artifacts through evaluation alongside a full-field logistic artifact, verify each prediction uses its own fitted fields, and exercise downstream calibration/scoring/dashboard consumers. A separate legacy equal-field persisted-artifact regression retains compatibility; split/build/eligibility mismatches still fail. Current winner-only stability reports selection frequencies over all completed splits, conditional metric support and unavailable absent/singleton uncertainty; a faithful three-split case verifies 1/3 versus 2/3 and no aggregate promotion. Legacy all-surface aggregation remains separate.
