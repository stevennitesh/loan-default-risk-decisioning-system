# Loan Default Risk Decisioning System — Validation Plan

**Version:** 0.1  
**Status:** Validation requirements and historical evidence; correctness gaps remain open
**Owner:** Steven  
**Aligned spec:** [PROJECT_SPEC.md](../spec/PROJECT_SPEC.md)
**Last updated:** 2026-10-03

---

## 1. Purpose

This validation plan defines how to judge whether the model, thresholds, business-value analysis, and dashboard outputs are credible for a portfolio-grade financial decision-support project.

Testing asks:

> Did we implement the system correctly?

Validation asks:

> Are the model and decisioning outputs reasonable, useful, stable, and honestly represented?

This project is not a production underwriting model and does not claim regulatory approval. Validation is designed to show professional model-risk awareness and avoid naive credit-model claims.

## Current Evidence Status

As of 2026-10-03, the local pipeline is implemented and the repository preserves historical v1/post-v1 experiments for recruiter and hiring-manager review. This is not a certification that every gate below passes. The gates are requirements; report presence and passing fixture tests alone do not establish scientific correctness.

| Area | Verified current behavior and evidence limit | Required repair or assessment |
|---|---|---|
| Assessment populations | [Repeated-seed stability](../../src/model_stability.py) re-splits all labeled applicants; original seed-42 test applicants enter training/validation under other seeds. Test results were also repeatedly observed. | Treat saved test results as historical comparisons. Enforce a development/assessment boundary before new adaptive selection; the exact protocol in the remediation plan remains a proposal. |
| Feature selection | [SHAP export](../../src/explain.py) aggregates holdout and Kaggle scoring populations; [feature experiments](../../src/feature_experiments.py) consume that ranking. | Generate selection importance within development fitting data. Post-selection reporting SHAP can remain an interpretation aid. |
| Calibration | [Calibration fitting and selection](../../src/calibrate.py) share validation rows. [Method selection](../../src/calibration.py) can prefer sigmoid even when its individual gain is below the configured minimum. | Separate calibration fitting from method assessment; apply eligibility to each method before preference. Do not present fit-set calibration gains as independent evidence. |
| SQL feature meaning | [Installment aggregation](../../sql/05_feature_installments.sql) repeats owed amounts across split-payment rows and evaluates underpayment per payment row. [Last-k features](../../sql/05d_feature_last_k_temporal.sql) rank POS/card account records, not distinct applicant months. | Resolve obligation identity/version semantics and normalize split payments; label record windows accurately or implement distinct-month windows. Review missing-value handling and pre-application availability. No actual future-data leakage has been established by this audit. |
| Ranking vs policy | [Recall-at-capacity](../../src/metrics.py) measures capture among the highest scores. [Threshold scenarios](../../src/thresholding.py) use fixed validation quantiles; they do not enforce a hard capacity on a new population. | Separate ranking capture from actual queue capture and test capacity under ties and distribution shifts before claiming constrained review. |
| Action vs utility | [Scoring](../../src/score_batch.py) labels the high band `high_priority_review`, while the utility formula charges only the middle review band and gives the high band no disposition/value. | Reconcile action meaning, review cost, and disposition before calling the scenario an operational policy. Legacy expected-value fields are retrospective utility units, not measured currency profit. |
| Artifact integrity | [Artifact loading](../../src/model_artifacts.py) does not reject cross-split ID overlap and binds calibration by a reusable model-version string rather than exact fitted-run identity. | Reject invalid split manifests and bind dependent artifacts to the fitted parent. Dashboard display aliases are not release identifiers. |
| Training controls | [LightGBM presets](../../src/modeling.py) specify `subsample`, but leave `subsample_freq` at zero, disabling row bagging. | Reconcile configured tuning claims with effective fitted parameters before rerunning comparisons. |
| Reporting and reproduction | [Dashboard export](../../src/dashboard_exports.py) recomputes segment diagnostics and can replace selected-model probability metrics with calibrated values. Local generated bundles differ from curated snapshots; dependencies have lower bounds, not a lock. | Preserve snapshot lineage, distinguish raw/calibrated score views, and reconcile one deliberately generated bundle with PBIX visuals. Exact historical numerical reproduction is not currently certified. |

The [remediation plan](../implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) proposes a concrete repair sequence. None of its new population protocol, policy, lineage machinery, or proposed commands is implemented by this documentation update. Preserve historical results; do not silently replace them with locally regenerated metrics.

Terminology for current documentation:

- **PR-AUC** is the legacy name for scikit-learn average precision, not trapezoidal PR-curve area.
- **Recall at manual review capacity** is the legacy top-score ranking metric; use **top-10% default capture** for its 10% presentation, distinct from the middle review band.
- **Brier score** measures overall probability quality; it does not establish calibration on its own.
- **Expected value** is the legacy name for retrospective scenario utility using `1000`, `5000`, and `50` weights. No reviewer effectiveness or real loan economics is estimated.
- **Held-out test** is a saved within-run split label. Across this experiment history it is a reused comparison population, not an independent final lockbox.

Historical metric sources are curated [experiment-log](../../reports/experiments/experiment_log.csv) rows `000` and `015` and the numbered reports. Runtime outputs, PBIX snapshots, and screenshots must be identified separately; their existence does not prove mutual reconciliation.

---

## 2. Validation Scope

Validation covers:

- data and target sanity;
- train/validation/test split integrity;
- leakage checks;
- baseline comparison;
- LightGBM performance;
- calibration;
- lift and ranking quality;
- threshold and expected-value analysis;
- segment diagnostics;
- explainability reasonableness;
- scoring output review;
- dashboard reconciliation;
- documentation and limitations.

Validation does not cover:

- fair-lending compliance certification;
- adverse-action notice compliance;
- production monitoring infrastructure;
- live policy approval;
- legal review;
- model governance sign-off.

---

## 3. Validation Artifacts

Runtime report paths below are relative to the configured `reports_dir`. Scoped runs use `reports/v1/` or `reports/post_v1/`; bare step commands use the separate defaults in `configs/base.yaml`. Exact exported columns are owned by [src/report_contracts.py](../../src/report_contracts.py), not the abbreviated descriptions here.

| Artifact | Purpose |
|---|---|
| `reports/validation_report.md` | Main model validation summary |
| `reports/model_card.md` | Intended use, non-use, data, metrics, limitations |
| `reports/business_value_analysis.md` | Threshold and expected-value interpretation |
| `model_run_summary` | Model version, data version, config, feature count, split info |
| `model_metrics_summary` | ROC-AUC, PR-AUC, Brier score, lift, recall-at-capacity |
| `model_threshold_metrics` | Threshold scenario comparison |
| `model_lift_by_decile` | Decile-level ranking performance |
| `model_calibration_bins` | Predicted vs observed default by score bucket |
| `segment_performance_summary` | Diagnostic performance by broad segments |
| `model_feature_importance` | Global model drivers |
| `credit_risk_scores` | Final scored applicant output |

---

## 4. Validation Gates

### Gate 1 — Data and Target Validation

**When:** After ingestion and feature mart creation, before modeling.

**Checks:**

- Confirm target values are binary and non-null for labeled training rows.
- Confirm unlabeled Kaggle test rows do not have `TARGET`.
- Confirm positive-class rate.
- Confirm one row per `(SK_ID_CURR, source_population)` in the feature mart.
- Confirm major feature groups have plausible missingness rates.
- Confirm no forbidden model fields appear in the model feature list.
- Confirm no obvious target leakage fields exist.

**Required outputs:**

```text
reports/data_inventory.csv
reports/feature_inventory.csv
initial target-rate summary
missingness summary
```

**Pass condition:**

The data is suitable for baseline modeling, or issues are documented with mitigation.

---

### Gate 2 — Split and Preprocessing Validation

**When:** Before training baseline and LightGBM models.

**Checks:**

- Train/validation/test splits are disjoint by `SK_ID_CURR`.
- Split proportions match config.
- Positive-class rate is similar across splits.
- Preprocessing is fit only on the appropriate training data.
- Calibration and threshold selection are not fit on held-out test data.
- `application_test` is not used for performance evaluation.

**Required outputs:**

```text
split_summary table
model_run_summary split metadata
```

**Pass condition:**

Splits are valid and leakage controls are documented.

Current gap: ordinary split generation is disjoint within a run, but saved-artifact loading lacks a cross-split overlap check and repeated-seed experiments do not preserve a common assessment boundary. This gate is not satisfied across the historical experiment trail.

---

### Gate 3 — Baseline Model Validation

**When:** After logistic regression training.

**Checks:**

- Baseline trains end-to-end.
- Predicted scores are bounded between 0 and 1.
- Baseline ranking metrics are above random behavior.
- Baseline calibration is inspected.
- Baseline feature count and feature exclusions are documented.

**Required metrics:**

- ROC-AUC;
- PR-AUC;
- Brier score;
- top-decile lift;
- calibration bins.

**Pass condition:**

The baseline is stable enough to serve as a comparison point. If baseline performance is weak, document why and continue only if data/target checks are sound.

---

### Gate 4 — LightGBM Model Validation

**When:** After primary model training.

**Checks:**

- LightGBM outperforms or materially complements the logistic regression baseline on validation data.
- Improvements are evaluated with PR-AUC, ROC-AUC, lift, and expected-value behavior, not accuracy alone.
- Model does not rely on excluded fields.
- Top global drivers are plausible.
- Model score distribution is not degenerate.
- Missing-value handling is documented.

**Required metrics:**

| Metric | Validation question |
|---|---|
| ROC-AUC | Does the model rank applicants better than random? |
| PR-AUC | Does the model handle the minority default/difficulty class well? |
| Brier score | Are scores usable as probability-like outputs? |
| Top-decile lift | Does the model concentrate risk in the highest-score group? |
| Top-score capture at a reference rate | Does the ranking concentrate observed positives? This does not measure actual review-band capture. |
| Calibration bins | Do predicted rates match observed rates reasonably? |

**Pass condition:**

LightGBM becomes the primary model only if it improves the decisioning story. If it does not beat the baseline clearly, the README should say so and explain the tradeoff.

---

### Gate 5 — Calibration Validation

**When:** After model selection, before final threshold reporting.

**Checks:**

- Compare uncalibrated LightGBM scores against calibrated alternatives if implemented.
- Evaluate Brier score.
- Inspect calibration bins.
- Confirm calibration is fit on validation/calibration data, not held-out test.
- Assess fitted calibrators on rows not used to fit them; test each candidate's minimum-improvement eligibility before applying a preference rule.
- Confirm calibration does not materially reduce ranking usefulness.

**Calibration candidates:**

- uncalibrated LightGBM;
- Platt scaling;
- isotonic regression.

**Pass condition:**

The selected score representation is documented. If uncalibrated scores are used, state that they are treated primarily as risk scores, not perfect probabilities.

Current post-v1 fitting/selection shares validation data; documentation of sigmoid gains does not satisfy independent method assessment.

---

### Gate 6 — Threshold and Business-Value Validation

**When:** After model selection and calibration assessment.

**Checks:**

- Validation-derived threshold scenarios are evaluated on validation data.
- Growth-oriented, balanced, and risk-averse scenarios are defined.
- Report actual middle-review and high-priority-review volumes separately. Fixed quantiles are not a hard capacity guarantee.
- Expected-value assumptions are explicit and configurable.
- Threshold choices are fixed before test-set reporting.
- Business-value tables reconcile to confusion matrix/action counts.

**Required outputs:**

```text
model_threshold_metrics
model_confusion_matrix
reports/business_value_analysis.md
```

**Required scenario fields:**

```text
scenario_name
threshold_low
threshold_high
approval_rate
manual_review_rate
high_risk_rate
default_rate_approved
high_risk_default_capture_rate
expected_value
```

**Pass condition:**

At least three predefined threshold scenarios are reported with counts, utility units, and action assumptions. "Balanced" is the displayed reference scenario, not an optimized policy. Capacity enforcement and the high-priority-review cost/disposition mismatch remain open.

---

### Gate 7 — Held-Out Comparison and Generalization Validation

**When:** After each experiment decision is fixed using training/validation only.

**Checks:**

- Held-out test metrics may be observed across documented post-v1 experiments
  as comparison and generalization evidence.
- Held-out test results are not used in the formal model, feature, calibrator,
  or threshold selection rules.
- Held-out test lift and threshold behavior are compared to validation behavior.
- Differences between validation and test are documented.

**Required outputs:**

```text
model_metrics_summary rows for held-out test split
model_lift_by_decile rows for held-out test split
model_threshold_metrics rows for held-out test split
```

**Pass condition:**

Held-out test results are reasonably consistent with validation results, or
gaps are explained honestly. Because this set has been observed across post-v1
experiments, it is a comparison/generalization set rather than a completely
untouched final lockbox.

The limitation goes beyond repeated observation: original test applicants enter fitting under the stability seeds, and reporting-population SHAP importance feeds feature selection. Similar validation/test metrics therefore do not demonstrate independent generalization.

---

### Gate 8 — Segment and Model-Risk Diagnostics

**When:** After final scoring and held-out comparison-set evaluation.

**Purpose:** Show model-risk awareness without claiming fair-lending compliance.

**Implemented diagnostic segments:** `CODE_GENDER`, `NAME_FAMILY_STATUS`, `applicant_age_band`, `CNT_CHILDREN`, and `CNT_FAM_MEMBERS`, as defined in [src/dashboard_segments.py](../../src/dashboard_segments.py). These fields are excluded from model fitting. Income, loan-amount, contract-type, and missingness-group segment exports are possible extensions, not current outputs.

**Checks by segment:**

- applicant count;
- observed target rate;
- average score;
- approval/review/high-risk action distribution;
- ROC-AUC where sample size permits;
- PR-AUC where sample size permits;
- Brier score or calibration gap where sample size permits;
- false positive/false negative patterns where relevant.

**Important limitation language:**

The segment analysis is a diagnostic check only. It is not a fair-lending analysis, compliance approval, or production governance substitute.

**Pass condition:**

Major performance variation is either absent, explained, or documented as a limitation.

---

### Gate 9 — Explainability Validation

**When:** After SHAP outputs are generated.

**Checks:**

- Top global drivers are plausible for credit-risk decision support.
- Excluded demographic/protected-status-like fields do not appear in model drivers.
- Reason-code-style outputs are readable.
- Reason codes do not claim legal adverse-action compliance.
- Local explanations are directionally consistent with feature values where inspected.

**Preferred explanation examples:**

```text
High credit-to-income ratio
Low external risk score
Recent payment delays
High overdue amount in prior credit history
High annuity-to-income ratio
```

**Red-flag explanation examples:**

```text
Applicant gender
Applicant age
Marital status
Raw applicant ID
Target leakage field
```

**Pass condition:**

Explainability artifacts support interpretation and debugging without overclaiming legal compliance.

---

### Gate 10 — Dashboard and Reporting Validation

**When:** Before final README polish.

**Checks:**

- Power BI visuals reconcile to exported tables.
- KPI cards match `model_metrics_summary`.
- Threshold visuals match `model_threshold_metrics`.
- Lift chart matches `model_lift_by_decile`.
- Confusion matrix matches selected threshold scenario.
- Dashboard distinguishes evaluation population from unlabeled scoring demo where relevant.
- Dashboard screenshot supports the project story.

**Pass condition:**

A reviewer can understand the business tradeoff from the dashboard screenshot without running code.

---

## 5. Metric Reporting Standard

### 5.1 Headline metrics

Report these in README and validation report:

- PR-AUC;
- ROC-AUC;
- Brier score;
- top-decile lift;
- recall at review capacity;
- expected value for selected scenario;
- approval/review/high-risk rates.

### 5.2 Secondary or appendix metrics

- accuracy;
- F1 score;
- precision/recall at arbitrary thresholds;
- feature importance rank changes;
- segment-level diagnostics.

Accuracy should not be the lead metric because the target is imbalanced.

---

## 6. Model Selection Criteria

The implemented baseline-versus-LightGBM family choice uses validation average precision. LightGBM candidate tuning first excludes degenerate score distributions, then sorts by validation PR-AUC, top-decile lift, top-score capture, ROC-AUC, and lower Brier score. Expected value is not the model-selection objective.

The following are broader review considerations, not additional implemented selection rules:

| Criterion | Why it matters |
|---|---|
| PR-AUC | Minority-class usefulness |
| ROC-AUC | General ranking quality |
| Top-decile lift | Business value of ranking |
| Brier/calibration | Probability-like score quality |
| Threshold expected value | Decision usefulness |
| Simplicity | Readability and reproducibility |
| Explainability | SHAP and reason-code quality |
| Stability | Development-fold consistency; historical test similarity alone is insufficient |

A different formal selection rule would require an explicit protocol change before viewing comparison outcomes.

---

## 7. Business-Value Validation

Expected value is illustrative, not a claim about real Home Credit economics.

### Required assumptions

```yaml
business_assumptions:
  expected_margin_per_good_loan: 1000
  expected_loss_per_bad_loan: 5000
  manual_review_cost: 50
  manual_review_capacity_rate: 0.10
```

### Validation checks

- assumptions appear in config;
- assumptions are repeated in README/report;
- scenario results change when assumptions change;
- business-value calculations reconcile to counts;
- final interpretation avoids overstating actual dollars.

### Reporting language

Use:

> The predefined balanced scenario illustrates approval, review, high-risk volume, and retrospective utility under stated weights. It is not a proven optimal policy or a hard review-capacity allocation.

Avoid:

> The model generated $X of real profit.

---

## 8. Model Card Requirements

`reports/model_card.md` should include:

```text
Intended Use
Not Intended For
Dataset
Target Definition
Training Population
Scoring Population
Feature Groups
Excluded Features
Model Type
Metrics
Calibration
Threshold Policy
Business Assumptions
Explainability
Segment Diagnostics
Limitations
Monitoring Considerations
```

The model card should explicitly state:

- this is not a production underwriting system;
- model outputs are decision-support artifacts;
- legal/compliance review is outside scope;
- SHAP reason codes are not adverse-action notices;
- business assumptions are illustrative.

---

## 9. Validation Report Outline

`reports/validation_report.md` should use this structure:

```text
# Validation Report

## Executive Summary
## Data and Target Validation
## Split Strategy
## Feature and Leakage Controls
## Baseline Model Results
## LightGBM Model Results
## Calibration Analysis
## Lift and Decile Analysis
## Threshold Scenario Analysis
## Business-Value Analysis
## Segment Diagnostics
## Explainability Review
## Held-Out Comparison-Set Results
## Limitations
## Recommendation for Portfolio v1
```

---

## 10. Minimum Release Standard

This records the historical artifact checklist, not a current validation sign-off. Checked items below indicate reported artifacts or implemented paths; unresolved correctness gates follow.

- [x] data/target validation is documented;
- [x] train/validation/test split summary is documented;
- [x] leakage controls are documented;
- [x] baseline model results are reported;
- [x] LightGBM results are reported;
- [x] calibration is evaluated;
- [x] lift-by-decile table exists;
- [x] threshold scenarios exist;
- [x] business-value assumptions are explicit;
- [x] historical test metrics are reported with each experiment;
- [x] segment diagnostics are included;
- [x] SHAP/global driver outputs are reviewed;
- [x] scoring output is validated;
- [ ] current generated bundle, report narrative, PBIX visuals, and screenshots reconcile to one identified run;
- [x] README limitations are clear.

Pending correctness requirements:

- [ ] assessment applicants remain outside adaptive fitting and selection across runs;
- [ ] feature selection importance is confined to development data;
- [ ] calibration fitting and method assessment are separated, with per-method eligibility;
- [ ] installment obligations, observation windows, missingness, and availability semantics are verified;
- [ ] actual action queues and utility assumptions agree, with tested capacity semantics;
- [ ] split manifests and dependent artifacts identify and validate the exact fitted parent;
- [ ] effective tuning parameters and a controlled reproduction are verified.

---

## 11. Common Validation Failure Modes

| Failure mode | What to do |
|---|---|
| LightGBM barely beats baseline | Keep baseline comparison honest; focus on pipeline and decisioning value |
| Model has good ROC-AUC but poor PR-AUC | Emphasize imbalance; tune threshold/review capacity; do not headline accuracy |
| Calibration is poor | Treat scores as rank scores or add calibration; document limitations |
| Threshold scenario produces unrealistic approval/review volume | Adjust scenario assumptions and state tradeoffs clearly |
| Segment diagnostics show sharp performance gaps | Document as limitation; do not claim compliance |
| SHAP top features are sensitive/diagnostic fields | Fix feature exclusion and retrain |
| Validation/test gap is large | Check split, leakage, feature instability, and overfitting |
| Dashboard numbers do not match reports | Fix export/reconciliation before publishing |

---

## 12. Definition of Validated

For a corrected portfolio release, the model can be called adequately validated only when the pending requirements above and the following reporting requirements are met. The historical release is not currently certified against this standard:

- the data, target, and split strategy are documented;
- baseline and LightGBM results are compared honestly;
- metrics are appropriate for imbalanced financial outcomes;
- calibration and lift are evaluated;
- thresholds are derived from validation data and reported on the held-out
  comparison/generalization set;
- expected-value assumptions are explicit and configurable;
- segment diagnostics and limitations are included;
- explanations are plausible and do not expose excluded fields;
- dashboard outputs reconcile to validation tables;
- the README does not overclaim production, compliance, or underwriting readiness.

The validation standard is deliberately professional but scoped: strong enough for portfolio review, not presented as production model-risk governance.
