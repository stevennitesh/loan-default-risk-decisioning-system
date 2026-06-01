# Loan Default Risk Decisioning System — Validation Plan

**Version:** 0.1  
**Status:** Implemented validation plan for portfolio decision-support reporting
**Owner:** Steven  
**Aligned spec:** `docs/spec/PROJECT_SPEC.md` v0.3.1
**Last updated:** 2026-06-01

---

## 1. Purpose

This validation plan defines how to judge whether the model, thresholds, business-value analysis, and dashboard outputs are credible for a portfolio-grade financial decision-support project.

Testing asks:

> Did we implement the system correctly?

Validation asks:

> Are the model and decisioning outputs reasonable, useful, stable, and honestly represented?

This project is not a production underwriting model and does not claim regulatory approval. Validation is designed to show professional model-risk awareness and avoid naive credit-model claims.

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
- Confirm one row per `SK_ID_CURR` in the feature mart.
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
- Calibration and threshold selection are not fit on final test data.
- `application_test` is not used for performance evaluation.

**Required outputs:**

```text
split_summary table
model_run_summary split metadata
```

**Pass condition:**

Splits are valid and leakage controls are documented.

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
| Recall at review capacity | Can the model support constrained manual review? |
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
- Confirm calibration is fit on validation/calibration data, not final test.
- Confirm calibration does not materially reduce ranking usefulness.

**Calibration candidates:**

- uncalibrated LightGBM;
- Platt scaling;
- isotonic regression.

**Pass condition:**

The selected score representation is documented. If uncalibrated scores are used, state that they are treated primarily as risk scores, not perfect probabilities.

---

### Gate 6 — Threshold and Business-Value Validation

**When:** After model selection and calibration assessment.

**Checks:**

- Threshold grid is evaluated on validation data.
- Growth-oriented, balanced, and risk-averse scenarios are defined.
- Manual review capacity is respected or clearly reported.
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
default_capture_rate
expected_value
```

**Pass condition:**

At least three threshold scenarios are reported, and the selected balanced scenario is defensible under the stated assumptions.

---

### Gate 7 — Final Test-Set Validation

**When:** After model and thresholds are fixed using training/validation only.

**Checks:**

- Final metrics are computed once on held-out test data.
- Test-set results are not used to retune the model.
- Test-set lift and threshold behavior are compared to validation behavior.
- Differences between validation and test are documented.

**Required outputs:**

```text
final model_metrics_summary rows for test split
final model_lift_by_decile rows for test split
final model_threshold_metrics rows for test split
```

**Pass condition:**

Test-set results are reasonably consistent with validation results, or gaps are explained honestly.

---

### Gate 8 — Segment and Model-Risk Diagnostics

**When:** After final scoring and test-set evaluation.

**Purpose:** Show model-risk awareness without claiming fair-lending compliance.

**Diagnostic segments:**

- income band;
- loan amount band;
- contract type;
- application type where available;
- broad age band only if retained in a separate diagnostic layer;
- gender only if retained in a separate diagnostic layer and framed carefully;
- missingness groups for major feature families.

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
- Dashboard screenshot supports the recruiter story.

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

The primary model should be selected using a balanced view of:

| Criterion | Why it matters |
|---|---|
| PR-AUC | Minority-class usefulness |
| ROC-AUC | General ranking quality |
| Top-decile lift | Business value of ranking |
| Brier/calibration | Probability-like score quality |
| Threshold expected value | Decision usefulness |
| Simplicity | Recruiter readability and reproducibility |
| Explainability | SHAP and reason-code quality |
| Stability | Similar validation/test behavior |

A model with slightly lower AUC but better calibration, cleaner threshold behavior, and clearer explanations may be preferred.

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

> Under illustrative assumptions, the balanced threshold scenario produced the strongest tradeoff between approval volume, default capture, review workload, and expected value.

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
## Final Test-Set Results
## Limitations
## Recommendation for Portfolio v1
```

---

## 10. Minimum Release Standard

The project should not be released as v1 until:

- [ ] data/target validation is documented;
- [ ] train/validation/test split summary is documented;
- [ ] leakage controls are documented;
- [ ] baseline model results are reported;
- [ ] LightGBM results are reported;
- [ ] calibration is evaluated;
- [ ] lift-by-decile table exists;
- [ ] threshold scenarios exist;
- [ ] business-value assumptions are explicit;
- [ ] final test-set metrics are reported after thresholds are fixed;
- [ ] segment diagnostics are included or explicitly deferred;
- [ ] SHAP/global driver outputs are reviewed;
- [ ] scoring output is validated;
- [ ] dashboard screenshot reconciles to exported metrics;
- [ ] README limitations are clear.

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

For portfolio v1, the model is considered adequately validated when:

- the data, target, and split strategy are documented;
- baseline and LightGBM results are compared honestly;
- metrics are appropriate for imbalanced financial outcomes;
- calibration and lift are evaluated;
- thresholds are chosen on validation data and reported on final test data;
- expected-value assumptions are explicit and configurable;
- segment diagnostics and limitations are included;
- explanations are plausible and do not expose excluded fields;
- dashboard outputs reconcile to validation tables;
- the README does not overclaim production, compliance, or underwriting readiness.

The validation standard is deliberately professional but scoped: strong enough for recruiters, not presented as production model-risk governance.
