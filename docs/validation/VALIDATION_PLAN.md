# Loan Default Risk Decisioning System — Validation Plan

The [current case study](../../reports/portfolio/case_study.md) and [standalone offline report](../../reports/portfolio/index.html) translate current anonymous aggregate results into reader-facing terms. Exact assessment tables remain authoritative. Final probabilities can be unchanged raw probabilities when no adjustment wins. Highest-risk 10% capture is a ranking check, separate from middle-band manual review.


**Version:** 0.1  
**Status:** Named corrected protocol evidence, preserved historical comparisons, and native Power BI refresh limit
**Owner:** Steven  
**Aligned spec:** [PROJECT_SPEC.md](../spec/PROJECT_SPEC.md)
**Last updated:** 2026-10-04

---

## 1. Purpose

This validation plan defines how to judge whether the model, thresholds, business-value analysis, and dashboard outputs are credible for a portfolio-grade financial decision-support project.

Testing asks:

> Did we implement the system correctly?

Validation asks:

> Are the model and decisioning outputs reasonable, useful, stable, and honestly represented?

This project is not a production underwriting model and does not claim regulatory approval. Validation is designed to show professional model-risk awareness and avoid naive credit-model claims.

## Current Evidence Status

The completed [2026-10-04 tuning assessment](../../reports/tuning_20261004/assessment_report.md) owns current `nested_inner_cv_v3` results and anonymous evidence. Its matched five-fold assessment covers 261,384 development applicants and four workflows; the complete sampled-20k shuffled-label control, frozen artifact verification and locked same-host first-fold refit also completed. The [prior correctness assessment](../../reports/correctness_20261004/assessment_report.md) retains `nested_matched_holdout_v2` numeric evidence, source/mart reconciliation and artifact identities unchanged. Prior SQL/mart proofs are reused with hashes; current source fingerprints do not replace prior fingerprints. Only completed manifests are evidence.

| Area | Current behavior and evidence | Limit |
|---|---|---|
| Populations | Original 46,127 historical comparison IDs remain fixed. Nested assessment uses 261,384 development applicants, one outer prediction per declared workflow per repeat. Unlabeled Kaggle rows are scoring-only. | Prior dataset exploration remains; historical comparison is not an untouched lockbox. Random folds do not establish future-cohort validity. |
| Selection | Five outer folds retain disjoint 70% base fitting / 15% calibration / 15% method-threshold selection. Three inner folds within base fitting search 24 unique joint feature/parameter recipes per LightGBM workflow. Each inner fit reserves separate stopping rows; preprocessing/raw ranking use fitting rows alone. Final rounds are median inner best iterations. History surfaces are 40/80/full 174; application uses 31 SQL static fields. | Bounded search does not establish global optimality. No reserved or outer label enters search/stopping or post-calibration refit. Fold SD is descriptive, not a confidence interval; no pooled cross-fold average precision. |
| Matched controls | Independent training-prevalence, tuned unweighted logistic (four C values) and SQL-origin application-only LightGBM use the same roles/outer IDs. Fold-paired differences retain workflow/family identities. | Different workflows have declared different optimization budgets. The preserved r2 balanced fixed-C1 logistic has a different identity/recipe. Outer results do not select/promote a family. Ablation is conditional engineering evidence, not causal attribution. |
| Calibration and metrics | Mean inner Brier/log loss must be within +0.002/+0.01 of fitting-prevalence losses for candidate acceptance; AP then leads the declared ranking ties. All-failed fallback reports failure without retuning. Reserved calibration fitting and disjoint method selection retain minimum gain and sigmoid simplicity rules; calibrated selection-role acceptance is recorded. Log loss accompanies Brier/AP/ROC. Top-rate metrics average fractional boundary-tie membership with `ceil(n*rate)`. Reliability bins keep identical scores together and show counts. | Brier alone does not establish reliability. Rank-bin display ordering by score/ID remains a separate target-blind rule. Legacy metric names remain qualified below. |
| Source meaning | SQL normalizes unambiguous obligations, preserves ambiguous/unknown support, uses distinct applicant months and matched ratio operands. Independent Python checks actual split, on-time/late, arrears, competing-version, unknown and multi-account records. Full contracts pass. | No amendment timestamps or unique cashflow ID permit optimistic version guesses or silent deduplication. Actual future/identical-record examples are absent; synthetic regressions check these boundaries. |
| Bureau eligibility | One SQL owner admits only finite loan origins before application day; bureau, child balance and recency consume it. Separate coverage retains 25 rejected day-zero origins, with no positive/unknown origins in the actual source. Planned future maturities on eligible loans remain known contract information. | Day zero is not proven future leakage. Relative offsets cannot establish intraday/vendor ingestion availability; application external-score availability cannot be independently certified. |
| Leakage control | A predeclared stratified 20,000-development-row seed-913 shuffled-label control executes the complete five-fold inner recipe and independent comparators. Outer label/covariate mutation regressions leave frozen choices/preprocessors unchanged. | Chance behavior is interpreted with sampling variation; this sampled diagnostic cannot prove real-world availability or erase historical exploration. |
| Sensitivities | Predeclared selected-recipe model seeds 101/211/307 measure selection-role variation without choosing a seed. Ranking seeds 101/211/307 have a separate fitting-only role. Frozen known/unknown/no-history segments and all 27 margin/loss/review multiplier combinations 0.5/1/2 around 1000/5000/50. | Conditional population comparisons; no schedule guessing or outer-cost tuning. Utility is retrospective, with no real profit, reviewer effectiveness or rejected-loan counterfactual estimate. |
| Policy and artifacts | Raw-score quantile bands simulate approve/review/decline; only middle review incurs cost. Feature/model/calibrator/evaluation identities reject stale dependent outputs. Historical inputs remain recoverable in separate scopes. | No hard queue capacity is promised. Local checks/fingerprints are scientific reproducibility evidence, not a production immutable release/tamper system. |
| Reproduction and presentation | Project dependency closure is hash-locked. A second isolated environment refits declared first fold, same roles/seeds/inputs, matching choices and predictions within 1e-10. Raw/config/source hashes plus dirty Git identity qualify provenance. Corrected CSV model/split/scenario/score/actions/counts reconcile; new anonymous PNG/HTML evidence is generated. | Same-host numeric proof does not certify cross-hardware portability. Power BI Desktop/tooling is absent; native PBIX refresh and opaque DAX/relationship/import queries remain unverified. Original PBIX/screenshots are byte-identical historical assets. |

The [remediation proposal](../implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) retains dated historical designs. Inner CV is now implemented under the explicitly authorized bounded procedure described in the [methodology](ASSESSMENT_METHODOLOGY.md); hard capacity, broader release/lineage infrastructure and bootstrap remain outside this scope. History calibrated AP is 0.265862 versus prior 0.265750, essentially unchanged; application-only AP is 0.231021 versus 0.232638, and tuned logistic AP is 0.251083 versus prior fixed logistic 0.248427. These descriptive comparisons do not establish significance or guaranteed improvement. V3 history raw Brier/log loss are 0.066655/0.240720 versus prior raw 0.157305/0.477672; calibrated history probability losses are similar to prior. The search comparison alone cannot attribute this difference to one parameter; the separate controlled weighting follow-up below addresses that question.

The separate [class-weighting diagnostic](../../reports/class_weighting_20261004/assessment_report.md) now resolves the main raw-probability attribution question conditional on the frozen history recipes. Twenty fixed-recipe fits change only positive-class weighting on the same five applicant groups. Removing weighting improves Brier/log loss in every recipe/group pair and essentially reproduces the original loss reduction; adding the earlier weight to the current recipe reverses it. All ten reference refits reproduce frozen scores exactly. This retrospective model-setting intervention is distinct from feature-set comparison, population causal claims and tuning-policy assessment. It does not promote a model or change prior numeric evidence.

The separate [current model-input diagnostic](../../reports/model_inputs_20261004/methods.md) reuses all five frozen current history models and saved fitting-only preprocessing, without fitting or selection. A predeclared uniform target-blind sample of 1,000 assessment rows per fold (seed 20261004) supports native TreeSHAP magnitudes. Absolute encoded effects sum by raw field and source group, then average within fold and equally across folds, in natural-log-odds units. Intercept-inclusive signed additivity is checked separately. Imported scores lead individual magnitudes; group size and correlated/derived fields limit cumulative attribution. This is sampled fitted-model behavior, not causality, predictive-value ablation or adverse-action reasons. Anonymous final tables and source identities remain separate from original empirical assessment fingerprints. The presentation also shows matched installment-only segment comparisons and all 27 existing fixed-policy utility assumptions without cost optimization.

Terminology:

- **PR-AUC** is legacy scikit-learn average precision, not trapezoidal PR area.
- **Recall at manual review capacity** means top-score capture, distinct from the middle review band; boundary ties receive equal expected membership.
- **Brier/log loss** assess probability quality; reliability bins provide a separate descriptive check.
- **Expected value** is legacy retrospective scenario utility in illustrative weight units.
- **Held-out test** remains the saved within-run label for a reused historical comparison.

Historical numbers stay in [experiment-log](../../reports/experiments/experiment_log.csv) rows `000` and `015`. The separate named reports identify corrected source/model and current tuning evidence without rewriting that history or claiming a refreshed Power BI report.

---

## 2. Validation Scope

Validation covers:

- data and target sanity;
- training/calibration/validation/test split integrity;
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
| `model_metrics_summary` | ROC-AUC, average precision, Brier/log loss, tie-aware lift/top-score capture |
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

- Training, calibration, validation, and test roles are disjoint by `SK_ID_CURR` (v1 omits calibration).
- Split proportions match config; post-v1 divides the configured validation budget equally between calibration fitting and selection validation.
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

Current safeguards reject cross-split overlap, preserve saved test membership across retraining/stability seeds, rank features only within training, and reserve separate calibration-fitting rows. Historical cross-run reuse and repeated dataset exploration still prevent an independent-assessment claim across the experiment trail.

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

Historical post-v1 fitting/selection shared validation data. Current fitting uses a reserved calibration role and disjoint method selection; named outer assessment evaluates the frozen choice independently within each fold.

---

### Gate 6 — Threshold and Business-Value Validation

**When:** After model selection and calibration assessment.

**Checks:**

- Validation-derived threshold scenarios are evaluated on validation data.
- Growth-oriented, balanced, and risk-averse scenarios are defined.
- Report simulated middle-review and decline volumes separately. Fixed quantiles are not a hard capacity guarantee.
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

At least three predefined threshold scenarios are reported with counts, utility units, and action assumptions. "Balanced" is the displayed reference scenario, not an optimized policy. The high band now simulates decline with zero modeled value/cost, aligning action and utility. Quantiles are a rate reference; hard capacity enforcement is outside the declared scope.

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

Historically, original test applicants entered fitting under stability seeds and reporting-population SHAP fed feature selection. Corrected code removes those paths, while prior exploration still prevents untouched external-generalization claims.

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
- [x] corrected CSV bundle, report narrative and newly generated static evidence reconcile to one identified run;
- [ ] native PBIX visuals and refreshed screenshots reconcile; required native tooling is unavailable here and original historical assets are preserved;
- [x] README limitations are clear.

Correctness gates for the named protocol:

- [x] each outer prediction excludes its applicant from that workflow's fitting/selection; original comparison IDs never enter nested/control fitting;
- [x] feature selection importance uses only base-fitting rows;
- [x] calibration fitting and method assessment are separated, with per-method eligibility;
- [x] declared obligation/window/missingness/relative-availability semantics pass actual-source and distinguishing synthetic checks; real ingestion/intraday availability remains unproven;
- [x] simulated actions and utility assumptions agree; quantile rate references and tie-aware rank metrics are tested without a hard-capacity claim;
- [x] split manifests and dependent artifacts identify and validate the fitted parent;
- [x] effective tuning parameters and a controlled locked same-host fold reproduction are recorded by the named evidence report.

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
