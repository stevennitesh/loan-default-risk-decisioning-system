# Model card: public credit-risk portfolio

## Current assessment and intended use

This project asks whether prior loan and repayment history improves ranking beyond application fields. It is a local decision-support portfolio using public Home Credit data. The outcome, `TARGET`, records repayment difficulty; it does not measure financial loss. It is not an underwriting, compliance, fair-lending or adverse-action system.

[The HTML project report](https://stevennitesh.github.io/loan-default-risk-decisioning-system/) explains the results and contains current charts; its [standalone HTML](portfolio/index.html) supports offline reading. The completed five-group assessment compares models on 261,384 labeled development applicants. Mean average precision is 0.266 for application and loan history LightGBM, 0.251 for history logistic regression, 0.231 for application-only LightGBM and 0.081 for the constant training-outcome-rate benchmark. Average precision measures ranking, not accuracy. The history model's mean Brier/log loss is 0.067/0.241 (lower is better), and ROC AUC is 0.776. [Exact results and identities](tuning_20261004/assessment_report.md) remain the numerical authority.

All five history fits selected 174 eligible inputs and unchanged raw probabilities after probability-adjustment testing. Different model families have different declared search budgets. These results assess the selection procedure fitted inside each applicant group; they do not describe one promoted saved dashboard model. Each workflow predicts the same assessment applicants, with fitting, stopping, probability-adjustment and selection roles kept separate. No outer result promotes a recipe, input surface, model family or seed.

## Controlled probability-quality follow-up

[A controlled follow-up](class_weighting_20261004/assessment_report.md) identifies positive-class weighting as the dominant explanation for improved raw probabilities in these history-model recipes. With the earlier recipe otherwise fixed, removing weighting changes raw Brier/log loss from 0.157305/0.477672 to 0.066572/0.240121, essentially reproducing current losses of 0.066655/0.240720. The reverse intervention worsens the current recipe; both losses improve without weighting in all ten recipe/group pairs. This diagnostic preserves selected inputs, preprocessing, applicant roles, seed and other parameters. It does not retune or promote a model, separately assess the search screen's effect, or establish optimal weights for future populations. Earlier probability adjustment had already corrected much of the raw-scale error; final probability quality changed little.

## Inputs and limits

SQL owns applicant-level feature extraction; Python owns orchestration, models, scoring and reporting. Identifiers, the outcome and direct demographic/protected-status-like fields are excluded from model inputs. Diagnostic fields remain separate. Prior public-data exploration means this is not an untouched final test. Calendar application, input-availability and outcome-maturity timestamps are missing; random applicant groups do not establish future-cohort validity. Correlated or proxy inputs may remain, and exclusion alone is not fairness certification.

Unlabeled Kaggle applications are scored only as a demonstration. Simulated approval/manual review/decline bands use fixed selection-derived cutoffs, with review cost in the middle band only. Utility weights are illustrative units, not dollars or profit. Reviewer effectiveness, rejected-loan outcomes and a hard review-queue limit are not estimated. SHAP contributions describe model behavior, not causal explanations or adverse-action notices.

[Current evidence status](../docs/validation/VALIDATION_PLAN.md#current-evidence-status) owns the validation boundaries; [assessment methodology](../docs/validation/ASSESSMENT_METHODOLOGY.md) explains the applicant roles. The saved Power BI files/screenshots remain historical and unrefreshed. Current report regeneration requires only committed anonymous aggregates (`make portfolio`), not raw data or training.

## Historical model card archive

The text below preserves the earlier model identities and numeric history. Its dated model descriptions and next actions are historical context, not the current reader journey or instructions.

<details>
<summary>Earlier application-and-history model card</summary>

# Model Card: Historical V1 and Post-v1 Credit-Risk Portfolio

## Model Summary

| Field | Value |
|---|---|
| Model version | `lightgbm_credit_risk_v1` |
| Model type | LightGBM binary classifier |
| Baseline | Logistic regression |
| Data scope | v1 Home Credit source files |
| Prediction target | Repayment difficulty indicator, `TARGET` |
| Primary use | Portfolio decision-support simulation |
| Production readiness | Not production-ready |

This model card preserves the frozen v1 baseline and summarizes the post-v1 improvement path. Scores are used to demonstrate threshold tradeoffs, batch scoring, explainability, and Power BI reporting. v1 scores should be treated as ranking scores, not fitted calibrated default probabilities.

**Evidence status (2026-10-04):** this is a resume portfolio for recruiters and hiring managers. The metrics below preserve historical experiments; they are not an independent final-test certification or live local metrics. Original test applicants entered fitting in repeated-seed runs, reporting-population SHAP fed feature selection, and post-v1 calibration fitting/selection shared validation rows. Current code fixes those reuse paths, counts normalized installment obligations, uses distinct pre-application months, enables row bagging, and checks local artifact identities. Post-v1 roles default to 70% training, 7.5% calibration fitting, 7.5% selection validation, and 15% historical comparison. New corrected full-data evidence and aggregate figures are in [the correctness assessment](correctness_20261004/assessment_report.md); historical PBIX/screenshots remain unchanged. [Current evidence status](../docs/validation/VALIDATION_PLAN.md#current-evidence-status) owns verification and limits; the [remediation plan](../docs/implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) retains broader proposed work.

Legacy PR-AUC means average precision; recall at 10% review capacity means top-10% default capture, not actual middle-review recall. Expected-value fields contain retrospective utility units. "Held-out test" retains the saved within-run split label, but it is a reused comparison population across the historical experiment trail.

## Current Assessment Protocol

The code also provides a separate nested assessment of the declared LightGBM
feature/model-setting/calibration selection procedure. Five outer folds hold out
assessment applicants; inner fitting/calibration/selection roles remain disjoint.
Feature ranking averages model-seed repeats within fitting rows. Partition and
model seeds are separate. This protocol has real-data matched controls, a sampled shuffled-label diagnostic, reliability/segment sensitivities and locked same-host reproduction. It does not replace the historical
numbers below or evaluate future application cohorts. Details belong to the
[assessment methodology](../docs/validation/ASSESSMENT_METHODOLOGY.md).

## Intended Use

The intended use is a portfolio project that demonstrates applied financial ML decision-support:

- rank applicants by repayment-difficulty risk;
- compare LightGBM against a logistic regression baseline;
- evaluate imbalanced-class metrics, lift, calibration, and threshold behavior;
- simulate approval, manual-review, and high-risk action bands;
- export scored applicants and reporting tables for Power BI.

## Non-Use

This model must not be used for:

- automated lending or underwriting decisions;
- real credit approval, pricing, line assignment, or collections;
- legally compliant adverse-action notice generation;
- fair-lending certification;
- production risk management without additional governance, monitoring, compliance review, and validation.

## Data

v1 uses these public Kaggle Home Credit files:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`

Metrics are computed only from labeled splits of `application_train`. Kaggle `application_test` rows are unlabeled and are scored only for production-like batch-scoring demonstration.

## Feature Scope

Feature engineering is SQL-first and produces one row per `(SK_ID_CURR, source_population)` in `mart_credit_risk_features`.

Feature groups include:

- current application attributes and affordability ratios;
- external score aggregates;
- bureau credit-history aggregates;
- previous-application approval/refusal and amount-ratio features;
- installment payment timing and payment-ratio features.

Identifiers, target fields, and v1 demographic/protected-status-like exclusions are removed from the model feature list. Excluded diagnostic fields may be inspected separately for limitation checks, but they are not model drivers.

The historically selected post-v1 candidate extends this scope with bureau-balance, POS-cash, credit-card, recency-deterioration, and last-k behavior features. Its windows counted account records and installment aggregation repeated owed amounts across split payments. Current SQL groups unambiguous due obligations, accumulates pre-application payments, distinguishes incomplete arrears from completed delays, and flags ambiguous versions/unknown amounts. POS/card windows now contain distinct applicant months strictly before month zero. Named corrected model fits use these features; conservative snapshot filtering does not prove production point-in-time availability.

## Training and Selection

The pipeline trains:

1. logistic regression baseline;
2. tuned LightGBM primary model.

LightGBM tuning uses validation metrics with a non-degenerate score guard, then ranks by PR-AUC, top-decile lift, top-score capture, ROC-AUC, and lower Brier score. The baseline-versus-LightGBM family choice uses validation PR-AUC. Test reporting follows within-run selection, but historical cross-run reuse prevents an independent final-test claim. Historical fits left row subsampling disabled; current code sets `subsample_freq=1`. Retraining and stability preserve saved test membership, and stability varies training/calibration/validation roles with training-only gain ranking. None of these repairs changes the historical metrics below.

No Platt/sigmoid or isotonic calibration layer is fitted in v1. Brier score and calibration bins are reported to evaluate score quality, but they do not make the raw LightGBM scores calibrated probabilities.

Post-v1 historically selected a sigmoid layer fitted and assessed on shared validation rows. Current code fits transforms on reserved calibration rows and selects methods on disjoint validation rows, requiring each candidate's minimum gain before applying sigmoid preference. Saved calibrators bind to the parent model and have their own run identity; replacing one requires rescoring before export. Sigmoid preserved ranking in the historical outputs. These fixes do not re-estimate the recorded results or restore independent historical assessment.

| Post-v1 calibration result | Uncalibrated | Sigmoid calibrated | Difference |
|---|---:|---:|---:|
| Validation Brier score | 0.174335 | 0.066500 | -0.107835 |
| Held-out test Brier score | 0.173301 | 0.066460 | -0.106842 |
| Validation weighted bin error | 0.289885 | 0.003634 | -0.286251 |
| Held-out test weighted bin error | 0.288304 | 0.002709 | -0.285595 |

Batch scoring and dashboard exports now retain both `raw_risk_score` and `calibrated_risk_score`, with `calibration_method` documenting the applied sigmoid layer. The original `score` column remains the rank-policy score used by the current threshold workflow. Post-v1 dashboard exports relabel the selected model as `lightgbm_credit_risk_post_v1` so the improved comparison bundle is distinct from frozen v1.

Post-v1 Experiments 005-014 record simplification, repeated-seed stability, pressure interactions, recency, last-k record behavior, and cleanup. Experiment 012 historically promoted the 168-feature setup under its validation ranking rule. Experiments 013 and 014 did not select a smaller SHAP-ranked surface by mean validation PR-AUC. Those decisions explain the saved feature scope; they do not close the methodology gaps or establish a permanently optimal feature set.

The historical comparison also records slightly worse weighted calibration-bin error versus the prior candidate despite lower Brier. This is an additional caveat alongside the assessment, selection, feature, and policy limits above. See [the historical comparison](experiments/v1_to_post_v1_model_diff.md).

Historical v1 selected candidate recorded in [experiment 000](experiments/000_v1_baseline.md). The regenerated `reports/v1/lightgbm_tuning_summary.csv` is a separate local artifact and may differ:

| Candidate | PR-AUC | ROC-AUC | Brier | Top-decile lift | Recall at 10% review capacity |
|---|---:|---:|---:|---:|---:|
| `feature_subsample_regularized` | 0.260173 | 0.770420 | 0.171640 | 3.490643 | 0.349087 |

## Metrics

Frozen v1 LightGBM metrics:

| Split | PR-AUC | ROC-AUC | Brier | Top-decile lift | Recall at 10% review capacity |
|---|---:|---:|---:|---:|---:|
| Validation | 0.260173 | 0.770420 | 0.171640 | 3.490643 | 0.349087 |
| Held-out test | 0.258236 | 0.770385 | 0.171245 | 3.482588 | 0.348281 |

Validation comparison to logistic regression:

| Metric | Logistic regression | LightGBM | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.244617 | 0.260173 | +0.015556 |
| ROC-AUC | 0.757608 | 0.770420 | +0.012812 |
| Brier score | 0.200474 | 0.171640 | -0.028835 |
| Top-decile lift | 3.337592 | 3.490643 | +0.153051 |
| Recall at 10% review capacity | 0.333781 | 0.349087 | +0.015306 |

Post-v1 improvement summary:

| Metric | Frozen v1 | Best post-v1 | Difference |
|---|---:|---:|---:|
| Feature count | 68 | 168 | +100 |
| Validation PR-AUC | 0.260173 | 0.272184 | +0.012011 |
| Validation ROC-AUC | 0.770420 | 0.778732 | +0.008312 |
| Validation Brier score | 0.171640 | 0.066500 | -0.105139 |
| Validation top-decile lift | 3.490643 | 3.659805 | +0.169162 |
| Validation recall at 10% review capacity | 0.349087 | 0.366004 | +0.016917 |
| Validation balanced EV / applicant | 571.52 | 577.24 | +5.72 |

The post-v1 values preserve the curated final dashboard snapshot for the historically selected 168-feature candidate. They may differ from current generated local artifacts. Historical test comparisons are reported in [the model comparison](experiments/v1_to_post_v1_model_diff.md); their reuse prevents independent generalization claims.

## Threshold Policy

Scores are mapped to simulated actions:

| Score range | Risk band | Simulated action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `T_low` to `< T_high` | Medium risk | Manual review |
| `>= T_high` | High risk | High-priority review |

Thresholds are selected from validation scores and applied unchanged to the held-out labeled test split.

The quantile scenarios do not enforce a hard review capacity. "Balanced" is the displayed reference, not an optimized policy. The current utility formula charges only the middle review band while the high band is labeled high-priority review; its cost/disposition is not modeled. Top-score capture and actual middle-review capture are different quantities.

The thresholds below are cutoffs on uncalibrated model scores. They are valid for rank-based scenario comparison in this project, but they should not be interpreted as calibrated default-probability thresholds.

| Scenario | `T_low` | `T_high` | Test approval rate | Test review rate | Test high-risk rate | Test EV / applicant |
|---|---:|---:|---:|---:|---:|---:|
| Growth-oriented | 0.635669 | 0.766724 | 0.8503 | 0.0976 | 0.0520 | 583.62 |
| Balanced | 0.580982 | 0.695323 | 0.8010 | 0.0967 | 0.1023 | 572.03 |
| Risk-averse | 0.485034 | 0.580982 | 0.7009 | 0.1001 | 0.1990 | 537.84 |

## Expected-Value Assumptions

Expected value is illustrative and not a claim about real Home Credit economics.

| Assumption | Value |
|---|---:|
| Expected margin per good approved loan | 1000 |
| Expected loss per bad approved loan | 5000 |
| Manual review cost | 50 |
| Review-rate scenario reference (not a hard cap) | 10% of applicants |

These values are utility weights for scenario comparison, not calibrated loan-level economics. The `1000` good-loan margin and `5000` bad-loan loss encode a simple 5:1 penalty ratio so approval, review, and high-risk threshold choices can be compared in a readable v1 dashboard. They do not estimate actual interest income, funding cost, exposure at default, recovery, loss given default, servicing cost, or loan term.

A production-style value model would use exposure-based assumptions, for example:

```text
good_loan_value = margin_rate * AMT_CREDIT
bad_loan_loss = loss_given_default_rate * AMT_CREDIT
```

Formula:

```text
approved_good_count * expected_margin_per_good_loan
- approved_bad_count * expected_loss_per_bad_loan
- manual_review_count * manual_review_cost
```

## Explainability

SHAP is used for global feature importance and reason-code-style debugging outputs. Top global drivers include external source aggregates, prior application amount ratios, requested credit/goods amounts, employment length, and payment-delay behavior.

SHAP outputs are not adverse-action notices and should not be presented as legally compliant customer explanations.

## Limitations

- The target is a proxy for observed repayment difficulty, not a complete default or loss model.
- The dataset is public, static, and not representative of a live lending environment.
- Expected-value assumptions are simplified scenario parameters.
- Threshold actions are simulated and not policy-approved credit decisions.
- No production monitoring, drift management, fair-lending review, compliance approval, or model governance is implemented.
- The frozen v1 model excludes richer monthly history tables; post-v1 experiments now include them, including recency and last-k temporal candidates. The historical post-v1 candidate was the 168-feature last-k temporal model; the corrected mart has 174 model fields and new fold choices are recorded separately; cleanup experiments did not justify a smaller promoted surface.
- Calibration is evaluated with Brier score and calibration bins; no final Platt or isotonic calibration model is fitted in v1.

## Reproducibility

Use the explicit scoped pipeline commands to regenerate local outputs from downloaded data:

```bash
make pipeline-v1
make pipeline-post-v1
```

Check the code with synthetic fixtures, without raw Kaggle data:

```bash
make lint
make format-check
make test
```

`configs/v1.yaml` writes frozen-v1 artifacts under `models/v1`, `reports/v1`, and `reports/dashboard_data`. `configs/post_v1.yaml` writes post-v1 artifacts under `models/post_v1`, `reports/post_v1`, and `reports/dashboard_data_post_v1`.

See [the run guide](../docs/RUNNING.md) for dependency installation, explicit step configs, and the Windows `PYTHON=python` override. The corrections require regenerating the full scoped pipeline from features through exports. Export-only targets require matching feature/model/evaluation/scoring/calibration identities. `requirements.lock` pins the named run, with clean same-host fold reproduction and CSV reconciliation. Exact historical numbers and native screenshot refresh are not certified. Model-version names, including the post-v1 dashboard display alias, do not guarantee exact fitted-run lineage.

Key generated artifacts:

- `reports/v1/model_metrics_summary.csv`
- `reports/post_v1/model_metrics_summary.csv`
- `reports/v1/lightgbm_tuning_summary.csv`
- `reports/post_v1/lightgbm_tuning_summary.csv`
- `reports/v1/model_threshold_metrics.csv`
- `reports/post_v1/model_threshold_metrics.csv`
- `reports/v1/business_value_analysis.md`
- `reports/post_v1/business_value_analysis.md`
- `reports/v1/model_feature_importance.csv`
- `reports/post_v1/model_feature_importance.csv`
- `reports/dashboard_data/`
- `reports/dashboard_data_post_v1/`
- `reports/v1/`
- `reports/post_v1/`
- `powerbi/archive/screenshots/`

## Current tuning evidence

The named [v3 tuning report](tuning_20261004/assessment_report.md) owns the new
training-only joint-CV results, probability acceptance, seed variation and runtime
limits. The [methodology](../docs/validation/ASSESSMENT_METHODOLOGY.md) specifies
24 LightGBM candidates per workflow, three inner folds and disjoint stopping,
calibration and selection roles. These results remain separate from the preserved
r2 and historical numbers above. Tuned versus fixed logistic identity is explicit;
no outer quality promotes a family. No true chronology, global optimality or
native Power BI refresh is claimed.

</details>
