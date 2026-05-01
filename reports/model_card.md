# Model Card: Loan Default Risk Decisioning v1

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

Feature engineering is SQL-first and produces one row per `SK_ID_CURR` in `mart_credit_risk_features`.

Feature groups include:

- current application attributes and affordability ratios;
- external score aggregates;
- bureau credit-history aggregates;
- previous-application approval/refusal and amount-ratio features;
- installment payment timing and payment-ratio features.

Identifiers, target fields, and v1 demographic/protected-status-like exclusions are removed from the model feature list. Excluded diagnostic fields may be inspected separately for limitation checks, but they are not model drivers.

The active post-v1 candidate extends this feature scope with monthly bureau-balance, POS-cash, credit-card, recency-deterioration, and last-k temporal repayment behavior features.

## Training and Selection

The pipeline trains:

1. logistic regression baseline;
2. tuned LightGBM primary model.

LightGBM tuning is bounded and validation-only. Candidate selection uses a non-degenerate score-distribution guard, then ranks by PR-AUC, top-decile lift, recall at manual-review capacity, ROC-AUC, and Brier score. Final test metrics are reported after model and threshold choices are fixed.

No Platt/sigmoid or isotonic calibration layer is fitted in v1. Brier score and calibration bins are reported to evaluate score quality, but they do not make the raw LightGBM scores calibrated probabilities.

Post-v1 fits a separate sigmoid calibration layer on the validation split for the selected LightGBM model. This materially improves probability-quality metrics while preserving rank metrics, and it is documented as an experiment artifact rather than a production probability-of-default model.

| Post-v1 calibration result | Uncalibrated | Sigmoid calibrated | Difference |
|---|---:|---:|---:|
| Validation Brier score | 0.174335 | 0.066500 | -0.107835 |
| Held-out test Brier score | 0.173301 | 0.066460 | -0.106842 |
| Validation weighted bin error | 0.289885 | 0.003634 | -0.286251 |
| Held-out test weighted bin error | 0.288304 | 0.002709 | -0.285595 |

Batch scoring and dashboard exports now retain both `raw_risk_score` and `calibrated_risk_score`, with `calibration_method` documenting the applied sigmoid layer. The original `score` column remains the rank-policy score used by the current threshold workflow. Post-v1 dashboard exports relabel the selected model as `lightgbm_credit_risk_post_v1` so the improved comparison bundle is distinct from frozen v1.

Post-v1 Experiments 005-014 form a validation-first learning trail rather than a leaderboard replication. The sequence tested simplification, repeated-seed stability, risk-pressure interactions, recency deterioration, source-informed last-k temporal repayment behavior, and final cleanup. Experiment 012 promotes the 168-feature last-k temporal setup after repeated-seed validation improved mean PR-AUC, PR-AUC stability, ROC-AUC, calibrated Brier, lift, precision, recall, and balanced expected value versus the prior 152-feature candidate. Experiments 013 and 014 then tested whether a smaller SHAP-ranked surface could preserve those gains; it could not, so feature expansion stops at the 168-feature candidate.

The post-v1 caveat is calibration-bin behavior: weighted calibration error worsens slightly versus the prior post-v1 candidate, even though Brier improves. The full v1-to-post-v1 summary is in `reports/experiments/v1_to_post_v1_model_diff.md`.

Frozen v1 selected candidate from `reports/v1/lightgbm_tuning_summary.csv`:

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

The post-v1 values summarize the frozen final dashboard export for the promoted 168-feature candidate. Held-out test remains a post-selection generalization check and is reported in `reports/experiments/v1_to_post_v1_model_diff.md`.

## Threshold Policy

Scores are mapped to simulated actions:

| Score range | Risk band | Simulated action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `T_low` to `< T_high` | Medium risk | Manual review |
| `>= T_high` | High risk | Decline or high-priority review |

Thresholds are selected from validation scores and applied unchanged to the held-out labeled test split.

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
| Manual review capacity | 10% of applicants |

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
- The frozen v1 model excludes richer monthly history tables; post-v1 experiments now include them, including recency and last-k temporal candidates. The active post-v1 candidate is the 168-feature last-k temporal model; cleanup experiments did not justify a smaller promoted surface.
- Calibration is evaluated with Brier score and calibration bins; no final Platt or isotonic calibration model is fitted in v1.

## Reproducibility

Primary commands:

```bash
make ingest
make features
make train
make evaluate
make score
make dashboard-data
make dashboard-data-post-v1
make test
```

The dashboard comparison bundles are reproducible from the current codebase with two explicit scopes:

```bash
make pipeline-v1
make pipeline-post-v1
```

`configs/v1.yaml` writes frozen-v1 artifacts under `models/v1`, `reports/v1`, and `reports/dashboard_data`. `configs/post_v1.yaml` writes post-v1 artifacts under `models/post_v1`, `reports/post_v1`, and `reports/dashboard_data_post_v1`.

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
- `powerbi/screenshots/`
