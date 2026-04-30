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

This model produces an applicant-level repayment-difficulty risk score. Scores are used to demonstrate threshold tradeoffs, batch scoring, explainability, and Power BI reporting. v1 scores should be treated as ranking scores, not fitted calibrated default probabilities.

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

## Training and Selection

The pipeline trains:

1. logistic regression baseline;
2. tuned LightGBM primary model.

LightGBM tuning is bounded and validation-only. Candidate selection uses a non-degenerate score-distribution guard, then ranks by PR-AUC, top-decile lift, recall at manual-review capacity, ROC-AUC, and Brier score. Final test metrics are reported after model and threshold choices are fixed.

No Platt/sigmoid or isotonic calibration layer is fitted in v1. Brier score and calibration bins are reported to evaluate score quality, but they do not make the raw LightGBM scores calibrated probabilities.

Post-v1 Experiment 004 fits a separate sigmoid calibration layer on the validation split for the Experiment 003 LightGBM model. This materially improves probability-quality metrics while preserving rank metrics, and it is documented as an experiment artifact rather than a v1 production probability-of-default model.

| Post-v1 calibration result | Uncalibrated | Sigmoid calibrated | Difference |
|---|---:|---:|---:|
| Validation Brier score | 0.175712 | 0.066535 | -0.109176 |
| Held-out test Brier score | 0.174848 | 0.066550 | -0.108298 |
| Validation weighted bin error | 0.296805 | 0.002704 | -0.294101 |
| Held-out test weighted bin error | 0.295293 | 0.002823 | -0.292470 |

Batch scoring and dashboard exports now retain both `raw_risk_score` and `calibrated_risk_score`, with `calibration_method` documenting the applied sigmoid layer. The original `score` column remains the rank-policy score used by the current threshold workflow.

Post-v1 Experiments 005-010 explored simplification, stability, risk-pressure interactions, and recency-deterioration features. The first repeated-seed stability check selected the full 140-feature setup over the `top_100` simplification. Experiment 010 then promoted the 152-feature recency-deterioration setup as the leading post-v1 ranking/calibration candidate: repeated-seed mean validation PR-AUC, ROC-AUC, calibrated Brier, lift, precision, recall, and weighted calibration error improved slightly versus the 140-feature calibrated baseline. Mean validation expected value was slightly lower, so the 140-feature model remains a competitive alternative if expected value becomes the dominant selection objective.

Selected candidate from `reports/lightgbm_tuning_summary.csv`:

| Candidate | PR-AUC | ROC-AUC | Brier | Top-decile lift | Recall at 10% review capacity |
|---|---:|---:|---:|---:|---:|
| `feature_subsample_regularized` | 0.258667 | 0.769216 | 0.171864 | 3.506754 | 0.350698 |

## Metrics

| Split | PR-AUC | ROC-AUC | Brier | Top-decile lift | Recall at 10% review capacity |
|---|---:|---:|---:|---:|---:|
| Validation | 0.258667 | 0.769216 | 0.171864 | 3.506754 | 0.350698 |
| Held-out test | 0.257943 | 0.771017 | 0.171325 | 3.471847 | 0.347207 |

Validation comparison to logistic regression:

| Metric | Logistic regression | LightGBM | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.244617 | 0.258667 | +0.014050 |
| ROC-AUC | 0.757608 | 0.769216 | +0.011608 |
| Brier score | 0.200474 | 0.171864 | -0.028610 |
| Top-decile lift | 3.337592 | 3.506754 | +0.169162 |
| Recall at 10% review capacity | 0.333781 | 0.350698 | +0.016917 |

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
| Growth-oriented | 0.634183 | 0.766934 | 0.8490 | 0.0987 | 0.0522 | 582.09 |
| Balanced | 0.581632 | 0.694617 | 0.8008 | 0.0973 | 0.1019 | 575.44 |
| Risk-averse | 0.485847 | 0.581632 | 0.7013 | 0.0995 | 0.1992 | 539.41 |

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
- The frozen v1 model excludes richer monthly history tables; post-v1 experiments now include them, but promotion depends on validation stability.
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
make test
```

Key generated artifacts:

- `reports/model_metrics_summary.csv`
- `reports/lightgbm_tuning_summary.csv`
- `reports/model_threshold_metrics.csv`
- `reports/business_value_analysis.md`
- `reports/model_feature_importance.csv`
- `reports/dashboard_data/`
- `powerbi/screenshots/`
