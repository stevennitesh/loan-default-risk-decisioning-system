# Experiment 003: Credit Card Features

## Purpose

Test whether monthly credit-card balance and payment behavior improves the post-v1 decisioning model.

This is a cumulative experiment: it keeps Experiment 001 bureau-balance features and Experiment 002 POS cash features, then adds `credit_card_balance.csv` as the next single feature family.

## Hypothesis

Adding credit-card monthly balance, utilization, drawing, repayment, and delinquency features should improve ranking quality because revolving-credit behavior can reveal borrower stress that is not fully captured by application, bureau, previous-application, POS cash, or installment-payment aggregates.

## Change Tested

- Source tables changed: added `credit_card_balance.csv`.
- SQL feature tables changed: added `f_credit_card_agg`, joined into `mart_credit_risk_features`.
- Python/modeling files changed: ingestion, feature-build, and data-contract registries now include the credit-card source and aggregate table.
- Config changes: `data_scope_version` is `post_v1_003_credit_card`; `source_files` now includes `credit_card_balance`.

## Model

| Field | Value |
|---|---|
| Selected candidate | `regularized_low_learning_rate` |
| Data scope | `post_v1_003_credit_card` |
| Feature count | 140 |
| Feature count change vs v1 | +72 |
| Feature count change vs Experiment 002 | +33 |
| New aggregate row count | 103,558 applicants |
| New aggregate duplicate keys | 0 |

## Validation Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.258667 | 0.271066 | +0.012399 |
| ROC-AUC | 0.769216 | 0.778197 | +0.008981 |
| Brier score | 0.171864 | 0.175712 | +0.003848 |
| Top-decile lift | 3.506754 | 3.641009 | +0.134255 |
| Precision at top decile | 0.283113 | 0.293952 | +0.010839 |
| Recall at 10% review capacity | 0.350698 | 0.364125 | +0.013427 |

## Held-Out Test Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.257943 | 0.268457 | +0.010514 |
| ROC-AUC | 0.771017 | 0.778805 | +0.007788 |
| Brier score | 0.171325 | 0.174848 | +0.003523 |
| Top-decile lift | 3.471847 | 3.592677 | +0.120830 |
| Precision at top decile | 0.280295 | 0.290050 | +0.009755 |
| Recall at 10% review capacity | 0.347207 | 0.359291 | +0.012084 |

## Balanced Scenario Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| Validation EV / applicant | 570.48 | 576.07 | +5.59 |
| Test EV / applicant | 575.44 | 584.14 | +8.70 |
| Validation high-risk default capture | 0.3507 | 0.3641 | +0.0134 |
| Test high-risk default capture | 0.3523 | 0.3649 | +0.0126 |

## Incremental Comparison To Experiment 002

| Metric | Experiment 002 | Experiment 003 | Difference |
|---|---:|---:|---:|
| Validation PR-AUC | 0.269216 | 0.271066 | +0.001850 |
| Validation top-decile lift | 3.619528 | 3.641009 | +0.021481 |
| Validation recall at 10% review capacity | 0.361976 | 0.364125 | +0.002149 |
| Test PR-AUC | 0.266340 | 0.268457 | +0.002117 |
| Test top-decile lift | 3.571196 | 3.592677 | +0.021481 |
| Test recall at 10% review capacity | 0.357143 | 0.359291 | +0.002148 |
| Validation EV / applicant | 574.25 | 576.07 | +1.82 |
| Test EV / applicant | 575.55 | 584.14 | +8.59 |
| Validation Brier score | 0.168560 | 0.175712 | +0.007152 |
| Test Brier score | 0.168068 | 0.174848 | +0.006780 |

## Feature Exploration Notes

- Coverage is narrower than POS cash: most credit-card features are missing for about 70.93% of applicants.
- Payment-to-minimum features are missing for about 80.38% of applicants, so they should be treated as sparse signals.
- Top new SHAP drivers: `Credit card avg drawing count` ranked 23; `Credit card avg credit utilization` ranked 28; `Credit card balance to limit ratio` ranked 62.
- The most useful credit-card signals were utilization/drawings features rather than raw DPD count features.
- Feature-risk check: the new fields are revolving-credit balance, payment, utilization, and delinquency aggregates, not direct demographic or protected-status-like fields.

## Conclusion

Improved with a calibration tradeoff. Credit-card features improve the ranking and business-value story against both frozen v1 and Experiment 002: validation/test PR-AUC, ROC-AUC, top-decile lift, precision, recall at review capacity, expected value, and high-risk default capture all increase. The drawback is worse Brier score, so this is a stronger ranking model but not a better-calibrated probability model. If this feature set becomes the preferred post-v1 direction, a calibration-focused experiment should follow before presenting probability quality as improved.

## Next Action

Run a calibration comparison on the current best feature set, or first run `004_all_monthly_features` only if we want to test whether adding the remaining monthly table changes ranking enough to justify the extra complexity.
