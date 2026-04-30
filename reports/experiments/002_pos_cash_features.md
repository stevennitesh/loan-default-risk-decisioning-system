# Experiment 002: POS Cash Features

## Purpose

Test whether monthly point-of-sale cash-loan status history improves the post-v1 decisioning model.

This is a cumulative experiment: it keeps Experiment 001 bureau-balance features and adds `POS_CASH_balance.csv` as the next single feature family.

## Hypothesis

Adding POS cash monthly repayment-status features should improve ranking quality because installment progress, active/completed status patterns, future installment counts, and DPD behavior expose recent repayment stress not captured by the base application, bureau, previous-application, or installment-payment aggregates alone.

## Change Tested

- Source tables changed: added `POS_CASH_balance.csv`.
- SQL feature tables changed: added `f_pos_cash_agg`, joined into `mart_credit_risk_features`.
- Python/modeling files changed: ingestion, feature-build, and data-contract registries now include the POS cash source and aggregate table.
- Config changes: `data_scope_version` is `post_v1_002_pos_cash`; `source_files` now includes `pos_cash_balance`.

## Model

| Field | Value |
|---|---|
| Selected candidate | `feature_subsample_regularized` |
| Data scope | `post_v1_002_pos_cash` |
| Feature count | 107 |
| Feature count change vs v1 | +39 |
| Feature count change vs Experiment 001 | +21 |
| New aggregate row count | 337,252 applicants |
| New aggregate duplicate keys | 0 |

## Validation Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.258667 | 0.269216 | +0.010549 |
| ROC-AUC | 0.769216 | 0.775945 | +0.006729 |
| Brier score | 0.171864 | 0.168560 | -0.003304 |
| Top-decile lift | 3.506754 | 3.619528 | +0.112774 |
| Precision at top decile | 0.283113 | 0.292218 | +0.009105 |
| Recall at 10% review capacity | 0.350698 | 0.361976 | +0.011278 |

## Held-Out Test Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.257943 | 0.266340 | +0.008397 |
| ROC-AUC | 0.771017 | 0.776070 | +0.005053 |
| Brier score | 0.171325 | 0.168068 | -0.003257 |
| Top-decile lift | 3.471847 | 3.571196 | +0.099349 |
| Precision at top decile | 0.280295 | 0.288316 | +0.008021 |
| Recall at 10% review capacity | 0.347207 | 0.357143 | +0.009936 |

## Balanced Scenario Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| Validation EV / applicant | 570.48 | 574.25 | +3.77 |
| Test EV / applicant | 575.44 | 575.55 | +0.11 |
| Validation high-risk default capture | 0.3507 | 0.3620 | +0.0113 |
| Test high-risk default capture | 0.3523 | 0.3604 | +0.0081 |

## Incremental Comparison To Experiment 001

| Metric | Experiment 001 | Experiment 002 | Difference |
|---|---:|---:|---:|
| Validation PR-AUC | 0.261040 | 0.269216 | +0.008176 |
| Validation top-decile lift | 3.469162 | 3.619528 | +0.150366 |
| Validation recall at 10% review capacity | 0.346939 | 0.361976 | +0.015037 |
| Test PR-AUC | 0.255433 | 0.266340 | +0.010907 |
| Test top-decile lift | 3.485273 | 3.571196 | +0.085923 |
| Test recall at 10% review capacity | 0.348550 | 0.357143 | +0.008593 |
| Validation EV / applicant | 570.35 | 574.25 | +3.90 |
| Test EV / applicant | 571.82 | 575.55 | +3.73 |

## Feature Exploration Notes

- Coverage is strong: most POS cash features are missing for about 5.33% of applicants.
- The exception is `pos_cash_recent_dpd_month_rate`, missing for about 32.57% of applicants where no recent POS months exist.
- Top new SHAP drivers: `Pos cash avg future installments` ranked 2; `Pos cash recent month count` ranked 12; `Pos cash active month count` ranked 25.
- Additional useful POS drivers included earliest/latest POS month, completed-month rate, month count, and recent DPD month rate.
- Feature-risk check: the new fields are repayment-history and contract-status aggregates, not direct demographic or protected-status-like fields.

## Conclusion

Improved. POS cash features improve the model story across the main v1 decisioning metrics: validation and held-out test PR-AUC, ROC-AUC, top-decile lift, precision, recall at review capacity, Brier score, and balanced high-risk default capture all improve against the frozen v1 baseline. Expected value improves meaningfully on validation and slightly on held-out test. Compared with Experiment 001, this is a clear incremental lift in ranking performance, though Brier score is less strong than the bureau-balance-only run.

## Next Action

Run `003_credit_card_features` as the next separate monthly-history experiment, then compare whether credit-card repayment behavior adds signal beyond bureau balance plus POS cash.
