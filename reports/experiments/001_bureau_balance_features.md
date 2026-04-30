# Experiment 001: Bureau Balance Features

## Purpose

Test whether monthly bureau-status history improves the post-v1 decisioning model beyond the frozen v1 baseline.

## Hypothesis

Adding `bureau_balance.csv` should improve applicant ranking because monthly delinquency, closed, and unknown-status behavior may capture credit-history risk not visible in one-row-per-bureau aggregates.

## Change Tested

- Source tables changed: added `bureau_balance.csv`.
- SQL feature tables changed: added `f_bureau_balance_agg`, joined into `mart_credit_risk_features`.
- Python/modeling files changed: ingestion, feature-build, and data-contract registries now include the new source and aggregate table.
- Config changes: `data_scope_version` is `post_v1_001_bureau_balance`; `source_files` now includes `bureau_balance`.

## Model

| Field | Value |
|---|---|
| Selected candidate | `lighter_weight_calibrated` |
| Data scope | `post_v1_001_bureau_balance` |
| Feature count | 86 |
| Feature count change | +18 |
| New aggregate row count | 134,542 applicants |
| New aggregate duplicate keys | 0 |

## Validation Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.258667 | 0.261040 | +0.002373 |
| ROC-AUC | 0.769216 | 0.768999 | -0.000217 |
| Brier score | 0.171864 | 0.155995 | -0.015869 |
| Top-decile lift | 3.506754 | 3.469162 | -0.037592 |
| Precision at top decile | 0.283113 | 0.280078 | -0.003035 |
| Recall at 10% review capacity | 0.350698 | 0.346939 | -0.003759 |

## Held-Out Test Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.257943 | 0.255433 | -0.002510 |
| ROC-AUC | 0.771017 | 0.769643 | -0.001374 |
| Brier score | 0.171325 | 0.155688 | -0.015637 |
| Top-decile lift | 3.471847 | 3.485273 | +0.013426 |
| Precision at top decile | 0.280295 | 0.281379 | +0.001084 |
| Recall at 10% review capacity | 0.347207 | 0.348550 | +0.001343 |

## Balanced Scenario Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| Validation EV / applicant | 570.48 | 570.35 | -0.13 |
| Test EV / applicant | 575.44 | 571.82 | -3.62 |
| Validation high-risk default capture | 0.3507 | 0.3469 | -0.0038 |
| Test high-risk default capture | 0.3523 | 0.3518 | -0.0005 |

## Feature Exploration Notes

- Missingness risk: the new mart features are missing for about 62% of applicants because many applicants do not have joined monthly bureau-balance rows.
- Top new SHAP drivers: `Bureau balance month count` ranked 45; `Bureau balance earliest month` ranked 54; `Bureau balance dpd 0 count` ranked 60.
- Most delinquency-rate features were lower-impact: `Bureau balance dpd 1plus rate` ranked 80 and `Bureau balance recent dpd 1plus rate` ranked 100.
- Feature-risk check: the new fields are bureau repayment-status aggregates, not direct demographic or protected-status-like fields.

## Conclusion

No clear improvement. The change improved validation PR-AUC and materially improved Brier score on validation and test, but it did not improve the main ranking story consistently: validation lift, precision, recall, and balanced EV declined, while held-out test PR-AUC and ROC-AUC also declined. The held-out test lift and recall gains are real but small, so this should stay as an experiment record rather than replacing the v1 baseline as the default story.

## Next Action

Try the next monthly-history source as a separate experiment: `POS_CASH_balance.csv`, using the same pattern of one applicant-grain aggregate, one pipeline rerun, and one comparison report.
