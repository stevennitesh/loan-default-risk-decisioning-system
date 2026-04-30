# Experiment 000: v1 Baseline

## Purpose

Freeze the completed v1 model and reporting metrics before post-v1 feature engineering, calibration, or model-family changes begin.

This baseline is the comparison point for all later experiments.

## Change Tested

None. This is the completed v1 system.

## Model

| Field | Value |
|---|---|
| Model version | `lightgbm_credit_risk_v1` |
| Model type | LightGBM binary classifier |
| Selected candidate | `feature_subsample_regularized` |
| Data scope | v1 source files |
| Feature count | 68 |

v1 source files:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`

## Validation Metrics

| Metric | Value |
|---|---:|
| PR-AUC | 0.258667 |
| ROC-AUC | 0.769216 |
| Brier score | 0.171864 |
| Top-decile lift | 3.506754 |
| Precision at top decile | 0.283113 |
| Recall at 10% review capacity | 0.350698 |

## Held-Out Test Metrics

| Metric | Value |
|---|---:|
| PR-AUC | 0.257943 |
| ROC-AUC | 0.771017 |
| Brier score | 0.171325 |
| Top-decile lift | 3.471847 |
| Precision at top decile | 0.280295 |
| Recall at 10% review capacity | 0.347207 |

## Balanced Scenario Metrics

| Split | Approval rate | Review rate | High-risk rate | High-risk default capture | EV / applicant |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8000 | 0.1000 | 0.1000 | 0.3507 | 570.48 |
| Held-out test | 0.8008 | 0.0973 | 0.1019 | 0.3523 | 575.44 |

## Top SHAP Drivers

Top global drivers from `reports/model_feature_importance.csv`:

1. Ext source mean
2. Avg credit to application ratio
3. Amt goods price
4. Amt credit
5. Ext source max
6. Ext source min
7. Employment length days
8. Max payment delay days
9. Name education type: Higher education
10. Amt annuity

## Conclusion

This is the frozen v1 benchmark. Future experiments should be judged by whether they improve the decisioning story relative to this baseline, especially PR-AUC, top-decile lift, recall at review capacity, Brier score, and balanced-scenario expected value.

No post-v1 change should be considered an improvement if it only increases one metric while materially degrading calibration, test stability, or business-value behavior.
