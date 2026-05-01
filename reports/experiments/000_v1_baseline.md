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
| PR-AUC | 0.260173 |
| ROC-AUC | 0.770420 |
| Brier score | 0.171640 |
| Top-decile lift | 3.490643 |
| Precision at top decile | 0.281812 |
| Recall at 10% review capacity | 0.349087 |

Calibration note: v1 reports Brier score and calibration bins, but it does not fit a Platt/sigmoid or isotonic calibration layer. Scores should be treated as ranking scores, not calibrated default probabilities.

## Held-Out Test Metrics

| Metric | Value |
|---|---:|
| PR-AUC | 0.258236 |
| ROC-AUC | 0.770385 |
| Brier score | 0.171245 |
| Top-decile lift | 3.482588 |
| Precision at top decile | 0.281162 |
| Recall at 10% review capacity | 0.348281 |

## Balanced Scenario Metrics

| Split | Approval rate | Review rate | High-risk rate | High-risk default capture | EV / applicant |
|---|---:|---:|---:|---:|---:|
| Validation | 0.8000 | 0.1000 | 0.1000 | 0.3491 | 571.52 |
| Held-out test | 0.8010 | 0.0967 | 0.1023 | 0.3539 | 572.03 |

## Top SHAP Drivers

Top global drivers from `reports/model_feature_importance.csv`:

1. Ext source mean
2. Avg credit to application ratio
3. Amt goods price
4. Ext source max
5. Amt credit
6. Employment length days
7. Ext source min
8. Max payment delay days
9. Name education type: Higher education
10. Total credit debt

## Conclusion

This is the frozen v1 benchmark. Future experiments should be judged by whether they improve the decisioning story relative to this baseline, especially PR-AUC, top-decile lift, recall at review capacity, Brier score, and balanced-scenario expected value.

No post-v1 change should be considered an improvement if it only increases one metric while materially degrading calibration, test stability, or business-value behavior.
