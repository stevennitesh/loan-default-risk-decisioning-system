# Experiment 007: Risk-Pressure Interaction Features

## Purpose

Test whether compact cross-domain pressure features improve the current post-v1 full-feature model beyond the calibrated 140-feature setup.

## Change Tested

Added `f_risk_pressure_features` as a SQL feature family and joined eight new features into `mart_credit_risk_features`:

- `external_score_credit_pressure`
- `external_score_annuity_pressure`
- `total_credit_exposure_to_income_ratio`
- `bureau_debt_to_income_ratio`
- `monthly_delinquency_pressure`
- `revolving_utilization_delinquency_pressure`
- `prior_refusal_delay_pressure`
- `payment_shortfall_ratio`

These combine existing application, bureau, bureau-balance, POS-cash, credit-card, previous-application, and installment signals. No new source tables or demographic/protected-status-like fields were added.

## Hypothesis

Risk may be better captured by interactions between weak external scores, affordability pressure, prior debt, delinquency, and repayment shortfall than by the raw component features alone.

## Results

Comparison target: Experiment 004 calibrated full model with 140 features.

| Metric | Experiment 004 | Experiment 007 | Difference |
|---|---:|---:|---:|
| Feature count | 140 | 148 | +8 |
| Validation PR-AUC | 0.271066 | 0.272038 | +0.000973 |
| Validation ROC-AUC | 0.778197 | 0.777827 | -0.000370 |
| Validation Brier, sigmoid | 0.066535 | 0.066542 | +0.000007 |
| Validation top-decile lift | 3.641009 | 3.638324 | -0.002685 |
| Validation recall at review capacity | 0.364125 | 0.363856 | -0.000269 |
| Validation balanced EV/applicant | 576.07 | 575.81 | -0.26 |
| Test PR-AUC | 0.268457 | 0.266119 | -0.002339 |
| Test Brier, sigmoid | 0.066550 | 0.066632 | +0.000082 |
| Test top-decile lift | 3.592677 | 3.595362 | +0.002685 |
| Test recall at review capacity | 0.359291 | 0.359560 | +0.000269 |
| Test balanced EV/applicant | 584.14 | 581.79 | -2.35 |

## Feature Use

Several engineered features were used by the model:

| Feature | SHAP rank |
|---|---:|
| `bureau_debt_to_income_ratio` | 20 |
| `external_score_annuity_pressure` | 25 |
| `payment_shortfall_ratio` | 26 |
| `external_score_credit_pressure` | 35 |
| `total_credit_exposure_to_income_ratio` | 47 |
| `prior_refusal_delay_pressure` | 71 |
| `monthly_delinquency_pressure` | 92 |
| `revolving_utilization_delinquency_pressure` | 163 |

## Interpretation

This is a mixed result, not a clean improvement. The new features improve validation PR-AUC, which is the primary optimization metric, and the SHAP ranks show that some engineered pressure features carry signal. However, validation lift, recall, Brier score, and balanced expected value are flat to slightly worse, and held-out test PR-AUC declines.

Under the validation-only discipline, this experiment is worth keeping as evidence that interaction features can add ranking signal. It should not replace the current full calibrated model story without a stability pass or a tighter feature family.

## Next Action

Run a narrower follow-up or stability check before promoting these features. The strongest candidates to keep exploring are `bureau_debt_to_income_ratio`, `external_score_annuity_pressure`, `payment_shortfall_ratio`, and `external_score_credit_pressure`.
