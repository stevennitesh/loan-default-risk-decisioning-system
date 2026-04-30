# Experiment 008: Narrow Risk-Pressure Features

## Purpose

Retest the risk-pressure idea using only the four engineered features that showed meaningful SHAP signal in Experiment 007.

## Change Tested

Kept these SQL interaction features in `f_risk_pressure_features` and `mart_credit_risk_features`:

- `bureau_debt_to_income_ratio`
- `external_score_annuity_pressure`
- `payment_shortfall_ratio`
- `external_score_credit_pressure`

Removed the weaker/sparser Experiment 007 pressure features from the model surface:

- `total_credit_exposure_to_income_ratio`
- `monthly_delinquency_pressure`
- `revolving_utilization_delinquency_pressure`
- `prior_refusal_delay_pressure`

No new source tables or demographic/protected-status-like fields were added.

## Hypothesis

The broad Experiment 007 family may have diluted useful pressure signals. Keeping only the strongest pressure features should preserve signal while reducing noise.

## Results

Comparison target: Experiment 004 calibrated full model with 140 features.

| Metric | Experiment 004 | Experiment 008 | Difference |
|---|---:|---:|---:|
| Feature count | 140 | 144 | +4 |
| Validation PR-AUC | 0.271066 | 0.271644 | +0.000578 |
| Validation ROC-AUC | 0.778197 | 0.779268 | +0.001071 |
| Validation Brier, sigmoid | 0.066535 | 0.066519 | -0.000017 |
| Validation top-decile lift | 3.641009 | 3.641009 | +0.000000 |
| Validation recall at review capacity | 0.364125 | 0.364125 | +0.000000 |
| Validation balanced EV/applicant | 576.07 | 574.90 | -1.17 |
| Test PR-AUC | 0.268457 | 0.268472 | +0.000014 |
| Test Brier, sigmoid | 0.066550 | 0.066496 | -0.000054 |
| Test top-decile lift | 3.592677 | 3.571196 | -0.021481 |
| Test recall at review capacity | 0.359291 | 0.357143 | -0.002148 |
| Test balanced EV/applicant | 584.14 | 583.73 | -0.42 |

## Feature Use

All four retained pressure features were used by the model:

| Feature | SHAP rank |
|---|---:|
| `bureau_debt_to_income_ratio` | 17 |
| `external_score_annuity_pressure` | 25 |
| `payment_shortfall_ratio` | 27 |
| `external_score_credit_pressure` | 43 |

## Interpretation

This is stronger than Experiment 007, but still not a clean replacement for the current full calibrated model. Validation PR-AUC, ROC-AUC, and calibrated Brier improve, while validation lift and recall are tied. The weakness is validation balanced EV, which declines by about `$1.17` per applicant under the illustrative balanced scenario.

Under validation-only selection discipline, this is a promising feature-engineering candidate because the primary ranking metric improves and the retained features have plausible SHAP support. It should get a repeated-seed stability check before promotion because the gains are small and the business-value metric did not improve.

## Next Action

Run a repeated-seed stability comparison between the current full calibrated model and the 144-feature narrow-pressure model. If the validation PR-AUC/Brier improvement persists without worsening lift, recall, or EV materially, promote the narrow-pressure features.
