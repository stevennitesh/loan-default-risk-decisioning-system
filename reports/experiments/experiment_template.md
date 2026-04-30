# Experiment NNN: Short Name

## Purpose

State the single change being tested and why it might improve the model or decisioning story.

## Hypothesis

Example:

> Adding monthly bureau delinquency status features will improve PR-AUC and top-decile lift because recent delinquency patterns should improve ranking of repayment-difficulty risk.

## Change Tested

- Source tables changed:
- SQL feature tables changed:
- Python/modeling files changed:
- Config changes:

## Validation Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.258667 |  |  |
| ROC-AUC | 0.769216 |  |  |
| Brier score | 0.171864 |  |  |
| Top-decile lift | 3.506754 |  |  |
| Precision at top decile | 0.283113 |  |  |
| Recall at 10% review capacity | 0.350698 |  |  |

## Held-Out Test Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.257943 |  |  |
| ROC-AUC | 0.771017 |  |  |
| Brier score | 0.171325 |  |  |
| Top-decile lift | 3.471847 |  |  |
| Precision at top decile | 0.280295 |  |  |
| Recall at 10% review capacity | 0.347207 |  |  |

## Balanced Scenario Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| Validation EV / applicant | 570.48 |  |  |
| Test EV / applicant | 575.44 |  |  |
| Validation high-risk default capture | 0.3507 |  |  |
| Test high-risk default capture | 0.3523 |  |  |

## Feature Exploration Notes

- Feature count change:
- Largest missingness risks:
- Top new SHAP drivers:
- SHAP rank changes:
- Any excluded/protected-status-like feature risk:

## Conclusion

Choose one:

- Improved
- No clear improvement
- Worse
- Inconclusive

Explain the evidence in one paragraph.

## Next Action

State the next single experiment or cleanup action.
