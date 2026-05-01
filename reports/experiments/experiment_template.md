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
| PR-AUC | 0.260173 |  |  |
| ROC-AUC | 0.770420 |  |  |
| Brier score | 0.171640 |  |  |
| Top-decile lift | 3.490643 |  |  |
| Precision at top decile | 0.281812 |  |  |
| Recall at 10% review capacity | 0.349087 |  |  |

## Held-Out Test Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| PR-AUC | 0.258236 |  |  |
| ROC-AUC | 0.770385 |  |  |
| Brier score | 0.171245 |  |  |
| Top-decile lift | 3.482588 |  |  |
| Precision at top decile | 0.281162 |  |  |
| Recall at 10% review capacity | 0.348281 |  |  |

## Balanced Scenario Metrics

| Metric | v1 baseline | Experiment | Difference |
|---|---:|---:|---:|
| Validation EV / applicant | 571.52 |  |  |
| Test EV / applicant | 572.03 |  |  |
| Validation high-risk default capture | 0.3491 |  |  |
| Test high-risk default capture | 0.3539 |  |  |

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
