# V1 to Best Post-v1 Model Diff

## Executive Summary

The v1 model was a complete end-to-end decision-support baseline: SQL feature mart, LightGBM training, threshold analysis, SHAP explainability, batch scoring, and Power BI outputs. Post-v1 work did not change the project into a leaderboard exercise. It used controlled experiments to answer a narrower question:

> Can richer repayment-history features and calibration improve the decisioning story without making the model surface unnecessarily complex?

The answer is yes. The best supported post-v1 candidate is the 168-feature last-k temporal model with sigmoid calibration. It improves ranking, calibration, lift, review-capacity recall, and expected value versus v1. A final cleanup pass tested whether the model could be simplified, but the smaller surfaces did not beat the full 168-feature setup on repeated-seed validation aggregates.

## Model Diff

| Area | V1 baseline | Best post-v1 candidate |
|---|---|---|
| Feature count | 68 | 168 |
| Feature scope | Application, bureau, previous-application, and installment aggregates | V1 scope plus bureau-balance, POS-cash, credit-card, recency-deterioration, and last-k temporal behavior |
| Score treatment | Raw LightGBM ranking score; calibration evaluated but not fitted | Raw ranking score retained plus sigmoid calibrated score |
| Selection discipline | Frozen v1 validation/test split | Validation-first experiments plus repeated-seed stability checks |
| Active decision | Complete v1 project baseline | Promoted post-v1 candidate after stability and cleanup checks |
| Main caveat | Raw scores should not be read as calibrated default probabilities | Weighted calibration-bin error is slightly worse than the prior post-v1 candidate, even though Brier improves |

## Metric Diff

The table below compares the frozen v1 dashboard export with the frozen post-v1 dashboard export. Historical experiments still document the learning trail, including repeated-seed stability checks, but these final values are the current artifacts that feed the Power BI comparison bundle.

| Metric | V1 baseline | Best post-v1 | Difference |
|---|---:|---:|---:|
| Feature count | 68 | 168 | +100 |
| Validation PR-AUC | 0.260173 | 0.272184 | +0.012011 |
| Validation ROC-AUC | 0.770420 | 0.778732 | +0.008312 |
| Validation Brier score | 0.171640 | 0.066500 | -0.105139 |
| Validation top-decile lift | 3.490643 | 3.659805 | +0.169162 |
| Validation recall at 10% review capacity | 0.349087 | 0.366004 | +0.016917 |
| Validation balanced EV / applicant | 571.52 | 577.24 | +5.72 |
| Held-out test PR-AUC | 0.258236 | 0.269925 | +0.011689 |
| Held-out test ROC-AUC | 0.770385 | 0.780208 | +0.009823 |
| Held-out test Brier score | 0.171245 | 0.066460 | -0.104786 |
| Held-out test top-decile lift | 3.482588 | 3.600733 | +0.118145 |
| Held-out test recall at 10% review capacity | 0.348281 | 0.360097 | +0.011815 |
| Held-out test balanced EV / applicant | 572.03 | 581.58 | +9.55 |

Lower Brier score is better. The large Brier improvement is mainly the result of adding sigmoid calibration, not just adding more features.

## How We Got There

| Step | What was tried | What we learned |
|---|---|---|
| V1 baseline | Built the complete baseline with SQL features, LightGBM, thresholding, scoring, SHAP, and dashboard exports. | The workflow was complete, but the model still left room for better repayment-history signal and calibrated score quality. |
| Monthly behavior tables | Added bureau-balance, POS-cash, and credit-card feature families. | POS-cash and credit-card behavior added ranking signal; bureau-balance alone was weaker. Richer history helped, but not every new source improved every metric. |
| Calibration | Added sigmoid calibration after seeing uncalibrated Brier behavior. | Calibration was the cleanest improvement: probability-quality metrics improved sharply without changing rank metrics. |
| Feature selection | Tested smaller SHAP-ranked feature surfaces. | Simpler was not automatically better. A one-shot top-N result was not enough to promote without stability. |
| Stability checks | Re-ran candidates across seeds `17`, `29`, and `43`. | Some apparent wins were split-sensitive. Repeated-seed validation made the active-candidate story more credible. |
| Pressure features | Tested broad and narrowed interaction features. | Plausible financial interactions were not enough by themselves. The model needed behavior over time, not just static pressure ratios. |
| Recency features | Added recent-vs-lifetime deterioration signals. | Recent repayment deterioration helped and became a promoted candidate, but the gains were still modest. |
| Last-k temporal features | Added source-informed last-3 and last-loan repayment behavior features in SQL. | Recent repayment behavior was the strongest feature-engineering direction: it improved repeated-seed PR-AUC, Brier, lift, recall, and EV. |
| Cleanup | Tested `top_100`, `top_120`, `top_140`, `top_152`, and full 168-feature surfaces. | The cleanup attempt did not justify removing features. The full 168-feature model stayed stronger on repeated-seed validation aggregates. |

## Final Read

The post-v1 work shows a real learning loop:

- We did not just keep adding features. We tested new sources, calibration, stability, interactions, recency, temporal behavior, and cleanup.
- We used validation results for selection and held-out test only as a generalization check.
- We kept caveats visible: calibration-bin error is not perfect, expected-value assumptions are illustrative, and this is not a production underwriting model.
- We stopped feature expansion once the cleanup experiment showed that further complexity was not justified for this project.

The best concise takeaway is:

> I built a complete v1 credit-risk decision-support pipeline, then improved it through a validation-first experiment trail. The final post-v1 candidate uses calibrated LightGBM scores and recent repayment behavior features, improving PR-AUC, Brier score, lift, review-capacity recall, and expected value while preserving clear documentation of what worked, what did not, and why feature engineering stopped.

## Supporting Reports

- `reports/experiments/000_v1_baseline.md`
- `reports/experiments/004_calibration_experiment.md`
- `reports/experiments/010_recency_model_stability.md`
- `reports/experiments/012_last_k_model_stability.md`
- `reports/experiments/014_feature_cleanup_stability.md`
- `reports/experiments/experiment_log.csv`
