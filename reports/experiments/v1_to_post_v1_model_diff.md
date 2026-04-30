# V1 to Best Post-v1 Model Diff

## Recruiter Summary

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

The table below compares the frozen v1 baseline with the active post-v1 candidate recorded in Experiment 014. V1 is a frozen single-split baseline. The post-v1 values are repeated-seed validation means for the promoted 168-feature candidate, so this is best read as a portfolio-level improvement summary rather than a perfectly identical single-run comparison.

| Metric | V1 baseline | Best post-v1 | Difference |
|---|---:|---:|---:|
| Feature count | 68 | 168 | +100 |
| Validation PR-AUC | 0.258667 | 0.271879 | +0.013212 |
| Validation ROC-AUC | 0.769216 | 0.780531 | +0.011315 |
| Validation Brier score | 0.171864 | 0.066419 | -0.105445 |
| Validation top-decile lift | 3.506754 | 3.651750 | +0.144996 |
| Validation recall at 10% review capacity | 0.350698 | 0.365199 | +0.014501 |
| Validation balanced EV / applicant | 570.48 | 580.80 | +10.32 |
| Held-out test PR-AUC | 0.257943 | 0.264566 | +0.006623 |
| Held-out test Brier score | 0.171325 | 0.066677 | -0.104648 |
| Held-out test top-decile lift | 3.471847 | 3.610578 | +0.138731 |
| Held-out test recall at 10% review capacity | 0.347207 | 0.361081 | +0.013874 |
| Held-out test balanced EV / applicant | 575.44 | 577.91 | +2.47 |

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

The best recruiter-friendly takeaway is:

> I built a complete v1 credit-risk decision-support pipeline, then improved it through a validation-first experiment trail. The final post-v1 candidate uses calibrated LightGBM scores and recent repayment behavior features, improving PR-AUC, Brier score, lift, review-capacity recall, and expected value while preserving clear documentation of what worked, what did not, and why feature engineering stopped.

## Supporting Reports

- `reports/experiments/000_v1_baseline.md`
- `reports/experiments/004_calibration_experiment.md`
- `reports/experiments/010_recency_model_stability.md`
- `reports/experiments/012_last_k_model_stability.md`
- `reports/experiments/014_feature_cleanup_stability.md`
- `reports/experiments/experiment_log.csv`
