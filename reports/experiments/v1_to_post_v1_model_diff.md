# Historical V1 to Post-v1 Model Comparison

## Executive Summary

The v1 model was a complete end-to-end decision-support baseline: SQL feature mart, LightGBM training, threshold analysis, SHAP explainability, batch scoring, and Power BI outputs. Post-v1 work did not change the project into a leaderboard exercise. It used controlled experiments to answer a narrower question:

> Can richer repayment-history features and calibration improve the decisioning story without making the model surface unnecessarily complex?

The historical experiments retained a 168-feature last-k record-window model with sigmoid calibration and recorded better ranking and probability-quality metrics. Smaller SHAP-ranked feature surfaces did not win by the historical mean-validation rule. These are exploratory findings, not a verified independent-test improvement: repeated-seed runs reused original test applicants in fitting, selection importance used reporting populations, and calibration fitting and selection shared validation rows.

This is the historical state preceding the repairs completed on 2026-10-04. Source semantics, fitting/selection boundaries and artifact checks were subsequently repaired and assessed; broader proposals remain distinct from completed work. Start with [the current case study](../portfolio/case_study.md), [current assessment](../tuning_20261004/assessment_report.md) and [current evidence status](../../docs/validation/VALIDATION_PLAN.md#current-evidence-status). The [earlier repaired assessment](../correctness_20261004/assessment_report.md) preserves the prior corrected protocol. Historical numbers and conclusions below remain unchanged.

## Model Diff

| Area | V1 baseline | Best post-v1 candidate |
|---|---|---|
| Feature count | 68 | 168 |
| Feature scope | Application, bureau, previous-application, and installment aggregates | V1 scope plus bureau-balance, POS-cash, credit-card, recency-deterioration, and last-k temporal behavior |
| Score treatment | Raw LightGBM ranking score; calibration evaluated but not fitted | Raw ranking score retained plus sigmoid calibrated score |
| Selection procedure | Saved within-run stratified split | Validation-based sorting; repeated-seed re-splitting and reporting-population importance limit independent assessment |
| Active decision | Complete v1 project baseline | Promoted post-v1 candidate after stability and cleanup checks |
| Main caveat | Raw scores are ranking scores; installment and policy semantics require repair | Shared calibration fit/selection, reused assessment applicants, reporting-population selection, and unresolved feature/policy semantics |

## Metric Diff

The table preserves curated experiment-log rows `000` and `015`, which record the historical final dashboard comparison. Current local exports can differ. PR-AUC is average precision, recall-at-capacity is highest-score capture rather than middle-review capture, and EV is retrospective utility per applicant rather than real currency profit. "Held-out test" retains the original within-run split label; across these experiments that population was reused.

| Metric | V1 baseline | Best post-v1 | Difference |
|---|---:|---:|---:|
| Feature count | 68 | 168 | +100 |
| Validation PR-AUC | 0.260173 | 0.272184 | +0.012011 |
| Validation ROC-AUC | 0.770420 | 0.778732 | +0.008312 |
| Validation Brier score | 0.171640 | 0.066500 | -0.105139 |
| Validation top-decile lift | 3.490643 | 3.659805 | +0.169162 |
| Validation top-10% default capture | 0.349087 | 0.366004 | +0.016917 |
| Validation balanced utility / applicant | 571.52 | 577.24 | +5.72 |
| Held-out test PR-AUC | 0.258236 | 0.269925 | +0.011689 |
| Held-out test ROC-AUC | 0.770385 | 0.780208 | +0.009823 |
| Held-out test Brier score | 0.171245 | 0.066460 | -0.104786 |
| Held-out test top-decile lift | 3.482588 | 3.600733 | +0.118145 |
| Historical test top-10% default capture | 0.348281 | 0.360097 | +0.011815 |
| Historical test balanced utility / applicant | 572.03 | 581.58 | +9.55 |

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
- Formal selection sorted validation metrics, but cross-run fitting reuse and reporting-population SHAP prevent an independent generalization claim.
- We kept caveats visible: calibration-bin error is not perfect, expected-value assumptions are illustrative, and this is not a production underwriting model.
- We stopped feature expansion once the cleanup experiment showed that further complexity was not justified for this project.

The best concise takeaway is:

> I built a SQL-to-Power-BI credit-risk portfolio with reproducible command interfaces, logistic regression and LightGBM comparisons, calibration experiments, and batch scoring. The historical experiment trail records gains and unsuccessful simplification attempts. A correctness review identified assessment reuse, feature-grain, calibration, and action-value gaps, which are documented separately from proposed repairs.

## Supporting Reports

- `reports/experiments/000_v1_baseline.md`
- `reports/experiments/004_calibration_experiment.md`
- `reports/experiments/010_recency_model_stability.md`
- `reports/experiments/012_last_k_model_stability.md`
- `reports/experiments/014_feature_cleanup_stability.md`
- `reports/experiments/experiment_log.csv`
