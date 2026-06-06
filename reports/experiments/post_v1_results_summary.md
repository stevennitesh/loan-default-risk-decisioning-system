# Post-v1 Experiment Summary

This file summarizes the post-v1 model-improvement trail. Selection decisions use validation results only; held-out test results are reported after the decision as a generalization check.

| ID | Change | Validation result | Decision |
|---|---|---|---|
| 001 | Bureau-balance monthly features | PR-AUC improved slightly, but ranking and business-value metrics were mixed. | Keep as evidence, not a clear standalone improvement. |
| 002 | POS-cash monthly features | PR-AUC, lift, recall, and expected value improved versus prior setup. | Keep. |
| 003 | Credit-card monthly features | Ranking and business-value metrics improved, but uncalibrated Brier worsened. | Keep, with calibration follow-up. |
| 004 | Sigmoid calibration | Brier score and calibration-bin error improved sharply with no ranking loss. | Keep as strongest post-v1 improvement. |
| 005 | SHAP-ranked feature selection | `top_100` won on one validation split and reduced the feature surface. | Promising simplification, but not enough alone. |
| 006 | Repeated-seed model stability | The full 140-feature setup had the best mean validation PR-AUC across seeds. | Keep full model as active candidate; do not promote `top_100` yet. |
| 007 | Risk-pressure interaction features | Validation PR-AUC improved slightly, but lift, recall, Brier, and EV were flat to slightly worse. | Keep as mixed feature-engineering evidence; do not promote without stability. |
| 008 | Narrow risk-pressure features | Validation PR-AUC, ROC-AUC, and calibrated Brier improved; lift and recall tied; validation EV declined slightly. | Promising candidate; run stability before promotion. |
| 009 | Recency-deterioration features | Validation PR-AUC, calibrated Brier, and balanced EV improved; lift and recall tied. | Strongest one-shot feature candidate so far; run stability before promotion. |
| 010 | Recency model stability | Repeated-seed validation PR-AUC, Brier, lift, and recall improved slightly versus the 140-feature calibrated baseline; validation EV was slightly lower and PR-AUC variance higher. | Promote as leading post-v1 ranking/calibration candidate, with EV caveat. |
| 011 | Last-k temporal behavior features | Source-informed recent behavior features improved one-shot validation PR-AUC, ROC-AUC, calibrated Brier, lift, precision, and recall versus the 152-feature recency setup; validation EV declined slightly. | Strongest one-shot ranking candidate so far; do not promote until repeated-seed stability confirms it. |
| 012 | Last-k temporal model stability | Repeated-seed validation PR-AUC, PR-AUC stability, ROC-AUC, calibrated Brier, lift, precision, recall, and EV improved versus the promoted 152-feature recency setup; weighted calibration error worsened slightly. | Promote as leading post-v1 candidate, with calibration-bin caveat. |
| 013 | Feature cleanup top-N comparison | Smaller SHAP-ranked surfaces did not beat the full 168-feature setup by the validation ranking rule; `top_152` was closest and improved one-shot validation EV. | Do not promote from one-shot cleanup; run focused stability for `top_152`. |
| 014 | Feature cleanup stability | `top_152` won two of three individual seeds, but full 168 had better mean validation PR-AUC, lower variance, Brier, lift, recall, calibration-bin error, and EV. | Keep 168-feature active candidate; stop feature expansion for this project. |
| 015 | Final pipeline freeze | Frozen dashboard artifacts report validation PR-AUC `0.272184`, ROC-AUC `0.778732`, Brier `0.066500`, lift `3.659805`, recall `0.366004`, and balanced EV/applicant `577.24`. | Use this row for final dashboard/docs alignment; do not rerun the pipeline. |

## Current Read

The best fully supported post-v1 story is now the 168-feature last-k temporal model with sigmoid calibration. Experiment 012 clears the repeated-seed validation check by the existing validation-first selection rule: mean validation PR-AUC, PR-AUC stability, ROC-AUC, calibrated Brier, top-decile lift, precision, recall, and balanced expected value all improve versus the promoted 152-feature recency setup. Experiment 015 records the final frozen single-run artifacts used by the dashboard and final docs.

This is not a clean win on every metric. Mean validation weighted calibration error is slightly worse (`0.002915` vs `0.002870`), and mean held-out test weighted calibration error is also slightly worse. Brier score improves, so this is a calibration-bin caveat rather than a broad probability-quality failure. Under the current ranking/calibration/business-value rule, the last-k temporal model becomes the leading post-v1 candidate.

Experiment 011 remains important because it records the source-informed research framing: public solution research suggested recent temporal behavior is a useful mechanism, but the implementation is this project's own compact SQL feature family, evaluated through the existing validation-first process.

Experiments 013 and 014 tested whether the project could simplify the active feature surface. The evidence does not support promoting a smaller model: `top_152` is close, but the full 168-feature setup has better repeated-seed validation aggregates across the main ranking, calibration, lift, recall, and expected-value metrics. This is a reasonable stopping point for post-v1 feature engineering.

The active 168-feature model story is packaged in `reports/experiments/v1_to_post_v1_model_diff.md` as the concise comparison of v1 versus the best post-v1 candidate. The remaining high-ROI work is final artifact consistency and presentation polish, not more feature expansion.
