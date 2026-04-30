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

## Current Read

The best fully supported post-v1 story is now the 152-feature recency-deterioration model with sigmoid calibration. Experiment 010 clears the repeated-seed validation check by the existing validation-first selection rule: mean validation PR-AUC, ROC-AUC, calibrated Brier, top-decile lift, precision, recall, and weighted calibration error all improve slightly versus the 140-feature calibrated baseline.

This is not a clean win on every metric. Mean validation expected value is slightly lower (`577.85` vs `578.15` EV/app), and validation PR-AUC variability is higher. If the project chooses expected value as the dominant objective, the 140-feature calibrated model remains competitive. Under the current ranking/calibration-first rule, the recency model becomes the leading post-v1 candidate.

The next high-ROI step is to keep the 152-feature setup as the active candidate and consider one narrow cleanup pass around the weakest recency deltas only if we want to reduce variance or simplify the feature surface.
