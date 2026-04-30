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

## Current Read

The best fully supported post-v1 story is still the full 140-feature model with sigmoid calibration. Experiment 008 is the strongest feature-engineering candidate so far because it improves validation PR-AUC, ROC-AUC, and calibrated Brier with only four added features, but it has not yet cleared the stability bar and does not improve validation expected value.

The next high-ROI step is a repeated-seed stability check for the 144-feature narrow-pressure setup against the 140-feature calibrated model.
