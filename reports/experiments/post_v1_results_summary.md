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

## Current Read

The best supported post-v1 story is still the full 140-feature model with sigmoid calibration. Feature selection and risk-pressure interactions both found useful signals, but neither has yet cleared the bar for replacing the full calibrated model as the active candidate.

The next high-ROI feature-engineering work should be narrow: isolate the strongest pressure features or test one additional behaviorally meaningful feature family, then require validation improvement across the same primary metrics before promoting it.
