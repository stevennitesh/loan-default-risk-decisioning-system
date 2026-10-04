# Post-v1 Experiment Summary

This file summarizes the historical post-v1 exploration. Formal sorting used validation metrics, but repeated-seed runs reused original test applicants in fitting, SHAP-ranked selection consumed reporting populations, and calibration fit/selection shared validation rows. The recorded improvements are exploratory comparisons, not independent final-test evidence. See [current evidence status](../../docs/validation/VALIDATION_PLAN.md#current-evidence-status).

The table's decisions and next actions describe the original experiment sequence, not a current execution queue. PR-AUC means average precision; recall means top-score capture; EV means retrospective utility units. Recorded numbers remain unchanged.

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
| 015 | Historical pipeline snapshot | Curated dashboard snapshot reports validation PR-AUC `0.272184`, ROC-AUC `0.778732`, Brier `0.066500`, lift `3.659805`, recall `0.366004`, and balanced EV/applicant `577.24`. | Preserve this numeric snapshot; a deliberate regeneration must identify new artifacts and must not silently overwrite historical evidence. |

## Historical Interpretation

The historical selection retained the 168-feature last-k record-window model with sigmoid calibration. Experiment 012 records improvements in mean validation ranking, PR-AUC variability, Brier, lift, precision, capture, and utility versus the prior 152-feature setup. These summaries compare the historical procedures; they do not close the assessment-boundary or calibration-selection gaps. Row 015 records the curated final single-run snapshot, not necessarily today's local generated bundle.

Mean validation weighted calibration error is slightly worse (`0.002915` vs `0.002870`), and mean historical test weighted calibration error is also slightly worse. Brier score improves in the recorded comparison. The original ranking rule selected the last-k setup; utility and calibration-bin behavior were additional reported tradeoffs, not evidence that the pending calibration-assessment gate passed.

Experiment 011 remains important because it records the source-informed research framing: public solution research suggested recent temporal behavior is a useful mechanism, but the implementation is this project's own compact SQL feature family, evaluated through the existing validation-first process.

Experiments 013 and 014 tested simplification. The full 168-feature setup won the historical mean-validation rule, while `top_152` was close. This explains the saved feature scope; it is not a final choice under the proposed corrected protocol.

The concise [v1/post-v1 comparison](v1_to_post_v1_model_diff.md) preserves the experiment story. Current priorities are correctness repair, artifact reconciliation, and presentation within the existing local scope; the [remediation plan](../../docs/implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) remains a proposal. Feature expansion alone would not resolve the identified gaps.
