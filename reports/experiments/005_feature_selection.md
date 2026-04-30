# Experiment 005: Feature Selection

## Purpose

Compare top-N feature subsets against the full post-v1 feature set to see whether the model can keep most of the ranking and calibration gains with a cleaner feature surface.

## Selection Rule

Feature subsets are selected from `reports/model_feature_importance.csv`, mapping human-readable SHAP labels back to raw model columns. The selected setup is chosen on validation results using PR-AUC first, then top-decile lift, recall at review capacity, ROC-AUC, lower Brier score, and finally fewer features as a tie-breaker. Held-out test is not the optimization target; test metrics are reported only after selection to check whether the validation-selected setup generalizes closely enough.

## Results

| Feature set | Feature count | Calibration | Val PR-AUC | Val Brier | Val lift | Val EV/app | Test PR-AUC | Test Brier | Test lift | Test EV/app | Selected |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| top_40 | 40 | sigmoid | 0.264804 | 0.066849 | 3.552401 | 575.16 | 0.262483 | 0.066907 | 3.514809 | 576.92 | False |
| top_60 | 60 | sigmoid | 0.270175 | 0.066605 | 3.641009 | 577.11 | 0.266188 | 0.066631 | 3.573881 | 582.88 | False |
| top_80 | 80 | sigmoid | 0.271671 | 0.066537 | 3.654435 | 576.98 | 0.267303 | 0.066569 | 3.627584 | 581.94 | False |
| top_100 | 100 | sigmoid | 0.272361 | 0.066507 | 3.657120 | 576.33 | 0.268083 | 0.066552 | 3.592677 | 580.98 | True |
| full | 140 | sigmoid | 0.271066 | 0.066535 | 3.641009 | 576.07 | 0.268457 | 0.066550 | 3.592677 | 584.14 | False |

## Selected Setup

Selected feature set: `top_100` with 100 features and `sigmoid` calibration.

Selected raw feature columns are written to `reports/experiments/005_selected_features.csv`.

## Interpretation

`top_100` is the selected setup under the validation-only rule. It has the strongest validation selection score across PR-AUC, top-decile lift, recall at review capacity, ROC-AUC, Brier score, and feature-count tie-breaks. It removes 40 features compared with the full setup.

The full model has the stronger PR-AUC and balanced expected value on held-out test, but held-out test is a final generalization check, not the optimization target. This does not override the validation-selected `top_100` choice; it means the test gap should be recorded as stability evidence. The current gap is small enough to report, not large enough to overrule validation selection; a larger or repeated gap would point to a better model-generation method in a follow-up experiment.

## Notes

This experiment changes the model feature surface only. It does not add new source tables, demographic/protected-status-like fields, or a new decision policy.
