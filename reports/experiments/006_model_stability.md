# Experiment 006: Model Stability

## Purpose

Check whether the feature-selection result is stable across repeated split/training seeds before promoting a smaller model surface.

## Process

This experiment reruns the same LightGBM tuning and sigmoid/isotonic/uncalibrated calibration comparison across seeds `17, 29, 43`. Each seed creates a fresh stratified train/validation/test split from labeled `application_train`, then trains the candidate feature surfaces independently.

## Selection Rule

The selected setup is chosen with a validation-only aggregate rule: mean validation PR-AUC first, then validation win rate, mean top-decile lift, mean recall at review capacity, mean ROC-AUC, lower mean Brier score, lower validation PR-AUC variability, and fewer features. Held-out test is reported after selection as a generalization check and is not used to choose the setup.

## Results

| Feature set | Features | Seeds | Val win rate | Val PR-AUC mean | Val PR-AUC std | Val Brier mean | Val lift mean | Test PR-AUC mean | Test Brier mean | Test EV/app mean | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| top_40 | 40 | 3 | 0.00 | 0.263460 | 0.003294 | 0.066850 | 3.586412 | 0.257985 | 0.067045 | 574.18 | False |
| top_60 | 60 | 3 | 0.00 | 0.267847 | 0.003272 | 0.066649 | 3.590887 | 0.260091 | 0.066906 | 575.82 | False |
| top_80 | 80 | 3 | 0.33 | 0.267598 | 0.003967 | 0.066657 | 3.603418 | 0.260299 | 0.066894 | 576.05 | False |
| top_100 | 100 | 3 | 0.33 | 0.267635 | 0.003086 | 0.066610 | 3.627584 | 0.261762 | 0.066828 | 576.85 | False |
| full | 140 | 3 | 0.33 | 0.268848 | 0.003168 | 0.066573 | 3.619528 | 0.260816 | 0.066855 | 575.63 | True |

## Selected Setup

Selected feature set: `full` with 140 features.

## Generalization Check

For the selected setup, mean test PR-AUC minus mean validation PR-AUC is -0.008032, and mean test balanced EV minus mean validation balanced EV is -2.52. These held-out test values are final verification signals, not optimization inputs.

## Interpretation

The repeated-seed result does not support promoting the smaller `top_100` surface yet. The full feature set has the strongest mean validation PR-AUC under the validation-only aggregate rule, so it remains the better active model candidate until a smaller setup wins a stability pass or the project explicitly prioritizes simplicity over validation lift.

## Notes

This experiment changes the model-generation evidence only. It does not add new source tables, demographic/protected-status-like fields, or a new decision policy.
