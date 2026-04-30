# Experiment 014 - Feature Cleanup Stability

## Purpose

Check whether the closest cleanup candidate from Experiment 013, `top_152`, is stable enough to replace the full 168-feature last-k temporal setup.

## Process

The stability run used seeds `17`, `29`, and `43`. Each seed created a fresh stratified train/validation/test split from labeled `application_train`, then trained:

- `top_152`: the top 152 SHAP-ranked features from the 168-feature model;
- `full`: the full 168-feature last-k temporal setup.

Selection uses validation aggregates only. Held-out test is reported after the decision as a generalization check.

## Repeated-Seed Result

| Feature set | Features | Seeds | Val win rate | Val PR-AUC mean | Val PR-AUC std | Val Brier mean | Val lift mean | Val recall mean | Val EV/app mean | Test PR-AUC mean | Test EV/app mean | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `top_152` | 152 | 3 | 0.67 | 0.271528 | 0.004406 | 0.066451 | 3.620423 | 0.362066 | 579.49 | 0.263508 | 578.83 | False |
| `full` | 168 | 3 | 0.33 | 0.271879 | 0.003282 | 0.066419 | 3.651750 | 0.365199 | 580.80 | 0.264566 | 577.91 | True |

## Validation Deltas

`top_152` versus `full`:

| Metric | Delta |
|---|---:|
| Feature count | -16 |
| Validation PR-AUC mean | -0.000351 |
| Validation PR-AUC std | +0.001124 |
| Validation ROC-AUC mean | -0.000727 |
| Validation Brier mean | +0.000033 |
| Validation top-decile lift mean | -0.031326 |
| Validation precision at top decile mean | -0.002529 |
| Validation recall at review capacity mean | -0.003133 |
| Validation weighted calibration error mean | +0.000012 |
| Validation balanced EV / applicant mean | -1.30 |

## Held-Out Test Check

The held-out test result is mixed and is not used to choose the setup. `top_152` has slightly higher mean test EV, but full has better mean test PR-AUC, ROC-AUC, Brier, lift, precision, and recall.

| Metric | `top_152` | `full` | Delta |
|---|---:|---:|---:|
| Test PR-AUC mean | 0.263508 | 0.264566 | -0.001058 |
| Test Brier mean | 0.066734 | 0.066677 | +0.000056 |
| Test lift mean | 3.601628 | 3.610578 | -0.008950 |
| Test recall mean | 0.360186 | 0.361081 | -0.000895 |
| Test EV/app mean | 578.83 | 577.91 | +0.91 |

## Conclusion

The cleanup attempt does not justify replacing the full 168-feature setup. `top_152` wins two of three individual validation seeds, but the full model has better mean validation PR-AUC, lower PR-AUC variability, better ROC-AUC, Brier, lift, precision, recall, weighted calibration error, and expected value.

This is the right stopping signal for feature engineering. We pushed the feature surface far enough to find measurable improvement, then tested whether simplification could preserve it. The evidence says not to remove features purely for tidiness.

## Decision

Keep the 168-feature last-k temporal setup as the active post-v1 candidate. Do not create additional feature families for this project unless a future requirement changes the project scope.

## Artifacts

- `reports/014_feature_cleanup_stability_summary.csv`
- `reports/014_feature_cleanup_stability_seed_runs.csv`
- `reports/experiments/014_feature_cleanup_stability.md`
