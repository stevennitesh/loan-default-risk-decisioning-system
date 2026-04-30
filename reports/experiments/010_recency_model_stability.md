# Experiment 010 - Recency Model Stability

## Purpose

Check whether the Experiment 009 recency-deterioration feature setup holds up across repeated split/training seeds before promoting it over the calibrated 140-feature active candidate.

## Process

The stability run used seeds `17`, `29`, and `43`. Each seed created a fresh stratified train/validation/test split from labeled `application_train`, then trained the full 152-feature recency-deterioration setup. Selection uses validation aggregates only. Held-out test is reported after the decision as a generalization check.

The comparison baseline is the Experiment 006 full 140-feature calibrated stability row, which used the same seed set and validation-only aggregate rule.

## Repeated-Seed Result

| Feature setup | Features | Seeds | Val PR-AUC mean | Val PR-AUC std | Val Brier mean | Val lift mean | Val recall mean | Val EV/app mean | Test PR-AUC mean | Test Brier mean | Test lift mean | Test EV/app mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Experiment 006 full calibrated | 140 | 3 | 0.268848 | 0.003168 | 0.066573 | 3.619528 | 0.361976 | 578.15 | 0.260816 | 0.066855 | 3.575671 | 575.63 |
| Experiment 010 recency full | 152 | 3 | 0.269003 | 0.004353 | 0.066549 | 3.627584 | 0.362782 | 577.85 | 0.263617 | 0.066779 | 3.599837 | 576.14 |

## Validation Deltas

| Metric | Delta vs 140-feature baseline |
|---|---:|
| Validation PR-AUC mean | +0.000155 |
| Validation PR-AUC std | +0.001185 |
| Validation ROC-AUC mean | +0.001122 |
| Validation Brier mean | -0.000024 |
| Validation top-decile lift mean | +0.008055 |
| Validation precision at top decile mean | +0.000650 |
| Validation recall at review capacity mean | +0.000806 |
| Validation weighted calibration error mean | -0.000108 |
| Validation balanced EV / applicant mean | -0.30 |

## Held-Out Test Check

The held-out test aggregates are close and directionally supportive on most ranking metrics, but they are not used to choose the setup.

| Metric | Delta vs 140-feature baseline |
|---|---:|
| Test PR-AUC mean | +0.002801 |
| Test ROC-AUC mean | +0.000400 |
| Test Brier mean | -0.000077 |
| Test top-decile lift mean | +0.024166 |
| Test precision at top decile mean | +0.001951 |
| Test recall at review capacity mean | +0.002417 |
| Test weighted calibration error mean | +0.000143 |
| Test balanced EV / applicant mean | +0.51 |

## Conclusion

The 152-feature recency-deterioration setup passes the repeated-seed validation check by the existing selection rule. Mean validation PR-AUC, ROC-AUC, Brier, top-decile lift, precision, recall, and weighted calibration error all improve versus the 140-feature calibrated baseline.

The improvement is real but small. Mean validation expected value is slightly lower, and validation PR-AUC variability is higher. That means Experiment 010 should be promoted as the leading post-v1 ranking/calibration candidate, not described as a clean win on every business metric.

## Decision

Promote the 152-feature recency-deterioration model as the current post-v1 active candidate under the validation-first model-selection rule.

Keep the caveat visible: if the project chooses expected value as the dominant business objective rather than PR-AUC/ranking/calibration, the 140-feature calibrated model remains competitive.

## Artifacts

- `reports/010_recency_stability_summary.csv`
- `reports/010_recency_stability_seed_runs.csv`
- `reports/experiments/010_recency_model_stability.md`
