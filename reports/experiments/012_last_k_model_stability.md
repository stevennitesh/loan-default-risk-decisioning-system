# Experiment 012 - Last-K Temporal Model Stability

## Purpose

Check whether the Experiment 011 last-k temporal feature setup holds up across repeated split/training seeds before promoting it over the 152-feature recency-deterioration candidate from Experiment 010.

## Process

The stability run used seeds `17`, `29`, and `43`. Each seed created a fresh stratified train/validation/test split from labeled `application_train`, then trained the full 168-feature last-k temporal setup. Selection uses validation aggregates only. Held-out test is reported after the decision as a generalization check.

The comparison baseline is the Experiment 010 152-feature recency-deterioration stability row, which used the same seed set and validation-only aggregate rule.

## Repeated-Seed Result

| Feature setup | Features | Seeds | Val PR-AUC mean | Val PR-AUC std | Val Brier mean | Val lift mean | Val recall mean | Val EV/app mean | Test PR-AUC mean | Test Brier mean | Test lift mean | Test EV/app mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Experiment 010 recency full | 152 | 3 | 0.269003 | 0.004353 | 0.066549 | 3.627584 | 0.362782 | 577.85 | 0.263617 | 0.066779 | 3.599837 | 576.14 |
| Experiment 012 last-k temporal full | 168 | 3 | 0.271879 | 0.003282 | 0.066419 | 3.651750 | 0.365199 | 580.80 | 0.264566 | 0.066677 | 3.610578 | 577.91 |

## Validation Deltas

| Metric | Delta vs Experiment 010 |
|---|---:|
| Validation PR-AUC mean | +0.002876 |
| Validation PR-AUC std | -0.001071 |
| Validation ROC-AUC mean | +0.001568 |
| Validation Brier mean | -0.000130 |
| Validation top-decile lift mean | +0.024166 |
| Validation precision at top decile mean | +0.001951 |
| Validation recall at review capacity mean | +0.002417 |
| Validation weighted calibration error mean | +0.000045 |
| Validation balanced EV / applicant mean | +2.95 |

## Held-Out Test Check

The held-out test aggregates are directionally supportive on ranking, Brier, lift, recall, and expected value, but they are not used to choose the setup.

| Metric | Delta vs Experiment 010 |
|---|---:|
| Test PR-AUC mean | +0.000949 |
| Test ROC-AUC mean | +0.001147 |
| Test Brier mean | -0.000102 |
| Test top-decile lift mean | +0.010740 |
| Test precision at top decile mean | +0.000867 |
| Test recall at review capacity mean | +0.001074 |
| Test weighted calibration error mean | +0.000255 |
| Test balanced EV / applicant mean | +1.77 |

## Conclusion

The 168-feature last-k temporal setup passes the repeated-seed validation check. Mean validation PR-AUC, PR-AUC stability, ROC-AUC, calibrated Brier, top-decile lift, precision, recall, and balanced expected value all improve versus the promoted 152-feature recency setup.

This is a stronger promotion case than Experiment 010 because the validation expected-value metric also improves. The caveat is calibration-bin behavior: weighted calibration error is slightly worse on validation and test, even though Brier improves. That means the candidate should be described as a ranking, Brier, lift, recall, and EV improvement with a small calibration-bin caveat, not as a perfect calibration win.

## Decision

Promote the 168-feature last-k temporal model as the current post-v1 active candidate under the validation-first model-selection rule.

Keep the research framing visible: the feature idea was source-informed by public Home Credit solution patterns, but the implementation is this project's own compact SQL feature family and was accepted only after this validation stability check.

## Artifacts

- `reports/012_last_k_stability_summary.csv`
- `reports/012_last_k_stability_seed_runs.csv`
- `reports/experiments/012_last_k_model_stability.md`
