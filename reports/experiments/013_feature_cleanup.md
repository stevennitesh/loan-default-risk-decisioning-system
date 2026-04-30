# Experiment 013 - Feature Cleanup Comparison

## Purpose

Test whether the 168-feature last-k temporal model can be simplified without giving back the validation gains that justified Experiment 012.

This is a cleanup experiment, not another feature-engineering expansion. The goal is to reduce complexity only if a smaller SHAP-ranked surface remains competitive on validation ranking, calibration, lift, recall, and expected value.

## Process

Compared SHAP-ranked top-N feature surfaces from the current 168-feature model:

- `top_100`
- `top_120`
- `top_140`
- `top_152`
- `full`

The selected setup is chosen using validation results only: PR-AUC first, then top-decile lift, recall at review capacity, ROC-AUC, lower Brier score, and fewer features as the final tie-breaker. Held-out test is reported after selection as a generalization check.

## Validation Result

| Feature set | Features | Val PR-AUC | Val Brier | Val lift | Val recall | Val EV/app | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| `top_100` | 100 | 0.272520 | 0.066510 | 3.649064 | 0.364930 | 577.11 | False |
| `top_120` | 120 | 0.271801 | 0.066470 | 3.619528 | 0.361976 | 577.63 | False |
| `top_140` | 140 | 0.272407 | 0.066452 | 3.659805 | 0.366004 | 577.37 | False |
| `top_152` | 152 | 0.272938 | 0.066392 | 3.686656 | 0.368690 | 579.32 | False |
| `full` | 168 | 0.274934 | 0.066380 | 3.697396 | 0.369764 | 576.46 | True |

The full 168-feature setup remains the validation-selected candidate. `top_152` is the closest cleanup candidate and has better one-shot validation EV, but it gives back PR-AUC, lift, recall, and Brier versus the full setup.

## Held-Out Test Check

| Feature set | Test PR-AUC | Test Brier | Test lift | Test recall | Test EV/app |
|---|---:|---:|---:|---:|---:|
| `top_100` | 0.266362 | 0.066594 | 3.619528 | 0.361976 | 581.13 |
| `top_120` | 0.268477 | 0.066478 | 3.659805 | 0.366004 | 582.46 |
| `top_140` | 0.270304 | 0.066440 | 3.619528 | 0.361976 | 583.47 |
| `top_152` | 0.272193 | 0.066369 | 3.673230 | 0.367347 | 582.09 |
| `full` | 0.271270 | 0.066403 | 3.627584 | 0.362782 | 583.10 |

The held-out test table is mixed and is not used to choose the setup. It does show that `top_152` is worth one focused stability check because it is close on validation and supportive on some test metrics.

## Decision

Do not promote a smaller feature surface from the one-shot comparison. Use Experiment 014 to decide whether the closest cleanup candidate, `top_152`, holds up across repeated seeds.

## Artifacts

- `reports/013_feature_cleanup_comparison.csv`
- `reports/experiments/013_feature_cleanup.md`
- `reports/experiments/013_selected_features.csv`
