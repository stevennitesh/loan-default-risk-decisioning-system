# Experiment Reports

This folder tracks post-v1 model and feature experiments against the frozen v1 baseline.

The goal is not to make every change look successful. The goal is to preserve a clear, comparable trail showing which modifications improved the decisioning story and which did not.

## Baseline

The frozen v1 baseline is recorded in:

- `000_v1_baseline.md`
- `experiment_log.csv`

Do not edit the baseline values after post-v1 experiments begin unless v1 itself is intentionally re-frozen.

## Experiment Rule

Each experiment should change one meaningful thing:

- one new feature family;
- one new source table;
- one calibration method;
- one model-family comparison;
- one threshold policy change.

Avoid bundling multiple feature families into one first pass. If a combined run improves, but the individual source of lift is unclear, split the experiment.

## Required Report Fields

Each experiment report should include:

- experiment ID and short name;
- change tested;
- hypothesis;
- files or tables changed;
- validation metrics;
- held-out test metrics;
- business-value metrics for the balanced scenario;
- feature-count change;
- top SHAP driver changes;
- conclusion: improve, no clear improvement, or worse;
- next action.

## Metrics To Compare

Primary:

- PR-AUC;
- top-decile lift;
- recall at 10% manual-review capacity;
- Brier score;
- expected value per applicant for the balanced scenario.

Secondary:

- ROC-AUC;
- precision at top decile;
- calibration-bin behavior;
- feature importance stability;
- train/evaluation runtime if materially changed.

Do not optimize on accuracy. Accuracy is not a useful headline metric for this imbalanced credit-risk outcome.

## Selection Discipline

Model and threshold choices must be made using training/validation data only. Held-out test metrics are for final reporting after the experiment choice is fixed.

Do not promote or demote an experiment because held-out test looks better or worse than the validation-selected choice. If held-out test diverges materially from validation, record the gap as a stability/generalization signal and improve the model-generation method in a follow-up experiment.

## Suggested File Naming

```text
001_bureau_balance_features.md
002_pos_cash_features.md
003_credit_card_features.md
004_calibration_experiment.md
005_feature_selection.md
006_model_stability.md
007_risk_pressure_features.md
post_v1_results_summary.md
```
