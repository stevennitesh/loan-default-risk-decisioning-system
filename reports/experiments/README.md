# Experiment Reports

This folder tracks post-v1 model and feature experiments against the frozen v1 baseline.

The goal is not to make every change look successful. The goal is to preserve a clear, comparable trail showing which modifications improved the decisioning story and which did not.

## Archive Interpretation

Reports `000`–`014` and log row `015` are historical experiment records. Their original conclusions and next actions describe the exploration at the time; they are not a current work queue or an independent-test certification. The [current evidence status](../../docs/validation/VALIDATION_PLAN.md#current-evidence-status) owns the known limitations and pending gates. The [remediation plan](../../docs/implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) is proposed work, not implemented methodology.

Original test applicants enter fitting under the repeated stability seeds; SHAP-ranked selection consumes holdout/Kaggle scoring importance; calibration fitting and selection share validation rows. Formal validation-based sorting therefore does not establish independent assessment across the experiment trail. Keep recorded numbers, but qualify claims of generalization and calibration.

Legacy PR-AUC means average precision, recall at 10% review capacity means highest-score default capture, and EV means retrospective utility units. Last-k POS/card features rank account records, not distinct applicant months. "Frozen" denotes preserved snapshots; local regenerated artifacts may differ.

## Baseline

The frozen v1 baseline is recorded in:

- `000_v1_baseline.md`
- `experiment_log.csv`

The baseline was intentionally re-frozen after the separate `pipeline-v1` and `pipeline-post-v1` artifact paths were added. `000_v1_baseline.md` and row `000` in `experiment_log.csv` now reflect the final frozen v1 dashboard data. Historical experiment reports `001`-`014` preserve the metrics recorded at the time of each experiment; row `015` records the final post-v1 dashboard freeze used for the concise comparison.

## Experiment Rule

Each experiment should change one meaningful thing:

- one new feature family;
- one new source table;
- one calibration method;
- one model-family comparison;
- one threshold policy change.

Avoid bundling multiple feature families into one first pass. If a combined run improves, but the individual source of lift is unclear, split the experiment.

## Required Report Fields

For future experiments, use [the template](experiment_template.md) and record:

- experiment ID and short name;
- change tested;
- hypothesis;
- files or tables changed;
- validation metrics;
- assessment metrics with the actual population role and reuse history;
- business-value metrics for the balanced scenario;
- feature-count change;
- top SHAP driver changes;
- conclusion: improve, no clear improvement, or worse;
- next action.

Also identify the config, code/input snapshot, exact model/calibrator artifacts, score representation, split memberships and sample counts, effective fitted parameters, and fitting/selection/assessment roles. Mark missing identity or evidence explicitly. Compare only compatible protocols; an old snapshot is historical context, not an automatic baseline for a corrected experiment.

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

Model and threshold choices must be made using development fitting/selection evidence only. The existing test population is historical comparison; do not call it untouched or use its outcomes to select new candidates. The new assessment protocol has not yet been implemented.

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
008_narrow_pressure_features.md
009_recency_deterioration_features.md
010_recency_model_stability.md
011_last_k_temporal_features.md
012_last_k_model_stability.md
013_feature_cleanup.md
014_feature_cleanup_stability.md
015 final pipeline freeze row in experiment_log.csv
v1_to_post_v1_model_diff.md
post_v1_results_summary.md
```
