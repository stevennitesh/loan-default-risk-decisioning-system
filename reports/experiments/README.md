# Experiment Reports

## Current controlled follow-up

[The class-weighting comparison](../class_weighting_20261004/assessment_report.md) isolates one model-setting change in the frozen earlier/current recipes, rather than appending a new selection result to this historical archive. It explains why raw probability quality improved and retains ranking tradeoffs, source identities and limits. Its diagnostic variants do not replace the selected models.

## Historical archive navigation

These numbered experiments preserve earlier findings, values and dated next actions. They are historical exploratory evidence, not the current selection procedure or a queue of work. Recruiters should start with [the current case study](../portfolio/case_study.md) and [offline report](../portfolio/index.html), then consult [current assessment evidence](../tuning_20261004/assessment_report.md). The earlier 68-input/168-input and probability-adjustment comparisons below remain available for the development trail; do not combine them with current group metrics as if they were one model.


This folder tracks post-v1 model and feature experiments against the frozen v1 baseline.

The goal is not to make every change look successful. The goal is to preserve a clear, comparable trail showing which modifications improved the decisioning story and which did not.

## Archive Interpretation

Reports `000`–`014` and log row `015` are historical experiment records. Their original conclusions and next actions describe the exploration at the time; they are not a current work queue or an independent-test certification. [Current evidence status](../../docs/validation/VALIDATION_PLAN.md#current-evidence-status) owns the corrected implementation and pending gates. The [remediation plan](../../docs/implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md) retains broader proposed work.

In the historical experiments, original test applicants entered fitting under repeated stability seeds, SHAP-ranked selection consumed holdout/Kaggle scoring importance, and calibration fitting/selection shared validation rows. Current code preserves saved comparison membership, ranks features within each training role, and fits calibrators on separate reserved rows. These corrections do not retroactively make the archive independent. Keep recorded numbers and identify newly generated evidence separately.

Legacy PR-AUC means average precision, recall at 10% review capacity means highest-score default capture, and EV means retrospective utility units. Historical last-k POS/card features ranked account records; current code uses distinct applicant months before month zero and normalized installment obligations. "Frozen" denotes preserved snapshots; local regenerated artifacts differ in methodology as well as values.

## Corrected Assessment

New corrected assessment uses `make assess-post-v1` and writes a separate run
directory with a protocol manifest. It assesses a fold-local LightGBM selection
procedure; the old repeated-seed command remains development sensitivity. Mean
feature ranks use several model seeds within each fitting population, with split
and model randomness separated. New full-data evidence is recorded separately in [the named correctness assessment](../correctness_20261004/assessment_report.md). See
[the explanation](../../docs/validation/ASSESSMENT_METHODOLOGY.md) before interpreting
or deliberately curating a new run. Historical reports below are not rewritten.

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

Model and threshold choices must be made using development fitting/selection evidence only. The existing test population is historical comparison; do not call it untouched or use its outcomes to select new candidates. The implemented nested assessment is documented in [the current methodology](../../docs/validation/ASSESSMENT_METHODOLOGY.md); diagnostic follow-ups do not promote models using assessment outcomes.

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

The named correctness assessment is a distinct protocol/evidence series; it does not append a revised recipe to historical selection evidence or rewrite numbered experiment outcomes.
