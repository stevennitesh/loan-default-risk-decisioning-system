# Experiment NNN: Short Name

## Question and hypothesis

State the one change being tested, its mechanism, and what evidence could reject the hypothesis. Read [archive interpretation](README.md#archive-interpretation) and [current evidence status](../../docs/validation/VALIDATION_PLAN.md#current-evidence-status) before choosing a baseline or assessment population.

## Run identity and evidence status

- Date, experiment/config/code/input snapshot:
- Baseline run and comparison compatibility:
- Saved model, feature list, calibration, policy, and export identities (state unavailable identity explicitly):
- Effective fitted parameters and feature count:
- Fitting, feature-selection, calibration-fitting, method-selection, and assessment populations; memberships/counts, seeds, and any prior reuse:
- Score representation for each metric (raw or calibrated):
- Proposed, executed, or historical status; command used and generated artifact locations:

Do not label the existing comparison population an untouched final test. Record any protocol change before viewing its outcomes. Proposed remediation machinery is not yet implemented.

## Change tested

- Source/SQL feature meaning, including grain, time availability, and missingness:
- Python/model/config changes:
- Controls and unchanged assumptions:

## Metric comparison

Populate from identified compatible runs; leave unavailable results blank with a reason. Do not copy historical baseline values into a new protocol automatically.

| Population role / score kind | Metric | Baseline | Experiment | Difference |
|---|---|---:|---:|---:|
| Selection | Average precision (legacy PR-AUC) | | | |
| Selection | ROC-AUC | | | |
| Selection | Brier score | | | |
| Selection | Top-decile lift | | | |
| Selection | Top-score capture at declared rate | | | |
| Assessment or historical comparison (specify) | Average precision | | | |
| Assessment or historical comparison (specify) | Brier score | | | |

Report fitting-set calibration separately from method assessment. Distinguish fold/seed variability from sampling uncertainty and independent evaluation. Explain incompatible populations or missing evidence before interpreting a difference.

## Policy and utility comparison

- Scenario, thresholds, action definitions, evaluated population/count:
- Approval, middle-review, and high-priority/high-risk counts and rates:
- Actual queue capture versus top-score ranking capture:
- Capacity meaning and any overflow (fixed quantiles do not enforce a hard cap):
- Utility weights/units and which actions are costed; disposition assumptions:
- Retrospective utility per applicant; no real-profit claim:

## Feature and interpretation checks

- Split-payment obligations, record/month windows, unknown values, and availability assumptions:
- Excluded identifiers, target, and diagnostic-only fields:
- Feature-selection importance population versus post-selection reporting SHAP:
- Missingness, top drivers, and instability:

## Conclusion and next action

State improved, no clear improvement, worse, or inconclusive relative to the declared compatible selection evidence. Include limitations and assessment reuse. Explain what this adds to the portfolio story. A proposed next action is not authorization or completed implementation.
