# Historical remediation decisions

The main correctness repairs and current assessment were completed on 2026-10-04.
This document summarizes the engineering decisions behind that work and separates
completed changes from remaining research and dashboard recommendations.

Read the [current case study](../../reports/portfolio/case_study.md) for the
project story, the [validation status](../validation/VALIDATION_PLAN.md#current-evidence-status)
for evidence and limits, and the [assessment methodology](../validation/ASSESSMENT_METHODOLOGY.md)
for the implemented evaluation procedure.

## Completed repairs

- **Repayment accounting:** SQL counts each unambiguous installment obligation once
  across split payments. Unknown payments and competing schedules remain explicit.
  Pre-application dates and matched ratio support govern historical features.
- **Monthly history:** recent windows use distinct applicant months. Duplicate
  account-month records fail validation rather than silently changing the grain.
- **Population boundaries:** labeled assessment and unlabeled scoring remain
  separate. Historical comparison applicants are preserved; fitting, stopping,
  calibration, selection and outer assessment roles have explicit memberships.
- **Model selection:** fitting-only preprocessing and feature ranking support a
  bounded three-fold inner search. Model, partition and ranking seeds have
  separate purposes. A training-prevalence benchmark and simpler models provide
  matched comparisons.
- **Probability checks:** calibration fitting and method selection use disjoint
  applicants. Candidate eligibility is tested separately from simplicity
  preferences. A controlled follow-up isolates the effect of class weighting.
- **Simulated actions:** high-risk cases represent simulated decline. Review cost
  applies only to the middle band. Utility uses illustrative units.
- **Artifact consistency:** build, model and calibrator identities reject stale
  dependent outputs. Failed feature rebuilds preserve the previous valid mart.
- **Verification and reporting:** synthetic regressions, source reconciliation,
  identified real-data assessments and locked same-host reproduction support the
  results. Public reports contain anonymous aggregate evidence.

The [earlier repaired assessment](../../reports/correctness_20261004/assessment_report.md)
uses `nested_matched_holdout_v2`. The [current assessment](../../reports/tuning_20261004/assessment_report.md)
uses `nested_inner_cv_v3`. They remain distinct records with their original numeric
results and execution fingerprints.

## Remaining limits and recommendations

| Topic | Current status | What would establish stronger evidence |
|---|---|---|
| Future cohorts | Random applicant groups and prior public-data exploration limit generalization claims. | A separately governed dataset with application, field-availability and outcome-maturity timestamps. |
| Review capacity | Cutoffs define simulated bands; they do not enforce a hard review queue. | An explicitly scoped allocator and tests for ties, overflow and per-batch capacity. |
| Utility | Assumed margins, losses and review costs describe a partial simulation. | Evidence for reviewer outcomes, rejected-loan counterfactuals and actual financial costs. |
| Uncertainty | Fold variation is descriptive. | A declared uncertainty analysis whose estimand and dependence assumptions match the assessment. |
| Native Power BI | The saved dashboards are historical and unrefreshed. | Native inspection of DAX, relationships, cached data and import widths, followed by visual reconciliation to one identified export bundle. |
| Portability | Same-host reproduction was checked. | Independent reproduction on another supported environment. |

These recommendations are outside the completed local portfolio scope. They are
not implemented capabilities or prerequisites for reading the demonstrated work.
The historical Power BI files are retained in [the labeled archive](../../powerbi/archive/README.md).

## Scope and maintained owners

The project remains a local SQL/DuckDB/Python decision-support portfolio using
public Home Credit data. It does not claim production underwriting, compliance,
fair-lending or adverse-action readiness. Identifiers, the recorded outcome and
direct demographic/protected-status-like fields remain excluded from model inputs.

The [project specification](../spec/PROJECT_SPEC.md) owns domain and architecture
requirements. The [command map](IMPLEMENTATION_PLAN.md#5-command-to-artifact-map)
owns execution paths. The [testing guide](../testing/TESTING_PLAN.md) owns code
verification, and the [report policy](../../reports/README.md) owns curated evidence.
