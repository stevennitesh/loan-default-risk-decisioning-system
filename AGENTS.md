# AGENTS.md

## Project

Resume portfolio for recruiters and hiring managers, using public Home Credit
data in a local SQL/DuckDB/Python/Power BI decision-support pipeline. Keep work
within that scope; production extensions require an explicit scope change.
Do not claim underwriting, compliance, fair-lending, or adverse-action readiness.

## Core Constraints

- SQL owns feature extraction; Python owns orchestration, modeling, scoring,
  interpretation, and exports.
- Preserve one mart row per `(SK_ID_CURR, source_population)` and keep labeled
  evaluation separate from unlabeled Kaggle scoring.
- Exclude identifiers, `TARGET`, and direct demographic/protected-status-like
  fields from model features; diagnostics remain separate.
- Historical test results are comparisons, not an untouched lockbox or selection
  evidence. Plans and historical next actions do not start work.

## Load Context For The Task

Before the decision or edit it governs, load the matching owner below and follow
its relevant sections and source links. These are conditional routes, not a
read-all checklist. Resolve source/test/contract disagreements before rewriting
accepted behavior or claims.

| When the task involves | Load |
|---|---|
| Portfolio story or presentation | [README.md](README.md); its HTML report path owns the reader journey. |
| Running the project or reproduction | [Run guide](docs/RUNNING.md); it owns config scopes, reproduction boundaries and the Windows interpreter override. |
| Domain meaning, scope, architecture, SQL features, or population contracts | [Project spec](docs/spec/PROJECT_SPEC.md); read the sections governing the change. |
| Training, calibration, scoring, experiments, metrics, or result claims | [Current evidence status](docs/validation/VALIDATION_PLAN.md#current-evidence-status), then the relevant validation gates. |
| Orchestration, config, or command behavior | [Makefile](Makefile), the applicable file in `configs/`, and the [command-to-artifact map](docs/implementation/IMPLEMENTATION_PLAN.md#5-command-to-artifact-map). |
| Report or dashboard export schemas | [src/report_contracts.py](src/report_contracts.py); exact columns belong here. |
| Code verification, fixtures, or CI | [Testing plan](docs/testing/TESTING_PLAN.md) and [run guide](docs/RUNNING.md#check-the-code-without-data) for host/interpreter setup; default gates are `make lint`, `make format-check`, and `make test`. |
| Container behavior | [Dockerfile](Dockerfile); testing expectations remain in the testing plan. |
| Generated artifacts, curated evidence, or Git inclusion | [Reports policy](reports/README.md); use `.tmp/` for scratch work. |
| Power BI reports, visuals, screenshots, or refresh | [Power BI guide](powerbi/README.md). |
| Interpreting or recording experiments | [Experiment guide](reports/experiments/README.md); preserve historical numbers and identify new evidence separately. |
| Explicitly requested remediation work | [Remediation proposal](docs/implementation/PORTFOLIO_INTEGRITY_REMEDIATION_PLAN.md); its status note distinguishes implemented repairs from remaining recommendations. |
| An assigned issue or other tracker-backed work | [Tracker guide](docs/agents/issue-tracker.md), then its linked label mapping. Ordinary coding requires no issue. |
