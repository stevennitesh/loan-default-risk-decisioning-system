# AGENTS.md

Repo-specific guidance for coding agents working in this project. Global agent
instructions cover general workflow, safety, verification, reporting, and skill
selection; this file only adds project context and repo contracts.

## Project Identity

This is an applied financial ML decision-support portfolio project built around
the public Home Credit dataset. Treat it as a reproducible analytics and ML
pipeline, not a Kaggle notebook exercise and not a production underwriting
system.

The intended story is:

- raw Kaggle CSVs become Parquet files;
- Parquet files load into DuckDB staging tables;
- SQL builds applicant-level feature tables and a one-row-per-applicant mart;
- Python trains, evaluates, calibrates, scores, explains, and exports reports;
- Power BI consumes explicit dashboard export tables.

Do not claim production credit-decision, compliance, fair-lending, or adverse
action readiness.

## First Files To Read

Before changing code or docs, read the file that owns the task's contract:

- `README.md`: recruiter-facing story, run interface, dashboard narrative.
- `docs/spec/PROJECT_SPEC.md`: scope, non-goals, public contracts, acceptance
  criteria, model-risk posture.
- `docs/implementation/IMPLEMENTATION_PLAN.md`: command-to-artifact
  expectations and build sequence.
- `docs/testing/TESTING_PLAN.md`: fixture strategy, required checks, CI
  expectations.
- `docs/validation/VALIDATION_PLAN.md`: model and reporting validation gates.
- `reports/README.md`: which report artifacts are intentionally committed
  versus regenerated locally.

If docs disagree with source or tests, identify the stale side before editing.
The more specific contract for the touched behavior should win.

## Pipeline Invariants

- SQL owns feature extraction. Python owns orchestration, training, evaluation,
  calibration, scoring, explainability, and exports.
- Preserve one row per `SK_ID_CURR` and `source_population` in
  `mart_credit_risk_features`; join expansion is a build failure.
- Keep evaluation and scoring populations separate. Metrics come only from
  labeled `application_train` splits. Kaggle `application_test` is for
  production-like batch scoring demonstration, not validation metrics.
- Do not let identifiers, `TARGET`, or direct demographic/protected-status-like
  fields into model features. Diagnostic fields stay in
  `segment_diagnostics`.
- Select models, thresholds, feature sets, and calibrators from validation
  evidence. Held-out test metrics are post-selection generalization checks.
- Do not headline accuracy for this imbalanced credit-risk use case. Prefer
  PR-AUC, ROC-AUC, Brier score, top-decile lift, recall at review capacity,
  calibration, and expected-value tradeoffs.
- SHAP outputs are interpretation/debugging aids only. Do not present them as
  legally compliant adverse-action notices.

## Scope Boundaries

Keep v1 and post-v1 comparison work focused on the existing local pipeline.
Do not add APIs, Postgres, Spark, MLflow, deep learning, broad fairness tooling,
or heavy hyperparameter search unless the user explicitly changes scope.

Prefer repo-native commands, checks, SQL, fixtures, and existing helper modules
over new frameworks or new service boundaries.

## Commands

Use the Makefile as the primary interface:

```bash
make setup
make lint
make format-check
make test
make ingest
make features
make train
make evaluate
make score
make dashboard-data
make dashboard-data-post-v1
make pipeline-v1
make pipeline-post-v1
```

Notes:

- `make lint`, `make format-check`, and `make test` are the default local and
  CI quality gates.
- Full pipeline targets require local Kaggle raw CSVs under `data/raw/`.
- Tests use synthetic fixtures and should not require full Kaggle data.
- The Docker image is a test container; its default command is `make test`.

## Artifacts And Git Hygiene

Keep raw Kaggle data, DuckDB files, Parquet files, model binaries, generated
runtime reports, generated figures, and dashboard CSV exports out of Git unless
a curated artifact is explicitly part of the portfolio evidence.

Committed portfolio evidence currently includes curated experiment summaries,
selected comparison CSVs, `reports/model_card.md`, Power BI files/screenshots,
and documentation. Runtime output directories are preserved with `.gitkeep`
files where needed.

Use `.tmp/` for scratch work and local experiment debris.

## Review Priorities

When reviewing or changing this repo, treat these as blocking risks:

- target leakage or threshold selection on test data;
- duplicate applicant rows or broken feature grain;
- mixing labeled evaluation populations with unlabeled scoring populations;
- stale README/report claims that contradict committed experiment evidence;
- dashboard exports that bypass `src/report_contracts.py` schemas;
- generated artifacts accidentally committed outside the curated evidence set;
- Docker, CI, or Makefile drift from the repo's documented command interface.
