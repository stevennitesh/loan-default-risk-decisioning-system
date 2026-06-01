# AGENTS.md

## Purpose

Build this as an applied financial ML decision-support project, not a notebook-only Kaggle exercise and not a production underwriting system. Be a lean, evidence-governed partner: infer the real job, choose the smallest useful path, validate the result, and keep uncertainty visible.

North star: make the right path cheaper than the sloppy path.

## Grounding

Before changing code or docs, read the relevant source-of-truth document:

- `docs/spec/PROJECT_SPEC.md` defines scope, contracts, non-goals, model-risk posture, and acceptance criteria.
- `docs/implementation/IMPLEMENTATION_PLAN.md` defines build order and command-to-artifact expectations.
- `docs/testing/TESTING_PLAN.md` defines what must be tested and how to use fixtures.
- `docs/validation/VALIDATION_PLAN.md` defines model and reporting gates.
- `README.md` defines the recruiter-facing story and run interface.

Keep `AGENTS.md` subordinate to those docs. If docs conflict, prefer the more specific contract for the task and call out the conflict instead of silently inventing a rule.

## Operating Loop

For nontrivial work, use this loop:

1. Goal: state the user-facing outcome.
2. Assumptions: name what is known, inferred, or uncertain.
3. Smallest useful approach: avoid broad rewrites and optional subsystems.
4. Output: make the narrow change.
5. Checks: run the relevant tests, command, diff, or artifact inspection.
6. Risks: say what remains unproven.
7. Next action: recommend the next high-ROI step.

For bugs or failed validation, use:

1. Baseline the exact failure.
2. Form one concrete hypothesis.
3. Make the smallest change that addresses the mechanism.
4. Retest the same path.
5. Keep, retest, or discard based on evidence.

## Implementation Rules

- Prove the data pipeline before modeling: CSV to Parquet to DuckDB staging to SQL feature mart.
- Let SQL own feature extraction. Use Python for orchestration, training, evaluation, scoring, and exports.
- Preserve one row per `SK_ID_CURR` in the feature mart. Join expansion is a build failure.
- Keep evaluation and scoring populations separate. Metrics come from labeled `application_train` splits; Kaggle `application_test` is for production-like scoring demonstration.
- Exclude direct demographic and protected-status-like fields from v1 model features. If inspected, keep them in a separate diagnostic layer only.
- Define Power BI-facing table contracts before dashboard work. Do not wire dashboards to ad hoc notebook outputs.
- Every major step should produce a concrete artifact: file, table, metric, test, report, or screenshot.

## Scope Discipline

Do not add APIs, Postgres, Spark, MLflow, deep learning, heavy hyperparameter search, or broad fairness tooling before v1 is complete unless the user explicitly changes scope. Prefer deletion, simplification, checklists, validators, and reusable commands over new frameworks or subsystems.

Raw Kaggle data, generated databases, trained model binaries, and generated report/export artifacts should stay out of Git unless a specific deliverable is intentionally curated.

Use `.tmp/` at the repo root for scratch files, temp files, local caches, pytest cache, and any other local garbage.

## Validation Standards

- Test business-critical logic first: ingestion contracts, feature grain, leakage controls, thresholding, expected value, scoring outputs, and dashboard exports.
- Use small synthetic fixtures for tests. Tests should not require the full Kaggle dataset.
- Treat leakage, duplicate applicant rows, population mixing, or threshold selection on test data as blocking failures.
- Do not headline accuracy for this imbalanced credit-risk use case. Prefer PR-AUC, ROC-AUC, Brier score, top-decile lift, recall at review capacity, calibration, and expected-value tradeoffs as specified.
- Do not present SHAP outputs as legally compliant adverse-action notices.

## Done Criteria

Finish nontrivial work with:

- what changed;
- what evidence supports it;
- what remains uncertain;
- the next useful action.

Do not claim production credit-decision readiness. This project is a portfolio decision-support simulation with explicit limitations.
