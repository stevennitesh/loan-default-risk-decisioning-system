# Running and reproducing the project

For the project story and results, read the [HTML project report](https://stevennitesh.github.io/loan-default-risk-decisioning-system/). Return to the [project overview](../README.md) for the main code links.

This guide covers code checks, report generation, new pipeline runs and exact historical reproduction. Run all commands from the repository root. These goals require different inputs and establish different kinds of evidence.

## Choose a reproduction goal

Choose the reproduction goal before running commands:

| Goal | Required inputs | What the run establishes |
|---|---|---|
| Recreate the presentation | Committed anonymous aggregates and project dependencies | Regenerates the reported charts and story without applicant data or models. |
| Run a new pipeline and assessment | Downloaded source data and current code/configuration | Creates new local models, memberships and results under the implemented procedure. |
| Reproduce an exact historical execution | Retained original models, memberships, source archives and locked environment | Repeats the identified execution; this was checked on the author's host. These local inputs are not distributed in Git. |

## Check the code without data

To check the code without downloading data (Python 3.12 and Make):

```bash
make setup
make lint
make format-check
make test
```

`make setup` installs dependencies into the selected interpreter; it does not create a virtual environment. On Windows where `python3` is unavailable, append `PYTHON=python` to Make commands, for example `make test PYTHON=python`. The Docker image defaults to `make test` and is a test container.

## Recreate the presentation without data

Install the project dependencies first using [Check the code without data](#check-the-code-without-data), then generate the presentation:

```bash
make portfolio
```

On Windows where `python3` is unavailable:

```powershell
make portfolio PYTHON=python
```

The renderer reads only the committed anonymous results in `reports/tuning_20261004/` and the controlled weighting evidence in `reports/class_weighting_20261004/` and anonymous frozen-model input magnitudes in `reports/model_inputs_20261004/`. It writes `reports/portfolio/`, checks matched population counts, fold-summary agreement and weighting-run identity, and records input/renderer/output hashes. It does not download data, load model artifacts, fit models or replace frozen scientific evidence. The HTML embeds images and styles; PNG/SVG files support GitHub viewing and reuse. A short main story leads to expandable supporting evidence. Phone charts use portrait layouts with stacked panels, keeping labels beside their values. Charts can also be enlarged within the report. Both layouts preserve the plotted values, axis limits and units. Exact technical keys remain in the numeric appendix and metric dictionary. `make portfolio-site` verifies the reviewed final-file allowlist and hashes, then copies only the static presentation into `.tmp/portfolio-site/`; the Pages workflow publishes this folder. Repository-wide data/models/scratch files are never uploaded. Source evidence links point to GitHub; adjacent downloads work online and offline when the presentation folder is retained. The HTML alone keeps all eight charts and text offline.

`make portfolio-inputs` is a separate local diagnostic requiring the completed current five frozen models and identified mart. It reads the database in read-only mode and reuses saved preprocessing/native TreeSHAP on 1,000 target-blind assessment rows per fold (seed 20261004), with no fitting or selection. Only anonymous per-fold/mean magnitudes, the input dictionary and provenance are curated; the ordinary presentation renderer does not need those models or data.

## Run a new pipeline

Raw Kaggle data is not committed. Download the [official dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data), including its column-description dictionary, and place the CSV files in `data/raw/`. Account access and acceptance of the competition rules are required.

To regenerate scoped local artifacts with the downloaded data:

```bash
make pipeline-v1
make pipeline-post-v1
```

These commands overwrite generated local outputs. They do not refresh committed snapshots, screenshots, or PBIX visuals. `requirements.lock` pins the dependency closure used for the named correctness run; current artifacts still differ from historical experiment metrics. See the retained-local-evidence instructions below.

For step-by-step post-v1 work, pass the same config at every step:

```bash
make ingest CONFIG=configs/post_v1.yaml
make features CONFIG=configs/post_v1.yaml
make train CONFIG=configs/post_v1.yaml
make evaluate CONFIG=configs/post_v1.yaml
make calibrate CONFIG=configs/post_v1.yaml
make score CONFIG=configs/post_v1.yaml
make explain CONFIG=configs/post_v1.yaml
```

Bare step targets use `configs/base.yaml`, a post-v1 feature scope with separate default output paths. For v1, use `CONFIG=configs/v1.yaml` and omit calibration. Power BI consumes `reports/dashboard_data/` for v1 and `reports/dashboard_data_post_v1/` for post-v1. `make dashboard-data` and `make dashboard-data-post-v1` export from existing scoped artifacts without retraining; export also recomputes segment diagnostics and, for post-v1, calibrated probability-quality metrics.

After the 2026-10-04 methodology corrections, regenerate dependent artifacts in order with `make pipeline-post-v1 PYTHON=python` (omit the interpreter override where `python3` works). Calibrating an old model alone is insufficient: corrected models require an identified feature build and reserved calibration IDs. Feature builds, fitted models, evaluation thresholds, scored rows, and calibrators now carry local identity checks; replacing a parent or calibrator requires rebuilding its dependent outputs. `make pipeline-v1` likewise uses corrected repayment features and row bagging, so it is a regeneration of the v1 source scope, not an exact reproduction of historical numbers. Both commands overwrite ignored runtime outputs; curated experiment numbers, PBIX files, and screenshots remain historical.

## Assessing the selection procedure

After generating compatible post-v1 features/models, run:

```bash
make assess-post-v1 PYTHON=python
```

This separate command assesses history-selected LightGBM and three declared
comparators on five matched outer folds with the current training-only inner-CV selection. Within each
fold, fitting, calibration and selection remain disjoint; each frozen workflow
predicts the same outer applicants. History feature rankings average three model-seed repeats within fitting
data. Split seeds and model seeds are configured independently. Each run creates
an ignored folder under `reports/post_v1/nested_assessment/` with exact local role
IDs, settings/seeds, selected fold artifacts, predictions, and a readable report.
It does not change the saved dashboard model or historical snapshots.

[The methodology explanation](../docs/validation/ASSESSMENT_METHODOLOGY.md) covers
population sizes, why selection metrics can be optimistic, what nested assessment
estimates, and why random folds cannot establish future-cohort performance.
The completed results for this selection procedure are in [the current application-and-loan-history assessment](../reports/tuning_20261004/assessment_report.md), including matched controls and a sampled shuffled-outcome diagnostic. [The earlier corrected assessment](../reports/correctness_20261004/assessment_report.md) preserves the prior reserved-holdout selection procedure and source-data validation. The default current comparison includes top-40, top-80, and full input groups
with one 24-candidate joint tuning budget, so this command performs more fits than
ordinary training. It is intentionally separate from the main pipeline.

## Locked correctness evidence

This section describes reproduction using retained local artifacts. Exact historical comparison memberships, original fitted models and executed source archives are required alongside the source data. They are ignored by Git and are not available from a fresh clone. The checked reproduction was on the author's host; the public commands above support presentation regeneration and new pipeline runs.

The [earlier repaired assessment report](../reports/correctness_20261004/assessment_report.md) separates full-data outer evidence, saved-model reused comparison, and historical Power BI snapshots. Its [corrected aggregate HTML](../reports/correctness_20261004/corrected_evidence.html) is generated from the verified CSV bundle; it is not a refreshed PBIX.

```powershell
uv venv .tmp/assessment-env --python 3.12 --cache-dir .tmp/uv-cache
uv pip sync requirements.lock --require-hashes --python .tmp/assessment-env/Scripts/python.exe --cache-dir .tmp/uv-cache
make lint format-check test PYTHON=.tmp/assessment-env/Scripts/python.exe
```

The completed local scopes are `configs/correctness_20261004_post_v1_r2.yaml` and `configs/correctness_20261004_v1_r2.yaml`. Their isolated outputs preserve the original defaults and historical artifacts. Before the first training run in any fresh scope, copy the original scoped LightGBM artifact into its new model directory to retain original comparison IDs; retain a local `historical_split_reference.json`. A fresh artifact directory without that reference would create a different population. Keep exact role IDs local and do not publish applicant-level evidence.

`python -m src.correctness_run --config <fresh-scoped-config>` executes the ordered raw-to-dashboard pipeline and writes a phase manifest/logs. It requires a fresh output directory and retains failures. `src.source_reconciliation` performs independent Python checks of actual source examples. `src.nested_assessment` executes the declared matched protocol; `configs/correctness_20261004_negative_control_r2.yaml` selects a stratified 20,000-row seed-913 shuffled-label diagnostic using the same full recipe. `src.assessment_diagnostics`, `src.assessment_reproduce` and `src.correctness_summary` generate frozen diagnostics, clean-environment reproduction and anonymous presentation evidence. Exact completed commands and paths are in the evidence report/manifest. No model promotion uses outer results.

## Evidence limits and history

This is a local portfolio decision-support simulation. It does not establish underwriting, compliance, fair-lending or adverse-action readiness. Calendar application, field-availability and label-maturity timestamps are unavailable; relative offsets cannot certify real-time availability. Historical public-data exploration remains a limitation even with disjoint fitting/selection roles. Same-host reproduction is not cross-hardware certification. See [current evidence status](../docs/validation/VALIDATION_PLAN.md#current-evidence-status).

The [HTML project report](https://stevennitesh.github.io/loan-default-risk-decisioning-system/) is the main reading destination. The [current assessment](../reports/tuning_20261004/assessment_report.md) provides supporting technical evidence. The saved [Power BI files and screenshots](../powerbi/README.md) are **unrefreshed historical demonstrations** in [the labeled Power BI archive](../powerbi/archive/README.md). Native Desktop/DAX refresh was unavailable on this host. These assets are preserved; the new HTML and plots are separate outputs.

For the documented development trail, use the [historical experiment archive](../reports/experiments/README.md). Original numbers and dated conclusions are retained there, including the earlier 68-input and 168-input comparisons. They are historical comparisons, not current selection instructions or independent final-test evidence. The [prior corrected assessment](../reports/correctness_20261004/assessment_report.md) preserves the earlier repaired protocol; the [current tuning assessment](../reports/tuning_20261004/assessment_report.md) identifies the current completed procedure.

Downloaded source files, generated applicant-level CSVs, exact memberships, fitted Python models and intermediate runtime outputs stay local. Git includes curated final reports, charts and anonymous aggregate tables, plus the explicitly historical Power BI archive. The embedded PBIX contents have not been certified as aggregate-only; [archive limits](../powerbi/archive/README.md) and [reports policy](../reports/README.md) describe that boundary.

## Repository guide

| Folder | Contents |
|---|---|
| `sql/`, `src/`, `tests/` | Model-input SQL, Python pipeline/reporting and synthetic verification fixtures. |
| `configs/`, `docs/` | Declared settings, scope, assessment/testing and command ownership. |
| `reports/portfolio/` | Main HTML report, text alternative, charts and anonymous numeric appendix. |
| `reports/tuning_20261004/`, `reports/correctness_20261004/` | Identified current/prior technical evidence. |
| `reports/experiments/`, `powerbi/archive/` | Historical development trail and clearly labeled Power BI snapshots. |
| `data/`, `models/`, `.tmp/` | Local source/intermediate data, model artifacts and scratch work; ignored by Git. |

## Development history

The default branch contains the reviewed portfolio release. The earlier development
commits are preserved under the [`archive/pre-cleanup-2026-10-06` tag](https://github.com/stevennitesh/loan-default-risk-decisioning-system/tree/archive/pre-cleanup-2026-10-06).
Commit IDs recorded in scientific provenance still identify those original executions;
the history cleanup does not replace source fingerprints or reported results.
