# How current tuning and assessment are separated

## Plain-language reading path

[The HTML project report](https://stevennitesh.github.io/loan-default-risk-decisioning-system/) explains the current assessment; its [standalone HTML](../../reports/portfolio/index.html) supports offline reading. A **feature** is a model input; the **mart** is a one-applicant modeling table. **Cross-validation** repeats fitting/selection across applicant groups. An **outer fold** is an applicant test group predicted after the training procedure is frozen. **Calibration** is a probability adjustment; the final method can leave raw probabilities unchanged. Exact protocol/CSV keys remain technical identifiers.


The current protocol is `nested_inner_cv_v3`. It evaluates a declared bounded
selection procedure on the already explored public dataset. It does not claim
global parameter optimality, future-cohort validation or underwriting readiness.
The completed `nested_matched_holdout_v2` evidence and eight-preset recipe remain
[separate historical evidence](CORRECTNESS_V2_METHODOLOGY.md), including the
[2026-10-04 correctness report](../../reports/correctness_20261004/assessment_report.md).

## Population and fitting roles

The original 46,127 historical comparison IDs never enter the new search,
calibration, selection or outer assessment. Only the remaining 261,384 labeled
development applicants enter five stratified outer folds. Kaggle applicants remain
unlabeled scoring-only. Within every outer-training partition, base fitting,
calibration fitting and method/threshold selection remain disjoint 70%/15%/15%
roles. Every development applicant receives exactly one outer prediction per
workflow and split-seed repeat. Role membership and target support are checked.

Three stratified inner CV folds occur **only inside base fitting**. Each inner
non-score partition reserves an additional stratified 15% stopping subset.
Preprocessing, imputation, encoding and raw-feature gain rankings fit only the
remaining fitting rows. Early stopping observes only the stopping rows, never the
inner scoring fold. Rankings average raw-feature ranks over seeds 101/211/307,
aggregate encoded columns back to raw columns and break rank ties by field name.
Each inner scoring fold supplies held-out selection metrics. No calibration,
selection-validation, historical comparison or outer labels enter this search.

The shared owner [src/tuning.py](../../src/tuning.py) caches transformed fitting,
stopping and scoring matrices by feature surface within each exact fold. Caches
are discarded at the fold boundary. Each classifier creates its own dataset and
sets `feature_pre_filter=False` while minimum leaf size varies.

## Controlled probability follow-up

The [controlled class-weighting follow-up](../../reports/class_weighting_20261004/assessment_report.md) separately tests why raw probabilities improved. It refits both frozen history recipes with and without their earlier positive-class weight, holding applicants, selected inputs, fitting-only preprocessing, seed, tree counts and every other classifier setting fixed within each pair. Removing weighting improves raw Brier and log loss in all ten recipe/group pairs and essentially reproduces the full original probability-loss reduction. Original-weight refits match frozen predictions exactly. This isolates a model-setting effect conditional on the selected recipes; it does not measure the probability-quality screen's separate selection effect or establish weights for future cohorts. No diagnostic variant is promoted, and no new calibration comparison is made.

## Declared search and acceptance

Each LightGBM workflow spends **24 unique joint feature/parameter candidates**,
with three inner folds each. History jointly searches top-40, top-80 and all 174
allowed raw fields, allocating eight candidates per surface. Application-only
uses all 31 allowed SQL `f_applicant_static` fields and 24 parameter candidates.
An explicit scoped feature-experiment request can declare other surfaces; it still
spends one shared joint budget per workflow. No outer result selects a family.

The expanded history surface also includes `external_score_credit_pressure` and
`external_score_annuity_pressure`, two interactions calculated entirely from
application fields in [risk-pressure SQL](../../sql/05b_feature_risk_pressure.sql).
They are absent from the application-only comparator. Each workflow selects its
own model recipe, so the matched comparison estimates the combined expanded
workflow rather than isolating the information value of loan history alone.

Candidate parameters are a deterministic bounded random draw: learning rate
0.025/0.04/0.06/0.08; leaves 15/31/47/63; depth unlimited/6/8; minimum child support
30/60/120/240; row/column fractions 0.7/0.85/1; L1 0/0.1/1; L2 1/4/12. Row bagging
has frequency one. Positive-class weight is `1 + fraction*(fit_negative/fit_positive-1)`
with fraction 0/0.25/0.5/1. Thus each surface explicitly includes unweighted,
lighter and balanced fits, and its ratio uses fitting labels alone.

Stopping uses average precision, a maximum 600 rounds and patience 40. Final
boosting rounds are the integer median of the chosen candidate's three best
iterations, at least one. The final pipeline can fit **all base-fitting rows**
with that fixed count. A selected top-N surface is reranked using base-fitting
rows alone. Calibration never changes this fitted base pipeline.

Raw probability acceptance compares mean inner-fold Brier and log loss against
fold-specific fitting-prevalence predictions on the same scoring rows. Tolerances
are absolute Brier +0.002 and log loss +0.01. Candidates passing both take
precedence. Within that pool average precision is primary, followed by top-decile
lift, top-score capture, ROC AUC and lower Brier as declared ties. If none passes,
the ranking choice is retained with an explicit failed probability-acceptance flag;
there is no label borrowing or silent retuning. This status is informative and
does not imply underwriting suitability.

The tuned logistic comparator searches C=0.01/0.1/1/10 on the same training-only
inner scoring partitions, with unweighted lbfgs, max_iter=1000, training-only
median/mode imputation, scaling and unknown-safe sparse encoding. Its classifier
has no stopping, so it uses all non-score inner rows. `logistic_tuned` and the
supported historical `logistic_fixed` (C=1, balanced weights) have distinct
workflow and candidate identities. Prevalence uses only base-fitting labels and
has no search, calibration or policy utility. A selected constant-score v3 recipe also omits quantile-policy utility rather than inventing cut points.

## Reserved calibration, seeds and interpretation

Sigmoid/isotonic fit only calibration-role rows. Raw/sigmoid/isotonic selection
uses the separate selection-validation role with the existing minimum Brier gain
and simplicity rules. Raw quantile scenario thresholds also come from that role.
Calibrated Brier/log loss versus base-fitting prevalence and the predefined
tolerances are recorded on selection validation. Their flags do not change the
search recipe. No post-calibration refit uses any reserved or outer rows.

Outer seed and model seed are 42. Search seed is 20261004; CV seed is 42.
Ranking seeds 101/211/307 have a separate purpose. A small predeclared sensitivity
refits the selected feature list, parameters and fixed rounds at model seeds
101/211/307, fits each calibrator only on calibration rows and applies the
already selected method to selection-validation rows. It reports variation;
it never chooses a lucky seed or uses outer quality for promotion.

The seed-913 shuffled-label control selects a stratified 20,000-row development
sample before permutation and executes the complete five-fold search, calibration
and independent comparator procedures. It is a sampled diagnostic, not full-data
performance. Tied-score metrics and reliability bins retain the v2 score-group
rules. Paired fold differences are descriptive; fold SD is not a confidence
interval. There is no pooled cross-fold average precision claim.

## Reproduction and artifact scopes

Current `configs/base.yaml` and `configs/post_v1.yaml` explicitly use
`model.lightgbm_tuning.mode: bounded_inner_cv`. Training, feature experiments,
feature selection, split stability and nested assessment share this owner.
Ordinary evaluation compares shared split/build identities while each model uses
its own eligible selected fields. Current split stability records only each
split's joint winner: frequencies use all completed splits, metrics are
conditional on selection, and absent/singleton variability is unavailable. It
does not choose an aggregate surface or promote a model across these summaries.
`mode: legacy_presets` reproduces the historical bounded presets. Frozen pre-v3
configs without a mode retain their historical interpretation; they are not
current defaults. `configs/v1.yaml` retains its historical v1 demonstration scope.

For a new assessment using retained local inputs:

The commands below require the corrected local database and original reference
model that preserve the historical comparison memberships. Those inputs and the
exact executed source archives are ignored by Git. They are available for the
author's checked same-host reproduction, rather than distributed with a fresh
clone. The [public run guide](../RUNNING.md#choose-a-reproduction-goal) separates regenerating
the presentation from running a new pipeline with downloaded data.

```bash
make assess-tuning PYTHON=.tmp/assessment-env/Scripts/python.exe
make assess-tuning-control PYTHON=.tmp/assessment-env/Scripts/python.exe
python -m src.verify_frozen_assessment --assessment-dir <completed-run>
python -m src.assessment_reproduce --assessment-dir <completed-original-run>
python -m src.tuning_summary --assessment-dir <completed-original-run> --control-dir <completed-control-run>
```

The named configs read the corrected r2 database and original model only as a
population/build reference via `paths.reference_model_dir`. Generated results,
fold joblibs, role IDs and predictions go into unique ignored run directories
under `reports/generated/tuning_20261004*`. The r2 configs, source fingerprints,
reports, models and runtime dependency lock are not refreshed to v3 source.
Exact executed v3 source is retained in `.tmp/tuning_20261004_execution_source`; post-run ordinary-caller artifact metadata/report-path fixes are recorded as delivery deltas without changing execution fingerprints.
Pre-edit dirty scientific source/configs were archived with hashes under
`.tmp/tuning_20261004_preedit`; HEAD alone would not reproduce that prior source.

Each run persists the frozen config/source identity, exact outer roles, candidate
specification digest, per-inner-fold fitting/stopping/scoring IDs, rankings, CV
metrics, best iterations, acceptance flags, selected fixed rounds, seed variation
and timing. Only complete manifests are evidence. CSV schema ownership remains
[src/report_contracts.py](../../src/report_contracts.py); anonymous aggregate
curation belongs to [reports policy](../../reports/README.md).

Calendar application timestamps, actual feature availability and label-maturity
times are unavailable. Applicant IDs and relative DAYS offsets cannot reconstruct
true chronology; no chronological splitter is invented. Reliable future-cohort
validation requires different evidence. Native Power BI refresh remains unverified
and historical PBIX/screenshots are preserved. The current empirical status and
prior/new results are owned by [validation](VALIDATION_PLAN.md#current-evidence-status).
