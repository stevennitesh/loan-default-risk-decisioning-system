# Preserved correctness v2 methodology

This document describes the completed r2 evidence recipe. Current tuning uses [v3](ASSESSMENT_METHODOLOGY.md); these historical design choices and numbers are preserved.

The portfolio now has two kinds of evaluation. The ordinary pipeline chooses a
model for its SQL-to-Power BI demonstration. The nested assessment command tests
the declared LightGBM feature, model-setting, and calibration selection procedure
on applicants excluded from that fold's fitting and selection. These answer
different questions and produce separate artifacts. Matched independent training-prevalence, fixed logistic-regression and application-only LightGBM controls use the same fold memberships and reserved roles. The assessment never selects between model families.

This protocol demonstrates careful modeling on the available public dataset.
It does not undo prior exploration, establish future-cohort performance, or
certify a lending model. Current empirical status remains in the
[validation owner](VALIDATION_PLAN.md#current-evidence-status).

## Population flow

Only labeled `application_train` applicants enter assessment. The existing
saved test IDs remain the historical comparison population, approximately 15%
of labeled applicants. Those applicants are excluded from nested fitting,
calibration, selection, and assessment. Unlabeled `application_test` applicants
remain scoring-only.

The remaining development applicants, approximately 85%, are divided into five
stratified outer folds. For each fold:

```text
Development population
├─ Outer assessment: one fold, approximately 20% of development
└─ Outer training: the other four folds, approximately 80% of development
   ├─ Base fitting: 70% of outer training
   ├─ Calibration fitting: 15% of outer training
   └─ Selection validation: 15% of outer training
```

All roles are disjoint within a fold and stratified by the binary target.
Every role must contain both classes; insufficient support fails rather than
borrowing assessment rows. An applicant can fit another fold's model, but their
own assessment prediction comes only from the workflow that excluded them.
Each development applicant receives exactly one assessment prediction per declared workflow per outer split-seed repeat. The protocol is `nested_matched_holdout_v2`; workflow/family identity is part of each new CSV schema.

The inner procedure uses one reserved holdout, not three-fold inner CV. This is
a bounded nested assessment: outer K-fold evaluation around an inner selection
procedure. Default fractions are project choices, not an industry prescription.
With five folds, each base fit uses about 47.6% of all labeled applicants
(`0.85 × 0.80 × 0.70`), so these results describe the fold-trained procedure,
not the separately saved dashboard model fitted on 70% of all labeled rows.

## What happens inside a fold

1. Fit preprocessing and ranking models using only base-fitting applicants.
   Aggregate encoded-column gain back to raw fields. Repeat with ranking model
   seeds `101`, `211`, and `307`, average raw-feature ranks, and break ties by
   raw field name. Rank dispersion is recorded. Repeats use the same fitting
   rows and measure sensitivity to model randomness, not new independent data.
2. Compare top-40, top-80, and full feature surfaces by default. Each surface
   uses the existing bounded LightGBM candidate grid, fitted on base-fitting
   rows and chosen using selection-validation metrics.
3. Fit sigmoid/isotonic transforms only on calibration-fitting rows. Select
   raw/sigmoid/isotonic using selection-validation Brier score and the existing
   minimum-improvement/simplicity rules.
4. Choose the feature surface using the existing validation ranking rule:
   average precision, lift, top-score capture, ROC-AUC, lower Brier, and fewer
   features. Derive raw-score scenario thresholds from selection validation.
   These validation results are selection evidence and may be optimistic.
5. Freeze the selected pipeline, feature list, calibrator, and thresholds.
   Only the chosen history feature/grid/calibration workflow predicts the outer fold. Unselected history candidates receive no outer metrics. Separately declared independent comparators each predict that exact fold; their results are never promotion inputs. No refit on reserved rows
   occurs after selection; doing so would change the calibrated base model.

The candidate-fitting interface accepts only fitting, calibration, and
selection roles. The assessment interface makes no choices and rejects overlap
with those roles. Historical comparison labels never enter the nested runner.

## Seeds and reproducibility

| Setting | Controls |
|---|---|
| `project.split_seed` | Ordinary pipeline partitioning; default 42 |
| `project.model_seed` | Base-model and calibration fitting; default 42, fixed across outer folds |
| `feature_selection.ranking_seeds` | Model repeats for feature ranking; defaults 101, 211, 307 |
| `assessment.split_seeds` | Outer-fold assignments; default `[42]` |
| Derived inner split seed | Deterministic from outer seed and fold; persisted in the manifest |

`project.random_seed` remains the fallback for older configs. Changing model
randomness does not change partition membership. The existing stability runner
varies split seeds `17`, `29`, and `43` with fixed model/ranking seeds; its
validation summaries remain development sensitivity checks. It does not replace
outer assessment. Its legacy `seed` CSV column aliases `split_seed`.

Declare settings before inspecting results. Outer summaries do not automatically
promote a feature surface or rewrite the dashboard model. If results motivate a
new recipe, disclose that reuse when assessing the revision.

## Running and inspecting the evidence

After building/training compatible post-v1 artifacts:

```bash
make assess-post-v1 PYTHON=python
```

Omit `PYTHON=python` where the default interpreter works. To use another scoped
config, pass `CONFIG_POST_V1=path/to/config.yaml`. The command reads the existing
mart/model population reference, recomputes eligible raw feature columns from
the mart contract, and leaves dashboard model artifacts and database tables
unchanged. It is separate from `pipeline-post-v1` because it fits multiple
workflows. Default work is five folds, three feature surfaces, up to eight
base-model candidates per surface, plus three ranking fits per fold, eight application-only candidates, one fixed logistic fit and one prevalence fit per fold. Baseline optimization budgets are deliberately different and explicitly recorded.

Each invocation creates a unique ignored directory:
`<report_dir>/nested_assessment/<run_id>/`. Inspect:

| Artifact | Meaning |
|---|---|
| `manifest.json` | Protocol/settings, source build identity, reference-model identity, exact local role IDs, ID/target/config fingerprints, seeds, and completion status |
| `inner_selection.csv` | Fold-local candidate metrics and selected surface |
| `training_rankings.csv` | Mean raw-feature ranks and dispersion across ranking seeds |
| `fold_<split_seed>_<fold>_<workflow>.joblib` | Frozen selected pipeline, calibrators, feature list, and thresholds |
| `assessment_predictions.csv` | One outer prediction per development applicant per workflow/repeat, with raw/calibrated views |
| `fold_metrics.csv`, `summary.csv` | Outer metrics and descriptive mean/SD across folds, separately by split seed and score view |
| `assessment_report.md` | Readable assessment results and limitations |

Exact CSV columns belong to [report contracts](../../src/report_contracts.py).
Partial runs retain a failed manifest and must not be presented as complete
evidence. Local IDs/build identities and fingerprints do not constitute full
immutable content lineage. Curated historical reports, PBIX files, and screenshots
are not rewritten by this command.

## Matched comparators and declared diagnostics

Application-only fields come from SQL `f_applicant_static`, intersected with allowed mart columns; name-prefix heuristics do not define origin. Its full application surface uses eight LightGBM candidates and the same reserved calibration/method-selection roles. Fixed logistic regression uses all allowed history fields, training-only median/mode imputation, numeric scaling, unknown-safe one-hot encoding, C=1, lbfgs, balanced class weights, max_iter=1000 and seed 42. It has no hyperparameter search. Training prevalence uses only fitting labels, no search/calibration, and no meaningful quantile policy threshold; utility is omitted.

The predeclared negative control selects a stratified 20,000-row development sample before seed-913 label permutation, then executes the complete five-fold ranking, feature/grid fitting/selection, calibration and frozen outer assessment. It remains a sampled diagnostic, separate from full-data performance. Historical comparison and Kaggle applicants never enter it.

Reliability bins use empirical score-group midranks and keep equal scores together; ten nominal bins can be empty and counts are mandatory. Top-rate evaluation uses equal fractional membership within the score tie crossing `ceil(n*rate)`, so flat scores have lift 1 independent of row order. Score/ID display rank bins are a separate target-blind ordering rule. Log loss accompanies Brier/AP/ROC. Fold-paired differences, known/unknown/no-history segment metrics and feature/calibration/model choices are frozen descriptive diagnostics. The utility grid uses all 27 combinations of margin/loss/review multipliers 0.5/1/2 at frozen raw balanced thresholds, with no outer tuning.

The named [correctness assessment](../../reports/correctness_20261004/assessment_report.md) records full raw checksums, source/config/dirty Git identity, locked environment, completed runs and clean first-fold reproduction within absolute prediction tolerance 1e-10. Exact local role IDs/joblibs remain ignored; only anonymous aggregates are public. Run `src.assessment_diagnostics` on a complete assessment, `src.assessment_reproduce` from a second locked environment, and `src.correctness_summary` after complete original-label/control/reproduction checks. These commands do not promote the dashboard model or refresh native Power BI assets.

## How to interpret the results

- Outer metrics assess the declared selection procedure. Different folds may
  select different features, model settings, or calibrators; that is expected.
- Fold SD is descriptive variability, not a confidence interval. Fitted models
  share training observations. The runner does not pool cross-fold scores into
  one average-precision claim, since score scales can differ between fits.
- Brier measures probability quality; inspect calibration separately. Utility
  retains illustrative weights and raw-score quantile scenarios, without a hard
  review cap or inferred reviewer effectiveness.
- This public dataset has already influenced past choices. Nested assessment
  improves separation within the new procedure, but does not manufacture an
  untouched external test.
- Reliable calendar application timestamps are unavailable in the input. Relative
  history offsets and applicant IDs do not establish application chronology.
  Future-cohort validation remains a documented dataset limitation.

The rationale follows [selection/assessment separation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
and [separate calibration data](https://scikit-learn.org/stable/modules/calibration.html).
[Banking model-risk guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)
also emphasizes purpose-appropriate testing and documented limitations; it does
not prescribe this portfolio's split fractions, fold count, or seed choices.
