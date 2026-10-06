# Current model-selection assessment, 2026-10-04

**Current completed procedure.** Start with [the case study](../portfolio/case_study.md) for the engineering story. The [earlier repaired assessment](../correctness_20261004/assessment_report.md) was completed on the same date under a different protocol; its values remain separate.

Search, seed roles and probability criteria were fixed before assessment results. No outer result promoted a model, family, surface or seed. The prior report and its source/config/evidence identities remain preserved; 155 protected files matched their pre-edit hashes.

| Model / input scope | Average precision | ROC AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Application fields only · LightGBM | 0.231021 | 0.748056 | 0.068423 | 0.248987 |
| Application and loan history · LightGBM | 0.265862 | 0.775862 | 0.066655 | 0.240720 |
| Application and loan history · logistic regression | 0.251083 | 0.766675 | 0.067396 | 0.243982 |
| Training outcome rate · constant benchmark | 0.080728 | 0.500000 | 0.074211 | 0.280544 |

These are means of five matched applicant test-group metrics. Average precision measures ranking, not accuracy; Brier/log loss measure probability error (lower is better). Final probabilities reflect the selected method, including unchanged raw probabilities when no adjustment is selected. All five current history fits retained raw probabilities. Descriptive sample SD is in
`summary.csv`. No pooled cross-fold average precision or optimality claim is made.
`prior_protocol_comparison.csv` records v3 minus prior v2 results; tuned unweighted
logistic is compared explicitly with the prior balanced fixed-C1 comparator, so
that difference mixes changed recipe and optimization budget. Paired differences
within v3 use identical outer applicants, not independent samples or causal effects.
History final-probability average precision changed by +0.000112, essentially unchanged.
Raw history probability losses are much lower than r2; final probability losses are
similar. Application ranking fell slightly and tuned logistic ranking rose.
Changed recipes and budgets prevent this search comparison alone from attributing
differences to one parameter, and these descriptive results do not establish
significance or guaranteed gains.

## Follow-up: isolate the class-weighting effect

The separate [controlled class-weighting comparison](../class_weighting_20261004/assessment_report.md) changes only `scale_pos_weight` within each frozen earlier/current history recipe and applicant group. Removing the earlier 9–11× positive weight reduces earlier-recipe raw Brier/log loss from 0.157305/0.477672 to 0.066572/0.240121, essentially reproducing current losses of 0.066655/0.240720. Mean raw predicted risk falls from 34.20% to 7.94%, against 8.07% observed difficulty. Adding the earlier weight to the current recipe worsens raw Brier/log loss to 0.176169/0.525676.

Both losses improve without weighting in all ten recipe/group pairs. All ten original-weight refits reproduce frozen raw predictions exactly. This identifies weighting as the dominant explanation for the raw-scale improvement conditional on these selected recipes; it does not isolate the search screen's selection effect, establish future-cohort validity or promote a diagnostic model. The original numerical evidence and execution fingerprints in this directory remain unchanged. The diagnostic examines raw probabilities only; the original final-method results remain authoritative.

## Fitting and selection methods

All 261,384 development applicants receive one frozen prediction per declared
workflow, excluding the same 46,127 historical IDs. V3 and r2 outer and reserved
role memberships match exactly. Three-fold CV remains entirely within the 70%
base-fitting role; separate stopping rows, per-fold training-only raw rankings and
preprocessing prevent inner score-fold leakage. Twenty-four unique joint candidates
per LightGBM workflow include unweighted/lighter-weight settings. Final fits use
the median chosen inner best iterations before disjoint calibration and method
selection. Searchable history surfaces are 40/80/full 174; application-only uses 31
SQL-origin static fields. Tuned logistic searches four C values; prevalence has no
optimization or policy utility. Constant-score recipes omit quantile-policy utility.

`cv_search_summary.csv` includes all 240 LightGBM candidate summaries (24 ×
two workflows × five outer folds), each averaging three inner score folds,
with CV losses versus fitting prevalence and chosen
round evidence. Full exact local memberships/rankings/iterations remain ignored.
`probability_acceptance.csv` records predefined calibrated selection-role checks;
these cannot promote or retune using outer quality. `model_seed_sensitivity.csv`
reports fixed-recipe selection-role variation at seeds 101/211/307 without choosing
a lucky seed; logistic lbfgs may have no seed variation. Ranking seeds have a
separate training-only role.
60 of 240 joint candidates pass mean CV probability criteria;
all 10 selected LightGBM recipes pass. Original-label history
selects unweighted full-field recipes with median rounds ranging
380–579.
The largest within-fold history AP range across sensitivity seeds is
0.003783; this describes variation on selection rows.
`independent_verification.json` records recomputed metric/paired-difference
agreement, coverage, probability flags and sensitivity ranges without applicant IDs.

## Complete shuffled-label diagnostic

| Model / input scope | Average precision | ROC AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Application fields only · LightGBM | 0.083875 | 0.501050 | 0.074353 | 0.281354 |
| Application and loan history · LightGBM | 0.083556 | 0.504891 | 0.074406 | 0.281720 |
| Application and loan history · logistic regression | 0.087017 | 0.508619 | 0.074242 | 0.280676 |
| Training outcome rate · constant benchmark | 0.080750 | 0.500000 | 0.074229 | 0.280597 |

The stratified sampled 20k development control permutes labels at seed 913 and runs
the complete five-fold selection/calibration/comparator procedure. Its chance
behavior has sampling variation and cannot prove feature availability or erase
historical exploration. It is separate from full-data original-label performance.

## Runtime, reproduction and limits

The real-data pilot used 20k/3k/3k fitting/calibration/selection rows and peaked at
1.47 GiB. Original assessment took
42.18 minutes; control took 4.41 minutes.
Model jobs ran serially with four-thread LightGBM/BLAS limits. All original/control
frozen artifacts reproduce within absolute score tolerance 1e-10. A first-fold
refit in the second locked environment matches selected features/parameters,
calibration and predictions within 1e-10. `provenance.json` records measured
timings, locked environment, source/config/build identities and preservation proof.
Exact executed source is retained in the ignored execution archive. Post-run
ordinary-caller metadata/report-path corrections, support for independently
selected model fields in evaluation, and winner-only stability frequency and
conditional uncertainty corrections are recorded as delivery source deltas.
Regression fixtures force top-40/top-80 CV winners through ordinary evaluation,
calibration, scoring, explanation and dashboard export; legacy equal-field
artifacts remain supported. Current stability summaries make no aggregate
selection or promotion. Executed joint-search/nested scientific source and all
frozen numeric evidence remain unchanged; execution fingerprints stay fixed.

Source/mart reconciliation from r2 is reused with its fingerprints. Native Power BI
Desktop/tooling is unavailable, so PBIX refresh is unverified and historical
screenshots remain unchanged. Dataset IDs/relative offsets do not identify calendar
application chronology, feature availability or label maturity. This remains an
already explored public static dataset, not an untouched external or future cohort.
Utility uses illustrative weights and is not profit or a hard queue guarantee.
No guarantee of improved performance or global tuning optimality is implied.

## Technical run details

- Protocol: `nested_inner_cv_v3`
- Assessment: `b45cb7fb871d4322bfdbb52e6e8d1d38`
- Sampled shuffled-outcome diagnostic: `706d7db1cbf5487aa96261cc4b5d9ab9`
- CSV keys remain machine identifiers: `calibrated` means final probabilities and can be unchanged raw values.
