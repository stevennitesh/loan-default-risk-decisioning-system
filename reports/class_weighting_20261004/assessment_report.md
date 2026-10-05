# Why raw probabilities improved: controlled class-weighting comparison

Removing positive-class weighting is the dominant explanation for the improvement in raw probability quality in these fixed recipes. Brier score and log loss improve in every one of the ten recipe/group pairs; adding the earlier weight to the current recipe reverses the benefit.

## What changed in this experiment

This retrospective diagnostic refits the earlier and current history-model recipes with and without positive-class weighting. Within each recipe/group pair, the only changed classifier parameter is `scale_pos_weight`: one means no weighting; the weighted condition uses that group's actual earlier fitted value (9.109–11.388). Applicant roles and order, selected inputs, frozen fitting-only preprocessing, every other model parameter, random seed 42, four threads and fixed tree counts remain unchanged. There is no retuning, early stopping, new calibration fitting or model promotion.

The two source assessments have identical role memberships and the same SQL feature build. The experiment makes 20 fits on five matched groups covering 261,384 distinct assessment applicants. Ten original-weight refits reproduce the corresponding saved raw predictions within 1e-10. The comparison conditions were written before fitting. These applicants were already exposed in earlier work; this is a mechanism check, not a fresh assessment.

## Results

| Recipe held fixed | Positive weighting | Mean predicted risk | Raw Brier ↓ | Raw log loss ↓ | Average precision ↑ |
|---|---|---:|---:|---:|---:|
| Earlier | Earlier 9–11× | 34.20% | 0.157305 | 0.477672 | 0.265750 |
| Earlier | None (1×) | 7.94% | 0.066572 | 0.240121 | 0.267124 |
| Current | Earlier 9–11× | 37.91% | 0.176169 | 0.525676 | 0.264770 |
| Current | None (1×) | 8.03% | 0.066655 | 0.240720 | 0.265862 |


The observed repayment-difficulty rate is **8.07%** in every condition's combined assessment population. Table values are means of five separate group metrics; sample SD and every paired effect are in the CSVs. Average predicted risk matching the observed rate alone does not prove calibration. Brier score and log loss provide additional probability-quality checks.

On the earlier-recipe path, changing weighting alone reproduces 100.1% of the previously observed raw Brier reduction and 100.3% of the raw log-loss reduction. These are descriptive ratios, not an additive causal decomposition: changing the other recipe settings can interact with weighting. Values above 100% can occur if the earlier unweighted recipe has slightly lower loss than the current one. The four conditions retain any ranking tradeoffs rather than treating better probability quality as better ranking.

## Interpretation

Weighting makes errors on the rare repayment-difficulty class more costly during fitting. The resulting score can remain useful for ranking while overstating probabilities at the actual outcome prevalence. The [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) warns that class weighting can impair individual class probability estimates and suggests calibration.

The current search allowed unweighted candidates and screened raw Brier/log loss before ranking among acceptable candidates. This experiment isolates the effect of the chosen models' weight setting; it does not separately test the search screen's contribution to choosing them. Earlier sigmoid adjustment had already repaired much of the raw scale distortion, explaining why final probability quality changed little in the original search comparison. This diagnostic assesses raw probabilities only; it makes no new calibrated-performance claim.

## Scope and evidence

Conditional on these selected history-model recipes, feature surfaces, five random groups and one model seed, the intervention isolates class weighting. It does not establish a causal relationship between applicant characteristics and repayment difficulty, optimal weighting for other models, significance across independent training resamples, or performance in future cohorts. All scientific assessment artifacts remain unchanged; no diagnostic variant is promoted.

[Fold results](fold_metrics.csv) · [Descriptive means and SD](summary.csv) · [Paired effects, unweighted minus weighted](paired_differences.csv) · [Run declaration, inputs and reproduction checks](provenance.json) · [Current case study](../portfolio/case_study.md)

To repeat using local frozen assessments and the original feature database, run `python -m src.class_weighting_ablation --earlier-dir <earlier-completed-assessment> --current-dir <current-completed-assessment> --output <fresh-ignored-output>`. Then curate with `python -m src.class_weighting_report --run-dir <completed-output> --output <fresh-final-evidence-folder>`. Exact source-run identities and parameter/hash records are in provenance; exact original applicant memberships and fitted models remain local. The offline portfolio renderer needs only these anonymous aggregate files.
