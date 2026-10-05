"""Verify and curate anonymous results from the fixed-recipe weighting diagnostic."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.evidence import file_sha256

METRICS = ("pr_auc", "roc_auc", "brier_score", "log_loss", "mean_probability")
FILES = ("fold_metrics.csv", "summary.csv", "paired_differences.csv", "provenance.json")


def load_evidence(source: Path, current_run: str | None = None) -> dict:
    """Check matched anonymous aggregates and independently recompute their means."""
    manifest = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "complete"
        or manifest["protocol"] != "fixed_recipe_class_weighting_v1"
    ):
        raise ValueError("Requires a completed fixed-recipe class-weighting diagnostic")
    if current_run is not None and manifest["source_runs"]["current"] != current_run:
        raise ValueError(
            "Weighting diagnostic belongs to a different current assessment"
        )
    for name, digest in manifest["aggregate_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("Weighting aggregate fingerprint differs from provenance")
    folds = pd.read_csv(source / "fold_metrics.csv")
    expected = {
        (recipe, weighting, fold)
        for recipe in ("earlier", "current")
        for weighting in ("earlier_weight", "unweighted")
        for fold in range(1, 6)
    }
    grain = ["recipe", "weighting", "outer_fold"]
    if len(folds) != 20 or set(map(tuple, folds[grain].to_numpy())) != expected:
        raise ValueError(
            "Requires exactly four recipe/weight conditions on five groups"
        )
    if {"SK_ID_CURR", "applicant_id", "target", "probability"} & set(folds):
        raise ValueError("Weighting evidence must contain anonymous aggregates only")
    for _fold, rows in folds.groupby("outer_fold"):
        if any(
            rows[key].nunique() != 1
            for key in (
                "split_seed",
                "training_count",
                "applicant_count",
                "observed_rate",
            )
        ):
            raise ValueError(
                "Weighting comparisons must use matched applicants and outcomes"
            )
        for _recipe, pair in rows.groupby("recipe"):
            if pair.feature_count.nunique() != 1:
                raise ValueError("Within-recipe features changed")
        if (
            not rows.loc[rows.weighting == "unweighted", "positive_class_weight"]
            .eq(1)
            .all()
        ):
            raise ValueError("Unweighted condition must have weight one")
        weights = rows.loc[rows.weighting == "earlier_weight", "positive_class_weight"]
        if weights.nunique() != 1 or weights.iloc[0] <= 1:
            raise ValueError("Both recipes must use the same earlier positive weight")
    checks = manifest["reference_checks"]
    if (
        len(checks) != 10
        or {(c["recipe"], c["outer_fold"]) for c in checks}
        != {(recipe, fold) for recipe in ("earlier", "current") for fold in range(1, 6)}
        or any(c["max_absolute_prediction_difference"] > 1e-10 for c in checks)
    ):
        raise ValueError("Original recipe prediction reproduction is incomplete")
    count = int(
        folds.loc[
            (folds.recipe == "current") & (folds.weighting == "unweighted"),
            "applicant_count",
        ].sum()
    )
    if count != manifest["applicant_count"]:
        raise ValueError("Weighting applicant count differs from provenance")
    summary = pd.read_csv(source / "summary.csv")
    expected_summary = folds.groupby(["recipe", "weighting"])[list(METRICS)].agg(
        ["mean", "std"]
    )
    expected_summary.columns = [f"{m}_{s}" for m, s in expected_summary.columns]
    actual_summary = summary.set_index(["recipe", "weighting"]).sort_index()
    if actual_summary.shape != expected_summary.shape or not np.allclose(
        actual_summary, expected_summary, rtol=0, atol=1e-12
    ):
        raise ValueError("Weighting summary disagrees with fold metrics")
    paired = (
        pd.read_csv(source / "paired_differences.csv")
        .set_index(["recipe", "outer_fold"])
        .sort_index()
    )
    pivot = folds.pivot(
        index=["recipe", "outer_fold"], columns="weighting", values=list(METRICS)
    )
    expected_pairs = pd.DataFrame(
        {
            f"{metric}_unweighted_minus_weighted": pivot[(metric, "unweighted")]
            - pivot[(metric, "earlier_weight")]
            for metric in METRICS
        }
    )
    if paired.shape != expected_pairs.shape or not np.allclose(
        paired, expected_pairs, rtol=0, atol=1e-12
    ):
        raise ValueError("Weighting paired effects disagree with fold metrics")
    probability_improved = bool(
        (paired.brier_score_unweighted_minus_weighted < 0).all()
        and (paired.log_loss_unweighted_minus_weighted < 0).all()
    )
    return {
        "provenance": manifest,
        "fold_metrics": folds,
        "summary": summary,
        "paired_differences": paired,
        "probability_improved_all_pairs": probability_improved,
        "observed_rate": float(
            np.average(folds.observed_rate, weights=folds.applicant_count)
        ),
    }


def curate(run_dir: Path, destination: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Cannot curate an incomplete run")
    destination.mkdir(parents=True, exist_ok=False)
    for name in FILES[:-1]:
        shutil.copy2(run_dir / name, destination / name)
    manifest["aggregate_sha256"] = {
        name: file_sha256(destination / name) for name in FILES[:-1]
    }
    manifest["curation_source_sha256"] = file_sha256(Path(__file__))
    (destination / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    evidence = load_evidence(destination)
    rows = evidence["summary"].set_index(["recipe", "weighting"])
    table = "| Recipe held fixed | Positive weighting | Mean predicted risk | Raw Brier ↓ | Raw log loss ↓ | Average precision ↑ |\n|---|---|---:|---:|---:|---:|\n"
    for recipe, weighting in (
        ("earlier", "earlier_weight"),
        ("earlier", "unweighted"),
        ("current", "earlier_weight"),
        ("current", "unweighted"),
    ):
        row = rows.loc[(recipe, weighting)]
        table += f"| {recipe.capitalize()} | {'Earlier 9–11×' if weighting == 'earlier_weight' else 'None (1×)'} | {row.mean_probability_mean:.2%} | {row.brier_score_mean:.6f} | {row.log_loss_mean:.6f} | {row.pr_auc_mean:.6f} |\n"
    earlier_weighted, earlier_unweighted = (
        rows.loc[("earlier", "earlier_weight")],
        rows.loc[("earlier", "unweighted")],
    )
    current_unweighted = rows.loc[("current", "unweighted")]
    fractions = {
        metric: float(
            (earlier_weighted[f"{metric}_mean"] - earlier_unweighted[f"{metric}_mean"])
            / (
                earlier_weighted[f"{metric}_mean"]
                - current_unweighted[f"{metric}_mean"]
            )
        )
        for metric in ("brier_score", "log_loss")
    }
    conclusion = (
        "Removing positive-class weighting is the dominant explanation for the improvement in raw probability quality in these fixed recipes. Brier score and log loss improve in every one of the ten recipe/group pairs; adding the earlier weight to the current recipe reverses the benefit."
        if evidence["probability_improved_all_pairs"]
        else "The class-weighting effect is not consistent across every recipe/group pair. Inspect the paired results before making an attribution claim."
    )
    report = f"""# Why raw probabilities improved: controlled class-weighting comparison

{conclusion}

## What changed in this experiment

This retrospective diagnostic refits the earlier and current history-model recipes with and without positive-class weighting. Within each recipe/group pair, the only changed classifier parameter is `scale_pos_weight`: one means no weighting; the weighted condition uses that group's actual earlier fitted value (9.109–11.388). Applicant roles and order, selected inputs, frozen fitting-only preprocessing, every other model parameter, random seed 42, four threads and fixed tree counts remain unchanged. There is no retuning, early stopping, new calibration fitting or model promotion.

The two source assessments have identical role memberships and the same SQL feature build. The experiment makes 20 fits on five matched groups covering {manifest["applicant_count"]:,} distinct assessment applicants. Ten original-weight refits reproduce the corresponding saved raw predictions within 1e-10. The comparison conditions were written before fitting. These applicants were already exposed in earlier work; this is a mechanism check, not a fresh assessment.

## Results

{table}

The observed repayment-difficulty rate is **{evidence["observed_rate"]:.2%}** in every condition's combined assessment population. Table values are means of five separate group metrics; sample SD and every paired effect are in the CSVs. Average predicted risk matching the observed rate alone does not prove calibration. Brier score and log loss provide additional probability-quality checks.

On the earlier-recipe path, changing weighting alone reproduces {fractions["brier_score"]:.1%} of the previously observed raw Brier reduction and {fractions["log_loss"]:.1%} of the raw log-loss reduction. These are descriptive ratios, not an additive causal decomposition: changing the other recipe settings can interact with weighting. Values above 100% can occur if the earlier unweighted recipe has slightly lower loss than the current one. The four conditions retain any ranking tradeoffs rather than treating better probability quality as better ranking.

## Interpretation

Weighting makes errors on the rare repayment-difficulty class more costly during fitting. The resulting score can remain useful for ranking while overstating probabilities at the actual outcome prevalence. The [LightGBM documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html) warns that class weighting can impair individual class probability estimates and suggests calibration.

The current search allowed unweighted candidates and screened raw Brier/log loss before ranking among acceptable candidates. This experiment isolates the effect of the chosen models' weight setting; it does not separately test the search screen's contribution to choosing them. Earlier sigmoid adjustment had already repaired much of the raw scale distortion, explaining why final probability quality changed little in the original search comparison. This diagnostic assesses raw probabilities only; it makes no new calibrated-performance claim.

## Scope and evidence

Conditional on these selected history-model recipes, feature surfaces, five random groups and one model seed, the intervention isolates class weighting. It does not establish a causal relationship between applicant characteristics and repayment difficulty, optimal weighting for other models, significance across independent training resamples, or performance in future cohorts. All scientific assessment artifacts remain unchanged; no diagnostic variant is promoted.

[Fold results](fold_metrics.csv) · [Descriptive means and SD](summary.csv) · [Paired effects, unweighted minus weighted](paired_differences.csv) · [Run declaration, inputs and reproduction checks](provenance.json) · [Current case study](../portfolio/case_study.md)

To repeat using local frozen assessments and the original feature database, run `python -m src.class_weighting_ablation --earlier-dir <earlier-completed-assessment> --current-dir <current-completed-assessment> --output <fresh-ignored-output>`. Then curate with `python -m src.class_weighting_report --run-dir <completed-output> --output <fresh-final-evidence-folder>`. Exact source-run identities and parameter/hash records are in provenance; exact original applicant memberships and fitted models remain local. The offline portfolio renderer needs only these anonymous aggregate files.
"""
    (destination / "assessment_report.md").write_text(report, encoding="utf-8")
    return {
        "run_id": manifest["run_id"],
        "probability_improved_all_pairs": evidence["probability_improved_all_pairs"],
        "earlier_recipe_recovery_fraction": fractions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(curate(args.run_dir, args.output)))


if __name__ == "__main__":
    main()
