"""Isolate class weighting in frozen history-model recipes; never select a model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
import uuid
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evidence import file_sha256
from src.feature_experiments import load_split_frames
from src.metrics import probability_metrics
from src.runtime import REPO_ROOT, feature_frame, resolve_config_path

PROTOCOL = "fixed_recipe_class_weighting_v1"
METRICS = ("pr_auc", "roc_auc", "brier_score", "log_loss", "mean_probability")
DECLARATION = {
    "question": "Does removing positive-class weighting explain improved raw probability quality?",
    "workflow": "history_selected",
    "recipes": ["earlier", "current"],
    "variants": ["earlier_weight", "unweighted"],
    "intervention": "Only classifier scale_pos_weight changes within each recipe/fold pair.",
    "held_fixed": [
        "applicant role memberships and row order",
        "feature build and selected input columns",
        "frozen fitting-only preprocessing",
        "all other classifier parameters, including seed, threads and number of trees",
    ],
    "positive_weight": "Use that fold's actual earlier fitted weight for both recipes; unweighted is 1.",
    "deciding_observation": "Unweighted raw Brier score and log loss improve in both recipes across all five matched folds; compare predicted prevalence with observed prevalence and report ranking tradeoffs.",
    "reference_check": "Each original-weight refit must reproduce its frozen raw predictions within 1e-10.",
    "selection": "No search, early stopping, calibration fitting, model promotion or threshold changes.",
    "limits": "Retrospective mechanism check on previously exposed random groups, conditional on selected recipes and model seed. Not a population causal claim, full search-policy ablation, or future-cohort proof.",
}


def validate_manifests(earlier: dict, current: dict) -> None:
    """Require the identical scientific population and all role memberships."""
    for manifest in (earlier, current):
        if manifest["status"] != "complete" or manifest["negative_control"]:
            raise ValueError("Requires completed original-label assessments")
    for key in (
        "feature_build_id",
        "development_id_target_sha256",
        "historical_comparison_sha256",
        "feature_columns",
    ):
        if not earlier.get(key) or earlier[key] != current[key]:
            raise ValueError(f"Assessments disagree on {key}")
    if earlier["folds"] != current["folds"]:
        raise ValueError("Assessments must have identical applicant roles and ordering")
    if len(current["folds"]) != 5:
        raise ValueError("This declared diagnostic requires five matched folds")
    assessment_ids = []
    historical = set(current["historical_comparison_ids"])
    for plan in current["folds"]:
        seen = set()
        for ids in plan["applicant_ids"].values():
            if len(ids) != len(set(ids)) or seen.intersection(ids):
                raise ValueError("Applicant roles contain duplicates or overlap")
            if historical.intersection(ids):
                raise ValueError(
                    "Historical comparison applicants entered the diagnostic"
                )
            seen.update(ids)
        assessment_ids.extend(plan["applicant_ids"]["assessment"])
    if len(assessment_ids) != len(set(assessment_ids)):
        raise ValueError("Assessment applicants must occur once across folds")


def weight_variant(classifier, weight: float):
    """Clone an unfitted classifier, rejecting competing weighting mechanisms."""
    params = classifier.get_params()
    if params.get("class_weight") is not None or params.get("is_unbalance", False):
        raise ValueError(
            "Competing class weighting prevents a one-parameter comparison"
        )
    if not np.isfinite(weight) or weight < 1:
        raise ValueError("Positive-class weight must be finite and at least one")
    changed = clone(classifier).set_params(scale_pos_weight=float(weight))
    if {k: v for k, v in params.items() if k != "scale_pos_weight"} != {
        k: v for k, v in changed.get_params().items() if k != "scale_pos_weight"
    }:
        raise ValueError("A parameter other than class weighting changed")
    return changed


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep fold means descriptive and retain paired weighting effects."""
    summary = rows.groupby(["recipe", "weighting"], sort=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    pairs = rows.pivot(
        index=["recipe", "outer_fold"], columns="weighting", values=list(METRICS)
    )
    differences = pd.DataFrame(index=pairs.index)
    for metric in METRICS:
        differences[f"{metric}_unweighted_minus_weighted"] = (
            pairs[(metric, "unweighted")] - pairs[(metric, "earlier_weight")]
        )
    return summary, differences.reset_index()


def run(earlier_dir: Path, current_dir: Path, output: Path) -> dict:
    sources = {"earlier": earlier_dir.resolve(), "current": current_dir.resolve()}
    manifests = {
        name: json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for name, path in sources.items()
    }
    validate_manifests(manifests["earlier"], manifests["current"])
    output.mkdir(parents=True, exist_ok=False)
    config = manifests["current"]["config"]
    database = resolve_config_path(config, "duckdb_path")
    manifest = {
        "status": "running",
        "protocol": PROTOCOL,
        "run_id": uuid.uuid4().hex,
        "declaration": DECLARATION,
        "source_runs": {k: m["run_id"] for k, m in manifests.items()},
        "source_manifests_sha256": {
            k: file_sha256(p / "manifest.json") for k, p in sources.items()
        },
        "feature_build_id": manifests["current"]["feature_build_id"],
        "database_sha256": file_sha256(database),
        "source_sha256": {
            name: file_sha256(REPO_ROOT / name)
            for name in (
                "src/class_weighting_ablation.py",
                "src/feature_experiments.py",
                "src/mart_access.py",
                "src/runtime.py",
                "src/modeling.py",
                "src/metrics.py",
                "requirements.lock",
            )
        },
        "python": platform.python_version(),
        "dependencies": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "recipe_parameters": [],
        "reference_checks": [],
    }
    manifest_path = output / "manifest.json"

    def persist():
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    # This records comparison conditions before any decisive fit.
    persist()
    started = time.perf_counter()
    rows = []
    try:
        for plan in manifests["current"]["folds"]:
            fold = plan["outer_fold"]
            filename = f"fold_{plan['split_seed']}_{fold}_history_selected.joblib"
            workflows = {k: joblib.load(p / filename) for k, p in sources.items()}
            weight = float(
                workflows["earlier"]["pipeline"]
                .named_steps["classifier"]
                .get_params()["scale_pos_weight"]
            )
            if (
                weight <= 1
                or workflows["current"]["pipeline"]
                .named_steps["classifier"]
                .get_params()["scale_pos_weight"]
                != 1
            ):
                raise ValueError(
                    "Expected the earlier weighted and current unweighted reference models"
                )
            with duckdb.connect(str(database), read_only=True) as con:
                frames = load_split_frames(
                    con,
                    {k: plan["applicant_ids"][k] for k in ("train", "assessment")},
                    manifests["current"]["feature_columns"],
                )
            for recipe, workflow in workflows.items():
                columns = workflow["feature_columns"]
                prefix = workflow["pipeline"][:-1]
                x_train = prefix.transform(feature_frame(frames["train"], columns))
                x_test = prefix.transform(feature_frame(frames["assessment"], columns))
                y_train, y_test = (
                    frames["train"]["TARGET"],
                    frames["assessment"]["TARGET"],
                )
                original = workflow["pipeline"].named_steps["classifier"]
                frozen_scores = original.predict_proba(x_test)[:, 1]
                manifest["recipe_parameters"].append(
                    {
                        "recipe": recipe,
                        "outer_fold": fold,
                        "feature_count": len(columns),
                        "artifact_sha256": file_sha256(sources[recipe] / filename),
                        "feature_columns": columns,
                        "parameters": original.get_params(),
                    }
                )
                for variant, value in (("earlier_weight", weight), ("unweighted", 1.0)):
                    classifier = weight_variant(original, value)
                    fit_started = time.perf_counter()
                    classifier.fit(x_train, y_train)
                    scores = classifier.predict_proba(x_test)[:, 1]
                    if value == original.get_params()["scale_pos_weight"]:
                        difference = float(np.max(np.abs(scores - frozen_scores)))
                        if difference > 1e-10:
                            raise ValueError(
                                f"Frozen recipe refit differs by {difference}"
                            )
                        manifest["reference_checks"].append(
                            {
                                "recipe": recipe,
                                "outer_fold": fold,
                                "max_absolute_prediction_difference": difference,
                            }
                        )
                    measured = probability_metrics(y_test, scores, 0.1)
                    row = {
                        "recipe": recipe,
                        "weighting": variant,
                        "split_seed": plan["split_seed"],
                        "outer_fold": fold,
                        "positive_class_weight": value,
                        "feature_count": len(columns),
                        "training_count": len(y_train),
                        "applicant_count": len(y_test),
                        "observed_rate": float(y_test.mean()),
                        "mean_probability": float(scores.mean()),
                        **{
                            key: float(measured[key])
                            for key in METRICS
                            if key in measured
                        },
                    }
                    rows.append(row)
                    print(
                        json.dumps(
                            {**row, "fit_seconds": time.perf_counter() - fit_started}
                        ),
                        flush=True,
                    )
                del x_train, x_test
            pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
            persist()
        frame = pd.DataFrame(rows)
        summary, differences = summarize(frame)
        summary.to_csv(output / "summary.csv", index=False)
        differences.to_csv(output / "paired_differences.csv", index=False)
        manifest.update(
            status="complete",
            duration_seconds=time.perf_counter() - started,
            applicant_count=int(
                frame.loc[
                    (frame.recipe == "current") & (frame.weighting == "unweighted"),
                    "applicant_count",
                ].sum()
            ),
        )
        persist()
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier-dir", required=True, type=Path)
    parser.add_argument("--current-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.earlier_dir, args.current_dir, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "run_id", "duration_seconds", "applicant_count")
            }
        )
    )


if __name__ == "__main__":
    main()
