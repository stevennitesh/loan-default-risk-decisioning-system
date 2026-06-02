from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.cli import add_config_argument
from src.cli import exit_with_error
from src.cli import format_int_csv
from src.cli import parse_int_csv
from src.config import DEFAULT_CONFIG_PATH
from src.config import load_config
from src.config import manual_review_capacity_rate
from src.feature_experiments import DEFAULT_FEATURE_LIMITS
from src.feature_experiments import FeatureExperimentError
from src.feature_experiments import load_lightgbm_artifact
from src.feature_experiments import prepare_feature_set_specs
from src.feature_experiments import run_single_feature_set
from src.feature_experiments import select_feature_set
from src.runtime import created_at_utc
from src.runtime import ensure_directories
from src.runtime import require_existing_path
from src.runtime import resolve_config_path
from src.runtime import write_csv
from src.modeling import load_labeled_training_frame
from src.modeling import split_labeled_frame
from src.report_contracts import MODEL_STABILITY_AGGREGATE_COLUMNS
from src.report_contracts import MODEL_STABILITY_RUN_COLUMNS


DEFAULT_STABILITY_SEEDS = (17, 29, 43)
MODEL_STABILITY_REPORT_NAME = "006_model_stability.md"

MEAN_STD_METRICS = [
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_review_capacity",
    "validation_weighted_calibration_error",
    "validation_balanced_ev_per_applicant",
    "test_pr_auc",
    "test_roc_auc",
    "test_brier_score",
    "test_top_decile_lift",
    "test_precision_at_top_decile",
    "test_recall_at_review_capacity",
    "test_weighted_calibration_error",
    "test_balanced_ev_per_applicant",
]


class ModelStabilityError(FeatureExperimentError):
    """Raised when the model-stability experiment cannot run safely."""


def run_model_stability_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    seeds: tuple[int, ...] = DEFAULT_STABILITY_SEEDS,
    feature_limits: tuple[int, ...] = DEFAULT_FEATURE_LIMITS,
    include_full: bool = True,
    seed_runs_name: str = "model_stability_seed_runs.csv",
    summary_name: str = "model_stability_summary.csv",
    report_name: str = MODEL_STABILITY_REPORT_NAME,
) -> dict[str, Any]:
    seeds = _normalize_seeds(seeds)
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    require_existing_path(duckdb_path, "DuckDB database", ModelStabilityError)

    base_artifact = load_lightgbm_artifact(model_dir, error_cls=ModelStabilityError)
    full_feature_columns = list(base_artifact["feature_columns"])

    created_at = created_at_utc()
    review_capacity_rate = manual_review_capacity_rate(config)
    feature_set_specs = prepare_feature_set_specs(
        report_dir,
        full_feature_columns,
        feature_limits,
        include_full,
        error_cls=ModelStabilityError,
    )
    run_rows: list[dict[str, Any]] = []
    with duckdb.connect(str(duckdb_path)) as connection:
        training_frame = load_labeled_training_frame(
            connection,
            full_feature_columns,
            error_cls=ModelStabilityError,
        )
        for seed in seeds:
            split_frames = split_labeled_frame(
                training_frame,
                config,
                seed,
                error_cls=ModelStabilityError,
            )
            seed_rows = []
            for feature_set_name, feature_columns, feature_limit in feature_set_specs:
                seed_rows.append(
                    run_single_feature_set(
                        config,
                        feature_set_name,
                        feature_columns,
                        feature_limit,
                        split_frames,
                        review_capacity_rate,
                        created_at,
                        random_seed=seed,
                        error_cls=ModelStabilityError,
                    )
                )
            seed_winner = select_feature_set(seed_rows)
            for row in seed_rows:
                run_row = dict(row)
                run_row.pop("selected", None)
                run_rows.append(
                    {
                        "seed": seed,
                        "seed_validation_winner": run_row["feature_set"] == seed_winner,
                        **run_row,
                    }
                )

    aggregate_rows = aggregate_stability_rows(run_rows, created_at)
    selected_feature_set = select_stability_feature_set(aggregate_rows)
    for row in aggregate_rows:
        row["selected"] = row["feature_set"] == selected_feature_set

    experiments_dir = report_dir / "experiments"
    ensure_directories(report_dir, experiments_dir)
    seed_runs_path = report_dir / seed_runs_name
    summary_path = report_dir / summary_name
    report_path = experiments_dir / report_name
    write_csv(seed_runs_path, MODEL_STABILITY_RUN_COLUMNS, run_rows)
    write_csv(summary_path, MODEL_STABILITY_AGGREGATE_COLUMNS, aggregate_rows)
    _write_report(report_path, aggregate_rows, selected_feature_set, seeds)

    return {
        "selected_feature_set": selected_feature_set,
        "run_rows": run_rows,
        "aggregate_rows": aggregate_rows,
        "seed_runs_path": seed_runs_path,
        "summary_path": summary_path,
        "report_path": report_path,
    }


def aggregate_stability_rows(
    run_rows: list[dict[str, Any]],
    created_at: str,
) -> list[dict[str, Any]]:
    if not run_rows:
        raise ModelStabilityError("At least one seed run row is required")

    frame = pd.DataFrame(run_rows)
    for metric in MEAN_STD_METRICS:
        frame[metric] = frame[metric].astype(float)
    seed_winners = {
        seed: select_feature_set(group.to_dict("records"))
        for seed, group in frame.groupby("seed", sort=True)
    }
    aggregate_rows = []
    for feature_set, group in frame.groupby("feature_set", sort=False):
        row: dict[str, Any] = {
            "feature_set": feature_set,
            "selected": False,
            "feature_count": int(group["feature_count"].iloc[0]),
            "feature_limit": group["feature_limit"].iloc[0],
            "seed_count": int(group["seed"].nunique()),
            "validation_win_count": sum(
                1 for seed, winner in seed_winners.items() if winner == feature_set
            ),
            "created_at": created_at,
        }
        row["validation_win_rate"] = row["validation_win_count"] / row["seed_count"]
        for metric in MEAN_STD_METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = _std(values)
        row["pr_auc_generalization_gap"] = row["test_pr_auc_mean"] - row["validation_pr_auc_mean"]
        row["abs_pr_auc_generalization_gap"] = abs(row["pr_auc_generalization_gap"])
        row["balanced_ev_generalization_gap"] = (
            row["test_balanced_ev_per_applicant_mean"]
            - row["validation_balanced_ev_per_applicant_mean"]
        )
        row["abs_balanced_ev_generalization_gap"] = abs(row["balanced_ev_generalization_gap"])
        aggregate_rows.append(row)
    return aggregate_rows


def select_stability_feature_set(rows: list[dict[str, Any]]) -> str:
    selected = max(rows, key=_stability_selection_key)
    return str(selected["feature_set"])


def _normalize_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ModelStabilityError("At least one seed is required")
    if len(set(normalized)) != len(normalized):
        raise ModelStabilityError("Seeds must be unique")
    return normalized


def _stability_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, int]:
    return (
        float(row["validation_pr_auc_mean"]),
        float(row["validation_win_rate"]),
        float(row["validation_top_decile_lift_mean"]),
        float(row["validation_recall_at_review_capacity_mean"]),
        float(row["validation_roc_auc_mean"]),
        -float(row["validation_brier_score_mean"]),
        -float(row["validation_pr_auc_std"]),
        -int(row["feature_count"]),
    )


def _write_report(
    path: Path,
    aggregate_rows: list[dict[str, Any]],
    selected_feature_set: str,
    seeds: tuple[int, ...],
) -> None:
    table_lines = "\n".join(
        "| {feature_set} | {feature_count} | {seed_count} | {validation_win_rate:.2f} | "
        "{validation_pr_auc_mean:.6f} | {validation_pr_auc_std:.6f} | "
        "{validation_brier_score_mean:.6f} | {validation_top_decile_lift_mean:.6f} | "
        "{test_pr_auc_mean:.6f} | {test_brier_score_mean:.6f} | "
        "{test_balanced_ev_per_applicant_mean:.2f} | {selected} |".format(**row)
        for row in aggregate_rows
    )
    selected_row = next(row for row in aggregate_rows if row["feature_set"] == selected_feature_set)
    interpretation_text = _interpretation_text(selected_row)
    text = f"""# Experiment 006: Model Stability

## Purpose

Check whether the feature-selection result is stable across repeated split/training seeds before promoting a smaller model surface.

## Process

This experiment reruns the same LightGBM tuning and sigmoid/isotonic/uncalibrated calibration comparison across seeds `{_seed_list(seeds)}`. Each seed creates a fresh stratified train/validation/test split from labeled `application_train`, then trains the candidate feature surfaces independently.

## Selection Rule

The selected setup is chosen with a validation-only aggregate rule: mean validation PR-AUC first, then validation win rate, mean top-decile lift, mean recall at review capacity, mean ROC-AUC, lower mean Brier score, lower validation PR-AUC variability, and fewer features. Held-out test is reported after selection as a generalization check and is not used to choose the setup.

## Results

| Feature set | Features | Seeds | Val win rate | Val PR-AUC mean | Val PR-AUC std | Val Brier mean | Val lift mean | Test PR-AUC mean | Test Brier mean | Test EV/app mean | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table_lines}

## Selected Setup

Selected feature set: `{selected_feature_set}` with {selected_row['feature_count']} features.

## Generalization Check

For the selected setup, mean test PR-AUC minus mean validation PR-AUC is {selected_row['pr_auc_generalization_gap']:.6f}, and mean test balanced EV minus mean validation balanced EV is {selected_row['balanced_ev_generalization_gap']:.2f}. These held-out test values are final verification signals, not optimization inputs.

## Interpretation

{interpretation_text}

## Notes

This experiment changes the model-generation evidence only. It does not add new source tables, demographic/protected-status-like fields, or a new decision policy.
"""
    path.write_text(text, encoding="utf-8")


def _interpretation_text(selected_row: dict[str, Any]) -> str:
    selected_feature_set = str(selected_row["feature_set"])
    if selected_feature_set == "full":
        return (
            "The repeated-seed result does not support promoting the smaller `top_100` surface yet. "
            "The full feature set has the strongest mean validation PR-AUC under the validation-only "
            "aggregate rule, so it remains the better active model candidate until a smaller setup "
            "wins a stability pass or the project explicitly prioritizes simplicity over validation lift."
        )
    return (
        f"`{selected_feature_set}` remains the selected setup after repeated-seed validation. "
        "That supports promoting the smaller feature surface because the improvement was not limited "
        "to a single split/training seed."
    )


def _std(values: pd.Series) -> float:
    result = float(values.std(ddof=0))
    if np.isnan(result):
        return 0.0
    return result


def _seed_list(seeds: tuple[int, ...]) -> str:
    return ", ".join(str(seed) for seed in seeds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare model-generation stability across repeated seeds.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "--seeds",
        default=format_int_csv(DEFAULT_STABILITY_SEEDS),
        help="Comma-separated random seeds for split/training repeats.",
    )
    parser.add_argument(
        "--feature-limits",
        default=format_int_csv(DEFAULT_FEATURE_LIMITS),
        help="Comma-separated top-N feature limits to compare.",
    )
    parser.add_argument("--skip-full", action="store_true", help="Do not include the full feature set.")
    parser.add_argument(
        "--seed-runs-name",
        default="model_stability_seed_runs.csv",
        help="CSV filename for per-seed stability rows under the report directory.",
    )
    parser.add_argument(
        "--summary-name",
        default="model_stability_summary.csv",
        help="CSV filename for aggregate stability rows under the report directory.",
    )
    parser.add_argument(
        "--report-name",
        default=MODEL_STABILITY_REPORT_NAME,
        help="Markdown report filename under reports/experiments.",
    )
    args = parser.parse_args()

    seeds = parse_int_csv(args.seeds)
    feature_limits = parse_int_csv(args.feature_limits)
    try:
        run_model_stability_experiment(
            args.config,
            seeds=seeds,
            feature_limits=feature_limits,
            include_full=not args.skip_full,
            seed_runs_name=args.seed_runs_name,
            summary_name=args.summary_name,
            report_name=args.report_name,
        )
    except ModelStabilityError as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
