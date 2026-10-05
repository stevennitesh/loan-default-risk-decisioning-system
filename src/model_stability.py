from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cli import add_config_argument, exit_with_error, format_int_csv, parse_int_csv
from src.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    manual_review_capacity_rate,
    project_model_seed,
)
from src.feature_experiments import (
    DEFAULT_FEATURE_LIMITS,
    FeatureExperimentError,
    load_lightgbm_artifact,
    load_split_frames,
    prepare_feature_set_specs,
    ranking_seeds,
    run_joint_feature_sets,
    run_single_feature_set,
    select_feature_set,
)
from src.metrics import target_class_values
from src.model_artifacts import validate_feature_build
from src.presentation import feature_set_label
from src.report_contracts import (
    MODEL_STABILITY_AGGREGATE_COLUMNS,
    MODEL_STABILITY_RUN_COLUMNS,
    MODEL_STABILITY_SELECTED_AGGREGATE_COLUMNS,
)
from src.runtime import (
    created_at_utc,
    ensure_directories,
    require_existing_path,
    resolve_config_path,
    write_csv,
)

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
    """Run repeated-seed model stability experiments and write summary outputs."""
    seeds = _normalize_seeds(seeds)
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    require_existing_path(duckdb_path, "DuckDB database", ModelStabilityError)

    base_artifact = load_lightgbm_artifact(model_dir, error_cls=ModelStabilityError)
    if "calibration" not in base_artifact["split_applicant_ids"]:
        raise ModelStabilityError(
            "Missing reserved calibration split; retrain with the corrected protocol"
        )
    full_feature_columns = list(
        base_artifact.get("eligible_feature_columns", base_artifact["feature_columns"])
    )

    created_at = created_at_utc()
    review_capacity_rate = manual_review_capacity_rate(config)
    model_seed = project_model_seed(config)
    run_rows: list[dict[str, Any]] = []
    with duckdb.connect(str(duckdb_path)) as connection:
        validate_feature_build(connection, base_artifact, error_cls=ModelStabilityError)
        saved_split_frames = load_split_frames(
            connection,
            base_artifact["split_applicant_ids"],
            full_feature_columns,
            error_cls=ModelStabilityError,
        )
        for seed in seeds:
            split_frames = _resplit_development(saved_split_frames, seed)
            from src.tuning import uses_inner_cv

            if uses_inner_cv(config):
                seed_rows, _ = run_joint_feature_sets(
                    config,
                    split_frames,
                    full_feature_columns,
                    feature_limits,
                    include_full,
                    review_capacity_rate,
                    created_at,
                    ModelStabilityError,
                )
            else:
                feature_set_specs = prepare_feature_set_specs(
                    report_dir,
                    full_feature_columns,
                    feature_limits,
                    include_full,
                    error_cls=ModelStabilityError,
                    training_frame=split_frames["train"],
                    config=config,
                )
                seed_rows = []
                for (
                    feature_set_name,
                    feature_columns,
                    feature_limit,
                ) in feature_set_specs:
                    seed_rows.append(
                        run_single_feature_set(
                            config,
                            feature_set_name,
                            feature_columns,
                            feature_limit,
                            split_frames,
                            review_capacity_rate,
                            created_at,
                            random_seed=model_seed,
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
                        "split_seed": seed,
                        "model_seed": model_seed,
                        "ranking_seeds": format_int_csv(
                            ranking_seeds(config, ModelStabilityError)
                        ),
                        "seed_validation_winner": run_row["feature_set"] == seed_winner,
                        **run_row,
                    }
                )

    from src.tuning import uses_inner_cv

    current = uses_inner_cv(config)
    feature_specs = {f"top_{limit}": (limit, limit) for limit in feature_limits}
    if include_full:
        feature_specs["full"] = (len(full_feature_columns), None)
    aggregate_rows = aggregate_stability_rows(
        run_rows,
        created_at,
        winner_only=current,
        completed_seeds=seeds,
        feature_specs=feature_specs,
    )
    selected_feature_set = (
        None if current else select_stability_feature_set(aggregate_rows)
    )
    for row in aggregate_rows:
        row["selected"] = row["feature_set"] == selected_feature_set

    output_dir = report_dir / "tuning" / "split_stability" if current else report_dir
    experiments_dir = output_dir if current else report_dir / "experiments"
    ensure_directories(output_dir, experiments_dir)
    seed_runs_path = output_dir / seed_runs_name
    summary_path = output_dir / summary_name
    report_path = experiments_dir / report_name
    write_csv(seed_runs_path, MODEL_STABILITY_RUN_COLUMNS, run_rows)
    write_csv(
        summary_path,
        MODEL_STABILITY_SELECTED_AGGREGATE_COLUMNS
        if current
        else MODEL_STABILITY_AGGREGATE_COLUMNS,
        aggregate_rows,
    )
    if current:
        _write_selected_report(report_path, aggregate_rows, seeds)
    else:
        _write_report(report_path, aggregate_rows, selected_feature_set, seeds)

    return {
        "selected_feature_set": selected_feature_set,
        "run_rows": run_rows,
        "aggregate_rows": aggregate_rows,
        "seed_runs_path": seed_runs_path,
        "summary_path": summary_path,
        "report_path": report_path,
    }


def _resplit_development(
    saved_split_frames: dict[str, pd.DataFrame], seed: int
) -> dict[str, pd.DataFrame]:
    """Vary train/validation membership without moving saved test applicants."""
    development = pd.concat(
        [saved_split_frames[name] for name in ("train", "calibration", "validation")],
        ignore_index=True,
    ).sort_values("SK_ID_CURR")
    try:
        train, selection_pool = train_test_split(
            development,
            test_size=len(saved_split_frames["validation"])
            + len(saved_split_frames["calibration"]),
            stratify=development["TARGET"].astype(int),
            random_state=seed,
        )
        calibration, validation = train_test_split(
            selection_pool,
            test_size=len(saved_split_frames["validation"]),
            stratify=selection_pool["TARGET"].astype(int),
            random_state=seed,
        )
    except ValueError as error:
        raise ModelStabilityError(
            f"Cannot stratify saved development applicants for seed {seed}: {error}"
        ) from error
    split_frames = {
        "train": train.sort_values("SK_ID_CURR").reset_index(drop=True),
        "validation": validation.sort_values("SK_ID_CURR").reset_index(drop=True),
        "calibration": calibration.sort_values("SK_ID_CURR").reset_index(drop=True),
        "test": saved_split_frames["test"],
    }
    for name in ("train", "calibration", "validation"):
        if target_class_values(
            split_frames[name]["TARGET"], error_cls=ModelStabilityError
        ) != {0, 1}:
            raise ModelStabilityError(f"{name} split must contain both target classes")
    return split_frames


def aggregate_stability_rows(
    run_rows: list[dict[str, Any]],
    created_at: str,
    *,
    winner_only: bool = False,
    completed_seeds: tuple[int, ...] | None = None,
    feature_specs: dict[str, tuple[int, int | None]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate all-surface legacy runs or conditional current selections."""
    if not run_rows:
        raise ModelStabilityError("At least one seed run row is required")
    if winner_only:
        return _aggregate_selected_rows(
            run_rows, created_at, completed_seeds, feature_specs
        )
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
        row["pr_auc_generalization_gap"] = (
            row["test_pr_auc_mean"] - row["validation_pr_auc_mean"]
        )
        row["abs_pr_auc_generalization_gap"] = abs(row["pr_auc_generalization_gap"])
        row["balanced_ev_generalization_gap"] = (
            row["test_balanced_ev_per_applicant_mean"]
            - row["validation_balanced_ev_per_applicant_mean"]
        )
        row["abs_balanced_ev_generalization_gap"] = abs(
            row["balanced_ev_generalization_gap"]
        )
        aggregate_rows.append(row)
    return aggregate_rows


def _aggregate_selected_rows(run_rows, created_at, completed_seeds, feature_specs):
    frame = pd.DataFrame(run_rows)
    seeds = set(completed_seeds or frame["seed"].unique())
    if not seeds or frame["seed"].duplicated().any() or set(frame["seed"]) != seeds:
        raise ModelStabilityError(
            "Winner-only stability requires one selected recipe per completed split"
        )
    specs = feature_specs or {
        row["feature_set"]: (row["feature_count"], row["feature_limit"])
        for row in run_rows
    }
    if not set(frame["feature_set"]) <= set(specs):
        raise ModelStabilityError("Selected stability surface was not declared")
    rows = []
    for name, (count, limit) in specs.items():
        group = frame.loc[frame["feature_set"] == name]
        support = len(group)
        row = {
            "feature_set": name,
            "feature_count": count,
            "feature_limit": limit,
            "selected": False,
            "seed_count": support,
            "completed_split_count": len(seeds),
            "metric_scope": "conditional_on_selection",
            "validation_win_count": support,
            "validation_win_rate": support / len(seeds),
            "created_at": created_at,
        }
        for metric in MEAN_STD_METRICS:
            values = group[metric].astype(float).dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        for prefix, test, validation in [
            ("pr_auc", "test_pr_auc_mean", "validation_pr_auc_mean"),
            (
                "balanced_ev",
                "test_balanced_ev_per_applicant_mean",
                "validation_balanced_ev_per_applicant_mean",
            ),
        ]:
            gap = (
                row[test] - row[validation]
                if row[test] is not None and row[validation] is not None
                else None
            )
            row[f"{prefix}_generalization_gap"] = gap
            row[f"abs_{prefix}_generalization_gap"] = (
                abs(gap) if gap is not None else None
            )
        rows.append(row)
    return rows


def _write_selected_report(path, rows, seeds):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    table = "\n".join(
        f"| {feature_set_label(row['feature_set'])} | {row['seed_count']}/{row['completed_split_count']} | "
        f"{row['validation_win_rate']:.6f} | {number(row['validation_pr_auc_mean'])} | "
        f"{number(row['validation_pr_auc_std'])} | {number(row['test_pr_auc_mean'])} |"
        for row in rows
    )
    path.write_text(
        f"""# Sensitivity to development applicant grouping

A feature is a model input. Cross-validation repeats training/selection across applicant groups; calibration is a probability adjustment and may leave raw probabilities unchanged. Average precision assesses ranking, not accuracy. Utility uses illustrative units, not dollars.

Each completed development split (`{_seed_list(seeds)}`) executes one training-only
joint CV budget and records its chosen recipe. Model and ranking seeds stay fixed;
saved historical test membership stays fixed. Calibration fitting and selection
remain disjoint. Separate fixed-recipe model-seed sensitivity measures randomness.

Selection frequency divides each surface's selected count by all {len(seeds)}
completed splits. Metric means and sample standard deviations are conditional on
that surface being selected. An absent surface has no measured metrics; zero or
one measured value has unavailable standard deviation. Different surfaces have
different selection support, so these conditional means cannot establish
across-split superiority. There is no aggregate feature-set selection or model
promotion. These development and historical comparison diagnostics do not provide
an independent nested performance estimate or undo historical public exploration.

| Input group | Selected/completed groups | Selection frequency | Conditional selection average precision | Conditional selection variation (sample SD) | Conditional historical average precision |
|---|---:|---:|---:|---:|---:|
{table}
""",
        encoding="utf-8",
    )


def select_stability_feature_set(rows: list[dict[str, Any]]) -> str:
    """Select the best feature set from aggregate stability rows."""
    selected = max(rows, key=_stability_selection_key)
    return str(selected["feature_set"])


def _normalize_seeds(seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Validate and normalize repeated-run seed values."""
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ModelStabilityError("At least one seed is required")
    if len(set(normalized)) != len(normalized):
        raise ModelStabilityError("Seeds must be unique")
    return normalized


def _stability_selection_key(
    row: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, int]:
    """Return the aggregate validation-first ranking key for stability selection."""
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
    """Write the markdown model-stability experiment report."""
    table_lines = "\n".join(
        "| {feature_display} | {feature_count} | {seed_count} | {validation_win_rate:.2f} | "
        "{validation_pr_auc_mean:.6f} | {validation_pr_auc_std:.6f} | "
        "{validation_brier_score_mean:.6f} | {validation_top_decile_lift_mean:.6f} | "
        "{test_pr_auc_mean:.6f} | {test_brier_score_mean:.6f} | "
        "{test_balanced_ev_per_applicant_mean:.2f} | {selected} |".format(
            **row, feature_display=feature_set_label(row["feature_set"])
        )
        for row in aggregate_rows
    )
    selected_row = next(
        row for row in aggregate_rows if row["feature_set"] == selected_feature_set
    )
    interpretation_text = _interpretation_text(selected_row)
    text = f"""# Experiment 006: Model Stability

## Purpose

Check how feature selection changes across development partition seeds, keeping model and ranking seeds fixed.

## Process

This development sensitivity experiment varies partition seeds `{_seed_list(seeds)}` while keeping model randomness fixed by `project.model_seed`. Ranking seeds are independently configured under `feature_selection.ranking_seeds`; mean raw-feature ranks use only the current training applicants. Each split seed stratifies the saved training, calibration, and validation applicants into disjoint roles, preserving counts and keeping saved test membership fixed. Calibrators fit the reserved calibration role; validation selects candidates. The `seed` CSV column remains a compatibility alias for `split_seed`; model and ranking seeds are explicit. Use the separate nested assessment command to assess the whole selection procedure.

## Selection Rule

The selected setup is chosen with a validation-only aggregate rule: mean validation average precision first, then validation win rate, mean top-decile lift, mean highest-score case capture (separate from middle-band review), mean ROC-AUC, lower mean Brier score, lower validation average precision variability, and fewer features. Historical comparison is reported after selection using fixed historical comparison membership. It is not used by the selection rule, but its prior use prevents an untouched final-test claim.

## Results

| Input group | Inputs | Grouping seeds | Selection win rate | Selection average precision | Selection variation (SD) | Selection Brier | Selection lift | Historical average precision | Historical Brier | Historical utility (units/applicant) | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table_lines}

## Selected Setup

Selected input group: {feature_set_label(selected_feature_set)}, with {selected_row["feature_count"]} inputs.

## Generalization Check

For the selected setup, mean test average precision minus mean validation average precision is {selected_row["pr_auc_generalization_gap"]:.6f}, and mean historical scenario utility minus mean model-selection utility (units/applicant) is {selected_row["balanced_ev_generalization_gap"]:.2f}. These are historical comparison diagnostics. Each partition averages ranking-model repeats within its own training role and fits calibrators on a disjoint calibration role. Validation still selects candidates, so validation metrics are selection evidence rather than an independent performance estimate. This sensitivity run does not undo historical dataset exploration or provide nested assessment.

## Interpretation

{interpretation_text}

## Notes

Scenario utility uses raw model scores, matching the existing threshold policy. Probability-quality metrics use the selected calibration method. This experiment does not add source tables, demographic/protected-status-like fields, or a new decision policy.
"""
    path.write_text(text, encoding="utf-8")


def _interpretation_text(selected_row: dict[str, Any]) -> str:
    """Build interpretation text for the selected stability result."""
    selected_feature_set = str(selected_row["feature_set"])
    if selected_feature_set == "full":
        return (
            "The repeated-seed result does not support promoting the smaller `top_100` surface yet. "
            "The full feature set has the strongest mean validation average precision under the validation-only "
            "aggregate rule, so it remains the better active model candidate until a smaller setup "
            "wins a stability pass or the project explicitly prioritizes simplicity over validation lift."
        )
    return (
        f"`{selected_feature_set}` remains the selected setup after repeated-seed validation. "
        "That supports considering the smaller input group within this development rule because the improvement was not limited "
        "to a single grouping seed. Historical exploration still prevents independent confirmation."
    )


def _std(values: pd.Series) -> float:
    """Return population standard deviation with NaN normalized to zero."""
    result = float(values.std(ddof=0))
    if np.isnan(result):
        return 0.0
    return result


def _seed_list(seeds: tuple[int, ...]) -> str:
    """Format seed values for markdown report text."""
    return ", ".join(str(seed) for seed in seeds)


def main() -> None:
    """Run the model-stability experiment CLI."""
    parser = argparse.ArgumentParser(
        description="Check sensitivity to development applicant groupings, keeping model randomness fixed.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "--seeds",
        default=format_int_csv(DEFAULT_STABILITY_SEEDS),
        help="Comma-separated partition seeds; model/ranking seeds remain fixed by config.",
    )
    parser.add_argument(
        "--feature-limits",
        default=format_int_csv(DEFAULT_FEATURE_LIMITS),
        help="Comma-separated top-N feature limits to compare.",
    )
    parser.add_argument(
        "--skip-full", action="store_true", help="Do not include the full feature set."
    )
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
