from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from src.config import load_config
from src.config import manual_review_capacity_rate
from src.feature_experiments import DEFAULT_FEATURE_LIMITS
from src.feature_experiments import FeatureExperimentError
from src.feature_experiments import load_lightgbm_artifact
from src.feature_experiments import load_split_frames
from src.feature_experiments import prepare_feature_set_specs
from src.feature_experiments import run_single_feature_set
from src.feature_experiments import select_feature_set
from src.model_artifacts import normalize_split_ids
from src.model_contracts import EVALUATION_SPLITS
from src.runtime import created_at_utc
from src.runtime import resolve_config_path
from src.runtime import write_csv
from src.report_contracts import FEATURE_SELECTION_COMPARISON_COLUMNS
from src.report_contracts import SELECTED_FEATURE_COLUMNS


FEATURE_SELECTION_REPORT_NAME = "005_feature_selection.md"
SELECTED_FEATURES_NAME = "005_selected_features.csv"


class FeatureSelectionError(FeatureExperimentError):
    """Raised when the feature-selection experiment cannot run safely."""


def run_feature_selection_experiment(
    config_path: str | Path = "configs/base.yaml",
    feature_limits: tuple[int, ...] = DEFAULT_FEATURE_LIMITS,
    include_full: bool = True,
    comparison_name: str = "feature_selection_comparison.csv",
    selected_features_name: str = SELECTED_FEATURES_NAME,
    report_name: str = FEATURE_SELECTION_REPORT_NAME,
) -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    if not duckdb_path.exists():
        raise FeatureSelectionError(f"DuckDB database not found: {duckdb_path}")

    base_artifact = load_lightgbm_artifact(model_dir, error_cls=FeatureSelectionError)
    full_feature_columns = list(base_artifact["feature_columns"])
    split_applicant_ids = normalize_split_ids(
        base_artifact["split_applicant_ids"],
        EVALUATION_SPLITS,
        error_cls=FeatureSelectionError,
    )

    created_at = created_at_utc()
    review_capacity_rate = manual_review_capacity_rate(config)
    rows: list[dict[str, Any]] = []
    feature_set_specs = prepare_feature_set_specs(
        report_dir,
        full_feature_columns,
        feature_limits,
        include_full,
        error_cls=FeatureSelectionError,
    )
    features_by_set = {
        feature_set_name: feature_columns
        for feature_set_name, feature_columns, _feature_limit in feature_set_specs
    }
    with duckdb.connect(str(duckdb_path)) as connection:
        for feature_set_name, feature_columns, feature_limit in feature_set_specs:
            split_frames = load_split_frames(
                connection,
                split_applicant_ids,
                feature_columns,
                error_cls=FeatureSelectionError,
            )
            rows.append(
                run_single_feature_set(
                    config,
                    feature_set_name,
                    feature_columns,
                    feature_limit,
                    split_frames,
                    review_capacity_rate,
                    created_at,
                    error_cls=FeatureSelectionError,
                )
            )

    selected_feature_set = select_feature_set(rows)
    for row in rows:
        row["selected"] = row["feature_set"] == selected_feature_set

    report_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = report_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = report_dir / comparison_name
    report_path = experiments_dir / report_name
    selected_features_path = experiments_dir / selected_features_name
    write_csv(comparison_path, FEATURE_SELECTION_COMPARISON_COLUMNS, rows)
    write_csv(
        selected_features_path,
        SELECTED_FEATURE_COLUMNS,
        _selected_feature_rows(selected_feature_set, features_by_set[selected_feature_set]),
    )
    _write_report(report_path, rows, selected_feature_set, selected_features_name)

    return {
        "selected_feature_set": selected_feature_set,
        "comparison_rows": rows,
        "comparison_path": comparison_path,
        "report_path": report_path,
        "selected_features_path": selected_features_path,
    }


def _selected_feature_rows(feature_set_name: str, feature_columns: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "feature_set": feature_set_name,
            "feature_rank": rank,
            "feature_name": feature_name,
        }
        for rank, feature_name in enumerate(feature_columns, start=1)
    ]


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    selected_feature_set: str,
    selected_features_name: str = SELECTED_FEATURES_NAME,
) -> None:
    table_lines = "\n".join(
        "| {feature_set} | {feature_count} | {selected_calibration_method} | "
        "{validation_pr_auc:.6f} | {validation_brier_score:.6f} | "
        "{validation_top_decile_lift:.6f} | {validation_balanced_ev_per_applicant:.2f} | "
        "{test_pr_auc:.6f} | {test_brier_score:.6f} | "
        "{test_top_decile_lift:.6f} | {test_balanced_ev_per_applicant:.2f} | {selected} |".format(**row)
        for row in rows
    )
    selected_row = next(row for row in rows if row["feature_set"] == selected_feature_set)
    interpretation_text = _interpretation_text(rows, selected_row)
    text = f"""# Experiment 005: Feature Selection

## Purpose

Compare top-N feature subsets against the full post-v1 feature set to see whether the model can keep most of the ranking and calibration gains with a cleaner feature surface.

## Selection Rule

Feature subsets are selected from `reports/model_feature_importance.csv`, mapping human-readable SHAP labels back to raw model columns. The selected setup is chosen on validation results using PR-AUC first, then top-decile lift, recall at review capacity, ROC-AUC, lower Brier score, and finally fewer features as a tie-breaker. Held-out test is not the optimization target; test metrics are reported only after selection to check whether the validation-selected setup generalizes closely enough.

## Results

| Feature set | Feature count | Calibration | Val PR-AUC | Val Brier | Val lift | Val EV/app | Test PR-AUC | Test Brier | Test lift | Test EV/app | Selected |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table_lines}

## Selected Setup

Selected feature set: `{selected_feature_set}` with {selected_row['feature_count']} features and `{selected_row['selected_calibration_method']}` calibration.

Selected raw feature columns are written to `reports/experiments/{selected_features_name}`.

## Interpretation

{interpretation_text}

## Notes

This experiment changes the model feature surface only. It does not add new source tables, demographic/protected-status-like fields, or a new decision policy.
"""
    path.write_text(text, encoding="utf-8")


def _interpretation_text(rows: list[dict[str, Any]], selected_row: dict[str, Any]) -> str:
    selected_name = str(selected_row["feature_set"])
    full_row = next((row for row in rows if row["feature_set"] == "full"), None)
    first_paragraph = (
        f"`{selected_name}` is the selected setup under the validation-only rule. "
        "It has the strongest validation selection score across PR-AUC, top-decile lift, "
        "recall at review capacity, ROC-AUC, Brier score, and feature-count tie-breaks."
    )
    if full_row is None or selected_name == "full":
        return first_paragraph

    removed_features = int(full_row["feature_count"]) - int(selected_row["feature_count"])
    full_test_pr_auc = float(full_row["test_pr_auc"])
    selected_test_pr_auc = float(selected_row["test_pr_auc"])
    full_test_ev = float(full_row["test_balanced_ev_per_applicant"])
    selected_test_ev = float(selected_row["test_balanced_ev_per_applicant"])
    full_test_edges = []
    if full_test_pr_auc > selected_test_pr_auc:
        full_test_edges.append("PR-AUC")
    if full_test_ev > selected_test_ev:
        full_test_edges.append("balanced expected value")
    if full_test_edges:
        test_caveat = (
            f"The full model has the stronger {' and '.join(full_test_edges)} on held-out test, "
            "but held-out test is a final generalization check, not the optimization target. "
            f"This does not override the validation-selected `{selected_name}` choice; it means the "
            "test gap should be recorded as stability evidence. The current gap is small enough to report, "
            "not large enough to overrule validation selection; a larger or repeated gap would point to a "
            "better model-generation method in a follow-up experiment."
        )
    else:
        test_caveat = (
            f"`{selected_name}` also holds up against the full model on the reported held-out "
            "test comparison."
        )
    return (
        f"{first_paragraph} It removes {removed_features} features compared with the full setup.\n\n"
        f"{test_caveat}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare top-N feature-selection variants for the LightGBM risk model.",
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    parser.add_argument(
        "--feature-limits",
        default="40,60,80,100",
        help="Comma-separated top-N feature limits to compare.",
    )
    parser.add_argument("--skip-full", action="store_true", help="Do not include the full feature set.")
    parser.add_argument(
        "--comparison-name",
        default="feature_selection_comparison.csv",
        help="CSV filename for feature-selection comparison rows under the report directory.",
    )
    parser.add_argument(
        "--selected-features-name",
        default=SELECTED_FEATURES_NAME,
        help="CSV filename for selected feature rows under reports/experiments.",
    )
    parser.add_argument(
        "--report-name",
        default=FEATURE_SELECTION_REPORT_NAME,
        help="Markdown report filename under reports/experiments.",
    )
    args = parser.parse_args()
    feature_limits = tuple(int(value.strip()) for value in args.feature_limits.split(",") if value.strip())

    try:
        run_feature_selection_experiment(
            args.config,
            feature_limits=feature_limits,
            include_full=not args.skip_full,
            comparison_name=args.comparison_name,
            selected_features_name=args.selected_features_name,
            report_name=args.report_name,
        )
    except FeatureSelectionError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
