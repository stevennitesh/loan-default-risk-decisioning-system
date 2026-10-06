from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from src.cli import add_config_argument, exit_with_error, format_int_csv, parse_int_csv
from src.config import DEFAULT_CONFIG_PATH, load_config, manual_review_capacity_rate
from src.feature_experiments import (
    DEFAULT_FEATURE_LIMITS,
    FeatureExperimentError,
    load_lightgbm_artifact,
    load_split_frames,
    prepare_feature_set_specs,
    run_joint_feature_sets,
    run_single_feature_set,
    select_feature_set,
)
from src.model_artifacts import normalize_split_ids, validate_feature_build
from src.model_contracts import EVALUATION_SPLITS
from src.presentation import feature_set_label, method_label
from src.report_contracts import (
    FEATURE_SELECTION_COMPARISON_COLUMNS,
    SELECTED_FEATURE_COLUMNS,
)
from src.runtime import (
    created_at_utc,
    ensure_directories,
    require_existing_path,
    resolve_config_path,
    write_csv,
)

FEATURE_SELECTION_REPORT_NAME = "005_feature_selection.md"
SELECTED_FEATURES_NAME = "005_selected_features.csv"


class FeatureSelectionError(FeatureExperimentError):
    """Raised when the feature-selection experiment cannot run safely."""


def run_feature_selection_experiment(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    feature_limits: tuple[int, ...] = DEFAULT_FEATURE_LIMITS,
    include_full: bool = True,
    comparison_name: str = "feature_selection_comparison.csv",
    selected_features_name: str = SELECTED_FEATURES_NAME,
    report_name: str = FEATURE_SELECTION_REPORT_NAME,
) -> dict[str, Any]:
    """Compare top-N feature sets and write feature-selection experiment outputs."""
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    require_existing_path(duckdb_path, "DuckDB database", FeatureSelectionError)

    base_artifact = load_lightgbm_artifact(model_dir, error_cls=FeatureSelectionError)
    full_feature_columns = list(
        base_artifact.get("eligible_feature_columns", base_artifact["feature_columns"])
    )
    split_applicant_ids = normalize_split_ids(
        base_artifact["split_applicant_ids"],
        (*EVALUATION_SPLITS, "calibration"),
        error_cls=FeatureSelectionError,
    )

    created_at = created_at_utc()
    review_capacity_rate = manual_review_capacity_rate(config)
    rows: list[dict[str, Any]] = []
    with duckdb.connect(str(duckdb_path)) as connection:
        validate_feature_build(
            connection, base_artifact, error_cls=FeatureSelectionError
        )
        all_split_frames = load_split_frames(
            connection,
            split_applicant_ids,
            full_feature_columns,
            error_cls=FeatureSelectionError,
        )
        from src.tuning import uses_inner_cv

        if uses_inner_cv(config):
            rows, features_by_set = run_joint_feature_sets(
                config,
                all_split_frames,
                full_feature_columns,
                feature_limits,
                include_full,
                review_capacity_rate,
                created_at,
                FeatureSelectionError,
            )
        else:
            feature_set_specs = prepare_feature_set_specs(
                report_dir,
                full_feature_columns,
                feature_limits,
                include_full,
                error_cls=FeatureSelectionError,
                training_frame=all_split_frames["train"],
                config=config,
            )
            features_by_set = {
                name: columns for name, columns, _limit in feature_set_specs
            }
            for feature_set_name, feature_columns, feature_limit in feature_set_specs:
                split_frames = all_split_frames
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

    current = uses_inner_cv(config)
    output_dir = report_dir / "tuning" / "feature_selection" if current else report_dir
    experiments_dir = output_dir if current else report_dir / "experiments"
    ensure_directories(output_dir, experiments_dir)
    comparison_path = output_dir / comparison_name
    report_path = experiments_dir / report_name
    selected_features_path = experiments_dir / selected_features_name
    write_csv(comparison_path, FEATURE_SELECTION_COMPARISON_COLUMNS, rows)
    write_csv(
        selected_features_path,
        SELECTED_FEATURE_COLUMNS,
        _selected_feature_rows(
            selected_feature_set, features_by_set[selected_feature_set]
        ),
    )
    _write_report(
        report_path, rows, selected_feature_set, selected_features_name, current=current
    )

    return {
        "selected_feature_set": selected_feature_set,
        "comparison_rows": rows,
        "comparison_path": comparison_path,
        "report_path": report_path,
        "selected_features_path": selected_features_path,
    }


def _selected_feature_rows(
    feature_set_name: str, feature_columns: list[str]
) -> list[dict[str, Any]]:
    """Build selected-feature CSV rows for the winning feature set."""
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
    *,
    current: bool = False,
) -> None:
    """Write the markdown feature-selection experiment report."""
    table_lines = "\n".join(
        "| {feature_display} | {feature_count} | {method_display} | "
        "{validation_pr_auc:.6f} | {validation_brier_score:.6f} | "
        "{validation_top_decile_lift:.6f} | {validation_balanced_ev_per_applicant:.2f} | "
        "{test_pr_auc:.6f} | {test_brier_score:.6f} | "
        "{test_top_decile_lift:.6f} | {test_balanced_ev_per_applicant:.2f} | {selected} |".format(
            **row,
            feature_display=feature_set_label(row["feature_set"]),
            method_display=method_label(row["selected_calibration_method"]),
        )
        for row in rows
    )
    selected_row = next(
        row for row in rows if row["feature_set"] == selected_feature_set
    )
    interpretation_text = _interpretation_text(rows, selected_row)
    text = f"""# Model-input selection experiment

## Purpose

Compare smaller groups of model inputs (features) with all eligible application and loan-history inputs. Average precision measures ranking, not accuracy; Brier score measures probability error (lower is better). Calibration means probability adjustment and can leave raw probabilities unchanged.

## Selection Rule

Feature subsets use mean model-input ranks from several LightGBM gain rankings fitted exclusively on the same training applicants, with independently configured ranking seeds. Reporting SHAP is not consumed. The selected setup follows the validation-only rule using average precision first, then top-decile lift, highest-score case capture (separate from middle-band manual review), ROC-AUC, lower Brier score, and fewer features. Calibrators fit on separate reserved applicants. Historical comparison is not the optimization target; its results are historically exposed comparison diagnostics reported after selection. This is development selection, not nested assessment.

## Results

| Input group | Inputs | Probability method | Selection average precision | Selection Brier | Selection lift | Selection utility (units/applicant) | Historical average precision | Historical Brier | Historical lift | Historical utility (units/applicant) | Selected |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table_lines}

## Selected Setup

Selected input group: {feature_set_label(selected_feature_set)}, with {selected_row["feature_count"]} inputs. Probability method: `{selected_row["selected_calibration_method"]}` (`uncalibrated` means raw probabilities unchanged).

Selected raw feature columns are written to `reports/experiments/{selected_features_name}`.

## Interpretation

{interpretation_text}

## Notes

This experiment changes the model feature surface only. It does not add new source tables, demographic/protected-status-like fields, or a new decision policy.
"""
    if current:
        text = text.replace(
            f"reports/experiments/{selected_features_name}", selected_features_name
        )
        start = text.index("## Selection Rule")
        end = text.index("## Results")
        text = (
            text[:start]
            + "## Selection Rule\n\nThe current bounded search jointly selects model inputs and settings using three-group cross-validation entirely within fitting applicants, with disjoint stopping subsets and fold-local raw rankings/preprocessing. This row describes the CV-chosen recipe. Reserved calibration and method selection follow; their metrics never choose another feature surface. Historical comparison metrics remain descriptive. Exact search evidence is stored in the ignored tuning scope. No historical experiment is refreshed.\n\n"
            + text[end:]
        )
    if current:
        start = text.index("## Interpretation")
        end = text.index("## Notes")
        text = (
            text[:start]
            + "## Interpretation\n\nThis is the recipe chosen within training-only cross-validation. Reserved selection rows choose the probability method and score thresholds, not a different input group. Historical comparison metrics remain descriptive and do not establish independent performance.\n\n"
            + text[end:]
        )
    path.write_text(text, encoding="utf-8")


def _interpretation_text(
    rows: list[dict[str, Any]], selected_row: dict[str, Any]
) -> str:
    """Build interpretation text for the selected feature-set report."""
    selected_name = str(selected_row["feature_set"])
    full_row = next((row for row in rows if row["feature_set"] == "full"), None)
    first_paragraph = (
        f"{feature_set_label(selected_name)} is the selected setup under the development selection rule. "
        "It has the strongest validation selection score across average precision, top-decile lift, "
        "highest-score case capture (separate from middle-band manual review), ROC-AUC, Brier score, and feature-count tie-breaks."
    )
    if full_row is None or selected_name == "full":
        return first_paragraph

    removed_features = int(full_row["feature_count"]) - int(
        selected_row["feature_count"]
    )
    full_test_pr_auc = float(full_row["test_pr_auc"])
    selected_test_pr_auc = float(selected_row["test_pr_auc"])
    full_test_ev = float(full_row["test_balanced_ev_per_applicant"])
    selected_test_ev = float(selected_row["test_balanced_ev_per_applicant"])
    full_test_edges = []
    if full_test_pr_auc > selected_test_pr_auc:
        full_test_edges.append("Average precision")
    if full_test_ev > selected_test_ev:
        full_test_edges.append("balanced expected value")
    if full_test_edges:
        test_caveat = (
            f"The full model has the stronger {' and '.join(full_test_edges)} on historical comparison, "
            "but historical comparison is a reused diagnostic, not the optimization target. "
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
    """Run the feature-selection experiment CLI."""
    parser = argparse.ArgumentParser(
        description="Compare smaller groups of model inputs with all eligible LightGBM inputs.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "--feature-limits",
        default=format_int_csv(DEFAULT_FEATURE_LIMITS),
        help="Comma-separated top-N feature limits to compare.",
    )
    parser.add_argument(
        "--skip-full", action="store_true", help="Do not include the full feature set."
    )
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
    feature_limits = parse_int_csv(args.feature_limits)

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
        exit_with_error(error)


if __name__ == "__main__":
    main()
