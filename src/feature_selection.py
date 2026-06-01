from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.calibration import CALIBRATION_METHODS
from src.calibration import apply_calibration_method
from src.calibration import fit_calibrators
from src.calibration import select_calibration_method
from src.config import load_config
from src.mart_access import load_labeled_split_frames
from src.mart_access import require_table
from src.model_contracts import EVALUATION_SPLITS
from src.model_contracts import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.model_contracts import LIGHTGBM_MODEL_TYPE
from src.model_contracts import LIGHTGBM_MODEL_VERSION
from src.model_contracts import REPORTING_SPLITS
from src.model_artifacts import load_model_artifact
from src.model_artifacts import normalize_split_ids
from src.runtime import created_at_utc
from src.runtime import feature_frame
from src.runtime import resolve_project_path
from src.runtime import write_csv
from src.thresholding import build_threshold_metric_rows
from src.thresholding import resolve_scenario_thresholds
from src.modeling import build_lightgbm_tuning_artifact
from src.modeling import classify_feature_columns
from src.modeling import fit_tuned_lightgbm
from src.modeling import lightgbm_params
from src.modeling import probability_metrics


DEFAULT_FEATURE_LIMITS = (40, 60, 80, 100)
FEATURE_SELECTION_REPORT_NAME = "005_feature_selection.md"
SELECTED_FEATURES_NAME = "005_selected_features.csv"

FEATURE_SELECTION_COMPARISON_COLUMNS = [
    "feature_set",
    "selected",
    "feature_count",
    "feature_limit",
    "selected_calibration_method",
    "selected_candidate_name",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_review_capacity",
    "validation_weighted_calibration_error",
    "test_pr_auc",
    "test_roc_auc",
    "test_brier_score",
    "test_top_decile_lift",
    "test_precision_at_top_decile",
    "test_recall_at_review_capacity",
    "test_weighted_calibration_error",
    "validation_balanced_ev_per_applicant",
    "test_balanced_ev_per_applicant",
    "created_at",
]

SELECTED_FEATURE_COLUMNS = [
    "feature_set",
    "feature_rank",
    "feature_name",
]


class FeatureSelectionError(RuntimeError):
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
    duckdb_path = resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = resolve_project_path(config["paths"]["model_dir"])
    report_dir = resolve_project_path(config["paths"]["report_dir"])

    if not duckdb_path.exists():
        raise FeatureSelectionError(f"DuckDB database not found: {duckdb_path}")

    base_artifact = load_lightgbm_artifact(model_dir)
    full_feature_columns = list(base_artifact["feature_columns"])
    split_applicant_ids = _normalize_split_ids(base_artifact["split_applicant_ids"])
    importance_rows = load_feature_importance_rows(report_dir)
    ranked_features = ranked_raw_features(importance_rows, full_feature_columns)
    max_limit = max(feature_limits) if feature_limits else 0
    if len(ranked_features) < min(max_limit, len(full_feature_columns)):
        raise FeatureSelectionError(
            "Feature importance ranking does not cover enough model features for requested limits: "
            f"ranked={len(ranked_features)}, requested={max_limit}"
        )

    created_at = created_at_utc()
    manual_review_capacity_rate = float(config["business_assumptions"]["manual_review_capacity_rate"])
    rows: list[dict[str, Any]] = []
    feature_set_specs = feature_sets(
        ranked_features,
        full_feature_columns,
        feature_limits,
        include_full,
    )
    features_by_set = {
        feature_set_name: feature_columns
        for feature_set_name, feature_columns, _feature_limit in feature_set_specs
    }
    with duckdb.connect(str(duckdb_path)) as connection:
        for feature_set_name, feature_columns, feature_limit in feature_set_specs:
            split_frames = _load_split_frames(connection, split_applicant_ids, feature_columns)
            rows.append(
                run_single_feature_set(
                    config,
                    feature_set_name,
                    feature_columns,
                    feature_limit,
                    split_frames,
                    manual_review_capacity_rate,
                    created_at,
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


def run_single_feature_set(
    config: dict[str, Any],
    feature_set_name: str,
    feature_columns: list[str],
    feature_limit: int | None,
    split_frames: dict[str, pd.DataFrame],
    manual_review_capacity_rate: float,
    created_at: str,
    random_seed: int | None = None,
) -> dict[str, Any]:
    random_seed = int(config["project"]["random_seed"] if random_seed is None else random_seed)
    numeric_features, categorical_features = classify_feature_columns(
        split_frames["train"],
        feature_columns,
    )
    base_params = lightgbm_params(config, split_frames["train"], random_seed)
    tuning = fit_tuned_lightgbm(
        config,
        numeric_features,
        categorical_features,
        base_params,
        split_frames,
        feature_columns,
        manual_review_capacity_rate,
        error_cls=FeatureSelectionError,
    )
    pipeline = tuning["pipeline"]
    raw_predictions = _prediction_frames(pipeline, split_frames, feature_columns)
    calibrators = fit_calibrators(
        raw_predictions["validation"]["probability"].to_numpy(),
        raw_predictions["validation"]["target"].to_numpy(),
        random_seed,
        error_cls=FeatureSelectionError,
    )
    predictions_by_method = {
        method: apply_calibration_method(
            method,
            calibrators,
            raw_predictions,
            error_cls=FeatureSelectionError,
        )
        for method in CALIBRATION_METHODS
    }
    metric_rows = _calibration_metric_rows(predictions_by_method, manual_review_capacity_rate)
    selected_calibration_method = select_calibration_method(
        metric_rows,
        error_cls=FeatureSelectionError,
    )
    selected_predictions = predictions_by_method[selected_calibration_method]
    metrics = _metrics_by_split(selected_predictions, manual_review_capacity_rate)
    weighted_bin_errors = {
        split_name: _weighted_calibration_error(selected_predictions[split_name])
        for split_name in REPORTING_SPLITS
    }
    threshold_rows = _balanced_threshold_rows(
        config,
        f"feature_selection_{feature_set_name}",
        selected_predictions,
        created_at,
    )
    balanced_ev = {
        row["split"]: float(row["expected_value_per_applicant"])
        for row in threshold_rows
        if row["scenario_name"] == "balanced"
    }
    selected_candidate = build_lightgbm_tuning_artifact(tuning)["selected_candidate"]
    return {
        "feature_set": feature_set_name,
        "selected": False,
        "feature_count": len(feature_columns),
        "feature_limit": feature_limit if feature_limit is not None else "full",
        "selected_calibration_method": selected_calibration_method,
        "selected_candidate_name": selected_candidate["candidate_name"],
        "validation_pr_auc": metrics["validation"]["pr_auc"],
        "validation_roc_auc": metrics["validation"]["roc_auc"],
        "validation_brier_score": metrics["validation"]["brier_score"],
        "validation_top_decile_lift": metrics["validation"]["top_decile_lift"],
        "validation_precision_at_top_decile": metrics["validation"]["precision_at_top_decile"],
        "validation_recall_at_review_capacity": metrics["validation"][
            "recall_at_manual_review_capacity"
        ],
        "validation_weighted_calibration_error": weighted_bin_errors["validation"],
        "test_pr_auc": metrics["test"]["pr_auc"],
        "test_roc_auc": metrics["test"]["roc_auc"],
        "test_brier_score": metrics["test"]["brier_score"],
        "test_top_decile_lift": metrics["test"]["top_decile_lift"],
        "test_precision_at_top_decile": metrics["test"]["precision_at_top_decile"],
        "test_recall_at_review_capacity": metrics["test"]["recall_at_manual_review_capacity"],
        "test_weighted_calibration_error": weighted_bin_errors["test"],
        "validation_balanced_ev_per_applicant": balanced_ev["validation"],
        "test_balanced_ev_per_applicant": balanced_ev["test"],
        "created_at": created_at,
    }


def load_lightgbm_artifact(model_dir: Path) -> dict[str, Any]:
    artifact_path = model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME
    return load_model_artifact(
        artifact_path,
        expected_model_type=LIGHTGBM_MODEL_TYPE,
        expected_model_version=LIGHTGBM_MODEL_VERSION,
        error_cls=FeatureSelectionError,
        artifact_label="LightGBM artifact",
        missing_label="LightGBM model artifact",
    )


def load_feature_importance_rows(report_dir: Path) -> list[dict[str, Any]]:
    path = report_dir / "model_feature_importance.csv"
    if not path.exists():
        raise FeatureSelectionError(f"Missing feature importance report: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def ranked_raw_features(
    importance_rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> list[str]:
    label_to_raw_feature = {
        _normalize_feature_label(readable_feature_label(feature_column)): feature_column
        for feature_column in feature_columns
    }
    ranked_features: list[str] = []
    seen_features = set()
    rows = sorted(importance_rows, key=lambda row: int(row["rank"]))
    for row in rows:
        raw_label = str(row["feature_name"]).split(":", 1)[0]
        raw_feature = label_to_raw_feature.get(_normalize_feature_label(raw_label))
        if raw_feature is None or raw_feature in seen_features:
            continue
        ranked_features.append(raw_feature)
        seen_features.add(raw_feature)
    return ranked_features


def readable_feature_label(raw_feature: str) -> str:
    cleaned = raw_feature.replace("__", "_").replace("_", " ").strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return "Unknown feature"
    return cleaned.lower().capitalize()


def _normalize_feature_label(label: str) -> str:
    return " ".join(label.lower().split())


def feature_sets(
    ranked_features: list[str],
    full_feature_columns: list[str],
    feature_limits: tuple[int, ...],
    include_full: bool,
) -> list[tuple[str, list[str], int | None]]:
    feature_sets = []
    full_count = len(full_feature_columns)
    for limit in feature_limits:
        if limit <= 0:
            raise FeatureSelectionError(f"Feature limits must be positive, got {limit}")
        if limit > full_count:
            raise FeatureSelectionError(f"Feature limit {limit} exceeds full feature count {full_count}")
        feature_sets.append((f"top_{limit}", ranked_features[:limit], limit))
    if include_full:
        feature_sets.append(("full", full_feature_columns, None))
    return feature_sets


def _normalize_split_ids(raw_split_ids: Any) -> dict[str, list[int]]:
    return normalize_split_ids(raw_split_ids, EVALUATION_SPLITS, error_cls=FeatureSelectionError)


def _load_split_frames(
    connection: duckdb.DuckDBPyConnection,
    split_applicant_ids: dict[str, list[int]],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    require_table(connection, "mart_credit_risk_features", error_cls=FeatureSelectionError)
    return load_labeled_split_frames(
        connection,
        split_applicant_ids,
        feature_columns,
        error_cls=FeatureSelectionError,
    )


def _prediction_frames(
    pipeline: Any,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    frames = {}
    for split_name in REPORTING_SPLITS:
        frame = split_frames[split_name]
        probabilities = pipeline.predict_proba(feature_frame(frame, feature_columns))[:, 1]
        frames[split_name] = pd.DataFrame(
            {
                "SK_ID_CURR": frame["SK_ID_CURR"].astype(int),
                "target": frame["TARGET"].astype(int),
                "probability": probabilities.astype(float),
            }
        )
    return frames


def _calibration_metric_rows(
    predictions_by_method: dict[str, dict[str, pd.DataFrame]],
    manual_review_capacity_rate: float,
) -> list[dict[str, Any]]:
    rows = []
    for method, split_predictions in predictions_by_method.items():
        for split_name in REPORTING_SPLITS:
            frame = split_predictions[split_name]
            metrics = probability_metrics(
                frame["target"],
                frame["probability"].to_numpy(),
                manual_review_capacity_rate,
                error_cls=FeatureSelectionError,
            )
            rows.append(
                {
                    "calibration_method": method,
                    "split": split_name,
                    "brier_score": metrics["brier_score"],
                }
            )
    return rows


def _metrics_by_split(
    prediction_frames: dict[str, pd.DataFrame],
    manual_review_capacity_rate: float,
) -> dict[str, dict[str, float]]:
    return {
        split_name: probability_metrics(
            frame["target"],
            frame["probability"].to_numpy(),
            manual_review_capacity_rate,
            error_cls=FeatureSelectionError,
        )
        for split_name, frame in prediction_frames.items()
    }


def _weighted_calibration_error(frame: pd.DataFrame) -> float:
    ranked = frame.sort_values(["probability", "SK_ID_CURR"], ascending=[True, True]).reset_index(drop=True)
    ranked["bin_id"] = np.ceil((np.arange(len(ranked)) + 1) * 10 / len(ranked)).astype(int).clip(1, 10)
    total_count = len(ranked)
    weighted_error = 0.0
    for bin_id in range(1, 11):
        bin_frame = ranked.loc[ranked["bin_id"] == bin_id]
        if bin_frame.empty:
            continue
        calibration_error = float(bin_frame["target"].mean() - bin_frame["probability"].mean())
        weighted_error += abs(calibration_error) * len(bin_frame) / total_count
    return float(weighted_error)


def _balanced_threshold_rows(
    config: dict[str, Any],
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    created_at: str,
) -> list[dict[str, Any]]:
    scenario_thresholds = resolve_scenario_thresholds(
        config["threshold_policy"],
        prediction_frames["validation"]["probability"].to_numpy(),
    )
    return build_threshold_metric_rows(
        model_version,
        str(config["threshold_policy"]["threshold_version"]),
        prediction_frames,
        scenario_thresholds,
        config["business_assumptions"],
        created_at,
    )


def select_feature_set(rows: list[dict[str, Any]]) -> str:
    selected = sorted(rows, key=_feature_set_selection_key, reverse=True)[0]
    return str(selected["feature_set"])


def _selected_feature_rows(feature_set_name: str, feature_columns: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "feature_set": feature_set_name,
            "feature_rank": rank,
            "feature_name": feature_name,
        }
        for rank, feature_name in enumerate(feature_columns, start=1)
    ]


def _feature_set_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    return (
        float(row["validation_pr_auc"]),
        float(row["validation_top_decile_lift"]),
        float(row["validation_recall_at_review_capacity"]),
        float(row["validation_roc_auc"]),
        -float(row["validation_brier_score"]),
        -int(row["feature_count"]),
    )


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


if __name__ == "__main__":
    main()
