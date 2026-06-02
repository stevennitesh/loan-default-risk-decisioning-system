from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.calibration import (
    CALIBRATION_METHODS,
    apply_calibration_method,
    fit_calibrators,
    select_calibration_method,
)
from src.config import (
    business_assumptions,
    project_random_seed,
    threshold_policy,
    threshold_version,
)
from src.feature_labels import readable_feature_label
from src.mart_access import load_labeled_split_frames
from src.metrics import probability_metrics, with_probability_rank_bin
from src.model_artifacts import load_model_artifact
from src.model_contracts import (
    LIGHTGBM_MODEL_ARTIFACT_NAME,
    LIGHTGBM_MODEL_TYPE,
    LIGHTGBM_MODEL_VERSION,
    REPORTING_SPLITS,
)
from src.modeling import (
    build_lightgbm_tuning_artifact,
    classify_feature_columns,
    fit_tuned_lightgbm,
    lightgbm_params,
    predict_probabilities,
    prediction_frame,
)
from src.runtime import read_csv
from src.thresholding import (
    BALANCED_SCENARIO,
    build_threshold_metric_rows,
    resolve_scenario_thresholds,
)

DEFAULT_FEATURE_LIMITS = (40, 60, 80, 100)


class FeatureExperimentError(RuntimeError):
    """Raised when a shared feature experiment cannot run safely."""


def run_single_feature_set(
    config: dict[str, Any],
    feature_set_name: str,
    feature_columns: list[str],
    feature_limit: int | None,
    split_frames: dict[str, pd.DataFrame],
    manual_review_capacity_rate: float,
    created_at: str,
    random_seed: int | None = None,
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, Any]:
    random_seed = (
        project_random_seed(config) if random_seed is None else int(random_seed)
    )
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
        error_cls=error_cls,
    )
    pipeline = tuning["pipeline"]
    raw_predictions = prediction_frames(
        pipeline,
        split_frames,
        feature_columns,
        feature_set_name,
        error_cls,
    )
    calibrators = fit_calibrators(
        raw_predictions["validation"]["probability"].to_numpy(),
        raw_predictions["validation"]["target"].to_numpy(),
        random_seed,
        error_cls=error_cls,
    )
    predictions_by_method = {
        method: apply_calibration_method(
            method,
            calibrators,
            raw_predictions,
            error_cls=error_cls,
        )
        for method in CALIBRATION_METHODS
    }
    metric_rows = calibration_metric_rows(
        predictions_by_method, manual_review_capacity_rate, error_cls
    )
    selected_calibration_method = select_calibration_method(
        metric_rows,
        error_cls=error_cls,
    )
    selected_predictions = predictions_by_method[selected_calibration_method]
    metrics = metrics_by_split(
        selected_predictions, manual_review_capacity_rate, error_cls
    )
    weighted_bin_errors = {
        split_name: weighted_calibration_error(selected_predictions[split_name])
        for split_name in REPORTING_SPLITS
    }
    threshold_rows = balanced_threshold_rows(
        config,
        f"feature_selection_{feature_set_name}",
        selected_predictions,
        created_at,
    )
    balanced_ev = {
        row["split"]: float(row["expected_value_per_applicant"])
        for row in threshold_rows
        if row["scenario_name"] == BALANCED_SCENARIO
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
        "validation_precision_at_top_decile": metrics["validation"][
            "precision_at_top_decile"
        ],
        "validation_recall_at_review_capacity": metrics["validation"][
            "recall_at_manual_review_capacity"
        ],
        "validation_weighted_calibration_error": weighted_bin_errors["validation"],
        "test_pr_auc": metrics["test"]["pr_auc"],
        "test_roc_auc": metrics["test"]["roc_auc"],
        "test_brier_score": metrics["test"]["brier_score"],
        "test_top_decile_lift": metrics["test"]["top_decile_lift"],
        "test_precision_at_top_decile": metrics["test"]["precision_at_top_decile"],
        "test_recall_at_review_capacity": metrics["test"][
            "recall_at_manual_review_capacity"
        ],
        "test_weighted_calibration_error": weighted_bin_errors["test"],
        "validation_balanced_ev_per_applicant": balanced_ev["validation"],
        "test_balanced_ev_per_applicant": balanced_ev["test"],
        "created_at": created_at,
    }


def load_lightgbm_artifact(
    model_dir: Path,
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, Any]:
    artifact_path = model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME
    return load_model_artifact(
        artifact_path,
        expected_model_type=LIGHTGBM_MODEL_TYPE,
        expected_model_version=LIGHTGBM_MODEL_VERSION,
        error_cls=error_cls,
        artifact_label="LightGBM artifact",
        missing_label="LightGBM model artifact",
    )


def load_feature_importance_rows(
    report_dir: Path,
    error_cls: type[Exception] = FeatureExperimentError,
) -> list[dict[str, Any]]:
    path = report_dir / "model_feature_importance.csv"
    if not path.exists():
        raise error_cls(f"Missing feature importance report: {path}")
    return read_csv(path)


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


def feature_sets(
    ranked_features: list[str],
    full_feature_columns: list[str],
    feature_limits: tuple[int, ...],
    include_full: bool,
    error_cls: type[Exception] = FeatureExperimentError,
) -> list[tuple[str, list[str], int | None]]:
    candidate_sets = []
    full_count = len(full_feature_columns)
    for limit in feature_limits:
        if limit <= 0:
            raise error_cls(f"Feature limits must be positive, got {limit}")
        if limit > full_count:
            raise error_cls(
                f"Feature limit {limit} exceeds full feature count {full_count}"
            )
        candidate_sets.append((f"top_{limit}", ranked_features[:limit], limit))
    if include_full:
        candidate_sets.append(("full", full_feature_columns, None))
    return candidate_sets


def prepare_feature_set_specs(
    report_dir: Path,
    full_feature_columns: list[str],
    feature_limits: tuple[int, ...],
    include_full: bool,
    error_cls: type[Exception] = FeatureExperimentError,
) -> list[tuple[str, list[str], int | None]]:
    importance_rows = load_feature_importance_rows(report_dir, error_cls=error_cls)
    ranked_features = ranked_raw_features(importance_rows, full_feature_columns)
    max_limit = max(feature_limits) if feature_limits else 0
    if len(ranked_features) < min(max_limit, len(full_feature_columns)):
        raise error_cls(
            "Feature importance ranking does not cover enough model features for requested limits: "
            f"ranked={len(ranked_features)}, requested={max_limit}"
        )
    return feature_sets(
        ranked_features,
        full_feature_columns,
        feature_limits,
        include_full,
        error_cls=error_cls,
    )


def load_split_frames(
    connection: duckdb.DuckDBPyConnection,
    split_applicant_ids: dict[str, list[int]],
    feature_columns: list[str],
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, pd.DataFrame]:
    return load_labeled_split_frames(
        connection,
        split_applicant_ids,
        feature_columns,
        error_cls=error_cls,
    )


def prediction_frames(
    pipeline: Any,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    label_prefix: str = "feature experiment",
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, pd.DataFrame]:
    frames = {}
    artifact = {"pipeline": pipeline, "model_version": label_prefix}
    for split_name in REPORTING_SPLITS:
        frame = split_frames[split_name]
        probabilities = predict_probabilities(
            artifact,
            frame,
            feature_columns,
            f"{label_prefix}_{split_name}",
            error_cls,
        )
        frames[split_name] = prediction_frame(frame, probabilities)
    return frames


def calibration_metric_rows(
    predictions_by_method: dict[str, dict[str, pd.DataFrame]],
    manual_review_capacity_rate: float,
    error_cls: type[Exception] = FeatureExperimentError,
) -> list[dict[str, Any]]:
    rows = []
    for method, split_predictions in predictions_by_method.items():
        for split_name in REPORTING_SPLITS:
            frame = split_predictions[split_name]
            metrics = probability_metrics(
                frame["target"],
                frame["probability"].to_numpy(),
                manual_review_capacity_rate,
                error_cls=error_cls,
            )
            rows.append(
                {
                    "calibration_method": method,
                    "split": split_name,
                    "brier_score": metrics["brier_score"],
                }
            )
    return rows


def metrics_by_split(
    prediction_frames_by_split: dict[str, pd.DataFrame],
    manual_review_capacity_rate: float,
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, dict[str, float]]:
    return {
        split_name: probability_metrics(
            frame["target"],
            frame["probability"].to_numpy(),
            manual_review_capacity_rate,
            error_cls=error_cls,
        )
        for split_name, frame in prediction_frames_by_split.items()
    }


def weighted_calibration_error(frame: pd.DataFrame) -> float:
    ranked = with_probability_rank_bin(frame, "bin_id", descending=False)
    total_count = len(ranked)
    weighted_error = 0.0
    for bin_id in range(1, 11):
        bin_frame = ranked.loc[ranked["bin_id"] == bin_id]
        if bin_frame.empty:
            continue
        calibration_error = float(
            bin_frame["target"].mean() - bin_frame["probability"].mean()
        )
        weighted_error += abs(calibration_error) * len(bin_frame) / total_count
    return float(weighted_error)


def balanced_threshold_rows(
    config: dict[str, Any],
    model_version: str,
    prediction_frames_by_split: dict[str, pd.DataFrame],
    created_at: str,
) -> list[dict[str, Any]]:
    scenario_thresholds = resolve_scenario_thresholds(
        threshold_policy(config),
        prediction_frames_by_split["validation"]["probability"].to_numpy(),
    )
    return build_threshold_metric_rows(
        model_version,
        threshold_version(config),
        prediction_frames_by_split,
        scenario_thresholds,
        business_assumptions(config),
        created_at,
    )


def select_feature_set(rows: list[dict[str, Any]]) -> str:
    selected = max(rows, key=feature_set_selection_key)
    return str(selected["feature_set"])


def feature_set_selection_key(
    row: dict[str, Any],
) -> tuple[float, float, float, float, float, int]:
    return (
        float(row["validation_pr_auc"]),
        float(row["validation_top_decile_lift"]),
        float(row["validation_recall_at_review_capacity"]),
        float(row["validation_roc_auc"]),
        -float(row["validation_brier_score"]),
        -int(row["feature_count"]),
    )


def _normalize_feature_label(label: str) -> str:
    return " ".join(label.lower().split())
