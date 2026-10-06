from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.calibration import (
    CALIBRATION_FIT_SPLIT,
    CALIBRATION_METHODS,
    apply_calibration_method,
    fit_calibrators,
    select_calibration_method,
)
from src.config import (
    business_assumptions,
    project_model_seed,
    threshold_policy,
    threshold_version,
)
from src.feature_labels import humanize_feature_token, readable_feature_label
from src.mart_access import load_labeled_split_frames
from src.metrics import probability_metrics, with_reliability_bin
from src.model_artifacts import load_model_artifact
from src.model_contracts import (
    LIGHTGBM_MODEL_ARTIFACT_NAME,
    LIGHTGBM_MODEL_TYPE,
    LIGHTGBM_MODEL_VERSION,
    REPORTING_SPLITS,
)
from src.modeling import (
    build_lightgbm_pipeline,
    build_lightgbm_tuning_artifact,
    classify_feature_columns,
    fit_tuned_lightgbm,
    lightgbm_params,
    predict_probabilities,
    prediction_frame,
)
from src.runtime import feature_frame, read_csv
from src.thresholding import (
    BALANCED_SCENARIO,
    build_threshold_metric_rows,
    resolve_scenario_thresholds,
)

DEFAULT_FEATURE_LIMITS = (40, 80)


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
    fitted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, calibrate, threshold, and summarize one feature-set candidate."""
    if CALIBRATION_FIT_SPLIT not in split_frames:
        raise error_cls(
            "Missing reserved calibration split; retrain with the corrected protocol"
        )
    random_seed = (
        project_model_seed(config) if random_seed is None else int(random_seed)
    )
    fitted = fitted or fit_feature_candidate(
        config,
        feature_set_name,
        feature_columns,
        feature_limit,
        {name: split_frames[name] for name in ("train", "calibration", "validation")},
        manual_review_capacity_rate,
        random_seed,
        error_cls=error_cls,
    )
    from src.tuning import uses_inner_cv

    if uses_inner_cv(config):
        feature_columns = list(fitted["feature_columns"])
        feature_set_name = fitted["selection_row"]["feature_set"]
        chosen_limit = fitted["selection_row"]["feature_limit"]
        feature_limit = None if chosen_limit == "full" else int(chosen_limit)
    pipeline = fitted["pipeline"]
    raw_predictions = prediction_frames(
        pipeline,
        split_frames,
        feature_columns,
        feature_set_name,
        error_cls,
    )
    selected_calibration_method = fitted["selected_calibration_method"]
    selected_predictions = apply_calibration_method(
        selected_calibration_method,
        fitted["calibrators"],
        raw_predictions,
        error_cls=error_cls,
    )
    metrics = metrics_by_split(
        selected_predictions, manual_review_capacity_rate, error_cls
    )
    weighted_bin_errors = {
        split_name: weighted_calibration_error(selected_predictions[split_name])
        for split_name in REPORTING_SPLITS
    }
    threshold_rows = (
        []
        if fitted["scenario_thresholds"] is None
        else balanced_threshold_rows(
            config,
            f"feature_selection_{feature_set_name}",
            raw_predictions,
            created_at,
        )
    )
    balanced_ev = {
        row["split"]: float(row["expected_value_per_applicant"])
        for row in threshold_rows
        if row["scenario_name"] == BALANCED_SCENARIO
    }
    selected_candidate = fitted["selected_candidate"]
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
        "validation_balanced_ev_per_applicant": balanced_ev.get(
            "validation", float("nan")
        ),
        "test_balanced_ev_per_applicant": balanced_ev.get("test", float("nan")),
        "created_at": created_at,
    }


def fit_feature_candidate(
    config: dict[str, Any],
    feature_set_name: str,
    feature_columns: list[str],
    feature_limit: int | None,
    roles: dict[str, pd.DataFrame],
    review_capacity_rate: float,
    model_seed: int,
    *,
    error_cls: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Fit/select a candidate without accepting an assessment population."""
    if set(roles) != {"train", "calibration", "validation"}:
        raise error_cls(
            "Candidate fitting accepts only train/calibration/validation roles"
        )
    numeric, categorical = classify_feature_columns(roles["train"], feature_columns)
    tuning = fit_tuned_lightgbm(
        config,
        numeric,
        categorical,
        lightgbm_params(config, roles["train"], model_seed),
        roles,
        feature_columns,
        review_capacity_rate,
        error_cls=error_cls,
    )
    pipeline = tuning["pipeline"]
    feature_columns = tuning.get("feature_columns", feature_columns)
    feature_set_name = tuning.get("feature_set", feature_set_name)
    selected = build_lightgbm_tuning_artifact(tuning)["selected_candidate"]
    fitted = calibrate_candidate(
        config,
        pipeline,
        feature_set_name,
        feature_columns,
        feature_limit,
        roles,
        review_capacity_rate,
        model_seed,
        selected,
        error_cls=error_cls,
    )
    if "search_evidence" in tuning:
        fitted["search_evidence"] = tuning["search_evidence"]
    return fitted


def fit_joint_workflow(config, roles, columns, review_rate, surfaces=None):
    """Joint selection before the reserved calibration/method-selection roles."""
    from src.tuning import joint_search

    if set(roles) != {"train", "calibration", "validation"}:
        raise ValueError(
            "Joint workflow accepts only train/calibration/validation roles"
        )
    from src.model_artifacts import normalize_split_ids

    normalize_split_ids(
        {role: list(frame["SK_ID_CURR"]) for role, frame in roles.items()},
        tuple(roles),
        error_cls=ValueError,
    )
    tuning = joint_search(config, roles["train"], columns, review_rate, surfaces)
    name = tuning["feature_set"]
    fitted = calibrate_candidate(
        config,
        tuning["pipeline"],
        name,
        tuning["feature_columns"],
        None if name == "full" else int(name.split("_")[1]),
        roles,
        review_rate,
        project_model_seed(config),
        build_lightgbm_tuning_artifact(tuning)["selected_candidate"],
    )
    fitted["search_evidence"] = tuning["search_evidence"]
    fitted["final_rankings"] = tuning["final_rankings"]
    fitted["selection_row"]["selected"] = True
    from src.tuning import seed_sensitivity

    fitted["seed_sensitivity"] = seed_sensitivity(config, roles, fitted, review_rate)
    return fitted


def run_joint_feature_sets(
    config,
    split_frames,
    columns,
    limits,
    include_full,
    review_rate,
    created_at,
    error_cls=ValueError,
):
    """Current experiment callers spend one joint budget per declared workflow."""
    surfaces = [f"top_{limit}" for limit in limits] + (["full"] if include_full else [])
    if not surfaces or any(limit > len(columns) for limit in limits):
        raise error_cls("Invalid joint feature surfaces")
    fitted = fit_joint_workflow(
        config,
        {role: split_frames[role] for role in ("train", "calibration", "validation")},
        columns,
        review_rate,
        surfaces,
    )
    row = run_single_feature_set(
        config,
        fitted["selection_row"]["feature_set"],
        fitted["feature_columns"],
        None
        if fitted["selection_row"]["feature_set"] == "full"
        else int(fitted["selection_row"]["feature_set"].split("_")[1]),
        split_frames,
        review_rate,
        created_at,
        error_cls=error_cls,
        fitted=fitted,
    )
    import json
    from uuid import uuid4

    from src.runtime import resolve_config_path

    output = resolve_config_path(config, "report_dir") / "tuning" / uuid4().hex
    output.mkdir(parents=True)
    (output / "search.json").write_text(
        json.dumps(fitted["search_evidence"], indent=2), encoding="utf-8"
    )
    return [row], {row["feature_set"]: fitted["feature_columns"]}


def calibrate_candidate(
    config,
    pipeline,
    feature_set_name,
    feature_columns,
    feature_limit,
    roles,
    review_capacity_rate,
    model_seed,
    selected,
    *,
    error_cls=ValueError,
):
    """Fit calibration and select a method using the same reserved inner roles."""
    if set(roles) != {"train", "calibration", "validation"}:
        raise error_cls("Calibration selection accepts only inner roles")
    artifact = {"pipeline": pipeline, "model_version": feature_set_name}
    predictions = {
        role: prediction_frame(
            frame,
            predict_probabilities(artifact, frame, feature_columns, role, error_cls),
        )
        for role, frame in roles.items()
        if role != "train"
    }
    calibrators = fit_calibrators(
        predictions["calibration"]["probability"].to_numpy(),
        predictions["calibration"]["target"].to_numpy(),
        model_seed,
        error_cls=error_cls,
    )
    validation_by_method = {
        method: apply_calibration_method(
            method,
            calibrators,
            {"validation": predictions["validation"]},
            error_cls=error_cls,
        )
        for method in CALIBRATION_METHODS
    }
    comparison = [
        {
            "split": "validation",
            "calibration_method": method,
            "brier_score": probability_metrics(
                frames["validation"]["target"],
                frames["validation"]["probability"].to_numpy(),
                review_capacity_rate,
                error_cls,
            )["brier_score"],
        }
        for method, frames in validation_by_method.items()
    ]
    method = select_calibration_method(comparison, error_cls=error_cls)
    metrics = probability_metrics(
        predictions["validation"]["target"],
        validation_by_method[method]["validation"]["probability"].to_numpy(),
        review_capacity_rate,
        error_cls,
    )
    from src.tuning import _quality, settings, uses_inner_cv

    quality = {}
    if uses_inner_cv(config):
        reference = probability_metrics(
            predictions["validation"]["target"],
            np.full(len(predictions["validation"]), roles["train"]["TARGET"].mean()),
            review_capacity_rate,
        )
        quality = {
            "selection_probability_accepted": _quality(
                metrics, reference, settings(config)
            ),
            "selection_brier_score": metrics["brier_score"],
            "selection_log_loss": metrics["log_loss"],
            "selection_prevalence_brier": reference["brier_score"],
            "selection_prevalence_log_loss": reference["log_loss"],
        }
    row = {
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "feature_limit": feature_limit if feature_limit is not None else "full",
        "selected_calibration_method": method,
        "selected_candidate_name": selected["candidate_name"],
        **{
            f"validation_{name}": metrics[name]
            for name in ("pr_auc", "roc_auc", "brier_score", "top_decile_lift")
        },
        "validation_recall_at_review_capacity": metrics[
            "recall_at_manual_review_capacity"
        ],
    }
    return {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "calibrators": calibrators,
        "selected_calibration_method": method,
        "selected_candidate": selected,
        "selection_row": row,
        "probability_quality": quality,
        "scenario_thresholds": None
        if uses_inner_cv(config)
        and predictions["validation"]["probability"].nunique() < 2
        else resolve_scenario_thresholds(
            threshold_policy(config),
            predictions["validation"]["probability"].to_numpy(),
        ),
    }


def load_lightgbm_artifact(
    model_dir: Path,
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, Any]:
    """Load the base LightGBM artifact for feature experiments."""
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
    """Load feature-importance rows that seed top-N feature subsets."""
    path = report_dir / "model_feature_importance.csv"
    if not path.exists():
        raise error_cls(f"Missing feature importance report: {path}")
    return read_csv(path)


def ranked_raw_features(
    importance_rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> list[str]:
    """Map ranked display labels back to raw model feature names."""
    aliases: dict[str, set[str]] = {}
    for feature_column in feature_columns:
        for label in (
            humanize_feature_token(feature_column),  # Literal historical display text.
            readable_feature_label(feature_column),
        ):
            aliases.setdefault(_normalize_feature_label(label), set()).add(
                feature_column
            )
    ranked_features: list[str] = []
    seen_features = set()
    rows = sorted(importance_rows, key=lambda row: int(row["rank"]))
    for row in rows:
        raw_label = str(row["feature_name"]).split(":", 1)[0]
        if raw_label in feature_columns:
            raw_feature = raw_label
        else:
            matches = aliases.get(_normalize_feature_label(raw_label), set())
            if len(matches) > 1:
                raise FeatureExperimentError(
                    f"Ambiguous feature display label: {raw_label}"
                )
            raw_feature = next(iter(matches), None)
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
    """Build top-N and optional full feature-set specifications."""
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
    if not candidate_sets:
        raise error_cls("At least one feature set must be requested")
    return candidate_sets


def prepare_feature_set_specs(
    report_dir: Path,
    full_feature_columns: list[str],
    feature_limits: tuple[int, ...],
    include_full: bool,
    error_cls: type[Exception] = FeatureExperimentError,
    *,
    training_frame: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> list[tuple[str, list[str], int | None]]:
    """Rank candidates using a model fitted only on the current training role."""
    if not feature_limits:
        return feature_sets([], full_feature_columns, (), include_full, error_cls)
    if training_frame is None or config is None:
        raise error_cls(
            "Feature selection requires the current training population; reporting SHAP is not a selection input"
        )
    ranking_rows = training_feature_ranking(
        training_frame, full_feature_columns, config, error_cls=error_cls
    )
    ranked_features = [row["feature_name"] for row in ranking_rows]
    return feature_sets(
        ranked_features, full_feature_columns, feature_limits, include_full, error_cls
    )


def ranking_seeds(
    config: dict[str, Any], error_cls: type[Exception] = ValueError
) -> tuple[int, ...]:
    """Use a fixed, independently configured collection of model seeds."""
    values = config.get("feature_selection", {}).get("ranking_seeds", [101, 211, 307])
    if (
        not isinstance(values, (list, tuple))
        or not values
        or any(type(value) is not int or not 0 <= value < 2**32 for value in values)
        or len(values) != len(set(values))
    ):
        raise error_cls(
            "feature_selection.ranking_seeds must be distinct nonnegative integer seeds"
        )
    return tuple(values)


def training_feature_ranking(
    training_frame: pd.DataFrame,
    full_feature_columns: list[str],
    config: dict[str, Any],
    *,
    error_cls: type[Exception] = ValueError,
) -> list[dict[str, Any]]:
    """Average raw-feature ranks across model repeats of exactly these rows."""
    seeds = ranking_seeds(config, error_cls)
    ranks = {column: [] for column in full_feature_columns}
    for seed in seeds:
        ordered = _single_training_ranking(
            training_frame, full_feature_columns, config, seed, error_cls
        )
        for rank, column in enumerate(ordered, 1):
            ranks[column].append(rank)
    rows = [
        {
            "feature_name": column,
            "mean_rank": float(np.mean(values)),
            "rank_std": float(np.std(values)),
            "ranking_seed_count": len(seeds),
        }
        for column, values in ranks.items()
    ]
    return sorted(rows, key=lambda row: (row["mean_rank"], row["feature_name"]))


def _single_training_ranking(
    training_frame, full_feature_columns, config, seed, error_cls
):
    numeric, categorical = classify_feature_columns(
        training_frame, full_feature_columns
    )
    ranking_model = build_lightgbm_pipeline(
        numeric, categorical, lightgbm_params(config, training_frame, seed)
    )
    ranking_model.fit(
        feature_frame(training_frame, full_feature_columns),
        training_frame["TARGET"].astype(int),
    )
    preprocessor = ranking_model.named_steps["preprocessor"]
    expanded_raw_columns = list(numeric)
    if categorical:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        for raw_column, categories in zip(
            categorical, encoder.categories_, strict=True
        ):
            expanded_raw_columns.extend([raw_column] * len(categories))
    gains = ranking_model.named_steps["classifier"].booster_.feature_importance(
        importance_type="gain"
    )
    if len(gains) != len(expanded_raw_columns):
        raise error_cls("Training importance does not reconcile to raw feature columns")
    raw_gains = dict.fromkeys(full_feature_columns, 0.0)
    for column, gain in zip(expanded_raw_columns, gains, strict=True):
        raw_gains[column] += float(gain)
    ranked_features = sorted(
        full_feature_columns, key=lambda column: (-raw_gains[column], column)
    )
    return ranked_features


def load_split_frames(
    connection: duckdb.DuckDBPyConnection,
    split_applicant_ids: dict[str, list[int]],
    feature_columns: list[str],
    error_cls: type[Exception] = FeatureExperimentError,
) -> dict[str, pd.DataFrame]:
    """Load experiment split frames for a candidate feature surface."""
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
    """Build validation and test prediction frames for a candidate pipeline."""
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
    """Build calibration-selection rows with validation and test Brier scores."""
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
    """Return the standard probability metric bundle keyed by split."""
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
    """Calculate applicant-weighted absolute calibration-bin error."""
    ranked = with_reliability_bin(frame)
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
    """Build balanced-scenario threshold rows for a feature experiment."""
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
    """Select the best feature set using the validation-only ranking key."""
    selected = max(rows, key=feature_set_selection_key)
    return str(selected["feature_set"])


def feature_set_selection_key(
    row: dict[str, Any],
) -> tuple[float, float, float, float, float, int]:
    """Return the validation-first ranking key for feature-set selection."""
    return (
        float(row["validation_pr_auc"]),
        float(row["validation_top_decile_lift"]),
        float(row["validation_recall_at_review_capacity"]),
        float(row["validation_roc_auc"]),
        -float(row["validation_brier_score"]),
        -int(row["feature_count"]),
    )


def _normalize_feature_label(label: str) -> str:
    """Normalize display labels before mapping them back to raw features."""
    return " ".join(label.lower().split())
