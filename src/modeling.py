from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.metrics import precision_at_rate
from src.metrics import recall_at_rate
from src.metrics import top_decile_lift
from src.metrics import validate_probabilities
from src.runtime import feature_frame
from src.runtime import sql_identifier

LIGHTGBM_SELECTION_METRIC_ORDER = [
    "nonconstant_score_distribution",
    "pr_auc",
    "top_decile_lift",
    "recall_at_manual_review_capacity",
    "roc_auc",
    "brier_score",
]


def load_labeled_training_frame(
    connection: duckdb.DuckDBPyConnection,
    feature_columns: list[str],
    error_cls: type[Exception] = ValueError,
) -> pd.DataFrame:
    if not feature_columns:
        raise error_cls("No model feature columns are available for training")

    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    query = f"""
        SELECT {", ".join(sql_identifier(column) for column in selected_columns)}
        FROM mart_credit_risk_features
        WHERE source_population = 'application_train'
        ORDER BY SK_ID_CURR
    """
    frame = connection.execute(query).fetch_df()
    if frame.empty:
        raise error_cls("No labeled application_train rows are available for training")
    target_values = set(frame["TARGET"].dropna().astype(int).unique())
    if target_values != {0, 1}:
        raise error_cls(f"Training TARGET must contain both binary classes, got {sorted(target_values)}")
    return frame


def split_labeled_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    random_seed: int,
    error_cls: type[Exception] = ValueError,
) -> dict[str, pd.DataFrame]:
    split_config = config["split"]
    validation_size = float(split_config["validation_size"])
    test_size = float(split_config["test_size"])
    holdout_size = validation_size + test_size
    if holdout_size <= 0:
        raise error_cls("Validation and test split sizes must be positive")

    train_frame, holdout_frame = train_test_split(
        frame,
        test_size=holdout_size,
        stratify=frame["TARGET"].astype(int),
        random_state=random_seed,
    )
    validation_test_ratio = test_size / holdout_size
    validation_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=validation_test_ratio,
        stratify=holdout_frame["TARGET"].astype(int),
        random_state=random_seed,
    )
    split_frames = {
        "train": train_frame.sort_values("SK_ID_CURR").reset_index(drop=True),
        "validation": validation_frame.sort_values("SK_ID_CURR").reset_index(drop=True),
        "test": test_frame.sort_values("SK_ID_CURR").reset_index(drop=True),
    }
    for split_name, split_frame in split_frames.items():
        split_targets = set(split_frame["TARGET"].astype(int).unique())
        if split_targets != {0, 1}:
            raise error_cls(f"{split_name} split must contain both target classes")
    return split_frames


def classify_feature_columns(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(train_frame[column])
    ]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    return numeric_features, categorical_features


def build_baseline_pipeline(
    config: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    random_seed: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    class_weight = "balanced" if config["model"]["use_class_weighting"] else None
    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_seed,
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_lightgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    lightgbm_params: dict[str, Any],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    classifier = LGBMClassifier(**lightgbm_params)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def lightgbm_params(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    random_seed: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": random_seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if config["model"]["use_class_weighting"]:
        positive_count = int(train_frame["TARGET"].sum())
        negative_count = len(train_frame) - positive_count
        params["scale_pos_weight"] = negative_count / positive_count
    return params


def fit_tuned_lightgbm(
    config: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    base_params: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    manual_review_capacity_rate: float,
    error_cls: type[Exception] = ValueError,
) -> dict[str, Any]:
    tuning_config = config["model"].get("lightgbm_tuning", {})
    tuning_enabled = bool(tuning_config.get("enabled", True))
    max_candidates = int(tuning_config.get("max_candidates", 8))
    if max_candidates < 1:
        raise error_cls("model.lightgbm_tuning.max_candidates must be at least 1")

    candidate_specs = _lightgbm_candidate_specs(base_params, max_candidates, tuning_enabled)
    x_train = feature_frame(split_frames["train"], feature_columns)
    y_train = split_frames["train"]["TARGET"].astype(int)
    validation_frame = split_frames["validation"]
    y_validation = validation_frame["TARGET"].astype(int)
    x_validation = feature_frame(validation_frame, feature_columns)

    candidates: list[dict[str, Any]] = []
    for candidate_spec in candidate_specs:
        pipeline = build_lightgbm_pipeline(
            numeric_features,
            categorical_features,
            candidate_spec["params"],
        )
        pipeline.fit(x_train, y_train)
        validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        validation_metrics = probability_metrics(
            y_validation,
            validation_probabilities,
            manual_review_capacity_rate,
            error_cls=error_cls,
        )
        candidates.append(
            {
                **candidate_spec,
                "pipeline": pipeline,
                "validation_metrics": validation_metrics,
                "selection_key": _lightgbm_selection_key(validation_metrics),
                "validation_selection_score": _lightgbm_selection_score(validation_metrics),
            }
        )

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: candidate["selection_key"],
        reverse=True,
    )
    selected_candidate = ranked_candidates[0]
    return {
        "enabled": tuning_enabled,
        "max_candidates": max_candidates,
        "candidate_count": len(candidates),
        "selection_metric_order": LIGHTGBM_SELECTION_METRIC_ORDER,
        "candidates": candidates,
        "ranked_candidates": ranked_candidates,
        "selected_candidate": selected_candidate,
        "pipeline": selected_candidate["pipeline"],
    }


def probability_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    manual_review_capacity_rate: float,
    error_cls: type[Exception] = ValueError,
) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "min_predicted_probability": float(np.min(probabilities)),
        "max_predicted_probability": float(np.max(probabilities)),
        "top_decile_lift": top_decile_lift(y_true, probabilities, error_cls=error_cls),
        "precision_at_top_decile": precision_at_rate(y_true, probabilities, 0.10, error_cls=error_cls),
        "recall_at_manual_review_capacity": recall_at_rate(
            y_true,
            probabilities,
            manual_review_capacity_rate,
            error_cls=error_cls,
        ),
    }


def predict_probabilities(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    label: str,
    error_cls: type[Exception] = ValueError,
) -> np.ndarray:
    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "predict_proba"):
        raise error_cls(f"Model {artifact['model_version']} does not expose predict_proba")
    probabilities = pipeline.predict_proba(feature_frame(frame, feature_columns))[:, 1]
    validate_probabilities(probabilities, label, error_cls=error_cls)
    return probabilities.astype(float)


def prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": frame["SK_ID_CURR"].astype(int),
            "target": frame["TARGET"].astype(int),
            "probability": probabilities.astype(float),
        }
    )


def build_lightgbm_tuning_artifact(tuning_result: dict[str, Any]) -> dict[str, Any]:
    selected = tuning_result["selected_candidate"]
    selected_rank = next(
        rank
        for rank, candidate in enumerate(tuning_result["ranked_candidates"], start=1)
        if candidate["candidate_name"] == selected["candidate_name"]
    )
    return {
        "enabled": tuning_result["enabled"],
        "max_candidates": tuning_result["max_candidates"],
        "candidate_count": tuning_result["candidate_count"],
        "selection_metric_order": tuning_result["selection_metric_order"],
        "selected_candidate": {
            "candidate_rank": selected_rank,
            "candidate_name": selected["candidate_name"],
            "candidate_source": selected["candidate_source"],
            "validation_selection_score": selected["validation_selection_score"],
            "validation_metrics": selected["validation_metrics"],
            "params": selected["params"],
        },
    }


def _lightgbm_candidate_specs(
    base_params: dict[str, Any],
    max_candidates: int,
    tuning_enabled: bool,
) -> list[dict[str, Any]]:
    base_scale_pos_weight = float(base_params.get("scale_pos_weight", 1.0))
    presets = [
        (
            "baseline_current",
            "current_default",
            {},
        ),
        (
            "regularized_low_learning_rate",
            "prior_informed",
            {
                "n_estimators": 500,
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.05,
                "reg_lambda": 4.0,
            },
        ),
        (
            "rank_focused_class_weighted",
            "prior_informed",
            {
                "n_estimators": 650,
                "learning_rate": 0.025,
                "num_leaves": 63,
                "max_depth": -1,
                "min_child_samples": 35,
                "subsample": 0.85,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.0,
                "reg_lambda": 3.0,
                "scale_pos_weight": base_scale_pos_weight * 1.15,
            },
        ),
        (
            "shallow_calibrated",
            "prior_informed",
            {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 5,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.10,
                "reg_lambda": 8.0,
            },
        ),
        (
            "high_recall_weighted",
            "prior_informed",
            {
                "n_estimators": 450,
                "learning_rate": 0.04,
                "num_leaves": 47,
                "max_depth": -1,
                "min_child_samples": 25,
                "subsample": 0.80,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 2.0,
                "scale_pos_weight": base_scale_pos_weight * 1.35,
            },
        ),
        (
            "feature_subsample_regularized",
            "prior_informed",
            {
                "n_estimators": 600,
                "learning_rate": 0.03,
                "num_leaves": 47,
                "max_depth": -1,
                "min_child_samples": 60,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.10,
                "reg_lambda": 6.0,
            },
        ),
        (
            "compact_conservative",
            "prior_informed",
            {
                "n_estimators": 350,
                "learning_rate": 0.05,
                "num_leaves": 11,
                "max_depth": 4,
                "min_child_samples": 60,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "reg_alpha": 0.20,
                "reg_lambda": 8.0,
            },
        ),
        (
            "lighter_weight_calibrated",
            "prior_informed",
            {
                "n_estimators": 500,
                "learning_rate": 0.035,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 45,
                "subsample": 0.85,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.05,
                "reg_lambda": 4.0,
                "scale_pos_weight": base_scale_pos_weight * 0.80,
            },
        ),
    ]
    candidate_limit = max_candidates if tuning_enabled else 1
    specs: list[dict[str, Any]] = []
    for candidate_name, candidate_source, overrides in presets[:candidate_limit]:
        params = {**base_params, **overrides}
        specs.append(
            {
                "candidate_name": candidate_name,
                "candidate_source": candidate_source,
                "params": params,
            }
        )
    return specs


def _lightgbm_selection_key(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        1.0 if _has_nonconstant_score_distribution(metrics) else 0.0,
        metrics["pr_auc"],
        metrics["top_decile_lift"],
        metrics["recall_at_manual_review_capacity"],
        metrics["roc_auc"],
        -metrics["brier_score"],
    )


def _lightgbm_selection_score(metrics: dict[str, float]) -> float:
    return float(metrics["pr_auc"])


def _has_nonconstant_score_distribution(metrics: dict[str, float]) -> bool:
    return metrics["max_predicted_probability"] > metrics["min_predicted_probability"]
