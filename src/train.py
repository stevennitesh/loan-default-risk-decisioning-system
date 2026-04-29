from __future__ import annotations

import argparse
import csv
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
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

from src.config import load_config
from src.data_contracts import DataContractError
from src.data_contracts import get_model_feature_columns
from src.data_contracts import validate_data_contracts


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_VERSION = "logistic_regression_baseline_v1"
BASELINE_MODEL_TYPE = "logistic_regression"
BASELINE_MODEL_ARTIFACT_NAME = "logistic_regression_baseline.joblib"
LIGHTGBM_MODEL_VERSION = "lightgbm_credit_risk_v1"
LIGHTGBM_MODEL_TYPE = "lightgbm"
LIGHTGBM_MODEL_ARTIFACT_NAME = "lightgbm_credit_risk.joblib"

MODEL_RUN_SUMMARY_COLUMNS = [
    "model_version",
    "run_id",
    "model_type",
    "data_scope_version",
    "train_rows",
    "validation_rows",
    "test_rows",
    "feature_count",
    "positive_rate_train",
    "random_seed",
    "created_at",
]

MODEL_METRICS_SUMMARY_COLUMNS = [
    "model_version",
    "split",
    "metric_name",
    "metric_value",
    "created_at",
]

SPLIT_SUMMARY_COLUMNS = [
    "model_version",
    "run_id",
    "split",
    "row_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "created_at",
]

MODEL_COMPARISON_SUMMARY_COLUMNS = [
    "metric_name",
    "baseline_metric_value",
    "lightgbm_metric_value",
    "lightgbm_minus_baseline",
    "selected_model_type",
]

LIGHTGBM_TUNING_SUMMARY_COLUMNS = [
    "candidate_rank",
    "selected",
    "candidate_name",
    "candidate_source",
    "validation_selection_score",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_manual_review_capacity",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
    "created_at",
]

LIGHTGBM_SELECTION_METRIC_ORDER = [
    "nonconstant_score_distribution",
    "pr_auc",
    "top_decile_lift",
    "recall_at_manual_review_capacity",
    "roc_auc",
    "brier_score",
]


class TrainingError(RuntimeError):
    """Raised when baseline training cannot satisfy the Milestone 4 contract."""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def run_training(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])
    report_dir = _resolve_project_path(config["paths"]["report_dir"])

    if not duckdb_path.exists():
        raise TrainingError(f"DuckDB database not found: {duckdb_path}")

    created_at = _created_at()
    run_id = f"model_training_v1_{created_at.replace('-', '').replace(':', '').replace('Z', '')}"
    random_seed = int(config["project"]["random_seed"])
    manual_review_capacity_rate = float(config["business_assumptions"]["manual_review_capacity_rate"])

    with duckdb.connect(str(duckdb_path)) as connection:
        try:
            validate_data_contracts(connection, config)
        except DataContractError as error:
            raise TrainingError(f"Data contract validation failed before training: {error}") from error

        feature_columns = get_model_feature_columns(connection, config)
        training_frame = _load_labeled_training_frame(connection, feature_columns)

        split_frames = _split_labeled_frame(training_frame, config, random_seed)
        numeric_features, categorical_features = _classify_feature_columns(
            split_frames["train"],
            feature_columns,
        )
        baseline_pipeline = _build_baseline_pipeline(
            config,
            numeric_features,
            categorical_features,
            random_seed,
        )
        base_lightgbm_params = _lightgbm_params(
            config,
            split_frames["train"],
            random_seed,
        )

        x_train = _feature_frame(split_frames["train"], feature_columns)
        y_train = split_frames["train"]["TARGET"].astype(int)
        baseline_pipeline.fit(x_train, y_train)
        lightgbm_tuning = _fit_tuned_lightgbm(
            config,
            numeric_features,
            categorical_features,
            base_lightgbm_params,
            split_frames,
            feature_columns,
            manual_review_capacity_rate,
        )
        lightgbm_pipeline = lightgbm_tuning["pipeline"]
        lightgbm_params = lightgbm_tuning["selected_candidate"]["params"]

        split_summary_rows = _build_split_summary(split_frames, run_id, created_at)
        baseline_metric_rows = _build_metric_rows(
            BASELINE_MODEL_VERSION,
            baseline_pipeline,
            split_frames,
            feature_columns,
            created_at,
            manual_review_capacity_rate,
        )
        lightgbm_metric_rows = _build_metric_rows(
            LIGHTGBM_MODEL_VERSION,
            lightgbm_pipeline,
            split_frames,
            feature_columns,
            created_at,
            manual_review_capacity_rate,
        )
        metric_rows = [*baseline_metric_rows, *lightgbm_metric_rows]
        comparison_rows = _build_model_comparison_rows(baseline_metric_rows, lightgbm_metric_rows)
        tuning_summary_rows = _build_lightgbm_tuning_summary_rows(lightgbm_tuning, created_at)
        run_summary_rows = [
            _build_run_summary_row(
                config,
                run_id,
                BASELINE_MODEL_VERSION,
                BASELINE_MODEL_TYPE,
                split_summary_rows,
                len(feature_columns),
                created_at,
                random_seed,
            ),
            _build_run_summary_row(
                config,
                run_id,
                LIGHTGBM_MODEL_VERSION,
                LIGHTGBM_MODEL_TYPE,
                split_summary_rows,
                len(feature_columns),
                created_at,
                random_seed,
            ),
        ]

        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        common_artifact_fields = {
            "run_id": run_id,
            "feature_columns": feature_columns,
            "numeric_feature_columns": numeric_features,
            "categorical_feature_columns": categorical_features,
            "split_config": config["split"],
            "split_summary": split_summary_rows,
            "split_applicant_ids": {
                split_name: [int(value) for value in frame["SK_ID_CURR"].tolist()]
                for split_name, frame in split_frames.items()
            },
            "created_at": created_at,
        }
        baseline_artifact = {
            **common_artifact_fields,
            "pipeline": baseline_pipeline,
            "model_version": BASELINE_MODEL_VERSION,
            "model_type": BASELINE_MODEL_TYPE,
            "metric_rows": baseline_metric_rows,
        }
        lightgbm_artifact = {
            **common_artifact_fields,
            "pipeline": lightgbm_pipeline,
            "model_version": LIGHTGBM_MODEL_VERSION,
            "model_type": LIGHTGBM_MODEL_TYPE,
            "metric_rows": lightgbm_metric_rows,
            "lightgbm_params": lightgbm_params,
            "lightgbm_tuning": _build_lightgbm_tuning_artifact(lightgbm_tuning),
        }
        joblib.dump(baseline_artifact, model_dir / BASELINE_MODEL_ARTIFACT_NAME)
        joblib.dump(lightgbm_artifact, model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME)

        _write_csv(report_dir / "model_run_summary.csv", MODEL_RUN_SUMMARY_COLUMNS, run_summary_rows)
        _write_csv(report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS, metric_rows)
        _write_csv(report_dir / "split_summary.csv", SPLIT_SUMMARY_COLUMNS, split_summary_rows)
        _write_csv(
            report_dir / "lightgbm_tuning_summary.csv",
            LIGHTGBM_TUNING_SUMMARY_COLUMNS,
            tuning_summary_rows,
        )
        _write_csv(
            report_dir / "model_comparison_summary.csv",
            MODEL_COMPARISON_SUMMARY_COLUMNS,
            comparison_rows,
        )
        _replace_duckdb_table(connection, "model_run_summary", run_summary_rows)
        _replace_duckdb_table(connection, "model_metrics_summary", metric_rows)
        _replace_duckdb_table(connection, "split_summary", split_summary_rows)
        _replace_duckdb_table(connection, "lightgbm_tuning_summary", tuning_summary_rows)
        _replace_duckdb_table(connection, "model_comparison_summary", comparison_rows)

    return {
        "run_id": run_id,
        "feature_columns": feature_columns,
        "artifacts": {
            BASELINE_MODEL_TYPE: model_dir / BASELINE_MODEL_ARTIFACT_NAME,
            LIGHTGBM_MODEL_TYPE: model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME,
        },
        "run_summary": run_summary_rows,
        "metric_rows": metric_rows,
        "comparison_rows": comparison_rows,
        "split_summary": split_summary_rows,
        "lightgbm_tuning_rows": tuning_summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and primary credit-risk models.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    try:
        run_training(args.config)
    except TrainingError as error:
        raise SystemExit(str(error)) from error


def _load_labeled_training_frame(
    connection: duckdb.DuckDBPyConnection,
    feature_columns: list[str],
) -> pd.DataFrame:
    if not feature_columns:
        raise TrainingError("No model feature columns are available for training")

    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    query = f"""
        SELECT {", ".join(_sql_identifier(column) for column in selected_columns)}
        FROM mart_credit_risk_features
        WHERE source_population = 'application_train'
        ORDER BY SK_ID_CURR
    """
    frame = connection.execute(query).fetch_df()
    if frame.empty:
        raise TrainingError("No labeled application_train rows are available for training")
    target_values = set(frame["TARGET"].dropna().astype(int).unique())
    if target_values != {0, 1}:
        raise TrainingError(f"Training TARGET must contain both binary classes, got {sorted(target_values)}")
    return frame


def _split_labeled_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    split_config = config["split"]
    validation_size = float(split_config["validation_size"])
    test_size = float(split_config["test_size"])
    holdout_size = validation_size + test_size
    if holdout_size <= 0:
        raise TrainingError("Validation and test split sizes must be positive")

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
            raise TrainingError(f"{split_name} split must contain both target classes")
    return split_frames


def _classify_feature_columns(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(train_frame[column])
    ]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    return numeric_features, categorical_features


def _build_baseline_pipeline(
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


def _build_lightgbm_pipeline(
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


def _lightgbm_params(
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


def _fit_tuned_lightgbm(
    config: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    base_params: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    manual_review_capacity_rate: float,
) -> dict[str, Any]:
    tuning_config = config["model"].get("lightgbm_tuning", {})
    tuning_enabled = bool(tuning_config.get("enabled", True))
    max_candidates = int(tuning_config.get("max_candidates", 8))
    if max_candidates < 1:
        raise TrainingError("model.lightgbm_tuning.max_candidates must be at least 1")

    candidate_specs = _lightgbm_candidate_specs(base_params, max_candidates, tuning_enabled)
    x_train = _feature_frame(split_frames["train"], feature_columns)
    y_train = split_frames["train"]["TARGET"].astype(int)
    validation_frame = split_frames["validation"]
    y_validation = validation_frame["TARGET"].astype(int)
    x_validation = _feature_frame(validation_frame, feature_columns)

    candidates: list[dict[str, Any]] = []
    for candidate_spec in candidate_specs:
        pipeline = _build_lightgbm_pipeline(
            numeric_features,
            categorical_features,
            candidate_spec["params"],
        )
        pipeline.fit(x_train, y_train)
        validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        validation_metrics = _probability_metrics(
            y_validation,
            validation_probabilities,
            manual_review_capacity_rate,
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


def _build_lightgbm_tuning_summary_rows(
    tuning_result: dict[str, Any],
    created_at: str,
) -> list[dict[str, Any]]:
    selected_name = tuning_result["selected_candidate"]["candidate_name"]
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(tuning_result["ranked_candidates"], start=1):
        metrics = candidate["validation_metrics"]
        params = candidate["params"]
        rows.append(
            {
                "candidate_rank": rank,
                "selected": candidate["candidate_name"] == selected_name,
                "candidate_name": candidate["candidate_name"],
                "candidate_source": candidate["candidate_source"],
                "validation_selection_score": candidate["validation_selection_score"],
                "validation_pr_auc": metrics["pr_auc"],
                "validation_roc_auc": metrics["roc_auc"],
                "validation_brier_score": metrics["brier_score"],
                "validation_top_decile_lift": metrics["top_decile_lift"],
                "validation_precision_at_top_decile": metrics["precision_at_top_decile"],
                "validation_recall_at_manual_review_capacity": metrics[
                    "recall_at_manual_review_capacity"
                ],
                "n_estimators": params.get("n_estimators"),
                "learning_rate": params.get("learning_rate"),
                "num_leaves": params.get("num_leaves"),
                "max_depth": params.get("max_depth"),
                "min_child_samples": params.get("min_child_samples"),
                "subsample": params.get("subsample"),
                "colsample_bytree": params.get("colsample_bytree"),
                "reg_alpha": params.get("reg_alpha"),
                "reg_lambda": params.get("reg_lambda"),
                "scale_pos_weight": params.get("scale_pos_weight"),
                "created_at": created_at,
            }
        )
    return rows


def _build_lightgbm_tuning_artifact(tuning_result: dict[str, Any]) -> dict[str, Any]:
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


def _feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    return features.where(pd.notna(features), np.nan)


def _build_split_summary(
    split_frames: dict[str, pd.DataFrame],
    run_id: str,
    created_at: str,
) -> list[dict[str, Any]]:
    rows = []
    for split_name, frame in split_frames.items():
        positives = int(frame["TARGET"].sum())
        row_count = len(frame)
        rows.append(
            {
                "model_version": BASELINE_MODEL_VERSION,
                "run_id": run_id,
                "split": split_name,
                "row_count": row_count,
                "positive_count": positives,
                "negative_count": row_count - positives,
                "positive_rate": positives / row_count,
                "created_at": created_at,
            }
        )
    return rows


def _build_metric_rows(
    model_version: str,
    pipeline: Pipeline,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    created_at: str,
    manual_review_capacity_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, frame in split_frames.items():
        y_true = frame["TARGET"].astype(int)
        probabilities = pipeline.predict_proba(_feature_frame(frame, feature_columns))[:, 1]
        metrics = _probability_metrics(y_true, probabilities, manual_review_capacity_rate)
        rows.extend(
            {
                "model_version": model_version,
                "split": split_name,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "created_at": created_at,
            }
            for metric_name, metric_value in metrics.items()
        )
    return rows


def _probability_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    manual_review_capacity_rate: float,
) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "min_predicted_probability": float(np.min(probabilities)),
        "max_predicted_probability": float(np.max(probabilities)),
        "top_decile_lift": _top_decile_lift(y_true, probabilities),
        "precision_at_top_decile": _precision_at_rate(y_true, probabilities, 0.10),
        "recall_at_manual_review_capacity": _recall_at_rate(
            y_true,
            probabilities,
            manual_review_capacity_rate,
        ),
    }


def _top_decile_lift(y_true: pd.Series, probabilities: np.ndarray) -> float:
    portfolio_positive_rate = float(y_true.mean())
    top_precision = _precision_at_rate(y_true, probabilities, 0.10)
    return float(top_precision / portfolio_positive_rate)


def _precision_at_rate(y_true: pd.Series, probabilities: np.ndarray, rate: float) -> float:
    top_count = _top_count(len(y_true), rate)
    frame = pd.DataFrame({"target": y_true.to_numpy(), "probability": probabilities})
    return float(frame.sort_values("probability", ascending=False).head(top_count)["target"].mean())


def _recall_at_rate(y_true: pd.Series, probabilities: np.ndarray, rate: float) -> float:
    top_count = _top_count(len(y_true), rate)
    frame = pd.DataFrame({"target": y_true.to_numpy(), "probability": probabilities})
    positives_in_top = int(frame.sort_values("probability", ascending=False).head(top_count)["target"].sum())
    total_positives = int(frame["target"].sum())
    return float(positives_in_top / total_positives) if total_positives else 0.0


def _top_count(row_count: int, rate: float) -> int:
    if rate <= 0 or rate > 1:
        raise TrainingError(f"Selection rate must be in (0, 1], got {rate}")
    return max(1, int(np.ceil(row_count * rate)))


def _build_model_comparison_rows(
    baseline_metric_rows: list[dict[str, Any]],
    lightgbm_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_validation = {
        row["metric_name"]: float(row["metric_value"])
        for row in baseline_metric_rows
        if row["split"] == "validation"
    }
    lightgbm_validation = {
        row["metric_name"]: float(row["metric_value"])
        for row in lightgbm_metric_rows
        if row["split"] == "validation"
    }
    selected_model_type = (
        LIGHTGBM_MODEL_TYPE
        if lightgbm_validation["pr_auc"] >= baseline_validation["pr_auc"]
        else BASELINE_MODEL_TYPE
    )
    return [
        {
            "metric_name": metric_name,
            "baseline_metric_value": baseline_validation[metric_name],
            "lightgbm_metric_value": lightgbm_validation[metric_name],
            "lightgbm_minus_baseline": lightgbm_validation[metric_name]
            - baseline_validation[metric_name],
            "selected_model_type": selected_model_type,
        }
        for metric_name in baseline_validation
    ]


def _build_run_summary_row(
    config: dict[str, Any],
    run_id: str,
    model_version: str,
    model_type: str,
    split_summary_rows: list[dict[str, Any]],
    feature_count: int,
    created_at: str,
    random_seed: int,
) -> dict[str, Any]:
    split_rows = {row["split"]: row for row in split_summary_rows}
    return {
        "model_version": model_version,
        "run_id": run_id,
        "model_type": model_type,
        "data_scope_version": config["project"]["data_scope_version"],
        "train_rows": split_rows["train"]["row_count"],
        "validation_rows": split_rows["validation"]["row_count"],
        "test_rows": split_rows["test"]["row_count"],
        "feature_count": feature_count,
        "positive_rate_train": split_rows["train"]["positive_rate"],
        "random_seed": random_seed,
        "created_at": created_at,
    }


def _replace_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(rows)
    connection.register("output_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {_sql_identifier(table_name)} AS SELECT * FROM output_frame")
    connection.unregister("output_frame")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _created_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


if __name__ == "__main__":
    main()
