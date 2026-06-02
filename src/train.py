from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

from src.cli import add_config_argument, exit_with_error
from src.config import (
    DEFAULT_CONFIG_PATH,
    data_scope_version,
    load_config,
    manual_review_capacity_rate,
    project_random_seed,
)
from src.data_contracts import (
    DataContractError,
    get_model_feature_columns,
    validate_data_contracts,
)
from src.metrics import build_probability_metric_rows
from src.model_contracts import (
    BASELINE_MODEL_ARTIFACT_NAME,
    BASELINE_MODEL_TYPE,
    BASELINE_MODEL_VERSION,
    LIGHTGBM_MODEL_ARTIFACT_NAME,
    LIGHTGBM_MODEL_TYPE,
    LIGHTGBM_MODEL_VERSION,
    select_model_type_by_validation_pr_auc,
)
from src.modeling import (
    build_baseline_pipeline,
    build_lightgbm_tuning_artifact,
    classify_feature_columns,
    fit_tuned_lightgbm,
    lightgbm_params,
    load_labeled_training_frame,
    predict_probabilities,
    prediction_frame,
    split_labeled_frame,
)
from src.report_contracts import (
    LIGHTGBM_TUNING_SUMMARY_COLUMNS,
    MODEL_COMPARISON_SUMMARY_COLUMNS,
    MODEL_METRICS_SUMMARY_COLUMNS,
    MODEL_RUN_SUMMARY_COLUMNS,
    SPLIT_SUMMARY_COLUMNS,
)
from src.runtime import (
    created_at_utc,
    ensure_directories,
    feature_frame,
    replace_duckdb_table,
    require_existing_path,
    resolve_config_path,
    write_csv,
)


class TrainingError(RuntimeError):
    """Raised when baseline training cannot satisfy the Milestone 4 contract."""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def run_training(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    require_existing_path(duckdb_path, "DuckDB database", TrainingError)

    created_at = created_at_utc()
    run_id = f"model_training_v1_{created_at.replace('-', '').replace(':', '').replace('Z', '')}"
    random_seed = project_random_seed(config)
    review_capacity_rate = manual_review_capacity_rate(config)

    with duckdb.connect(str(duckdb_path)) as connection:
        # Training is gated by the mart contract so artifacts cannot be created from invalid grain.
        try:
            validate_data_contracts(connection, config)
        except DataContractError as error:
            raise TrainingError(f"Data contract validation failed before training: {error}") from error

        feature_columns = get_model_feature_columns(connection, config)
        training_frame = load_labeled_training_frame(
            connection,
            feature_columns,
            error_cls=TrainingError,
        )

        # Baseline and LightGBM share the exact split so model comparisons are apples-to-apples.
        split_frames = split_labeled_frame(
            training_frame,
            config,
            random_seed,
            error_cls=TrainingError,
        )
        numeric_features, categorical_features = classify_feature_columns(
            split_frames["train"],
            feature_columns,
        )
        baseline_pipeline = build_baseline_pipeline(
            config,
            numeric_features,
            categorical_features,
            random_seed,
        )
        base_lightgbm_params = lightgbm_params(
            config,
            split_frames["train"],
            random_seed,
        )

        x_train = feature_frame(split_frames["train"], feature_columns)
        y_train = split_frames["train"]["TARGET"].astype(int)
        baseline_pipeline.fit(x_train, y_train)
        lightgbm_tuning = fit_tuned_lightgbm(
            config,
            numeric_features,
            categorical_features,
            base_lightgbm_params,
            split_frames,
            feature_columns,
            review_capacity_rate,
            error_cls=TrainingError,
        )
        lightgbm_pipeline = lightgbm_tuning["pipeline"]
        selected_lightgbm_params = lightgbm_tuning["selected_candidate"]["params"]

        split_summary_rows = _build_split_summary(split_frames, run_id, created_at)
        baseline_metric_rows = _build_metric_rows(
            BASELINE_MODEL_VERSION,
            baseline_pipeline,
            split_frames,
            feature_columns,
            created_at,
            review_capacity_rate,
        )
        lightgbm_metric_rows = _build_metric_rows(
            LIGHTGBM_MODEL_VERSION,
            lightgbm_pipeline,
            split_frames,
            feature_columns,
            created_at,
            review_capacity_rate,
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

        ensure_directories(model_dir, report_dir)
        # Downstream evaluation, scoring, and explainability depend on these persisted split IDs.
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
            "lightgbm_params": selected_lightgbm_params,
            "lightgbm_tuning": build_lightgbm_tuning_artifact(lightgbm_tuning),
        }
        joblib.dump(baseline_artifact, model_dir / BASELINE_MODEL_ARTIFACT_NAME)
        joblib.dump(lightgbm_artifact, model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME)

        write_csv(report_dir / "model_run_summary.csv", MODEL_RUN_SUMMARY_COLUMNS, run_summary_rows)
        write_csv(report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS, metric_rows)
        write_csv(report_dir / "split_summary.csv", SPLIT_SUMMARY_COLUMNS, split_summary_rows)
        write_csv(
            report_dir / "lightgbm_tuning_summary.csv",
            LIGHTGBM_TUNING_SUMMARY_COLUMNS,
            tuning_summary_rows,
        )
        write_csv(
            report_dir / "model_comparison_summary.csv",
            MODEL_COMPARISON_SUMMARY_COLUMNS,
            comparison_rows,
        )
        replace_duckdb_table(connection, "model_run_summary", run_summary_rows)
        replace_duckdb_table(connection, "model_metrics_summary", metric_rows)
        replace_duckdb_table(connection, "split_summary", split_summary_rows)
        replace_duckdb_table(connection, "lightgbm_tuning_summary", tuning_summary_rows)
        replace_duckdb_table(connection, "model_comparison_summary", comparison_rows)

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
    pipeline: Any,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    created_at: str,
    manual_review_capacity_rate: float,
) -> list[dict[str, Any]]:
    artifact = {"pipeline": pipeline, "model_version": model_version}
    prediction_frames = {}
    for split_name, frame in split_frames.items():
        probabilities = predict_probabilities(
            artifact,
            frame,
            feature_columns,
            f"{model_version} {split_name}",
            TrainingError,
        )
        prediction_frames[split_name] = prediction_frame(frame, probabilities)
    return build_probability_metric_rows(
        model_version,
        prediction_frames,
        created_at,
        manual_review_capacity_rate,
        error_cls=TrainingError,
    )


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
    selected_model_type = select_model_type_by_validation_pr_auc(
        baseline_validation["pr_auc"],
        lightgbm_validation["pr_auc"],
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
        "data_scope_version": data_scope_version(config),
        "train_rows": split_rows["train"]["row_count"],
        "validation_rows": split_rows["validation"]["row_count"],
        "test_rows": split_rows["test"]["row_count"],
        "feature_count": feature_count,
        "positive_rate_train": split_rows["train"]["positive_rate"],
        "random_seed": random_seed,
        "created_at": created_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and primary credit-risk models.")
    add_config_argument(parser)
    args = parser.parse_args()

    try:
        run_training(args.config)
    except TrainingError as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
