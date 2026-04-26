from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
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
MODEL_VERSION = "logistic_regression_baseline_v1"
MODEL_TYPE = "logistic_regression"
MODEL_ARTIFACT_NAME = "logistic_regression_baseline.joblib"

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


class TrainingError(RuntimeError):
    """Raised when baseline training cannot satisfy the Milestone 4 contract."""


def run_training(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])
    report_dir = _resolve_project_path(config["paths"]["report_dir"])

    if not duckdb_path.exists():
        raise TrainingError(f"DuckDB database not found: {duckdb_path}")

    created_at = _created_at()
    run_id = f"{MODEL_VERSION}_{created_at.replace('-', '').replace(':', '').replace('Z', '')}"
    random_seed = int(config["project"]["random_seed"])

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
        pipeline = _build_pipeline(config, numeric_features, categorical_features, random_seed)

        x_train = _feature_frame(split_frames["train"], feature_columns)
        y_train = split_frames["train"]["TARGET"].astype(int)
        pipeline.fit(x_train, y_train)

        split_summary_rows = _build_split_summary(split_frames, run_id, created_at)
        metric_rows = _build_metric_rows(pipeline, split_frames, feature_columns, created_at)
        run_summary_rows = [
            _build_run_summary_row(
                config,
                run_id,
                split_summary_rows,
                len(feature_columns),
                created_at,
                random_seed,
            )
        ]

        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "pipeline": pipeline,
            "model_version": MODEL_VERSION,
            "run_id": run_id,
            "model_type": MODEL_TYPE,
            "feature_columns": feature_columns,
            "numeric_feature_columns": numeric_features,
            "categorical_feature_columns": categorical_features,
            "split_config": config["split"],
            "split_summary": split_summary_rows,
            "split_applicant_ids": {
                split_name: [int(value) for value in frame["SK_ID_CURR"].tolist()]
                for split_name, frame in split_frames.items()
            },
            "metric_rows": metric_rows,
            "created_at": created_at,
        }
        joblib.dump(artifact, model_dir / MODEL_ARTIFACT_NAME)

        _write_csv(report_dir / "model_run_summary.csv", MODEL_RUN_SUMMARY_COLUMNS, run_summary_rows)
        _write_csv(report_dir / "model_metrics_summary.csv", MODEL_METRICS_SUMMARY_COLUMNS, metric_rows)
        _write_csv(report_dir / "split_summary.csv", SPLIT_SUMMARY_COLUMNS, split_summary_rows)
        _replace_duckdb_table(connection, "model_run_summary", run_summary_rows)
        _replace_duckdb_table(connection, "model_metrics_summary", metric_rows)
        _replace_duckdb_table(connection, "split_summary", split_summary_rows)

    return {
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "model_type": MODEL_TYPE,
        "feature_columns": feature_columns,
        "run_summary": run_summary_rows,
        "metric_rows": metric_rows,
        "split_summary": split_summary_rows,
        "artifact_path": model_dir / MODEL_ARTIFACT_NAME,
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


def _build_pipeline(
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
                "model_version": MODEL_VERSION,
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
    pipeline: Pipeline,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    created_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, frame in split_frames.items():
        y_true = frame["TARGET"].astype(int)
        probabilities = pipeline.predict_proba(_feature_frame(frame, feature_columns))[:, 1]
        metrics = {
            "roc_auc": roc_auc_score(y_true, probabilities),
            "pr_auc": average_precision_score(y_true, probabilities),
            "brier_score": brier_score_loss(y_true, probabilities),
            "min_predicted_probability": float(np.min(probabilities)),
            "max_predicted_probability": float(np.max(probabilities)),
        }
        rows.extend(
            {
                "model_version": MODEL_VERSION,
                "split": split_name,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "created_at": created_at,
            }
            for metric_name, metric_value in metrics.items()
        )
    return rows


def _build_run_summary_row(
    config: dict[str, Any],
    run_id: str,
    split_summary_rows: list[dict[str, Any]],
    feature_count: int,
    created_at: str,
    random_seed: int,
) -> dict[str, Any]:
    split_rows = {row["split"]: row for row in split_summary_rows}
    return {
        "model_version": MODEL_VERSION,
        "run_id": run_id,
        "model_type": MODEL_TYPE,
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
