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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.evaluate import EVALUATION_SPLITS
from src.evaluate import REPORTING_SPLITS
from src.train import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.train import LIGHTGBM_MODEL_TYPE
from src.train import LIGHTGBM_MODEL_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ARTIFACT_NAME = "lightgbm_credit_risk_calibration.joblib"
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")
CALIBRATION_FIT_SPLIT = "validation"
CALIBRATION_MIN_BRIER_IMPROVEMENT = 0.0005
SIGMOID_SIMPLICITY_TOLERANCE = 0.0005
MODEL_CALIBRATION_COMPARISON_COLUMNS = [
    "model_version",
    "base_model_version",
    "calibration_method",
    "split",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "min_predicted_probability",
    "max_predicted_probability",
    "top_decile_lift",
    "precision_at_top_decile",
    "recall_at_manual_review_capacity",
    "mean_absolute_bin_error",
    "weighted_calibration_error",
    "max_absolute_bin_error",
    "created_at",
]
MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS = [
    "model_version",
    "base_model_version",
    "calibration_method",
    "split",
    "bin_id",
    "applicant_count",
    "average_predicted_score",
    "observed_default_rate",
    "calibration_error",
    "created_at",
]


class CalibrationError(RuntimeError):
    """Raised when the post-v1 calibration experiment cannot run safely."""


def run_calibration_experiment(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])
    report_dir = _resolve_project_path(config["paths"]["report_dir"])

    if not duckdb_path.exists():
        raise CalibrationError(f"DuckDB database not found: {duckdb_path}")

    artifact_path = model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME
    if not artifact_path.exists():
        raise CalibrationError(f"Missing LightGBM model artifact: {artifact_path}")

    artifact = joblib.load(artifact_path)
    _validate_lightgbm_artifact(artifact, artifact_path)

    feature_columns = list(artifact["feature_columns"])
    split_applicant_ids = _normalize_split_ids(artifact["split_applicant_ids"])
    created_at = _created_at()
    manual_review_capacity_rate = float(config["business_assumptions"]["manual_review_capacity_rate"])

    with duckdb.connect(str(duckdb_path)) as connection:
        split_frames = _load_split_frames(connection, split_applicant_ids, feature_columns)
        uncalibrated_predictions = _build_uncalibrated_predictions(
            artifact,
            split_frames,
            feature_columns,
        )
        calibrators = _fit_calibrators(
            uncalibrated_predictions[CALIBRATION_FIT_SPLIT]["probability"].to_numpy(),
            uncalibrated_predictions[CALIBRATION_FIT_SPLIT]["target"].to_numpy(),
            int(config["project"]["random_seed"]),
        )
        calibrated_predictions = {
            method: _apply_calibration_method(method, calibrators, uncalibrated_predictions)
            for method in CALIBRATION_METHODS
        }
        comparison_rows, bin_rows = _build_comparison_outputs(
            calibrated_predictions,
            manual_review_capacity_rate,
            created_at,
        )
        selected_method = _select_calibration_method(comparison_rows)

        report_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            report_dir / "model_calibration_comparison.csv",
            MODEL_CALIBRATION_COMPARISON_COLUMNS,
            comparison_rows,
        )
        _write_csv(
            report_dir / "model_calibration_bins_comparison.csv",
            MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS,
            bin_rows,
        )
        _replace_duckdb_table(connection, "model_calibration_comparison", comparison_rows)
        _replace_duckdb_table(connection, "model_calibration_bins_comparison", bin_rows)

    calibration_artifact = {
        "base_model_version": LIGHTGBM_MODEL_VERSION,
        "base_model_type": LIGHTGBM_MODEL_TYPE,
        "calibration_fit_split": CALIBRATION_FIT_SPLIT,
        "selected_method": selected_method,
        "selection_rule": (
            "Require at least 0.0005 validation Brier improvement over uncalibrated scores; "
            "prefer sigmoid when it is within 0.0005 Brier of isotonic because it is simpler and "
            "rank-preserving. Test metrics are held out for reporting only."
        ),
        "calibrators": calibrators,
        "fit_applicant_ids": split_applicant_ids[CALIBRATION_FIT_SPLIT],
        "split_applicant_ids": split_applicant_ids,
        "feature_columns": feature_columns,
        "comparison_rows": comparison_rows,
        "created_at": created_at,
    }
    joblib.dump(calibration_artifact, model_dir / CALIBRATION_ARTIFACT_NAME)

    return {
        "selected_method": selected_method,
        "comparison_rows": comparison_rows,
        "bin_rows": bin_rows,
        "artifact": model_dir / CALIBRATION_ARTIFACT_NAME,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-v1 probability calibration comparison for the LightGBM model.",
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    try:
        run_calibration_experiment(args.config)
    except CalibrationError as error:
        raise SystemExit(str(error)) from error


def _validate_lightgbm_artifact(artifact: dict[str, Any], artifact_path: Path) -> None:
    if not isinstance(artifact, dict):
        raise CalibrationError(f"LightGBM artifact must be a dict: {artifact_path}")
    required_keys = {
        "pipeline",
        "model_version",
        "model_type",
        "feature_columns",
        "split_applicant_ids",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise CalibrationError(f"LightGBM artifact is missing required keys: {missing_keys}")
    if artifact["model_version"] != LIGHTGBM_MODEL_VERSION:
        raise CalibrationError(
            f"LightGBM artifact has model_version={artifact['model_version']}, "
            f"expected {LIGHTGBM_MODEL_VERSION}"
        )
    if artifact["model_type"] != LIGHTGBM_MODEL_TYPE:
        raise CalibrationError(
            f"LightGBM artifact has model_type={artifact['model_type']}, expected {LIGHTGBM_MODEL_TYPE}"
        )
    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise CalibrationError("LightGBM artifact pipeline does not expose predict_proba")


def _normalize_split_ids(raw_split_ids: Any) -> dict[str, list[int]]:
    if not isinstance(raw_split_ids, dict):
        raise CalibrationError("split_applicant_ids must be a mapping")
    missing_splits = [split for split in EVALUATION_SPLITS if split not in raw_split_ids]
    if missing_splits:
        raise CalibrationError(f"split_applicant_ids is missing splits: {missing_splits}")

    split_ids: dict[str, list[int]] = {}
    for split_name in EVALUATION_SPLITS:
        ids = [int(value) for value in raw_split_ids[split_name]]
        if not ids:
            raise CalibrationError(f"split_applicant_ids[{split_name}] must not be empty")
        if len(ids) != len(set(ids)):
            raise CalibrationError(f"split_applicant_ids[{split_name}] contains duplicate applicants")
        split_ids[split_name] = ids
    return split_ids


def _load_split_frames(
    connection: duckdb.DuckDBPyConnection,
    split_applicant_ids: dict[str, list[int]],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    split_frames = {}

    for split_name, applicant_ids in split_applicant_ids.items():
        ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
        connection.register("split_ids", ids_frame)
        try:
            frame = connection.execute(
                f"""
                SELECT {", ".join(_sql_identifier(column) for column in selected_columns)}
                FROM mart_credit_risk_features
                INNER JOIN split_ids USING (SK_ID_CURR)
                WHERE source_population = 'application_train'
                ORDER BY SK_ID_CURR
                """
            ).fetch_df()
        finally:
            connection.unregister("split_ids")

        if len(frame) != len(applicant_ids):
            raise CalibrationError(f"Saved split IDs no longer reconcile for {split_name}")
        target_values = set(frame["TARGET"].astype(int).unique())
        if target_values != {0, 1}:
            raise CalibrationError(f"{split_name} split must contain binary TARGET classes")
        split_frames[split_name] = frame.reset_index(drop=True)
    return split_frames


def _build_uncalibrated_predictions(
    artifact: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, pd.DataFrame]:
    pipeline = artifact["pipeline"]
    prediction_frames = {}
    for split_name, frame in split_frames.items():
        probabilities = pipeline.predict_proba(_feature_frame(frame, feature_columns))[:, 1]
        _validate_probabilities(probabilities, f"{LIGHTGBM_MODEL_VERSION}_{split_name}")
        prediction_frames[split_name] = pd.DataFrame(
            {
                "SK_ID_CURR": frame["SK_ID_CURR"].astype(int),
                "target": frame["TARGET"].astype(int),
                "probability": probabilities.astype(float),
            }
        )
    return prediction_frames


def _fit_calibrators(
    validation_probabilities: np.ndarray,
    validation_targets: np.ndarray,
    random_seed: int,
) -> dict[str, Any]:
    _validate_probabilities(validation_probabilities, "validation calibration input")
    target_values = set(int(value) for value in validation_targets)
    if target_values != {0, 1}:
        raise CalibrationError("Calibration fit split must contain both target classes")

    sigmoid = LogisticRegression(max_iter=1000, random_state=random_seed)
    sigmoid.fit(_logit_features(validation_probabilities), validation_targets.astype(int))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(validation_probabilities, validation_targets.astype(int))
    return {
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }


def _apply_calibration_method(
    method: str,
    calibrators: dict[str, Any],
    uncalibrated_predictions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    calibrated = {}
    for split_name, frame in uncalibrated_predictions.items():
        probabilities = frame["probability"].to_numpy()
        if method == "uncalibrated":
            adjusted_probabilities = probabilities
        elif method == "sigmoid":
            adjusted_probabilities = calibrators["sigmoid"].predict_proba(
                _logit_features(probabilities),
            )[:, 1]
        elif method == "isotonic":
            adjusted_probabilities = calibrators["isotonic"].predict(probabilities)
        else:
            raise CalibrationError(f"Unknown calibration method: {method}")
        _validate_probabilities(adjusted_probabilities, f"{method} {split_name}")
        calibrated[split_name] = frame.assign(probability=adjusted_probabilities.astype(float))
    return calibrated


def _build_comparison_outputs(
    predictions_by_method: dict[str, dict[str, pd.DataFrame]],
    manual_review_capacity_rate: float,
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_rows: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []
    for method, split_predictions in predictions_by_method.items():
        model_version = f"{LIGHTGBM_MODEL_VERSION}_{method}"
        bin_rows = _build_bin_rows(model_version, method, split_predictions, created_at)
        all_bin_rows.extend(bin_rows)
        bin_errors = _bin_error_summary(bin_rows)
        for split_name in REPORTING_SPLITS:
            frame = split_predictions[split_name]
            probabilities = frame["probability"].to_numpy()
            y_true = frame["target"]
            split_bin_errors = bin_errors[split_name]
            comparison_rows.append(
                {
                    "model_version": model_version,
                    "base_model_version": LIGHTGBM_MODEL_VERSION,
                    "calibration_method": method,
                    "split": split_name,
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
                    "mean_absolute_bin_error": split_bin_errors["mean_absolute_bin_error"],
                    "weighted_calibration_error": split_bin_errors["weighted_calibration_error"],
                    "max_absolute_bin_error": split_bin_errors["max_absolute_bin_error"],
                    "created_at": created_at,
                }
            )
    return comparison_rows, all_bin_rows


def _build_bin_rows(
    model_version: str,
    method: str,
    prediction_frames: dict[str, pd.DataFrame],
    created_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in REPORTING_SPLITS:
        frame = _with_score_bin(prediction_frames[split_name], "bin_id")
        for bin_id in range(1, 11):
            bin_frame = frame.loc[frame["bin_id"] == bin_id]
            average_predicted_score = _nullable_mean(bin_frame["probability"])
            observed_default_rate = _nullable_mean(bin_frame["target"])
            rows.append(
                {
                    "model_version": model_version,
                    "base_model_version": LIGHTGBM_MODEL_VERSION,
                    "calibration_method": method,
                    "split": split_name,
                    "bin_id": bin_id,
                    "applicant_count": len(bin_frame),
                    "average_predicted_score": average_predicted_score,
                    "observed_default_rate": observed_default_rate,
                    "calibration_error": observed_default_rate - average_predicted_score
                    if observed_default_rate is not None and average_predicted_score is not None
                    else None,
                    "created_at": created_at,
                }
            )
    return rows


def _bin_error_summary(bin_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for split_name in REPORTING_SPLITS:
        split_rows = [
            row
            for row in bin_rows
            if row["split"] == split_name and row["calibration_error"] is not None
        ]
        total_count = sum(int(row["applicant_count"]) for row in split_rows)
        absolute_errors = [abs(float(row["calibration_error"])) for row in split_rows]
        weighted_error = (
            sum(abs(float(row["calibration_error"])) * int(row["applicant_count"]) for row in split_rows)
            / total_count
            if total_count
            else 0.0
        )
        summaries[split_name] = {
            "mean_absolute_bin_error": float(np.mean(absolute_errors)) if absolute_errors else 0.0,
            "weighted_calibration_error": float(weighted_error),
            "max_absolute_bin_error": float(np.max(absolute_errors)) if absolute_errors else 0.0,
        }
    return summaries


def _select_calibration_method(comparison_rows: list[dict[str, Any]]) -> str:
    validation_rows = [
        row for row in comparison_rows if row["split"] == CALIBRATION_FIT_SPLIT
    ]
    by_method = {
        str(row["calibration_method"]): float(row["brier_score"])
        for row in validation_rows
    }
    uncalibrated_brier = by_method["uncalibrated"]
    best_method = min(by_method, key=by_method.get)
    best_brier = by_method[best_method]

    if uncalibrated_brier - best_brier < CALIBRATION_MIN_BRIER_IMPROVEMENT:
        return "uncalibrated"
    if (
        "sigmoid" in by_method
        and by_method["sigmoid"] - best_brier <= SIGMOID_SIMPLICITY_TOLERANCE
    ):
        return "sigmoid"
    return best_method


def _feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    return features.where(pd.notna(features), np.nan)


def _logit_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


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
        raise CalibrationError(f"Selection rate must be in (0, 1], got {rate}")
    return max(1, int(np.ceil(row_count * rate)))


def _with_score_bin(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["probability", "SK_ID_CURR"],
        ascending=[True, True],
    ).reset_index(drop=True)
    ranked[column_name] = np.ceil((np.arange(len(ranked)) + 1) * 10 / len(ranked)).astype(int)
    ranked[column_name] = ranked[column_name].clip(1, 10)
    return ranked


def _nullable_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.mean())


def _validate_probabilities(probabilities: np.ndarray, label: str) -> None:
    if probabilities.ndim != 1:
        raise CalibrationError(f"{label} probabilities must be one-dimensional")
    if not np.isfinite(probabilities).all():
        raise CalibrationError(f"{label} probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise CalibrationError(f"{label} probabilities must be in [0, 1]")


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
