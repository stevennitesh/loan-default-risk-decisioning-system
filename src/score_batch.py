from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.calibrate import CALIBRATION_ARTIFACT_NAME
from src.calibration import apply_saved_calibration_artifact
from src.cli import add_config_argument, exit_with_error
from src.config import DEFAULT_CONFIG_PATH, load_config
from src.mart_access import (
    load_application_test_frame,
    load_labeled_split_frame,
    require_table,
)
from src.metrics import validate_probabilities, with_probability_rank_bin
from src.model_artifacts import (
    load_calibration_artifact,
    load_selected_model_artifact,
    load_selected_model_type,
    normalize_split_ids,
)
from src.model_contracts import (
    LIGHTGBM_MODEL_TYPE,
    MODEL_ARTIFACTS,
    SUPPORTED_MODEL_TYPES,
)
from src.modeling import predict_probabilities
from src.report_contracts import CREDIT_RISK_SCORE_COLUMNS
from src.runtime import (
    current_utc_datetime,
    replace_duckdb_table,
    require_existing_path,
    resolve_config_path,
)
from src.thresholding import BALANCED_SCENARIO, assign_risk_bands

ACTION_LABELS = {
    "approve": ("low_risk", "approve"),
    "manual_review": ("medium_risk", "manual_review"),
    "high_risk": ("high_risk", "high_priority_review"),
}


class ScoringError(RuntimeError):
    """Raised when batch scoring cannot satisfy the Milestone 8 contract."""


def run_scoring(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")

    require_existing_path(duckdb_path, "DuckDB database", ScoringError)

    scored_at = current_utc_datetime()
    with duckdb.connect(str(duckdb_path)) as connection:
        selected_model_type = load_selected_model_type(
            connection,
            SUPPORTED_MODEL_TYPES,
            error_cls=ScoringError,
        )
        artifact = load_selected_model_artifact(
            model_dir,
            selected_model_type,
            MODEL_ARTIFACTS,
            error_cls=ScoringError,
        )
        calibration_artifact = load_calibration_artifact(
            model_dir,
            artifact,
            CALIBRATION_ARTIFACT_NAME,
            LIGHTGBM_MODEL_TYPE,
            error_cls=ScoringError,
        )
        feature_columns = list(artifact["feature_columns"])
        split_applicant_ids = normalize_split_ids(
            artifact["split_applicant_ids"],
            ("test",),
            error_cls=ScoringError,
            label="Selected model artifact split_applicant_ids",
        )
        threshold_policy = _load_balanced_threshold_policy(connection, str(artifact["model_version"]))

        holdout_frame = _load_holdout_test_frame(
            connection,
            split_applicant_ids["test"],
            feature_columns,
        )
        kaggle_frame = load_application_test_frame(connection, feature_columns, error_cls=ScoringError)
        score_rows = [
            *_score_population(
                artifact,
                holdout_frame,
                feature_columns,
                "holdout_test",
                threshold_policy,
                calibration_artifact,
                scored_at,
            ),
            *_score_population(
                artifact,
                kaggle_frame,
                feature_columns,
                "kaggle_test",
                threshold_policy,
                calibration_artifact,
                scored_at,
            ),
        ]
        _validate_output_rows(score_rows)
        replace_duckdb_table(connection, "credit_risk_scores", score_rows, CREDIT_RISK_SCORE_COLUMNS)

    return {
        "row_count": len(score_rows),
        "scoring_populations": sorted({row["scoring_population"] for row in score_rows}),
        "model_version": artifact["model_version"],
        "threshold_version": threshold_policy["threshold_version"],
        "calibration_method": calibration_artifact["selected_method"],
    }


def _load_balanced_threshold_policy(
    connection: duckdb.DuckDBPyConnection,
    model_version: str,
) -> dict[str, Any]:
    require_table(connection, "model_threshold_metrics", error_cls=ScoringError)
    rows = connection.execute(
        """
        SELECT threshold_version, threshold_low, threshold_high
        FROM model_threshold_metrics
        WHERE split = 'validation'
          AND scenario_name = ?
          AND model_version = ?
        """,
        [BALANCED_SCENARIO, model_version],
    ).fetchall()
    if len(rows) != 1:
        raise ScoringError(
            "model_threshold_metrics must contain exactly one validation balanced row "
            f"for model_version={model_version}, got {len(rows)}"
        )
    threshold_version, threshold_low, threshold_high = rows[0]
    return {
        "threshold_version": str(threshold_version),
        "threshold_low": float(threshold_low),
        "threshold_high": float(threshold_high),
    }


def _load_holdout_test_frame(
    connection: duckdb.DuckDBPyConnection,
    applicant_ids: list[int],
    feature_columns: list[str],
) -> pd.DataFrame:
    return load_labeled_split_frame(
        connection,
        applicant_ids,
        feature_columns,
        "holdout_test",
        error_cls=ScoringError,
        missing_context="holdout test split",
    )


def _score_population(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    scoring_population: str,
    threshold_policy: dict[str, Any],
    calibration_artifact: dict[str, Any],
    scored_at: datetime,
) -> list[dict[str, Any]]:
    raw_probabilities = predict_probabilities(
        artifact,
        frame,
        feature_columns,
        scoring_population,
        ScoringError,
    )
    calibrated_probabilities = apply_saved_calibration_artifact(
        raw_probabilities,
        calibration_artifact,
        error_cls=ScoringError,
        label="score calibration",
    )
    validate_probabilities(calibrated_probabilities, f"{scoring_population} calibrated", error_cls=ScoringError)
    risk_actions = assign_risk_bands(raw_probabilities, threshold_policy)
    ranked_frame = pd.DataFrame(
        {
            "SK_ID_CURR": frame["SK_ID_CURR"].astype(int),
            "observed_target": frame["TARGET"],
            "probability": raw_probabilities.astype(float),
            "raw_risk_score": raw_probabilities.astype(float),
            "calibrated_risk_score": calibrated_probabilities.astype(float),
            "risk_action": risk_actions,
        }
    )
    ranked_frame = with_probability_rank_bin(ranked_frame, "score_decile", descending=True)

    rows = []
    for record in ranked_frame.to_dict("records"):
        risk_band, recommended_action = ACTION_LABELS[record["risk_action"]]
        observed_target = record["observed_target"]
        rows.append(
            {
                "applicant_id": int(record["SK_ID_CURR"]),
                "scoring_population": scoring_population,
                "observed_target": None
                if pd.isna(observed_target)
                else int(observed_target),
                "score": float(record["probability"]),
                "raw_risk_score": float(record["raw_risk_score"]),
                "calibrated_risk_score": float(record["calibrated_risk_score"]),
                "calibration_method": calibration_artifact["selected_method"],
                "score_decile": int(record["score_decile"]),
                "risk_band": risk_band,
                "recommended_action": recommended_action,
                "threshold_version": threshold_policy["threshold_version"],
                "model_version": artifact["model_version"],
                "top_reason_1": None,
                "top_reason_2": None,
                "top_reason_3": None,
                "scored_at": scored_at,
            }
        )
    return rows


def _validate_output_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ScoringError("credit_risk_scores output must not be empty")
    frame = pd.DataFrame(rows, columns=CREDIT_RISK_SCORE_COLUMNS)
    duplicate_count = int(
        frame.duplicated(
            subset=["applicant_id", "scoring_population", "model_version", "threshold_version"]
        ).sum()
    )
    if duplicate_count:
        raise ScoringError(f"Duplicate credit_risk_scores output keys: {duplicate_count}")
    if frame["risk_band"].isna().any() or frame["recommended_action"].isna().any():
        raise ScoringError("Every scored row must have risk_band and recommended_action")
    score_columns = ["score", "raw_risk_score", "calibrated_risk_score"]
    if frame[score_columns].isna().any().any():
        raise ScoringError("Every scored row must have raw and calibrated score values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score applicants in batch and write DuckDB score outputs.")
    add_config_argument(parser)
    args = parser.parse_args()

    try:
        run_scoring(args.config)
    except ScoringError as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
