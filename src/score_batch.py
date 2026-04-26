from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.thresholding import assign_risk_bands
from src.train import BASELINE_MODEL_ARTIFACT_NAME
from src.train import BASELINE_MODEL_TYPE
from src.train import BASELINE_MODEL_VERSION
from src.train import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.train import LIGHTGBM_MODEL_TYPE
from src.train import LIGHTGBM_MODEL_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]

CREDIT_RISK_SCORE_COLUMNS = [
    "applicant_id",
    "scoring_population",
    "observed_target",
    "score",
    "score_decile",
    "risk_band",
    "recommended_action",
    "threshold_version",
    "model_version",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "scored_at",
]

MODEL_ARTIFACTS = {
    BASELINE_MODEL_TYPE: (BASELINE_MODEL_VERSION, BASELINE_MODEL_ARTIFACT_NAME),
    LIGHTGBM_MODEL_TYPE: (LIGHTGBM_MODEL_VERSION, LIGHTGBM_MODEL_ARTIFACT_NAME),
}

ACTION_LABELS = {
    "approve": ("low_risk", "approve"),
    "manual_review": ("medium_risk", "manual_review"),
    "high_risk": ("high_risk", "high_priority_review"),
}


class ScoringError(RuntimeError):
    """Raised when batch scoring cannot satisfy the Milestone 8 contract."""


def run_scoring(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])

    if not duckdb_path.exists():
        raise ScoringError(f"DuckDB database not found: {duckdb_path}")

    scored_at = _scored_at()
    with duckdb.connect(str(duckdb_path)) as connection:
        selected_model_type = _load_selected_model_type(connection)
        artifact = _load_selected_artifact(model_dir, selected_model_type)
        feature_columns = list(artifact["feature_columns"])
        split_applicant_ids = _normalize_split_ids(artifact)
        threshold_policy = _load_balanced_threshold_policy(connection, str(artifact["model_version"]))

        holdout_frame = _load_holdout_test_frame(
            connection,
            split_applicant_ids["test"],
            feature_columns,
        )
        kaggle_frame = _load_kaggle_test_frame(connection, feature_columns)
        score_rows = [
            *_score_population(
                artifact,
                holdout_frame,
                feature_columns,
                "holdout_test",
                threshold_policy,
                scored_at,
            ),
            *_score_population(
                artifact,
                kaggle_frame,
                feature_columns,
                "kaggle_test",
                threshold_policy,
                scored_at,
            ),
        ]
        _validate_output_rows(score_rows)
        _replace_duckdb_table(connection, "credit_risk_scores", score_rows)

    return {
        "row_count": len(score_rows),
        "scoring_populations": sorted({row["scoring_population"] for row in score_rows}),
        "model_version": artifact["model_version"],
        "threshold_version": threshold_policy["threshold_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score applicants in batch and write DuckDB score outputs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    try:
        run_scoring(args.config)
    except ScoringError as error:
        raise SystemExit(str(error)) from error


def _load_selected_model_type(connection: duckdb.DuckDBPyConnection) -> str:
    _require_table(connection, "model_comparison_summary")
    selected_values = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT selected_model_type FROM model_comparison_summary"
        ).fetchall()
    }
    if len(selected_values) != 1:
        raise ScoringError(
            f"model_comparison_summary must contain exactly one selected_model_type, got {sorted(selected_values)}"
        )
    selected_model_type = next(iter(selected_values))
    if selected_model_type not in MODEL_ARTIFACTS:
        raise ScoringError(f"Unsupported selected_model_type: {selected_model_type}")
    return str(selected_model_type)


def _load_selected_artifact(model_dir: Path, selected_model_type: str) -> dict[str, Any]:
    expected_model_version, artifact_name = MODEL_ARTIFACTS[selected_model_type]
    artifact_path = model_dir / artifact_name
    if not artifact_path.exists():
        raise ScoringError(f"Missing selected model artifact: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ScoringError(f"Selected model artifact must be a dict: {artifact_path}")

    required_keys = {
        "pipeline",
        "model_version",
        "model_type",
        "feature_columns",
        "split_applicant_ids",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise ScoringError(f"Selected model artifact is missing required keys: {missing_keys}")
    if artifact["model_type"] != selected_model_type:
        raise ScoringError(
            f"Selected artifact model_type={artifact['model_type']}, expected {selected_model_type}"
        )
    if artifact["model_version"] != expected_model_version:
        raise ScoringError(
            f"Selected artifact model_version={artifact['model_version']}, expected {expected_model_version}"
        )
    if not artifact["feature_columns"]:
        raise ScoringError("Selected model artifact does not contain feature_columns")
    return artifact


def _normalize_split_ids(artifact: dict[str, Any]) -> dict[str, list[int]]:
    raw_split_ids = artifact["split_applicant_ids"]
    if not isinstance(raw_split_ids, dict) or "test" not in raw_split_ids:
        raise ScoringError("Selected model artifact must contain split_applicant_ids['test']")
    test_ids = [int(value) for value in raw_split_ids["test"]]
    if not test_ids:
        raise ScoringError("Selected model artifact split_applicant_ids['test'] must not be empty")
    if len(test_ids) != len(set(test_ids)):
        raise ScoringError("Selected model artifact split_applicant_ids['test'] contains duplicate applicants")
    return {"test": test_ids}


def _load_balanced_threshold_policy(
    connection: duckdb.DuckDBPyConnection,
    model_version: str,
) -> dict[str, Any]:
    _require_table(connection, "model_threshold_metrics")
    rows = connection.execute(
        """
        SELECT threshold_version, threshold_low, threshold_high
        FROM model_threshold_metrics
        WHERE split = 'validation'
          AND scenario_name = 'balanced'
          AND model_version = ?
        """,
        [model_version],
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
    _require_table(connection, "mart_credit_risk_features")
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
    connection.register("holdout_ids", ids_frame)
    try:
        frame = connection.execute(
            f"""
            SELECT {", ".join(_sql_identifier(column) for column in selected_columns)}
            FROM mart_credit_risk_features
            INNER JOIN holdout_ids USING (SK_ID_CURR)
            WHERE source_population = 'application_train'
            ORDER BY SK_ID_CURR
            """
        ).fetch_df()
    finally:
        connection.unregister("holdout_ids")

    if len(frame) != len(applicant_ids):
        found_ids = set(frame["SK_ID_CURR"].astype(int).tolist()) if not frame.empty else set()
        missing_ids = sorted(set(applicant_ids).difference(found_ids))
        raise ScoringError(
            "Saved holdout test split IDs no longer reconcile to mart_credit_risk_features: "
            f"missing {missing_ids[:10]}"
        )
    if frame["TARGET"].isna().any():
        raise ScoringError("holdout_test rows must have observed TARGET values")
    return frame.reset_index(drop=True)


def _load_kaggle_test_frame(
    connection: duckdb.DuckDBPyConnection,
    feature_columns: list[str],
) -> pd.DataFrame:
    _require_table(connection, "mart_credit_risk_features")
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    frame = connection.execute(
        f"""
        SELECT {", ".join(_sql_identifier(column) for column in selected_columns)}
        FROM mart_credit_risk_features
        WHERE source_population = 'application_test'
        ORDER BY SK_ID_CURR
        """
    ).fetch_df()
    if frame.empty:
        raise ScoringError("No application_test rows are available for kaggle_test scoring")
    if frame["TARGET"].notna().any():
        raise ScoringError("kaggle_test rows must have NULL TARGET values")
    return frame.reset_index(drop=True)


def _score_population(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    scoring_population: str,
    threshold_policy: dict[str, Any],
    scored_at: datetime,
) -> list[dict[str, Any]]:
    probabilities = artifact["pipeline"].predict_proba(_feature_frame(frame, feature_columns))[:, 1]
    _validate_scores(probabilities, scoring_population)
    risk_actions = assign_risk_bands(probabilities, threshold_policy)
    ranked_frame = pd.DataFrame(
        {
            "applicant_id": frame["SK_ID_CURR"].astype(int),
            "observed_target": frame["TARGET"],
            "score": probabilities.astype(float),
            "risk_action": risk_actions,
        }
    )
    ranked_frame["score_decile"] = _score_deciles(ranked_frame)

    rows = []
    for record in ranked_frame.to_dict("records"):
        risk_band, recommended_action = ACTION_LABELS[record["risk_action"]]
        observed_target = record["observed_target"]
        rows.append(
            {
                "applicant_id": int(record["applicant_id"]),
                "scoring_population": scoring_population,
                "observed_target": None
                if pd.isna(observed_target)
                else int(observed_target),
                "score": float(record["score"]),
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


def _score_deciles(frame: pd.DataFrame) -> pd.Series:
    ranked = frame.sort_values(
        ["score", "applicant_id"],
        ascending=[False, True],
    ).reset_index()
    ranked["score_decile"] = np.ceil((np.arange(len(ranked)) + 1) * 10 / len(ranked)).astype(int)
    ranked["score_decile"] = ranked["score_decile"].clip(1, 10)
    return ranked.set_index("index").sort_index()["score_decile"]


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


def _validate_scores(probabilities: np.ndarray, scoring_population: str) -> None:
    if probabilities.ndim != 1:
        raise ScoringError(f"{scoring_population} probabilities must be one-dimensional")
    if not np.isfinite(probabilities).all():
        raise ScoringError(f"{scoring_population} probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ScoringError(f"{scoring_population} probabilities must be in [0, 1]")


def _feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = frame[feature_columns].copy()
    return features.where(pd.notna(features), np.nan)


def _replace_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(rows, columns=CREDIT_RISK_SCORE_COLUMNS)
    connection.register("output_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {_sql_identifier(table_name)} AS SELECT * FROM output_frame")
    connection.unregister("output_frame")


def _require_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    if table_name not in _existing_tables(connection):
        raise ScoringError(f"Missing required DuckDB table: {table_name}")


def _existing_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _scored_at() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


if __name__ == "__main__":
    main()
