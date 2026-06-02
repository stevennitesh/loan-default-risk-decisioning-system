from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import joblib

from src.calibration import CALIBRATION_METHODS, UNCALIBRATED_METHOD
from src.mart_access import existing_tables

REQUIRED_MODEL_ARTIFACT_KEYS = {
    "pipeline",
    "model_version",
    "model_type",
    "feature_columns",
    "split_applicant_ids",
}
REQUIRED_CALIBRATION_ARTIFACT_KEYS = {
    "base_model_version",
    "selected_method",
    "calibrators",
}


def load_model_artifact(
    path: Path,
    *,
    expected_model_type: str,
    expected_model_version: str,
    error_cls: type[Exception],
    artifact_label: str = "Model artifact",
    missing_label: str | None = None,
    require_feature_columns: bool = False,
    require_predict_proba: bool = False,
) -> dict[str, Any]:
    """Load and validate a persisted model artifact contract."""
    if not path.exists():
        display_label = missing_label or artifact_label
        raise error_cls(f"Missing {display_label}: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise error_cls(f"{artifact_label} must be a dict: {path}")

    missing_keys = sorted(REQUIRED_MODEL_ARTIFACT_KEYS.difference(artifact))
    if missing_keys:
        raise error_cls(f"{artifact_label} is missing required keys: {missing_keys}")
    if artifact["model_type"] != expected_model_type:
        raise error_cls(
            f"{artifact_label} has model_type={artifact['model_type']}, expected {expected_model_type}"
        )
    if artifact["model_version"] != expected_model_version:
        raise error_cls(
            f"{artifact_label} has model_version={artifact['model_version']}, expected {expected_model_version}"
        )
    if require_feature_columns and not artifact["feature_columns"]:
        raise error_cls(f"{artifact_label} does not contain feature_columns")
    if require_predict_proba and not hasattr(artifact["pipeline"], "predict_proba"):
        raise error_cls(f"{artifact_label} pipeline does not expose predict_proba")
    return artifact


def normalize_split_ids(
    raw_split_ids: Any,
    required_splits: tuple[str, ...],
    *,
    error_cls: type[Exception],
    label: str = "split_applicant_ids",
) -> dict[str, list[int]]:
    """Validate persisted split IDs and normalize them to integer lists."""
    if not isinstance(raw_split_ids, dict):
        raise error_cls(f"{label} must be a mapping")

    missing_splits = [split for split in required_splits if split not in raw_split_ids]
    if missing_splits:
        raise error_cls(f"{label} is missing splits: {missing_splits}")

    split_ids: dict[str, list[int]] = {}
    for split_name in required_splits:
        ids = [int(value) for value in raw_split_ids[split_name]]
        if not ids:
            raise error_cls(f"{label}[{split_name}] must not be empty")
        if len(ids) != len(set(ids)):
            raise error_cls(f"{label}[{split_name}] contains duplicate applicants")
        split_ids[split_name] = ids
    return split_ids


def selected_model_types(connection: Any) -> set[str]:
    """Read the selected model type values from DuckDB, if present."""
    if "model_comparison_summary" not in existing_tables(connection):
        return set()
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT selected_model_type FROM model_comparison_summary"
        ).fetchall()
    }


def load_selected_model_type(
    connection: Any,
    supported_model_types: Collection[str],
    *,
    error_cls: type[Exception],
) -> str:
    """Load the single selected model type from model_comparison_summary."""
    if "model_comparison_summary" not in existing_tables(connection):
        raise error_cls("Missing required DuckDB table: model_comparison_summary")

    selected_values = selected_model_types(connection)
    if len(selected_values) != 1:
        raise error_cls(
            f"model_comparison_summary must contain exactly one selected_model_type, got {sorted(selected_values)}"
        )
    selected_model_type = str(next(iter(selected_values)))
    if selected_model_type not in supported_model_types:
        raise error_cls(f"Unsupported selected_model_type: {selected_model_type}")
    return selected_model_type


def load_selected_model_artifact(
    model_dir: Path,
    selected_model_type: str,
    model_artifacts: dict[str, tuple[str, str]],
    *,
    error_cls: type[Exception],
) -> dict[str, Any]:
    """Load the artifact for the model type selected by evaluation."""
    expected_model_version, artifact_name = model_artifacts[selected_model_type]
    artifact_path = model_dir / artifact_name
    return load_model_artifact(
        artifact_path,
        expected_model_type=selected_model_type,
        expected_model_version=expected_model_version,
        error_cls=error_cls,
        artifact_label="Selected model artifact",
        missing_label="selected model artifact",
        require_feature_columns=True,
    )


def load_calibration_artifact(
    model_dir: Path,
    selected_artifact: dict[str, Any],
    calibration_artifact_name: str,
    calibrated_model_type: str,
    *,
    error_cls: type[Exception],
) -> dict[str, Any]:
    """Load a matching calibration artifact or return the uncalibrated contract."""
    artifact_path = model_dir / calibration_artifact_name
    if (
        selected_artifact["model_type"] != calibrated_model_type
        or not artifact_path.exists()
    ):
        return uncalibrated_calibration_artifact()

    calibration_artifact = joblib.load(artifact_path)
    if not isinstance(calibration_artifact, dict):
        raise error_cls(f"Calibration artifact must be a dict: {artifact_path}")

    missing_keys = sorted(
        REQUIRED_CALIBRATION_ARTIFACT_KEYS.difference(calibration_artifact)
    )
    if missing_keys:
        raise error_cls(
            f"Calibration artifact is missing required keys: {missing_keys}"
        )
    if calibration_artifact["base_model_version"] != selected_artifact["model_version"]:
        raise error_cls(
            "Calibration artifact base_model_version does not match selected model_version: "
            f"{calibration_artifact['base_model_version']} != {selected_artifact['model_version']}"
        )

    selected_method = str(calibration_artifact["selected_method"])
    if selected_method not in CALIBRATION_METHODS:
        raise error_cls(f"Unsupported calibration method: {selected_method}")
    calibrators = calibration_artifact["calibrators"]
    if selected_method != UNCALIBRATED_METHOD and selected_method not in calibrators:
        raise error_cls(
            f"Calibration artifact does not contain selected calibrator: {selected_method}"
        )
    return calibration_artifact


def uncalibrated_calibration_artifact() -> dict[str, Any]:
    """Return the no-op calibration artifact used when calibration is unavailable."""
    return {"selected_method": UNCALIBRATED_METHOD, "calibrators": {}}
