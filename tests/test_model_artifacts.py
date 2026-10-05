from pathlib import Path

import joblib
import numpy as np
import pytest

from src.model_artifacts import (
    load_calibration_artifact,
    load_model_artifact,
    normalize_split_ids,
)


def test_split_ids_normalize_integer_strings_without_changing_identity() -> None:
    assert normalize_split_ids(
        {"train": [np.int64(1), "2"], "validation": [3], "test": [4]},
        ("validation", "test"),
        error_cls=RuntimeError,
    ) == {"validation": [3], "test": [4]}


@pytest.mark.parametrize("invalid_id", [1.9, True, None, float("nan"), "1.9"])
def test_split_ids_reject_values_that_cannot_represent_integer_ids(invalid_id) -> None:
    with pytest.raises(RuntimeError, match="integer applicant IDs"):
        normalize_split_ids({"test": [invalid_id]}, ("test",), error_cls=RuntimeError)


@pytest.mark.parametrize("invalid_ids", [None, 1, "123", {"1": 2}])
def test_split_ids_reject_non_sequence_containers(invalid_ids) -> None:
    with pytest.raises(RuntimeError, match="sequence"):
        normalize_split_ids({"test": invalid_ids}, ("test",), error_cls=RuntimeError)


def test_split_ids_check_overlap_with_splits_outside_the_requested_subset() -> None:
    with pytest.raises(RuntimeError, match="overlap"):
        normalize_split_ids(
            {"train": [1, 2], "validation": [3], "test": [2, 4]},
            ("test",),
            error_cls=RuntimeError,
        )


def test_model_loading_rejects_overlapping_manifest(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    artifact = _model_artifact()
    artifact["split_applicant_ids"]["test"] = [1]
    joblib.dump(artifact, path)
    with pytest.raises(RuntimeError, match="overlap"):
        load_model_artifact(
            path,
            expected_model_type="lightgbm",
            expected_model_version="same_display_version",
            error_cls=RuntimeError,
        )


@pytest.mark.parametrize("parent_run", ["older_fit", None])
def test_calibration_rejects_wrong_or_missing_fitted_parent(
    tmp_path: Path, parent_run: str | None
) -> None:
    calibration = {
        "base_model_version": "same_display_version",
        "calibration_run_id": "current_calibration",
        "selected_method": "sigmoid",
        "calibrators": {"sigmoid": object()},
    }
    if parent_run is not None:
        calibration["base_model_run_id"] = parent_run
    joblib.dump(calibration, tmp_path / "calibration.joblib")
    with pytest.raises(RuntimeError, match="base_model_run_id"):
        load_calibration_artifact(
            tmp_path,
            _model_artifact(),
            "calibration.joblib",
            "lightgbm",
            error_cls=RuntimeError,
        )


def test_calibration_accepts_its_fitted_parent(tmp_path: Path) -> None:
    calibration = {
        "base_model_version": "same_display_version",
        "calibration_run_id": "current_calibration",
        "base_model_run_id": "current_fit",
        "selected_method": "uncalibrated",
        "calibrators": {},
    }
    joblib.dump(calibration, tmp_path / "calibration.joblib")
    assert (
        load_calibration_artifact(
            tmp_path,
            _model_artifact(),
            "calibration.joblib",
            "lightgbm",
            error_cls=RuntimeError,
        )
        == calibration
    )


def _model_artifact() -> dict:
    return {
        "pipeline": None,
        "model_type": "lightgbm",
        "model_version": "same_display_version",
        "run_id": "current_fit",
        "feature_columns": ["feature"],
        "split_applicant_ids": {"train": [1, 2], "validation": [3], "test": [4]},
    }
