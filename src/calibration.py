from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics import validate_probabilities

CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")
CALIBRATION_FIT_SPLIT = "validation"
CALIBRATION_MIN_BRIER_IMPROVEMENT = 0.0005
SIGMOID_SIMPLICITY_TOLERANCE = 0.0005


def fit_calibrators(
    validation_probabilities: np.ndarray,
    validation_targets: np.ndarray,
    random_seed: int,
    error_cls: type[Exception] = ValueError,
) -> dict[str, Any]:
    validate_probabilities(validation_probabilities, "validation calibration input", error_cls=error_cls)
    target_values = set(int(value) for value in validation_targets)
    if target_values != {0, 1}:
        raise error_cls("Calibration fit split must contain both target classes")

    sigmoid = LogisticRegression(max_iter=1000, random_state=random_seed)
    sigmoid.fit(logit_features(validation_probabilities), validation_targets.astype(int))

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(validation_probabilities, validation_targets.astype(int))
    return {
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }


def apply_calibration_method(
    method: str,
    calibrators: dict[str, Any],
    uncalibrated_predictions: dict[str, pd.DataFrame],
    error_cls: type[Exception] = ValueError,
) -> dict[str, pd.DataFrame]:
    calibrated = {}
    for split_name, frame in uncalibrated_predictions.items():
        probabilities = frame["probability"].to_numpy()
        adjusted_probabilities = apply_calibration_to_probabilities(
            method,
            calibrators,
            probabilities,
            error_cls=error_cls,
            label=f"{method} {split_name}",
        )
        calibrated[split_name] = frame.assign(probability=adjusted_probabilities.astype(float))
    return calibrated


def apply_calibration_to_probabilities(
    method: str,
    calibrators: dict[str, Any],
    probabilities: np.ndarray,
    error_cls: type[Exception] = ValueError,
    label: str | None = None,
) -> np.ndarray:
    if method == "uncalibrated":
        adjusted_probabilities = probabilities
    elif method == "sigmoid":
        adjusted_probabilities = calibrators["sigmoid"].predict_proba(
            logit_features(probabilities),
        )[:, 1]
    elif method == "isotonic":
        adjusted_probabilities = calibrators["isotonic"].predict(probabilities)
    else:
        raise error_cls(f"Unknown calibration method: {method}")

    validate_probabilities(adjusted_probabilities, label or method, error_cls=error_cls)
    return adjusted_probabilities


def select_calibration_method(
    comparison_rows: list[dict[str, Any]],
    error_cls: type[Exception] = ValueError,
) -> str:
    validation_rows = [
        row for row in comparison_rows if row["split"] == CALIBRATION_FIT_SPLIT
    ]
    by_method = {
        str(row["calibration_method"]): float(row["brier_score"])
        for row in validation_rows
    }
    missing_methods = set(CALIBRATION_METHODS).difference(by_method)
    if missing_methods:
        raise error_cls(f"Missing calibration comparison rows for: {sorted(missing_methods)}")

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


def logit_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)
