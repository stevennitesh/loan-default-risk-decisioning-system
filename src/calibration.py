from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics import target_class_values, validate_probabilities

UNCALIBRATED_METHOD = "uncalibrated"
SIGMOID_METHOD = "sigmoid"
ISOTONIC_METHOD = "isotonic"
CALIBRATION_METHODS = (UNCALIBRATED_METHOD, SIGMOID_METHOD, ISOTONIC_METHOD)
CALIBRATION_FIT_SPLIT = "calibration"
CALIBRATION_SELECTION_SPLIT = "validation"
CALIBRATION_MIN_BRIER_IMPROVEMENT = 0.0005
SIGMOID_SIMPLICITY_TOLERANCE = 0.0005


def fit_calibrators(
    validation_probabilities: np.ndarray,
    validation_targets: np.ndarray,
    random_seed: int,
    error_cls: type[Exception] = ValueError,
) -> dict[str, Any]:
    """Fit calibrators on reserved predictions, separate from method selection."""
    validate_probabilities(
        validation_probabilities, "reserved calibration input", error_cls=error_cls
    )
    target_values = target_class_values(validation_targets, error_cls=error_cls)
    if target_values != {0, 1}:
        raise error_cls("Calibration fit split must contain both target classes")

    sigmoid = LogisticRegression(max_iter=1000, random_state=random_seed)
    sigmoid.fit(
        logit_features(validation_probabilities), validation_targets.astype(int)
    )

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(validation_probabilities, validation_targets.astype(int))
    return {
        SIGMOID_METHOD: sigmoid,
        ISOTONIC_METHOD: isotonic,
    }


def apply_calibration_method(
    method: str,
    calibrators: dict[str, Any],
    uncalibrated_predictions: dict[str, pd.DataFrame],
    error_cls: type[Exception] = ValueError,
) -> dict[str, pd.DataFrame]:
    """Apply one calibration method to every split prediction frame."""
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
        calibrated[split_name] = frame.assign(
            probability=adjusted_probabilities.astype(float)
        )
    return calibrated


def apply_calibration_to_probabilities(
    method: str,
    calibrators: dict[str, Any],
    probabilities: np.ndarray,
    error_cls: type[Exception] = ValueError,
    label: str | None = None,
) -> np.ndarray:
    """Apply a configured calibration method to a probability vector."""
    if method == UNCALIBRATED_METHOD:
        adjusted_probabilities = probabilities
    elif method == SIGMOID_METHOD:
        adjusted_probabilities = calibrators[SIGMOID_METHOD].predict_proba(
            logit_features(probabilities),
        )[:, 1]
    elif method == ISOTONIC_METHOD:
        adjusted_probabilities = calibrators[ISOTONIC_METHOD].predict(probabilities)
    else:
        raise error_cls(f"Unknown calibration method: {method}")

    validate_probabilities(adjusted_probabilities, label or method, error_cls=error_cls)
    return adjusted_probabilities


def apply_saved_calibration_artifact(
    probabilities: np.ndarray,
    calibration_artifact: dict[str, Any],
    error_cls: type[Exception] = ValueError,
    label: str = "saved calibration",
) -> np.ndarray:
    """Apply the selected method from a saved calibration artifact."""
    return apply_calibration_to_probabilities(
        str(calibration_artifact["selected_method"]),
        calibration_artifact["calibrators"],
        probabilities,
        error_cls=error_cls,
        label=label,
    ).astype(float)


def select_calibration_method(
    comparison_rows: list[dict[str, Any]],
    error_cls: type[Exception] = ValueError,
) -> str:
    """Select calibration by validation Brier improvement and simplicity rules."""
    validation_rows = [
        row for row in comparison_rows if row["split"] == CALIBRATION_SELECTION_SPLIT
    ]
    by_method = {
        str(row["calibration_method"]): float(row["brier_score"])
        for row in validation_rows
    }
    if len(by_method) != len(validation_rows) or set(by_method) != set(
        CALIBRATION_METHODS
    ):
        raise error_cls(
            "Calibration comparison requires exactly one row per known method"
        )
    if any(
        not np.isfinite(brier) or not 0 <= brier <= 1 for brier in by_method.values()
    ):
        raise error_cls(
            "Calibration comparison Brier scores must be finite and in [0, 1]"
        )
    uncalibrated_brier = by_method[UNCALIBRATED_METHOD]
    eligible = {
        method: brier
        for method, brier in by_method.items()
        if method != UNCALIBRATED_METHOD
        and uncalibrated_brier - brier >= CALIBRATION_MIN_BRIER_IMPROVEMENT - 1e-12
    }
    if not eligible:
        return UNCALIBRATED_METHOD
    best_method = min(eligible, key=eligible.get)
    best_brier = eligible[best_method]
    if (
        SIGMOID_METHOD in eligible
        and eligible[SIGMOID_METHOD] - best_brier
        <= SIGMOID_SIMPLICITY_TOLERANCE + 1e-12
    ):
        return SIGMOID_METHOD
    return best_method


def logit_features(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to clipped logit features for sigmoid calibration."""
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)
