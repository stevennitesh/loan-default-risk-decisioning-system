from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCENARIO_NAMES = ("growth_oriented", "balanced", "risk_averse")
SCENARIO_QUANTILES = {
    "growth_oriented": (0.85, 0.95),
    "balanced": (0.80, 0.90),
    "risk_averse": (0.70, 0.80),
}


class ThresholdingError(RuntimeError):
    """Raised when threshold policy inputs or outputs violate the Milestone 7 contract."""


def validate_threshold_pair(
    threshold_low: Any,
    threshold_high: Any,
    scenario_name: str,
) -> dict[str, float]:
    low = _coerce_threshold(threshold_low, scenario_name, "threshold_low")
    high = _coerce_threshold(threshold_high, scenario_name, "threshold_high")
    if low >= high:
        raise ThresholdingError(
            f"{scenario_name} thresholds must satisfy threshold_low < threshold_high"
        )
    return {"threshold_low": low, "threshold_high": high}


def resolve_scenario_thresholds(
    threshold_policy_config: dict[str, Any],
    validation_probabilities: np.ndarray,
) -> dict[str, dict[str, float]]:
    scenario_config = threshold_policy_config["scenarios"]
    scenario_names = set(scenario_config)
    if scenario_names != set(SCENARIO_NAMES):
        raise ThresholdingError(f"Unexpected threshold scenarios: {sorted(scenario_names)}")

    thresholds = {}
    for scenario_name in SCENARIO_NAMES:
        configured = scenario_config[scenario_name]
        threshold_low = configured["threshold_low"]
        threshold_high = configured["threshold_high"]
        if (threshold_low is None) != (threshold_high is None):
            raise ThresholdingError(
                f"{scenario_name} must configure both thresholds or neither"
            )
        if threshold_low is None:
            low_quantile, high_quantile = SCENARIO_QUANTILES[scenario_name]
            threshold_low, threshold_high = _derive_threshold_pair(
                validation_probabilities,
                low_quantile,
                high_quantile,
            )
        thresholds[scenario_name] = validate_threshold_pair(
            threshold_low,
            threshold_high,
            scenario_name,
        )
    return thresholds


def assign_risk_bands(
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1:
        raise ThresholdingError("Scores must be one-dimensional")
    if not np.isfinite(scores).all():
        raise ThresholdingError("Scores must contain only finite, non-finite values are not allowed")
    if ((scores < 0) | (scores > 1)).any():
        raise ThresholdingError("Scores must be in [0, 1]")

    validated_thresholds = validate_threshold_pair(
        thresholds["threshold_low"],
        thresholds["threshold_high"],
        "scenario",
    )
    threshold_low = validated_thresholds["threshold_low"]
    threshold_high = validated_thresholds["threshold_high"]
    return np.select(
        [
            scores < threshold_low,
            scores < threshold_high,
        ],
        [
            "approve",
            "manual_review",
        ],
        default="high_risk",
    )


def calculate_expected_value(
    approved_good_count: int,
    approved_bad_count: int,
    manual_review_count: int,
    assumptions: dict[str, Any],
) -> float:
    return float(
        approved_good_count * assumptions["expected_margin_per_good_loan"]
        - approved_bad_count * assumptions["expected_loss_per_bad_loan"]
        - manual_review_count * assumptions["manual_review_cost"]
    )


def build_threshold_metric_rows(
    model_version: str,
    threshold_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    scenario_thresholds: dict[str, dict[str, float]],
    assumptions: dict[str, Any],
    created_at: str,
    splits: tuple[str, ...] = ("validation", "test"),
) -> list[dict[str, Any]]:
    rows = []
    for split_name in splits:
        frame = prediction_frames[split_name]
        total_defaults = int(frame["target"].sum())
        for scenario_name in SCENARIO_NAMES:
            thresholds = scenario_thresholds[scenario_name]
            bands = assign_risk_bands(frame["probability"].to_numpy(), thresholds)
            applicant_count = len(frame)
            approve_mask = bands == "approve"
            manual_review_mask = bands == "manual_review"
            high_risk_mask = bands == "high_risk"
            approved_targets = frame.loc[approve_mask, "target"]
            approved_good_count = int((approved_targets == 0).sum())
            approved_bad_count = int((approved_targets == 1).sum())
            manual_review_count = int(manual_review_mask.sum())
            high_risk_count = int(high_risk_mask.sum())
            high_risk_default_count = int(
                ((frame["target"] == 1) & high_risk_mask).sum()
            )
            expected_value = calculate_expected_value(
                approved_good_count,
                approved_bad_count,
                manual_review_count,
                assumptions,
            )
            approved_count = approved_good_count + approved_bad_count
            rows.append(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "threshold_version": threshold_version,
                    "scenario_name": scenario_name,
                    "threshold_low": thresholds["threshold_low"],
                    "threshold_high": thresholds["threshold_high"],
                    "applicant_count": applicant_count,
                    "approval_rate": approved_count / applicant_count,
                    "manual_review_rate": manual_review_count / applicant_count,
                    "high_risk_rate": high_risk_count / applicant_count,
                    "approved_good_count": approved_good_count,
                    "approved_bad_count": approved_bad_count,
                    "manual_review_count": manual_review_count,
                    "high_risk_count": high_risk_count,
                    "default_rate_approved": approved_bad_count / approved_count
                    if approved_count
                    else None,
                    "high_risk_default_capture_rate": high_risk_default_count / total_defaults
                    if total_defaults
                    else None,
                    "expected_value": expected_value,
                    "expected_value_per_applicant": expected_value / applicant_count,
                    "created_at": created_at,
                }
            )
    return rows


def build_confusion_matrix_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    scenario_thresholds: dict[str, dict[str, float]],
    splits: tuple[str, ...] = ("validation", "test"),
) -> list[dict[str, Any]]:
    rows = []
    for split_name in splits:
        frame = prediction_frames[split_name]
        for scenario_name in SCENARIO_NAMES:
            bands = assign_risk_bands(
                frame["probability"].to_numpy(),
                scenario_thresholds[scenario_name],
            )
            predicted_labels = pd.Series(bands == "high_risk", index=frame.index).astype(int)
            rows.extend(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "scenario_name": scenario_name,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": _confusion_count(frame, predicted_labels, true_label, predicted_label),
                }
                for true_label in [0, 1]
                for predicted_label in [0, 1]
            )
    return rows


def _confusion_count(
    frame: pd.DataFrame,
    predicted_labels: pd.Series,
    true_label: int,
    predicted_label: int,
) -> int:
    return int(((frame["target"] == true_label) & (predicted_labels == predicted_label)).sum())


def _derive_threshold_pair(
    probabilities: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[float, float]:
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1:
        raise ThresholdingError("Validation scores must be one-dimensional")
    if not np.isfinite(scores).all():
        raise ThresholdingError("Validation scores must contain only finite values")
    if len(scores) == 0:
        raise ThresholdingError("Validation scores must not be empty")
    unique_probabilities = np.unique(scores)
    if len(unique_probabilities) < 2:
        raise ThresholdingError(
            "Cannot derive scenario thresholds from a constant validation score"
        )

    threshold_low = float(np.quantile(scores, low_quantile))
    threshold_high = float(np.quantile(scores, high_quantile))
    if threshold_low < threshold_high:
        return threshold_low, threshold_high

    low_index = min(
        int(np.floor(low_quantile * (len(unique_probabilities) - 1))),
        len(unique_probabilities) - 2,
    )
    high_index = max(
        int(np.ceil(high_quantile * (len(unique_probabilities) - 1))),
        low_index + 1,
    )
    high_index = min(high_index, len(unique_probabilities) - 1)
    threshold_low = float(unique_probabilities[low_index])
    threshold_high = float(unique_probabilities[high_index])
    if threshold_low >= threshold_high:
        raise ThresholdingError(
            "Could not derive ordered scenario thresholds from validation scores"
        )
    return threshold_low, threshold_high


def _coerce_threshold(value: Any, scenario_name: str, field_name: str) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ThresholdingError(
            f"{scenario_name} {field_name} must be numeric"
        ) from error
    if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
        raise ThresholdingError(f"{scenario_name} {field_name} must be in [0, 1]")
    return threshold
