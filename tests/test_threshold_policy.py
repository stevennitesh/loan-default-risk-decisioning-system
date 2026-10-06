from __future__ import annotations

import numpy as np
import pytest

from src.thresholding import (
    SCENARIO_NAMES,
    ThresholdingError,
    assign_risk_bands,
    resolve_scenario_thresholds,
    validate_threshold_pair,
)


def test_assign_risk_bands_uses_documented_boundaries() -> None:
    bands = assign_risk_bands(
        np.array([0.10, 0.30, 0.50, 0.70, 0.90]),
        {"threshold_low": 0.30, "threshold_high": 0.70},
    )

    assert bands.tolist() == [
        "approve",
        "manual_review",
        "manual_review",
        "high_risk",
        "high_risk",
    ]


def test_assign_risk_bands_rejects_null_scores() -> None:
    with pytest.raises(ThresholdingError, match="non-finite"):
        assign_risk_bands(
            np.array([0.10, np.nan]),
            {"threshold_low": 0.30, "threshold_high": 0.70},
        )


def test_validate_threshold_pair_requires_ordered_unit_interval_thresholds() -> None:
    assert validate_threshold_pair(0.20, 0.80, "balanced") == {
        "threshold_low": 0.20,
        "threshold_high": 0.80,
    }

    with pytest.raises(ThresholdingError, match="threshold_low < threshold_high"):
        validate_threshold_pair(0.80, 0.20, "balanced")

    with pytest.raises(ThresholdingError, match=r"\[0, 1\]"):
        validate_threshold_pair(-0.01, 0.20, "balanced")


def test_resolve_scenario_thresholds_uses_config_overrides_and_default_quantiles() -> (
    None
):
    config = {
        "threshold_version": "threshold_v1",
        "scenarios": {
            "growth_oriented": {"threshold_low": None, "threshold_high": None},
            "balanced": {"threshold_low": 0.40, "threshold_high": 0.70},
            "risk_averse": {"threshold_low": None, "threshold_high": None},
        },
    }

    thresholds = resolve_scenario_thresholds(
        config,
        np.linspace(0.01, 1.00, 100),
    )

    assert set(thresholds) == set(SCENARIO_NAMES)
    assert thresholds["balanced"] == {"threshold_low": 0.40, "threshold_high": 0.70}
    assert thresholds["growth_oriented"]["threshold_low"] == pytest.approx(0.8515)
    assert thresholds["growth_oriented"]["threshold_high"] == pytest.approx(0.9505)
    assert thresholds["risk_averse"]["threshold_low"] == pytest.approx(0.703)
    assert thresholds["risk_averse"]["threshold_high"] == pytest.approx(0.802)


def test_resolve_scenario_thresholds_rejects_partial_config_override() -> None:
    config = {
        "threshold_version": "threshold_v1",
        "scenarios": {
            "growth_oriented": {"threshold_low": None, "threshold_high": None},
            "balanced": {"threshold_low": 0.40, "threshold_high": None},
            "risk_averse": {"threshold_low": None, "threshold_high": None},
        },
    }

    with pytest.raises(ThresholdingError, match="both thresholds or neither"):
        resolve_scenario_thresholds(config, np.linspace(0.01, 1.00, 100))
