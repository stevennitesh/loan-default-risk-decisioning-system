from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from src.class_weighting_ablation import (
    METRICS,
    summarize,
    validate_manifests,
    weight_variant,
)


def test_weight_intervention_preserves_every_other_model_parameter():
    original = LGBMClassifier(n_estimators=37, random_state=42, scale_pos_weight=11.4)
    changed = weight_variant(original, 1)
    assert changed.get_params() == {**original.get_params(), "scale_pos_weight": 1.0}
    assert original.get_params()["scale_pos_weight"] == 11.4
    assert not hasattr(changed, "booster_")
    with pytest.raises(ValueError, match="Competing class weighting"):
        weight_variant(LGBMClassifier(class_weight="balanced"), 1)


def manifest():
    return {
        "status": "complete",
        "negative_control": False,
        "feature_build_id": "build",
        "development_id_target_sha256": "development",
        "historical_comparison_sha256": "historical",
        "historical_comparison_ids": [99],
        "feature_columns": ["income"],
        "folds": [
            {
                "applicant_ids": {
                    "train": [10 + i],
                    "calibration": [20 + i],
                    "validation": [30 + i],
                    "assessment": [i],
                }
            }
            for i in range(5)
        ],
    }


@pytest.mark.parametrize(
    "corruption", ["roles", "build", "status", "overlap", "historical"]
)
def test_attribution_rejects_unmatched_or_contaminated_inputs(corruption):
    earlier = manifest()
    current = deepcopy(earlier)
    if corruption == "roles":
        current["folds"][0]["applicant_ids"]["train"].reverse()
        current["folds"][0]["applicant_ids"]["train"].append(77)
    elif corruption == "build":
        current["feature_build_id"] = "different"
    elif corruption == "status":
        current["status"] = "failed"
    elif corruption == "overlap":
        for m in (earlier, current):
            m["folds"][0]["applicant_ids"]["train"] = [0]
    else:
        for m in (earlier, current):
            m["folds"][0]["applicant_ids"]["train"] = [99]
    with pytest.raises(ValueError):
        validate_manifests(earlier, current)


def test_paired_effect_direction_is_unweighted_minus_weighted():
    rows = pd.DataFrame(
        [
            {
                "recipe": recipe,
                "outer_fold": fold,
                "weighting": weight,
                **{metric: value for metric in METRICS},
            }
            for recipe in ("earlier", "current")
            for fold in range(1, 6)
            for weight, value in (
                ("earlier_weight", 0.4 + fold / 100),
                ("unweighted", 0.1 + fold / 100),
            )
        ]
    )
    summary, paired = summarize(rows)
    assert len(summary) == 4
    assert len(paired) == 10
    assert paired["brier_score_unweighted_minus_weighted"].tolist() == pytest.approx(
        [-0.3] * 10
    )
    assert summary.loc[
        summary.weighting == "unweighted", "brier_score_mean"
    ].tolist() == pytest.approx([0.13] * 2)
