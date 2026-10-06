from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.assessment_model_inputs import (
    contribution_magnitudes,
    sampled_ids,
    source_group,
    transformed_raw_fields,
)


def test_sampling_is_order_invariant_unique_and_target_blind():
    ids = list(range(1700))
    chosen = sampled_ids(ids)
    np.testing.assert_array_equal(chosen, sampled_ids(ids[::-1]))
    assert len(chosen) == len(set(chosen)) == 1000
    assert set(chosen) <= set(ids)
    with pytest.raises(ValueError, match="unique"):
        sampled_ids([1, 1])


def test_one_hot_mapping_uses_output_ownership_even_with_prefix_collisions():
    # alpha_long_yes is an alpha category despite the alpha_long input name.
    frame = pd.DataFrame({"alpha": ["long_yes", "no"], "alpha_long": ["yes", "no"]})
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                ["alpha", "alpha_long"],
            )
        ]
    ).fit(frame)
    mapping = transformed_raw_fields(preprocessor)
    assert mapping == ["alpha", "alpha", "alpha_long", "alpha_long"]
    assert preprocessor.get_feature_names_out()[0] == "categorical__alpha_long_yes"
    assert mapping[0] == "alpha"


def test_absolute_encoded_magnitudes_and_signed_intercept_accounting_are_separate():
    features, groups, residual = contribution_magnitudes(
        np.array([[1.0, -1.0, 0.5], [-2.0, 2.0, 0.5]]),
        np.array([0.5, 0.5]),
        ["category", "category"],
        {"category": "context"},
    )
    assert features == {"category": 3.0}
    assert groups == {"context": 3.0}
    assert residual == 0
    with pytest.raises(ValueError, match="additivity"):
        contribution_magnitudes(
            np.array([[1.0, -1.0, 0.5]]),
            np.array([0.6]),
            ["a", "b"],
            {"a": "g", "b": "g"},
        )


def test_source_groups_keep_external_scores_and_installment_meaning_explicit():
    assert source_group("EXT_SOURCE_2", ["EXT_SOURCE_2"]) == "external_scores"
    assert (
        source_group("installment_unknown_payment_obligation_count", [])
        == "installments"
    )
    assert source_group("external_score_credit_pressure", []) == "cross_source"
    with pytest.raises(ValueError, match="Unmapped"):
        source_group("unreviewed_feature", [])
