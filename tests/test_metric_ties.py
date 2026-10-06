import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    precision_at_rate,
    probability_metrics,
    recall_at_rate,
    with_reliability_bin,
)


@pytest.mark.parametrize("reverse", [False, True])
def test_flat_scores_report_expected_capture_and_lift(reverse):
    target = pd.Series([1, 1] + [0] * 18)
    if reverse:
        target = target.iloc[::-1]
    scores = np.full(20, 0.5)
    assert precision_at_rate(target, scores, 0.1) == pytest.approx(0.1)
    assert recall_at_rate(target, scores, 0.1) == pytest.approx(0.1)
    assert probability_metrics(target, scores, 0.1)["top_decile_lift"] == pytest.approx(
        1
    )
    assert probability_metrics(target, scores, 0.1)["log_loss"] == pytest.approx(
        np.log(2)
    )


def test_mixed_boundary_tie_and_ceiling_are_order_independent():
    target = pd.Series([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    # ceil(5*.3)=2: one certain positive plus 1/3 of the tied positive.
    for order in (np.arange(5), np.array([4, 2, 0, 3, 1])):
        assert precision_at_rate(
            target.iloc[order], scores[order], 0.3
        ) == pytest.approx(2 / 3)
        assert recall_at_rate(target.iloc[order], scores[order], 0.3) == pytest.approx(
            4 / 9
        )


def test_reliability_keeps_score_groups_and_exposes_counts():
    frame = pd.DataFrame({"probability": [0.2] * 12 + [0.7] * 8, "target": [0, 1] * 10})
    bins = with_reliability_bin(frame)
    assert bins.groupby("probability")["bin_id"].nunique().max() == 1
    assert len(bins) == 20
    assert with_reliability_bin(frame.assign(probability=0.5))["bin_id"].nunique() == 1
    assert (
        bins["bin_id"].tolist()
        == with_reliability_bin(frame.iloc[::-1])["bin_id"].tolist()[::-1]
    )


@pytest.mark.parametrize(
    "target,scores,rate",
    [
        ([], [], 0.1),
        ([0, 1], [0.1], 0.1),
        ([0, 0.5], [0.1, 0.2], 0.1),
        ([0, 1], [0.1, np.nan], 0.1),
        ([0, 1], [0.1, 1.2], 0.1),
        ([0, 1], [0.1, 0.2], np.nan),
    ],
)
def test_rank_metrics_reject_invalid_inputs(target, scores, rate):
    with pytest.raises(ValueError):
        recall_at_rate(pd.Series(target), np.array(scores), rate)
