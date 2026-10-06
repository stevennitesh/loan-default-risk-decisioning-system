from __future__ import annotations

import pytest

from src.thresholding import calculate_expected_value

ASSUMPTIONS = {
    "expected_margin_per_good_loan": 1000,
    "expected_loss_per_bad_loan": 5000,
    "manual_review_cost": 50,
    "manual_review_capacity_rate": 0.10,
}


def test_expected_value_formula_matches_hand_computed_counts() -> None:
    value = calculate_expected_value(
        approved_good_count=8,
        approved_bad_count=2,
        manual_review_count=5,
        assumptions=ASSUMPTIONS,
    )

    assert value == pytest.approx(-2250)


def test_expected_value_changes_predictably_with_assumptions() -> None:
    richer_margin = {
        **ASSUMPTIONS,
        "expected_margin_per_good_loan": 1200,
    }

    baseline_value = calculate_expected_value(8, 2, 5, ASSUMPTIONS)
    richer_value = calculate_expected_value(8, 2, 5, richer_margin)

    assert richer_value - baseline_value == pytest.approx(8 * 200)


def test_manual_review_cost_is_subtracted_only_for_manual_reviews() -> None:
    value_with_reviews = calculate_expected_value(0, 0, 3, ASSUMPTIONS)
    value_without_reviews = calculate_expected_value(0, 0, 0, ASSUMPTIONS)

    assert value_with_reviews == pytest.approx(-150)
    assert value_without_reviews == pytest.approx(0)


def test_approved_bad_loans_incur_expected_loss() -> None:
    value = calculate_expected_value(0, 4, 0, ASSUMPTIONS)

    assert value == pytest.approx(-20000)


def test_high_risk_declined_counts_do_not_enter_expected_value_formula() -> None:
    value = calculate_expected_value(
        approved_good_count=1,
        approved_bad_count=1,
        manual_review_count=1,
        assumptions=ASSUMPTIONS,
    )

    assert value == pytest.approx(1000 - 5000 - 50)
