from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import pytest

from src.calibrate import run_calibration_experiment
from src.evaluate import run_evaluation
from src.report_contracts import CREDIT_RISK_SCORE_COLUMNS
from src.score_batch import ScoringError, run_scoring
from src.thresholding import BALANCED_SCENARIO
from src.train import run_training
from tests.helpers import (
    assert_table_missing,
    create_training_database,
    query_value,
    read_table_columns,
    table_row_count,
)

VALID_RISK_BANDS = {"low_risk", "medium_risk", "high_risk"}
VALID_ACTIONS = {"approve", "manual_review", "high_priority_review"}

pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names.*:UserWarning"
)


def test_scoring_fails_clearly_without_model_selection(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)

    with pytest.raises(ScoringError, match="model_comparison_summary"):
        run_scoring(project_config_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert_table_missing(connection, "credit_risk_scores")


def test_scoring_fails_clearly_without_threshold_metrics(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    run_training(project_config_path)

    with pytest.raises(ScoringError, match="model_threshold_metrics"):
        run_scoring(project_config_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert_table_missing(connection, "credit_risk_scores")


def test_scoring_fails_clearly_without_selected_model_artifact(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    run_training(project_config_path)
    run_evaluation(project_config_path)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        selected_model_type = query_value(
            connection,
            "SELECT selected_model_type FROM model_comparison_summary LIMIT 1",
        )
    artifact_name = (
        "lightgbm_credit_risk.joblib"
        if selected_model_type == "lightgbm"
        else "logistic_regression_baseline.joblib"
    )
    (scratch_path / "models" / artifact_name).unlink()

    with pytest.raises(ScoringError, match="Missing selected model artifact"):
        run_scoring(project_config_path)


def test_run_scoring_creates_credit_risk_scores_for_holdout_and_kaggle_populations(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    run_training(project_config_path)
    run_evaluation(project_config_path)
    calibration_result = run_calibration_experiment(project_config_path)

    result = run_scoring(project_config_path)

    artifact = joblib.load(scratch_path / "models" / "lightgbm_credit_risk.joblib")
    expected_holdout_rows = len(artifact["split_applicant_ids"]["test"])
    expected_kaggle_rows = 12
    expected_total_rows = expected_holdout_rows + expected_kaggle_rows

    assert result["row_count"] == expected_total_rows
    assert set(result["scoring_populations"]) == {"holdout_test", "kaggle_test"}

    with duckdb.connect(str(database_path), read_only=True) as connection:
        columns = read_table_columns(connection, "credit_risk_scores")
        assert columns == CREDIT_RISK_SCORE_COLUMNS
        assert table_row_count(connection, "credit_risk_scores") == expected_total_rows
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM (
                SELECT applicant_id, scoring_population, model_version, threshold_version
                FROM credit_risk_scores
                GROUP BY applicant_id, scoring_population, model_version, threshold_version
                HAVING COUNT(*) > 1
            )
            """,
            )
            == 0
        )

        population_rows = dict(
            connection.execute(
                """
                SELECT scoring_population, COUNT(*)
                FROM credit_risk_scores
                GROUP BY scoring_population
                """
            ).fetchall()
        )
        assert population_rows == {
            "holdout_test": expected_holdout_rows,
            "kaggle_test": expected_kaggle_rows,
        }
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM credit_risk_scores
            WHERE scoring_population = 'holdout_test'
              AND observed_target IS NOT NULL
            """,
            )
            == expected_holdout_rows
        )
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM credit_risk_scores
            WHERE scoring_population = 'kaggle_test'
              AND observed_target IS NULL
            """,
            )
            == expected_kaggle_rows
        )

        min_score, max_score = connection.execute(
            "SELECT MIN(score), MAX(score) FROM credit_risk_scores"
        ).fetchone()
        assert 0 <= min_score <= max_score <= 1
        raw_min, raw_max, calibrated_min, calibrated_max = connection.execute(
            """
            SELECT
                MIN(raw_risk_score),
                MAX(raw_risk_score),
                MIN(calibrated_risk_score),
                MAX(calibrated_risk_score)
            FROM credit_risk_scores
            """
        ).fetchone()
        assert 0 <= raw_min <= raw_max <= 1
        assert 0 <= calibrated_min <= calibrated_max <= 1
        assert (
            query_value(
                connection,
                "SELECT COUNT(*) FROM credit_risk_scores WHERE score < 0 OR score > 1",
            )
            == 0
        )
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM credit_risk_scores
            WHERE raw_risk_score < 0
               OR raw_risk_score > 1
               OR calibrated_risk_score < 0
               OR calibrated_risk_score > 1
            """,
            )
            == 0
        )
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT calibration_method FROM credit_risk_scores"
            ).fetchall()
        } == {calibration_result["selected_method"]}
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM credit_risk_scores
            WHERE ABS(score - raw_risk_score) > 1e-12
            """,
            )
            == 0
        )
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT risk_band FROM credit_risk_scores"
            ).fetchall()
        }.issubset(VALID_RISK_BANDS)
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT recommended_action FROM credit_risk_scores"
            ).fetchall()
        }.issubset(VALID_ACTIONS)
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM credit_risk_scores
            WHERE top_reason_1 IS NOT NULL
               OR top_reason_2 IS NOT NULL
               OR top_reason_3 IS NOT NULL
            """,
            )
            == 0
        )

        for population in ["holdout_test", "kaggle_test"]:
            decile_min, decile_max = connection.execute(
                """
                SELECT MIN(score_decile), MAX(score_decile)
                FROM credit_risk_scores
                WHERE scoring_population = ?
                """,
                [population],
            ).fetchone()
            assert 1 <= decile_min <= decile_max <= 10
            high_risk_decile_score = query_value(
                connection,
                """
                SELECT AVG(score)
                FROM credit_risk_scores
                WHERE scoring_population = ?
                  AND score_decile = 1
                """,
                [population],
            )
            low_risk_decile_score = query_value(
                connection,
                """
                SELECT AVG(score)
                FROM credit_risk_scores
                WHERE scoring_population = ?
                  AND score_decile = 10
                """,
                [population],
            )
            assert high_risk_decile_score >= low_risk_decile_score

        threshold_low, threshold_high, threshold_version, model_version = (
            connection.execute(
                """
            SELECT threshold_low, threshold_high, threshold_version, model_version
            FROM model_threshold_metrics
            WHERE split = 'validation'
              AND scenario_name = ?
            """,
                [BALANCED_SCENARIO],
            ).fetchone()
        )
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT threshold_version FROM credit_risk_scores"
            ).fetchall()
        } == {threshold_version}
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT model_version FROM credit_risk_scores"
            ).fetchall()
        } == {model_version}
        assert (
            query_value(
                connection,
                """
            SELECT COUNT(*)
            FROM credit_risk_scores
            WHERE (score < ? AND NOT (risk_band = 'low_risk' AND recommended_action = 'approve'))
               OR (score >= ? AND score < ? AND NOT (
                    risk_band = 'medium_risk' AND recommended_action = 'manual_review'
               ))
               OR (score >= ? AND NOT (
                    risk_band = 'high_risk' AND recommended_action = 'high_priority_review'
               ))
            """,
                [threshold_low, threshold_low, threshold_high, threshold_high],
            )
            == 0
        )
