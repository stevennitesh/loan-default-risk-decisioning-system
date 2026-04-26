from __future__ import annotations

from pathlib import Path

import duckdb
import joblib
import pytest

from src.explain import MODEL_FEATURE_IMPORTANCE_COLUMNS
from src.explain import ExplainabilityError
from src.explain import run_explain
from src.score_batch import run_scoring
from src.train import LIGHTGBM_MODEL_VERSION
from src.train import run_training
from tests.test_train import create_training_database
from tests.test_train import read_csv_rows


FORBIDDEN_EXPLANATION_TERMS = {
    "SK_ID_CURR",
    "TARGET",
    "source_population",
    "SK_ID_PREV",
    "SK_ID_BUREAU",
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "DAYS_BIRTH",
    "applicant_age_years",
    "applicant_age_band",
    "employment_to_age_ratio",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}

pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def test_explain_fails_clearly_without_credit_risk_scores(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    _create_lightgbm_training_state(database_path, project_config_path)

    with pytest.raises(ExplainabilityError, match="credit_risk_scores"):
        run_explain(project_config_path)

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert "model_feature_importance" not in {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def test_explain_fails_when_selected_model_is_not_lightgbm(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    run_training(project_config_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("UPDATE model_comparison_summary SET selected_model_type = 'logistic_regression'")

    with pytest.raises(ExplainabilityError, match="LightGBM-only"):
        run_explain(project_config_path)


def test_explain_fails_without_selected_lightgbm_artifact(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    _create_scored_lightgbm_state(database_path, project_config_path)
    (scratch_path / "models" / "lightgbm_credit_risk.joblib").unlink()

    with pytest.raises(ExplainabilityError, match="Missing LightGBM model artifact"):
        run_explain(project_config_path)


def test_run_explain_creates_importance_outputs_and_updates_reason_fields(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    _create_scored_lightgbm_state(database_path, project_config_path)

    result = run_explain(project_config_path)

    report_dir = scratch_path / "reports"
    importance_rows = read_csv_rows(
        report_dir / "model_feature_importance.csv",
        MODEL_FEATURE_IMPORTANCE_COLUMNS,
    )
    assert result["model_version"] == LIGHTGBM_MODEL_VERSION
    assert result["explained_row_count"] > 0
    assert result["feature_importance_row_count"] == len(importance_rows)
    assert (report_dir / "figures" / "shap_summary.png").stat().st_size > 0

    assert importance_rows
    assert [int(row["rank"]) for row in importance_rows] == list(range(1, len(importance_rows) + 1))
    for row in importance_rows:
        assert row["model_version"] == LIGHTGBM_MODEL_VERSION
        assert row["feature_name"]
        assert row["importance_type"] == "mean_abs_shap"
        assert float(row["importance_value"]) >= 0
        _assert_no_forbidden_terms(row["feature_name"])

    with duckdb.connect(str(database_path), read_only=True) as connection:
        table_rows = connection.execute(
            "SELECT COUNT(*) FROM model_feature_importance"
        ).fetchone()[0]
        assert table_rows == len(importance_rows)
        reason_rows = connection.execute(
            """
            SELECT top_reason_1, top_reason_2, top_reason_3
            FROM credit_risk_scores
            WHERE model_version = ?
            ORDER BY scoring_population, applicant_id
            """,
            [LIGHTGBM_MODEL_VERSION],
        ).fetchall()

    flattened_reasons = [reason for row in reason_rows for reason in row if reason is not None]
    assert flattened_reasons
    for reason in flattened_reasons:
        assert isinstance(reason, str)
        assert "Higher risk:" in reason
        assert "__" not in reason
        assert " " in reason
        _assert_no_forbidden_terms(reason)


def _create_lightgbm_training_state(database_path: Path, config_path: Path) -> None:
    create_training_database(database_path, train_rows=80, test_rows=12)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            UPDATE mart_credit_risk_features
            SET
                credit_to_income_ratio = CASE
                    WHEN TARGET = 1 THEN 4.0
                    WHEN TARGET = 0 THEN 0.2
                    ELSE credit_to_income_ratio
                END,
                payment_amount_ratio = CASE
                    WHEN TARGET = 1 THEN 0.35
                    WHEN TARGET = 0 THEN 1.15
                    ELSE payment_amount_ratio
                END,
                category_feature = CASE
                    WHEN TARGET = 1 THEN 'high'
                    WHEN TARGET = 0 THEN 'low'
                    ELSE category_feature
                END
            """
        )
        connection.execute(
            """
            UPDATE mart_credit_risk_features
            SET
                credit_to_income_ratio = CASE WHEN SK_ID_CURR % 2 = 0 THEN 4.0 ELSE 0.2 END,
                payment_amount_ratio = CASE WHEN SK_ID_CURR % 2 = 0 THEN 0.35 ELSE 1.15 END,
                category_feature = CASE WHEN SK_ID_CURR % 2 = 0 THEN 'high' ELSE 'low' END
            WHERE source_population = 'application_test'
            """
        )
    run_training(config_path)
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("UPDATE model_comparison_summary SET selected_model_type = 'lightgbm'")
        connection.execute(
            """
            CREATE OR REPLACE TABLE model_threshold_metrics AS
            SELECT
                ?::VARCHAR AS model_version,
                'validation'::VARCHAR AS split,
                'threshold_v1'::VARCHAR AS threshold_version,
                'balanced'::VARCHAR AS scenario_name,
                0.30::DOUBLE AS threshold_low,
                0.70::DOUBLE AS threshold_high
            """,
            [LIGHTGBM_MODEL_VERSION],
        )


def _create_scored_lightgbm_state(database_path: Path, config_path: Path) -> None:
    _create_lightgbm_training_state(database_path, config_path)
    run_scoring(config_path)
    artifact = joblib.load(database_path.parents[1] / "models" / "lightgbm_credit_risk.joblib")
    assert artifact["model_version"] == LIGHTGBM_MODEL_VERSION


def _assert_no_forbidden_terms(text: str) -> None:
    normalized_text = text.lower().replace("_", " ")
    for forbidden in FORBIDDEN_EXPLANATION_TERMS:
        assert forbidden.lower().replace("_", " ") not in normalized_text
