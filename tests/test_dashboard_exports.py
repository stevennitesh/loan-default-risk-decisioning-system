from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest

from src.dashboard_exports import DASHBOARD_EXPORT_TABLES
from src.dashboard_exports import POST_V1_DASHBOARD_MODEL_VERSION
from src.dashboard_exports import DashboardExportError
from src.dashboard_exports import run_dashboard_export
from src.calibrate import CALIBRATION_ARTIFACT_NAME
from src.model_contracts import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.model_contracts import LIGHTGBM_MODEL_VERSION
from src.report_contracts import CREDIT_RISK_SCORE_COLUMNS
from src.report_contracts import MODEL_CALIBRATION_BINS_COLUMNS
from src.report_contracts import MODEL_CONFUSION_MATRIX_COLUMNS
from src.report_contracts import MODEL_FEATURE_IMPORTANCE_COLUMNS
from src.report_contracts import MODEL_LIFT_BY_DECILE_COLUMNS
from src.report_contracts import MODEL_METRICS_SUMMARY_COLUMNS
from src.report_contracts import MODEL_THRESHOLD_METRICS_COLUMNS
from src.report_contracts import SEGMENT_PERFORMANCE_SUMMARY_COLUMNS
from src.runtime import sql_identifier
from src.thresholding import SCENARIO_NAMES
from src.train import run_training
from tests.helpers import create_training_database
from tests.helpers import read_csv_rows
from tests.helpers import table_exists


pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


EXPECTED_EXPORT_COLUMNS = {
    "credit_risk_scores": CREDIT_RISK_SCORE_COLUMNS,
    "model_metrics_summary": MODEL_METRICS_SUMMARY_COLUMNS,
    "model_threshold_metrics": MODEL_THRESHOLD_METRICS_COLUMNS,
    "model_lift_by_decile": MODEL_LIFT_BY_DECILE_COLUMNS,
    "model_calibration_bins": MODEL_CALIBRATION_BINS_COLUMNS,
    "model_confusion_matrix": MODEL_CONFUSION_MATRIX_COLUMNS,
    "model_feature_importance": MODEL_FEATURE_IMPORTANCE_COLUMNS,
    "segment_performance_summary": SEGMENT_PERFORMANCE_SUMMARY_COLUMNS,
}

SCENARIOS = set(SCENARIO_NAMES)
SEGMENTS = {
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "applicant_age_band",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
}


class ConstantSigmoidCalibrator:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        positive_probability = np.full(values.shape[0], 0.10)
        return np.column_stack([1 - positive_probability, positive_probability])


def test_dashboard_export_fails_without_required_prerequisites(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    create_training_database(database_path, train_rows=80, test_rows=12)
    run_training(project_config_path)

    with pytest.raises(DashboardExportError, match="Missing required DuckDB tables"):
        run_dashboard_export(project_config_path)

    export_dir = scratch_path / "reports" / "dashboard_data"
    assert not any(export_dir.glob("*.csv")) if export_dir.exists() else True


def test_run_dashboard_export_creates_power_bi_csv_bundle_and_segment_table(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    split_sizes = _create_dashboard_ready_state(database_path, project_config_path)

    result = run_dashboard_export(project_config_path)

    export_dir = scratch_path / "reports" / "dashboard_data"
    assert set(result["exported_tables"]) == set(DASHBOARD_EXPORT_TABLES)
    assert result["export_dir"] == export_dir
    for table_name, expected_columns in EXPECTED_EXPORT_COLUMNS.items():
        export_path = export_dir / f"{table_name}.csv"
        assert export_path.exists()
        rows = read_csv_rows(export_path, expected_columns)
        assert len(rows) == result["row_counts"][table_name]
        if table_name == "credit_risk_scores":
            assert all(row["raw_risk_score"] != "" for row in rows)
            assert all(row["calibrated_risk_score"] != "" for row in rows)
            assert {row["calibration_method"] for row in rows} == {"uncalibrated"}
            assert all(0 <= float(row["raw_risk_score"]) <= 1 for row in rows)
            assert all(0 <= float(row["calibrated_risk_score"]) <= 1 for row in rows)
        if table_name == "model_metrics_summary":
            assert LIGHTGBM_MODEL_VERSION in {row["model_version"] for row in rows}

    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert table_exists(connection, "segment_performance_summary")
        for table_name in DASHBOARD_EXPORT_TABLES:
            duckdb_count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            assert duckdb_count == result["row_counts"][table_name]

        segment_rows = connection.execute(
            """
            SELECT *
            FROM segment_performance_summary
            ORDER BY split, segment_name, segment_value
            """
        ).fetch_df()

    assert set(segment_rows["split"]) == {"validation", "test"}
    assert set(segment_rows["segment_name"]) == SEGMENTS
    assert set(segment_rows["model_version"]) == {LIGHTGBM_MODEL_VERSION}
    for split in ["validation", "test"]:
        split_rows = segment_rows.loc[segment_rows["split"] == split]
        for segment_name in SEGMENTS:
            segment_count = int(
                split_rows.loc[split_rows["segment_name"] == segment_name, "applicant_count"].sum()
            )
            assert segment_count == split_sizes[split]

    assert segment_rows["average_score"].between(0, 1).all()
    assert segment_rows["observed_default_rate"].between(0, 1).all()
    assert segment_rows["brier_score"].between(0, 1).all()
    single_class_rows = segment_rows.loc[segment_rows["segment_name"] == "CODE_GENDER"]
    assert single_class_rows["roc_auc"].isna().all()
    assert single_class_rows["pr_auc"].isna().all()

    threshold_rows = read_csv_rows(
        export_dir / "model_threshold_metrics.csv",
        MODEL_THRESHOLD_METRICS_COLUMNS,
    )
    confusion_rows = read_csv_rows(
        export_dir / "model_confusion_matrix.csv",
        MODEL_CONFUSION_MATRIX_COLUMNS,
    )
    assert {row["scenario_name"] for row in threshold_rows} == SCENARIOS
    assert {row["scenario_name"] for row in confusion_rows} == SCENARIOS

    lift_rows = read_csv_rows(export_dir / "model_lift_by_decile.csv", MODEL_LIFT_BY_DECILE_COLUMNS)
    for split in ["validation", "test"]:
        assert sum(
            int(row["applicant_count"])
            for row in lift_rows
            if row["split"] == split
        ) == split_sizes[split]


def test_run_dashboard_export_can_write_same_bundle_to_post_v1_folder(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    _create_dashboard_ready_state(database_path, project_config_path)

    post_v1_export_dir = scratch_path / "reports" / "dashboard_data_post_v1"
    result = run_dashboard_export(project_config_path, export_dir=post_v1_export_dir)

    default_export_dir = scratch_path / "reports" / "dashboard_data"
    assert result["export_dir"] == post_v1_export_dir
    if default_export_dir.exists():
        assert not any(default_export_dir.glob("*.csv"))
    for table_name, expected_columns in EXPECTED_EXPORT_COLUMNS.items():
        export_path = post_v1_export_dir / f"{table_name}.csv"
        assert export_path.exists()
        rows = read_csv_rows(export_path, expected_columns)
        assert len(rows) == result["row_counts"][table_name]


def test_dashboard_export_uses_selected_calibration_for_probability_quality_tables(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    _create_dashboard_ready_state(database_path, project_config_path)
    _write_constant_sigmoid_calibration_artifact(scratch_path / "models")

    result = run_dashboard_export(project_config_path, use_calibrated_probability_quality=True)

    export_dir = scratch_path / "reports" / "dashboard_data"
    assert result["selected_model_version"] == POST_V1_DASHBOARD_MODEL_VERSION
    metric_rows = read_csv_rows(
        export_dir / "model_metrics_summary.csv",
        MODEL_METRICS_SUMMARY_COLUMNS,
    )
    selected_probability_range_rows = [
        row
        for row in metric_rows
        if row["model_version"] == POST_V1_DASHBOARD_MODEL_VERSION
        and row["split"] in {"validation", "test"}
        and row["metric_name"] in {"min_predicted_probability", "max_predicted_probability"}
    ]
    assert selected_probability_range_rows
    assert all(float(row["metric_value"]) == pytest.approx(0.10) for row in selected_probability_range_rows)
    assert POST_V1_DASHBOARD_MODEL_VERSION in {row["model_version"] for row in metric_rows}
    assert LIGHTGBM_MODEL_VERSION not in {row["model_version"] for row in metric_rows}
    assert len({row["model_version"] for row in metric_rows}) > 1

    calibration_rows = read_csv_rows(
        export_dir / "model_calibration_bins.csv",
        MODEL_CALIBRATION_BINS_COLUMNS,
    )
    assert {row["model_version"] for row in calibration_rows} == {POST_V1_DASHBOARD_MODEL_VERSION}
    assert all(float(row["average_predicted_score"]) == pytest.approx(0.10) for row in calibration_rows)

    segment_rows = read_csv_rows(
        export_dir / "segment_performance_summary.csv",
        SEGMENT_PERFORMANCE_SUMMARY_COLUMNS,
    )
    assert {row["model_version"] for row in segment_rows} == {POST_V1_DASHBOARD_MODEL_VERSION}
    assert all(float(row["average_score"]) == pytest.approx(0.10) for row in segment_rows)

    for table_name, expected_columns in EXPECTED_EXPORT_COLUMNS.items():
        if table_name == "model_metrics_summary":
            continue
        rows = read_csv_rows(export_dir / f"{table_name}.csv", expected_columns)
        assert {row["model_version"] for row in rows} == {POST_V1_DASHBOARD_MODEL_VERSION}


def test_dashboard_export_keeps_raw_v1_probability_quality_by_default(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    database_path = scratch_path / "db" / "credit_risk.duckdb"
    _create_dashboard_ready_state(database_path, project_config_path)
    _write_constant_sigmoid_calibration_artifact(scratch_path / "models")

    run_dashboard_export(project_config_path)

    export_dir = scratch_path / "reports" / "dashboard_data"
    metric_rows = read_csv_rows(
        export_dir / "model_metrics_summary.csv",
        MODEL_METRICS_SUMMARY_COLUMNS,
    )
    selected_probability_range_rows = [
        row
        for row in metric_rows
        if row["model_version"] == LIGHTGBM_MODEL_VERSION
        and row["split"] in {"validation", "test"}
        and row["metric_name"] in {"min_predicted_probability", "max_predicted_probability"}
    ]
    assert selected_probability_range_rows
    assert any(float(row["metric_value"]) != pytest.approx(0.10) for row in selected_probability_range_rows)

    calibration_rows = read_csv_rows(
        export_dir / "model_calibration_bins.csv",
        MODEL_CALIBRATION_BINS_COLUMNS,
    )
    assert any(float(row["average_predicted_score"]) != pytest.approx(0.10) for row in calibration_rows)


def test_dashboard_export_cli_smoke(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    _create_dashboard_ready_state(scratch_path / "db" / "credit_risk.duckdb", project_config_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--config",
            str(project_config_path),
            "--export-dashboard-data",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (scratch_path / "reports" / "dashboard_data" / "segment_performance_summary.csv").exists()


def test_dashboard_export_cli_can_write_post_v1_folder(
    scratch_path: Path,
    project_config_path: Path,
) -> None:
    _create_dashboard_ready_state(scratch_path / "db" / "credit_risk.duckdb", project_config_path)
    post_v1_export_dir = scratch_path / "reports" / "dashboard_data_post_v1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--config",
            str(project_config_path),
            "--export-dashboard-data",
            "--dashboard-export-dir",
            str(post_v1_export_dir),
            "--use-calibrated-dashboard-metrics",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (post_v1_export_dir / "segment_performance_summary.csv").exists()
    assert not (scratch_path / "reports" / "dashboard_data" / "segment_performance_summary.csv").exists()

    metric_rows = read_csv_rows(
        post_v1_export_dir / "model_metrics_summary.csv",
        MODEL_METRICS_SUMMARY_COLUMNS,
    )
    assert POST_V1_DASHBOARD_MODEL_VERSION in {row["model_version"] for row in metric_rows}
    assert LIGHTGBM_MODEL_VERSION not in {row["model_version"] for row in metric_rows}


def _create_dashboard_ready_state(database_path: Path, config_path: Path) -> dict[str, int]:
    create_training_database(database_path, train_rows=80, test_rows=12)
    run_training(config_path)
    artifact_path = database_path.parents[1] / "models" / LIGHTGBM_MODEL_ARTIFACT_NAME
    artifact = joblib.load(artifact_path)
    split_sizes = {split: len(ids) for split, ids in artifact["split_applicant_ids"].items()}

    with duckdb.connect(str(database_path)) as connection:
        connection.execute("UPDATE model_comparison_summary SET selected_model_type = 'lightgbm'")
        _create_credit_risk_scores(connection, artifact)
        _create_model_threshold_metrics(connection, split_sizes)
        _create_lift_rows(connection, split_sizes)
        _create_calibration_rows(connection, split_sizes)
        _create_confusion_rows(connection, split_sizes)
        _create_feature_importance(connection)

    return split_sizes


def _create_credit_risk_scores(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
) -> None:
    feature_columns = list(artifact["feature_columns"])
    test_ids = [int(value) for value in artifact["split_applicant_ids"]["test"]]
    holdout_frame = _mart_frame(
        connection,
        feature_columns,
        "application_train",
        test_ids,
    )
    kaggle_frame = _mart_frame(connection, feature_columns, "application_test", None)
    rows = [
        *_score_rows(artifact, holdout_frame, feature_columns, "holdout_test"),
        *_score_rows(artifact, kaggle_frame, feature_columns, "kaggle_test"),
    ]
    _replace_table(connection, "credit_risk_scores", pd.DataFrame(rows, columns=CREDIT_RISK_SCORE_COLUMNS))


def _write_constant_sigmoid_calibration_artifact(model_dir: Path) -> None:
    calibration_artifact = {
        "base_model_version": LIGHTGBM_MODEL_VERSION,
        "selected_method": "sigmoid",
        "calibrators": {"sigmoid": ConstantSigmoidCalibrator()},
    }
    joblib.dump(calibration_artifact, model_dir / CALIBRATION_ARTIFACT_NAME)


def _mart_frame(
    connection: duckdb.DuckDBPyConnection,
    feature_columns: list[str],
    source_population: str,
    applicant_ids: list[int] | None,
) -> pd.DataFrame:
    selected_columns = ["SK_ID_CURR", "TARGET", *feature_columns]
    if applicant_ids is not None:
        ids_frame = pd.DataFrame({"SK_ID_CURR": applicant_ids})
        connection.register("selected_ids", ids_frame)
        try:
            return connection.execute(
                f"""
                SELECT {", ".join(f'"{column}"' for column in selected_columns)}
                FROM mart_credit_risk_features
                INNER JOIN selected_ids USING (SK_ID_CURR)
                WHERE source_population = ?
                ORDER BY SK_ID_CURR
                """,
                [source_population],
            ).fetch_df()
        finally:
            connection.unregister("selected_ids")
    return connection.execute(
        f"""
        SELECT {", ".join(f'"{column}"' for column in selected_columns)}
        FROM mart_credit_risk_features
        WHERE source_population = ?
        ORDER BY SK_ID_CURR
        """,
        [source_population],
    ).fetch_df()


def _score_rows(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    scoring_population: str,
) -> list[dict[str, Any]]:
    probabilities = artifact["pipeline"].predict_proba(frame[feature_columns])[:, 1]
    rows = []
    for index, record in enumerate(frame.to_dict("records")):
        score = float(probabilities[index])
        rows.append(
            {
                "applicant_id": int(record["SK_ID_CURR"]),
                "scoring_population": scoring_population,
                "observed_target": None if pd.isna(record["TARGET"]) else int(record["TARGET"]),
                "score": score,
                "raw_risk_score": score,
                "calibrated_risk_score": score,
                "calibration_method": "uncalibrated",
                "score_decile": max(1, min(10, int(np.ceil((index + 1) * 10 / len(frame))))),
                "risk_band": "high_risk" if score >= 0.7 else "medium_risk" if score >= 0.3 else "low_risk",
                "recommended_action": "high_priority_review"
                if score >= 0.7
                else "manual_review"
                if score >= 0.3
                else "approve",
                "threshold_version": "threshold_v1",
                "model_version": artifact["model_version"],
                "top_reason_1": "Higher risk: Credit to income ratio",
                "top_reason_2": None,
                "top_reason_3": None,
                "scored_at": "2026-01-01T00:00:00Z",
            }
        )
    return rows


def _create_model_threshold_metrics(
    connection: duckdb.DuckDBPyConnection,
    split_sizes: dict[str, int],
) -> None:
    rows = []
    for split in ["validation", "test"]:
        row_count = split_sizes[split]
        for scenario in sorted(SCENARIOS):
            rows.append(
                {
                    "model_version": LIGHTGBM_MODEL_VERSION,
                    "split": split,
                    "threshold_version": "threshold_v1",
                    "scenario_name": scenario,
                    "threshold_low": 0.3,
                    "threshold_high": 0.7,
                    "applicant_count": row_count,
                    "approval_rate": 0.5,
                    "manual_review_rate": 0.25,
                    "high_risk_rate": 0.25,
                    "approved_good_count": row_count // 2,
                    "approved_bad_count": 0,
                    "manual_review_count": row_count // 4,
                    "high_risk_count": row_count - (row_count // 2) - (row_count // 4),
                    "default_rate_approved": 0.0,
                    "high_risk_default_capture_rate": 0.5,
                    "expected_value": 1000.0,
                    "expected_value_per_applicant": 1000.0 / row_count,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            )
    _replace_table(connection, "model_threshold_metrics", pd.DataFrame(rows, columns=MODEL_THRESHOLD_METRICS_COLUMNS))


def _create_lift_rows(connection: duckdb.DuckDBPyConnection, split_sizes: dict[str, int]) -> None:
    rows = []
    for split in ["validation", "test"]:
        counts = _ten_bin_counts(split_sizes[split])
        for decile, count in enumerate(counts, start=1):
            rows.append(
                {
                    "model_version": LIGHTGBM_MODEL_VERSION,
                    "split": split,
                    "decile": decile,
                    "applicant_count": count,
                    "average_score": 1 - decile / 20,
                    "observed_default_rate": 0.5,
                    "portfolio_default_rate": 0.5,
                    "lift": 1.0,
                    "cumulative_default_capture_rate": min(1.0, decile / 10),
                }
            )
    _replace_table(connection, "model_lift_by_decile", pd.DataFrame(rows, columns=MODEL_LIFT_BY_DECILE_COLUMNS))


def _create_calibration_rows(connection: duckdb.DuckDBPyConnection, split_sizes: dict[str, int]) -> None:
    rows = []
    for split in ["validation", "test"]:
        counts = _ten_bin_counts(split_sizes[split])
        for bin_id, count in enumerate(counts, start=1):
            rows.append(
                {
                    "model_version": LIGHTGBM_MODEL_VERSION,
                    "split": split,
                    "bin_id": bin_id,
                    "applicant_count": count,
                    "average_predicted_score": bin_id / 12,
                    "observed_default_rate": 0.5,
                    "calibration_error": 0.5 - bin_id / 12,
                }
            )
    _replace_table(
        connection,
        "model_calibration_bins",
        pd.DataFrame(rows, columns=MODEL_CALIBRATION_BINS_COLUMNS),
    )


def _create_confusion_rows(connection: duckdb.DuckDBPyConnection, split_sizes: dict[str, int]) -> None:
    rows = []
    for split in ["validation", "test"]:
        for scenario in sorted(SCENARIOS):
            counts = [split_sizes[split] // 4] * 4
            counts[-1] += split_sizes[split] - sum(counts)
            for row_index, (true_label, predicted_label) in enumerate(
                [(0, 0), (0, 1), (1, 0), (1, 1)]
            ):
                rows.append(
                    {
                        "model_version": LIGHTGBM_MODEL_VERSION,
                        "split": split,
                        "scenario_name": scenario,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": counts[row_index],
                    }
                )
    _replace_table(connection, "model_confusion_matrix", pd.DataFrame(rows, columns=MODEL_CONFUSION_MATRIX_COLUMNS))


def _create_feature_importance(connection: duckdb.DuckDBPyConnection) -> None:
    rows = [
        {
            "model_version": LIGHTGBM_MODEL_VERSION,
            "feature_name": "Credit to income ratio",
            "importance_type": "mean_abs_shap",
            "importance_value": 1.0,
            "rank": 1,
        },
        {
            "model_version": LIGHTGBM_MODEL_VERSION,
            "feature_name": "Payment amount ratio",
            "importance_type": "mean_abs_shap",
            "importance_value": 0.5,
            "rank": 2,
        },
    ]
    _replace_table(connection, "model_feature_importance", pd.DataFrame(rows, columns=MODEL_FEATURE_IMPORTANCE_COLUMNS))


def _ten_bin_counts(row_count: int) -> list[int]:
    return [
        int(np.ceil((index + 1) * row_count / 10) - np.ceil(index * row_count / 10))
        for index in range(10)
    ]


def _replace_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    connection.register("table_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {sql_identifier(table_name)} AS SELECT * FROM table_frame")
    connection.unregister("table_frame")
