import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    required_directories = [
        "configs",
        "src",
        "tests",
        "sql",
        "data/raw",
        "data/parquet",
        "data/db",
        "models",
        "reports/figures",
        "reports/dashboard_data",
        "reports/dashboard_data_post_v1",
        "powerbi/screenshots",
    ]

    for directory in required_directories:
        assert (ROOT / directory).is_dir(), f"Missing required directory: {directory}"


def test_makefile_exposes_required_targets() -> None:
    makefile_path = ROOT / "Makefile"
    assert makefile_path.exists(), "Makefile must exist"
    makefile_text = makefile_path.read_text(encoding="utf-8")

    for target in [
        "setup",
        "ingest",
        "features",
        "train",
        "evaluate",
        "score",
        "calibrate",
        "explain",
        "dashboard-data",
        "pipeline-v1",
        "pipeline-post-v1",
        "dashboard-data-post-v1",
        "test",
    ]:
        assert f"{target}:" in makefile_text


def test_required_sql_files_exist() -> None:
    required_sql_files = [
        "02_feature_applicant.sql",
        "03_feature_bureau.sql",
        "03b_feature_bureau_balance.sql",
        "04_feature_previous_applications.sql",
        "04b_feature_pos_cash.sql",
        "04c_feature_credit_card.sql",
        "05_feature_installments.sql",
        "05b_feature_risk_pressure.sql",
        "05c_feature_recency_deterioration.sql",
        "05d_feature_last_k_temporal.sql",
        "06_build_feature_mart.sql",
        "06_build_feature_mart_v1.sql",
        "07_create_score_tables.sql",
    ]

    for filename in required_sql_files:
        assert (ROOT / "sql" / filename).exists(), f"Missing SQL file: {filename}"


def test_generated_artifact_paths_are_gitignored() -> None:
    generated_paths = [
        "data/raw/application_train.csv",
        "data/parquet/application_train.parquet",
        "data/db/credit_risk.duckdb",
        "models/lightgbm_credit_risk.joblib",
        "reports/ingestion_summary.csv",
        "reports/feature_mart_profile.csv",
        "reports/data_inventory.csv",
        "reports/feature_inventory.csv",
        "reports/model_run_summary.csv",
        "reports/model_metrics_summary.csv",
        "reports/model_comparison_summary.csv",
        "reports/lightgbm_tuning_summary.csv",
        "reports/split_summary.csv",
        "reports/model_lift_by_decile.csv",
        "reports/model_calibration_bins.csv",
        "reports/model_confusion_matrix.csv",
        "reports/model_threshold_metrics.csv",
        "reports/validation_report.md",
        "reports/business_value_analysis.md",
        "reports/model_feature_importance.csv",
        "reports/figures/roc_curve.png",
        "reports/figures/pr_curve.png",
        "reports/figures/calibration_curve.png",
        "reports/figures/lift_chart.png",
        "reports/figures/shap_summary.png",
        "reports/dashboard_data/credit_risk_scores.csv",
        "reports/dashboard_data/model_metrics_summary.csv",
        "reports/dashboard_data/segment_performance_summary.csv",
        "reports/dashboard_data_post_v1/credit_risk_scores.csv",
        "reports/dashboard_data_post_v1/model_metrics_summary.csv",
        "reports/dashboard_data_post_v1/segment_performance_summary.csv",
        "reports/v1/model_metrics_summary.csv",
        "reports/post_v1/model_metrics_summary.csv",
        "reports/figures/generated/lift_chart.png",
        ".tmp/scratch.txt",
    ]

    for path in generated_paths:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0, f"Expected git to ignore {path}"
