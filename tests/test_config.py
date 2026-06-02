from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "base.yaml"
V1_CONFIG_PATH = ROOT / "configs" / "v1.yaml"
POST_V1_CONFIG_PATH = ROOT / "configs" / "post_v1.yaml"

V1_SOURCE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
    "installments_payments": "installments_payments.csv",
}

POST_V1_SOURCE_FILES = {
    **V1_SOURCE_FILES,
    "bureau_balance": "bureau_balance.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


def _load_config() -> dict:
    assert CONFIG_PATH.exists(), "configs/base.yaml must exist"
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_base_config_has_required_sections() -> None:
    config = _load_config()

    required_sections = {
        "project",
        "paths",
        "source_files",
        "split",
        "model",
        "excluded_features",
        "business_assumptions",
        "threshold_policy",
    }

    assert required_sections.issubset(config)


def test_config_loader_returns_validated_config() -> None:
    from src.config import load_config as load_project_config

    config = load_project_config(CONFIG_PATH)

    assert config["project"]["name"] == "loan-default-decisioning"


def test_source_files_include_post_v1_bureau_balance_inputs() -> None:
    config = _load_config()

    assert config["source_files"] == POST_V1_SOURCE_FILES


def test_v1_and_post_v1_configs_are_valid_reproducible_pipeline_scopes() -> None:
    from src.config import load_config as load_project_config

    v1_config = load_project_config(V1_CONFIG_PATH)
    post_v1_config = load_project_config(POST_V1_CONFIG_PATH)

    assert v1_config["project"]["data_scope_version"] == "v1"
    assert v1_config["source_files"] == V1_SOURCE_FILES
    assert v1_config["paths"]["duckdb_path"] == "data/db/credit_risk_v1.duckdb"
    assert v1_config["paths"]["model_dir"] == "models/v1"
    assert v1_config["paths"]["report_dir"] == "reports/v1"
    assert v1_config["paths"]["dashboard_export_dir"] == "reports/dashboard_data"

    assert post_v1_config["project"]["data_scope_version"].startswith("post_v1")
    assert post_v1_config["source_files"] == POST_V1_SOURCE_FILES
    assert post_v1_config["paths"]["duckdb_path"] == "data/db/credit_risk_post_v1.duckdb"
    assert post_v1_config["paths"]["model_dir"] == "models/post_v1"
    assert post_v1_config["paths"]["report_dir"] == "reports/post_v1"
    assert post_v1_config["paths"]["dashboard_export_dir"] == "reports/dashboard_data_post_v1"


def test_split_fractions_sum_to_one() -> None:
    config = _load_config()
    split = config["split"]

    total = split["train_size"] + split["validation_size"] + split["test_size"]

    assert total == 1.0
    assert split["stratify"] is True


def test_excluded_feature_groups_cover_required_leakage_controls() -> None:
    config = _load_config()
    excluded_features = config["excluded_features"]

    assert "SK_ID_CURR" in excluded_features["identifiers"]
    assert "TARGET" in excluded_features["target"]
    assert {
        "CODE_GENDER",
        "NAME_FAMILY_STATUS",
        "DAYS_BIRTH",
        "applicant_age_years",
        "applicant_age_band",
        "employment_to_age_ratio",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
    }.issubset(excluded_features["sensitive_or_protected_status_like"])


def test_business_assumptions_cover_threshold_value_analysis() -> None:
    config = _load_config()

    assert config["business_assumptions"] == {
        "expected_margin_per_good_loan": 1000,
        "expected_loss_per_bad_loan": 5000,
        "manual_review_cost": 50,
        "manual_review_capacity_rate": 0.10,
    }
