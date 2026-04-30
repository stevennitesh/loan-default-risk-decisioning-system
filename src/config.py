from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "project",
    "paths",
    "source_files",
    "split",
    "model",
    "excluded_features",
    "business_assumptions",
    "threshold_policy",
}

REQUIRED_SOURCE_FILES = {
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "pos_cash_balance",
    "credit_card_balance",
    "previous_application",
    "installments_payments",
}

REQUIRED_BUSINESS_ASSUMPTIONS = {
    "expected_margin_per_good_loan",
    "expected_loss_per_bad_loan",
    "manual_review_cost",
    "manual_review_capacity_rate",
}


class ConfigError(ValueError):
    """Raised when the project config violates the documented contract."""


def load_config(path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigError("Config must parse to a mapping")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing_sections = REQUIRED_SECTIONS.difference(config)
    if missing_sections:
        raise ConfigError(f"Missing config sections: {sorted(missing_sections)}")

    source_keys = set(config["source_files"])
    if source_keys != REQUIRED_SOURCE_FILES:
        raise ConfigError(f"Unexpected source_files keys: {sorted(source_keys)}")

    split = config["split"]
    split_total = split["train_size"] + split["validation_size"] + split["test_size"]
    if round(split_total, 10) != 1.0:
        raise ConfigError("Split fractions must sum to 1.0")
    if split["train_size"] <= 0 or split["validation_size"] <= 0 or split["test_size"] <= 0:
        raise ConfigError("Split fractions must be positive")

    excluded_features = config["excluded_features"]
    if "SK_ID_CURR" not in excluded_features["identifiers"]:
        raise ConfigError("SK_ID_CURR must be excluded from model features")
    if "TARGET" not in excluded_features["target"]:
        raise ConfigError("TARGET must be excluded from model features")

    assumptions = set(config["business_assumptions"])
    if assumptions != REQUIRED_BUSINESS_ASSUMPTIONS:
        raise ConfigError(f"Unexpected business assumptions: {sorted(assumptions)}")
