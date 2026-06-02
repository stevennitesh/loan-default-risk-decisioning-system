from __future__ import annotations

BASELINE_MODEL_VERSION = "logistic_regression_baseline_v1"
BASELINE_MODEL_TYPE = "logistic_regression"
BASELINE_MODEL_ARTIFACT_NAME = "logistic_regression_baseline.joblib"

LIGHTGBM_MODEL_VERSION = "lightgbm_credit_risk_v1"
LIGHTGBM_MODEL_TYPE = "lightgbm"
LIGHTGBM_MODEL_ARTIFACT_NAME = "lightgbm_credit_risk.joblib"

MODEL_ARTIFACTS = {
    BASELINE_MODEL_TYPE: (BASELINE_MODEL_VERSION, BASELINE_MODEL_ARTIFACT_NAME),
    LIGHTGBM_MODEL_TYPE: (LIGHTGBM_MODEL_VERSION, LIGHTGBM_MODEL_ARTIFACT_NAME),
}

EVALUATION_SPLITS = ("train", "validation", "test")
REPORTING_SPLITS = ("validation", "test")


def select_model_type_by_validation_pr_auc(
    baseline_pr_auc: float,
    lightgbm_pr_auc: float,
) -> str:
    return LIGHTGBM_MODEL_TYPE if lightgbm_pr_auc >= baseline_pr_auc else BASELINE_MODEL_TYPE
