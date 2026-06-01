from __future__ import annotations

CREDIT_RISK_SCORE_COLUMNS = [
    "applicant_id",
    "scoring_population",
    "observed_target",
    "score",
    "raw_risk_score",
    "calibrated_risk_score",
    "calibration_method",
    "score_decile",
    "risk_band",
    "recommended_action",
    "threshold_version",
    "model_version",
    "top_reason_1",
    "top_reason_2",
    "top_reason_3",
    "scored_at",
]

MODEL_METRICS_SUMMARY_COLUMNS = [
    "model_version",
    "split",
    "metric_name",
    "metric_value",
    "created_at",
]

MODEL_THRESHOLD_METRICS_COLUMNS = [
    "model_version",
    "split",
    "threshold_version",
    "scenario_name",
    "threshold_low",
    "threshold_high",
    "applicant_count",
    "approval_rate",
    "manual_review_rate",
    "high_risk_rate",
    "approved_good_count",
    "approved_bad_count",
    "manual_review_count",
    "high_risk_count",
    "default_rate_approved",
    "high_risk_default_capture_rate",
    "expected_value",
    "expected_value_per_applicant",
    "created_at",
]

MODEL_LIFT_BY_DECILE_COLUMNS = [
    "model_version",
    "split",
    "decile",
    "applicant_count",
    "average_score",
    "observed_default_rate",
    "portfolio_default_rate",
    "lift",
    "cumulative_default_capture_rate",
]

MODEL_CALIBRATION_BINS_COLUMNS = [
    "model_version",
    "split",
    "bin_id",
    "applicant_count",
    "average_predicted_score",
    "observed_default_rate",
    "calibration_error",
]

MODEL_CALIBRATION_COMPARISON_COLUMNS = [
    "model_version",
    "base_model_version",
    "calibration_method",
    "split",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "min_predicted_probability",
    "max_predicted_probability",
    "top_decile_lift",
    "precision_at_top_decile",
    "recall_at_manual_review_capacity",
    "mean_absolute_bin_error",
    "weighted_calibration_error",
    "max_absolute_bin_error",
    "created_at",
]

MODEL_CALIBRATION_BINS_COMPARISON_COLUMNS = [
    "model_version",
    "base_model_version",
    "calibration_method",
    "split",
    "bin_id",
    "applicant_count",
    "average_predicted_score",
    "observed_default_rate",
    "calibration_error",
    "created_at",
]

MODEL_CONFUSION_MATRIX_COLUMNS = [
    "model_version",
    "split",
    "scenario_name",
    "true_label",
    "predicted_label",
    "count",
]

MODEL_FEATURE_IMPORTANCE_COLUMNS = [
    "model_version",
    "feature_name",
    "importance_type",
    "importance_value",
    "rank",
]

SEGMENT_PERFORMANCE_SUMMARY_COLUMNS = [
    "model_version",
    "split",
    "segment_name",
    "segment_value",
    "applicant_count",
    "observed_default_rate",
    "average_score",
    "roc_auc",
    "pr_auc",
    "brier_score",
]
