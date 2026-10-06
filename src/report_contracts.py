from __future__ import annotations

INGESTION_SUMMARY_COLUMNS = [
    "source_name",
    "source_file",
    "raw_path",
    "parquet_path",
    "staging_table",
    "csv_rows",
    "parquet_rows",
    "duckdb_rows",
    "created_at_utc",
]

FEATURE_PROFILE_COLUMNS = [
    "table_name",
    "row_count",
    "distinct_applicant_count",
    "duplicate_key_count",
    "column_count",
    "created_at_utc",
]

DATA_INVENTORY_COLUMNS = [
    "table_name",
    "layer",
    "grain_key",
    "row_count",
    "distinct_applicant_count",
    "duplicate_grain_key_count",
    "has_target_column",
    "target_non_null_count",
    "target_null_count",
    "created_at_utc",
]

FEATURE_INVENTORY_COLUMNS = [
    "table_name",
    "column_name",
    "duckdb_type",
    "is_model_feature",
    "exclusion_group",
    "missing_count",
    "missing_rate",
    "distinct_value_count",
    "created_at_utc",
]

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

MODEL_RUN_SUMMARY_COLUMNS = [
    "model_version",
    "run_id",
    "model_type",
    "data_scope_version",
    "train_rows",
    "validation_rows",
    "test_rows",
    "feature_count",
    "positive_rate_train",
    "random_seed",
    "created_at",
]

SPLIT_SUMMARY_COLUMNS = [
    "model_version",
    "run_id",
    "split",
    "row_count",
    "positive_count",
    "negative_count",
    "positive_rate",
    "created_at",
]

MODEL_COMPARISON_SUMMARY_COLUMNS = [
    "metric_name",
    "baseline_metric_value",
    "lightgbm_metric_value",
    "lightgbm_minus_baseline",
    "selected_model_type",
]

LIGHTGBM_TUNING_SUMMARY_COLUMNS = [
    "candidate_rank",
    "selected",
    "candidate_name",
    "candidate_source",
    "validation_selection_score",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_manual_review_capacity",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
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

FEATURE_SELECTION_COMPARISON_COLUMNS = [
    "feature_set",
    "selected",
    "feature_count",
    "feature_limit",
    "selected_calibration_method",
    "selected_candidate_name",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_review_capacity",
    "validation_weighted_calibration_error",
    "test_pr_auc",
    "test_roc_auc",
    "test_brier_score",
    "test_top_decile_lift",
    "test_precision_at_top_decile",
    "test_recall_at_review_capacity",
    "test_weighted_calibration_error",
    "validation_balanced_ev_per_applicant",
    "test_balanced_ev_per_applicant",
    "created_at",
]

SELECTED_FEATURE_COLUMNS = [
    "feature_set",
    "feature_rank",
    "feature_name",
]

MODEL_STABILITY_RUN_COLUMNS = [
    "seed",
    "split_seed",
    "model_seed",
    "ranking_seeds",
    "feature_set",
    "seed_validation_winner",
    "feature_count",
    "feature_limit",
    "selected_calibration_method",
    "selected_candidate_name",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_precision_at_top_decile",
    "validation_recall_at_review_capacity",
    "validation_weighted_calibration_error",
    "test_pr_auc",
    "test_roc_auc",
    "test_brier_score",
    "test_top_decile_lift",
    "test_precision_at_top_decile",
    "test_recall_at_review_capacity",
    "test_weighted_calibration_error",
    "validation_balanced_ev_per_applicant",
    "test_balanced_ev_per_applicant",
    "created_at",
]

MODEL_STABILITY_AGGREGATE_COLUMNS = [
    "feature_set",
    "selected",
    "feature_count",
    "feature_limit",
    "seed_count",
    "validation_win_count",
    "validation_win_rate",
    "validation_pr_auc_mean",
    "validation_pr_auc_std",
    "validation_roc_auc_mean",
    "validation_roc_auc_std",
    "validation_brier_score_mean",
    "validation_brier_score_std",
    "validation_top_decile_lift_mean",
    "validation_top_decile_lift_std",
    "validation_precision_at_top_decile_mean",
    "validation_precision_at_top_decile_std",
    "validation_recall_at_review_capacity_mean",
    "validation_recall_at_review_capacity_std",
    "validation_weighted_calibration_error_mean",
    "validation_weighted_calibration_error_std",
    "validation_balanced_ev_per_applicant_mean",
    "validation_balanced_ev_per_applicant_std",
    "test_pr_auc_mean",
    "test_pr_auc_std",
    "test_roc_auc_mean",
    "test_roc_auc_std",
    "test_brier_score_mean",
    "test_brier_score_std",
    "test_top_decile_lift_mean",
    "test_top_decile_lift_std",
    "test_precision_at_top_decile_mean",
    "test_precision_at_top_decile_std",
    "test_recall_at_review_capacity_mean",
    "test_recall_at_review_capacity_std",
    "test_weighted_calibration_error_mean",
    "test_weighted_calibration_error_std",
    "test_balanced_ev_per_applicant_mean",
    "test_balanced_ev_per_applicant_std",
    "pr_auc_generalization_gap",
    "abs_pr_auc_generalization_gap",
    "balanced_ev_generalization_gap",
    "abs_balanced_ev_generalization_gap",
    "created_at",
]

MODEL_STABILITY_SELECTED_AGGREGATE_COLUMNS = [
    *MODEL_STABILITY_AGGREGATE_COLUMNS,
    "completed_split_count",
    "metric_scope",
]

NESTED_CONTEXT_COLUMNS = [
    "run_id",
    "workflow",
    "model_family",
    "split_seed",
    "outer_fold",
    "model_seed",
    "feature_set",
    "calibration_method",
    "created_at",
]
NESTED_METRIC_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "score_kind",
    "metric_name",
    "metric_value",
    "applicant_count",
]
NESTED_PREDICTION_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "SK_ID_CURR",
    "target",
    "raw_score",
    "calibrated_score",
]
NESTED_CANDIDATE_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "feature_count",
    "feature_limit",
    "selected",
    "selected_calibration_method",
    "selected_candidate_name",
    "validation_pr_auc",
    "validation_roc_auc",
    "validation_brier_score",
    "validation_top_decile_lift",
    "validation_recall_at_review_capacity",
]
NESTED_RANKING_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "feature_name",
    "mean_rank",
    "rank_std",
    "ranking_seed_count",
]
NESTED_SUMMARY_COLUMNS = [
    "workflow",
    "model_family",
    "split_seed",
    "score_kind",
    "metric_name",
    "fold_count",
    "fold_mean",
    "fold_std",
]

NESTED_RELIABILITY_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "score_kind",
    "bin_id",
    "applicant_count",
    "average_predicted_score",
    "observed_default_rate",
    "calibration_error",
]
NESTED_SENSITIVITY_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "margin_multiplier",
    "loss_multiplier",
    "review_multiplier",
    "utility_per_applicant",
]

NESTED_SEED_SENSITIVITY_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "score_kind",
    "candidate_name",
    "evaluation_role",
    "promotes_seed",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "log_loss",
    "min_predicted_probability",
    "max_predicted_probability",
    "top_decile_lift",
    "precision_at_top_decile",
    "recall_at_manual_review_capacity",
]
NESTED_PROBABILITY_ACCEPTANCE_COLUMNS = [
    *NESTED_CONTEXT_COLUMNS,
    "selection_probability_accepted",
    "selection_brier_score",
    "selection_log_loss",
    "selection_prevalence_brier",
    "selection_prevalence_log_loss",
]
TUNING_CV_SUMMARY_COLUMNS = [
    "workflow",
    "split_seed",
    "outer_fold",
    "candidate_name",
    "feature_set",
    "weight_fraction",
    "selected",
    "rounds_median",
    "rounds_min",
    "rounds_max",
    "cv_seconds",
    "mean_pr_auc",
    "mean_brier_score",
    "mean_log_loss",
    "mean_prevalence_brier_score",
    "mean_prevalence_log_loss",
]
