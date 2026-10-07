"""Human-facing vocabulary; machine export keys remain owned by report_contracts."""

POS_CASH_HISTORY_DESCRIPTION = (
    "Monthly loan status, days past due, remaining installments and recent/last-loan "
    "deterioration. Delinquency rates and last-three-month windows use distinct "
    "applicant months; record counts retain account-month grain."
)

WORKFLOW_LABELS = {
    "history_selected": "Application and loan history · LightGBM",
    "application_only": "Application fields only · LightGBM",
    "logistic_tuned": "Application and loan history · logistic regression",
    "logistic_fixed": "Application and loan history · earlier logistic regression",
    "training_prevalence": "Training outcome rate · constant benchmark",
}
METRIC_LABELS = {
    "pr_auc": "Average precision",
    "roc_auc": "ROC AUC",
    "brier_score": "Brier score",
    "log_loss": "Log loss",
    "recall_at_manual_review_capacity": "Repayment-difficulty cases captured in highest-risk group",
    "top_decile_lift": "Highest-risk 10% concentration relative to overall rate",
    "precision_at_top_decile": "Repayment-difficulty rate in highest-risk 10%",
    "balanced_utility_per_applicant": "Illustrative utility per applicant",
    "expected_value_per_applicant": "Illustrative utility per applicant",
}
SPLIT_LABELS = {
    "train": "Model fitting applicants",
    "calibration": "Probability-adjustment fitting applicants",
    "validation": "Model selection applicants",
    "test": "Historical comparison applicants",
    "holdout_test": "Historical comparison applicants",
    "application_test": "Unlabeled application scoring demonstration",
    "kaggle_test": "Unlabeled application scoring demonstration",
}
ACTION_LABELS = {
    "approve": "Simulated approval",
    "manual_review": "Manual review",
    "simulated_decline": "Simulated decline",
}


def workflow_label(value: str) -> str:
    return WORKFLOW_LABELS.get(value, value.replace("_", " ").capitalize())


def metric_label(value: str) -> str:
    return METRIC_LABELS.get(value, value.replace("_", " ").capitalize())


def split_label(value: str) -> str:
    return SPLIT_LABELS.get(value, value.replace("_", " ").capitalize())


def score_label(value: str) -> str:
    return {"calibrated": "Final probabilities", "raw": "Raw probabilities"}.get(
        value, value
    )


def feature_set_label(value: str) -> str:
    if value == "full":
        return "All eligible inputs"
    if value.startswith("top_"):
        return f"Top {value[4:]} training-ranked inputs"
    return value.replace("_", " ").capitalize()


SCENARIO_LABELS = {
    "growth_oriented": "More approvals",
    "balanced": "Balanced reference",
    "risk_averse": "Fewer approvals",
}


def scenario_label(value: str) -> str:
    return SCENARIO_LABELS.get(value, value.replace("_", " ").capitalize())


def method_label(value: str) -> str:
    return {
        "uncalibrated": "Raw probabilities unchanged",
        "sigmoid": "Sigmoid probability adjustment",
        "isotonic": "Isotonic probability adjustment",
    }.get(value, value)


METRIC_DETAILS = {
    "pr_auc": (
        "Start with the highest-risk applicants, then include progressively more. Average precision summarizes the share of each group with recorded repayment difficulty, weighted by the additional difficulty cases captured. It measures ranking, not accuracy.",
        "Higher is better",
        "Unitless, 0 to 1",
    ),
    "roc_auc": (
        "How often a repayment-difficulty case receives a higher score than a case without difficulty, with half credit for ties.",
        "Higher is better",
        "Unitless, 0 to 1",
    ),
    "brier_score": (
        "Mean squared difference between predicted probability and the recorded binary outcome.",
        "Lower is better",
        "Squared probability error",
    ),
    "log_loss": (
        "Probability error that penalizes confident incorrect predictions more strongly; uses natural logarithms.",
        "Lower is better",
        "Unitless",
    ),
    "recall_at_manual_review_capacity": (
        "Share of recorded repayment-difficulty cases captured among the highest-risk fraction of applicants (10% in this assessment), using equal expected membership for boundary ties. Separate from middle-band manual review.",
        "Higher is better at the same group size",
        "Fraction; displayed as a percentage",
    ),
    "top_decile_lift": (
        "Repayment-difficulty rate in the highest-risk 10% divided by the overall rate, with boundary ties handled equally.",
        "Higher is better",
        "Ratio; 1 is the random-ranking reference",
    ),
    "precision_at_top_decile": (
        "Repayment-difficulty rate among applicants in the highest-risk 10%.",
        "Higher is better for ranking at the same group size",
        "Fraction; displayed as a percentage",
    ),
    "balanced_utility_per_applicant": (
        "Assumed approved-loan margins minus assumed losses and middle-band review costs, divided by applicant count. Not dollars or measured profit.",
        "Higher only under the same assumptions",
        "Illustrative utility units per applicant",
    ),
    "expected_value_per_applicant": (
        "Assumed approved-loan margins minus assumed losses and middle-band review costs, divided by applicant count. Not dollars or measured profit.",
        "Higher only under the same assumptions",
        "Illustrative utility units per applicant",
    ),
}


def model_type_label(value: str) -> str:
    return {
        "lightgbm": "LightGBM decision-tree model",
        "logistic_regression": "logistic regression benchmark",
    }.get(value, value.replace("_", " "))
