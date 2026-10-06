from __future__ import annotations

# Display-only names shared with the frozen-input diagnostic's semantic labels.
# Application fields: sql/02_feature_applicant.sql; cash-loan history:
# sql/04b_feature_pos_cash.sql and sql/05d_feature_last_k_temporal.sql.
FEATURE_LABELS = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "ext_source_mean": "Mean external credit score",
    "ext_source_min": "Minimum external credit score",
    "ext_source_max": "Maximum external credit score",
    "ext_source_missing_count": "Missing external-score count",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_ANNUITY": "Application annuity amount",
    "AMT_INCOME_TOTAL": "Reported income",
    "AMT_GOODS_PRICE": "Goods purchase price",
    "FLAG_OWN_CAR": "Reported car ownership",
    "FLAG_OWN_REALTY": "Reported real-estate ownership",
    "NAME_CONTRACT_TYPE": "Loan contract category",
    "NAME_EDUCATION_TYPE": "Reported education category",
    "NAME_INCOME_TYPE": "Income category",
    "NAME_HOUSING_TYPE": "Housing category",
    "ORGANIZATION_TYPE": "Employer organization category",
    "OCCUPATION_TYPE": "Occupation category",
    "employment_length_days": "Employment length in days",
    "credit_to_income_ratio": "Requested credit relative to income",
    "annuity_to_income_ratio": "Annuity relative to income",
    "goods_price_to_income_ratio": "Goods price relative to income",
    "avg_credit_to_application_ratio": "Prior granted credit / requested amount",
    "pos_cash_avg_future_installments": "Remaining cash-loan installments",
    "pos_cash_last_3_future_installment_ratio": "Remaining / scheduled cash-loan installments, latest 3 months",
    "bureau_debt_to_income_ratio": "Bureau debt relative to income",
    "payment_shortfall_ratio": "Installment payment shortfall ratio",
}


def readable_feature_label(raw_feature: str, category_value: str | None = None) -> str:
    """Convert a raw feature and optional encoded category into display text."""
    label = FEATURE_LABELS.get(
        raw_feature,
        FEATURE_LABELS.get(raw_feature.lower(), humanize_feature_token(raw_feature)),
    )
    if category_value is None:
        return label
    return f"{label}: {humanize_feature_token(category_value)}"


def humanize_feature_token(value: str) -> str:
    """Turn a snake-case or encoded feature token into sentence-style text."""
    cleaned = value.replace("__", "_").replace("_", " ").strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return "Unknown feature"
    return cleaned.lower().capitalize()
