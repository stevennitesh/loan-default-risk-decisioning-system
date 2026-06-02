from __future__ import annotations


def readable_feature_label(raw_feature: str, category_value: str | None = None) -> str:
    """Convert a raw feature and optional encoded category into display text."""
    label = humanize_feature_token(raw_feature)
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
