from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.calibration import apply_saved_calibration_artifact
from src.mart_access import load_labeled_segment_split_frame
from src.metrics import pr_auc_or_none, roc_auc_or_none, validate_probabilities
from src.model_artifacts import normalize_split_ids
from src.model_contracts import REPORTING_SPLITS
from src.modeling import predict_probabilities

SEGMENT_DIMENSIONS = [
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "applicant_age_band",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]


def build_segment_performance_rows(
    connection: duckdb.DuckDBPyConnection,
    artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
    dashboard_model_version: str,
    error_cls: type[Exception] = ValueError,
) -> list[dict[str, Any]]:
    feature_columns = list(artifact["feature_columns"])
    split_applicant_ids = normalize_split_ids(
        artifact["split_applicant_ids"],
        REPORTING_SPLITS,
        error_cls=error_cls,
    )
    rows: list[dict[str, Any]] = []

    for split_name in REPORTING_SPLITS:
        split_frame = load_labeled_segment_split_frame(
            connection,
            split_applicant_ids[split_name],
            feature_columns,
            SEGMENT_DIMENSIONS,
            split_name,
            error_cls=error_cls,
        )
        probabilities = predict_probabilities(
            artifact,
            split_frame,
            feature_columns,
            split_name,
            error_cls,
        )
        probabilities = apply_saved_calibration_artifact(
            probabilities,
            calibration_artifact,
            error_cls=error_cls,
            label="dashboard calibration",
        )
        validate_probabilities(
            probabilities, f"{split_name} calibrated", error_cls=error_cls
        )
        split_frame = split_frame.copy()
        split_frame["probability"] = probabilities.astype(float)

        for segment_name in SEGMENT_DIMENSIONS:
            for segment_value, segment_frame in split_frame.groupby(
                segment_name, dropna=False, sort=True
            ):
                segment_targets = segment_frame["TARGET"].astype(int)
                segment_probabilities = segment_frame["probability"].to_numpy(
                    dtype=float
                )
                rows.append(
                    {
                        "model_version": dashboard_model_version,
                        "split": split_name,
                        "segment_name": segment_name,
                        "segment_value": _segment_value(segment_value),
                        "applicant_count": len(segment_frame),
                        "observed_default_rate": float(segment_targets.mean()),
                        "average_score": float(segment_probabilities.mean()),
                        "roc_auc": roc_auc_or_none(
                            segment_targets, segment_probabilities
                        ),
                        "pr_auc": pr_auc_or_none(
                            segment_targets, segment_probabilities
                        ),
                        "brier_score": float(
                            brier_score_loss(segment_targets, segment_probabilities)
                        ),
                    }
                )

    if not rows:
        raise error_cls("segment_performance_summary must not be empty")
    return rows


def _segment_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
