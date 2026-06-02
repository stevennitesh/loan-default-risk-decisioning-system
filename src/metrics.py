from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

TError = TypeVar("TError", bound=Exception)


def validate_probabilities(
    probabilities: np.ndarray,
    label: str,
    *,
    error_cls: type[TError],
) -> None:
    """Validate that model probabilities are one-dimensional values in [0, 1]."""
    if probabilities.ndim != 1:
        raise error_cls(f"{label} probabilities must be one-dimensional")
    if not np.isfinite(probabilities).all():
        raise error_cls(f"{label} probabilities contain non-finite values")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise error_cls(f"{label} probabilities must be in [0, 1]")


def top_decile_lift(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    error_cls: type[TError] = ValueError,
) -> float:
    """Return default lift among the highest-risk decile."""
    portfolio_positive_rate = float(y_true.mean())
    top_precision = precision_at_rate(y_true, probabilities, 0.10, error_cls=error_cls)
    return float(top_precision / portfolio_positive_rate)


def precision_at_rate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    rate: float,
    *,
    error_cls: type[TError] = ValueError,
) -> float:
    """Return target precision among the top-ranked applicants at a rate."""
    frame = _probability_frame(y_true, probabilities)
    return float(_top_rate_frame(frame, rate, error_cls)["target"].mean())


def recall_at_rate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    rate: float,
    *,
    error_cls: type[TError] = ValueError,
) -> float:
    """Return target recall among the top-ranked applicants at a rate."""
    frame = _probability_frame(y_true, probabilities)
    positives_in_top = int(_top_rate_frame(frame, rate, error_cls)["target"].sum())
    total_positives = int(frame["target"].sum())
    return float(positives_in_top / total_positives) if total_positives else 0.0


def top_count(row_count: int, rate: float, *, error_cls: type[TError]) -> int:
    """Convert a positive selection rate to at least one selected row."""
    if rate <= 0 or rate > 1:
        raise error_cls(f"Selection rate must be in (0, 1], got {rate}")
    return max(1, int(np.ceil(row_count * rate)))


def with_probability_rank_bin(
    frame: pd.DataFrame, column_name: str, *, descending: bool
) -> pd.DataFrame:
    """Add a deterministic 1-10 probability rank bin column."""
    ranked = frame.sort_values(
        ["probability", "SK_ID_CURR"],
        ascending=[not descending, True],
    ).reset_index(drop=True)
    ranked[column_name] = np.ceil(
        (np.arange(len(ranked)) + 1) * 10 / len(ranked)
    ).astype(int)
    ranked[column_name] = ranked[column_name].clip(1, 10)
    return ranked


def nullable_mean(series: pd.Series) -> float | None:
    """Return a float mean, or None for empty report groups."""
    if series.empty:
        return None
    return float(series.mean())


def target_class_values(
    targets: pd.Series | np.ndarray, *, dropna: bool = False
) -> set[int]:
    """Return observed integer target classes, optionally dropping missing values."""
    target_series = pd.Series(targets)
    if dropna:
        target_series = target_series.dropna()
    return set(target_series.astype(int).unique())


def roc_auc_or_none(targets: pd.Series, probabilities: np.ndarray) -> float | None:
    """Return ROC-AUC when a segment contains both classes, otherwise None."""
    if not _has_binary_targets(targets):
        return None
    return float(roc_auc_score(targets, probabilities))


def pr_auc_or_none(targets: pd.Series, probabilities: np.ndarray) -> float | None:
    """Return PR-AUC when a segment contains both classes, otherwise None."""
    if not _has_binary_targets(targets):
        return None
    return float(average_precision_score(targets, probabilities))


def probability_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    manual_review_capacity_rate: float,
    error_cls: type[Exception] = ValueError,
) -> dict[str, float]:
    """Build the standard credit-risk probability metric bundle."""
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "min_predicted_probability": float(np.min(probabilities)),
        "max_predicted_probability": float(np.max(probabilities)),
        "top_decile_lift": top_decile_lift(y_true, probabilities, error_cls=error_cls),
        "precision_at_top_decile": precision_at_rate(
            y_true, probabilities, 0.10, error_cls=error_cls
        ),
        "recall_at_manual_review_capacity": recall_at_rate(
            y_true,
            probabilities,
            manual_review_capacity_rate,
            error_cls=error_cls,
        ),
    }


def build_probability_metric_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    created_at: str,
    manual_review_capacity_rate: float,
    *,
    error_cls: type[TError] = ValueError,
) -> list[dict[str, object]]:
    """Build long-form probability metric rows for each prediction split."""
    rows: list[dict[str, object]] = []
    for split_name, frame in prediction_frames.items():
        y_true = frame["target"]
        probabilities = frame["probability"].to_numpy(dtype=float)
        metrics = probability_metrics(
            y_true, probabilities, manual_review_capacity_rate, error_cls
        )
        rows.extend(
            {
                "model_version": model_version,
                "split": split_name,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "created_at": created_at,
            }
            for metric_name, metric_value in metrics.items()
        )
    return rows


def build_calibration_bin_rows(
    model_version: str,
    prediction_frames: dict[str, pd.DataFrame],
    reporting_splits: tuple[str, ...],
) -> list[dict[str, object]]:
    """Build calibration-bin rows for the configured reporting splits."""
    rows: list[dict[str, object]] = []
    for split_name in reporting_splits:
        frame = with_probability_rank_bin(
            prediction_frames[split_name], "bin_id", descending=False
        )
        for bin_id in range(1, 11):
            bin_frame = frame.loc[frame["bin_id"] == bin_id]
            average_predicted_score = nullable_mean(bin_frame["probability"])
            observed_default_rate = nullable_mean(bin_frame["target"])
            rows.append(
                {
                    "model_version": model_version,
                    "split": split_name,
                    "bin_id": bin_id,
                    "applicant_count": len(bin_frame),
                    "average_predicted_score": average_predicted_score,
                    "observed_default_rate": observed_default_rate,
                    "calibration_error": observed_default_rate - average_predicted_score
                    if observed_default_rate is not None
                    and average_predicted_score is not None
                    else None,
                }
            )
    return rows


def _top_rate_frame(
    frame: pd.DataFrame,
    rate: float,
    error_cls: type[TError],
) -> pd.DataFrame:
    """Return the top rows by probability for the requested selection rate."""
    count = top_count(len(frame), rate, error_cls=error_cls)
    return frame.sort_values("probability", ascending=False).head(count)


def _probability_frame(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Build the canonical target/probability frame used by rank metrics."""
    return pd.DataFrame({"target": y_true.to_numpy(), "probability": probabilities})


def _has_binary_targets(targets: pd.Series) -> bool:
    """Return whether targets contain exactly the two binary classes."""
    return target_class_values(targets) == {0, 1}
