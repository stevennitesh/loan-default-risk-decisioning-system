from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from src.cli import add_config_argument
from src.cli import exit_with_error
from src.config import DEFAULT_CONFIG_PATH
from src.config import load_config
from src.config import project_random_seed
from src.feature_labels import readable_feature_label
from src.mart_access import fetch_count
from src.mart_access import require_table
from src.mart_access import require_table_columns
from src.model_contracts import BASELINE_MODEL_TYPE
from src.model_contracts import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.model_contracts import LIGHTGBM_MODEL_TYPE
from src.model_contracts import LIGHTGBM_MODEL_VERSION
from src.model_artifacts import load_model_artifact
from src.model_artifacts import load_selected_model_type
from src.runtime import ensure_directories
from src.runtime import replace_duckdb_table
from src.runtime import replace_duckdb_table_from_frame
from src.runtime import require_existing_path
from src.runtime import resolve_config_path
from src.runtime import sql_identifier
from src.runtime import write_csv
from src.report_contracts import CREDIT_RISK_SCORE_COLUMNS
from src.report_contracts import MODEL_FEATURE_IMPORTANCE_COLUMNS


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAX_SHAP_SUMMARY_ROWS = 5_000
SHAP_BATCH_SIZE = 10_000

REASON_COLUMNS = ["top_reason_1", "top_reason_2", "top_reason_3"]


class ExplainabilityError(RuntimeError):
    """Raised when explainability outputs cannot satisfy the Milestone 9 contract."""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def run_explain(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = resolve_config_path(config, "duckdb_path")
    model_dir = resolve_config_path(config, "model_dir")
    report_dir = resolve_config_path(config, "report_dir")

    require_existing_path(duckdb_path, "DuckDB database", ExplainabilityError)

    excluded_terms = _excluded_output_terms(config)

    with duckdb.connect(str(duckdb_path)) as connection:
        selected_model_type = load_selected_model_type(
            connection,
            {BASELINE_MODEL_TYPE, LIGHTGBM_MODEL_TYPE},
            error_cls=ExplainabilityError,
        )
        if selected_model_type != LIGHTGBM_MODEL_TYPE:
            raise ExplainabilityError(
                "Milestone 9 explainability is LightGBM-only; "
                f"model_comparison_summary selected {selected_model_type}"
            )

        artifact = _load_lightgbm_artifact(model_dir)
        model_version = str(artifact["model_version"])
        feature_columns = list(artifact["feature_columns"])
        scored_frame = _load_scored_feature_frame(connection, model_version, feature_columns)
        # SHAP runs on the fitted transformed feature space, then maps labels back to raw features.
        transformed_features, transformed_feature_names, classifier = _transform_features(
            artifact,
            scored_frame[feature_columns],
        )
        feature_labels = _readable_transformed_feature_labels(
            transformed_feature_names,
            feature_columns,
            list(artifact.get("categorical_feature_columns", [])),
            excluded_terms,
        )
        shap_values = _compute_shap_values(classifier, transformed_features, len(feature_labels))
        importance_rows = _build_feature_importance_rows(model_version, feature_labels, shap_values)
        _validate_explanation_texts(
            [row["feature_name"] for row in importance_rows],
            excluded_terms,
            "model_feature_importance",
        )
        reason_rows = _build_reason_rows(scored_frame, feature_labels, shap_values)
        _validate_explanation_texts(
            [
                reason
                for row in reason_rows
                for reason in (row["top_reason_1"], row["top_reason_2"], row["top_reason_3"])
                if reason is not None
            ],
            excluded_terms,
            "credit_risk_scores reason fields",
        )

        figures_dir = report_dir / "figures"
        ensure_directories(report_dir, figures_dir)
        write_csv(
            report_dir / "model_feature_importance.csv",
            MODEL_FEATURE_IMPORTANCE_COLUMNS,
            importance_rows,
        )
        _write_shap_summary(
            figures_dir / "shap_summary.png",
            transformed_features,
            shap_values,
            feature_labels,
            project_random_seed(config),
        )
        replace_duckdb_table(connection, "model_feature_importance", importance_rows)
        _update_credit_risk_score_reasons(connection, reason_rows)

    return {
        "model_version": model_version,
        "explained_row_count": len(scored_frame),
        "feature_importance_row_count": len(importance_rows),
        "feature_importance_path": report_dir / "model_feature_importance.csv",
        "shap_summary_path": report_dir / "figures" / "shap_summary.png",
    }


def _load_lightgbm_artifact(model_dir: Path) -> dict[str, Any]:
    artifact_path = model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME
    artifact = load_model_artifact(
        artifact_path,
        expected_model_type=LIGHTGBM_MODEL_TYPE,
        expected_model_version=LIGHTGBM_MODEL_VERSION,
        error_cls=ExplainabilityError,
        artifact_label="LightGBM model artifact",
        missing_label="LightGBM model artifact",
        require_feature_columns=True,
    )
    required_keys = {
        "numeric_feature_columns",
        "categorical_feature_columns",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise ExplainabilityError(f"LightGBM model artifact is missing required keys: {missing_keys}")
    return artifact


def _load_scored_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    model_version: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    require_table(connection, "credit_risk_scores", error_cls=ExplainabilityError)
    require_table(connection, "mart_credit_risk_features", error_cls=ExplainabilityError)
    require_table_columns(
        connection,
        "mart_credit_risk_features",
        feature_columns,
        error_cls=ExplainabilityError,
    )

    scored_row_count = fetch_count(
        connection,
        "SELECT COUNT(*) FROM credit_risk_scores WHERE model_version = ?",
        ExplainabilityError,
        [model_version],
    )
    if scored_row_count == 0:
        raise ExplainabilityError(f"credit_risk_scores has no rows for model_version={model_version}")

    feature_select = ", ".join(f"m.{sql_identifier(column)}" for column in feature_columns)
    frame = connection.execute(
        f"""
        SELECT
            s.applicant_id,
            s.scoring_population,
            s.model_version,
            s.threshold_version,
            s.score,
            {feature_select}
        FROM credit_risk_scores AS s
        INNER JOIN mart_credit_risk_features AS m
            ON m.SK_ID_CURR = s.applicant_id
           -- Holdout rows come from the labeled training population; Kaggle scoring rows stay unlabeled.
           AND (
                (s.scoring_population = 'holdout_test' AND m.source_population = 'application_train')
                OR (s.scoring_population = 'kaggle_test' AND m.source_population = 'application_test')
           )
        WHERE s.model_version = ?
        ORDER BY s.scoring_population, s.applicant_id
        """,
        [model_version],
    ).fetch_df()
    if len(frame) != scored_row_count:
        raise ExplainabilityError(
            "credit_risk_scores rows do not reconcile to mart_credit_risk_features for explainability: "
            f"scored={scored_row_count}, joined={len(frame)}"
        )
    return frame.reset_index(drop=True)


def _transform_features(
    artifact: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[Any, list[str], Any]:
    pipeline = artifact["pipeline"]
    if not hasattr(pipeline, "named_steps"):
        raise ExplainabilityError("LightGBM artifact pipeline must expose named_steps")
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        raise ExplainabilityError("LightGBM artifact pipeline must contain preprocessor and classifier steps")
    if not hasattr(classifier, "booster_"):
        raise ExplainabilityError("LightGBM classifier is not fitted with a booster_")

    features = frame.copy()
    transformed = preprocessor.transform(features.where(pd.notna(features), np.nan))
    try:
        transformed_feature_names = list(preprocessor.get_feature_names_out())
    except AttributeError as error:
        raise ExplainabilityError("LightGBM preprocessor must expose transformed feature names") from error
    if not transformed_feature_names:
        raise ExplainabilityError("LightGBM preprocessor produced no transformed feature names")
    return transformed, transformed_feature_names, classifier


def _compute_shap_values(
    classifier: Any,
    transformed_features: Any,
    transformed_feature_count: int,
) -> np.ndarray:
    row_count = transformed_features.shape[0]
    batches = []
    for start in range(0, row_count, SHAP_BATCH_SIZE):
        stop = min(start + SHAP_BATCH_SIZE, row_count)
        contributions = classifier.booster_.predict(
            transformed_features[start:stop],
            pred_contrib=True,
        )
        contribution_array = np.asarray(contributions, dtype=float)
        if contribution_array.ndim != 2:
            raise ExplainabilityError(
                f"LightGBM SHAP contributions must be a 2D array, got shape {contribution_array.shape}"
            )
        batches.append(contribution_array)

    if not batches:
        raise ExplainabilityError("No rows were available for SHAP computation")
    contribution_array = np.vstack(batches)
    expected_width = transformed_feature_count + 1
    if contribution_array.shape != (row_count, expected_width):
        raise ExplainabilityError(
            "LightGBM SHAP contribution shape does not match transformed features: "
            f"got {contribution_array.shape}, expected {(row_count, expected_width)}"
        )
    # LightGBM appends the expected value as the final contribution column; outputs use feature effects only.
    shap_values = contribution_array[:, :-1]
    if not np.isfinite(shap_values).all():
        raise ExplainabilityError("LightGBM SHAP values contain non-finite values")
    return shap_values


def _readable_transformed_feature_labels(
    transformed_feature_names: list[str],
    feature_columns: list[str],
    categorical_feature_columns: list[str],
    excluded_terms: set[str],
) -> list[str]:
    labels = []
    for transformed_name in transformed_feature_names:
        raw_feature, category_value = _raw_feature_for_transformed_name(
            transformed_name,
            feature_columns,
            categorical_feature_columns,
        )
        if _contains_excluded_term(raw_feature, excluded_terms):
            raise ExplainabilityError(f"Excluded field appeared in SHAP feature output: {raw_feature}")
        labels.append(readable_feature_label(raw_feature, category_value))
    _validate_explanation_texts(labels, excluded_terms, "SHAP feature labels")
    return labels


def _raw_feature_for_transformed_name(
    transformed_name: str,
    feature_columns: list[str],
    categorical_feature_columns: list[str],
) -> tuple[str, str | None]:
    name_without_transformer = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    for category_feature in sorted(categorical_feature_columns, key=len, reverse=True):
        if name_without_transformer == category_feature:
            return category_feature, None
        category_prefix = f"{category_feature}_"
        if name_without_transformer.startswith(category_prefix):
            return category_feature, name_without_transformer[len(category_prefix) :]
    for feature_name in sorted(feature_columns, key=len, reverse=True):
        if name_without_transformer == feature_name:
            return feature_name, None
    return name_without_transformer, None


def _build_feature_importance_rows(
    model_version: str,
    feature_labels: list[str],
    shap_values: np.ndarray,
) -> list[dict[str, Any]]:
    mean_abs_values = np.abs(shap_values).mean(axis=0)
    importance_by_label: dict[str, float] = {}
    for feature_label, importance_value in zip(feature_labels, mean_abs_values, strict=True):
        importance_by_label[feature_label] = importance_by_label.get(feature_label, 0.0) + float(
            importance_value
        )

    sorted_importance = sorted(
        (
            (feature_name, importance_value)
            for feature_name, importance_value in importance_by_label.items()
            if importance_value > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not sorted_importance:
        raise ExplainabilityError("All SHAP feature importance values are zero")
    return [
        {
            "model_version": model_version,
            "feature_name": feature_name,
            "importance_type": "mean_abs_shap",
            "importance_value": importance_value,
            "rank": rank,
        }
        for rank, (feature_name, importance_value) in enumerate(sorted_importance, start=1)
    ]


def _build_reason_rows(
    scored_frame: pd.DataFrame,
    feature_labels: list[str],
    shap_values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, scored_record in scored_frame.iterrows():
        local_values = shap_values[row_index]
        sorted_indexes = np.argsort(-local_values)
        reasons = []
        seen_labels = set()
        for feature_index in sorted_indexes:
            contribution = float(local_values[feature_index])
            if contribution <= 0:
                break
            feature_label = feature_labels[feature_index]
            if feature_label in seen_labels:
                continue
            # Reason fields are directional debugging signals, not adverse-action notices.
            reasons.append(f"Higher risk: {feature_label}")
            seen_labels.add(feature_label)
            if len(reasons) == 3:
                break
        while len(reasons) < 3:
            reasons.append(None)
        rows.append(
            {
                "applicant_id": int(scored_record["applicant_id"]),
                "scoring_population": str(scored_record["scoring_population"]),
                "model_version": str(scored_record["model_version"]),
                "threshold_version": str(scored_record["threshold_version"]),
                "top_reason_1": reasons[0],
                "top_reason_2": reasons[1],
                "top_reason_3": reasons[2],
            }
        )
    return rows


def _update_credit_risk_score_reasons(
    connection: duckdb.DuckDBPyConnection,
    reason_rows: list[dict[str, Any]],
) -> None:
    score_frame = connection.execute("SELECT * FROM credit_risk_scores").fetch_df()
    if score_frame.empty:
        raise ExplainabilityError("credit_risk_scores must not be empty")
    missing_columns = sorted(set(CREDIT_RISK_SCORE_COLUMNS).difference(score_frame.columns))
    if missing_columns:
        raise ExplainabilityError(f"credit_risk_scores is missing required columns: {missing_columns}")

    key_columns = ["applicant_id", "scoring_population", "model_version", "threshold_version"]
    score_frame = score_frame[CREDIT_RISK_SCORE_COLUMNS].copy()
    for column in REASON_COLUMNS:
        score_frame[column] = score_frame[column].astype("object")
    score_indexed = score_frame.set_index(key_columns)
    reason_frame = pd.DataFrame(reason_rows).set_index(key_columns)
    missing_reason_keys = reason_frame.index.difference(score_indexed.index)
    if len(missing_reason_keys):
        raise ExplainabilityError("Reason rows contain keys missing from credit_risk_scores")
    for column in REASON_COLUMNS:
        score_indexed.loc[reason_frame.index, column] = reason_frame[column]
    updated_frame = score_indexed.reset_index()[CREDIT_RISK_SCORE_COLUMNS]
    replace_duckdb_table_from_frame(connection, "credit_risk_scores", updated_frame)


def _write_shap_summary(
    path: Path,
    transformed_features: Any,
    shap_values: np.ndarray,
    feature_labels: list[str],
    random_seed: int,
) -> None:
    sample_indexes = _summary_sample_indexes(shap_values.shape[0], random_seed)
    sampled_shap_values = shap_values[sample_indexes]
    sampled_features = _to_dense(transformed_features[sample_indexes])
    if _write_shap_package_summary(path, sampled_features, sampled_shap_values, feature_labels):
        return

    feature_order = np.argsort(-np.abs(sampled_shap_values).mean(axis=0))[: min(20, len(feature_labels))]

    figure, axis = plt.subplots(figsize=(9, max(5, len(feature_order) * 0.35)))
    rng = np.random.default_rng(random_seed)
    for display_position, feature_index in enumerate(reversed(feature_order)):
        x_values = sampled_shap_values[:, feature_index]
        feature_values = sampled_features[:, feature_index]
        y_values = np.full(len(x_values), display_position, dtype=float)
        y_values += rng.normal(loc=0.0, scale=0.08, size=len(x_values))
        scatter = axis.scatter(
            x_values,
            y_values,
            c=feature_values,
            cmap="coolwarm",
            s=14,
            alpha=0.75,
            linewidths=0,
        )
    axis.axvline(0, color="gray", linewidth=1, alpha=0.7)
    axis.set_yticks(range(len(feature_order)))
    axis.set_yticklabels([feature_labels[index] for index in reversed(feature_order)])
    axis.set_xlabel("SHAP contribution to default-risk score")
    axis.set_title("LightGBM SHAP Summary")
    axis.grid(True, axis="x", alpha=0.25)
    if len(feature_order):
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
        colorbar.set_label("Transformed feature value")
    _save_shap_figure(path, figure)


def _write_shap_package_summary(
    path: Path,
    sampled_features: np.ndarray,
    sampled_shap_values: np.ndarray,
    feature_labels: list[str],
) -> bool:
    try:
        import shap
    except ImportError:
        return False

    plot_frame = pd.DataFrame(sampled_features, columns=feature_labels)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        sampled_shap_values,
        plot_frame,
        feature_names=feature_labels,
        max_display=min(20, len(feature_labels)),
        show=False,
    )
    figure = plt.gcf()
    _save_shap_figure(path, figure)
    return True


def _save_shap_figure(path: Path, figure: Any) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    if not path.exists() or path.stat().st_size == 0:
        raise ExplainabilityError(f"Failed to write SHAP summary figure: {path}")


def _summary_sample_indexes(row_count: int, random_seed: int) -> np.ndarray:
    if row_count <= 0:
        raise ExplainabilityError("Cannot sample SHAP summary rows from an empty explanation set")
    if row_count <= MAX_SHAP_SUMMARY_ROWS:
        return np.arange(row_count)
    random_generator = np.random.default_rng(random_seed)
    return np.sort(random_generator.choice(row_count, size=MAX_SHAP_SUMMARY_ROWS, replace=False))


def _to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _validate_explanation_texts(texts: list[str], excluded_terms: set[str], output_name: str) -> None:
    for text in texts:
        if _contains_excluded_term(text, excluded_terms):
            raise ExplainabilityError(f"Excluded field appeared in {output_name}: {text}")


def _contains_excluded_term(text: str, excluded_terms: set[str]) -> bool:
    normalized_text = _normalize_output_text(text)
    return any(_normalize_output_text(term) in normalized_text for term in excluded_terms)


def _excluded_output_terms(config: dict[str, Any]) -> set[str]:
    terms = {"source_population"}
    for column_names in config["excluded_features"].values():
        terms.update(str(column_name) for column_name in column_names)
    return terms


def _normalize_output_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance and reason-code-style outputs.")
    add_config_argument(parser)
    args = parser.parse_args()

    try:
        run_explain(args.config)
    except ExplainabilityError as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
