from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any

import duckdb
import joblib
import matplotlib
import numpy as np
import pandas as pd

from src.config import load_config
from src.score_batch import CREDIT_RISK_SCORE_COLUMNS
from src.train import LIGHTGBM_MODEL_ARTIFACT_NAME
from src.train import LIGHTGBM_MODEL_TYPE
from src.train import LIGHTGBM_MODEL_VERSION


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_SHAP_SUMMARY_ROWS = 5_000
SHAP_BATCH_SIZE = 10_000

MODEL_FEATURE_IMPORTANCE_COLUMNS = [
    "model_version",
    "feature_name",
    "importance_type",
    "importance_value",
    "rank",
]

REASON_COLUMNS = ["top_reason_1", "top_reason_2", "top_reason_3"]


class ExplainabilityError(RuntimeError):
    """Raised when explainability outputs cannot satisfy the Milestone 9 contract."""


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def run_explain(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    duckdb_path = _resolve_project_path(config["paths"]["duckdb_path"])
    model_dir = _resolve_project_path(config["paths"]["model_dir"])
    report_dir = _resolve_project_path(config["paths"]["report_dir"])

    if not duckdb_path.exists():
        raise ExplainabilityError(f"DuckDB database not found: {duckdb_path}")

    excluded_terms = _excluded_output_terms(config)

    with duckdb.connect(str(duckdb_path)) as connection:
        selected_model_type = _load_selected_model_type(connection)
        if selected_model_type != LIGHTGBM_MODEL_TYPE:
            raise ExplainabilityError(
                "Milestone 9 explainability is LightGBM-only; "
                f"model_comparison_summary selected {selected_model_type}"
            )

        artifact = _load_lightgbm_artifact(model_dir)
        model_version = str(artifact["model_version"])
        feature_columns = list(artifact["feature_columns"])
        scored_frame = _load_scored_feature_frame(connection, model_version, feature_columns)
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

        report_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = report_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            report_dir / "model_feature_importance.csv",
            MODEL_FEATURE_IMPORTANCE_COLUMNS,
            importance_rows,
        )
        _write_shap_summary(
            figures_dir / "shap_summary.png",
            transformed_features,
            shap_values,
            feature_labels,
            int(config["project"]["random_seed"]),
        )
        _replace_duckdb_table(connection, "model_feature_importance", importance_rows)
        _update_credit_risk_score_reasons(connection, reason_rows)

    return {
        "model_version": model_version,
        "explained_row_count": len(scored_frame),
        "feature_importance_row_count": len(importance_rows),
        "feature_importance_path": report_dir / "model_feature_importance.csv",
        "shap_summary_path": report_dir / "figures" / "shap_summary.png",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP feature importance and reason-code-style outputs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the project config file.")
    args = parser.parse_args()

    try:
        run_explain(args.config)
    except ExplainabilityError as error:
        raise SystemExit(str(error)) from error


def _load_selected_model_type(connection: duckdb.DuckDBPyConnection) -> str:
    _require_table(connection, "model_comparison_summary")
    selected_values = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT selected_model_type FROM model_comparison_summary"
        ).fetchall()
    }
    if len(selected_values) != 1:
        raise ExplainabilityError(
            f"model_comparison_summary must contain exactly one selected_model_type, got {sorted(selected_values)}"
        )
    return str(next(iter(selected_values)))


def _load_lightgbm_artifact(model_dir: Path) -> dict[str, Any]:
    artifact_path = model_dir / LIGHTGBM_MODEL_ARTIFACT_NAME
    if not artifact_path.exists():
        raise ExplainabilityError(f"Missing LightGBM model artifact: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ExplainabilityError(f"LightGBM model artifact must be a dict: {artifact_path}")

    required_keys = {
        "pipeline",
        "model_version",
        "model_type",
        "feature_columns",
        "numeric_feature_columns",
        "categorical_feature_columns",
    }
    missing_keys = sorted(required_keys.difference(artifact))
    if missing_keys:
        raise ExplainabilityError(f"LightGBM model artifact is missing required keys: {missing_keys}")
    if artifact["model_type"] != LIGHTGBM_MODEL_TYPE:
        raise ExplainabilityError(
            f"LightGBM artifact model_type={artifact['model_type']}, expected {LIGHTGBM_MODEL_TYPE}"
        )
    if artifact["model_version"] != LIGHTGBM_MODEL_VERSION:
        raise ExplainabilityError(
            f"LightGBM artifact model_version={artifact['model_version']}, expected {LIGHTGBM_MODEL_VERSION}"
        )
    if not artifact["feature_columns"]:
        raise ExplainabilityError("LightGBM model artifact does not contain feature_columns")
    return artifact


def _load_scored_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    model_version: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    _require_table(connection, "credit_risk_scores")
    _require_table(connection, "mart_credit_risk_features")
    mart_columns = set(_table_columns(connection, "mart_credit_risk_features"))
    missing_feature_columns = sorted(set(feature_columns).difference(mart_columns))
    if missing_feature_columns:
        raise ExplainabilityError(
            f"mart_credit_risk_features is missing selected model feature columns: {missing_feature_columns}"
        )

    scored_row_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM credit_risk_scores WHERE model_version = ?",
            [model_version],
        ).fetchone()[0]
    )
    if scored_row_count == 0:
        raise ExplainabilityError(f"credit_risk_scores has no rows for model_version={model_version}")

    feature_select = ", ".join(f"m.{_sql_identifier(column)}" for column in feature_columns)
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
        labels.append(_readable_feature_label(raw_feature, category_value))
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
    _replace_duckdb_table_from_frame(connection, "credit_risk_scores", updated_frame)


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
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    if not path.exists() or path.stat().st_size == 0:
        raise ExplainabilityError(f"Failed to write SHAP summary figure: {path}")


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
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    if not path.exists() or path.stat().st_size == 0:
        raise ExplainabilityError(f"Failed to write SHAP summary figure: {path}")
    return True


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


def _readable_feature_label(raw_feature: str, category_value: str | None = None) -> str:
    label = _humanize_token(raw_feature)
    if category_value is None:
        return label
    return f"{label}: {_humanize_token(category_value)}"


def _humanize_token(value: str) -> str:
    cleaned = value.replace("__", "_").replace("_", " ").strip()
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return "Unknown feature"
    return cleaned.lower().capitalize()


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


def _replace_duckdb_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(rows)
    _replace_duckdb_table_from_frame(connection, table_name, frame)


def _replace_duckdb_table_from_frame(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    connection.register("output_frame", frame)
    connection.execute(f"CREATE OR REPLACE TABLE {_sql_identifier(table_name)} AS SELECT * FROM output_frame")
    connection.unregister("output_frame")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _require_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    if table_name not in _existing_tables(connection):
        raise ExplainabilityError(f"Missing required DuckDB table: {table_name}")


def _existing_tables(connection: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in connection.execute("SHOW TABLES").fetchall()}


def _table_columns(connection: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    return {
        row[1]: row[2]
        for row in connection.execute(f"PRAGMA table_info({_sql_literal(table_name)})").fetchall()
    }


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _sql_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) + chr(34))}"'


def _sql_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) + chr(39))}'"


if __name__ == "__main__":
    main()
