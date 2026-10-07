"""Bounded, target-blind interpretation of already frozen assessment models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.explain import _transform_features
from src.feature_labels import readable_feature_label
from src.mart_access import feature_build_id
from src.presentation import POS_CASH_HISTORY_DESCRIPTION
from src.runtime import sql_identifier

ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "reports/generated/tuning_20261004/nested_assessment/b45cb7fb871d4322bfdbb52e6e8d1d38"
)
DESTINATION = ROOT / "reports/model_inputs_20261004"
SAMPLE_SEED = 20261004
MAX_ROWS = 1000
GROUP_DETAILS = {
    "external_scores": (
        "Imported external credit scores",
        "Three supplied external-score fields and their mean, minimum, maximum and missing-count summaries. Their original construction and real-time availability cannot be independently certified.",
    ),
    "application": (
        "Application finances and context",
        "Requested credit, income, annuity, financial ratios, employment and application context. Direct demographic/protected-status-like fields are excluded; remaining context can still contain proxies.",
    ),
    "bureau": (
        "Bureau loans and balances",
        "Counts, amounts, debt, overdue status and relative dates for bureau loans originating before application day; monthly balance status and recent deterioration.",
    ),
    "previous_applications": (
        "Previous loan applications",
        "Prior Home Credit application outcomes, requested and granted amounts, approval/refusal rates and relative decision dates.",
    ),
    "installments": (
        "Installment repayment history",
        "Payment timing, shortfalls, arrears and recent repayment behavior. Obligations are counted once across split payments; ambiguity and unknown payment support remain explicit.",
    ),
    "pos_cash": (
        "Cash-loan monthly history",
        POS_CASH_HISTORY_DESCRIPTION,
    ),
    "credit_card": (
        "Credit-card monthly history",
        "Balances, limits, utilization, drawings, payment support and delinquency, with recent and last-loan summaries.",
    ),
    "cross_source": (
        "Combined financial pressure",
        "Ratios combining external scores or application income/credit with bureau debt and installment payment shortfall. These groups overlap information sources.",
    ),
}
EXTERNAL = {
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "ext_source_mean",
    "ext_source_min",
    "ext_source_max",
    "ext_source_missing_count",
}
PREVIOUS = {
    "previous_application_count",
    "approved_application_count",
    "refused_application_count",
    "canceled_application_count",
    "approval_rate",
    "refusal_rate",
    "avg_application_amount",
    "avg_previous_credit_amount",
    "total_previous_credit_amount",
    "avg_credit_to_application_ratio",
    "avg_days_decision",
    "earliest_days_decision",
    "latest_days_decision",
}
INSTALLMENTS = {
    "late_payment_count",
    "avg_payment_delay_days",
    "max_payment_delay_days",
    "underpayment_count",
    "total_instalment_amount",
    "total_payment_amount",
    "payment_amount_ratio",
    "avg_payment_to_instalment_ratio",
}
PRESSURE = {
    "external_score_credit_pressure",
    "external_score_annuity_pressure",
    "bureau_debt_to_income_ratio",
    "payment_shortfall_ratio",
}
FEATURE_LABELS = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "ext_source_mean": "Mean external credit score",
    "ext_source_min": "Minimum external credit score",
    "ext_source_max": "Maximum external credit score",
    "ext_source_missing_count": "Missing external-score count",
    "avg_credit_to_application_ratio": "Prior granted credit / requested amount",
    "pos_cash_avg_future_installments": "Remaining cash-loan installments",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_ANNUITY": "Application annuity amount",
    "AMT_INCOME_TOTAL": "Reported income",
    "AMT_GOODS_PRICE": "Goods purchase price",
    "employment_length_days": "Employment length",
    "DAYS_LAST_PHONE_CHANGE": "Time since last phone change",
    "DAYS_ID_PUBLISH": "Time since identity-document issue",
    "DAYS_REGISTRATION": "Time since registration change",
    "NAME_EDUCATION_TYPE": "Reported education category",
    "NAME_INCOME_TYPE": "Income category",
    "NAME_HOUSING_TYPE": "Housing category",
    "NAME_CONTRACT_TYPE": "Loan contract category",
    "ORGANIZATION_TYPE": "Employer organization category",
    "OCCUPATION_TYPE": "Occupation category",
    "external_score_credit_pressure": "Credit amount relative to external-score support",
    "external_score_annuity_pressure": "Annuity relative to external-score support",
    "bureau_debt_to_income_ratio": "Bureau debt relative to income",
    "payment_shortfall_ratio": "Installment payment shortfall ratio",
    "credit_to_income_ratio": "Requested credit relative to income",
    "annuity_to_income_ratio": "Annuity relative to income",
    "goods_price_to_income_ratio": "Goods price relative to income",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_group(field: str, application_columns: list[str]) -> str:
    if field in EXTERNAL:
        return "external_scores"
    if field in PRESSURE:
        return "cross_source"
    if field in application_columns:
        return "application"
    if field.startswith("pos_cash_"):
        return "pos_cash"
    if field.startswith("credit_card_"):
        return "credit_card"
    if field.startswith("installment") or field in INSTALLMENTS:
        return "installments"
    if field in PREVIOUS:
        return "previous_applications"
    if field.startswith(
        (
            "bureau_",
            "active_credit_",
            "closed_credit_",
            "overdue_credit_",
            "max_credit_day_",
            "total_credit_",
            "avg_credit_",
            "avg_days_credit",
            "earliest_days_credit",
            "latest_days_credit",
            "avg_days_enddate",
        )
    ):
        return "bureau"
    raise ValueError(f"Unmapped model input: {field}")


def transformed_raw_fields(preprocessor) -> list[str]:
    """Use encoder output order, never category-name prefix guessing."""
    mapping = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            if transformer != "drop":
                raise ValueError("Unexpected remainder inputs")
            continue
        names = transformer.get_feature_names_out(columns)
        encoder = transformer.named_steps.get("encoder")
        if encoder is None:
            encoder = transformer.named_steps.get("onehot")
        if encoder is None:
            if len(names) != len(columns):
                raise ValueError("Numeric transformation changed input width")
            mapping.extend(columns)
        else:
            if encoder.min_frequency is not None or encoder.max_categories is not None:
                raise ValueError("Grouped one-hot categories require explicit mapping")
            for index, (column, categories) in enumerate(
                zip(columns, encoder.categories_, strict=True)
            ):
                dropped = (
                    encoder.drop_idx_ is not None
                    and encoder.drop_idx_[index] is not None
                )
                mapping.extend([column] * (len(categories) - int(dropped)))
    if len(mapping) != len(preprocessor.get_feature_names_out()):
        raise ValueError("Transformed input mapping does not reconcile")
    return mapping


def contribution_magnitudes(contributions, raw_margin, fields, groups):
    """Sum absolute encoded-column effects; keep signed additivity separate."""
    values = np.asarray(contributions, dtype=float)
    if (
        values.shape != (len(raw_margin), len(fields) + 1)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Invalid native contribution shape or values")
    residual = float(np.max(np.abs(values.sum(axis=1) - raw_margin)))
    if residual > 1e-8:
        raise ValueError("Native contributions fail intercept-inclusive additivity")
    feature_means = {}
    group_means = {}
    for field, magnitude in zip(
        fields, np.abs(values[:, :-1]).mean(axis=0), strict=True
    ):
        feature_means[field] = feature_means.get(field, 0.0) + float(magnitude)
        group = groups[field]
        group_means[group] = group_means.get(group, 0.0) + float(magnitude)
    return feature_means, group_means, residual


def sampled_ids(ids, seed=SAMPLE_SEED, maximum=MAX_ROWS):
    ordered = np.array(sorted(ids), dtype=np.int64)
    if len(ordered) != len(set(ids)) or not len(ordered):
        raise ValueError("Assessment membership must be nonempty and unique")
    return np.sort(
        np.random.default_rng(seed).choice(
            ordered, min(maximum, len(ordered)), replace=False
        )
    )


def run(source: Path = RUN, destination: Path = DESTINATION) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["negative_control"]:
        raise ValueError(
            "Interpretation requires the completed original-label assessment"
        )
    if {(f["split_seed"], f["outer_fold"]) for f in manifest["folds"]} != {
        (42, n) for n in range(1, 6)
    }:
        raise ValueError("Expected all five declared current models")
    columns = manifest["feature_columns"]
    excluded = {"source_population"}.union(
        *(set(v) for v in manifest["config"]["excluded_features"].values())
    )
    if set(columns) & excluded:
        raise ValueError("Excluded identifier, target or diagnostic model input")
    groups = {c: source_group(c, manifest["application_columns"]) for c in columns}
    dictionary = [
        {
            "raw_field": c,
            "display_label": FEATURE_LABELS.get(c, readable_feature_label(c)),
            "source_group": groups[c],
            "group_label": GROUP_DETAILS[groups[c]][0],
            "group_description": GROUP_DETAILS[groups[c]][1],
        }
        for c in columns
    ]
    rows, checks, model_hashes = [], [], {}
    db = ROOT / manifest["config"]["paths"]["duckdb_path"]
    with duckdb.connect(str(db), read_only=True) as connection:
        if feature_build_id(connection) != manifest["feature_build_id"]:
            raise ValueError("Interpretation feature-build identity mismatch")
        for plan in manifest["folds"]:
            fold = plan["outer_fold"]
            model_path = source / f"fold_42_{fold}_history_selected.joblib"
            artifact = joblib.load(model_path)
            context = artifact["assessment_context"]
            if (
                context["run_id"] != manifest["run_id"]
                or context["workflow"] != "history_selected"
                or context["outer_fold"] != fold
                or context["split_seed"] != 42
            ):
                raise ValueError("Frozen model assessment identity mismatch")
            if (
                artifact["feature_columns"] != columns
                or artifact["selected_calibration_method"] != "uncalibrated"
            ):
                raise ValueError(
                    "Current interpretation requires the declared full raw-score models"
                )
            outer = plan["applicant_ids"]["assessment"]
            if set(outer) & set(artifact["fit_applicant_ids"]):
                raise ValueError("Assessment rows overlap fitting/selection membership")
            chosen = sampled_ids(outer)
            connection.register(
                "interpretation_membership", pd.DataFrame({"SK_ID_CURR": chosen})
            )
            select = ",".join("m." + sql_identifier(c) for c in columns)
            # TARGET is deliberately never queried or available to sampling/contributions.
            frame = connection.execute(
                f"SELECT m.SK_ID_CURR,{select} FROM mart_credit_risk_features m JOIN interpretation_membership s USING(SK_ID_CURR) WHERE m.source_population='application_train' ORDER BY m.SK_ID_CURR"
            ).fetch_df()
            if not np.array_equal(frame.SK_ID_CURR.to_numpy(), chosen):
                raise ValueError(
                    "Sample does not reconcile to one labeled-source mart row"
                )
            transformed, names, classifier = _transform_features(
                artifact, frame[columns]
            )
            fields = transformed_raw_fields(
                artifact["pipeline"].named_steps["preprocessor"]
            )
            if len(fields) != len(names) or set(fields) != set(columns):
                raise ValueError("Transformed raw-field ownership mismatch")
            contributions = classifier.booster_.predict(
                transformed, pred_contrib=True, num_threads=4
            )
            raw_margin = classifier.booster_.predict(
                transformed, raw_score=True, num_threads=4
            )
            features, families, residual = contribution_magnitudes(
                contributions, raw_margin, fields, groups
            )
            for level, magnitudes in [
                ("feature", features),
                ("source_group", families),
            ]:
                for key, magnitude in magnitudes.items():
                    rows.append(
                        {
                            "assessment_run_id": manifest["run_id"],
                            "split_seed": 42,
                            "outer_fold": fold,
                            "workflow": "history_selected",
                            "sample_count": len(chosen),
                            "level": level,
                            "input_key": key,
                            "mean_absolute_contribution": magnitude,
                        }
                    )
            checks.append(
                {
                    "split_seed": 42,
                    "outer_fold": fold,
                    "assessment_count": len(outer),
                    "sample_count": len(chosen),
                    "raw_input_count": len(columns),
                    "transformed_input_count": len(fields),
                    "maximum_additivity_residual": residual,
                }
            )
            model_hashes[model_path.name] = sha256(model_path)
    destination.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(rows).sort_values(["level", "input_key", "outer_fold"])
    summary = (
        fold_frame.groupby(["level", "input_key"])
        .mean_absolute_contribution.agg(
            fold_mean="mean", fold_sd="std", fold_min="min", fold_max="max"
        )
        .reset_index()
    )
    fold_frame.to_csv(destination / "fold_importance.csv", index=False)
    summary.to_csv(destination / "importance_summary.csv", index=False)
    pd.DataFrame(dictionary).to_csv(destination / "input_dictionary.csv", index=False)
    provenance = {
        "purpose": "Frozen-model behavior diagnostic; no fitting, selection, scores or memberships published",
        "assessment_run_id": manifest["run_id"],
        "protocol": manifest["protocol_version"],
        "feature_build_id": manifest["feature_build_id"],
        "assessment_config_sha256": manifest["config_sha256"],
        "workflow": "history_selected",
        "status": "complete",
        "sample_seed": SAMPLE_SEED,
        "maximum_rows_per_fold": MAX_ROWS,
        "sampling": "Sorted assessment IDs; uniform without replacement, same fixed RNG seed independently per fold; target-blind, TARGET never queried",
        "aggregation": "Sum absolute native TreeSHAP encoded-column contributions by raw input, then source group; mean within each fold sample; equal mean of five fold magnitudes. Group sums are cumulative magnitudes, not signed net effects or predictive performance shares.",
        "units": "Raw margin (natural-log odds)",
        "limitations": "Sampled fitted-model behavior; correlated/derived inputs can share attribution. Group size affects cumulative magnitude. Not causal effects, predictive-value ablation, probability-point changes or adverse-action reasons.",
        "checks": checks,
        "frozen_models_sha256": model_hashes,
        "local_manifest_sha256": sha256(source / "manifest.json"),
        "scientific_source_fingerprints": manifest["fingerprints"],
        "diagnostic_source_sha256": {
            "src/assessment_model_inputs.py": sha256(Path(__file__)),
            "src/explain.py": sha256(ROOT / "src/explain.py"),
        },
    }
    # Scientific fingerprints contain dirty-status/source hashes, never role IDs.
    provenance["outputs_sha256"] = {
        n: sha256(destination / n)
        for n in (
            "fold_importance.csv",
            "importance_summary.csv",
            "input_dictionary.csv",
        )
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "methods.md").write_text(
        "# Current assessed model inputs\n\nAll five frozen current history models; 1,000 uniformly sampled assessment applicants per fold, seed 20261004, declared before computation. Sampling never reads the outcome. Saved fitting-only preprocessing and native LightGBM TreeSHAP are reused without fitting or selection.\n\nAbsolute encoded-column contributions are summed by original field (including all one-hot categories), then by source group; each sample is averaged and five fold magnitudes receive equal weight. Quantities are cumulative mean absolute raw-margin contributions in natural-log-odds units, not signed net effects, probability points or shares of predictive performance. Larger groups can accumulate more magnitude; correlated and derived inputs share information.\n\nIntercept-inclusive signed contributions reproduce each sampled raw margin within 1e-8; residuals and source/model identities are in provenance.json. Exact IDs, individual scores and SHAP arrays stay local and are not exported. This describes fitted-model behavior, not causality, predictive-value ablation or adverse-action reasons. The renderer reads these anonymous tables without data/models.\n",
        encoding="utf-8",
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(
        f"Interpreted five frozen models; {sum(c['sample_count'] for c in result['checks'])} sampled rows"
    )


if __name__ == "__main__":
    main()
