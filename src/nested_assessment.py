"""Assess the declared LightGBM selection procedure on fold-local unseen rows."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.calibration import apply_saved_calibration_artifact
from src.cli import add_config_argument, exit_with_error
from src.config import (
    business_assumptions,
    load_config,
    manual_review_capacity_rate,
    project_model_seed,
    threshold_version,
)
from src.data_contracts import get_model_feature_columns
from src.evidence import fingerprints
from src.feature_experiments import (
    calibrate_candidate,
    feature_sets,
    fit_feature_candidate,
    fit_joint_workflow,
    load_lightgbm_artifact,
    load_split_frames,
    ranking_seeds,
    select_feature_set,
    training_feature_ranking,
)
from src.mart_access import table_columns
from src.metrics import (
    build_calibration_bin_rows,
    probability_metrics,
    target_class_values,
)
from src.model_artifacts import normalize_split_ids, validate_feature_build
from src.modeling import (
    build_baseline_pipeline,
    classify_feature_columns,
    predict_probabilities,
    prediction_frame,
)
from src.presentation import metric_label, score_label, workflow_label
from src.report_contracts import (
    NESTED_CANDIDATE_COLUMNS,
    NESTED_METRIC_COLUMNS,
    NESTED_PREDICTION_COLUMNS,
    NESTED_PROBABILITY_ACCEPTANCE_COLUMNS,
    NESTED_RANKING_COLUMNS,
    NESTED_RELIABILITY_COLUMNS,
    NESTED_SEED_SENSITIVITY_COLUMNS,
    NESTED_SENSITIVITY_COLUMNS,
    NESTED_SUMMARY_COLUMNS,
)
from src.runtime import created_at_utc, feature_frame, resolve_config_path, write_csv
from src.thresholding import BALANCED_SCENARIO, build_threshold_metric_rows

PROTOCOL_VERSION = "nested_matched_holdout_v2"
FIT_ROLES = ("train", "calibration", "validation")
WORKFLOWS = {
    "history_selected": "lightgbm",
    "training_prevalence": "prevalence",
    "logistic_fixed": "logistic_regression",
    "logistic_tuned": "logistic_regression",
    "application_only": "lightgbm",
}


class NestedAssessmentError(RuntimeError):
    """Raised when an assessment cannot preserve its population boundaries."""


def assessment_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "outer_folds": 5,
        "split_seeds": [42],
        "fit_fraction": 0.70,
        "calibration_fraction": 0.15,
        "selection_fraction": 0.15,
        "feature_limits": [40, 80],
        "include_full": True,
        "workflows": [
            "history_selected",
            "training_prevalence",
            "logistic_tuned"
            if config.get("model", {}).get("lightgbm_tuning", {}).get("mode")
            == "bounded_inner_cv"
            else "logistic_fixed",
            "application_only",
        ],
        "negative_control_sample_size": None,
        "negative_control_seed": 913,
        "utility_multipliers": [0.5, 1.0, 2.0],
        **config.get("assessment", {}),
    }
    folds = settings["outer_folds"]
    if type(folds) is not int or folds < 2:
        raise NestedAssessmentError("assessment.outer_folds must be an integer >= 2")
    seeds = settings["split_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise NestedAssessmentError(
            "assessment.split_seeds must be distinct nonnegative integer seeds"
        )
    fractions = [
        settings[name]
        for name in ("fit_fraction", "calibration_fraction", "selection_fraction")
    ]
    if any(
        type(value) not in (int, float) or not np.isfinite(value) or value <= 0
        for value in fractions
    ) or not np.isclose(sum(fractions), 1, rtol=0, atol=1e-12):
        raise NestedAssessmentError(
            "assessment role fractions must be positive and sum to one"
        )
    limits = settings["feature_limits"]
    if (
        not isinstance(limits, list)
        or any(type(limit) is not int or limit <= 0 for limit in limits)
        or len(set(limits)) != len(limits)
    ):
        raise NestedAssessmentError(
            "assessment.feature_limits must contain distinct positive integers"
        )
    if type(settings["include_full"]) is not bool or not (
        limits or settings["include_full"]
    ):
        raise NestedAssessmentError("assessment must request at least one feature set")
    ranking_seeds(config, NestedAssessmentError)
    if (
        not isinstance(settings["workflows"], list)
        or not settings["workflows"]
        or len(set(settings["workflows"])) != len(settings["workflows"])
        or not set(settings["workflows"]) <= WORKFLOWS.keys()
    ):
        raise NestedAssessmentError(
            "assessment.workflows must be distinct declared workflows"
        )
    sample_size = settings["negative_control_sample_size"]
    if sample_size is not None and (type(sample_size) is not int or sample_size < 100):
        raise NestedAssessmentError(
            "negative control sample size must be an integer >= 100"
        )
    multipliers = settings["utility_multipliers"]
    if (
        not isinstance(multipliers, list)
        or not multipliers
        or any(
            type(x) not in (float, int) or not np.isfinite(x) or x <= 0
            for x in multipliers
        )
    ):
        raise NestedAssessmentError(
            "utility multipliers must be finite positive numbers"
        )
    if not 0 <= project_model_seed(config) < 2**32:
        raise NestedAssessmentError("project.model_seed must be in [0, 2**32)")
    return settings


def make_fold_plan(
    development: pd.DataFrame, comparison_ids: list[int], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Freeze all fold/role memberships before fitting any candidate."""
    frame = development.sort_values("SK_ID_CURR").reset_index(drop=True)
    if frame["SK_ID_CURR"].isna().any() or frame["SK_ID_CURR"].duplicated().any():
        raise NestedAssessmentError(
            "Development applicant IDs must be known and unique"
        )
    normalize_split_ids(
        {
            "development": list(frame["SK_ID_CURR"]),
            "historical_comparison": comparison_ids,
        },
        ("development", "historical_comparison"),
        error_cls=NestedAssessmentError,
    )
    if target_class_values(frame["TARGET"], error_cls=NestedAssessmentError) != {0, 1}:
        raise NestedAssessmentError("Development must contain both target classes")
    if frame["TARGET"].value_counts().min() < settings["outer_folds"]:
        raise NestedAssessmentError(
            "Each class needs at least outer_folds development applicants"
        )
    plans = []
    for seed in settings["split_seeds"]:
        splitter = StratifiedKFold(
            n_splits=settings["outer_folds"], shuffle=True, random_state=seed
        )
        for fold, (fit_positions, assessment_positions) in enumerate(
            splitter.split(frame, frame["TARGET"]), 1
        ):
            outer_train = frame.iloc[fit_positions]
            inner_seed = int(
                np.random.SeedSequence([seed, fold, 739]).generate_state(1)[0]
            )
            try:
                fit, reserved = train_test_split(
                    outer_train,
                    test_size=settings["calibration_fraction"]
                    + settings["selection_fraction"],
                    stratify=outer_train["TARGET"],
                    random_state=inner_seed,
                )
                calibration, selection = train_test_split(
                    reserved,
                    test_size=settings["selection_fraction"]
                    / (
                        settings["calibration_fraction"]
                        + settings["selection_fraction"]
                    ),
                    stratify=reserved["TARGET"],
                    random_state=inner_seed,
                )
            except ValueError as error:
                raise NestedAssessmentError(
                    f"Insufficient class support for fold {fold} roles: {error}"
                ) from error
            roles = {
                "train": fit,
                "calibration": calibration,
                "validation": selection,
                "assessment": frame.iloc[assessment_positions],
            }
            for name, role in roles.items():
                if target_class_values(
                    role["TARGET"], error_cls=NestedAssessmentError
                ) != {0, 1}:
                    raise NestedAssessmentError(
                        f"Fold {fold} {name} must contain both classes"
                    )
            ids = {
                name: sorted(int(value) for value in role["SK_ID_CURR"])
                for name, role in roles.items()
            }
            normalize_split_ids(
                {**ids, "historical_comparison": comparison_ids},
                (*ids, "historical_comparison"),
                error_cls=NestedAssessmentError,
            )
            plans.append(
                {
                    "split_seed": seed,
                    "outer_fold": fold,
                    "inner_split_seed": inner_seed,
                    "applicant_ids": ids,
                }
            )
    return plans


def fit_fold_workflow(
    config: dict[str, Any],
    roles: dict[str, pd.DataFrame],
    columns: list[str],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose the entire workflow using only the three inner roles."""
    if set(roles) != set(FIT_ROLES):
        raise NestedAssessmentError(
            "Workflow fitting must not receive assessment/comparison rows"
        )
    normalize_split_ids(
        {name: list(frame["SK_ID_CURR"]) for name, frame in roles.items()},
        FIT_ROLES,
        error_cls=NestedAssessmentError,
    )
    from src.tuning import uses_inner_cv

    if uses_inner_cv(config):
        surfaces = [f"top_{limit}" for limit in settings["feature_limits"]] + (
            ["full"] if settings["include_full"] else []
        )
        selected = fit_joint_workflow(
            config, roles, columns, manual_review_capacity_rate(config), surfaces
        )
        selected["model_family"] = "lightgbm"
        selected["fit_applicant_ids"] = sorted(
            set().union(*(set(frame["SK_ID_CURR"]) for frame in roles.values()))
        )
        return selected, [selected["selection_row"]], selected["final_rankings"]
    ranking = (
        training_feature_ranking(
            roles["train"], columns, config, error_cls=NestedAssessmentError
        )
        if settings["feature_limits"]
        else []
    )
    specs = feature_sets(
        [row["feature_name"] for row in ranking],
        columns,
        tuple(settings["feature_limits"]),
        settings["include_full"],
        NestedAssessmentError,
    )
    candidates = {}
    rows = []
    for name, features, limit in specs:
        fitted = fit_feature_candidate(
            config,
            name,
            features,
            limit,
            roles,
            manual_review_capacity_rate(config),
            project_model_seed(config),
            error_cls=NestedAssessmentError,
        )
        candidates[name] = fitted
        rows.append(fitted["selection_row"])
    selected_name = select_feature_set(rows)
    for row in rows:
        row["selected"] = row["feature_set"] == selected_name
    selected = candidates[selected_name]
    selected["model_family"] = "lightgbm"
    selected["fit_applicant_ids"] = sorted(
        set().union(*(set(frame["SK_ID_CURR"]) for frame in roles.values()))
    )
    return selected, rows, ranking


def fit_comparator(config, roles, columns, application_columns, name):
    """Fit independent declared baselines on exactly the same reserved roles."""
    if set(roles) != set(FIT_ROLES):
        raise NestedAssessmentError("Comparator fitting accepts only inner roles")
    normalize_split_ids(
        {role: list(frame["SK_ID_CURR"]) for role, frame in roles.items()},
        FIT_ROLES,
        error_cls=NestedAssessmentError,
    )
    seed = project_model_seed(config)
    if name == "application_only":
        if not application_columns:
            raise NestedAssessmentError(
                "Application-only workflow needs SQL-owned application features"
            )
        from src.tuning import uses_inner_cv

        if uses_inner_cv(config):
            fitted = fit_joint_workflow(
                config,
                roles,
                application_columns,
                manual_review_capacity_rate(config),
                ["full"],
            )
            fitted["selection_row"]["feature_set"] = name
        else:
            fitted = fit_feature_candidate(
                config,
                name,
                application_columns,
                None,
                roles,
                manual_review_capacity_rate(config),
                seed,
                error_cls=NestedAssessmentError,
            )
    elif name == "logistic_tuned":
        from src.tuning import tuned_logistic

        pipeline, selected = tuned_logistic(
            config, roles["train"], columns, manual_review_capacity_rate(config)
        )
        fitted = calibrate_candidate(
            config,
            pipeline,
            name,
            columns,
            None,
            roles,
            manual_review_capacity_rate(config),
            seed,
            selected,
            error_cls=NestedAssessmentError,
        )
        fitted["search_evidence"] = selected["search_evidence"]
        from src.tuning import seed_sensitivity

        fitted["seed_sensitivity"] = seed_sensitivity(
            config, roles, fitted, manual_review_capacity_rate(config)
        )
    elif name == "logistic_fixed":
        numeric, categorical = classify_feature_columns(roles["train"], columns)
        pipeline = build_baseline_pipeline(config, numeric, categorical, seed)
        pipeline.fit(
            feature_frame(roles["train"], columns), roles["train"]["TARGET"].astype(int)
        )
        fitted = calibrate_candidate(
            config,
            pipeline,
            name,
            columns,
            None,
            roles,
            manual_review_capacity_rate(config),
            seed,
            {
                "candidate_name": "fixed_C1_lbfgs",
                "params": pipeline.named_steps["classifier"].get_params(),
            },
            error_cls=NestedAssessmentError,
        )
    elif name == "training_prevalence":
        pipeline = DummyClassifier(strategy="prior")
        pipeline.fit(
            feature_frame(roles["train"], []), roles["train"]["TARGET"].astype(int)
        )
        fitted = {
            "pipeline": pipeline,
            "feature_columns": [],
            "selected_calibration_method": "uncalibrated",
            "calibrators": {},
            "scenario_thresholds": None,
            "selected_candidate": {"candidate_name": "training_prevalence"},
            "selection_row": {
                "feature_set": name,
                "selected_calibration_method": "uncalibrated",
            },
        }
    else:
        raise NestedAssessmentError(f"Unknown comparator {name}")
    fitted["model_family"] = WORKFLOWS[name]
    fitted["fit_applicant_ids"] = sorted(
        set().union(*(set(frame["SK_ID_CURR"]) for frame in roles.values()))
    )
    return fitted


def assess_workflow(
    workflow: dict[str, Any],
    assessment: pd.DataFrame,
    config: dict[str, Any],
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply a frozen selected workflow; this function makes no choices."""
    if set(assessment["SK_ID_CURR"]) & set(workflow["fit_applicant_ids"]):
        raise NestedAssessmentError(
            "Assessment applicants overlap fitted/selection roles"
        )
    if target_class_values(assessment["TARGET"], error_cls=NestedAssessmentError) != {
        0,
        1,
    }:
        raise NestedAssessmentError("Assessment must contain both target classes")
    raw = predict_probabilities(
        {"pipeline": workflow["pipeline"], "model_version": PROTOCOL_VERSION},
        assessment,
        workflow["feature_columns"],
        "outer assessment",
        NestedAssessmentError,
    )
    calibrated = apply_saved_calibration_artifact(
        raw,
        {
            "selected_method": workflow["selected_calibration_method"],
            "calibrators": workflow["calibrators"],
        },
        error_cls=NestedAssessmentError,
    )
    raw_frame = prediction_frame(assessment, raw)
    threshold_rows = (
        build_threshold_metric_rows(
            PROTOCOL_VERSION,
            threshold_version(config),
            {"assessment": raw_frame},
            workflow["scenario_thresholds"],
            business_assumptions(config),
            context["created_at"],
            splits=("assessment",),
        )
        if workflow["scenario_thresholds"] is not None
        else []
    )
    utility = (
        next(
            row["expected_value_per_applicant"]
            for row in threshold_rows
            if row["scenario_name"] == BALANCED_SCENARIO
        )
        if threshold_rows
        else None
    )
    rows = []
    for kind, probabilities in (("raw", raw), ("calibrated", calibrated)):
        metrics = probability_metrics(
            assessment["TARGET"],
            probabilities,
            manual_review_capacity_rate(config),
            NestedAssessmentError,
        )
        if kind == "raw" and utility is not None:
            metrics["balanced_utility_per_applicant"] = utility
        for name, value in metrics.items():
            rows.append(
                {
                    **context,
                    "score_kind": kind,
                    "metric_name": name,
                    "metric_value": value,
                    "applicant_count": len(assessment),
                }
            )
    predictions = [
        {
            **context,
            "SK_ID_CURR": int(applicant),
            "target": int(target),
            "raw_score": float(raw_score),
            "calibrated_score": float(cal_score),
        }
        for applicant, target, raw_score, cal_score in zip(
            assessment["SK_ID_CURR"], assessment["TARGET"], raw, calibrated, strict=True
        )
    ]
    return rows, predictions


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_nested_assessment(
    config_path: str | Path = "configs/post_v1.yaml",
) -> dict[str, Any]:
    started = perf_counter()
    config = load_config(config_path)
    settings = assessment_settings(config)
    from src.tuning import uses_inner_cv

    if uses_inner_cv(config):
        settings["protocol_version"] = "nested_inner_cv_v3"
    artifact = load_lightgbm_artifact(
        resolve_config_path(
            config,
            "reference_model_dir"
            if "reference_model_dir" in config["paths"]
            else "model_dir",
        ),
        NestedAssessmentError,
    )
    ids = normalize_split_ids(
        artifact["split_applicant_ids"],
        (*FIT_ROLES, "test"),
        error_cls=NestedAssessmentError,
    )
    with duckdb.connect(
        str(resolve_config_path(config, "duckdb_path")), read_only=True
    ) as connection:
        validate_feature_build(connection, artifact, error_cls=NestedAssessmentError)
        columns = get_model_feature_columns(connection, config)
        application_columns = [
            column
            for column in table_columns(connection, "f_applicant_static")
            if column in columns
        ]
        roles = load_split_frames(
            connection,
            {name: ids[name] for name in FIT_ROLES},
            columns,
            NestedAssessmentError,
        )
    development = (
        pd.concat(list(roles.values()), ignore_index=True)
        .sort_values("SK_ID_CURR")
        .reset_index(drop=True)
    )
    negative_control = settings["negative_control_sample_size"] is not None
    if negative_control:
        size = settings["negative_control_sample_size"]
        if size > len(development):
            raise NestedAssessmentError(
                "Negative-control sample exceeds development population"
            )
        if size < len(development):
            development, _ = train_test_split(
                development,
                train_size=size,
                stratify=development["TARGET"],
                random_state=settings["negative_control_seed"],
            )
        development = development.sort_values("SK_ID_CURR").reset_index(drop=True)
        original_label_digest = _digest(
            development[["SK_ID_CURR", "TARGET"]].astype(int).values.tolist()
        )
        development["TARGET"] = np.random.default_rng(
            settings["negative_control_seed"]
        ).permutation(development["TARGET"].to_numpy())
    plans = make_fold_plan(development, ids["test"], settings)
    run_id = uuid4().hex
    created_at = created_at_utc()
    output = resolve_config_path(config, "report_dir") / "nested_assessment" / run_id
    output.mkdir(parents=True)
    manifest = {
        "protocol_version": settings.get("protocol_version", PROTOCOL_VERSION),
        "fingerprints": fingerprints(config),
        "model_family": "lightgbm",
        "workflow_families": {name: WORKFLOWS[name] for name in settings["workflows"]},
        "negative_control": negative_control,
        "original_sample_label_sha256": original_label_digest
        if negative_control
        else None,
        "application_columns": application_columns,
        "diagnostic_declarations": {
            "history_segments": "no obligations; any ambiguous/unknown obligation; remaining known history",
            "utility_grid": "all combinations of declared margin/loss/review multipliers; frozen balanced raw-score thresholds",
            "clean_reproduction": "first split seed, first fold, history_selected; absolute prediction tolerance 1e-10",
            "negative_control": "stratified sample before label permutation; seed 913; same full inner recipe and matched comparators",
        },
        "run_id": run_id,
        "created_at": created_at,
        "status": "running",
        "feature_build_id": artifact["feature_build_id"],
        "reference_model_run_id": artifact["run_id"],
        "config": config,
        "config_sha256": _digest(config),
        "settings": settings,
        "model_seed": project_model_seed(config),
        "ranking_seeds": list(ranking_seeds(config)),
        "feature_columns": columns,
        "development_id_target_sha256": _digest(
            development[["SK_ID_CURR", "TARGET"]].astype(int).values.tolist()
        ),
        "historical_comparison_ids": ids["test"],
        "historical_comparison_sha256": _digest(sorted(ids["test"])),
        "folds": plans,
        "limitations": "Previously explored dataset; random stratified assessment, not temporal or external validation. Local build identity is not full content lineage.",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metric_rows, prediction_rows, candidate_rows, rank_rows = [], [], [], []
    reliability_rows, sensitivity_rows, seed_rows, quality_rows = [], [], [], []
    try:
        for plan in plans:
            memberships = plan["applicant_ids"]
            fitting = {
                name: development.loc[
                    development["SK_ID_CURR"].isin(memberships[name])
                ].reset_index(drop=True)
                for name in FIT_ROLES
            }
            selected, choices, ranking = fit_fold_workflow(
                config, fitting, columns, settings
            )
            workflows = (
                {"history_selected": selected}
                if "history_selected" in settings["workflows"]
                else {}
            )
            workflows.update(
                {
                    name: fit_comparator(
                        config, fitting, columns, application_columns, name
                    )
                    for name in settings["workflows"]
                    if name != "history_selected"
                }
            )
            context = {
                "run_id": run_id,
                "workflow": "history_selected",
                "model_family": "lightgbm",
                "split_seed": plan["split_seed"],
                "outer_fold": plan["outer_fold"],
                "model_seed": project_model_seed(config),
                "feature_set": selected["selection_row"]["feature_set"],
                "calibration_method": selected["selected_calibration_method"],
                "created_at": created_at,
            }
            assessment = development.loc[
                development["SK_ID_CURR"].isin(memberships["assessment"])
            ].reset_index(drop=True)
            for name, frozen in workflows.items():
                workflow_context = {
                    **context,
                    "workflow": name,
                    "model_family": WORKFLOWS[name],
                    "feature_set": frozen["selection_row"]["feature_set"],
                    "calibration_method": frozen["selected_calibration_method"],
                }
                metrics, predictions = assess_workflow(
                    frozen, assessment, config, workflow_context
                )
                metric_rows.extend(metrics)
                prediction_rows.extend(predictions)
                for kind in ("raw", "calibrated"):
                    frame = pd.DataFrame(predictions).rename(
                        columns={f"{kind}_score": "probability"}
                    )
                    reliability_rows.extend(
                        {
                            **workflow_context,
                            "score_kind": kind,
                            **{
                                k: v
                                for k, v in row.items()
                                if k not in ("model_version", "split")
                            },
                        }
                        for row in build_calibration_bin_rows(
                            PROTOCOL_VERSION, {"assessment": frame}, ("assessment",)
                        )
                    )
                sensitivity_rows.extend(
                    utility_sensitivity(
                        frozen,
                        assessment,
                        predictions,
                        config,
                        settings,
                        workflow_context,
                    )
                )
                seed_rows.extend(
                    {**workflow_context, **row}
                    for row in frozen.get("seed_sensitivity", [])
                )
                quality_rows.append(
                    {**workflow_context, **frozen.get("probability_quality", {})}
                )
                if "search_evidence" in frozen:
                    (
                        output
                        / f"search_{plan['split_seed']}_{plan['outer_fold']}_{name}.json"
                    ).write_text(
                        json.dumps(frozen["search_evidence"], indent=2),
                        encoding="utf-8",
                    )
                frozen["assessment_context"] = workflow_context
                joblib.dump(
                    frozen,
                    output
                    / f"fold_{plan['split_seed']}_{plan['outer_fold']}_{name}.joblib",
                )
            candidate_rows.extend(
                {
                    **context,
                    **row,
                    "calibration_method": row["selected_calibration_method"],
                }
                for row in choices
            )
            rank_rows.extend({**context, **row} for row in ranking)
            selected["assessment_context"] = context
            joblib.dump(
                selected,
                output / f"fold_{plan['split_seed']}_{plan['outer_fold']}.joblib",
            )
            manifest.setdefault("completed_folds", []).append(
                {"split_seed": plan["split_seed"], "outer_fold": plan["outer_fold"]}
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"Completed assessment fold {plan['outer_fold']} seed {plan['split_seed']}",
                flush=True,
            )
        summary_rows = summarize_metrics(metric_rows)
        expected_ids = set(development["SK_ID_CURR"])
        for seed, name in itertools.product(
            settings["split_seeds"], settings["workflows"]
        ):
            observed = [
                row["SK_ID_CURR"]
                for row in prediction_rows
                if row["split_seed"] == seed and row["workflow"] == name
            ]
            if len(observed) != len(expected_ids) or set(observed) != expected_ids:
                raise NestedAssessmentError(
                    "Each development applicant must have exactly one assessment prediction per split seed per workflow"
                )
        write_csv(output / "fold_metrics.csv", NESTED_METRIC_COLUMNS, metric_rows)
        write_csv(
            output / "assessment_predictions.csv",
            NESTED_PREDICTION_COLUMNS,
            prediction_rows,
        )
        write_csv(
            output / "inner_selection.csv", NESTED_CANDIDATE_COLUMNS, candidate_rows
        )
        write_csv(output / "training_rankings.csv", NESTED_RANKING_COLUMNS, rank_rows)
        write_csv(output / "summary.csv", NESTED_SUMMARY_COLUMNS, summary_rows)
        write_csv(
            output / "reliability_bins.csv",
            NESTED_RELIABILITY_COLUMNS,
            reliability_rows,
        )
        write_csv(
            output / "utility_sensitivity.csv",
            NESTED_SENSITIVITY_COLUMNS,
            sensitivity_rows,
        )
        if seed_rows:
            write_csv(
                output / "model_seed_sensitivity.csv",
                NESTED_SEED_SENSITIVITY_COLUMNS,
                seed_rows,
            )
            write_csv(
                output / "probability_acceptance.csv",
                NESTED_PROBABILITY_ACCEPTANCE_COLUMNS,
                quality_rows,
            )
        write_paired_comparisons(output, metric_rows)
        manifest["selected_workflows"] = [
            {
                "workflow": name,
                "split_seed": frozen["assessment_context"]["split_seed"],
                "outer_fold": frozen["assessment_context"]["outer_fold"],
                "feature_columns": frozen["feature_columns"],
                "calibration_method": frozen["selected_calibration_method"],
                "candidate": frozen["selected_candidate"],
            }
            for path in sorted(output.glob("fold_*_*.joblib"))
            if not path.name.replace(".joblib", "").split("_")[-1].isdigit()
            for frozen in [joblib.load(path)]
            for name in [frozen["assessment_context"]["workflow"]]
        ]
        _write_report(output / "assessment_report.md", manifest, summary_rows)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["duration_seconds"] = perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "output_dir": output,
        "manifest": manifest_path,
        "summary_rows": summary_rows,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    return [
        {
            "workflow": workflow,
            "model_family": family,
            "split_seed": seed,
            "score_kind": kind,
            "metric_name": metric,
            "fold_count": len(group),
            "fold_mean": float(group["metric_value"].mean()),
            "fold_std": float(group["metric_value"].std(ddof=1)),
        }
        for (workflow, family, seed, kind, metric), group in frame.groupby(
            ["workflow", "model_family", "split_seed", "score_kind", "metric_name"],
            sort=True,
        )
    ]


def utility_sensitivity(workflow, assessment, predictions, config, settings, context):
    """Evaluate a predeclared grid at frozen raw-score scenario thresholds."""
    if workflow["scenario_thresholds"] is None:
        return []
    frame = pd.DataFrame(predictions).rename(columns={"raw_score": "probability"})
    assumptions = business_assumptions(config)
    rows = []
    for margin, loss, review in itertools.product(
        settings["utility_multipliers"], repeat=3
    ):
        weights = {
            **assumptions,
            "expected_margin_per_good_loan": assumptions[
                "expected_margin_per_good_loan"
            ]
            * margin,
            "expected_loss_per_bad_loan": assumptions["expected_loss_per_bad_loan"]
            * loss,
            "manual_review_cost": assumptions["manual_review_cost"] * review,
        }
        metrics = build_threshold_metric_rows(
            PROTOCOL_VERSION,
            threshold_version(config),
            {"assessment": frame},
            workflow["scenario_thresholds"],
            weights,
            context["created_at"],
            splits=("assessment",),
        )
        value = next(
            row["expected_value_per_applicant"]
            for row in metrics
            if row["scenario_name"] == BALANCED_SCENARIO
        )
        rows.append(
            {
                **context,
                "margin_multiplier": margin,
                "loss_multiplier": loss,
                "review_multiplier": review,
                "utility_per_applicant": value,
            }
        )
    return rows


def write_paired_comparisons(output, rows):
    """Match metrics within each fold; these differences never select workflows."""
    frame = pd.DataFrame(rows)
    keys = ["split_seed", "outer_fold", "score_kind", "metric_name"]
    reference = frame.loc[
        frame["workflow"] == "history_selected", keys + ["metric_value"]
    ]
    differences = frame.loc[frame["workflow"] != "history_selected"].merge(
        reference, on=keys, suffixes=("_comparator", "_history"), validate="many_to_one"
    )
    differences["history_minus_comparator"] = (
        differences["metric_value_history"] - differences["metric_value_comparator"]
    )
    differences.to_csv(output / "paired_differences.csv", index=False)


def _write_report(
    path: Path, manifest: dict[str, Any], summaries: list[dict[str, Any]]
) -> None:
    lines = "\n".join(
        f"| {workflow_label(row['workflow'])} | {row['split_seed']} | {score_label(row['score_kind'])} | {metric_label(row['metric_name'])} | {row['fold_mean']:.6f} | {row['fold_std']:.6f} |"
        for row in summaries
        if row["metric_name"]
        in (
            "pr_auc",
            "roc_auc",
            "brier_score",
            "log_loss",
            "balanced_utility_per_applicant",
        )
    )
    path.write_text(
        f"""# Matched applicant-group assessment

Applicant test groups (outer stratified folds) assess application-and-loan-history LightGBM, application-only LightGBM, logistic regression and the constant training-outcome-rate benchmark on the same applicants. Cross-validation repeats fitting/selection across groups; calibration is a probability adjustment, which may leave raw probabilities unchanged. Features are model inputs. Within each outer-training partition, fitting, calibration, and selection are disjoint. History model-input rankings average several fitting-only repeats. Model randomness is configured independently of applicant-grouping randomness. Inner selection chooses features, bounded model settings, calibration, and raw-score thresholds before the frozen workflow predicts the outer fold. Each development applicant is assessed exactly once per workflow per split seed. Historical comparison and unlabeled Kaggle applicants do not enter this assessment. Comparator optimization budgets differ as declared; outer results do not select between model families. Prevalence has flat scores and no quantile policy utility.

{"The v3 procedure uses three inner stratified CV folds entirely within base fitting, with additional disjoint stopping subsets, 24 joint candidates per LightGBM workflow, fixed median best rounds and train-only per-fold rankings/preprocessing. Tuned logistic searches four C values. Raw probability acceptance compares mean inner Brier/log loss to training prevalence plus declared tolerances. Search and seed sensitivity evidence never uses outer labels." if manifest["protocol_version"] == "nested_inner_cv_v3" else "The inner procedure uses a reserved holdout, not inner K-fold CV."} Results estimate this fold-trained selection procedure; they are not the metrics of the separately saved dashboard model. Different folds may choose different features/settings. The outer-training fitting fraction uses less data than the final dashboard fit.

| Model / input scope | Split seed | Probability view | Metric | Mean across applicant test groups | Descriptive group variation (SD) |
|---|---|---|---|---:|---:|
{lines}

## Interpretation and limits

Outer metrics are assessment evidence, not inputs to automatic promotion. Selecting a revised recipe after seeing these metrics makes them exploratory evidence for that revision. Fold SD describes variability and is not a confidence interval; fold fits share training data. No cross-fold pooled average precision is reported. This dataset was historically explored; nesting cannot restore untouched external evidence. Reliable calendar application dates are unavailable, so random folds do not establish future-cohort performance. Utility uses illustrative weights and raw-score quantile scenarios, with no hard review cap or estimated reviewer effectiveness.

## Technical run details

Run `{manifest["run_id"]}`; protocol `{manifest["protocol_version"]}`; feature build `{manifest["feature_build_id"]}`.
Model seed: `{manifest["model_seed"]}`; ranking seeds: `{manifest["ranking_seeds"]}`.

`manifest.json` records local role IDs, settings, seeds, and fingerprints. `inner_selection.csv` records the fold-local choices; `training_rankings.csv` records ranking dispersion. Fold artifacts reproduce the selected pipelines and calibrators. These outputs are local generated evidence and do not refresh curated metrics or Power BI snapshots.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess model selection on matched applicant test groups, keeping fitting and selection separate."
    )
    add_config_argument(parser)
    parser.set_defaults(config="configs/post_v1.yaml")
    args = parser.parse_args()
    try:
        result = run_nested_assessment(args.config)
        print(f"Nested assessment written to {result['output_dir']}")
    except (NestedAssessmentError, ValueError) as error:
        exit_with_error(error)


if __name__ == "__main__":
    main()
