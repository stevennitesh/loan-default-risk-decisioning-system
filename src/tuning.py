"""One bounded, training-only owner for joint feature/parameter selection."""

from __future__ import annotations

import hashlib
import json
from time import perf_counter

import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from src.config import project_model_seed
from src.metrics import probability_metrics
from src.modeling import (
    _lightgbm_selection_key,
    _lightgbm_selection_score,
    build_baseline_pipeline,
    build_lightgbm_pipeline,
    classify_feature_columns,
    lightgbm_params,
)
from src.runtime import feature_frame

PROTOCOL = "nested_inner_cv_v3"


def uses_inner_cv(config):
    # Missing mode is the preserved, pre-v3 config contract. Current configs
    # explicitly declare bounded_inner_cv; historical files stay byte-identical.
    return (
        config.get("model", {}).get("lightgbm_tuning", {}).get("mode")
        == "bounded_inner_cv"
    )


def settings(config):
    value = {
        "max_candidates": 24,
        "inner_folds": 3,
        "search_seed": 20261004,
        "cv_seed": 42,
        "stopping_fraction": 0.15,
        "max_rounds": 600,
        "stopping_rounds": 40,
        "brier_tolerance": 0.002,
        "log_loss_tolerance": 0.01,
        "sensitivity_seeds": [101, 211, 307],
        **config.get("model", {}).get("lightgbm_tuning", {}),
    }
    for key in ("max_candidates", "inner_folds", "max_rounds", "stopping_rounds"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"Invalid tuning {key}")
    if value["inner_folds"] < 2 or not 0 < value["stopping_fraction"] < 0.5:
        raise ValueError(
            "Inner CV requires >=2 folds and disjoint stopping fraction in (0, .5)"
        )
    for key in ("brier_tolerance", "log_loss_tolerance"):
        if not np.isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"Invalid tuning {key}")
    for key in ("search_seed", "cv_seed"):
        if type(value[key]) is not int or not 0 <= value[key] < 2**32:
            raise ValueError(f"Invalid tuning {key}")
    seeds = value["sensitivity_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError("Sensitivity seeds must be distinct uint32 integers")
    return value


def candidate_specs(config, surfaces):
    """Deterministic unique JOINT candidates, evenly spanning declared surfaces."""
    cfg = settings(config)
    rng = np.random.default_rng(cfg["search_seed"])
    rows, seen = [], set()
    for index in range(cfg["max_candidates"]):
        surface = surfaces[index % len(surfaces)]
        while True:
            params = {
                "learning_rate": float(rng.choice([0.025, 0.04, 0.06, 0.08])),
                "num_leaves": int(rng.choice([15, 31, 47, 63])),
                "max_depth": int(rng.choice([-1, 6, 8])),
                "min_child_samples": int(rng.choice([30, 60, 120, 240])),
                "subsample": float(rng.choice([0.7, 0.85, 1.0])),
                "colsample_bytree": float(rng.choice([0.7, 0.85, 1.0])),
                "reg_alpha": float(rng.choice([0.0, 0.1, 1.0])),
                "reg_lambda": float(rng.choice([1.0, 4.0, 12.0])),
            }
            # Each surface sees unweighted, light and balanced cases; ratio is
            # computed from that fitting partition, never held-out labels.
            weight = [0.0, 0.25, 0.5, 1.0][(index // len(surfaces)) % 4]
            identity = json.dumps([surface, params, weight], sort_keys=True)
            if identity not in seen:
                seen.add(identity)
                break
        rows.append(
            {
                "candidate_name": f"random_{index + 1:02d}_{surface}",
                "candidate_source": PROTOCOL,
                "feature_set": surface,
                "weight_fraction": weight,
                "search_params": params,
            }
        )
    return rows


def inner_partitions(frame, config):
    """Score folds are never stopping folds; all rows belong to base fitting."""
    cfg = settings(config)
    splitter = StratifiedKFold(
        cfg["inner_folds"], shuffle=True, random_state=cfg["cv_seed"]
    )
    plans = []
    for fold, (training, scoring) in enumerate(
        splitter.split(frame, frame["TARGET"]), 1
    ):
        fitting, stopping = train_test_split(
            training,
            test_size=cfg["stopping_fraction"],
            stratify=frame.iloc[training]["TARGET"],
            random_state=cfg["cv_seed"] + fold,
        )
        plans.append(
            {
                "inner_fold": fold,
                "fitting": fitting,
                "stopping": stopping,
                "scoring": scoring,
            }
        )
    return plans


def _params(config, fitting, spec, seed, rounds):
    params = lightgbm_params(config, fitting, seed)
    ratio = (len(fitting) - fitting["TARGET"].sum()) / fitting["TARGET"].sum()
    params.update(spec["search_params"])
    params.update(
        n_estimators=rounds,
        feature_pre_filter=False,
        metric="average_precision",
        scale_pos_weight=1.0 + spec["weight_fraction"] * (ratio - 1.0),
    )
    return params


def _features(surface, columns, ranking):
    return (
        list(columns)
        if surface == "full"
        else [row["feature_name"] for row in ranking][: int(surface.split("_")[1])]
    )


def _quality(metrics, reference, cfg):
    return all(
        metrics[name] <= reference[name] + cfg[tolerance]
        for name, tolerance in [
            ("brier_score", "brier_tolerance"),
            ("log_loss", "log_loss_tolerance"),
        ]
    )


def joint_search(config, training, columns, review_rate, surfaces=None):
    """Search only base fitting; refit selected fixed rounds on that entire role."""
    from src.feature_experiments import training_feature_ranking

    cfg = settings(config)
    training = training.sort_values("SK_ID_CURR").reset_index(drop=True)
    surfaces = surfaces or [
        f"top_{limit}" for limit in (40, 80) if limit < len(columns)
    ] + ["full"]
    specs = candidate_specs(config, surfaces)
    seed = project_model_seed(config)
    evaluations, ranking_rows, memberships = [], [], []
    results = {row["candidate_name"]: [] for row in specs}
    started = perf_counter()
    for plan in inner_partitions(training, config):
        fold = plan["inner_fold"]
        fit, stop, score = [
            training.iloc[plan[role]] for role in ("fitting", "stopping", "scoring")
        ]
        memberships.append(
            {
                "inner_fold": fold,
                **{
                    role: training.iloc[plan[role]]["SK_ID_CURR"].astype(int).tolist()
                    for role in ("fitting", "stopping", "scoring")
                },
            }
        )
        ranking = (
            training_feature_ranking(fit, columns, config)
            if any(s != "full" for s in surfaces)
            else []
        )
        ranking_rows.extend({"inner_fold": fold, **row} for row in ranking)
        cache = {}
        reference = probability_metrics(
            score["TARGET"], np.full(len(score), fit["TARGET"].mean()), review_rate
        )
        for spec in specs:
            surface = spec["feature_set"]
            if surface not in cache:
                features = _features(surface, columns, ranking)
                numeric, categorical = classify_feature_columns(fit, features)
                transforms = build_lightgbm_pipeline(
                    numeric,
                    categorical,
                    _params(config, fit, spec, seed, cfg["max_rounds"]),
                )[:-1]
                x_fit = transforms.fit_transform(
                    feature_frame(fit, features), fit["TARGET"]
                )
                cache[surface] = (
                    x_fit,
                    transforms.transform(feature_frame(stop, features)),
                    transforms.transform(feature_frame(score, features)),
                )
            x_fit, x_stop, x_score = cache[surface]
            params = _params(config, fit, spec, seed, cfg["max_rounds"])
            model = LGBMClassifier(**params)
            clock = perf_counter()
            model.fit(
                x_fit,
                fit["TARGET"],
                eval_X=x_stop,
                eval_y=stop["TARGET"],
                eval_metric="average_precision",
                callbacks=[
                    early_stopping(
                        cfg["stopping_rounds"], first_metric_only=True, verbose=False
                    )
                ],
            )
            metrics = probability_metrics(
                score["TARGET"], model.predict_proba(x_score)[:, 1], review_rate
            )
            row = {
                **spec,
                "inner_fold": fold,
                "best_iteration": int(model.best_iteration_),
                "seconds": perf_counter() - clock,
                "probability_accepted": _quality(metrics, reference, cfg),
                **metrics,
                **{
                    f"prevalence_{name}": reference[name]
                    for name in ("brier_score", "log_loss")
                },
            }
            evaluations.append(row)
            results[spec["candidate_name"]].append(row)
        # Transform caches die with this exact fold; they never cross row sets.
    candidates = []
    for spec in specs:
        rows = results[spec["candidate_name"]]
        metrics = {
            name: float(np.mean([row[name] for row in rows]))
            for name in (
                "pr_auc",
                "roc_auc",
                "brier_score",
                "log_loss",
                "top_decile_lift",
                "precision_at_top_decile",
                "recall_at_manual_review_capacity",
                "max_predicted_probability",
                "min_predicted_probability",
            )
        }
        reference = {
            name: float(np.mean([row[f"prevalence_{name}"] for row in rows]))
            for name in ("brier_score", "log_loss")
        }
        accepted = _quality(metrics, reference, cfg)
        candidates.append(
            {
                **spec,
                "params": _params(
                    config,
                    training,
                    spec,
                    seed,
                    max(1, int(np.median([row["best_iteration"] for row in rows]))),
                ),
                "validation_metrics": metrics,
                "probability_accepted": accepted,
                "selection_key": (accepted, *_lightgbm_selection_key(metrics)[1:]),
                "validation_selection_score": _lightgbm_selection_score(metrics),
            }
        )
    ranked = sorted(candidates, key=lambda row: row["selection_key"], reverse=True)
    chosen = ranked[0]
    # If no recipe passes, preserve ranking choice and report explicit failure;
    # do not borrow calibration/selection/outer labels to retry the search.
    ranking = (
        training_feature_ranking(training, columns, config)
        if chosen["feature_set"] != "full"
        else []
    )
    features = _features(chosen["feature_set"], columns, ranking)
    numeric, categorical = classify_feature_columns(training, features)
    pipeline = build_lightgbm_pipeline(numeric, categorical, chosen["params"])
    pipeline.fit(feature_frame(training, features), training["TARGET"])
    chosen["pipeline"] = pipeline
    identity = hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
    return {
        "enabled": True,
        "max_candidates": cfg["max_candidates"],
        "candidate_count": len(candidates),
        "selection_metric_order": [
            "probability_acceptance",
            "pr_auc",
            "top_decile_lift",
            "recall_at_manual_review_capacity",
            "roc_auc",
            "brier_score",
        ],
        "candidates": candidates,
        "ranked_candidates": ranked,
        "selected_candidate": chosen,
        "pipeline": pipeline,
        "feature_columns": features,
        "feature_set": chosen["feature_set"],
        "search_evidence": {
            "protocol": PROTOCOL,
            "settings": cfg,
            "candidate_sha256": identity,
            "candidate_specs": specs,
            "inner_memberships": memberships,
            "cv_results": evaluations,
            "training_rankings": ranking_rows,
            "selected_rounds": chosen["params"]["n_estimators"],
            "probability_accepted": chosen["probability_accepted"],
            "search_seconds": perf_counter() - started,
        },
        "final_rankings": ranking,
    }


def tuned_logistic(config, training, columns, review_rate):
    """Modest C search on the identical training-only inner score partitions."""
    cfg = settings(config)
    values = (
        config.get("model", {})
        .get("logistic_tuning", {})
        .get("c_values", [0.01, 0.1, 1.0, 10.0])
    )
    rows = []
    for plan in inner_partitions(training, config):
        # Logistic has no stopping: it may fit both non-score roles.
        fit = training.iloc[np.concatenate([plan["fitting"], plan["stopping"]])]
        score = training.iloc[plan["scoring"]]
        numeric, categorical = classify_feature_columns(fit, columns)
        template = build_baseline_pipeline(
            config, numeric, categorical, project_model_seed(config)
        )
        preprocessor = template.named_steps["preprocessor"]
        x_fit = preprocessor.fit_transform(feature_frame(fit, columns), fit["TARGET"])
        x_score = preprocessor.transform(feature_frame(score, columns))
        reference = probability_metrics(
            score["TARGET"], np.full(len(score), fit["TARGET"].mean()), review_rate
        )
        for c in values:
            from sklearn.base import clone

            model = clone(template.named_steps["classifier"])
            model.set_params(C=c, class_weight=None)
            with threadpool_limits(
                limits=config.get("resources", {}).get("model_threads", 4)
            ):
                model.fit(x_fit, fit["TARGET"])
            metrics = probability_metrics(
                score["TARGET"], model.predict_proba(x_score)[:, 1], review_rate
            )
            rows.append(
                {
                    "C": c,
                    "inner_fold": plan["inner_fold"],
                    **metrics,
                    **{
                        f"prevalence_{name}": reference[name]
                        for name in ("brier_score", "log_loss")
                    },
                }
            )
    choices = []
    for c in values:
        group = [row for row in rows if row["C"] == c]
        metrics = {
            name: float(np.mean([row[name] for row in group]))
            for name in (
                "pr_auc",
                "brier_score",
                "log_loss",
                "roc_auc",
                "top_decile_lift",
                "recall_at_manual_review_capacity",
                "max_predicted_probability",
                "min_predicted_probability",
            )
        }
        reference = {
            name: float(np.mean([row[f"prevalence_{name}"] for row in group]))
            for name in ("brier_score", "log_loss")
        }
        choices.append(
            (
                (
                    _quality(metrics, reference, cfg),
                    *_lightgbm_selection_key(metrics)[1:],
                ),
                c,
            )
        )
    chosen = max(choices)[1]
    numeric, categorical = classify_feature_columns(training, columns)
    pipeline = build_baseline_pipeline(
        config, numeric, categorical, project_model_seed(config)
    )
    pipeline.named_steps["classifier"].set_params(C=chosen, class_weight=None)
    with threadpool_limits(limits=config.get("resources", {}).get("model_threads", 4)):
        pipeline.fit(feature_frame(training, columns), training["TARGET"])
    return pipeline, {
        "candidate_name": f"tuned_C{chosen}_unweighted",
        "params": pipeline.named_steps["classifier"].get_params(),
        "search_evidence": {
            "protocol": PROTOCOL,
            "c_values": values,
            "cv_results": rows,
        },
    }


def seed_sensitivity(config, roles, fitted, review_rate):
    """Vary model seeds for the chosen fixed recipe; never choose a seed."""
    from sklearn.base import clone

    from src.calibration import apply_saved_calibration_artifact, fit_calibrators

    rows = []
    columns = fitted["feature_columns"]
    for seed in settings(config)["sensitivity_seeds"]:
        pipeline = clone(fitted["pipeline"])
        pipeline.named_steps["classifier"].set_params(random_state=seed)
        with threadpool_limits(
            limits=config.get("resources", {}).get("model_threads", 4)
        ):
            pipeline.fit(
                feature_frame(roles["train"], columns), roles["train"]["TARGET"]
            )
        raw_calibration = pipeline.predict_proba(
            feature_frame(roles["calibration"], columns)
        )[:, 1]
        calibrators = fit_calibrators(
            raw_calibration, roles["calibration"]["TARGET"].to_numpy(), seed
        )
        raw = pipeline.predict_proba(feature_frame(roles["validation"], columns))[:, 1]
        calibrated = apply_saved_calibration_artifact(
            raw,
            {
                "selected_method": fitted["selected_calibration_method"],
                "calibrators": calibrators,
            },
        )
        for kind, probabilities in [("raw", raw), ("calibrated", calibrated)]:
            rows.append(
                {
                    "model_seed": seed,
                    "score_kind": kind,
                    "candidate_name": fitted["selected_candidate"]["candidate_name"],
                    "calibration_method": fitted["selected_calibration_method"],
                    "evaluation_role": "selection_validation",
                    "promotes_seed": False,
                    **probability_metrics(
                        roles["validation"]["TARGET"], probabilities, review_rate
                    ),
                }
            )
    return rows
