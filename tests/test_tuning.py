import copy

import numpy as np
import pandas as pd
import pytest

from src.tuning import candidate_specs, inner_partitions, joint_search


def config():
    return {
        "project": {"model_seed": 42, "random_seed": 42},
        "resources": {"model_threads": 1},
        "model": {
            "use_class_weighting": True,
            "lightgbm_tuning": {
                "mode": "bounded_inner_cv",
                "max_candidates": 24,
                "max_rounds": 35,
                "stopping_rounds": 5,
            },
        },
    }


def frame():
    rng = np.random.default_rng(7)
    target = np.tile([0, 0, 0, 1], 100)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(400),
            "TARGET": target,
            "a": target + rng.normal(size=400),
            "b": rng.normal(size=400),
            "c": rng.normal(size=400),
        }
    )


def test_joint_budget_is_unique_reproducible_spans_surfaces_and_unweighted():
    cfg = config()
    first = candidate_specs(cfg, ["top_40", "top_80", "full"])
    assert first == candidate_specs(cfg, ["top_40", "top_80", "full"])
    assert len(first) == len({row["candidate_name"] for row in first}) == 24
    assert {row["feature_set"] for row in first} == {"top_40", "top_80", "full"}
    assert all(
        sum(row["feature_set"] == surface for row in first) == 8
        for surface in ["top_40", "top_80", "full"]
    )
    assert {row["weight_fraction"] for row in first} == {0, 0.25, 0.5, 1}
    cfg["project"]["model_seed"] = 211
    assert first == candidate_specs(cfg, ["top_40", "top_80", "full"])


def test_stopping_scoring_and_fitting_are_disjoint_and_score_once():
    data = frame()
    plans = inner_partitions(data, config())
    seen = []
    for plan in plans:
        fit, stop, score = [
            set(plan[role]) for role in ("fitting", "stopping", "scoring")
        ]
        assert not (fit & stop or fit & score or stop & score)
        assert fit | stop | score == set(range(len(data)))
        seen.extend(score)
    assert sorted(seen) == list(range(len(data)))


def test_ranking_stopping_and_actual_rounds_are_fold_local(monkeypatch):
    from lightgbm import LGBMClassifier

    from src.feature_experiments import training_feature_ranking

    data, cfg, rank_ids, stopping_ids = frame(), config(), [], []
    original_fit = LGBMClassifier.fit

    def rank(training, columns, config, **kwargs):
        rank_ids.append(set(training["SK_ID_CURR"]))
        return training_feature_ranking(training, columns, config, **kwargs)

    def fit(model, x, y, **kwargs):
        if "eval_X" in kwargs:
            stopping_ids.append((set(x.index), set(kwargs["eval_X"].index)))
        return original_fit(model, x, y, **kwargs)

    monkeypatch.setattr("src.feature_experiments.training_feature_ranking", rank)
    monkeypatch.setattr(LGBMClassifier, "fit", fit)
    result = joint_search(cfg, data, ["a", "b", "c"], 0.1, ["top_2", "full"])
    evidence = result["search_evidence"]
    assert result["candidate_count"] == 24
    assert len(evidence["cv_results"]) == 72
    assert len(stopping_ids) == 72
    for fold, membership in enumerate(evidence["inner_memberships"]):
        assert rank_ids[fold] == set(membership["fitting"])
        for fit_ids, stop_ids in stopping_ids[fold * 24 : (fold + 1) * 24]:
            assert fit_ids == set(membership["fitting"])
            assert stop_ids == set(membership["stopping"])
            assert not (fit_ids | stop_ids) & set(membership["scoring"])
    chosen = result["selected_candidate"]
    iterations = [
        row["best_iteration"]
        for row in evidence["cv_results"]
        if row["candidate_name"] == chosen["candidate_name"]
    ]
    assert chosen["params"]["n_estimators"] == max(1, int(np.median(iterations)))
    assert (
        result["pipeline"].named_steps["classifier"].n_estimators_
        == chosen["params"]["n_estimators"]
    )
    assert all(
        row["params"]["feature_pre_filter"] is False for row in result["candidates"]
    )


def test_model_seed_does_not_change_candidate_recipe_or_memberships():
    cfg = config()
    changed = copy.deepcopy(cfg)
    changed["project"]["model_seed"] = 101
    assert candidate_specs(cfg, ["full"]) == candidate_specs(changed, ["full"])
    for before, after in zip(
        inner_partitions(frame(), cfg), inner_partitions(frame(), changed), strict=True
    ):
        for role in ("fitting", "stopping", "scoring"):
            assert np.array_equal(before[role], after[role])


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_budget_is_rejected(value):
    cfg = config()
    cfg["model"]["lightgbm_tuning"]["max_candidates"] = value
    with pytest.raises(ValueError):
        candidate_specs(cfg, ["full"])
