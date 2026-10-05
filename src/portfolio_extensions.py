"""Validate and chart anonymous installment, utility and frozen-input evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.presentation import workflow_label

INPUT_SOURCE = Path(__file__).resolve().parents[1] / "reports/model_inputs_20261004"
INPUT_FILES = (
    "fold_importance.csv",
    "importance_summary.csv",
    "input_dictionary.csv",
    "provenance.json",
    "methods.md",
)
SEGMENTS = {
    "known_installment_history": "Known installment payments",
    "ambiguous_or_unknown_history": "Ambiguous or unknown payments",
    "no_installment_history": "No installment obligations",
}
UTILITY_WORKFLOWS = ("application_only", "logistic_tuned", "history_selected")


def validate_sensitivities(evidence: dict) -> None:
    segments = evidence["history_segment_metrics"]
    final = segments.loc[segments.score_kind == "calibrated"]
    grain = ["workflow", "split_seed", "outer_fold", "history_segment"]
    expected = set(
        itertools.product(
            evidence["fold_metrics"].workflow.unique(), [42], range(1, 6), SEGMENTS
        )
    )
    if (
        final.duplicated(grain).any()
        or set(final[grain].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Installment segments must cover every matched workflow/fold")
    reference = final.loc[final.workflow == "application_only"].set_index(
        ["outer_fold", "history_segment"]
    )
    for _, frame in final.groupby("workflow"):
        actual = frame.set_index(["outer_fold", "history_segment"])
        if not actual.applicant_count.equals(
            reference.applicant_count
        ) or not np.allclose(
            actual.target_rate, reference.target_rate, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Installment segment comparisons require identical populations"
            )
    fold_counts = (
        evidence["fold_metrics"]
        .loc[
            (evidence["fold_metrics"].workflow == "application_only")
            & (evidence["fold_metrics"].score_kind == "calibrated")
            & (evidence["fold_metrics"].metric_name == "pr_auc")
        ]
        .set_index("outer_fold")
        .applicant_count
    )
    if not reference.groupby("outer_fold").applicant_count.sum().equals(fold_counts):
        raise ValueError("Installment segments do not reconcile to assessment counts")
    utility = evidence["utility_sensitivity"]
    keys = [
        "workflow",
        "split_seed",
        "outer_fold",
        "margin_multiplier",
        "loss_multiplier",
        "review_multiplier",
    ]
    expected = set(
        itertools.product(
            UTILITY_WORKFLOWS,
            [42],
            range(1, 6),
            [0.5, 1.0, 2.0],
            [0.5, 1.0, 2.0],
            [0.5, 1.0, 2.0],
        )
    )
    if (
        utility.duplicated(keys).any()
        or set(utility[keys].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError(
            "Utility sensitivity requires all 27 assumptions for three workflows and five folds"
        )
    if set(utility.run_id) != {evidence["provenance"]["assessment_run_id"]}:
        raise ValueError("Utility sensitivity assessment identity mismatch")
    if not np.isfinite(utility.utility_per_applicant).all():
        raise ValueError("Nonfinite utility evidence")
    baseline = (
        utility.loc[
            (utility.margin_multiplier == 1)
            & (utility.loss_multiplier == 1)
            & (utility.review_multiplier == 1)
        ]
        .set_index(["workflow", "outer_fold"])
        .utility_per_applicant
    )
    reported = (
        evidence["fold_metrics"]
        .loc[
            (evidence["fold_metrics"].score_kind == "raw")
            & (evidence["fold_metrics"].metric_name == "balanced_utility_per_applicant")
            & evidence["fold_metrics"].workflow.isin(UTILITY_WORKFLOWS)
        ]
        .set_index(["workflow", "outer_fold"])
        .metric_value
    )
    if not np.allclose(baseline.sort_index(), reported.sort_index(), rtol=0, atol=1e-9):
        raise ValueError("Fixed-policy base utility disagrees with reported assessment")


def load_inputs(source: Path, assessment: dict) -> dict:
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    for key, expected in (
        ("assessment_run_id", assessment["assessment_run_id"]),
        ("feature_build_id", assessment["feature_build_id"]),
        ("assessment_config_sha256", assessment["config_sha256"]),
        ("protocol", assessment["protocol"]),
    ):
        if provenance.get(key) != expected:
            raise ValueError(
                "Model-input diagnostic and presentation assessment identity mismatch"
            )
    if (
        provenance.get("status") != "complete"
        or provenance.get("workflow") != "history_selected"
    ):
        raise ValueError(
            "Model-input evidence must describe the completed current history workflow"
        )
    frames = {
        name[:-4]: pd.read_csv(source / name)
        for name in INPUT_FILES
        if name.endswith(".csv")
    }
    schemas = {
        "fold_importance": {
            "assessment_run_id",
            "split_seed",
            "outer_fold",
            "workflow",
            "sample_count",
            "level",
            "input_key",
            "mean_absolute_contribution",
        },
        "importance_summary": {
            "level",
            "input_key",
            "fold_mean",
            "fold_sd",
            "fold_min",
            "fold_max",
        },
        "input_dictionary": {
            "raw_field",
            "display_label",
            "source_group",
            "group_label",
            "group_description",
        },
    }
    for name, frame in frames.items():
        if set(frame.columns) != schemas[name]:
            raise ValueError(
                "Model-input evidence must contain only the declared anonymous aggregate columns"
            )
    for name, digest in provenance["outputs_sha256"].items():
        if (
            name
            not in {
                "fold_importance.csv",
                "importance_summary.csv",
                "input_dictionary.csv",
            }
            or hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Model-input aggregate fingerprint mismatch")
    folds, summary, dictionary = (
        frames["fold_importance"],
        frames["importance_summary"],
        frames["input_dictionary"],
    )
    declared = next(
        w["feature_columns"]
        for w in assessment["selected_workflows"]
        if w["workflow"] == "history_selected"
    )
    if dictionary.raw_field.duplicated().any() or set(dictionary.raw_field) != set(
        declared
    ):
        raise ValueError(
            "Model-input dictionary differs from the assessed eligible inputs"
        )
    excluded = {"source_population"}.union(
        *(set(v) for v in assessment["config"]["excluded_features"].values())
    )
    if set(dictionary.raw_field) & excluded:
        raise ValueError(
            "Excluded identifier, target or diagnostic input in interpretation"
        )
    expected = set(
        itertools.product(["feature"], dictionary.raw_field, range(1, 6))
    ) | set(
        itertools.product(
            ["source_group"], dictionary.source_group.unique(), range(1, 6)
        )
    )
    grain = ["level", "input_key", "outer_fold"]
    if (
        folds.duplicated(grain).any()
        or set(folds[grain].itertuples(index=False, name=None)) != expected
        or set(folds.assessment_run_id) != {assessment["assessment_run_id"]}
        or set(folds.workflow) != {"history_selected"}
        or set(folds.split_seed) != {42}
        or not folds.sample_count.eq(1000).all()
    ):
        raise ValueError(
            "Model-input evidence requires all five declared frozen-model samples"
        )
    values = folds.mean_absolute_contribution
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Model-input magnitudes must be finite and nonnegative")
    recomputed = folds.groupby(["level", "input_key"]).mean_absolute_contribution.agg(
        fold_mean="mean", fold_sd="std", fold_min="min", fold_max="max"
    )
    actual = summary.set_index(["level", "input_key"]).sort_index()
    if (
        actual.index.duplicated().any()
        or not actual.index.equals(recomputed.index)
        or not np.allclose(actual, recomputed, rtol=0, atol=1e-12)
    ):
        raise ValueError("Model-input summary disagrees with five-fold magnitudes")
    grouped_features = (
        folds.loc[folds.level == "feature"]
        .merge(
            dictionary[["raw_field", "source_group"]],
            left_on="input_key",
            right_on="raw_field",
        )
        .groupby(["outer_fold", "source_group"])
        .mean_absolute_contribution.sum()
    )
    grouped = (
        folds.loc[folds.level == "source_group"]
        .set_index(["outer_fold", "input_key"])
        .mean_absolute_contribution.sort_index()
    )
    if not np.allclose(grouped_features.sort_index(), grouped, rtol=0, atol=1e-12):
        raise ValueError("Source-group magnitudes do not sum original-field magnitudes")
    checks = provenance["checks"]
    if {(c["split_seed"], c["outer_fold"]) for c in checks} != {
        (42, n) for n in range(1, 6)
    } or any(
        c["maximum_additivity_residual"] > 1e-8 or c["sample_count"] != 1000
        for c in checks
    ):
        raise ValueError("Model-input signed additivity checks are incomplete")
    frames["provenance"] = provenance
    return frames


def segment_chart(evidence: dict, destination: Path, save_chart) -> None:
    frame = evidence["history_segment_metrics"]
    frame = frame.loc[
        (frame.score_kind == "calibrated")
        & frame.workflow.isin(["application_only", "history_selected"])
    ]
    figure, axes = plt.subplots(
        1, 3, figsize=(12, 5.6), layout="constrained", sharey=True
    )
    for axis, (segment, label) in zip(axes, SEGMENTS.items(), strict=True):
        group = frame.loc[frame.history_segment == segment]
        ref = group.loc[group.workflow == "application_only"]
        for _, pair in group.pivot(
            index="outer_fold", columns="workflow", values="pr_auc"
        ).iterrows():
            axis.plot(
                [0, 1],
                [pair.application_only, pair.history_selected],
                color="#aab7c5",
                alpha=0.6,
                marker="o",
                markersize=4,
                linewidth=1,
            )
        means = group.groupby("workflow").pr_auc.mean()
        axis.scatter(
            [0, 1],
            [means.application_only, means.history_selected],
            c=["#5275a5", "#087f82"],
            marker="D",
            s=70,
            zorder=3,
        )
        for x, value in enumerate([means.application_only, means.history_selected]):
            axis.annotate(
                f"{value:.3f}",
                (x, value),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                weight="bold",
            )
        count = int(ref.applicant_count.sum())
        prevalence = np.average(ref.target_rate, weights=ref.applicant_count)
        axis.set_title(
            f"{label.replace(' or ', ' or\n')}\n{count:,} applicants · {prevalence:.1%} outcome rate",
            fontsize=10,
            pad=15,
        )
        axis.set_xticks([0, 1], ["Application\nonly", "Application\n+ history"])
        axis.set_xlim(-0.3, 1.3)
        axis.set_ylim(0.10, 0.33)
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Average precision · higher is better")
    figure.suptitle(
        "Compare models within each installment-history segment",
        weight="bold",
        fontsize=13,
    )
    figure.supxlabel(
        "Paired lines: five applicant test groups. Diamonds: mean of five metrics. Segment outcome rates differ;\ndo not compare AP across segments. No installment obligations does not mean no bureau or other loan history.",
        fontsize=9,
        color="#475467",
    )
    save_chart(figure, destination, "installment_segments")


def utility_chart(evidence: dict, destination: Path, save_chart) -> None:
    frame = evidence["utility_sensitivity"]
    means = frame.groupby(
        ["workflow", "margin_multiplier", "loss_multiplier", "review_multiplier"]
    ).utility_per_applicant.mean()
    figure, axes = plt.subplots(
        1, 3, figsize=(12.5, 5.7), layout="constrained", sharey=True
    )
    combos = list(itertools.product([0.5, 1.0, 2.0], repeat=2))
    for axis, margin in zip(axes, [0.5, 1.0, 2.0], strict=True):
        for workflow, color in zip(
            UTILITY_WORKFLOWS, ["#5275a5", "#8a6397", "#087f82"], strict=True
        ):
            values = [
                means.loc[(workflow, margin, loss, review)] for loss, review in combos
            ]
            axis.plot(
                range(9),
                values,
                marker="o",
                markersize=4,
                color=color,
                label=workflow_label(workflow).replace(" · ", "\n"),
            )
        axis.set_title(f"Approved-loan margin: {margin:g}×", fontsize=11, pad=12)
        axis.set_xticks(
            range(9),
            [f"{loss:g} / {review:g}" for loss, review in combos],
            rotation=55,
            ha="right",
            fontsize=9,
        )
        axis.set_xlabel("Loss multiplier / review-cost multiplier", fontsize=9)
        axis.axhline(0, color="#667085", linewidth=0.8)
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Illustrative utility units per applicant")
    axes[1].legend(fontsize=8, loc="upper left")
    figure.suptitle(
        "Fixed simulated policies across all 27 cost assumptions",
        weight="bold",
        fontsize=13,
    )
    figure.supxlabel(
        "Each point is a five-group mean; lines guide the eye across discrete assumptions, not a cost response curve.\nBase weights: margin 1,000, loss 5,000, review 50 units. Only middle-band reviews incur cost; declines contribute zero.",
        fontsize=9,
        color="#475467",
    )
    save_chart(figure, destination, "utility_sensitivity")


def input_chart(evidence: dict, destination: Path, save_chart) -> None:
    inputs = evidence["inputs"]
    summary, folds, dictionary = (
        inputs["importance_summary"],
        inputs["fold_importance"],
        inputs["input_dictionary"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.6), layout="constrained")
    groups = dictionary.groupby("source_group").agg(
        group_label=("group_label", "first"), input_count=("raw_field", "count")
    )
    labels = dictionary.set_index("raw_field").display_label
    for axis, level, title in zip(
        axes,
        ["source_group", "feature"],
        [
            "Source groups · cumulative magnitudes",
            "Largest individual input magnitudes",
        ],
        strict=True,
    ):
        rows = (
            summary.loc[summary.level == level]
            .nlargest(8, "fold_mean")
            .sort_values("fold_mean")
        )
        text = [
            f"{groups.loc[k, 'group_label']} ({groups.loc[k, 'input_count']} inputs)"
            if level == "source_group"
            else labels[k]
            for k in rows.input_key
        ]
        axis.barh(range(len(rows)), rows.fold_mean, color="#087f82", alpha=0.75)
        for index, row in enumerate(rows.itertuples()):
            dots = folds.loc[
                (folds.level == level) & (folds.input_key == row.input_key)
            ].mean_absolute_contribution
            axis.scatter(
                dots, np.full(5, index), color="#123e4b", s=12, alpha=0.65, zorder=3
            )
        axis.set_yticks(range(len(rows)), text, fontsize=9)
        axis.set_title(title, fontsize=11, pad=14)
        axis.set_xlabel("Mean absolute contribution · raw log-odds", fontsize=9)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle(
        "What the five frozen history models use", weight="bold", fontsize=13
    )
    figure.supxlabel(
        "Bars: equal mean of five 1,000-applicant samples. Dots: fold magnitudes. Absolute encoded effects sum by raw field/group.\nLarger groups can accumulate more; correlation shares attribution. Model behavior, not causal effects or predictive-value shares.",
        fontsize=9,
        color="#475467",
    )
    save_chart(figure, destination, "model_inputs")


def extra_narrative(evidence: dict) -> list[tuple[str, str, str]]:
    segments = evidence["history_segment_metrics"]
    segments = segments.loc[
        (segments.score_kind == "calibrated")
        & segments.workflow.isin(["application_only", "history_selected"])
    ]
    parts = []
    for segment, label in SEGMENTS.items():
        frame = (
            segments.loc[segments.history_segment == segment]
            .groupby("workflow")
            .pr_auc.mean()
        )
        parts.append(
            f"{label.lower()}: {frame.application_only:.3f} to {frame.history_selected:.3f}"
        )
    utility = (
        evidence["utility_sensitivity"]
        .groupby(
            ["margin_multiplier", "loss_multiplier", "review_multiplier", "workflow"]
        )
        .utility_per_applicant.mean()
        .unstack()
    )
    wins = int(utility.idxmax(axis=1).eq("history_selected").sum())
    inputs = evidence["inputs"]
    dictionary = inputs["input_dictionary"]
    groups = (
        inputs["importance_summary"]
        .loc[inputs["importance_summary"].level == "source_group"]
        .sort_values("fold_mean", ascending=False)
    )
    largest = dictionary.loc[
        dictionary.source_group == groups.iloc[0].input_key, "group_label"
    ].iloc[0]
    leading = (
        inputs["importance_summary"]
        .loc[inputs["importance_summary"].level == "feature"]
        .nlargest(2, "fold_mean")
        .input_key
    )
    labels = dictionary.set_index("raw_field").display_label
    leading_text = " and ".join(labels[key] for key in leading)
    return [
        (
            "segments",
            "Check the installment-history boundary",
            f"History improves average precision within each declared installment-history segment: {'; '.join(parts)}. The chart preserves matched fold variation and applicant counts. Segment prevalences differ, so average precision should be compared between models within each segment. These labels describe installment records and payment support; no installment obligations does not imply absence of bureau or other loan history. Results are descriptive, with no significance claim.",
        ),
        (
            "inputs",
            "Explain the current assessed model inputs",
            f"The five assessed history models use {len(dictionary)} eligible raw inputs. {leading_text} lead individual input magnitudes; {largest.lower()} have the largest cumulative group magnitude. Supplied credit scores are application inputs, so the signal should not be credited entirely to engineered history.\n\nThe chart describes how these fitted models use their inputs, on the model's internal log-odds score scale. It sums absolute contributions, so a group's total depends on its input count and correlated fields. These magnitudes are not signed net effects, probability changes, causal effects or shares of predictive performance.\n\nThe diagnostic reuses the five frozen models on target-blind samples without fitting or selection. Sampling, encoded-field aggregation and separate signed additivity checks are documented in the methods download; these are not adverse-action reasons.",
        ),
        (
            "utility",
            "Check sensitivity without re-optimizing policies",
            f"Application and loan history has the highest mean simulated utility among the three declared model workflows in {wins} of 27 assumption combinations. Each of the three workflows keeps the score cutoffs and actions from its five applicant-group fits fixed (15 fitted policies in total); varying margin, loss and review-cost weights does not optimize a new policy. Each point is a mean of five fold utilities, not a profit estimate. The constant ranking benchmark is not part of this declared utility grid.\n\nOnly the middle manual-review band incurs review cost; simulated declines contribute zero modeled utility. These descriptive results remain conditional on the illustrative formula and public-data population.",
        ),
    ]
