"""Render the public portfolio from frozen anonymous aggregates, without fitting."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.class_weighting_report import FILES as WEIGHTING_FILES
from src.class_weighting_report import load_evidence as load_weighting_evidence
from src.portfolio_extensions import (
    INPUT_FILES as MODEL_INPUT_FILES,
)
from src.portfolio_extensions import (
    INPUT_SOURCE,
    extra_narrative,
    input_chart,
    load_inputs,
    segment_chart,
    utility_chart,
    validate_sensitivities,
)
from src.presentation import (
    METRIC_DETAILS,
    METRIC_LABELS,
    WORKFLOW_LABELS,
    workflow_label,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/tuning_20261004"
DESTINATION = ROOT / "reports/portfolio"
WEIGHTING_SOURCE = ROOT / "reports/class_weighting_20261004"
WORKFLOWS = (
    "training_prevalence",
    "application_only",
    "logistic_tuned",
    "history_selected",
)
COLORS = ("#667085", "#5275a5", "#8a6397", "#087f82")
INPUT_FILES = (
    "summary.csv",
    "fold_metrics.csv",
    "reliability_bins.csv",
    "prior_protocol_comparison.csv",
    "negative_control_summary.csv",
    "selection_stability.csv",
    "history_segment_metrics.csv",
    "utility_sensitivity.csv",
    "provenance.json",
)


def load_evidence(source: Path) -> dict:
    """Check the curated aggregate grain and cross-file identity before rendering."""
    frames = {
        name[:-4]: pd.read_csv(source / name)
        for name in INPUT_FILES
        if name.endswith(".csv")
    }
    for frame in frames.values():
        if {
            "applicant_id",
            "SK_ID_CURR",
            "observed_target",
            "target",
            "probability",
        } & set(frame.columns):
            raise ValueError("Portfolio inputs must contain anonymous aggregates only")
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    folds = frames["fold_metrics"]
    if set(folds.run_id) != {provenance["assessment_run_id"]}:
        raise ValueError("Aggregate assessment identity differs from provenance")
    if set(folds.workflow) != set(WORKFLOWS):
        raise ValueError("The portfolio requires all four matched workflows")
    final = folds.loc[folds.score_kind == "calibrated"]
    grain = ["workflow", "split_seed", "outer_fold", "metric_name"]
    if final.duplicated(grain).any() or set(final.outer_fold) != {1, 2, 3, 4, 5}:
        raise ValueError(
            "Expected one metric per workflow and five applicant test groups"
        )
    summary = frames["summary"]
    for row in summary.loc[summary.score_kind == "calibrated"].itertuples():
        values = final.loc[
            (final.workflow == row.workflow) & (final.metric_name == row.metric_name),
            "metric_value",
        ]
        if len(values) != 5 or not np.isclose(
            values.mean(), row.fold_mean, rtol=0, atol=1e-12
        ):
            raise ValueError("Summary disagrees with the five fold metrics")
    counts = final.loc[final.metric_name == "pr_auc"].pivot(
        index="outer_fold", columns="workflow", values="applicant_count"
    )
    if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Workflow comparisons must use the same applicant groups")
    assumptions = provenance["config"]["business_assumptions"]
    if assumptions["manual_review_capacity_rate"] != 0.1:
        raise ValueError(
            "The curated capture presentation expects the declared 10% reference"
        )
    frames["business"] = assumptions
    reliability = frames["reliability_bins"]
    if set(reliability.run_id) != {provenance["assessment_run_id"]}:
        raise ValueError("Reliability bins belong to a different assessment")
    bin_counts = (
        reliability.loc[reliability.score_kind == "calibrated"]
        .groupby(["outer_fold", "workflow"])["applicant_count"]
        .sum()
        .unstack()
    )
    if not bin_counts.equals(counts.astype(bin_counts.dtypes.iloc[0])):
        raise ValueError(
            "Reliability-bin counts differ from the matched assessment population"
        )
    frames["provenance"] = provenance
    frames["applicant_count"] = int(counts.iloc[:, 0].sum())
    validate_sensitivities(frames)
    return frames


def mean(evidence: dict, workflow: str, metric: str) -> float:
    rows = evidence["summary"]
    values = rows.loc[
        (rows.workflow == workflow)
        & (rows.metric_name == metric)
        & (rows.score_kind == "calibrated"),
        "fold_mean",
    ]
    if len(values) != 1:
        raise ValueError("Expected one final-probability metric")
    return float(values.iloc[0])


def save_chart(figure, destination: Path, name: str) -> None:
    figure.savefig(destination / f"{name}.png", dpi=180, facecolor="white")
    figure.savefig(
        destination / f"{name}.svg", facecolor="white", metadata={"Date": None}
    )
    plt.close(figure)


def comparison_chart(evidence: dict, destination: Path, metric: str, name: str) -> None:
    figure, axis = plt.subplots(figsize=(10.6, 5.2), layout="constrained")
    folds = evidence["fold_metrics"]
    for index, (workflow, color) in enumerate(zip(WORKFLOWS, COLORS, strict=True)):
        values = folds.loc[
            (folds.workflow == workflow)
            & (folds.metric_name == metric)
            & (folds.score_kind == "calibrated"),
            "metric_value",
        ].to_numpy()
        axis.scatter(
            values, index + np.linspace(-0.10, 0.10, 5), color=color, alpha=0.5, s=30
        )
        value = mean(evidence, workflow, metric)
        axis.scatter(
            [value], [index], marker="D", s=75, color=color, edgecolor="white", zorder=3
        )
        label = f"{value:.1%}" if name == "risk_capture" else f"{value:.3f}"
        axis.annotate(
            label,
            (value, index),
            xytext=(11, 11),
            textcoords="offset points",
            color=color,
            weight="bold",
        )
    axis.set_yticks(
        range(4), [workflow_label(w).replace(" · ", "\n") for w in WORKFLOWS]
    )
    axis.set_xlim(0, 0.46 if name == "risk_capture" else 0.33)
    axis.grid(axis="x", alpha=0.17)
    axis.set_axisbelow(True)
    if name == "risk_capture":
        axis.axvline(0.1, color="#667085", linestyle="--", linewidth=1)
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(
            "Who appears in the highest-risk 10%?", loc="left", pad=20, weight="bold"
        )
        axis.set_xlabel(
            "Share of all repayment-difficulty cases captured · higher is better"
        )
        caption = "10% dashed line: random ranking reference. Boundary ties receive equal expected membership.\nThis is a ranking check, not capture in the middle manual-review band."
    else:
        axis.set_title(
            "Loan history adds useful ranking information",
            loc="left",
            pad=20,
            weight="bold",
        )
        axis.set_xlabel(
            "Average precision · higher is better · not classification accuracy"
        )
        caption = "Public labeled applications · same five applicant test groups for every model.\nSmall dots: individual group results. Diamonds: mean of five metrics; variation is descriptive."
    figure.supxlabel(caption, fontsize=9, color="#475467")
    save_chart(figure, destination, name)


def reliability_chart(evidence: dict, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), layout="constrained")
    bins = evidence["reliability_bins"]
    bins = bins.loc[
        (bins.workflow == "history_selected") & (bins.score_kind == "calibrated")
    ]
    for fold, group in bins.groupby("outer_fold"):
        group = group.sort_values("average_predicted_score")
        axis.plot(
            group.average_predicted_score,
            group.observed_default_rate,
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=f"Applicant test group {fold}",
        )
    maximum = float(
        max(bins.average_predicted_score.max(), bins.observed_default_rate.max())
    )
    upper = min(1.0, maximum * 1.12)
    axis.plot(
        [0, upper],
        [0, upper],
        color="#475467",
        linestyle="--",
        label="Predicted = observed",
    )
    axis.set(
        xlim=(0, upper),
        ylim=(0, upper),
        xlabel="Mean predicted repayment-difficulty probability",
        ylabel="Observed repayment-difficulty rate",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title(
        "Do predicted probabilities match observed rates?",
        loc="left",
        pad=18,
        weight="bold",
    )
    axis.grid(alpha=0.17)
    axis.legend(fontsize=9)
    figure.supxlabel(
        "Application and loan history · final probabilities. Each line keeps its own score groups;\nidentical scores stay together. No pooling or averaging of different group boundaries.",
        fontsize=9,
        color="#475467",
    )
    save_chart(figure, destination, "probability_reliability")


def search_chart(evidence: dict, destination: Path) -> None:
    frame = evidence["prior_protocol_comparison"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 4.8), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ["pr_auc", "brier_score", "log_loss"],
        [
            "Average precision\nHigher is better",
            "Brier score\nLower is better",
            "Log loss\nLower is better",
        ],
        strict=True,
    ):
        for score, color, label in [
            ("raw", "#8a6397", "Raw probabilities"),
            ("calibrated", "#087f82", "Final probabilities"),
        ]:
            row = frame.loc[
                (frame.workflow_v3 == "history_selected")
                & (frame.metric_name == metric)
                & (frame.score_kind == score)
            ].iloc[0]
            axis.plot(
                [0, 1],
                [row.fold_mean_v2, row.fold_mean_v3],
                marker="o",
                color=color,
                label=label,
            )
        axis.set_xticks([0, 1], ["Earlier\nsearch", "Current\nsearch"])
        axis.set_title(title, fontsize=11)
        axis.set_ylim(
            0, 0.29 if metric == "pr_auc" else 0.18 if metric == "brier_score" else 0.53
        )
        axis.grid(axis="y", alpha=0.17)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        "Broader search: ranking plateau, better raw probabilities", weight="bold"
    )
    figure.supxlabel(
        "Application and loan history · mean of five matched applicant test-group metrics.\nThe search comparison alone does not isolate a parameter; a separate controlled comparison tests class weighting.",
        fontsize=9,
        color="#475467",
    )
    save_chart(figure, destination, "search_comparison")


def weighting_chart(evidence: dict, destination: Path) -> None:
    frame = evidence["weighting"]["fold_metrics"]
    figure, axes = plt.subplots(1, 3, figsize=(11, 5.2), layout="constrained")
    for axis, metric, title in zip(
        axes,
        ["mean_probability", "brier_score", "log_loss"],
        [
            "Average predicted risk",
            "Brier score\nLower is better",
            "Log loss\nLower is better",
        ],
        strict=True,
    ):
        for recipe, color, offset in [
            ("earlier", "#8a6397", -0.045),
            ("current", "#087f82", 0.045),
        ]:
            pairs = frame.loc[frame.recipe == recipe].pivot(
                index="outer_fold", columns="weighting", values=metric
            )
            for row in pairs.itertuples():
                axis.plot(
                    np.array([0, 1]) + offset,
                    [row.earlier_weight, row.unweighted],
                    color=color,
                    alpha=0.25,
                    linewidth=1,
                    marker="o",
                    markersize=3,
                )
            axis.plot(
                np.array([0, 1]) + offset,
                [pairs.earlier_weight.mean(), pairs.unweighted.mean()],
                color=color,
                marker="D",
                markersize=6,
                linewidth=2,
                label=f"{recipe.capitalize()} recipe",
            )
        axis.set_xticks([0, 1], ["Earlier weight\n(9–11×)", "No weighting\n(1×)"])
        axis.set_title(title, fontsize=11)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.17)
    axes[0].axhline(
        evidence["weighting"]["observed_rate"],
        color="#667085",
        linestyle="--",
        label="Observed rate (8.07%)",
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "Changing class weighting isolates the probability-scale effect",
        weight="bold",
        fontsize=13,
    )
    figure.supxlabel(
        "History model · each line connects a matched group with only class weighting changed.\nDiamonds show five-group means. Inputs, applicants, preprocessing, seed and other settings stay fixed within each recipe.",
        fontsize=9,
        color="#475467",
    )
    save_chart(figure, destination, "class_weighting")


def narrative(evidence: dict) -> tuple[str, list[tuple[str, str, str]]]:
    ap = mean(evidence, "history_selected", "pr_auc")
    application = mean(evidence, "application_only", "pr_auc")
    capture = mean(evidence, "history_selected", "recall_at_manual_review_capacity")
    brier = mean(evidence, "history_selected", "brier_score")
    loss = mean(evidence, "history_selected", "log_loss")
    prior = evidence["prior_protocol_comparison"]
    prior_ap = float(
        prior.loc[
            (prior.workflow_v3 == "history_selected")
            & (prior.score_kind == "calibrated")
            & (prior.metric_name == "pr_auc"),
            "fold_mean_v2",
        ].iloc[0]
    )
    weights = evidence["business"]
    margin = weights["expected_margin_per_good_loan"]
    penalty = weights["expected_loss_per_bad_loan"]
    review_cost = weights["manual_review_cost"]
    selections = evidence["selection_stability"]
    history = selections.loc[selections.workflow == "history_selected"]
    method_text = (
        "All five history models selected the full 174 eligible inputs and raw probabilities; no probability-adjustment transform was selected."
        if (
            len(history) == 5
            and history.feature_count.eq(174).all()
            and history.calibration_method.eq("uncalibrated").all()
        )
        else "Input counts and probability-adjustment choices vary by applicant test group; see the technical evidence."
    )
    intro = "Can prior loan and repayment history improve risk ranking beyond an application form?"
    sections = [
        (
            "problem",
            "The question",
            "Loan applications contain current financial information, while separate tables record previous loans, monthly balances and repayments. This public Home Credit project asks whether joining that history improves ranking of applicants with observed repayment difficulty. The recorded target is a proxy for repayment difficulty, not measured financial loss.",
        ),
        (
            "engineering",
            "Build one trustworthy applicant table",
            "SQL converts several relational tables into one modeling-table row per applicant and source population. Installment obligations are counted once across split payments; ambiguous schedules and unknown payments remain explicit. Monthly history uses distinct applicant months, and bureau loans must originate before the application day. Python coordinates the steps, models, scoring, interpretation and exports. Identifiers, the outcome and direct demographic/protected-status-like fields are excluded from model inputs. Relative dates cannot certify the exact time a lender could have obtained each field.",
        ),
        (
            "assessment",
            "Compare models on the same applicants",
            f"The assessment covers {evidence['applicant_count']:,} labeled development applicants in five applicant test groups. Each group is predicted by models whose fitting and selection used the other groups. Inside that training population, three-group cross-validation selects inputs and model settings; separate applicants support stopping, probability-adjustment fitting and method/threshold selection. A constant outcome-rate benchmark, logistic regression, application-only LightGBM and history LightGBM predict the same test applicants. Logistic regression is the simpler linear benchmark; LightGBM combines decision trees. Means summarize five separate metrics, not one pooled cross-model score. Prior public-data exploration remains: this is not an untouched final test or a future-cohort assessment.",
        ),
        (
            "ranking",
            "History improves ranking in this assessment",
            f"Average precision is {ap:.3f} with application and loan history versus {application:.3f} with application fields only. Average precision summarizes how strongly repayment-difficulty cases concentrate near the top of the ranking; it is not accuracy. The history model captures {capture:.1%} of observed repayment-difficulty cases in the highest-risk 10% of applicants, against a 10% random-ranking reference. Tied boundary scores receive equal expected membership. This highest-risk group is separate from the middle manual-review band.",
        ),
        (
            "probabilities",
            "Check probabilities separately",
            f"The history model's Brier score is {brier:.3f} and log loss is {loss:.3f}; lower is better for both. These measure probability errors, while the reliability chart compares predicted and observed rates within each test group's own score bins. {method_text} Reliability is descriptive, not a guarantee for a new lending population.",
        ),
        (
            "search",
            "A broader search did not materially improve ranking",
            f"The earlier corrected history search had average precision {prior_ap:.6f}; the current search has {ap:.6f}. That is a practical ranking plateau in these descriptive results. Raw probability errors improved substantially, while final probability quality stayed similar. Multiple settings and search budgets changed together, so this search comparison alone cannot attribute the change to one parameter. The separate controlled comparison below tests class weighting. A 20,000-applicant shuffled-outcome diagnostic returned roughly chance ranking; it does not establish real-world field availability or erase earlier exploration.",
        ),
        (
            "actions",
            "Show decisions as assumptions, not lending advice",
            f"A lower score cutoff separates simulated approval from manual review; an upper cutoff separates manual review from simulated decline. Cutoffs come from model-selection applicants and are applied unchanged to assessment applicants. Only the middle band incurs review cost. Simulated declines issue no loan and contribute zero modeled margin, loss or review cost. The example weights are {margin:,.0f} units for an approved applicant without recorded difficulty, {penalty:,.0f} units of loss for an approved applicant with difficulty and {review_cost:,.0f} units per review. Utility is in illustrative units, not dollars or profit. Sensitivity checks vary each weight by 0.5, 1 and 2. No reviewer effectiveness, rejected-loan counterfactual or hard review-queue limit is estimated.",
        ),
        (
            "limits",
            "What this demonstrates—and what remains unknown",
            "The contribution is disciplined SQL data engineering, benchmark comparison, bounded model selection, reproducible assessment and readable reporting. Historical comparisons remain an archive. Unlabeled Kaggle applications demonstrate batch scoring only and contribute no outcome metrics. Calendar application, field-availability and outcome-maturity timestamps are missing; random applicant groups cannot validate performance in a future cohort. This portfolio does not establish underwriting, compliance, fair-lending or adverse-action readiness. The saved Power BI files and screenshots are unrefreshed historical demonstrations; the current charts and offline report are separate presentation artifacts.",
        ),
    ]
    weighting = evidence["weighting"]
    results = weighting["summary"].set_index(["recipe", "weighting"])
    before = results.loc[("earlier", "earlier_weight")]
    after = results.loc[("earlier", "unweighted")]
    finding = (
        "Removing class weighting is the dominant explanation for the better raw probabilities in these recipes. Probability errors improved in all ten recipe/group pairs, and putting the earlier weight into the current recipe reversed the benefit."
        if weighting["probability_improved_all_pairs"]
        else "The controlled comparison did not show a consistent probability-quality benefit in every recipe/group pair; the weighting explanation remains conditional."
    )
    sections.insert(
        6,
        (
            "weighting",
            "Test why the raw probabilities improved",
            f"{finding} A separate controlled comparison kept each recipe's applicants, selected inputs, fitting-only preprocessing, random seed, number of trees and all other model settings fixed. Only the weight assigned to repayment-difficulty cases changed, from the earlier 9–11 times weight to equal weighting. Using the earlier recipe alone, mean predicted risk fell from {before.mean_probability_mean:.1%} to {after.mean_probability_mean:.2%}, against an observed rate of {weighting['observed_rate']:.2%}; raw Brier score fell from {before.brier_score_mean:.4f} to {after.brier_score_mean:.4f}. Brier and log loss improved together; average-rate agreement alone does not prove calibration. Class weighting prioritizes the rare outcome during fitting and can distort the probability scale. This retrospective check makes no model promotion and does not establish the separate contribution of the search's probability-quality screen, globally optimal weights or future-cohort performance. Earlier probability adjustment had already repaired much of the scale error, so final probability quality changed little.",
        ),
    )
    return intro, sections


def render(
    source: Path = SOURCE,
    destination: Path = DESTINATION,
    weighting_source: Path = WEIGHTING_SOURCE,
    input_source: Path = INPUT_SOURCE,
) -> dict:
    evidence = load_evidence(source)
    evidence["weighting"] = load_weighting_evidence(
        weighting_source, evidence["provenance"]["assessment_run_id"]
    )
    evidence["inputs"] = load_inputs(input_source, evidence["provenance"])
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "home-credit-portfolio",
        }
    )
    comparison_chart(evidence, destination, "pr_auc", "model_comparison")
    comparison_chart(
        evidence, destination, "recall_at_manual_review_capacity", "risk_capture"
    )
    reliability_chart(evidence, destination)
    search_chart(evidence, destination)
    weighting_chart(evidence, destination)
    segment_chart(evidence, destination, save_chart)
    utility_chart(evidence, destination, save_chart)
    input_chart(evidence, destination, save_chart)
    intro, sections = narrative(evidence)
    extras = extra_narrative(evidence)
    sections.insert(4, extras[0])
    sections.insert(5, extras[1])
    sections.insert(-1, extras[2])
    input_dictionary = evidence["inputs"]["input_dictionary"]
    group_table = input_dictionary.groupby("source_group", sort=False).agg(
        label=("group_label", "first"),
        description=("group_description", "first"),
        count=("raw_field", "count"),
    )
    group_md = (
        "| Input group | Raw inputs | What it describes |\n|---|---:|---|\n"
        + "\n".join(
            f"| {r.label} | {r.count} | {r.description} |"
            for r in group_table.itertuples()
        )
        + "\n\n"
    )
    group_html = (
        '<div class="table-wrap"><table><thead><tr><th>Input group</th><th>Raw inputs</th><th>What it describes</th></tr></thead><tbody>'
        + "".join(
            f"<tr><td>{html.escape(r.label)}</td><td>{r.count}</td><td>{html.escape(r.description)}</td></tr>"
            for r in group_table.itertuples()
        )
        + "</tbody></table></div>"
    )
    metric_rows = []
    for workflow in WORKFLOWS:
        metric_rows.append(
            [
                workflow_label(workflow),
                *[
                    f"{mean(evidence, workflow, metric):.3f}"
                    for metric in ("pr_auc", "roc_auc", "brier_score", "log_loss")
                ],
            ]
        )
    headers = [
        "Model / input scope",
        "Average precision ↑",
        "ROC AUC ↑",
        "Brier score ↓",
        "Log loss ↓",
    ]
    table_md = (
        "| "
        + " | ".join(headers)
        + " |\n|---|---:|---:|---:|---:|\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in metric_rows)
    )
    figures = {
        "segments": (
            "installment_segments",
            "Application-only and history ranking within three installment-history segments; fold variation and counts retained.",
        ),
        "inputs": (
            "model_inputs",
            "Cumulative mean absolute raw-margin contributions by source group and individual input for five frozen history models.",
        ),
        "utility": (
            "utility_sensitivity",
            "Three fixed simulated policies across all 27 margin, loss and review-cost assumptions; five-group means.",
        ),
        "ranking": (
            "model_comparison",
            "Average precision for four models on five matched applicant test groups.",
        ),
        "probabilities": (
            "probability_reliability",
            "Predicted and observed repayment-difficulty rates in each test group's own bins.",
        ),
        "search": (
            "search_comparison",
            "Earlier and current search: ranking and raw/final probability quality.",
        ),
        "weighting": (
            "class_weighting",
            "Controlled class-weighting comparison: two fixed recipes, weighted and unweighted, on five matched groups.",
        ),
    }
    md = "# Credit risk from application and repayment history\n\n" + intro + "\n\n"
    for key, title, paragraph in sections:
        md += f"## {title}\n\n{paragraph}\n\n"
        if key == "ranking":
            md += (
                table_md
                + "\n\n![Model comparison](model_comparison.png)\n\n![Highest-risk group capture](risk_capture.png)\n\n"
            )
        elif key in figures:
            name, alt = figures[key]
            md += f"![{alt}]({name}.png)\n\n"
        if key == "inputs":
            md += group_md
    glossary_keys = [
        "pr_auc",
        "roc_auc",
        "brier_score",
        "log_loss",
        "recall_at_manual_review_capacity",
        "balanced_utility_per_applicant",
    ]
    glossary_rows = [
        (METRIC_LABELS[key], *METRIC_DETAILS[key]) for key in glossary_keys
    ]
    md += "## Metric glossary\n\n| Measure | Meaning | Direction | Units |\n|---|---|---|---|\n"
    md += "\n".join("| " + " | ".join(row) + " |" for row in glossary_rows) + "\n\n"
    md += "## Explore the evidence\n\n[Browser report](https://stevennitesh.github.io/loan-default-risk-decisioning-system/) · [Offline standalone report](index.html) · [Exact numeric appendix](metrics.csv) · [Metric dictionary](metric_dictionary.csv) · [Presentation provenance](provenance.json) · [Frozen-model input methods](model_input_methods.md) · [Input dictionary](model_input_dictionary.csv) · [Input magnitudes](model_input_summary.csv) · [Current technical assessment](https://github.com/stevennitesh/loan-default-risk-decisioning-system/blob/main/reports/tuning_20261004/assessment_report.md) · [Controlled weighting evidence](https://github.com/stevennitesh/loan-default-risk-decisioning-system/blob/main/reports/class_weighting_20261004/assessment_report.md) · [Historical experiment archive](https://github.com/stevennitesh/loan-default-risk-decisioning-system/blob/main/reports/experiments/README.md)\n\nInstall the project dependencies using [How To Run](../../README.md#how-to-run), then regenerate with `make portfolio` (Windows: `make portfolio PYTHON=python`). Only committed anonymous aggregate inputs are read; no raw data, saved model or new fitting is required.\n"
    (destination / "case_study.md").write_text(md, encoding="utf-8")
    nav = " ".join(
        f'<a href="#{key}">{html.escape(title)}</a>' for key, title, _ in sections
    )
    body = ""
    for key, title, paragraph in sections:
        body += f'<section id="{key}"><h2>{html.escape(title)}</h2><p>{html.escape(paragraph)}</p>'
        if key == "inputs":
            body += group_html
        if key == "ranking":
            body += (
                '<div class="table-wrap"><table><thead><tr>'
                + "".join(f"<th>{html.escape(v)}</th>" for v in headers)
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>"
                    + "".join(f"<td>{html.escape(v)}</td>" for v in row)
                    + "</tr>"
                    for row in metric_rows
                )
                + "</tbody></table></div>"
            )
        chart_list = [figures[key]] if key in figures else []
        if key == "ranking":
            chart_list.append(
                (
                    "risk_capture",
                    "Share of repayment-difficulty cases captured in the highest-risk 10%.",
                )
            )
        for name, alt in chart_list:
            encoded = base64.b64encode(
                (destination / f"{name}.png").read_bytes()
            ).decode("ascii")
            score_note = (
                "Within-segment model comparison; different segment prevalences prevent cross-segment AP interpretation."
                if key == "segments"
                else "Frozen-model behavior on target-blind bounded samples; cumulative absolute raw-log-odds magnitudes, not performance shares."
                if key == "inputs"
                else "Fixed balanced-reference actions; illustrative utility units, not money or profit."
                if key == "utility"
                else "Raw probabilities; only class weighting changes within each fixed recipe."
                if key == "weighting"
                else "Raw and final probabilities from different search procedures."
                if key == "search"
                else "Final method-choice probabilities."
            )
            body += f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)} Source: completed public labeled-applicant assessment; five matched test groups. {score_note}</figcaption></figure>'
        body += "</section>"
    appendix = f"Assessment {evidence['provenance']['assessment_run_id']}; protocol {evidence['provenance']['protocol']}; shuffled-label diagnostic {evidence['provenance']['control_run_id']}. Exact means, descriptive fold variation and preserved machine metric keys are in metrics.csv. The CSV score-kind key calibrated means the final method-choice view; it does not imply that a transform was selected. The presentation provenance lists aggregate and renderer hashes; scientific execution fingerprints remain unchanged."
    dictionary = "".join(
        f"<tr><td><code>{html.escape(key)}</code></td><td>{html.escape(label)}</td><td>{html.escape(METRIC_DETAILS[key][0])}</td><td>{html.escape(METRIC_DETAILS[key][1])}</td><td>{html.escape(METRIC_DETAILS[key][2])}</td></tr>"
        for key, label in METRIC_LABELS.items()
    )
    downloads = " ".join(
        f'<a href="{filename}" download>{html.escape(label)}</a>'
        for filename, label in [
            ("metrics.csv", "Exact metrics CSV"),
            ("metric_dictionary.csv", "Metric dictionary"),
            ("provenance.json", "Presentation provenance"),
            ("model_input_dictionary.csv", "Model inputs CSV"),
            ("model_input_summary.csv", "Input magnitudes CSV"),
            ("model_input_provenance.json", "Input provenance"),
            ("installment_segments.svg", "Installment chart SVG"),
            ("utility_sensitivity.svg", "Utility chart SVG"),
            ("model_inputs.svg", "Input chart SVG"),
        ]
    )
    evidence_links = (
        '<section id="downloads"><h2>Downloads and source evidence</h2><nav aria-label="Download anonymous evidence">'
        + downloads
        + '</nav><p>Downloads work online or alongside this HTML in the presentation folder. The charts and story remain readable when this HTML is opened alone offline.</p><p>Optional source links require a network connection: <a href="https://github.com/stevennitesh/loan-default-risk-decisioning-system">Project repository</a> · <a href="https://github.com/stevennitesh/loan-default-risk-decisioning-system/blob/main/reports/tuning_20261004/assessment_report.md">Current assessment</a> · <a href="https://github.com/stevennitesh/loan-default-risk-decisioning-system/blob/main/reports/class_weighting_20261004/assessment_report.md">Controlled weighting evidence</a> · <a href="model_input_methods.md" download>Frozen-model input methods</a></p></section>'
    )
    body += evidence_links
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Credit risk from application and repayment history</title><style>html{{scroll-behavior:smooth}}body{{margin:0;background:#f6f8fa;color:#182230;font:17px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1120px;margin:auto;padding:2rem 1.5rem}}header{{padding:2rem;background:#123e4b;color:white;border-radius:14px}}h1{{font-size:clamp(2rem,5vw,3.1rem);line-height:1.15;max-width:900px}}h2{{font-size:1.55rem;line-height:1.3;color:#123e4b}}.eyebrow{{letter-spacing:.08em;text-transform:uppercase;font-size:.8rem}}nav{{display:flex;flex-wrap:wrap;gap:.5rem 1rem;margin:1.5rem 0}}a{{color:#076c76;text-underline-offset:3px}}section{{background:white;padding:1.7rem 2rem;border:1px solid #dde4eb;border-radius:12px;margin:1.2rem 0;scroll-margin-top:1rem}}p{{max-width:90ch}}img{{display:block;width:100%;height:auto}}figure{{margin:1.5rem 0}}figcaption{{font-size:.85rem;color:#475467}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{text-align:left;padding:.65rem;border-bottom:1px solid #dde4eb}}th{{background:#edf5f6}}code{{overflow-wrap:anywhere}}details{{background:white;padding:1.5rem;border-radius:12px}}@media(max-width:600px){{main{{padding:.8rem}}header,section{{padding:1.2rem}}}}@media print{{body{{background:white}}nav{{display:none}}section{{break-inside:avoid}}}}</style></head><body><main><header><p class="eyebrow">Public-data portfolio · SQL · Python · credit risk</p><h1>Credit risk from application and repayment history</h1><p>{html.escape(intro)}</p><p>Current completed evidence · {evidence["applicant_count"]:,} labeled applicants · five matched applicant test groups</p></header><nav aria-label="Report contents">{nav}</nav>{body}<details id="technical-appendix"><summary>Technical appendix: exact identities and machine-key dictionary</summary><p>{html.escape(appendix)}</p><div class="table-wrap"><table><thead><tr><th>Preserved machine key</th><th>Measure</th><th>Meaning</th><th>Direction</th><th>Units</th></tr></thead><tbody>{dictionary}</tbody></table></div><p>Average precision summarizes ranking (higher is better); ROC AUC summarizes pairwise ordering (higher is better). Brier score is mean squared probability error; log loss penalizes incorrect confident probabilities (lower is better). No confidence interval is inferred from fold variation.</p></details><footer><p>This file is self-contained: charts, styles and text require no network connection. Adjacent PNG/SVG files are available for reuse; metrics.csv and provenance.json provide the technical evidence trail.</p></footer></main></body></html>"""
    (destination / "index.html").write_text(document, encoding="utf-8")
    for original, final_name in [
        ("input_dictionary.csv", "model_input_dictionary.csv"),
        ("importance_summary.csv", "model_input_summary.csv"),
        ("fold_importance.csv", "model_input_fold_importance.csv"),
        ("provenance.json", "model_input_provenance.json"),
        ("methods.md", "model_input_methods.md"),
    ]:
        shutil.copyfile(input_source / original, destination / final_name)
    evidence["summary"].to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "machine_key": key,
                "display_label": label,
                "definition": METRIC_DETAILS.get(
                    key,
                    (
                        "Model family and input scope used for the matched comparison.",
                        "Not a metric",
                        "Not applicable",
                    ),
                )[0],
                "direction": METRIC_DETAILS.get(
                    key, ("", "Not a metric", "Not applicable")
                )[1],
                "units": METRIC_DETAILS.get(key, ("", "", "Not applicable"))[2],
            }
            for key, label in {**METRIC_LABELS, **WORKFLOW_LABELS}.items()
        ]
    ).to_csv(destination / "metric_dictionary.csv", index=False)
    outputs = [
        p for p in destination.iterdir() if p.is_file() and p.name != "provenance.json"
    ]
    provenance = {
        "purpose": "Presentation only; no model fitting or scientific evidence replacement",
        "assessment_run_id": evidence["provenance"]["assessment_run_id"],
        "protocol": evidence["provenance"]["protocol"],
        "assessment_config_sha256": evidence["provenance"]["config_sha256"],
        "source_aggregates_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in INPUT_FILES
        },
        "weighting_run_id": evidence["weighting"]["provenance"]["run_id"],
        "model_input_assessment_run_id": evidence["inputs"]["provenance"][
            "assessment_run_id"
        ],
        "model_input_aggregates_sha256": {
            name: hashlib.sha256((input_source / name).read_bytes()).hexdigest()
            for name in MODEL_INPUT_FILES
        },
        "weighting_aggregates_sha256": {
            name: hashlib.sha256((weighting_source / name).read_bytes()).hexdigest()
            for name in WEIGHTING_FILES
        },
        "renderer_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "src/portfolio_report.py",
                "src/portfolio_extensions.py",
                "src/class_weighting_report.py",
                "src/presentation.py",
                "src/evaluation_reports.py",
                "src/nested_assessment.py",
                "src/feature_selection.py",
                "src/model_stability.py",
                "src/correctness_summary.py",
                "src/tuning_summary.py",
                "src/explain.py",
            ]
        },
        "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs
        },
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="Curated anonymous aggregate evidence folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DESTINATION,
        help="Final presentation output folder.",
    )
    parser.add_argument(
        "--input-source",
        type=Path,
        default=INPUT_SOURCE,
        help="Curated anonymous frozen-model input evidence folder.",
    )
    parser.add_argument(
        "--weighting-source",
        type=Path,
        default=WEIGHTING_SOURCE,
        help="Curated anonymous class-weighting diagnostic folder.",
    )
    args = parser.parse_args()
    render(args.source, args.output, args.weighting_source, args.input_source)
    print(f"Portfolio report: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
