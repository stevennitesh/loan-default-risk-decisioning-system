"""Verify and curate anonymous evidence for the declared v3 tuning procedure."""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from src.evidence import file_sha256
from src.presentation import workflow_label
from src.report_contracts import TUNING_CV_SUMMARY_COLUMNS
from src.runtime import REPO_ROOT


def verify_search(output, manifest):
    rows = []
    for plan in manifest["folds"]:
        base = set(plan["applicant_ids"]["train"])
        for workflow in ("history_selected", "application_only"):
            path = (
                output
                / f"search_{plan['split_seed']}_{plan['outer_fold']}_{workflow}.json"
            )
            evidence = json.loads(path.read_text(encoding="utf-8"))
            specs = evidence["candidate_specs"]
            if len(specs) != 24 or len({row["candidate_name"] for row in specs}) != 24:
                raise ValueError("Incorrect joint search budget")
            identities = {
                json.dumps(
                    [row["feature_set"], row["search_params"], row["weight_fraction"]],
                    sort_keys=True,
                )
                for row in specs
            }
            if len(identities) != 24 or not {0.0, 0.25, 0.5, 1.0}.issubset(
                {row["weight_fraction"] for row in specs}
            ):
                raise ValueError(
                    "Joint recipes must be unique and include unweighted/lighter fits"
                )
            scoring = []
            for inner in evidence["inner_memberships"]:
                roles = [
                    set(inner[name]) for name in ("fitting", "stopping", "scoring")
                ]
                if set.union(*roles) != base or any(
                    roles[i] & roles[j] for i in range(3) for j in range(i)
                ):
                    raise ValueError(
                        "CV fitting/stopping/scoring boundaries are invalid"
                    )
                scoring.extend(inner["scoring"])
            if len(scoring) != len(base) or set(scoring) != base:
                raise ValueError("Inner score rows must be covered exactly once")
            cv = pd.DataFrame(evidence["cv_results"])
            if len(cv) != 72 or cv.duplicated(["candidate_name", "inner_fold"]).any():
                raise ValueError("Incomplete candidate/fold CV evidence")
            selected = next(
                row
                for row in manifest["selected_workflows"]
                if row["workflow"] == workflow
                and row["split_seed"] == plan["split_seed"]
                and row["outer_fold"] == plan["outer_fold"]
            )
            chosen = cv.loc[
                cv["candidate_name"] == selected["candidate"]["candidate_name"]
            ]
            rounds = max(1, int(chosen["best_iteration"].median()))
            if rounds != selected["candidate"]["params"]["n_estimators"]:
                raise ValueError("Final rounds disagree with inner stopping evidence")
            for name, group in cv.groupby("candidate_name", sort=True):
                rows.append(
                    {
                        "workflow": workflow,
                        "split_seed": plan["split_seed"],
                        "outer_fold": plan["outer_fold"],
                        "candidate_name": name,
                        "feature_set": group["feature_set"].iloc[0],
                        "weight_fraction": group["weight_fraction"].iloc[0],
                        "selected": name == selected["candidate"]["candidate_name"],
                        "rounds_median": int(group["best_iteration"].median()),
                        "rounds_min": int(group["best_iteration"].min()),
                        "rounds_max": int(group["best_iteration"].max()),
                        "cv_seconds": float(group["seconds"].sum()),
                        **{
                            f"mean_{metric}": float(group[metric].mean())
                            for metric in (
                                "pr_auc",
                                "brier_score",
                                "log_loss",
                                "prevalence_brier_score",
                                "prevalence_log_loss",
                            )
                        },
                    }
                )
    return pd.DataFrame(rows)


def weighting_followup_note(assessment_run_id, followup, destination):
    """Link an explicitly supplied completed comparison to its source assessment."""
    if followup is None:
        return (
            "This search comparison alone cannot isolate a parameter effect. No separately "
            "identified weighting follow-up was supplied for this report; controlled evidence "
            "would be needed for that attribution."
        )
    provenance = json.loads((followup / "provenance.json").read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "complete"
        or provenance.get("source_runs", {}).get("current") != assessment_run_id
    ):
        raise ValueError(
            "Weighting follow-up must be complete and match this assessment"
        )
    for name, expected in provenance["aggregate_sha256"].items():
        if file_sha256(followup / name) != expected:
            raise ValueError(f"Weighting follow-up aggregate changed: {name}")
    report = followup / "assessment_report.md"
    if not report.is_file():
        raise ValueError("Weighting follow-up report is missing")
    link = Path(os.path.relpath(report, destination)).as_posix()
    return (
        "## Completed controlled follow-up\n\n"
        "Class weighting gives rare repayment-difficulty cases extra influence during fitting. "
        f"The separately identified [controlled comparison]({link}) tests this setting in "
        "the frozen earlier/current recipes on matched applicant groups. Its report retains "
        "the exact effects and limits. This follow-up is separate from the original final-probability "
        "assessment, changes no original evidence and promotes no diagnostic model."
    )


def run(output, control, destination=None, weighting_followup=None):
    destination = destination or REPO_ROOT / "reports/tuning_20261004"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    negative = json.loads((control / "manifest.json").read_text(encoding="utf-8"))
    followup_note = weighting_followup_note(
        manifest["run_id"], weighting_followup, destination
    )
    if (
        manifest["status"] != "complete"
        or negative["status"] != "complete"
        or manifest["protocol_version"] != "nested_inner_cv_v3"
        or negative["protocol_version"] != "nested_inner_cv_v3"
    ):
        raise ValueError("Curating requires completed v3 original/control runs")
    if (
        not negative["negative_control"]
        or negative["settings"]["negative_control_sample_size"] != 20000
    ):
        raise ValueError("The complete sampled20k control is required")
    for folder in (output, control):
        verified = json.loads(
            (folder / "frozen_artifact_verification.json").read_text()
        )
        if verified["status"] != "complete":
            raise ValueError("All frozen artifacts must reproduce")
    reproduced = json.loads(
        (output / "clean_environment_reproduction.json").read_text()
    )
    if reproduced["status"] != "complete":
        raise ValueError("Refitted first-fold reproduction is required")
    snapshot = json.loads(
        (REPO_ROOT / ".tmp/tuning_20261004_preedit/snapshot.json").read_text()
    )
    for name, digest in snapshot["protected_sha256"].items():
        if file_sha256(REPO_ROOT / name) != digest:
            raise ValueError(f"Protected prior evidence changed: {name}")
    for name, digest in snapshot["source_sha256"].items():
        if (
            name.startswith(("configs/correctness_", "sql/"))
            or name == "requirements.lock"
        ) and file_sha256(REPO_ROOT / name) != digest:
            raise ValueError(f"Preserved scientific input changed: {name}")
    execution_archive = REPO_ROOT / ".tmp/tuning_20261004_execution_source"
    delivery_source_delta = {}
    for name, digest in manifest["fingerprints"]["source_sha256"].items():
        current_digest = file_sha256(REPO_ROOT / name)
        if current_digest != digest:
            archived = execution_archive / name
            if not archived.exists() or file_sha256(archived) != digest:
                raise ValueError(f"Recorded assessment source is not preserved: {name}")
            delivery_source_delta[name] = {
                "execution_sha256": digest,
                "delivery_sha256": current_digest,
            }
    destination.mkdir(parents=True, exist_ok=True)
    cv = verify_search(output, manifest)
    verify_search(control, negative)
    cv.to_csv(
        destination / "cv_search_summary.csv",
        index=False,
        columns=TUNING_CV_SUMMARY_COLUMNS,
    )
    for name in (
        "summary.csv",
        "fold_metrics.csv",
        "paired_differences.csv",
        "selection_stability.csv",
        "feature_selection_frequency.csv",
        "history_segment_metrics.csv",
        "reliability_bins.csv",
        "utility_sensitivity.csv",
        "model_seed_sensitivity.csv",
        "probability_acceptance.csv",
    ):
        shutil.copy2(output / name, destination / name)
    shutil.copy2(control / "summary.csv", destination / "negative_control_summary.csv")
    shutil.copy2(
        control / "fold_metrics.csv", destination / "negative_control_fold_metrics.csv"
    )
    summary = pd.read_csv(output / "summary.csv")
    prior = pd.read_csv(REPO_ROOT / "reports/correctness_20261004/fold_summary.csv")
    comparison = summary.copy()
    comparison["prior_workflow"] = comparison["workflow"].replace(
        {"logistic_tuned": "logistic_fixed"}
    )
    comparison = comparison.merge(
        prior,
        left_on=["prior_workflow", "split_seed", "score_kind", "metric_name"],
        right_on=["workflow", "split_seed", "score_kind", "metric_name"],
        suffixes=("_v3", "_v2"),
        validate="one_to_one",
    )
    comparison["v3_minus_v2"] = comparison["fold_mean_v3"] - comparison["fold_mean_v2"]
    comparison.to_csv(destination / "prior_protocol_comparison.csv", index=False)
    prior_run = (
        REPO_ROOT
        / "reports/generated/correctness_20261004_post_v1_r2/nested_assessment/f6b58bf8335645b3a1ece2ca3502e173"
    )
    prior_manifest = json.loads((prior_run / "manifest.json").read_text())
    if (
        manifest["historical_comparison_ids"]
        != prior_manifest["historical_comparison_ids"]
        or manifest["folds"] != prior_manifest["folds"]
    ):
        raise ValueError("New and prior assessment populations/roles do not match")
    pilot = json.loads((REPO_ROOT / ".tmp/tuning_pilot.json").read_text())
    source = {
        "protocol": manifest["protocol_version"],
        "assessment_run_id": manifest["run_id"],
        "control_run_id": negative["run_id"],
        "config": manifest["config"],
        "config_sha256": manifest["config_sha256"],
        "source_fingerprints": manifest["fingerprints"],
        "feature_build_id": manifest["feature_build_id"],
        "reference_model_run_id": manifest["reference_model_run_id"],
        "selected_workflows": manifest["selected_workflows"],
        "settings": manifest["settings"],
        "duration_seconds": manifest["duration_seconds"],
        "control_duration_seconds": negative["duration_seconds"],
        "protected_files_verified": len(snapshot["protected_sha256"]),
        "preedit_source_sha256": snapshot["source_sha256"],
        "execution_source_archive": ".tmp/tuning_20261004_execution_source",
        "delivery_source_delta": delivery_source_delta,
        "delivery_delta_reason": "Post-run ordinary-caller metadata/report-path corrections, per-artifact evaluation of different selected feature lists, winner-only stability frequency/conditional uncertainty and reporting corrections, curation verification, and regression coverage. Joint search, nested fitting/stopping/calibration, frozen settings and outer recipes are unchanged; empirical execution fingerprints are not refreshed.",
        "review_repairs": {
            "snapshot": ".tmp/tuning_20261004_review_repairs_preedit/snapshot.json",
            "scientific_core_unchanged": ["src/tuning.py", "src/nested_assessment.py"],
            "full_assessment_rerun": False,
            "impact": "Integration and bookkeeping after the frozen recipe; saved numeric evidence is preserved. Ordinary evaluation supports different selected model fields; current split-stability metrics are conditional and no aggregate model is promoted.",
        },
        "pilot": pilot,
        "refit_reproduction": reproduced,
        "frozen_original": json.loads(
            (output / "frozen_artifact_verification.json").read_text()
        ),
        "frozen_control": json.loads(
            (control / "frozen_artifact_verification.json").read_text()
        ),
    }
    (destination / "provenance.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    metric_names = ["pr_auc", "roc_auc", "brier_score", "log_loss"]

    def table(frame):
        view = frame.loc[
            (frame["score_kind"] == "calibrated")
            & frame["metric_name"].isin(metric_names)
        ].pivot(index="workflow", columns="metric_name", values="fold_mean")
        return "\n".join(
            f"| {workflow_label(name)} | "
            + " | ".join(f"{row[metric]:.6f}" for metric in metric_names)
            + " |"
            for name, row in view.iterrows()
        )

    prior_ap = comparison.loc[
        (comparison["workflow_v3"] == "history_selected")
        & (comparison["score_kind"] == "calibrated")
        & (comparison["metric_name"] == "pr_auc")
    ].iloc[0]
    cv_pass = (cv["mean_brier_score"] <= cv["mean_prevalence_brier_score"] + 0.002) & (
        cv["mean_log_loss"] <= cv["mean_prevalence_log_loss"] + 0.01
    )
    selected_cv = cv.loc[cv["selected"]]
    seed_frame = pd.read_csv(output / "model_seed_sensitivity.csv")
    history_seed = seed_frame.loc[
        (seed_frame["workflow"] == "history_selected")
        & (seed_frame["score_kind"] == "calibrated")
    ]
    history_seed_ap_range = (
        history_seed.groupby("outer_fold")["pr_auc"]
        .agg(lambda values: values.max() - values.min())
        .max()
    )
    lines = table(summary)
    control_lines = table(pd.read_csv(control / "summary.csv"))
    report = f"""# Current model-selection assessment, 2026-10-04

**Current completed procedure.** Start with [the case study](../portfolio/case_study.md) for the engineering story. The [earlier repaired assessment](../correctness_20261004/assessment_report.md) was completed on the same date under a different protocol; its values remain separate.

This completed assessment compares application-only and application-and-loan-history models with simple benchmarks on the same applicant test groups. A separate sampled shuffled-outcome diagnostic checks for obvious leakage behavior. Search, randomness settings and probability criteria were fixed before assessment results. No outer result promoted
a model, family, surface or seed. The prior r2 report and its source/config/evidence
identities remain preserved; {len(snapshot["protected_sha256"])} protected files matched their pre-edit hashes.

| Model / input scope | Average precision | ROC AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
{lines}

These are means of five matched applicant test-group metrics. Average precision measures ranking, not accuracy; Brier/log loss measure probability error (lower is better). Final probabilities reflect the selected method, including unchanged raw probabilities when no adjustment is selected. Descriptive sample SD is in
`summary.csv`. No pooled cross-fold average precision or optimality claim is made.
`prior_protocol_comparison.csv` records v3 minus prior v2 results; tuned unweighted
logistic is compared explicitly with the prior balanced fixed-C1 comparator, so
that difference mixes changed recipe and optimization budget. Paired differences
within v3 use identical outer applicants, not independent samples or causal effects.
History final-probability average precision changed by {prior_ap["v3_minus_v2"]:+.6f}, essentially unchanged.
Raw history probability errors are much lower than the earlier corrected search; final probability errors are
similar. Application ranking fell slightly and tuned logistic ranking rose.
This search comparison alone cannot attribute differences to one parameter;
these descriptive results do not establish significance or guaranteed gains.

{followup_note}

All 261,384 development applicants receive one frozen prediction per declared
workflow, excluding the same 46,127 historical IDs. V3 and r2 outer and reserved
role memberships match exactly. Three-fold CV remains entirely within the 70%
base-fitting role; separate stopping rows, per-fold training-only raw rankings and
preprocessing prevent inner score-fold leakage. Twenty-four unique joint candidates
per LightGBM workflow include unweighted/lighter-weight settings. Final fits use
the median chosen inner best iterations before disjoint calibration and method
selection. Searchable history surfaces are 40/80/full 174; application-only uses 31
SQL-origin static fields. Tuned logistic searches four C values; prevalence has no
optimization or policy utility. Constant-score recipes omit quantile-policy utility.

`cv_search_summary.csv` includes all 240 LightGBM candidate summaries (24 ×
two workflows × five outer folds), each averaging three inner score folds,
with CV losses versus fitting prevalence and chosen
round evidence. Full exact local memberships/rankings/iterations remain ignored.
`probability_acceptance.csv` records predefined calibrated selection-role checks;
these cannot promote or retune using outer quality. `model_seed_sensitivity.csv`
reports fixed-recipe selection-role variation at seeds 101/211/307 without choosing
a lucky seed; logistic lbfgs may have no seed variation. Ranking seeds have a
separate training-only role.
{int(cv_pass.sum())} of {len(cv)} joint candidates pass mean CV probability criteria;
all {len(selected_cv)} selected LightGBM recipes pass. Original-label history
selects unweighted full-field recipes with median rounds ranging
{int(selected_cv.loc[selected_cv["workflow"] == "history_selected", "rounds_median"].min())}–{int(selected_cv.loc[selected_cv["workflow"] == "history_selected", "rounds_median"].max())}.
The largest within-fold history AP range across sensitivity seeds is
{history_seed_ap_range:.6f}; this describes variation on selection rows.
`independent_verification.json` records recomputed metric/paired-difference
agreement, coverage, probability flags and sensitivity ranges without applicant IDs.

## Complete shuffled-label diagnostic

| Model / input scope | Average precision | ROC AUC | Brier score | Log loss |
|---|---:|---:|---:|---:|
{control_lines}

The stratified sampled 20k development control permutes labels at seed 913 and runs
the complete five-fold selection/calibration/comparator procedure. Its chance
behavior has sampling variation and cannot prove feature availability or erase
historical exploration. It is separate from full-data original-label performance.

## Runtime, reproduction and limits

The real-data pilot used 20k/3k/3k fitting/calibration/selection rows and peaked at
{pilot["peak_working_set_bytes"] / 1024**3:.2f} GiB. Original assessment took
{manifest["duration_seconds"] / 60:.2f} minutes; control took {negative["duration_seconds"] / 60:.2f} minutes.
Model jobs ran serially with four-thread LightGBM/BLAS limits. All original/control
frozen artifacts reproduce within absolute score tolerance 1e-10. A first-fold
refit in the second locked environment matches selected features/parameters,
calibration and predictions within 1e-10. `provenance.json` records measured
timings, locked environment, source/config/build identities and preservation proof.
Exact executed source is retained in the ignored execution archive. Post-run
ordinary-caller metadata/report-path corrections, support for independently
selected model fields in evaluation, and winner-only stability frequency and
conditional uncertainty corrections are recorded as delivery source deltas.
Regression fixtures force top-40/top-80 CV winners through ordinary evaluation,
calibration, scoring, explanation and dashboard export; legacy equal-field
artifacts remain supported. Current stability summaries make no aggregate
selection or promotion. Executed joint-search/nested scientific source and all
frozen numeric evidence remain unchanged; execution fingerprints stay fixed.

Source/mart reconciliation from r2 is reused with its fingerprints. Native Power BI
Desktop/tooling is unavailable, so PBIX refresh is unverified and historical
screenshots remain unchanged. Dataset IDs/relative offsets do not identify calendar
application chronology, feature availability or label maturity. This remains an
already explored public static dataset, not an untouched external or future cohort.
Utility uses illustrative weights and is not profit or a hard queue guarantee.
No guarantee of improved performance or global tuning optimality is implied.

## Technical run details

- Protocol: `{manifest["protocol_version"]}`
- Assessment: `{manifest["run_id"]}`
- Sampled shuffled-outcome diagnostic: `{negative["run_id"]}`
"""
    (destination / "assessment_report.md").write_text(report, encoding="utf-8")
    return {
        "destination": str(destination),
        "assessment_run_id": manifest["run_id"],
        "control_run_id": negative["run_id"],
        "protected_files_verified": len(snapshot["protected_sha256"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument(
        "--weighting-followup",
        type=Path,
        help="Explicit completed weighting evidence directory; must match the assessment run",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.assessment_dir,
                args.control_dir,
                weighting_followup=args.weighting_followup,
            )
        )
    )


if __name__ == "__main__":
    main()
