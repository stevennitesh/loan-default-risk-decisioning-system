"""Reconcile a corrected bundle and publish anonymous aggregate portfolio evidence."""

import argparse
import html
import json
import shutil
import textwrap
from pathlib import Path

import joblib
import matplotlib

from src.presentation import ACTION_LABELS, metric_label, score_label, workflow_label

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import business_assumptions, load_config, manual_review_capacity_rate
from src.dashboard_exports import EXPORT_TABLE_COLUMNS
from src.evidence import file_sha256, fingerprints
from src.metrics import probability_metrics
from src.runtime import REPO_ROOT, resolve_config_path

NATIVE_LIMIT = "Power BI Desktop, pbi-tools, Tabular Editor and DAX Studio are unavailable on this host. Native PBIX refresh was not performed. Layout metadata confirms two pages and 15/8 visuals, but opaque DataModel measures/relationships/import queries prevent DAX, cross-table filters and CSV import-width certification. Historical dollar formats represent utility weights, not profit."
ORIGINAL_ASSETS = {
    "powerbi/dashboard.pbix": "a5596afc94c7ab52267c0e04c95b130506a715a15ff332cee7e5ff44114797e7",
    "powerbi/dashboard_post_v1.pbix": "42097f42dcb8a3adab31817fde321cb50a332d90693095a51a7f7943d607ea0b",
    "powerbi/screenshots/decisioning_overview.png": "21cebf6e2aae8928c4b5809c096691547b6608e1ab429b0e30206631a5a90704",
    "powerbi/screenshots/model_validation_appendix.png": "f9e1ac806386cb8322923067dad3c99df441e0e74670f4b61676726ecd81db99",
}


def reconcile_dashboard(config):
    folder = resolve_config_path(config, "dashboard_export_dir")
    tables = {
        name: pd.read_csv(folder / f"{name}.csv") for name in EXPORT_TABLE_COLUMNS
    }
    for name, columns in EXPORT_TABLE_COLUMNS.items():
        assert tables[name].columns.tolist() == columns, (
            f"Export schema mismatch {name}"
        )
    scores = tables["credit_risk_scores"]
    test = scores.loc[scores["scoring_population"] == "holdout_test"]
    kaggle = scores.loc[scores["scoring_population"] != "holdout_test"]
    assert (
        kaggle["observed_target"].isna().all() and test["observed_target"].notna().all()
    )
    assert not scores.duplicated(["applicant_id", "scoring_population"]).any()
    assert np.allclose(scores["score"], scores["raw_risk_score"], atol=0, rtol=0)
    artifact = joblib.load(
        resolve_config_path(config, "model_dir") / "lightgbm_credit_risk.joblib"
    )
    reference = json.loads(
        (
            resolve_config_path(config, "model_dir") / "historical_split_reference.json"
        ).read_text()
    )
    assert (
        set(test["applicant_id"])
        == set(artifact["split_applicant_ids"]["test"])
        == set(reference["split_applicant_ids"]["test"])
    )
    (alias,) = scores["model_version"].unique()
    metrics = tables["model_metrics_summary"]
    selected = metrics.loc[
        (metrics["model_version"] == alias) & (metrics["split"] == "test")
    ]
    assert not selected["metric_name"].duplicated().any()
    expected = probability_metrics(
        test["observed_target"].astype(int),
        test["calibrated_risk_score"].to_numpy(),
        manual_review_capacity_rate(config),
    )
    assert set(selected["metric_name"]) == set(expected)
    for row in selected.to_dict("records"):
        assert np.isclose(
            row["metric_value"], expected[row["metric_name"]], rtol=1e-10, atol=1e-12
        ), f"Metric mismatch {row['metric_name']}"
    threshold = tables["model_threshold_metrics"]
    assumptions = business_assumptions(config)
    utility = []
    for row in threshold.loc[
        (threshold["model_version"] == alias) & (threshold["split"] == "test")
    ].to_dict("records"):
        approve = test["raw_risk_score"] < row["threshold_low"]
        high = test["raw_risk_score"] >= row["threshold_high"]
        review = ~approve & ~high
        labels = test["observed_target"]
        value = (
            (approve & (labels == 0)).sum()
            * assumptions["expected_margin_per_good_loan"]
            - (approve & (labels == 1)).sum()
            * assumptions["expected_loss_per_bad_loan"]
            - review.sum() * assumptions["manual_review_cost"]
        ) / len(test)
        assert np.isclose(
            value, row["expected_value_per_applicant"], atol=1e-10, rtol=1e-10
        )
        if row["scenario_name"] == "balanced":
            actions = np.select(
                [approve, review, high],
                ["approve", "manual_review", "simulated_decline"],
                default="invalid",
            )
            assert np.array_equal(actions, test["recommended_action"])
        utility.append(
            {
                "scenario_name": row["scenario_name"],
                "utility_per_applicant": float(value),
                "approve_count": int(approve.sum()),
                "review_count": int(review.sum()),
                "simulated_decline_count": int(high.sum()),
            }
        )
    for name, digest in ORIGINAL_ASSETS.items():
        assert file_sha256(REPO_ROOT / name) == digest, (
            f"Historical asset changed {name}"
        )
    calibration = joblib.load(
        resolve_config_path(config, "model_dir")
        / "lightgbm_credit_risk_calibration.joblib"
    )
    return {
        "status": "complete",
        "model_run_id": artifact["run_id"],
        "feature_build_id": artifact["feature_build_id"],
        "calibration_run_id": calibration["calibration_run_id"],
        "dashboard_alias": alias,
        "labeled_comparison_count": len(test),
        "unlabeled_kaggle_count": len(kaggle),
        "schemas_checked": len(tables),
        "test_probability_metrics": expected,
        "scenario_checks": utility,
        "csv_sha256": {name: file_sha256(folder / f"{name}.csv") for name in tables},
        "historical_asset_sha256": ORIGINAL_ASSETS,
        "native_powerbi_limit": NATIVE_LIMIT,
    }, tables


def renewed_comparison(config, v1_config, destination):
    """Compare rebuilt saved models using the same historical membership."""
    artifacts = {}
    rows = []
    for scope, scoped_config in (("v1", v1_config), ("post_v1", config)):
        artifact = joblib.load(
            resolve_config_path(scoped_config, "model_dir")
            / "lightgbm_credit_risk.joblib"
        )
        artifacts[scope] = artifact
        metrics = pd.read_csv(
            resolve_config_path(scoped_config, "report_dir")
            / "model_metrics_summary.csv"
        )
        selected = metrics.loc[
            (metrics["model_version"] == artifact["model_version"])
            & (metrics["split"] == "test")
        ]
        assert len(selected) == 9 and not selected["metric_name"].duplicated().any()
        rows.extend(
            {
                "scope": scope,
                "score_kind": "raw",
                "model_run_id": artifact["run_id"],
                "feature_build_id": artifact["feature_build_id"],
                "feature_count": len(artifact["feature_columns"]),
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
            }
            for row in selected.itertuples()
        )
    v1_ids = artifacts["v1"]["split_applicant_ids"]
    post_ids = artifacts["post_v1"]["split_applicant_ids"]
    assert all(set(v1_ids[role]) == set(post_ids[role]) for role in ("train", "test"))
    assert set(v1_ids["validation"]) == set(post_ids["validation"]) | set(
        post_ids["calibration"]
    ), "Renewed comparison requires the same original reserved population"
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "renewed_saved_model_comparison.csv", index=False)
    return frame


def run(config_path, assessment, negative, destination, v1_config_path):
    assessment = assessment.resolve()
    negative = negative.resolve()
    destination = destination.resolve()
    config = load_config(config_path)
    assessment_manifest = json.loads((assessment / "manifest.json").read_text())
    negative_manifest = json.loads((negative / "manifest.json").read_text())
    assert assessment_manifest["status"] == negative_manifest["status"] == "complete"
    assert (
        not assessment_manifest["negative_control"]
        and negative_manifest["negative_control"]
    )
    reproduction = json.loads(
        (assessment / "clean_environment_reproduction.json").read_text()
    )
    assert reproduction["status"] == "complete"
    frozen_checks = {
        name: json.loads((folder / "frozen_artifact_verification.json").read_text())
        for name, folder in (
            ("full_assessment", assessment),
            ("negative_control", negative),
        )
    }
    assert all(check["status"] == "complete" for check in frozen_checks.values())
    result, tables = reconcile_dashboard(config)
    destination.mkdir(parents=True, exist_ok=True)
    provenance_path = assessment / "source_provenance.json"
    provenance_note = "Run-start `fingerprints` identify the source at assessment initialization; `delivery_fingerprints` identify the source at the later evidence publication. They are separate snapshots and are not asserted to be identical."
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        assert provenance["status"] == "complete"
        assert provenance["assessment_run_id"] == assessment_manifest["run_id"]
        assert provenance["current_nested_source_sha256"] == file_sha256(
            REPO_ROOT / "src/nested_assessment.py"
        )
        shutil.copy2(provenance_path, destination / "source_provenance.json")
        result["source_provenance_file"] = "source_provenance.json"
        provenance_note += " For this run, [source_provenance.json](source_provenance.json) records the exact drift audit. A byte-identical reconstruction matches the immutable start SHA256 of `src/nested_assessment.py`; the only two differences are its report title and explanatory paragraph, repaired after completion of the five-fold run. The module AST outside `_write_report` is identical. This reconstruction is not an originally retained source backup. The fitting/selection/assessment recipe did not drift. Post-start changes to reproduction, publication and fingerprint helpers concern proof/report metadata; the later capacity-getter repair affects post-fit diagnostics/reconciliation for non-default configurations. Refreshed recorded 10% outputs remain numerically and byte-for-byte unchanged. Original start fingerprints are preserved, never replaced with delivery hashes."
    renewed = renewed_comparison(config, load_config(v1_config_path), destination)
    report = resolve_config_path(config, "report_dir")
    source = json.loads((report / "source_reconciliation/manifest.json").read_text())
    assert source["status"] == "complete"
    summary = pd.read_csv(assessment / "summary.csv")
    negative_summary = pd.read_csv(negative / "summary.csv")
    for original, name in (
        (assessment / "summary.csv", "fold_summary.csv"),
        (assessment / "paired_differences.csv", "paired_differences.csv"),
        (negative / "summary.csv", "negative_control_summary.csv"),
        (assessment / "selection_stability.csv", "selection_stability.csv"),
        (
            assessment / "feature_selection_frequency.csv",
            "feature_selection_frequency.csv",
        ),
        (assessment / "history_segment_metrics.csv", "history_segment_metrics.csv"),
        (assessment / "reliability_bins.csv", "reliability_bins.csv"),
        (assessment / "utility_sensitivity.csv", "utility_sensitivity.csv"),
        (
            report / "source_reconciliation/coverage_by_target.csv",
            "source_coverage_by_target.csv",
        ),
    ):
        shutil.copy2(original, destination / name)
    public_source = {
        key: source[key]
        for key in (
            "status",
            "repayment_cases",
            "repayment_source_rows",
            "repayment_obligations",
            "repayment_numeric_checks",
            "identical_source_record_count",
            "duplicate_policy",
            "coverage_by_target",
            "schedule_counts",
            "feature_count",
            "excluded_model_features",
            "availability_counts",
            "bureau_origin_coverage_by_target",
            "verification_source_sha256",
            "limitations",
        )
    }
    public_source["monthly_applicant_count"] = len(source["monthly_cases"])
    public_source["monthly_numeric_checks"] = sum(
        row["checks"] for row in source["monthly_cases"]
    )
    (destination / "source_summary.json").write_text(
        json.dumps(public_source, indent=2), encoding="utf-8"
    )
    result.update(
        delivery_fingerprints=fingerprints(config),
        model_artifact_sha256={
            path.name: file_sha256(path)
            for path in resolve_config_path(config, "model_dir").glob("*.joblib")
        },
        protocol=assessment_manifest["protocol_version"],
        assessment_run_id=assessment_manifest["run_id"],
        negative_control_run_id=negative_manifest["run_id"],
        negative_control_size=negative_manifest["settings"][
            "negative_control_sample_size"
        ],
        negative_control_seed=negative_manifest["settings"]["negative_control_seed"],
        reproduction=reproduction,
        frozen_artifact_verification=frozen_checks,
        fingerprints=assessment_manifest["fingerprints"],
        local_evidence_directories={
            "pipeline": str(report.relative_to(REPO_ROOT)),
            "assessment": str(assessment.relative_to(REPO_ROOT)),
            "negative_control": str(negative.relative_to(REPO_ROOT)),
        },
    )
    # Fingerprints contain no applicant memberships; local exact role IDs remain
    # in the ignored assessment manifest, never copied to public evidence.
    (destination / "evidence_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    plot_evidence(destination, summary, tables, assessment)
    primary = summary.loc[
        (summary["score_kind"] == "calibrated")
        & summary["metric_name"].isin(["pr_auc", "roc_auc", "brier_score", "log_loss"])
    ]
    pivot = primary.pivot(index="workflow", columns="metric_name", values="fold_mean")
    table = (
        "| Model / input scope | "
        + " | ".join(metric_label(v) for v in pivot.columns)
        + " |\n|---|"
        + "---:|" * len(pivot.columns)
        + "\n"
    )
    table += "\n".join(
        "| "
        + workflow_label(name)
        + " | "
        + " | ".join(f"{value:.6f}" for value in row)
        + " |"
        for name, row in pivot.iterrows()
    )
    values = primary.set_index(["workflow", "metric_name"])["fold_mean"]
    raw_values = summary.loc[summary["score_kind"] == "raw"].set_index(
        ["workflow", "metric_name"]
    )["fold_mean"]
    paired = pd.read_csv(assessment / "paired_differences.csv")
    paired_ap = (
        paired.loc[
            (paired["score_kind"] == "calibrated") & (paired["metric_name"] == "pr_auc")
        ]
        .groupby("workflow")["history_minus_comparator"]
        .agg(["mean", "min", "max"])
    )
    segment = pd.read_csv(assessment / "history_segment_metrics.csv")
    segment = segment.loc[
        (segment["workflow"] == "history_selected")
        & (segment["score_kind"] == "calibrated")
    ]
    segment_lines = "\n".join(
        f"| {name} | {int(group.applicant_count.sum()):,} | {np.average(group.target_rate, weights=group.applicant_count):.6f} | {group.pr_auc.mean():.6f} | {group.brier_score.mean():.6f} |"
        for name, group in segment.groupby("history_segment")
    )
    renewed_values = renewed.set_index(["scope", "metric_name"])["metric_value"]
    utility = pd.read_csv(assessment / "utility_sensitivity.csv")
    history_utility = (
        utility.loc[utility["workflow"] == "history_selected"]
        .groupby(["margin_multiplier", "loss_multiplier", "review_multiplier"])[
            "utility_per_applicant"
        ]
        .mean()
    )
    # Avoid an extra markdown-table dependency in the scientific runtime.
    negative_rows = negative_summary.loc[
        (negative_summary["score_kind"] == "calibrated")
        & negative_summary["metric_name"].isin(
            ["pr_auc", "roc_auc", "brier_score", "log_loss"]
        )
    ]
    lines = [
        f"| {workflow_label(row.workflow)} | {metric_label(row.metric_name)} | {row.fold_mean:.6f} | {row.fold_std:.6f} |"
        for row in negative_rows.itertuples()
    ]
    text = f"""# Earlier repaired assessment, 2026-10-04

**Earlier completed procedure.** This `nested_matched_holdout_v2` assessment precedes the [current `nested_inner_cv_v3` assessment](../tuning_20261004/assessment_report.md), also completed on 2026-10-04. Start with [the current case study](../portfolio/case_study.md). This report compares models after repairing loan-history inputs and separating fitting from selection. It records the earlier corrected assessment, source-data checks and saved-model comparison; numbered historical experiments and Power BI assets retain their original values.

{table}

These are means of five applicant test-group metrics. Average precision measures ranking, not accuracy. Brier/log loss measure probability error (lower is better). Final probabilities reflect the selected adjustment method, including unchanged raw probabilities. `fold_summary.csv` includes raw/calibrated views and descriptive sample SD; SD is not a confidence interval. No pooled cross-fold average precision is reported. `paired_differences.csv` compares history with each declared comparator on exactly matched folds. Application-only ablation measures conditional engineering value, not causal attribution. Model families have different declared optimization budgets and are never chosen using outer metrics.

Application-and-loan-history final-probability average precision was {values["history_selected", "pr_auc"]:.6f}. Its matched-fold difference versus application-only averaged {paired_ap.loc["application_only", "mean"]:.6f} (fold range {paired_ap.loc["application_only", "min"]:.6f} to {paired_ap.loc["application_only", "max"]:.6f}); versus fixed logistic it averaged {paired_ap.loc["logistic_fixed", "mean"]:.6f} (range {paired_ap.loc["logistic_fixed", "min"]:.6f} to {paired_ap.loc["logistic_fixed", "max"]:.6f}). These observations support predictive engineering value within this population and recipe, with no causal or untouched external-performance claim.

Raw class-weighted history scores had worse probability losses than training prevalence: Brier {raw_values["history_selected", "brier_score"]:.6f} versus {values["training_prevalence", "brier_score"]:.6f}, and log loss {raw_values["history_selected", "log_loss"]:.6f} versus {values["training_prevalence", "log_loss"]:.6f}. Calibration reduced history Brier to {values["history_selected", "brier_score"]:.6f} and log loss to {values["history_selected", "log_loss"]:.6f}, below prevalence. Sigmoid was chosen in every fold for all three non-flat workflows. Its monotone mapping preserved ranking metrics, so calibration produced no average-precision/lift improvement. History selected 80 features in three folds and all 174 in two; selected settings varied across folds. This variation is documented, not used to change the predeclared holdout recipe.

## Renewed saved-model comparison

Both corrected v1 (74 features) and post-v1 (174 features) were rebuilt and fitted in isolated paths preserving the original 215,257 training and 46,127 comparison IDs. V1 uses its original 46,127 validation applicants; post-v1 partitions the same reserved population into 23,063 calibration and 23,064 selection applicants. `renewed_saved_model_comparison.csv` records fresh raw-score metrics and model/build identities, distinct from numbered historical snapshots and from the nested estimates. Historical comparison raw average precision was {renewed_values["v1", "pr_auc"]:.6f} for v1 and {renewed_values["post_v1", "pr_auc"]:.6f} for post-v1; ROC AUC was {renewed_values["v1", "roc_auc"]:.6f} and {renewed_values["post_v1", "roc_auc"]:.6f}. These differently engineered/selected saved-model recipes remain exploratory comparisons on reused IDs. The post-v1 calibrated dashboard comparison has average precision {result["test_probability_metrics"]["pr_auc"]:.6f}; calibration was fitted/selected on reserved roles rather than this comparison set.

The full population contains 307,511 labeled applicants. The original 46,127 comparison IDs remain excluded from all nested fitting/assessment; 261,384 development applicants receive one outer prediction per workflow. Five outer folds reserve inner roles 70% fitting, 15% calibration and 15% selection. History compares top-40/top-80/full surfaces with eight LightGBM candidates each and ranking seeds 101/211/307; application-only uses all SQL `f_applicant_static` model fields and eight candidates; logistic uses fixed C=1, lbfgs, balanced class weights, max_iter=1000 with training-only imputation/scaling/encoding. Training-prevalence fits only the base fitting labels, has no search/calibration/policy threshold. Model seed and outer seed are 42. No post-selection refit uses reserved rows.

## Source checks and sensitivity

The Python repayment oracle agreed on {source["repayment_numeric_checks"]:,} values across {source["repayment_obligations"]:,} actual obligations, using {source["repayment_source_rows"]:,} raw/staged payment rows. It retains repeated identical payment records because no unique cashflow key proves duplication. Independent Python monthly calculations agreed on {public_source["monthly_numeric_checks"]} values for {public_source["monthly_applicant_count"]} actual applicant histories, including multiple accounts/month and matched operand ratios. Full SQL contracts passed with 12,951,918 schedule groups, 180,085 ambiguous groups and 2,831 otherwise unambiguous unknown-payment groups. `source_summary.json` distinguishes actual examples from absent source cases; future censoring is also covered by synthetic regressions.

Strict finite bureau origin `<0` eligibility excludes 25 day-zero origins; no positive or unknown origins were present. Day-zero records are not proven future leakage. Bureau balance and recency use the same eligible-origin owner; planned future maturities remain available on eligible loans. The superseded partial run was stopped before this source correction; no outer metric informed it. Source-relative cutoffs cannot establish intraday/vendor-ingestion availability, and no reliable calendar application dates exist. Random folds cannot establish future-cohort performance.

`source_coverage_by_target.csv` reports ambiguity/unknown/no-history support by target. `history_segment_metrics.csv` assesses frozen predictions in prespecified no-history, ambiguous/unknown-history and remaining known-history populations. Their prevalence and performance differences are conditional population descriptions, with no optimistic alternate schedule inference. `selection_stability.csv` records fold-specific model/calibration/feature-count choices.

| Frozen history segment | Outer applicants | Target rate | Mean fold average precision | Mean fold Brier |
|---|---:|---:|---:|---:|
{segment_lines}

Applicant totals and target rates cover disjoint outer populations; performance columns are descriptive means of within-fold segment metrics. These segment comparisons have different prevalence, sample size and covariate support.

## Shuffled-outcome diagnostic (separate sampled population)

A stratified sample of 20,000 development applicants, selected before a seed-913 label permutation, traversed the complete five-fold inner ranking, fitting, feature/grid/method selection and frozen outer assessment with the same recipe. It remains a sampled diagnostic, separate from full-data performance.

| Model / input scope | Metric | Mean across test groups | Descriptive group SD |
|---|---|---:|---:|
{chr(10).join(lines)}

Chance-level behavior is judged with sampling variation, not an exact hard threshold. This control can detect strong leakage/selection bugs in the exercised workflow, but cannot prove real-world feature availability or undo historical dataset exploration. Regression checks prove that outer label/covariate changes cannot mutate fitted choices or preprocessors.

## Probability, policy and reproduction

Legacy `pr_auc` means average precision. Top-rate metrics use expected fractional membership within a cutoff score tie and selected mass `ceil(n*rate)`; flat scores have lift 1. Reliability bins use score-group midranks and keep equal scores together, with nominal ten bins and explicit applicant counts. Display rank deciles use target-blind score/ID ordering and are a separate descriptive rule. Log loss and Brier assess probability quality; raw weighted classifier scores are also kept as policy ranking scores.

`utility_sensitivity.csv` evaluates all 27 combinations of margin/loss/review multipliers 0.5/1/2 around 1000/5000/50 at each frozen balanced raw-score policy. It does not tune outer thresholds or estimate real profit, reviewer effectiveness, or rejected-loan counterfactuals. The low band simulates approval, middle band charges review cost, and high band simulates decline with zero modeled value/cost. Quantile scenarios have no hard queue cap. Flat prevalence scores have no defined quantile policy utility.

History's mean fold utility across this fixed grid ranges from {history_utility.min():.6f} to {history_utility.max():.6f} utility units; the original 1000/5000/50 weights give {history_utility.loc[(1, 1, 1)]:.6f}. The broad range reflects sensitivity to assumed weights, not measured business returns or a newly optimized policy.

The separate clean environment refitted declared seed-42 fold-1 history selection using the same data, role IDs, seeds and locked dependencies. Chosen features, settings and calibration matched; maximum absolute raw/calibrated prediction differences were {reproduction["max_absolute_prediction_difference"]}, within the predeclared 1e-10 tolerance. This is same-host numerical reproduction, not cross-hardware portability or independent sampling uncertainty. `requirements.lock` pins only this project's required dependency closure. The evidence manifest records full raw checksums, Git HEAD, dirty source/config hashes and dependency versions; uncommitted source hashes qualify the Git SHA.

{provenance_note}

Every frozen fold/workflow joblib was also reloaded and its raw/calibrated outer predictions regenerated without fitting: {frozen_checks["full_assessment"]["artifact_count"]} full-assessment artifacts reproduced {frozen_checks["full_assessment"]["prediction_count"]:,} predictions, and {frozen_checks["negative_control"]["artifact_count"]} sampled-control artifacts reproduced {frozen_checks["negative_control"]["prediction_count"]:,}. Each artifact's fitting-role IDs were checked to exclude its outer applicants. These checks are separate from the clean-environment refit.

## Executed local commands

The original default artifacts remain recoverable. The three `_r2` scopes below use isolated database/parquet/model/report paths; each new model directory was seeded with its original model artifact and a local membership reference before training. Completed scopes reject pipeline overwrite. Original configs/manifests/partial artifacts from before the bureau-origin repair are retained as superseded local evidence.

```powershell
uv pip compile requirements.txt --python C:\\Users\\steve\\miniforge3\\python.exe --output-file requirements.lock --generate-hashes --cache-dir .tmp/uv-cache
uv venv .tmp/assessment-env --python C:\\Users\\steve\\miniforge3\\python.exe
uv pip sync requirements.lock --python .tmp/assessment-env/Scripts/python.exe --require-hashes --cache-dir .tmp/uv-cache
$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
.tmp/assessment-env/Scripts/python.exe -m src.correctness_run --config configs/correctness_20261004_post_v1_r2.yaml
.tmp/assessment-env/Scripts/python.exe -m src.correctness_run --config configs/correctness_20261004_v1_r2.yaml
.tmp/assessment-env/Scripts/python.exe -m src.source_reconciliation --config configs/correctness_20261004_post_v1_r2.yaml
.tmp/assessment-env/Scripts/python.exe -m src.nested_assessment --config configs/correctness_20261004_post_v1_r2.yaml
.tmp/assessment-env/Scripts/python.exe -m src.nested_assessment --config configs/correctness_20261004_negative_control_r2.yaml
.tmp/assessment-env/Scripts/python.exe -m src.assessment_diagnostics --assessment-dir {assessment.relative_to(REPO_ROOT).as_posix()}
.tmp/assessment-env/Scripts/python.exe -m src.assessment_diagnostics --assessment-dir {negative.relative_to(REPO_ROOT).as_posix()}
uv venv .tmp/assessment-clean-env --python C:\\Users\\steve\\miniforge3\\python.exe
uv pip sync requirements.lock --python .tmp/assessment-clean-env/Scripts/python.exe --require-hashes --cache-dir .tmp/uv-cache
.tmp/assessment-clean-env/Scripts/python.exe -m src.assessment_reproduce --assessment-dir {assessment.relative_to(REPO_ROOT).as_posix()}
.tmp/assessment-env/Scripts/python.exe -m src.verify_frozen_assessment --assessment-dir {assessment.relative_to(REPO_ROOT).as_posix()}
.tmp/assessment-env/Scripts/python.exe -m src.verify_frozen_assessment --assessment-dir {negative.relative_to(REPO_ROOT).as_posix()}
.tmp/assessment-env/Scripts/python.exe -m src.correctness_summary --config configs/correctness_20261004_post_v1_r2.yaml --v1-config configs/correctness_20261004_v1_r2.yaml --assessment-dir {assessment.relative_to(REPO_ROOT).as_posix()} --negative-control-dir {negative.relative_to(REPO_ROOT).as_posix()}
make lint format-check test PYTHON=.tmp/assessment-env/Scripts/python.exe
```

Final local verification is recorded separately in `verification.json`. Dependency installation used project-local environments/cache and did not mutate the global interpreter environment.

## Dashboard proof and native gap

The identified corrected CSV bundle reconciled all {result["schemas_checked"]} schemas, exact original comparison IDs, calibrated comparison metrics, raw policy scores, balanced actions and scenario utility. It contains {result["labeled_comparison_count"]:,} labeled comparison rows and {result["unlabeled_kaggle_count"]:,} unlabeled Kaggle scores. Its model run is `{result["model_run_id"]}`, feature build `{result["feature_build_id"]}`, calibrator `{result["calibration_run_id"]}`; CSV hashes are recorded.

{NATIVE_LIMIT}

The historical overview filters model/`test` and uses balanced scenario filters for most KPIs; the comparison chart intentionally spans scenarios. The appendix filter is attached to feature importance, so cross-table propagation is unverified. Screenshot `held-out` means historical reused comparison. Historical score histograms mix labeled comparison and unlabeled scoring populations unless explicitly filtered. No visible binding certifies simulated-decline actions. These historical assets remain byte-identical. `corrected_evidence.html` and the PNG below are newly generated aggregate evidence from the verified bundle, not a refreshed PBIX.

![Corrected aggregate evidence](corrected_evidence.png)

## Technical run details

- Protocol: `{result["protocol"]}`
- Assessment: `{result["assessment_run_id"]}`
"""
    (destination / "assessment_report.md").write_text(text, encoding="utf-8")
    return result


def plot_evidence(destination, summary, tables, assessment):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    calibrated = summary.loc[
        (summary["score_kind"] == "calibrated") & (summary["metric_name"] == "pr_auc")
    ]
    axes[0, 0].barh(
        calibrated["workflow"].map(workflow_label),
        calibrated["fold_mean"],
        xerr=calibrated["fold_std"],
    )
    axes[0, 0].set(
        title="Earlier five-group assessment: 261,384 applicants\nAverage precision; descriptive group SD",
        xlabel="Average precision",
    )
    bins = pd.read_csv(assessment / "reliability_bins.csv")
    for fold, group in bins.loc[
        (bins["workflow"] == "history_selected")
        & (bins["score_kind"] == "calibrated")
        & (bins["applicant_count"] > 0)
    ].groupby("outer_fold"):
        axes[0, 1].plot(
            group["average_predicted_score"],
            group["observed_default_rate"],
            marker=".",
            label=f"Applicant test group {fold}",
        )
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 1].set(
        title="Earlier five-group assessment: 261,384 applicants\nFinal-probability reliability; ties stay together",
        xlabel="Mean predicted repayment-difficulty probability",
        ylabel="Observed difficulty rate",
        xlim=(0, 0.5),
        ylim=(0, 0.5),
    )
    axes[0, 1].legend()
    scores = tables["credit_risk_scores"]
    comparison = scores.loc[scores["scoring_population"] == "holdout_test"]
    counts = comparison["recommended_action"].value_counts()
    axes[1, 0].bar([ACTION_LABELS.get(v, v) for v in counts.index], counts.values)
    axes[1, 0].set(
        title=f"Saved post-v1 model: {len(comparison):,} reused applicants\nBalanced-reference simulated actions",
        ylabel="Applicant count",
    )
    for kind in ("raw_risk_score", "calibrated_risk_score"):
        axes[1, 1].hist(
            comparison[kind],
            bins=30,
            alpha=0.5,
            label="Raw probabilities"
            if kind == "raw_risk_score"
            else "Final probabilities",
        )
    axes[1, 1].set(
        title=f"Saved post-v1 model: {len(comparison):,} reused applicants\nSeparate from the five-group assessment",
        xlabel="Score",
        ylabel="Applicant count",
    )
    axes[1, 1].legend()
    fig.suptitle(
        "Earlier repaired assessment | 2026-10-04 | nested_matched_holdout_v2",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(destination / "corrected_evidence.png", dpi=160)
    plt.close(fig)
    rows = summary.loc[
        summary["metric_name"].isin(
            [
                "pr_auc",
                "roc_auc",
                "brier_score",
                "log_loss",
                "balanced_utility_per_applicant",
            ]
        )
    ].copy()
    rows["workflow"] = rows["workflow"].map(workflow_label)
    rows["metric_name"] = rows["metric_name"].map(metric_label)
    rows["score_kind"] = rows["score_kind"].map(score_label)
    rows = rows.rename(
        columns={
            "workflow": "Model / input scope",
            "metric_name": "Metric",
            "score_kind": "Probability view",
            "fold_mean": "Mean across test groups",
            "fold_std": "Descriptive group SD",
        }
    ).to_html(index=False, float_format=lambda value: f"{value:.6f}")
    (destination / "corrected_evidence.html").write_text(
        f"<!doctype html><html><meta charset='utf-8'><title>Corrected local assessment</title><style>body{{font:16px system-ui;margin:2rem;max-width:1300px}}table{{border-collapse:collapse}}td,th{{padding:.4rem;border:1px solid #ddd}}img{{max-width:100%}}</style><h1>Earlier repaired assessment, 2026-10-04</h1><p>Earlier protocol: nested_matched_holdout_v2. The <a href='../portfolio/index.html'>current case study</a> and <a href='../tuning_20261004/assessment_report.md'>current assessment</a> use nested_inner_cv_v3, completed later on the same date.</p><p>Upper panels: 261,384 development applicants; lower panels: 46,127 reused historical comparison applicants from the saved post-v1 model. These are separate model populations.</p><p>Five matched applicant test groups. Raw/final probability views are separate; final may select unchanged raw probabilities. Fold SD is descriptive, not a confidence interval. Saved-model comparison scores are reused historical comparison; unlabeled Kaggle scores are excluded from these visuals.</p><img src='corrected_evidence.png' alt='Corrected anonymous aggregate evidence'><p>{html.escape(NATIVE_LIMIT)}</p>{rows}</html>",
        encoding="utf-8",
    )


def refresh_presentation(destination):
    """Render the earlier repaired view from frozen anonymous aggregates only.

    This does not call curation, read applicant scores or alter empirical manifests.
    Returned hashes are presentation-delivery proof, separate from verification.json.
    """
    names = (
        "fold_summary.csv",
        "reliability_bins.csv",
        "renewed_saved_model_comparison.csv",
        "evidence_manifest.json",
    )
    inputs = {name: file_sha256(destination / name) for name in names}
    manifest = json.loads(
        (destination / "evidence_manifest.json").read_text(encoding="utf-8")
    )
    verification = json.loads(
        (destination / "verification.json").read_text(encoding="utf-8")
    )
    if (
        manifest["status"] != "complete"
        or manifest["protocol"] != "nested_matched_holdout_v2"
    ):
        raise ValueError(
            "Presentation refresh requires completed earlier repaired evidence"
        )
    for name in names:
        expected = verification["curated_artifact_sha256"].get(name)
        if expected is not None and inputs[name] != expected:
            raise ValueError(f"Frozen presentation input changed: {name}")
    summary = pd.read_csv(destination / "fold_summary.csv")
    bins = pd.read_csv(destination / "reliability_bins.csv")
    bins = bins.loc[
        (bins.workflow == "history_selected") & (bins.score_kind == "calibrated")
    ]
    assessment_count = int(bins.applicant_count.sum())
    comparison_count = manifest["labeled_comparison_count"]
    balanced = next(
        row for row in manifest["scenario_checks"] if row["scenario_name"] == "balanced"
    )
    counts = [
        balanced[key]
        for key in ("approve_count", "review_count", "simulated_decline_count")
    ]
    if sum(counts) != comparison_count or bins.outer_fold.nunique() != 5:
        raise ValueError("Presentation populations do not reconcile")
    renewed = pd.read_csv(destination / "renewed_saved_model_comparison.csv")
    raw_ap = renewed.loc[
        (renewed.score_kind == "raw") & (renewed.metric_name == "pr_auc")
    ]
    if set(raw_ap.scope) != {"v1", "post_v1"} or len(raw_ap) != 2:
        raise ValueError("Saved-model comparison must identify both earlier scopes")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    primary = summary.loc[
        (summary.score_kind == "calibrated") & (summary.metric_name == "pr_auc")
    ]
    axes[0, 0].barh(
        [textwrap.fill(workflow_label(value), 28) for value in primary.workflow],
        primary.fold_mean,
        xerr=primary.fold_std,
        color="#087f82",
    )
    axes[0, 0].set(
        title=f"Five-group assessment: {assessment_count:,} applicants\nFinal-probability ranking; descriptive group SD",
        xlabel="Average precision (higher is better; not accuracy)",
    )
    for fold, group in bins.groupby("outer_fold"):
        axes[0, 1].plot(
            group.average_predicted_score,
            group.observed_default_rate,
            marker=".",
            label=f"Group {fold}",
        )
    axes[0, 1].plot([0, 0.5], [0, 0.5], "k--", alpha=0.5)
    axes[0, 1].set(
        title=f"Five-group assessment: {assessment_count:,} applicants\nHistory final-probability reliability",
        xlabel="Mean predicted repayment-difficulty probability",
        ylabel="Observed repayment-difficulty rate",
        xlim=(0, 0.5),
        ylim=(0, 0.5),
    )
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].bar(
        ["Simulated\napproval", "Manual\nreview", "Simulated\ndecline"],
        counts,
        color="#087f82",
    )
    axes[1, 0].set(
        title=f"Saved post-v1 model: {comparison_count:,} reused applicants\nBalanced-reference simulated actions",
        ylabel="Applicant count",
    )
    axes[1, 1].bar(
        ["Saved v1\n74 inputs", "Saved post-v1\n174 inputs"],
        raw_ap.set_index("scope").loc[["v1", "post_v1"], "metric_value"],
        color=["#8a6397", "#087f82"],
    )
    axes[1, 1].set(
        title=f"Saved-model comparison: {comparison_count:,} reused applicants\nSeparate saved fits; raw-probability ranking",
        ylabel="Average precision (higher is better; not accuracy)",
        ylim=(0, 0.3),
    )
    for axis in axes.flat:
        axis.grid(axis="x" if axis == axes[0, 0] else "y", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Earlier repaired assessment | 2026-10-04 | nested_matched_holdout_v2",
        fontsize=16,
        weight="bold",
    )
    fig.supxlabel(
        "Upper panels: matched development assessment. Lower panels: reused historical comparison, not an untouched test.\nCurrent results: stevennitesh.github.io/loan-default-risk-decisioning-system/ | Presentation refresh only; no PBIX refresh.",
        fontsize=10,
    )
    fig.savefig(destination / "corrected_evidence.png", dpi=160)
    plt.close(fig)
    rows = summary.loc[
        summary.metric_name.isin(
            [
                "pr_auc",
                "roc_auc",
                "brier_score",
                "log_loss",
                "balanced_utility_per_applicant",
            ]
        )
    ].copy()
    rows["workflow"] = rows.workflow.map(workflow_label)
    rows["metric_name"] = rows.metric_name.map(metric_label)
    rows["score_kind"] = rows.score_kind.map(score_label)
    rows["model_family"] = rows.model_family.map(
        {
            "lightgbm": "Tree ensemble",
            "logistic_regression": "Logistic regression",
            "prevalence": "Constant benchmark",
        }
    )
    rows = rows.rename(
        columns={
            "workflow": "Model / input scope",
            "model_family": "Model family",
            "split_seed": "Grouping seed",
            "fold_count": "Applicant test groups",
            "metric_name": "Metric",
            "score_kind": "Probability view",
            "fold_mean": "Mean across groups",
            "fold_std": "Descriptive group SD",
        }
    )
    table = rows.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Earlier repaired assessment, 2026-10-04</title><style>*{{box-sizing:border-box}}body{{font:16px/1.6 system-ui;margin:0;background:#f6f8fa;color:#182230}}main{{max-width:1100px;margin:auto;padding:1.5rem}}img{{display:block;width:100%;height:auto}}figure{{margin:1.5rem 0;border:1px solid #cad6df;border-radius:10px;background:white;overflow:hidden}}figcaption{{padding:1rem;background:#e9f1f3}}.table-wrap{{max-width:100%;overflow-x:auto}}table{{border-collapse:collapse;font-size:.85rem}}td,th{{padding:.5rem;border:1px solid #ddd}}a{{color:#076c76}}p{{overflow-wrap:anywhere}}@media(max-width:600px){{main{{padding:.8rem}}}}</style></head><body><main><h1>Earlier repaired assessment, 2026-10-04</h1><p><strong>Earlier completed protocol: nested_matched_holdout_v2.</strong> The <a href="../portfolio/index.html">current report</a> and <a href="../tuning_20261004/assessment_report.md">current technical assessment</a> use nested_inner_cv_v3, completed later on the same date. These earlier values remain separate.</p><p>The upper panels assess {assessment_count:,} development applicants in five matched test groups. The lower panels compare separate saved models on {comparison_count:,} reused historical applicants. Unlabeled Kaggle applications are excluded. Random groups do not establish future-cohort performance; the historical comparison is not an untouched lockbox.</p><figure><img src="corrected_evidence.png" alt="Earlier repaired five-group ranking and reliability, with separate saved-model actions and ranking"><figcaption>Earlier aggregate evidence. Final probabilities are the chosen probability view and can equal raw probabilities; this earlier history procedure chose sigmoid adjustment. Group SD is descriptive, not a confidence interval. Actions are simulations, only middle-band reviews incur cost, and high-band simulated declines contribute zero utility. <a href="corrected_evidence.png">Full-size chart</a>.</figcaption></figure><h2>Exact earlier aggregate results</h2><p>Average precision measures ranking, not accuracy. Brier/log loss measure probability error (lower is better). Utility uses illustrative units, not money or profit.</p><div class="table-wrap">{table}</div><h2>Presentation and scientific provenance</h2><p>This presentation refresh reads frozen anonymous CSVs and the existing evidence manifest; it does not load models or applicant records. The lower-right raw average-precision panel replaces the original score histogram. Original verification.json HTML/PNG hashes apply to the original figures, recoverable at Git commit 7bcdee0ae22bd715d6527e2884b37bdcdc897823; the empirical evidence and execution fingerprints remain unchanged. Refresh hashes are recorded separately during delivery.</p><p>{html.escape(NATIVE_LIMIT)}</p><details><summary>Technical identity</summary><p>Assessment {html.escape(manifest["assessment_run_id"])}; protocol {html.escape(manifest["protocol"])}. Machine keys remain unchanged in the source CSVs.</p></details></main></body></html>"""
    (destination / "corrected_evidence.html").write_text(document, encoding="utf-8")
    return {
        "purpose": "Presentation-only refresh; no empirical recomputation",
        "assessment_run_id": manifest["assessment_run_id"],
        "source_sha256": inputs,
        "renderer_sha256": file_sha256(Path(__file__)),
        "output_sha256": {
            name: file_sha256(destination / name)
            for name in ("corrected_evidence.png", "corrected_evidence.html")
        },
        "original_output_sha256": {
            name: verification["curated_artifact_sha256"][name]
            for name in ("corrected_evidence.png", "corrected_evidence.html")
        },
        "assessment_count": assessment_count,
        "historical_comparison_count": comparison_count,
        "baseline_git": "7bcdee0ae22bd715d6527e2884b37bdcdc897823",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--assessment-dir", required=True, type=Path)
    parser.add_argument("--negative-control-dir", required=True, type=Path)
    parser.add_argument("--v1-config", required=True, type=Path)
    parser.add_argument(
        "--destination", default=Path("reports/correctness_20261004"), type=Path
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.assessment_dir,
                args.negative_control_dir,
                args.destination,
                args.v1_config,
            )
        )
    )


if __name__ == "__main__":
    main()
