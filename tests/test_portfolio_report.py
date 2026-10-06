from __future__ import annotations

import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from src.portfolio_extensions import INPUT_FILES as DRIVER_FILES
from src.portfolio_extensions import INPUT_SOURCE, load_inputs
from src.portfolio_report import INPUT_FILES, SOURCE, load_evidence, mean, render
from src.portfolio_site import FILES as SITE_FILES
from src.portfolio_site import prepare


def copy_evidence(destination: Path) -> Path:
    destination.mkdir()
    for name in INPUT_FILES:
        shutil.copy2(SOURCE / name, destination / name)
    return destination


def test_committed_aggregate_metrics_keep_their_scientific_mean() -> None:
    evidence = load_evidence(SOURCE)
    assert evidence["applicant_count"] == 261384
    assert mean(evidence, "history_selected", "pr_auc") == pytest.approx(
        0.2658621001843764, abs=1e-6
    )
    assert mean(evidence, "application_only", "pr_auc") == pytest.approx(
        0.23102126216808827
    )
    assert mean(evidence, "history_selected", "brier_score") < mean(
        evidence, "training_prevalence", "brier_score"
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "summary",
        "population",
        "applicant_rows",
        "reliability_population",
        "reliability_identity",
    ],
)
def test_report_rejects_inconsistent_or_applicant_level_inputs(
    tmp_path: Path, corruption: str
) -> None:
    source = copy_evidence(tmp_path / "source")
    if corruption == "summary":
        frame = pd.read_csv(source / "summary.csv")
        frame.loc[0, "fold_mean"] += 0.01
        frame.to_csv(source / "summary.csv", index=False)
        message = "Summary disagrees"
    elif corruption.startswith("reliability"):
        frame = pd.read_csv(source / "reliability_bins.csv")
        if corruption == "reliability_population":
            frame.loc[
                (frame.workflow == "history_selected")
                & (frame.score_kind == "calibrated"),
                "applicant_count",
            ] += 1
            message = "Reliability-bin counts"
        else:
            frame.loc[0, "run_id"] = "different-assessment"
            message = "different assessment"
        frame.to_csv(source / "reliability_bins.csv", index=False)
    else:
        frame = pd.read_csv(source / "fold_metrics.csv")
        if corruption == "population":
            frame.loc[
                (frame.workflow == "history_selected")
                & (frame.score_kind == "calibrated")
                & (frame.metric_name == "pr_auc"),
                "applicant_count",
            ] += 1
            message = "same applicant groups"
        else:
            frame["applicant_id"] = 123
            message = "anonymous aggregates"
        frame.to_csv(source / "fold_metrics.csv", index=False)
    with pytest.raises(ValueError, match=message):
        load_evidence(source)


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        self.references.extend(
            value for key, value in attrs if key in {"src", "srcset", "href"}
        )


def test_render_is_offline_preserves_keys_and_is_deterministic(tmp_path: Path) -> None:
    destination = tmp_path / "portfolio"
    provenance = render(SOURCE, destination)
    source_metrics = pd.read_csv(SOURCE / "summary.csv")
    rendered_metrics = pd.read_csv(destination / "metrics.csv")
    pd.testing.assert_frame_equal(source_metrics, rendered_metrics)
    glossary = pd.read_csv(destination / "metric_dictionary.csv").set_index(
        "machine_key"
    )
    assert {"display_label", "definition", "direction", "units"} <= set(
        glossary.columns
    )
    assert glossary.loc["pr_auc", "direction"] == "Higher is better"
    assert glossary.loc["brier_score", "direction"] == "Lower is better"
    assert (
        "not dollars"
        in glossary.loc["balanced_utility_per_applicant", "definition"].lower()
    )
    assert (
        "middle-band manual review"
        in glossary.loc["recall_at_manual_review_capacity", "definition"]
    )

    parser = Resources()
    parser.feed((destination / "index.html").read_text(encoding="utf-8"))
    # Both desktop and portrait phone charts must remain embedded offline.
    assert len([v for v in parser.references if v.startswith("data:image/png")]) == 16
    assert all(
        v.startswith(("data:", "#", "https://github.com/"))
        or (destination / v).is_file()
        for v in parser.references
    )
    for name in [
        "model_comparison",
        "risk_capture",
        "probability_reliability",
        "search_comparison",
        "class_weighting",
        "installment_segments",
        "utility_sensitivity",
        "model_inputs",
    ]:
        assert (destination / f"{name}.svg").exists()
    report = (destination / "case_study.md").read_text(encoding="utf-8")
    methods = (destination / "model_input_methods.md").read_text(encoding="utf-8")
    for document in (report, methods):
        for reference in re.findall(r"\]\(([^)]+)\)", document):
            assert not reference.startswith("../"), reference
            assert (
                reference.startswith("https://")
                or (destination / reference.split("#")[0]).is_file()
            ), reference
    assert "in provenance.json." not in methods
    assert "[model_input_provenance.json](model_input_provenance.json)" in methods
    assert (destination / "model_input_provenance.json").read_bytes() == (
        INPUT_SOURCE / "provenance.json"
    ).read_bytes()
    assert "in provenance.json." in (INPUT_SOURCE / "methods.md").read_text(
        encoding="utf-8"
    )
    assert "not accuracy" in report
    assert "middle manual-review band" in report
    assert "not dollars or profit" in report
    assert "not an untouched final test" in report
    assert "no probability-adjustment transform was selected" in report
    assert "Removing class weighting is the dominant explanation" in report
    assert "no installment obligations does not imply" in report
    assert "27 assumption combinations" in report
    assert "shares of predictive performance" in report
    assert "model_input_methods.md" in report
    assert "1,000 uniformly sampled assessment applicants per fold" in (
        destination / "model_input_methods.md"
    ).read_text(encoding="utf-8")
    assert (
        "only class weighting changes"
        in (destination / "index.html").read_text(encoding="utf-8").lower()
    )
    assert render(SOURCE, destination)["outputs_sha256"] == provenance["outputs_sha256"]
    assert (
        json.loads((destination / "provenance.json").read_text())["assessment_run_id"]
        == provenance["assessment_run_id"]
    )
    site = tmp_path / "site"
    result = prepare(destination, site)
    assert set(result["files_sha256"]) == set(SITE_FILES)
    assert {p.name for p in site.iterdir()} == set(SITE_FILES) | {
        ".nojekyll",
        "site_manifest.json",
    }
    (destination / "private_applicant_rows.csv").write_text(
        "applicant_id,score\n123,.5\n"
    )
    with pytest.raises(ValueError, match="unreviewed files"):
        prepare(destination, tmp_path / "rejected-site")
    (destination / "private_applicant_rows.csv").unlink()
    (destination / "metrics.csv").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        prepare(destination, tmp_path / "tampered-site")


@pytest.mark.parametrize(
    "corruption", ["segment_population", "missing_assumption", "utility_identity"]
)
def test_sensitivity_charts_reject_mixed_or_incomplete_evidence(tmp_path, corruption):
    source = copy_evidence(tmp_path / "source")
    if corruption == "segment_population":
        filename = "history_segment_metrics.csv"
        frame = pd.read_csv(source / filename)
        frame.loc[
            (frame.workflow == "history_selected") & (frame.score_kind == "calibrated"),
            "applicant_count",
        ] += 1
        message = "identical populations"
    else:
        filename = "utility_sensitivity.csv"
        frame = pd.read_csv(source / filename)
        if corruption == "missing_assumption":
            frame = frame.iloc[1:]
            message = "all 27 assumptions"
        else:
            frame.loc[0, "run_id"] = "different-assessment"
            message = "identity mismatch"
    frame.to_csv(source / filename, index=False)
    with pytest.raises(ValueError, match=message):
        load_evidence(source)


@pytest.mark.parametrize(
    "corruption", ["identity", "applicant_rows", "summary", "protected_field"]
)
def test_current_model_input_evidence_rejects_mismatch_and_leakage(
    tmp_path, corruption
):
    import hashlib

    source = tmp_path / "inputs"
    source.mkdir()
    for filename in DRIVER_FILES:
        shutil.copy2(INPUT_SOURCE / filename, source / filename)
    provenance = json.loads((source / "provenance.json").read_text())
    if corruption == "identity":
        provenance["assessment_run_id"] = "different-assessment"
        message = "identity mismatch"
    else:
        filename = (
            "input_dictionary.csv"
            if corruption == "protected_field"
            else "importance_summary.csv"
        )
        frame = pd.read_csv(source / filename)
        if corruption == "applicant_rows":
            frame["SK_ID_CURR"] = 123
            message = "anonymous aggregate columns"
        elif corruption == "protected_field":
            frame.loc[0, "raw_field"] = "TARGET"
            message = "dictionary differs"
        else:
            frame.loc[0, "fold_mean"] += 0.01
            message = "summary disagrees"
        frame.to_csv(source / filename, index=False)
        provenance["outputs_sha256"][filename] = hashlib.sha256(
            (source / filename).read_bytes()
        ).hexdigest()
    (source / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match=message):
        load_inputs(source, load_evidence(SOURCE)["provenance"])
