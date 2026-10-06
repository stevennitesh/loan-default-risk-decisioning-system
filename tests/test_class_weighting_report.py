from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from src.class_weighting_report import FILES, load_evidence
from src.evidence import file_sha256
from src.portfolio_report import WEIGHTING_SOURCE


def test_curated_diagnostic_retains_all_pairs_and_reference_checks():
    evidence = load_evidence(WEIGHTING_SOURCE)
    assert evidence["probability_improved_all_pairs"]
    assert evidence["provenance"]["applicant_count"] == 261384
    assert len(evidence["provenance"]["reference_checks"]) == 10
    with pytest.raises(ValueError, match="different current assessment"):
        load_evidence(WEIGHTING_SOURCE, "different-run")


@pytest.mark.parametrize(
    "corruption", ["summary", "paired", "reference", "applicant_rows"]
)
def test_weighting_story_rejects_bad_aggregates_even_with_updated_hashes(
    tmp_path, corruption
):
    source = tmp_path / "evidence"
    source.mkdir()
    for name in FILES:
        shutil.copy2(WEIGHTING_SOURCE / name, source / name)
    manifest = json.loads((source / "provenance.json").read_text())
    if corruption == "reference":
        manifest["reference_checks"][0]["max_absolute_prediction_difference"] = 0.01
        message = "reproduction is incomplete"
    else:
        filename = {
            "summary": "summary.csv",
            "paired": "paired_differences.csv",
            "applicant_rows": "fold_metrics.csv",
        }[corruption]
        frame = pd.read_csv(source / filename)
        if corruption == "summary":
            frame.loc[0, "brier_score_mean"] += 0.01
            message = "summary disagrees"
        elif corruption == "paired":
            frame.loc[0, "brier_score_unweighted_minus_weighted"] += 0.01
            message = "paired effects disagree"
        else:
            frame["applicant_id"] = 123
            message = "anonymous aggregates"
        frame.to_csv(source / filename, index=False)
        manifest["aggregate_sha256"][filename] = file_sha256(source / filename)
    (source / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_evidence(source)
