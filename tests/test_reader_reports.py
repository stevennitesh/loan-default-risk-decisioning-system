"""Evidence identity and preservation checks for presentation-only report paths."""

import json
import shutil
from pathlib import Path

import pytest

from src.correctness_summary import refresh_presentation
from src.evidence import file_sha256
from src.tuning_summary import weighting_followup_note

ROOT = Path(__file__).resolve().parents[1]


def copy_bundle(source, destination):
    shutil.copytree(ROOT / source, destination)
    return destination


def test_followup_requires_explicit_matching_completed_evidence(tmp_path):
    bundle = copy_bundle("reports/class_weighting_20261004", tmp_path / "weighting")
    provenance_path = bundle / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    run_id = provenance["source_runs"]["current"]
    destination = tmp_path / "assessment"
    note = weighting_followup_note(run_id, bundle, destination)
    assert "../weighting/assessment_report.md" in note
    assert "No separately identified" in weighting_followup_note(
        run_id, None, destination
    )
    with pytest.raises(ValueError, match="match this assessment"):
        weighting_followup_note("unrelated-new-assessment", bundle, destination)
    provenance["status"] = "running"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        weighting_followup_note(run_id, bundle, destination)


def test_followup_rejects_changed_frozen_aggregate(tmp_path):
    bundle = copy_bundle("reports/class_weighting_20261004", tmp_path / "weighting")
    provenance = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    with (bundle / "summary.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="aggregate changed"):
        weighting_followup_note(provenance["source_runs"]["current"], bundle, tmp_path)


def test_earlier_presentation_reads_only_frozen_anonymous_evidence(tmp_path):
    bundle = copy_bundle("reports/correctness_20261004", tmp_path / "earlier")
    frozen = {
        path.name: file_sha256(path)
        for path in bundle.iterdir()
        if path.suffix in {".csv", ".json"}
    }
    proof = refresh_presentation(bundle)
    assert proof["assessment_count"] == 261384
    assert proof["historical_comparison_count"] == 46127
    assert proof["assessment_run_id"] == "f6b58bf8335645b3a1ece2ca3502e173"
    for name, expected in frozen.items():
        assert file_sha256(bundle / name) == expected
    for name, expected in proof["output_sha256"].items():
        assert file_sha256(bundle / name) == expected
    with (bundle / "fold_summary.csv").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="Frozen presentation input changed"):
        refresh_presentation(bundle)
