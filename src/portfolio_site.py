"""Prepare a small reviewed static Pages bundle using only standard Python."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/portfolio"
CHARTS = (
    "model_comparison",
    "risk_capture",
    "probability_reliability",
    "search_comparison",
    "class_weighting",
    "installment_segments",
    "utility_sensitivity",
    "model_inputs",
)
FILES = (
    "index.html",
    "case_study.md",
    "metrics.csv",
    "metric_dictionary.csv",
    "provenance.json",
    "model_input_dictionary.csv",
    "model_input_summary.csv",
    "model_input_fold_importance.csv",
    "model_input_provenance.json",
    "model_input_methods.md",
    *(f"{chart}.{extension}" for chart in CHARTS for extension in ("png", "svg")),
)


def prepare(source: Path, output: Path) -> dict:
    """Reject unreviewed files; copy exact final allowlist and verify its hashes."""
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    declared = set(FILES) - {"provenance.json"}
    if set(provenance["outputs_sha256"]) != declared:
        raise ValueError(
            "Presentation provenance differs from the reviewed static allowlist"
        )
    actual = {p.name for p in source.iterdir()}
    if actual != set(FILES) or any(
        not (source / name).is_file() or (source / name).is_symlink() for name in FILES
    ):
        raise ValueError("Static source contains missing or unreviewed files")
    for name, expected in provenance["outputs_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Static presentation fingerprint mismatch: {name}")
    inputs = json.loads(
        (source / "model_input_provenance.json").read_text(encoding="utf-8")
    )
    if inputs["assessment_run_id"] != provenance["assessment_run_id"]:
        raise ValueError(
            "Static model-input and presentation assessment identity mismatch"
        )
    if (
        output.resolve() == source.resolve()
        or source.resolve() in output.resolve().parents
    ):
        raise ValueError("Static output must be separate from the curated source")
    allowed_output = set(FILES) | {".nojekyll", "site_manifest.json"}
    if output.exists() and any(
        p.name not in allowed_output or not p.is_file() or p.is_symlink()
        for p in output.iterdir()
    ):
        raise ValueError("Static output contains unreviewed files")
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, output / name)
    (output / ".nojekyll").write_text("", encoding="utf-8")
    manifest = {
        "purpose": "Reviewed static portfolio only; no data, models, source archives or memberships",
        "assessment_run_id": provenance["assessment_run_id"],
        "files_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in FILES
        },
    }
    (output / "site_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / ".tmp/portfolio-site")
    args = parser.parse_args()
    result = prepare(args.source, args.output)
    print(f"Prepared {len(result['files_sha256'])} reviewed files at {args.output}")


if __name__ == "__main__":
    main()
