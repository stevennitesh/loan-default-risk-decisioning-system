"""Prepare a small reviewed static Pages bundle using only standard Python."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
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
    "featured.jpg",
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
PORTRAIT_FILES = tuple(f"{chart}.portrait.png" for chart in CHARTS)
HOSTED_FILES = (*FILES, *PORTRAIT_FILES, "offline.html")
HTML_MAX_BYTES = 3 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DELIVERY_REPLACEMENTS = (
    (
        (
            "Downloads work online or alongside this HTML in the presentation folder. "
            "The charts and story remain readable when this HTML is opened alone offline."
        ),
        (
            "Charts and downloads use files alongside this HTML. "
            '<a href="offline.html" download>Download the offline report</a> '
            "for a single-file copy with embedded charts."
        ),
    ),
    (
        "This file is self-contained: charts, styles and text require no network connection.",
        (
            "This hosted page loads charts from adjacent PNG files. "
            '<a href="offline.html" download>Download the self-contained offline report</a> '
            "for reading without a network connection."
        ),
    ),
)


def hosted_html(source: Path) -> tuple[bytes, dict[str, bytes]]:
    """Externalize reviewed chart bytes without regenerating or altering charts."""
    document = (source / "index.html").read_text(encoding="utf-8")
    portraits = {}
    seen = set()

    def replace_figure(match: re.Match) -> str:
        chart, figure = match.group("chart"), match.group(0)
        if chart not in CHARTS or chart in seen:
            raise ValueError("Static HTML contains unexpected chart identity")
        seen.add(chart)
        for attribute, filename in (
            ("srcset", f"{chart}.portrait.png"),
            ("src", f"{chart}.png"),
        ):
            references = list(
                re.finditer(rf'\b{attribute}="(data:image/png;base64,[^"]+)"', figure)
            )
            if len(references) != 1:
                raise ValueError(
                    f"Static chart has missing or unexpected images: {chart}"
                )
            reference = references[0]
            try:
                image = base64.b64decode(
                    reference.group(1).split(",", 1)[1], validate=True
                )
            except (ValueError, binascii.Error) as error:
                raise ValueError(
                    f"Static chart has invalid PNG encoding: {chart}"
                ) from error
            if not image.startswith(PNG_SIGNATURE):
                raise ValueError(f"Static chart is not a PNG: {chart}")
            if attribute == "src":
                if image != (source / filename).read_bytes():
                    raise ValueError(
                        f"Static embedded chart differs from reviewed PNG: {chart}"
                    )
            else:
                portraits[filename] = image
            figure = (
                figure[: reference.start(1)] + filename + figure[reference.end(1) :]
            )
        return figure

    document = re.sub(
        r'<figure\b[^>]*\baria-labelledby="(?P<chart>[^"<>]+)-title"[^>]*>.*?</figure>',
        replace_figure,
        document,
        flags=re.DOTALL,
    )
    if seen != set(CHARTS) or "data:image/" in document:
        raise ValueError(
            "Static HTML differs from the reviewed responsive chart layout"
        )
    for original, hosted in DELIVERY_REPLACEMENTS:
        if document.count(original) != 1:
            raise ValueError(
                "Static HTML differs from the reviewed delivery instructions"
            )
        document = document.replace(original, hosted)
    encoded = document.encode("utf-8")
    if len(encoded) >= HTML_MAX_BYTES:
        raise ValueError("Hosted HTML exceeds the 3 MiB link-preview budget")
    return encoded, portraits


def prepare(source: Path, output: Path) -> dict:
    """Verify the curated bundle, then build a small HTML page and exact image assets."""
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
    html, portraits = hosted_html(source)
    allowed_output = set(HOSTED_FILES) | {".nojekyll", "site_manifest.json"}
    if output.exists() and any(
        p.name not in allowed_output or not p.is_file() or p.is_symlink()
        for p in output.iterdir()
    ):
        raise ValueError("Static output contains unreviewed files")
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(source / "index.html", output / "offline.html")
    (output / "index.html").write_bytes(html)
    for name, image in portraits.items():
        (output / name).write_bytes(image)
    (output / ".nojekyll").write_text("", encoding="utf-8")
    manifest = {
        "purpose": "Reviewed static portfolio only; no data, models, source archives or memberships",
        "assessment_run_id": provenance["assessment_run_id"],
        "files_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in HOSTED_FILES
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
