from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from src import portfolio_site


def curated_bundle(source: Path) -> Path:
    source.mkdir()
    for name in portfolio_site.FILES:
        (source / name).write_bytes(b"reviewed artifact\n")
    figures = []
    for chart in portfolio_site.CHARTS:
        desktop = portfolio_site.PNG_SIGNATURE + chart.encode()
        portrait = desktop + b"portrait"
        (source / f"{chart}.png").write_bytes(desktop)
        figures.append(
            f'<figure class="report-figure" aria-labelledby="{chart}-title">'
            f'<picture><source media="(max-width:600px)" '
            f'srcset="data:image/png;base64,{base64.b64encode(portrait).decode()}">'
            f'<img src="data:image/png;base64,{base64.b64encode(desktop).decode()}" '
            f'alt="{chart}"></picture><figcaption>Reviewed result</figcaption></figure>'
        )
    (source / "index.html").write_text(
        "<!doctype html><html><body>"
        + "".join(figures)
        + "".join(
            f"<p>{original}</p>" for original, _ in portfolio_site.DELIVERY_REPLACEMENTS
        )
        + "</body></html>",
        encoding="utf-8",
    )
    (source / "model_input_provenance.json").write_text(
        json.dumps({"assessment_run_id": "reviewed-assessment"}), encoding="utf-8"
    )
    refresh_provenance(source)
    return source


def refresh_provenance(source: Path) -> None:
    (source / "provenance.json").write_text(
        json.dumps(
            {
                "assessment_run_id": "reviewed-assessment",
                "outputs_sha256": {
                    name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                    for name in portfolio_site.FILES
                    if name != "provenance.json"
                },
            }
        ),
        encoding="utf-8",
    )


def test_hosted_build_preserves_source_charts_and_hashes_actual_assets(tmp_path):
    source = curated_bundle(tmp_path / "source")
    original = (source / "index.html").read_bytes()
    output = tmp_path / "site"
    manifest = portfolio_site.prepare(source, output)
    hosted = (output / "index.html").read_text(encoding="utf-8")
    expected = original.decode()
    for chart in portfolio_site.CHARTS:
        desktop = (source / f"{chart}.png").read_bytes()
        portrait = desktop + b"portrait"
        for image, filename in (
            (desktop, f"{chart}.png"),
            (portrait, f"{chart}.portrait.png"),
        ):
            expected = expected.replace(
                '"data:image/png;base64,' + base64.b64encode(image).decode() + '"',
                f'"{filename}"',
            )
            assert (output / filename).read_bytes() == image
    for original_copy, hosted_copy in portfolio_site.DELIVERY_REPLACEMENTS:
        expected = expected.replace(original_copy, hosted_copy)
        assert original_copy not in hosted
    assert hosted == expected
    assert (source / "index.html").read_bytes() == original
    assert (output / "offline.html").read_bytes() == original
    assert hosted.count('href="offline.html" download') == 2
    assert set(manifest["files_sha256"]) == set(portfolio_site.HOSTED_FILES)
    for filename, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    assert (
        manifest["files_sha256"]["index.html"] != hashlib.sha256(original).hexdigest()
    )
    assert portfolio_site.prepare(source, output) == manifest


@pytest.mark.parametrize("copy_index", [0, 1])
@pytest.mark.parametrize("corruption", ["missing", "duplicate"])
def test_rejects_stale_or_duplicate_delivery_instructions(
    tmp_path, copy_index, corruption
):
    source = curated_bundle(tmp_path / "source")
    document = (source / "index.html").read_text(encoding="utf-8")
    original, _ = portfolio_site.DELIVERY_REPLACEMENTS[copy_index]
    document = (
        document.replace(original, "Changed delivery copy")
        if corruption == "missing"
        else document + f"<p>{original}</p>"
    )
    (source / "index.html").write_text(document, encoding="utf-8")
    refresh_provenance(source)
    with pytest.raises(ValueError, match="delivery instructions"):
        portfolio_site.prepare(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()


@pytest.mark.parametrize("target", ["source", "child"])
def test_rejects_source_as_output_or_output_inside_source(tmp_path, target):
    source = curated_bundle(tmp_path / "source")
    output = source if target == "source" else source / "site"
    with pytest.raises(ValueError, match="separate"):
        portfolio_site.prepare(source, output)


def test_rejects_unreviewed_output_before_writing(tmp_path):
    source = curated_bundle(tmp_path / "source")
    output = tmp_path / "site"
    output.mkdir()
    (output / "unexpected.txt").write_text("retain me")
    with pytest.raises(ValueError, match="unreviewed files"):
        portfolio_site.prepare(source, output)
    assert (output / "unexpected.txt").read_text() == "retain me"
    assert not (output / "index.html").exists()


def test_rejects_tampered_reviewed_source(tmp_path):
    source = curated_bundle(tmp_path / "source")
    (source / "metrics.csv").write_text("tampered")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        portfolio_site.prepare(source, tmp_path / "site")


@pytest.mark.parametrize("corruption", ["desktop", "portrait", "missing", "extra"])
def test_rejects_changed_responsive_chart_contract(tmp_path, corruption):
    source = curated_bundle(tmp_path / "source")
    document = (source / "index.html").read_text(encoding="utf-8")
    chart = portfolio_site.CHARTS[0]
    if corruption == "desktop":
        (source / f"{chart}.png").write_bytes(
            portfolio_site.PNG_SIGNATURE + b"different"
        )
        message = "differs from reviewed PNG"
    elif corruption == "portrait":
        document = document.replace(
            base64.b64encode(
                portfolio_site.PNG_SIGNATURE + chart.encode() + b"portrait"
            ).decode(),
            "!!!invalid!!!",
        )
        message = "invalid PNG encoding"
    elif corruption == "missing":
        document = document.replace(
            f'aria-labelledby="{chart}-title"', 'aria-labelledby="other-title"'
        )
        message = "unexpected chart identity"
    else:
        document += '<img src="data:image/png;base64,aGVsbG8=">'
        message = "responsive chart layout"
    (source / "index.html").write_text(document, encoding="utf-8")
    refresh_provenance(source)
    with pytest.raises(ValueError, match=message):
        portfolio_site.prepare(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_html_budget_is_checked_before_any_output_write(tmp_path, monkeypatch):
    source = curated_bundle(tmp_path / "source")
    monkeypatch.setattr(portfolio_site, "HTML_MAX_BYTES", 100)
    with pytest.raises(ValueError, match="3 MiB"):
        portfolio_site.prepare(source, tmp_path / "site")
    assert not (tmp_path / "site").exists()
