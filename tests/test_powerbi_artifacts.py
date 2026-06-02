from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
POWERBI_DIR = ROOT / "powerbi"
SCREENSHOT_DIR = POWERBI_DIR / "screenshots"

MIN_PBIX_SIZE_BYTES = 100_000
MIN_SCREENSHOT_SIZE_BYTES = 25_000
MIN_SCREENSHOT_WIDTH = 1200
MIN_SCREENSHOT_HEIGHT = 700


def test_powerbi_dashboard_pbix_files_exist_and_are_non_empty() -> None:
    dashboard_paths = [
        POWERBI_DIR / "dashboard.pbix",
        POWERBI_DIR / "dashboard_post_v1.pbix",
    ]

    for dashboard_path in dashboard_paths:
        assert dashboard_path.exists(), f"Expected curated Power BI report at {dashboard_path}"
        assert dashboard_path.stat().st_size >= MIN_PBIX_SIZE_BYTES


def test_powerbi_dashboard_screenshots_are_readable_pngs() -> None:
    for screenshot_name in [
        "decisioning_overview.png",
        "model_validation_appendix.png",
    ]:
        screenshot_path = SCREENSHOT_DIR / screenshot_name

        assert screenshot_path.exists(), f"Missing Power BI screenshot: {screenshot_path}"
        assert screenshot_path.stat().st_size >= MIN_SCREENSHOT_SIZE_BYTES
        with Image.open(screenshot_path) as image:
            assert image.format == "PNG"
            assert image.width >= MIN_SCREENSHOT_WIDTH
            assert image.height >= MIN_SCREENSHOT_HEIGHT


def test_powerbi_readme_documents_refresh_pages_and_limitations() -> None:
    readme_path = POWERBI_DIR / "README.md"

    assert readme_path.exists(), "Expected Power BI authoring notes at powerbi/README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    assert "make dashboard-data" in readme_text
    assert "make dashboard-data-post-v1" in readme_text
    assert "reports/dashboard_data_post_v1/" in readme_text
    assert "lightgbm_credit_risk_post_v1" in readme_text
    assert "Metric Display Value" in readme_text
    assert "MIN(model_metrics_summary[metric_value])" in readme_text
    assert "Decisioning Overview" in readme_text
    assert "Model Validation Appendix" in readme_text
    assert "diagnostic-only" in readme_text
    assert "not a fairness certification" in readme_text
