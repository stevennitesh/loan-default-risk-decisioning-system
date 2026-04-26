# Power BI Dashboard

This folder contains the curated Power BI portfolio dashboard for the loan default risk decisioning project.

## Data Source

Power BI should load only the CSV bundle in:

```text
reports/dashboard_data/
```

Refresh the bundle before opening or refreshing the report:

```bash
make dashboard-data
```

That command expects the pipeline outputs from `make evaluate`, `make score`, and `python -m src.explain --config configs/base.yaml` to already exist.

## Required Report

The curated report artifact is:

```text
powerbi/dashboard.pbix
```

It should contain two pages:

- `Decisioning Overview`
- `Model Validation Appendix`

## Required Screenshots

Export page screenshots to:

```text
powerbi/screenshots/decisioning_overview.png
powerbi/screenshots/model_validation_appendix.png
```

The overview screenshot should be readable without interacting with slicers. Use the held-out labeled `test` split and keep the `balanced` threshold scenario visually highlighted.

## Dashboard Framing

The dashboard is a portfolio decision-support simulation, not a production underwriting system. Segment diagnostics are diagnostic-only, excluded from model training, and not a fairness certification. Reason-code-style fields are interpretability artifacts, not adverse-action notices.
