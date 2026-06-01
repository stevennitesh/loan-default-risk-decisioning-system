# Power BI Dashboard

This folder contains the curated Power BI portfolio dashboard for the loan default risk decisioning project.

## Data Source

The frozen v1 dashboard should load only the CSV bundle in:

```text
reports/dashboard_data/
```

Refresh the bundle before opening or refreshing the report:

```bash
make pipeline-v1
```

That command rebuilds the v1 artifacts from `configs/v1.yaml` and exports the v1 dashboard CSVs. If the upstream v1 artifacts already exist, `make dashboard-data` refreshes only the v1 CSV bundle.

`credit_risk_scores.csv` currently has 16 columns. If Power Query generated a fixed `Columns=13` CSV import step from an older refresh, update it to 16 columns or remove the fixed column-count argument so Power BI reads `top_reason_1`, `top_reason_2`, and `top_reason_3`.

`model_metrics_summary.csv` keeps the baseline and LightGBM rows. The Model Metrics visual must not use plain `Max of metric_value` for every row because Brier score is lower-is-better. Use a measure like this in the matrix values:

```DAX
Metric Display Value =
VAR MetricName = SELECTEDVALUE(model_metrics_summary[metric_name])
RETURN
    IF(
        MetricName = "brier_score",
        MIN(model_metrics_summary[metric_value]),
        MAX(model_metrics_summary[metric_value])
    )
```

The v1 bundle remains raw and uncalibrated, with the selected model identified as `lightgbm_credit_risk_v1`. The post-v1 bundle identifies the improved selected model as `lightgbm_credit_risk_post_v1` and applies the selected calibration artifact to probability-quality views: `model_metrics_summary`, `model_calibration_bins`, and `segment_performance_summary`. Rank-policy views such as lift, threshold scenarios, risk bands, and recommended actions remain based on the raw rank score because the calibrated sigmoid layer is monotonic and does not change applicant ordering.

The post-v1 comparison dashboard uses the same CSV filenames and schemas in:

```text
reports/dashboard_data_post_v1/
```

Refresh that bundle with:

```bash
make pipeline-post-v1
```

If the upstream post-v1 artifacts already exist, `make dashboard-data-post-v1` refreshes only the post-v1 CSV bundle.

The post-v1 report is maintained as `powerbi/dashboard_post_v1.pbix`. Its CSV folder/data-source path should remain `reports/dashboard_data_post_v1/`, while table names, columns, pages, visuals, and slicers stay aligned with the v1 report so the two dashboards remain directly comparable.

## Required Reports

The curated report artifacts are:

```text
powerbi/dashboard.pbix
powerbi/dashboard_post_v1.pbix
```

They should contain two pages:

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
