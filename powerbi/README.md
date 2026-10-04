# Power BI Dashboard

This folder contains curated Power BI snapshots for recruiters and hiring managers reviewing the local credit-risk portfolio. Read [current evidence status](../docs/validation/VALIDATION_PLAN.md#current-evidence-status) alongside the visuals; saved reports are not an independent model-validation sign-off.

## Data Source

The frozen v1 dashboard should load only the CSV bundle in:

```text
reports/dashboard_data/
```

The saved report can be inspected without rebuilding. To deliberately regenerate the local v1 bundle from downloaded Kaggle data:

```bash
make pipeline-v1
```

That command overwrites generated v1 artifacts from `configs/v1.yaml` and exports dashboard CSVs. If upstream v1 artifacts exist, `make dashboard-data` exports without retraining and recomputes segment diagnostics. Neither command refreshes PBIX visuals or committed screenshots. On Windows where `python3` is unavailable, append `PYTHON=python`.

`credit_risk_scores.csv` currently has 16 columns. If Power Query generated a fixed `Columns=13` CSV import step from an older refresh, update it to 16 columns or remove the fixed column-count argument so Power BI reads `top_reason_1`, `top_reason_2`, and `top_reason_3`.

`model_metrics_summary.csv` keeps baseline and LightGBM rows. Filter a metric visual to exactly one `model_version` and one `split` before aggregation. The legacy measure below includes lower-is-better Brier handling, but using it across models/splits would mix the best values into a fictitious model. Once the filters identify one row per metric, MIN and MAX return that same row:

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

The v1 bundle remains raw and uncalibrated, with the selected model displayed as `lightgbm_credit_risk_v1`. The post-v1 bundle uses the display alias `lightgbm_credit_risk_post_v1` and recomputes selected-model probability-quality views using the calibration artifact: `model_metrics_summary`, `model_calibration_bins`, and `segment_performance_summary`. The display alias is not an exact fitted-run identity. Raw runtime reports can differ from these calibrated views.

Lift, threshold scenarios, risk bands, and actions use the raw rank score. The historical selected sigmoid transform preserves ordering; do not generalize that result to every calibrator or to policy-capacity guarantees. `score` and `raw_risk_score` carry the policy score; `calibrated_risk_score` and `calibration_method` describe the separate probability-quality view.

The post-v1 comparison dashboard uses the same CSV filenames and schemas in:

```text
reports/dashboard_data_post_v1/
```

Refresh that bundle with:

```bash
make pipeline-post-v1
```

If upstream post-v1 artifacts exist, `make dashboard-data-post-v1` exports without retraining, recomputing calibrated probability-quality metrics and segment diagnostics. It does not update curated historical metrics or screenshots. Exact historical numeric reproduction is not certified with the current unlocked dependencies.

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

The two existing screenshots show the historical **post-v1** report, not a separate v1 preview. For a future screenshot refresh, use an identified labeled evaluation run, a single model/split filter, and the `balanced` display scenario. Record the input bundle before replacing the snapshots and verify displayed values against that bundle.

Tests check artifact presence and report page structure, not visual or numeric reconciliation. A current local export can differ from these saved snapshots; do not claim a live refresh was verified from the file tests alone.

## Dashboard Framing

The dashboard is a portfolio decision-support simulation, not a production underwriting system. Segment diagnostics are diagnostic-only, excluded from model training, and not a fairness certification. Reason-code-style fields are interpretability artifacts, not adverse-action notices.

Legacy visual labels require these qualifications:

- "Held-out test" is the saved within-run split label; historical stability runs reused those applicants in fitting.
- PR-AUC is average precision. Recall at review capacity measures highest-score capture, not capture in the middle manual-review band.
- The 10% setting is a scenario reference, not an enforced queue cap. The high band is labeled high-priority review but is not charged review cost by the current utility formula.
- Currency-formatted EV cards show illustrative utility units, not dollars of measured profit. Calibration fitting and selection share validation data.

These labels remain embedded in the saved PBIX/screenshots. Updating this guide does not certify that their visuals have been repaired or refreshed.
