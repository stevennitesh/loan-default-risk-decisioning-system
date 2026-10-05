# Power BI Dashboard

## Current results and historical Power BI assets

For current results, open [the standalone offline report](../reports/portfolio/index.html) or [case study](../reports/portfolio/case_study.md). The PNG/SVG charts are current static presentation artifacts, separately generated from anonymous assessment aggregates. They are not screenshots of a refreshed Power BI report.

Both PBIX files and existing screenshots below are unrefreshed historical demonstrations and remain byte-identical. Desktop/DAX/model refresh is unavailable on this host. Future Desktop work should first choose one identified export bundle and confirm model, labeled-versus-unlabeled population, split, scenario and score view on every visual; inspect DAX, relationships and import widths before reconciling numbers. Keep fold-trained assessment separate from the saved-model export bundle.

Use these display names while preserving machine field names:

| Machine term | Display name / meaning |
|---|---|
| `pr_auc` | Average precision (ranking; higher is better, not accuracy) |
| `recall_at_manual_review_capacity` | Repayment-difficulty cases captured in highest-risk 10%; separate from middle-band review |
| `calibrated_risk_score` | Final probability after method selection; unchanged raw probabilities if `uncalibrated` was selected |
| `expected_value_per_applicant` | Illustrative utility per applicant (units; no dollar symbol) |
| `holdout_test` / `test` | Historical comparison applicants |
| `application_test` | Unlabeled scoring demonstration; no outcome evaluation |
| `approve` / `manual_review` / `simulated_decline` | Simulated approval / Manual review / Simulated decline |

For the later native refresh, apply these labels, remove currency/profit formatting, verify middle-band-only review cost, then capture new screenshots with the bundle identity. Opening the saved file or passing artifact tests does not verify a refresh.


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

That command overwrites generated v1 artifacts from `configs/v1.yaml` and exports dashboard CSVs. Current code uses corrected repayment features and row bagging, so regenerated numbers differ from the frozen historical v1 snapshot. If compatible upstream artifacts exist, `make dashboard-data` exports without retraining and recomputes segment diagnostics. Neither command refreshes PBIX visuals or committed screenshots. On Windows where `python3` is unavailable, append `PYTHON=python`.

`credit_risk_scores.csv` currently has 16 columns. If Power Query generated a fixed `Columns=13` CSV import step from an older refresh, update it to 16 columns or remove the fixed column-count argument so Power BI reads `top_reason_1`, `top_reason_2`, and `top_reason_3`.

`model_metrics_summary.csv` keeps baseline and LightGBM rows. Filter a metric visual to exactly one `model_version` and one `split` before aggregation. The legacy measure below includes lower-is-better Brier handling, but using it across models/splits would mix the best values into a fictitious model. Once the filters identify one row per metric, MIN and MAX return that same row:

```DAX
Metric Display Value =
VAR MetricName = SELECTEDVALUE(model_metrics_summary[metric_name])
RETURN
    IF(
        MetricName IN { "brier_score", "log_loss" },
        MIN(model_metrics_summary[metric_value]),
        MAX(model_metrics_summary[metric_value])
    )
```

The v1 bundle remains raw and uncalibrated, with the selected model displayed as `lightgbm_credit_risk_v1`. The post-v1 bundle uses the display alias `lightgbm_credit_risk_post_v1` and recomputes selected-model probability-quality views using the calibration artifact: `model_metrics_summary`, `model_calibration_bins`, and `segment_performance_summary`. The display alias is not an exact fitted-run identity. Raw runtime reports can differ from these calibrated views.

Lift, threshold scenarios, risk bands, and actions use the raw rank score. The historical selected sigmoid transform preserves ordering; do not generalize that result to every calibrator or to policy-capacity guarantees. `score` and `raw_risk_score` carry the policy score; `calibrated_risk_score` and `calibration_method` describe the separate probability-quality view.

Current exports use `simulated_decline` for high-risk recommended actions, `manual_review` for the middle band, and `approve` for the low band. Update action filters/labels before refreshing a saved report: historical `high_priority_review` labels described a different action from the utility formula. The high band now explicitly represents no issued loan and zero modeled value/cost; only the middle band incurs review cost. These are simulated dispositions, with no estimated reviewer effectiveness.

The post-v1 comparison dashboard uses the same CSV filenames and schemas in:

```text
reports/dashboard_data_post_v1/
```

Refresh that bundle with:

```bash
make pipeline-post-v1
```

After the 2026-10-04 corrections, first regenerate with the full scoped pipeline; old artifacts lack the reserved calibration role and feature-build identity. For compatible upstream artifacts, `make dashboard-data-post-v1` exports without retraining, recomputing calibrated probability-quality metrics and segment diagnostics. Export rejects mismatched fitted runs and replaced calibrators until dependent outputs are regenerated. It does not update curated historical metrics or screenshots. The named correctness run uses hash-locked dependencies and verified CSV reconciliation; exact historical numeric reproduction and native PBIX refresh remain uncertified.

The post-v1 report is maintained as `powerbi/dashboard_post_v1.pbix`. Its CSV folder/data-source path should remain `reports/dashboard_data_post_v1/`, while table names, columns, pages, visuals, and slicers stay aligned with the v1 report so the two dashboards remain directly comparable.

## Nested Assessment Evidence

Nested assessment is a separate procedure-evaluation artifact. Its fold metrics
and out-of-fold predictions are not substituted into the existing dashboard CSV
bundle. The saved dashboard model and its raw/calibrated views describe a different
fit. Read the [assessment explanation](../docs/validation/ASSESSMENT_METHODOLOGY.md)
before presenting the two evidence types together; a synthetic assessment run
does not refresh or validate these historical visuals.

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
- Currency-formatted EV cards show illustrative utility units, not dollars of measured profit. The displayed historical calibration fit/selection used shared validation data; current code reserves separate fitting rows.

These labels remain embedded in the saved PBIX/screenshots. Updating this guide does not certify that their visuals have been repaired or refreshed.

## 2026-10-04 native inspection and corrected static evidence

The completed isolated bundles are `reports/generated/correctness_20261004_post_v1_r2/dashboard/` and `reports/generated/correctness_20261004_v1_r2/dashboard/`. Original default folders retain their previous artifacts; the new aggregate report does not certify those defaults as regenerated. Use the named bundle and its manifest identities for a future native refresh.

The original PBIX/screenshots remain byte-identical historical snapshots. Bounded inspection confirmed two pages and 15/8 visuals. Power BI Desktop, pbi-tools, Tabular Editor and DAX Studio are unavailable here, so no native refresh was performed. The PBIX DataModel is opaque: DAX definitions, relationship propagation, CSV source paths and fixed import widths cannot be certified from layout.

The overview has model alias and `test` filters; most KPIs use Balanced, while scenario comparison intentionally spans scenarios. The appendix filter is attached to feature importance, not directly to every displayed table, so its metric measure/cross-table filtering is unverified. Historical dollar formatting describes utility units, and historical held-out wording means reused comparison. The score histogram can mix labeled comparison and unlabeled Kaggle rows unless population filters are applied. No visible layout binding proves current simulated-decline action semantics.

[The new correctness report](../reports/correctness_20261004/assessment_report.md) identifies an independently reconciled corrected CSV bundle, model/build/calibrator identities and hashes. [Corrected aggregate HTML](../reports/correctness_20261004/corrected_evidence.html) and PNG are newly generated anonymous evidence from that bundle; they are not substituted claims of PBIX refresh. A later native refresh must select the identified corrected folder, confirm model/split/scenario/population filters per visual, inspect DAX/relationships/import width, apply simulated-decline labels and replace dollar/profit wording with utility units before publishing updated screenshots.
