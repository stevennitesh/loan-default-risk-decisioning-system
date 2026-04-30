# Experiment 009 - Recency-Deterioration Financial Features

## Change Tested

Added one compact SQL feature family, `f_recency_deterioration_features`, with eight recent-vs-lifetime monthly-history signals:

- `bureau_balance_recent_dpd_rate_delta`
- `bureau_balance_recent_status_delta`
- `pos_cash_recent_dpd_rate_delta`
- `pos_cash_remaining_installment_ratio_delta`
- `credit_card_recent_utilization_delta`
- `credit_card_recent_balance_ratio_delta`
- `credit_card_recent_drawings_ratio_delta`
- `credit_card_recent_dpd_rate_delta`

Recent is defined as `MONTHS_BALANCE >= -12`. The features are joined into `mart_credit_risk_features` at one row per `SK_ID_CURR` and `source_population`.

## Hypothesis

Default risk should not only depend on lifetime credit behavior. Recent deterioration should matter: rising delinquency rates, higher remaining installment burden, increasing revolving utilization, growing card balances, and increased card drawings are all plausible early stress signals. LightGBM can learn from the underlying recent and lifetime aggregates separately, but explicit deltas make this deterioration mechanism cheaper for the model to use.

## Files Changed

- `src/build_features.py`
- `sql/05c_feature_recency_deterioration.sql`
- `sql/06_build_feature_mart.sql`
- `configs/base.yaml`
- `tests/test_feature_sql.py`

## Validation Metrics

Metrics below use validation for the experiment decision. Brier values use the sigmoid-calibrated score so this remains comparable with Experiments 004-008.

| Metric | Experiment 008 | Experiment 009 | Difference |
|---|---:|---:|---:|
| Feature count | 144 | 152 | +8 |
| Validation PR-AUC | 0.271644 | 0.272970 | +0.001327 |
| Validation ROC-AUC | 0.779268 | 0.778720 | -0.000548 |
| Validation Brier | 0.066519 | 0.066473 | -0.000046 |
| Validation top-decile lift | 3.641009 | 3.641009 | +0.000000 |
| Validation precision at top decile | 0.293952 | 0.293952 | +0.000000 |
| Validation recall at review capacity | 0.364125 | 0.364125 | +0.000000 |
| Validation balanced EV / applicant | 574.90 | 577.63 | +2.73 |

Compared with the Experiment 004 calibrated 140-feature model, Experiment 009 improves one-shot validation PR-AUC by `+0.001904`, calibrated Brier by `-0.000062`, and balanced EV per applicant by `+1.56`. Lift and review-capacity recall remain tied.

## Held-Out Test Metrics

Held-out test is reported only after the validation decision. It is not used to select or tune the experiment.

| Metric | Experiment 008 | Experiment 009 | Difference |
|---|---:|---:|---:|
| Test PR-AUC | 0.268472 | 0.268340 | -0.000132 |
| Test ROC-AUC | 0.780392 | 0.779274 | -0.001118 |
| Test Brier | 0.066496 | 0.066547 | +0.000051 |
| Test top-decile lift | 3.571196 | 3.622213 | +0.051017 |
| Test precision at top decile | 0.288316 | 0.292434 | +0.004118 |
| Test recall at review capacity | 0.357143 | 0.362245 | +0.005102 |
| Test balanced EV / applicant | 583.73 | 582.04 | -1.69 |

The test result is mixed but close: rank PR-AUC/ROC/Brier are slightly lower than Experiment 008, while top-decile lift and recall are better. This is a stability signal, not a reason to tune against test.

## SHAP Read

The new family has one strong driver and several weaker supporting signals:

| Feature | SHAP rank |
|---|---:|
| `pos_cash_remaining_installment_ratio_delta` | 15 |
| `credit_card_recent_drawings_ratio_delta` | 76 |
| `pos_cash_recent_dpd_rate_delta` | 87 |
| `credit_card_recent_dpd_rate_delta` | 96 |
| `bureau_balance_recent_status_delta` | 118 |
| `credit_card_recent_balance_ratio_delta` | 133 |
| `bureau_balance_recent_dpd_rate_delta` | 136 |
| `credit_card_recent_utilization_delta` | 139 |

The standout signal is the POS/cash remaining-installment deterioration feature. That is financially plausible: a higher recent share of remaining installments can indicate less seasoned repayment history and higher near-term obligation burden.

## Conclusion

Experiment 009 is the strongest one-shot feature-engineering candidate so far on the validation criteria. It improves validation PR-AUC, calibrated Brier, and balanced expected value versus Experiment 008 and versus the calibrated 140-feature model, while preserving lift and review-capacity recall.

The follow-up stability check in Experiment 010 promotes this 152-feature setup as the leading post-v1 ranking/calibration candidate. The caveat remains that the repeated-seed validation expected value is slightly lower than the 140-feature calibrated baseline, so this is not a clean win on every business metric.

## Next Action

Keep the 152-feature recency-deterioration setup as the active post-v1 candidate. If we want a cleanup follow-up, narrow the family around `pos_cash_remaining_installment_ratio_delta` and the other higher-ranked deltas to see whether the validation PR-AUC edge can be retained with lower variance.
