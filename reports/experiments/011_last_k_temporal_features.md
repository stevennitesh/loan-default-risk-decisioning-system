# Experiment 011 - Last-K Temporal Behavior Features

## Change Tested

Added one source-informed SQL feature family, `f_last_k_temporal_features`, with 16 recent-payment-history signals from the last few observed months or installments:

- installments last-3 late-payment, underpayment, delay, and payment-ratio features;
- installments last-payment delay and payment-ratio features;
- POS-cash last-3 delinquency, future-installment, and deterioration features;
- POS-cash last-loan delinquency feature;
- credit-card last-3 utilization, payment-to-minimum, drawing-count, delinquency, and utilization-delta features.

The features are joined into `mart_credit_risk_features` at one row per `SK_ID_CURR` and `source_population`.

## Research Framing

This is a research-informed experiment, not a copied solution. Public Home Credit solution writeups consistently suggest that temporal windows, last-loan behavior, and recent repayment dynamics are useful for this dataset. We used that as evidence for what kind of mechanism to test, then implemented a compact, auditable SQL feature family inside this project's own data contracts, validation loop, and reporting rules.

The source pattern was informed by public educational references including:

- [Neptune/Open solution dynamic feature notes](https://github.com/minerva-ml/open-solution-home-credit/wiki/LightGBM-clean-dynamic-features)
- [NoxMoon Home Credit repository](https://github.com/NoxMoon/home-credit-default-risk)
- [Oskird Kaggle Home Credit solution repository](https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution)

The concise framing should be: "I studied public solution patterns to identify plausible financial mechanisms, then rebuilt a narrow, testable version in SQL and evaluated whether it improved my validation metrics." It should not be framed as a leaderboard replication.

## Hypothesis

Lifetime averages can dilute near-term borrower stress. Recent late payments, underpayment, credit-card utilization, payment-to-minimum behavior, drawings, and active-installment burden should help the model identify applicants whose current repayment behavior is deteriorating even if their full-history aggregates look acceptable.

## Files Changed

- `src/build_features.py`
- `sql/05d_feature_last_k_temporal.sql`
- `sql/06_build_feature_mart.sql`
- `configs/base.yaml`
- `tests/test_feature_sql.py`

## Validation Metrics

Metrics below use validation for the experiment decision. Brier values use the sigmoid-calibrated score so this remains comparable with Experiments 004-010. The comparison baseline is Experiment 009, the prior one-shot recency-deterioration candidate. Experiment 010 remains the currently promoted setup because it has a repeated-seed stability check.

| Metric | Experiment 009 | Experiment 011 | Difference |
|---|---:|---:|---:|
| Feature count | 152 | 168 | +16 |
| Validation PR-AUC | 0.272970 | 0.274934 | +0.001964 |
| Validation ROC-AUC | 0.778720 | 0.779251 | +0.000531 |
| Validation Brier | 0.066473 | 0.066380 | -0.000093 |
| Validation top-decile lift | 3.641009 | 3.697396 | +0.056387 |
| Validation precision at top decile | 0.293952 | 0.298504 | +0.004552 |
| Validation recall at review capacity | 0.364125 | 0.369764 | +0.005639 |
| Validation balanced EV / applicant | 577.63 | 576.46 | -1.17 |

This is the strongest one-shot validation ranking result so far, but it is not a clean win on every metric because validation expected value declined slightly.

## Held-Out Test Metrics

Held-out test is reported only after the validation decision. It is not used to select or tune the experiment.

| Metric | Experiment 009 | Experiment 011 | Difference |
|---|---:|---:|---:|
| Test PR-AUC | 0.268340 | 0.271270 | +0.002930 |
| Test ROC-AUC | 0.779274 | 0.780914 | +0.001640 |
| Test Brier | 0.066547 | 0.066403 | -0.000144 |
| Test top-decile lift | 3.622213 | 3.627584 | +0.005371 |
| Test precision at top decile | 0.292434 | 0.292868 | +0.000434 |
| Test recall at review capacity | 0.362245 | 0.362782 | +0.000537 |
| Test balanced EV / applicant | 582.04 | 583.10 | +1.06 |

The held-out test movement is directionally supportive and close to validation, but it should remain a generalization check rather than an optimization target.

## SHAP Read

Several new features entered the upper half of the global SHAP ranking, especially recent POS/cash obligation burden and credit-card utilization:

| Feature | SHAP rank |
|---|---:|
| `pos_cash_last_3_future_installment_ratio` | 9 |
| `credit_card_last_3_credit_utilization` | 23 |
| `installments_last_3_late_payment_rate` | 32 |
| `credit_card_last_3_payment_to_min_ratio` | 47 |
| `installments_last_3_avg_payment_delay_days` | 53 |
| `pos_cash_last_loan_dpd_rate` | 72 |
| `installments_last_3_payment_amount_ratio` | 73 |
| `credit_card_last_3_drawing_count` | 89 |
| `credit_card_last_3_utilization_delta` | 115 |
| `pos_cash_last_3_dpd_rate_delta` | 125 |

The strongest new signal is `pos_cash_last_3_future_installment_ratio`, which is financially plausible: a high recent share of future installments can indicate active obligation burden and less seasoned repayment progress.

## Conclusion

Experiment 011 is a strong one-shot feature-engineering result. It improves validation PR-AUC, ROC-AUC, calibrated Brier, top-decile lift, precision, and review-capacity recall versus the prior 152-feature recency setup. The expected-value caveat matters: validation balanced EV declines by about `1.17` per applicant, so this should not be promoted as a universal business-value win yet.

## Decision

Mark Experiment 011 as promising, source-informed, and stability-required. The follow-up stability check in Experiment 012 promotes the 168-feature last-k temporal setup as the current post-v1 active candidate.

## Next Action

Use Experiment 012 as the promotion evidence. Keep the source-informed framing explicit and avoid describing this as copied Kaggle solution work.
