# Experiment 004: Probability Calibration

## Purpose

Test whether post-hoc calibration can improve the probability-quality story for the current best post-v1 feature set without giving up the ranking gains from Experiment 003.

## Hypothesis

Experiment 003 improved PR-AUC, ROC-AUC, lift, recall, and expected value, but worsened Brier score. A post-model calibration layer should reduce Brier score and calibration-bin error while preserving ranking metrics if the selected calibrator is monotonic.

## Change Tested

- Base model: Experiment 003 LightGBM artifact, `lightgbm_credit_risk_v1`.
- Feature set: unchanged, 140 features.
- New command: `python -m src.calibrate --config configs/base.yaml`.
- New artifact: `models/lightgbm_credit_risk_calibration.joblib`.
- New outputs:
  - `reports/model_calibration_comparison.csv`
  - `reports/model_calibration_bins_comparison.csv`
  - DuckDB tables `model_calibration_comparison` and `model_calibration_bins_comparison`

## Fit and Selection Protocol

Calibration is fit on the saved validation split only. The held-out test split is used for reporting only.

Compared candidates:

- uncalibrated LightGBM scores;
- sigmoid / Platt-style calibration;
- isotonic calibration.

Selection rule: require at least `0.0005` validation Brier improvement over uncalibrated scores. Prefer sigmoid when it is within `0.0005` Brier of isotonic because sigmoid is simpler and rank-preserving. Under this rule, the selected calibrator is `sigmoid`.

## Calibration Comparison

| Method | Split | PR-AUC | ROC-AUC | Brier | Weighted bin error | Max bin error | Top-decile lift |
|---|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | Validation | 0.271066 | 0.778197 | 0.175712 | 0.296805 | 0.498483 | 3.641009 |
| Sigmoid | Validation | 0.271066 | 0.778197 | 0.066535 | 0.002704 | 0.011507 | 3.641009 |
| Isotonic | Validation | 0.265858 | 0.779744 | 0.066276 | 0.000690 | 0.002080 | 3.638324 |
| Uncalibrated | Held-out test | 0.268457 | 0.778805 | 0.174848 | 0.295293 | 0.494613 | 3.592677 |
| Sigmoid | Held-out test | 0.268457 | 0.778805 | 0.066550 | 0.002823 | 0.008458 | 3.592677 |
| Isotonic | Held-out test | 0.259784 | 0.778314 | 0.066597 | 0.004756 | 0.014658 | 3.606103 |

## Selected Calibration Result

| Metric | Experiment 003 uncalibrated | Experiment 004 sigmoid | Difference |
|---|---:|---:|---:|
| Validation PR-AUC | 0.271066 | 0.271066 | +0.000000 |
| Validation ROC-AUC | 0.778197 | 0.778197 | +0.000000 |
| Validation Brier score | 0.175712 | 0.066535 | -0.109176 |
| Validation weighted bin error | 0.296805 | 0.002704 | -0.294101 |
| Validation top-decile lift | 3.641009 | 3.641009 | +0.000000 |
| Test PR-AUC | 0.268457 | 0.268457 | +0.000000 |
| Test ROC-AUC | 0.778805 | 0.778805 | +0.000000 |
| Test Brier score | 0.174848 | 0.066550 | -0.108298 |
| Test weighted bin error | 0.295293 | 0.002823 | -0.292470 |
| Test top-decile lift | 3.592677 | 3.592677 | +0.000000 |

## Interpretation

This is a clear calibration improvement. Sigmoid calibration sharply reduces Brier score and calibration-bin error on both validation and held-out test while preserving PR-AUC, ROC-AUC, lift, precision at top decile, recall at review capacity, and expected value because it is monotonic.

Isotonic has the best validation Brier and bin error, but its gain over sigmoid is tiny on the fit split and it weakens held-out test PR-AUC. Sigmoid is the more conservative selected calibrator for this project.

## Limitations

- This is still a portfolio decision-support simulation, not production probability-of-default validation.
- The calibration fit uses the validation split, so validation calibration metrics are fit-split metrics. The held-out test metrics are the cleaner evidence of generalization.
- Business thresholds and EV are rank-based in the current workflow; sigmoid calibration improves probability scale but does not change rank-based action outcomes.

## Conclusion

Improved calibration with no ranking loss. The selected sigmoid calibration layer fixes the main Experiment 003 drawback: probability quality. The post-v1 model story is now stronger if presented as a ranked risk model with a separately validated calibration layer, not as a production underwriting model.

## Next Action

Decide whether to integrate the selected sigmoid-calibrated scores into downstream scoring/dashboard exports, or keep calibration as a documented post-v1 experiment artifact while moving to feature-selection and simplification.
