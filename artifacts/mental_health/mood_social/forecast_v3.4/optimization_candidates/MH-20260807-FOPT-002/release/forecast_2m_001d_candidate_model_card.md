# forecast_2m / 001D fusion candidate model card

## Purpose

This is an offline PSYCHE-D nominal-month forecasting candidate for a later higher PHQ-9 symptom-pattern label. It is not a diagnosis, is not a China-elderly or device-domain validation, and has no product decision authority.

## Method and data boundary

- Inputs: the frozen 27 activity/sleep fields plus strictly past anchor, delta and rolling history features selected inside outer-train inner folds.
- Prediction unit: participant-isolated target window; all metrics use strict outer OOF predictions.
- Candidate: non-negative ElasticNet/LightGBM/CatBoost fusion with fold-local calibration and threshold selection.
- Evaluation: participant-equal metrics, 2,000 participant-cluster bootstrap repetitions, paired on the frozen common window where applicable.
- Release status: `experimental / offline_only / shadow_only / product_visible=false`.

## Full-window result

| Metric | Frozen baseline | 001D candidate | Difference |
|---|---:|---:|---:|
| AUPRC | 0.486916 | 0.494514 | +0.007598 |
| AUROC | 0.669109 | 0.678700 | +0.009590 |
| Macro-F1 | 0.620268 | 0.627561 | +0.007293 |
| Sensitivity | 0.502685 | 0.512436 | +0.009751 |
| Specificity | 0.740091 | 0.745028 | +0.004937 |
| Brier | 0.203585 | 0.200531 | -0.003053 |
| ECE | 0.013785 | 0.013655 | -0.000130 |

## Stability and decision

- AUPRC bootstrap median: `+0.007802`.
- AUPRC bootstrap 95% interval: `[-0.003488, 0.018418]`.
- Non-decreasing outer folds: `5/5`; worst-fold delta: `+0.012253`.
- Horizon decision: `promote_candidate`.

The overall release decision requires both horizons to pass. The package-level decision is recorded in `rollback_decision.json`; because the 1-month horizon did not meet the AUPRC gate, the existing V3.4 offline baseline remains the recommended artifact and this candidate is retained for audit only.

## Limitations

PSYCHE-D supplies design-level nominal-month evidence rather than natural timestamps. Fitbit-derived fields do not establish transfer to the project's camera or sleep-monitor domain. Feature contributions describe association, not causation. This card must not be used to claim medical diagnosis or deployment efficacy.
