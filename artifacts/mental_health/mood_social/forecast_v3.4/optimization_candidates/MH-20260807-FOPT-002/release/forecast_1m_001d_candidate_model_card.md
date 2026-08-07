# forecast_1m / 001D fusion candidate model card

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
| AUPRC | 0.482186 | 0.482799 | +0.000614 |
| AUROC | 0.664142 | 0.671453 | +0.007311 |
| Macro-F1 | 0.616978 | 0.624983 | +0.008005 |
| Sensitivity | 0.489131 | 0.517858 | +0.028727 |
| Specificity | 0.745471 | 0.736158 | -0.009313 |
| Brier | 0.205124 | 0.201506 | -0.003619 |
| ECE | 0.012775 | 0.019376 | +0.006601 |

## Stability and decision

- AUPRC bootstrap median: `+0.000427`.
- AUPRC bootstrap 95% interval: `[-0.014126, 0.015335]`.
- Non-decreasing outer folds: `3/5`; worst-fold delta: `-0.014003`.
- Horizon decision: `retain_baseline`.

The overall release decision requires both horizons to pass. The package-level decision is recorded in `rollback_decision.json`; because the 1-month horizon did not meet the AUPRC gate, the existing V3.4 offline baseline remains the recommended artifact and this candidate is retained for audit only.

## Limitations

PSYCHE-D supplies design-level nominal-month evidence rather than natural timestamps. Fitbit-derived fields do not establish transfer to the project's camera or sleep-monitor domain. Feature contributions describe association, not causation. This card must not be used to claim medical diagnosis or deployment efficacy.
