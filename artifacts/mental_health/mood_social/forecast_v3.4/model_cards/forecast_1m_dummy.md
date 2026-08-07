# forecast_1m / dummy model card

## Purpose

Offline PSYCHE-D nominal-month conditional experiment for estimating a later higher PHQ-9 symptom-pattern label. This is not a diagnosis and has no product decision authority.

## Data and method

- Frozen raw inputs: 27 activity and sleep fields from the earlier nominal month.
- Prediction unit: participant-isolated target window.
- Evaluation: strict five-fold outer OOF with participant-equal primary weighting.
- Calibration: none; raw prior retained.
- Status: experimental, offline_only, shadow_only, product_visible=false.

## Primary OOF result

- AUPRC: 0.324935
- AUROC: 0.494758
- Macro-F1: 0.488469
- Sensitivity: 0.197526
- Specificity: 0.801249
- Brier: 0.220423
- ECE: 0.004303

## Limitations

The public matrix has design-level nominal-month evidence rather than row-level natural timestamps. PSYCHE-D is not an elderly-China validation cohort, and its Fitbit-derived fields do not establish transfer to the project's camera or sleep-monitor device domain. Feature contributions describe model association, not causation.
