# forecast_2m / lightgbm model card

## Purpose

Offline PSYCHE-D nominal-month conditional experiment for estimating a later higher PHQ-9 symptom-pattern label. This is not a diagnosis and has no product decision authority.

## Data and method

- Frozen raw inputs: 27 activity and sleep fields from the earlier nominal month.
- Prediction unit: participant-isolated target window.
- Evaluation: strict five-fold outer OOF with participant-equal primary weighting.
- Calibration: fold-local Isotonic Regression fitted on inner OOF only.
- Status: experimental, offline_only, shadow_only, product_visible=false.

## Primary OOF result

- AUPRC: 0.486916
- AUROC: 0.669109
- Macro-F1: 0.620268
- Sensitivity: 0.502685
- Specificity: 0.740091
- Brier: 0.203585
- ECE: 0.013785

## Limitations

The public matrix has design-level nominal-month evidence rather than row-level natural timestamps. PSYCHE-D is not an elderly-China validation cohort, and its Fitbit-derived fields do not establish transfer to the project's camera or sleep-monitor device domain. Feature contributions describe model association, not causation.
