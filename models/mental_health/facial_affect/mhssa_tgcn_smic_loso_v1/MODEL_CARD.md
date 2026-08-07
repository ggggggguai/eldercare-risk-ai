# MHSSA-TGCN SMIC LOSO Model Card

> Run: `MHSSA-SMIC-LOSO-20260806-001`  
> Status: `completed`  
> Release: `experimental / offline_only`  
> Annotation: `estimated / engineering_only`

## Scope

This package contains subject-level LOSO fold checkpoints for SMIC three-class micro-expression classification. It is not a psychological diagnosis model and is not connected to S10, backend alerts, or the psychological observation table.

## Aggregate Metrics

| Metric | Value |
|---|---:|
| Samples | 164 |
| UF1 / Macro-F1 | 0.462002 |
| UAR / Macro Recall | 0.471581 |
| Balanced Accuracy | 0.471581 |
| Accuracy | 0.457317 |

## Fold Results

| Test subject | Best epoch | UF1 | UAR | Balanced Accuracy |
|---|---:|---:|---:|---:|
| s01 | 34 | 0.4127 | 0.3889 | 0.3889 |
| s02 | 16 | 0.4167 | 0.5333 | 0.8000 |
| s03 | 21 | 0.4865 | 0.5101 | 0.5101 |
| s04 | 17 | 0.2222 | 0.3500 | 0.3500 |
| s05 | 18 | 0.6667 | 0.6667 | 1.0000 |
| s06 | 14 | 0.3333 | 0.3333 | 0.5000 |
| s08 | 23 | 0.2095 | 0.2870 | 0.4306 |
| s09 | 17 | 0.0000 | 0.0000 | 0.0000 |
| s11 | 22 | 0.2000 | 0.3333 | 0.3333 |
| s12 | 19 | 0.3897 | 0.5417 | 0.8125 |
| s13 | 11 | 0.0000 | 0.0000 | 0.0000 |
| s14 | 20 | 0.4667 | 0.4778 | 0.4778 |
| s15 | 38 | 0.5556 | 0.6667 | 0.6667 |
| s18 | 15 | 0.2963 | 0.2667 | 0.4000 |
| s19 | 15 | 0.6667 | 0.6667 | 1.0000 |
| s20 | 12 | 0.4195 | 0.4619 | 0.4619 |

## Reproducibility

- Artifact manifest SHA-256: `baa574855209d47a398f09c7b8f27aba6f442b0d9a25054346e73f372573cc86`
- Split manifest SHA-256: `dec3acd9dafbaa4f0d746b275d0ed5413e43f0a1fce7d2f92303b41fc6dc1b40`
- Resolved config SHA-256: `3762d1f9b956909226894f9f0febdf60f92599550e10ae06e754fc31fbe827b7`
- Checkpoint selection: validation Macro-F1, then validation loss
- Test fold used for selection: false

## Limitations

- SMIC apex frames are engineering estimates based on aligned motion energy.
- The dataset is small and class/subject distributions are imbalanced.
- Fold checkpoints are evaluation artifacts, not production deployment weights.
- Results do not claim reproduction of the paper's final metrics.
