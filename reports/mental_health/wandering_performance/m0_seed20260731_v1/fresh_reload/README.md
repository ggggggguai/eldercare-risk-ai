# TopoWander-MPT supervised development validation

- Candidate: M0
- Initialization: scratch
- Seed: 20260731
- Train / validation: 1257 / 278
- Best epoch: 13 of at most 50
- Parameters: 204466 total, 198258 optimized
- Training seconds: 687.486; average epoch: 32.737
- WP four-class macro-F1: 0.995833
- WP binary macro-F1: 1.000000
- SmartCare binary macro-F1: 1.000000
- Source-equal binary macro-F1: 1.000000
- Selection score: 0.995833
- Target >= 0.95: yes
- Fresh reload: PASS
- Best checkpoint: `/home/lenovo/runs/eldercare-wandering/m0_seed20260731_v1/checkpoints/epoch-0013`
- Last checkpoint: `/home/lenovo/runs/eldercare-wandering/m0_seed20260731_v1/checkpoints/epoch-0021`
- Largest shortfall: wp_four_class_recall:lapping = 0.983333

| WP class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| direct | 1.000000 | 1.000000 | 1.000000 | 60 |
| pacing | 0.983607 | 1.000000 | 0.991736 | 60 |
| lapping | 1.000000 | 0.983333 | 0.991597 | 60 |
| random | 1.000000 | 1.000000 | 1.000000 | 60 |

WP frozen test, SmartCare official, and sealed camera data were not consumed. These are public trajectory development-validation results, not camera, real-older-adult, product, or clinical evidence.
