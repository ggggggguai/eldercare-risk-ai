# TopoWander-MPT supervised development validation

- Candidate: M0-S
- Initialization: scratch
- Seed: 20260801
- Train / validation: 1257 / 278
- Best epoch: 4 of at most 50
- Parameters: 204466 total, 198258 optimized
- Training seconds: 405.768; average epoch: 33.814
- WP four-class macro-F1: 0.987497
- WP binary macro-F1: 0.994475
- SmartCare binary macro-F1: 0.973221
- Source-equal binary macro-F1: 0.983848
- Selection score: 0.983848
- Target >= 0.95: yes
- Fresh reload: PASS
- Best checkpoint: `checkpoints/epoch-0004`
- Last checkpoint: `checkpoints/epoch-0012`
- Largest shortfall: smartcare_binary_recall:direct_or_non_wandering = 0.941176

| WP class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| direct | 0.983607 | 1.000000 | 0.991736 | 60 |
| pacing | 0.967742 | 1.000000 | 0.983607 | 60 |
| lapping | 1.000000 | 0.983333 | 0.991597 | 60 |
| random | 1.000000 | 0.966667 | 0.983051 | 60 |

WP frozen test, SmartCare official, and sealed camera data were not consumed. These are public trajectory development-validation results, not camera, real-older-adult, product, or clinical evidence.
