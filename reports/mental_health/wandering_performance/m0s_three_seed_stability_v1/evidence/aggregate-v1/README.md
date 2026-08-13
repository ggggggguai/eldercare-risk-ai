# M0-S three-seed stability

- Gate: PASS
- Fixed seeds: [20260731, 20260801, 20260802]
- Frozen primary seed: 20260731
- RF/TCN comparisons reuse saved same-seed validation predictions; no baseline retraining.
- Shared preprocessing integrity parsing is distinguished from frozen-test exposure/use.

| Seed | WP 4-class macro-F1 | Source-equal binary macro-F1 | Best epoch | Train sec | Batch latency ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260731 | 0.987497 | 0.997238 | 5 | 432.8 | 1263.446 |
| 20260801 | 0.987497 | 0.983848 | 4 | 405.8 | 1238.822 |
| 20260802 | 0.987497 | 0.997238 | 8 | 520.9 | 1235.949 |

Frozen WP test, SmartCare official, and sealed camera data were not exposed or used. This is development-validation evidence, not camera, real-older-adult, product, or clinical evidence.
