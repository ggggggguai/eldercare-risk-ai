# M0-R fixed-candidate Release-Prep report

Date: 2026-08-12

## Outcome

Release-Prep passed for the fixed M0-S primary candidate. The release candidate remains TopoWander-MPT seed `20260731`, best epoch `5`; it was not retrained, ensembled, threshold-tuned, or replaced with historical M0. The new candidate path has not opened, inferred on, or scored WP public holdout. This report preserves the completed v2 Release-Prep evidence. A subsequent read-only audit found three score-entry hardening items that must be closed in a new v3 before requesting the one-time public-holdout authorization; they do not invalidate the v2 validation parity or require model retraining.

## Frozen identities

| Layer | Evidence | SHA-256 / identity |
|---|---|---|
| Training candidate | `identity/training_candidate_identity.json` | `c4e8576fce46a69bca3cebf81306197c9abfa439ff720bb3ced95ea74f580228` |
| Complete model state | M0-S seed `20260731`, epoch `5` | `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031` |
| Frozen training source bundle | `identity/training_source_bundle/` | bundle `70b0b92c9e8aea6b49f500dcaee11e16f266a58da88903e0c4d3dbea7f38feb8` |
| Release implementation v2 | `identity/release_implementation_identity_v2.json` | `d544e3e9198acf348fd64f6b435ed337f8543db81d4ca632e829df4c3bac1f6d` |
| Persistent release source archive | `artifacts/release_implementation_source_v2.zip` | `f16175206ee490df3b3688beb42cf376a356048a7590140a72cc6c1b93e6dad0` |
| Release-Prep candidate manifest v2 | `artifacts/topowander_m0r_candidate_v2/candidate_manifest.json` | `e977ce18ab042e33591994b93ad30787c01840f72c6320fdb40b78122f27cae0` |

The source identities are explicitly `current_machine_local_only`, with Git HEAD `2981583736a74b1328e0fa52bf39028d6a3f8f7f` recorded only as context. No release commit or push was created. The release ZIP was actually extracted during terminal audit; all three extracted files matched both the current frozen source bytes and the per-file identity hashes.

The formal compact bundle contains exactly four files: complete `model_state.npz`, `forward_config.yaml`, `performance_config.yaml`, and `candidate_manifest.json`. It contains no optimizer state, RNG state, training history, or other-seed checkpoint.

## Release path contract

- The loader requires a trusted manifest SHA supplied outside the bundle before parsing or loading artifacts.
- Artifact paths must be bounded relative paths; file size and SHA-256 are checked before the safe NPZ model loader runs.
- The model is CPU-only, in `eval` mode, and inference uses `torch.inference_mode()`.
- The validator and evaluator accept WanderingPatterns only and require explicit `expected_split=validation|test`; they reject split relabeling.
- A fixed cohort must contain exactly 240 unique IDs, 60 records per four-class label, and binary counts 60/180.
- Decisions remain fixed: binary `sigmoid >= 0.5`; four-class hierarchical probabilities followed by `argmax`; class order is `direct, pacing, lapping, random` and subtype order is `pacing, lapping, random`.
- Outputs are written through staging and refuse overwrite.

## Numerical batch-layout diagnosis and v2 resolution

The first v1 WP-only parity attempt preserved the candidate and metrics but produced a maximum probability difference of `2.1291036811366126e-07`, above the fixed `1e-7` limit. Read-only diagnosis showed that the frozen M0-S reference is reproduced with difference `0.0` under both `no_grad` and `inference_mode` when the historical batch positions are retained. The drift came only from changing the layout from the historical `[64,64,64,64,22]` to WP-only `[64,64,64,48]`.

The v2 manifest therefore makes the numerical protocol explicit: before fixed-cohort inference, it prefixes 38 temporary copies of the first already-validated WP record, uses batch size 64, and discards every prefix output before evaluation. The parity controller reads and validates the already-authorized development bundle, which includes SmartCare development rows, before filtering the 240 WP validation rows; it does not read SmartCare official/raw and performs no SmartCare candidate inference or scoring. The evaluator still receives exactly the original 240 unique WP IDs and labels. No probability rounding, probability replacement, score postprocessing, weight change, or tolerance change is used. The superseded v1 bundle remains preserved; it was not overwritten. Full evidence is in `diagnostics/wp_only_batch_layout_diagnosis.json`.

## Fresh primary-validation parity

Formal evidence: `validation_parity_v2/`.

| Check | Result |
|---|---:|
| WP sample count | 240 |
| Sample IDs and labels | exact |
| Maximum absolute probability difference | `0.0` |
| Maximum absolute WP-only metric difference | `0.0` |
| Required tolerance | `1e-7` |
| WP four-class macro-F1 | `0.9874973951700553` |
| WP binary macro-F1 | `0.9944750109348742` |
| Public-holdout access by release path | false |

Four-class confusion matrix, row/column order `direct, pacing, lapping, random`:

```text
[[60, 0,  0,  0],
 [ 0, 60, 0,  0],
 [ 0, 1, 59,  0],
 [ 1, 1,  0, 58]]
```

Binary confusion matrix, row/column order `direct_or_non_wandering, wandering_like`:

```text
[[60,   0],
 [ 1, 179]]
```

The parity file SHA-256 is `34b6f5c72a5bb35241ef2195afd9847cfbf0ea56186fee1ddd50ced03b4ce1d2`; predictions SHA-256 is `dd8f1d6f32434dd010ba84c04c5a5a94ebd4a66b16f6691d57f15495960300ec`.

## Verification

- `conda run -n eldercare-ai python -m pytest tests/test_wandering_release.py -q`: 7 passed.
- `conda run -n eldercare-ai python -m pytest tests/test_wandering_release.py tests/test_wandering_performance.py -q`: 19 passed, one existing CUDA-driver warning on a CPU-compatible test path.
- `conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py -q`: 236 passed plus 160 subtests, one existing CUDA-driver warning.
- Release CLI `--help`: passed.
- Editable install: points to this checkout in the `eldercare-ai` environment.
- Training-source live bytes versus frozen snapshot: all 6 files matched.
- Release-source archive extraction and per-file hash verification: all 3 files matched.

## Post-completion audit and next gate

This is engineering and primary-validation reproducibility evidence, not WP public-holdout performance, camera validation, or clinical validation. The release path did not read WP raw, SmartCare official/raw, or sealed-camera data. Existing historical/accessor contract facts do not make a future WP result project-level fully blind.

The read-only audit confirmed the model/config hashes, compact bundle, external manifest binding, validation parity, no-test boundary, and wandering regressions. It also found that the future formal score path must be hardened before authorization:

1. before any future authorized accessor call, test scoring must bind the trusted upstream input identity and must not trust an arbitrary balanced `--records` JSONL; during that authorized score, it must obtain the fixed 240 records internally from `BUNDLE_MODE_FROZEN_WP_TEST` and record canonical records/order hashes in execution evidence;
2. validation parity and formal scoring must share the manifest-bound CPU runtime of intra-op `8`, inter-op `1`, and `batch=64`;
3. scoring must verify the actual release/training source bytes against the two frozen identities before opening the test accessor, or execute from the verified frozen source archive.

The next action is therefore M0-R score-entry hardening without allowing the new candidate/controller path to obtain WP test for inference or scoring. Existing accessor contract regressions may continue to parse fixed records for access/count/schema checks. The hardening must preserve v1/v2, keep the same seed/epoch/model/config/threshold, generate a new release identity and v3 manifest, reproduce WP validation parity, and then stop for explicit owner authorization. Only that new score-ready manifest may later form one refusal-to-overwrite WP public-holdout result. The result may report only WP four-class/binary metrics, per-class recall, confusion, predictions, and errors; it must not calculate a SmartCare source-equal test metric or feed any observed score back into model, threshold, split, or protocol changes.
