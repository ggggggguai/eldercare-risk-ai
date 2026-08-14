# M0-S verification

## Execution identity

- WSL distribution: `Ubuntu-22.04`
- Conda environment: `eldercare-ai`
- Ext4 source: `/home/lenovo/work/eldercare-wandering-m0s-20260812-v1/source`
- Ext4 runs/evidence: `/home/lenovo/work/eldercare-wandering-m0s-20260812-v1/{runs,evidence}`
- Exact M0-S source bundle SHA-256: `70b0b92c9e8aea6b49f500dcaee11e16f266a58da88903e0c4d3dbea7f38feb8`
- Materialized development manifest SHA-256: `2a58370a3029a19f2665cb60e19c107146b0c473a71d13c78ee2ddf27a624548`
- Formal evidence copy tree SHA-256: `a35108c52592178299d04ca7c0ecde08150e8edb769656ceefb1516c66834bc5`
- Formal evidence copy: 169 files, 106,285,309 bytes

The source bundle is computed from the exact performance module, model module, preprocessing-bundle accessor, CLI, performance config, and forward config. Each run records the same file hashes and bundle hash in `resolved_config.json`; aggregation rejects a mismatch.

## Gates and commands

Environment binding was checked with:

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

The final source passed:

```bash
conda run -n eldercare-ai python -m pytest -q tests/test_wandering_performance.py
# 12 passed

conda run -n eldercare-ai python -m pytest -q tests/test_wandering_*.py
# 229 passed, 160 subtests passed

conda run -n eldercare-ai python -m pytest -q
# 710 passed, 239 subtests passed, 1 unrelated fall-runtime failure
```

The only warning was an NVIDIA-driver/PyTorch CUDA compatibility warning. The formal M0-S config is CPU-only and explicitly does not claim CUDA resume support.

The single full-suite failure was `tests/test_fall_runtime_fingerprint.py::FallRuntimeFingerprintTest::test_repository_acceptance_config_has_hashable_fixed_inputs`: the unrelated fall-risk fixed input `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi` is absent in this checkout. No M0-S or other wandering test failed.

Formal phases used fresh paths and refused overwrite:

```bash
conda run -n eldercare-ai python scripts/wandering/train_wandering_performance.py \
  --mode overfit-smoke --config configs/modules/wandering_performance_v1.yaml \
  --project-root . --run-dir /home/lenovo/work/eldercare-wandering-m0s-20260812-v1/evidence/overfit-smoke

conda run -n eldercare-ai python scripts/wandering/train_wandering_performance.py \
  --mode train --config configs/modules/wandering_performance_v1.yaml \
  --project-root . --run-dir <fresh-seed-dir> --seed <fixed-seed>

conda run -n eldercare-ai python scripts/wandering/train_wandering_performance.py \
  --mode evaluate --config configs/modules/wandering_performance_v1.yaml \
  --project-root . --run-dir <completed-seed-dir> --seed <same-fixed-seed>

conda run -n eldercare-ai python scripts/wandering/train_wandering_performance.py \
  --mode aggregate --config configs/modules/wandering_performance_v1.yaml \
  --project-root . \
  --runs-root /home/lenovo/work/eldercare-wandering-m0s-20260812-v1/runs \
  --run-dir /home/lenovo/work/eldercare-wandering-m0s-20260812-v1/evidence/aggregate-v1
```

Runs were sequential in the fixed order `20260731`, `20260801`, `20260802`. Each training process exited before its fresh-evaluation process started; the next seed did not start until the prior seed's fresh evaluation completed.

## Crash-recovery fault injection

`test_recovery_repairs_checkpoint_history_latest_and_best_after_fault_window` creates two complete numeric epoch checkpoints and then deliberately leaves history/latest/best at epoch 1. Recovery treats the atomically renamed epoch-2 checkpoint as authoritative, regenerates history/latest/best, reloads model/AdamW/CPU RNG at epoch 2, and reports each repaired index. This covers the minimal crash window that the original M0 resume probe did not cover.

## Copy and original-evidence integrity

The ext4 `runs/` + `evidence/` tree and the repository copy were hashed from sorted relative paths plus each file SHA-256. Both yielded:

```text
a35108c52592178299d04ca7c0ecde08150e8edb769656ceefb1516c66834bc5
169 files
```

The historical original M0 directory was hashed before work and after all formal runs/copying. Both checks yielded:

```text
reports/mental_health/wandering_performance/m0_seed20260731_v1
79 files
53,017,495 bytes
tree manifest SHA-256: f3a5f973283d1ec7f944cd2224fa7c44eba4acb249c325041fd82a9849e7a504
```

Therefore the original M0 outputs were not modified or overwritten.

## Stop boundary

No WP frozen-test accessor, inference, scoring, selection, report, or publication step was run. No SmartCare official/raw, WP raw, or sealed camera data was read. Work stopped after development-candidate freeze as required.
