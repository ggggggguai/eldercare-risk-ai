# KINECAL skeleton subset

Dataset: KINECAL: A Dataset for Falls-Risk Assessment and Balance Impairment Analysis

Official page: https://physionet.org/content/kinecal/1.0.3/

Direct file root: https://physionet.org/files/kinecal/1.0.3/

DOI: `10.13026/vvkp-ct80`

Version: `1.0.3`

License: CC0 1.0

## Local scope

This project downloads the minimum skeleton subset needed for lightweight gait TCN experiments:

- Official risk groups: `NF`, `FHs`, and `FHm`.
- Expected group counts: `33`, `15`, and `9`, for 57 participants in total.
- Movements: `3m-walk-Front-View`, `Get-Up-And-Go-Front-View`, and `STS-5`.
- Modality: Kinect V2 25-joint skeleton text files only.
- Excluded: all `depth/DepthUshort*.bin` files and the other eight balance movements.

Official v1.0.3 source coverage for this scope:

- 171 requested participant-movement combinations.
- 154 available skeleton recordings.
- 55,509 available frame files totaling 93,005,183 bytes.
- 17 participant-movement directories are absent from the official PhysioNet release; they are not local download failures.

The absent source recordings are:

| Participant | Missing movements |
|---|---|
| `SPPB3` | 3 m walk, TUG |
| `SPPB80` | 3 m walk, TUG |
| `SPPB81` | 3 m walk, TUG |
| `SPPB87` | STS-5 |
| `SPPB307` | STS-5 |
| `SPPB403` | 3 m walk |
| `SPPB404` | 3 m walk |
| `SPPB407` | TUG |
| `SPPB501` | 3 m walk, TUG |
| `SPPB502` | 3 m walk, TUG, STS-5 |
| `SPPB705` | STS-5 |

Local snapshot verified on 2026-07-17:

- Manifest rows and verified skeleton files: 55,509.
- Verification errors: 0.
- Unexpected `.bin` files and unfinished `.part` files: 0.
- Local directory usage: approximately 237 MiB, including the manifest and small-file filesystem allocation.

The source paper defines `NF/FHs/FHm` as older risk groups, but four records in `register.csv` have numeric ages of 60 or 64. The downloader preserves the official group labels and records these rows in `download_summary.json` under `age_group_mismatch`. Experiments should report a sensitivity analysis that excludes these four records.

## Download

Confirm that the project editable install points to this repository:

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

Download the selected skeleton subset:

```bash
conda run -n eldercare-ai python scripts/annotation/download_kinecal.py \
  --output-dir data/external/kinecal/raw \
  --workers 32 \
  --retries 3 \
  --allow-missing
```

The command is resumable. Files that already exist with the expected byte size are skipped. Downloads are written to `.part` files and atomically renamed after size validation. When the optional project `service` dependencies are installed, the downloader reuses HTTP connections through `httpx`; otherwise it falls back to the slower standard-library transport.

Strict mode deliberately returns a non-zero exit for this release because the 17 directories listed above are absent at the source. `--allow-missing` acknowledges those source gaps; it does not make them complete. The verification command below is still mandatory because it fails on any individual file download failure, size/hash mismatch or accidental depth file.

## Verify

```bash
conda run -n eldercare-ai python scripts/annotation/download_kinecal.py \
  --output-dir data/external/kinecal/raw \
  --verify-only
```

Verification checks required metadata, manifest JSON, file sizes, local SHA-256 hashes, unsafe paths and accidental `.bin` depth files.

## Local layout

```text
data/external/kinecal/raw/
  LICENSE.txt
  register.csv
  download_manifest.jsonl
  download_summary.json
  kinecal/
    <participant_number>/
      <participant_number>_<movement>/
        skel/
          <clock_tick>.txt
```

Raw files and generated manifests are ignored by Git. This README is tracked.

## Skeleton frame format

Each frame file normally contains 25 whitespace-separated rows:

```text
SpineBase Tracked 0.116 0.05266726 4.351724 1001.183 531.8633
KneeLeft Tracked 0.02743349 -0.3494925 4.337027 982.2243 630.4145
```

Columns:

```text
joint_name tracking_state x_3d y_3d z_3d x_pixel y_pixel
```

Order frame files by the numeric filename stem. Do not rely on filesystem iteration order. Preserve `tracking_state` as a quality mask; inferred or missing joints should not have the same weight as tracked observations.

## Prepare and train the lightweight TCN

Run these commands from the repository root. No manual parsing is required.

Step 1: convert the raw KINECAL 3 m walks into training tensors:

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_kinecal_gait_tcn.py \
  --input-dir data/external/kinecal/raw \
  --output-dir data/processed/fall_risk/kinecal_gait_tcn \
  --target-fps 30 \
  --max-gap-sec 0.10 \
  --window-frames 128 \
  --stride-frames 64 \
  --seed 42
```

The command performs the complete conversion:

```text
Kinect 25-joint frame text
  -> 12 joints shared with COCO
  -> derived pelvis and shoulder centers
  -> 14-joint canonical skeleton
  -> 30 FPS time resampling
  -> pelvis centering and torso-scale normalization
  -> x, y, dx, dy and quality channels
  -> fixed [128, 14, 5] windows
  -> participant-level train/validation/test split
```

It also treats Kinect `-∞` pixel projections as missing observations and supports source filenames that include a numeric prefix before the clock timestamp.

Generated files:

```text
data/processed/fall_risk/kinecal_gait_tcn/
  dataset.npz     model-ready arrays
  metadata.json   joint/channel contract, hashes and participant split
  samples.jsonl   one auditable record per window
```

The verified local preparation contains 50 participants and 219 windows. Seven of the 57 risk-group participants have no official 3 m walk recording. The participant split is 36 train, 7 validation and 7 test; no participant crosses partitions.

Step 2: train the lightweight TCN:

```bash
conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/kinecal_gait_tcn/dataset.npz \
  --output-dir reports/fall_risk/kinecal_gait_tcn \
  --epochs 80 \
  --batch-size 16 \
  --patience 15 \
  --seed 42 \
  --device cpu
```

Training writes:

```text
reports/fall_risk/kinecal_gait_tcn/
  best_model.pt
  metrics.json
  history.jsonl
  test_participant_predictions.jsonl
```

The model has 14,114 parameters. Metrics are calculated both by window and, more importantly, after averaging windows by participant. The current fixed-split baseline is not usable: its seven-person test balanced accuracy is `0.333` and ROC-AUC is `0.583`. See the tracked experiment report for interpretation and reproduction details.

Recommended task separation:

- Train the primary gait TCN on `3m-walk-Front-View`.
- Use TUG only after phase segmentation, or as an auxiliary/multitask sequence; it mixes sit-to-stand, walking, turning and sitting.
- Route `STS-5` to the sit-stand branch rather than treating it as gait.

Suggested labels:

| Source field | Meaning | Allowed use |
|---|---|---|
| `NF` | No reported fall in the source grouping | fall-history proxy negative |
| `FHs` | Single reported fall | fall-history proxy positive/subclass |
| `FHm` | Multiple reported falls | stronger fall-history proxy/subclass |
| `clinically-at-risk` | Source clinical risk flag | separate classification target |

Split train, validation and test by `part_id`. All movements and windows from one participant must remain in the same partition. The implemented command enforces this rule, but the first baseline uses only one fixed split. Repeated participant-level cross-validation is still required before model selection. Fall history is a retrospective proxy, not a future fall outcome.
