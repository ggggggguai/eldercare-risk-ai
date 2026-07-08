# LTMM raw download

Dataset: Long Term Movement Monitoring Database

Source record: https://physionet.org/content/ltmm/1.0.0/

Direct file index: https://physionet.org/files/ltmm/1.0.0/

Access: open PhysioNet project

License: Open Data Commons Attribution License v1.0

Local status:

| File or directory | Required now | Purpose |
|---|---|---|
| `raw/ClinicalDemogData_COFL.xlsx` | yes | Clinical and demographic table for control and faller groups |
| `raw/ReportHome75h.xlsx` | yes | Home monitoring report summary |
| `raw/CO*.hea`, `raw/FL*.hea` | optional | WFDB headers for long-term home accelerometer records |
| `raw/CO*.dat`, `raw/FL*.dat` | optional, large | Long-term home accelerometer records |
| `raw/LabWalks/*.hea`, `raw/LabWalks/*.dat` | optional | Laboratory walk accelerometer records |
| `raw/RECORDS` | recommended | Official record list |
| `raw/SHA256SUMS.txt` | recommended | Official checksum list |

Suggested minimal download:

```bash
mkdir -p data/external/ltmm/raw
curl -L https://physionet.org/files/ltmm/1.0.0/ClinicalDemogData_COFL.xlsx \
  -o data/external/ltmm/raw/ClinicalDemogData_COFL.xlsx
curl -L https://physionet.org/files/ltmm/1.0.0/ReportHome75h.xlsx \
  -o data/external/ltmm/raw/ReportHome75h.xlsx
curl -L https://physionet.org/files/ltmm/1.0.0/RECORDS \
  -o data/external/ltmm/raw/RECORDS
curl -L https://physionet.org/files/ltmm/1.0.0/SHA256SUMS.txt \
  -o data/external/ltmm/raw/SHA256SUMS.txt
```

Do not download all `.dat` files unless a long-term activity experiment needs them. The full dataset is large, and the raw signal files are ignored by git through `data/external/**/*`.

Generate a local manifest after download:

```bash
conda run -n eldercare-ai python scripts/annotation/build_ltmm_manifest.py \
  --raw-dir data/external/ltmm/raw \
  --output data/manifests/ltmm_manifest.jsonl
```

Use in this project:

- Long-term activity and personal-baseline experiments using waist accelerometer data.
- Fall-history, BBS, DGI, FSST, TUG and clinical proxy alignment when present in the LTMM tables.
- Cross-checking whether the baseline module can use multi-day movement summaries without video.

Do not use LTMM as video action labels:

```text
LTMM has accelerometer and clinical/functional assessment data, not ordinary RGB home video.
```

It should not be used to validate this chain:

```text
video -> detection/tracking -> pose -> pose quality
```

Use TOAGA and LE2I/IMViA for video and pose-chain validation, then use LTMM for long-term movement and fall-history risk calibration.
