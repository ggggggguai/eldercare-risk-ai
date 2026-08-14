# M0-CAM-C01-PREP verification

> This file records the initial PREP verification. A later independent synthetic red-team review found caller-supplied provenance and missing on-disk receipt/collection/source binding gaps. Those were subsequently closed by [M0-CAM-C01-F](../wandering_camera_c01_f_v1/VERIFICATION.md); the test results below remain evidence for the initial round-trip and receipt-first implementation only.

## Environment

- Date: 2026-08-13
- Environment: `eldercare-ai`
- Editable package: verified to point to this checkout before implementation
- Human/authorized camera data read: no
- Network access: no
- Shared fall tracker modified: no
- Candidate/model/release/threshold/label/split modified: no
- Commit/push performed: no

## Commands

All Python commands run in the project conda environment.

```bash
conda run --no-capture-output -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_collection.py \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_adapter.py -q

conda run --no-capture-output -n eldercare-ai python scripts/wandering/build_camera_tracking_pair.py --help
conda run --no-capture-output -n eldercare-ai python scripts/wandering/prepare_camera_session.py --help
```

The wider camera regression used all nine runnable `tests/test_wandering_camera_*.py` files. Help was also run for the new builder, pair-only preparation, annotation builder, authorized development runner, historical comparison-only camera runner, and fixed-primary camera runner.

## Automated contract coverage

- shared-tracker-shaped noncanonical JSONL drops `person_id`, `scene_region`, `center`, and `speed_px_per_sec`, rebuilds the sidecar, and reloads as a canonical five-field pair;
- source sidecar bytes remain unchanged and both source/output hashes and byte counts are recorded;
- staged round-trip failure removes staging and leaves no final output;
- missing, unapproved, inactive, expired, operation-mismatched, dataset-mismatched, purpose-mismatched, or scope-mismatched receipt fails before video resolution/hash/open, optional vision imports, tracker, raw temp, staging, or final output;
- unknown/duplicate source and pre-existing final output fail at the same pre-video boundary;
- the successful fake-media path spies on the shared tracker call, verifies fixed parameters and exact sidecar sources, reloads the output pair, and confirms raw-temp cleanup;
- tracker failure cleans raw temp and leaves no final output;
- CLI help exposes no caller-selected authorization/evidence/validation scope, test fixture, fake tracker/loader, skip, or bypass flag.

## Results

- C1 dataset/adapter/new controller narrow regression: `41 passed in 9.41s`.
- All runnable wandering camera tests: `157 passed in 77.22s`.
- Six relevant CLI `--help` commands: all exited 0.
- New CLI/API boundary: exact production API signature locked; no caller-selected authorization/evidence/validation scope, model, threshold, frame limit, test fixture, fake runtime/loader, skip, or bypass input.
- New C01 tests: self-contained `tmp_path`/in-memory inputs; no report artifact, candidate, real data, or network dependency.
- Owner templates: both were parsed from their Markdown code blocks and rejected by the formal validators as intended (`AUTHORIZATION_TEMPLATE_REJECTED=True`, `COLLECTION_TEMPLATE_REJECTED=True`) in a one-off conda audit; the temporary audit script was removed afterward.
- `git diff --check`: passed; only Git's existing LF→CRLF working-tree warnings were emitted.
- Strict UTF-8 decode: passed for all 21 changed/new Markdown and Python files present at audit time.
- C01 report and newly referenced local targets: passed (`C01_LOCAL_LINKS_OK=8`). A broad scan also found three absent generated fall-report JSON targets already present in `HEAD`'s `docs/README.md`; they were not introduced or changed by C01-PREP.
- Scoped source audit: no diff in `fall_risk/tracking.py`, primary config, wandering `model.py`, or wandering `release.py`.
- Git staging/commit/push: none.

## Interpretation

These are synthetic schema and software-contract results. They are not authorization, a real C1 artifact, a camera smoke, M0-CAM-D, or camera/human/older-adult/product/clinical performance evidence.
