# M0-CAM-C01-F verification

> Post-completion audit note: the commands and results below remain valid evidence for the original C01-F scope. A later synthetic red-team audit reopened bounded C01-F2 for active-root/samefile path identity, receipt-first external path resolution, pair-only versus video-direct source-SHA semantics, and detector/tracker component portability. These original counts did not by themselves prove F2 complete; C01-F2 subsequently completed those findings. Current status is maintained only in the task table, and real C0/C1 input remains closed.

## Environment and boundary

- Date: 2026-08-13
- Environment: `eldercare-ai`
- Editable package: verified to point to this checkout
- Human/authorized camera data read: no
- WP, SmartCare, sealed camera, or official test data read: no
- Camera QC, RF/TCN, fixed-primary inference, evaluator, or M0-CAM-D run: no
- Candidate/model/release/threshold/label/split/shared tracker modified: no
- Network access: no
- Commit/push performed: no

## Required commands

All Python commands use the project conda environment.

```bash
conda run --no-capture-output -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_collection.py -q

conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_*.py -q

conda run --no-capture-output -n eldercare-ai python scripts/wandering/build_camera_tracking_pair.py --help
conda run --no-capture-output -n eldercare-ai python scripts/wandering/prepare_camera_session.py --help
conda run --no-capture-output -n eldercare-ai python scripts/wandering/build_camera_annotations.py --help
conda run --no-capture-output -n eldercare-ai python scripts/wandering/run_camera_development.py --help
conda run --no-capture-output -n eldercare-ai python scripts/wandering/run_camera_inference.py --help
conda run --no-capture-output -n eldercare-ai python scripts/wandering/run_topowander_camera_inference.py --help
```

Repository/static checks:

```bash
git diff --check
git diff --cached --name-only
```

Strict UTF-8, Markdown fence balance, local-link targets, public signatures/exports, CLI forbidden terms, summary sensitive-tree, and protected source-scope checks are also run as focused automated or read-only audits.

## Attack regression coverage

- both public preparation signatures contain no context/provenance/status/scope/hook/fake input; removed keyword calls raise `TypeError`;
- an authorized sidecar passed to the public synthetic helper causes zero tracking loader calls and zero output;
- synthetic round-trip records only fixed false provenance and reloads canonical tracking/sidecar/summary;
- public authorized pair-only preparation records all execution/QC/model/M0-CAM-D flags as false;
- video-controller success independently recomputes receipt, collection, source, tracking, and sidecar bindings from input/output bytes;
- recursive summary scanning rejects raw sensitive structures, paths, URLs, headers, tokens, credentials, and ID lists while allowing aggregate scope counts;
- invalid `media_ref` and timezone families fail before protected tracking or all video/runtime/tracker/temp work;
- in-project and symlink-resolved production paths fail before video or protected input reads;
- same-length video mutation during tracker execution is detected by the post-tracker SHA and leaves no raw/staging/final output;
- source, receipt scope, controller, controller schema, detector, tracker, source SHA, metadata, and raw-count mismatches fail closed;
- both owner Markdown template JSON blocks are parsed and rejected by the formal validators;
- shared-tracker-shaped noncanonical JSONL still canonicalizes, rebinds its sidecar, and round-trips;
- the C01 builder/pair-only and authorized RD/annotation entry help exposes no caller-selected provenance/context, evidence/scope, bypass, skip, fake runtime/loader, test fixture, model, threshold, or frame-limit override. The historical comparison-only camera runner retains its pre-existing sidecar `--authorization-status` argument; it is not the C01 receipt-gated controller and was not changed or treated as authorization evidence.

## Results

- Editable package: `elderly-monitoring-algorithms==0.2.0`; editable project location resolved to this checkout.
- Focused dataset/controller regression: `67 passed in 7.13s`.
- All runnable `tests/test_wandering_camera_*.py`: `196 passed in 73.16s`.
- Six camera/RD CLI `--help` commands: all exited 0. The four C01/authorized-RD/annotation entry sources had zero forbidden provenance/context/evidence/scope/hook/fake/skip/bypass terms; both remaining historical/fixed-primary camera helps were runnable.
- Public API/export audit: exact three production signatures passed; `camera_collection.__all__` contains only the controller and error; private video path/type had zero CLI exposure.
- Template regression: both Markdown JSON blocks were parsed and rejected by the formal receipt/collection validators.
- Strict UTF-8 decode: 250 changed/untracked Markdown/Python files in the dirty shared worktree passed. Markdown fence balance passed for 80 files.
- C01 local links: all resolved. The broad changed-worktree scan reported six unrelated missing generated report targets, including three already documented in the earlier PREP audit; none is a C01 link or introduced by C01-F.
- `git diff --check`: passed with only existing LF-to-CRLF working-tree warnings.
- Git staged paths: empty. No commit or push was performed.
- Protected source audit: no diff in the shared fall tracker, primary config, wandering model, or wandering release code.

These passing software checks remain `synthetic_schema_contract_only`; they are not real-camera, human, older-adult, product, clinical, accuracy, F1, recall, FAR, calibration, episode, or deployment-latency evidence.
