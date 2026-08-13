# M0-CAM-C01-PREP receipt-first C1 handoff

## Initial completion state

```text
status_at_initial_completion=wandering_m0cam_c0_c1_handoff_ready_waiting_owner_inputs
evidence_scope=synthetic_schema_contract_only
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

## Post-completion audit note

The round-trip and receipt-first CLI findings below remain valid, but an independent synthetic red-team review reopened bounded C01-F before any real C0/C1 use. Its original public context and atomic-binding findings were subsequently addressed by [M0-CAM-C01-F](../wandering_camera_c01_f_v1/README.md); a later audit routed active-root/resolver/source/component findings to [C01-F2](../wandering_camera_c01_f2_v1/README.md), which has since completed. The former `wandering_m0cam_c01_f2_integrity_fix_required` value is only a historical audit state; current operational status is maintained in the task table. `C0=false`, `C1=false`, and all PREP/F/F2 work remains `synthetic_schema_contract_only` evidence.

This initial milestone implemented the round-trip and receipt-first handoff body; C01-F is the separate closing evidence for the later audit findings. PREP did not receive an authorization receipt, read human video, generate a real C1 pair, run Camera QC, run RF/TCN or primary inference, start M0-CAM-D, or create camera-performance evidence.

## What changed

### Canonical pair round-trip

`prepare_camera_input_pair()` now distinguishes and records:

- the SHA-256 of the caller's raw tracking bytes;
- the SHA-256 of the canonical five-field tracking output;
- the SHA-256 and byte count of the caller's unchanged sidecar bytes;
- the SHA-256 and byte count of the rebuilt output sidecar.

The rebuilt sidecar binds `tracking_jsonl_sha256` to the actual canonical `tracking.jsonl` bytes. The staged pair is reloaded through `load_camera_inputs()` before the fresh output directory is atomically committed. A failed staged reload removes staging and leaves no final output.

### Receipt-first video preparation

The new `camera_collection.py` controller and `build_camera_tracking_pair.py` CLI enforce this order:

1. validate the exact C0 receipt for approved, active, unexpired `camera_development` / `development` / `prepare_session` use;
2. read and validate the anonymous collection, then revalidate its complete participant/session/setup/source-group scope;
3. select exactly one declared `source_video_id` and verify the final output is absent;
4. validate operator-supplied deidentified metadata;
5. only then resolve/stat/hash/open the video, import the optional vision runtime, and create a private raw temporary directory;
6. call the existing shared `run_yolov8_bytetrack()` with fixed production parameters;
7. bind a temporary sidecar to the raw tracking SHA and call `prepare_authorized_camera_session()`;
8. let the existing staging transaction commit the canonical pair, then clean the raw temporary directory.

The controller fixes `authorization_status=authorized_camera_engineering_smoke`; it has no caller-selected authorization/evidence/validation scope, test fixture, fake runtime, loader, skip, bypass, model, threshold, or maximum-frame input. It produces only `tracking.jsonl`, `media_sidecar.json`, and `preparation_summary.json` for later QC.

## Sidecar provenance

- `source_video_id`, `source_group_id`, `device_id`, `setup_id`, and `stream_epoch`: the unique validated collection source;
- width, height, FPS, duration, and `source_sha256`: the opened authorized video;
- `tracking_jsonl_sha256`: the actual output tracking bytes;
- `media_ref`: operator-supplied deidentified relative POSIX reference;
- `camera_motion_state` and `deidentification_status`: required operator facts, never inferred;
- `capture_started_at` and `timezone`: both provided or both null;
- detector/tracker names, versions, and fixed execution parameters: sidecar plus atomically committed preparation summary.

No absolute video path, credential, authorization token, or direct identity is serialized.

## Operator handoff

Use [OPERATOR_CHECKLIST.md](OPERATOR_CHECKLIST.md). The two repository templates are intentionally invalid examples only:

- [authorization receipt template](templates/authorization_receipt.template.md)
- [anonymous collection template](templates/collection.template.md)

The execution AI must validate owner-provided facts; it must not create, sign, approve, or fill them. Real receipts, manifests, video, C1 outputs, and authorization evidence stay outside Git.

## Evidence boundary

All automated success paths used deterministic bytes under pytest `tmp_path` plus monkeypatched video metadata and tracker runtime. Those outputs were consumed by tests and were not retained as evidence. Passing tests establish software ordering, schema, hash, cleanup, and refusal contracts only. They do not establish target-camera, human, older-adult, product, clinical, accuracy, F1, recall, FAR, calibration, episode, or deployment-latency performance.

See [VERIFICATION.md](VERIFICATION.md) for commands and results.
