# M0-CAM-C01-F C1 provenance and final-binding hardening

## Initial completion state

```text
status_at_initial_completion=wandering_m0cam_c0_c1_handoff_hardened_waiting_owner_inputs
evidence_scope=synthetic_schema_contract_only
c01_f_complete=true
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

C01-F implemented its original bounded fixes after C01-PREP. It does not supply an authorization receipt, validate an owner’s real-world claims, read human or authorized camera media, create a real C1 artifact, run Camera QC or model inference, or start M0-CAM-D.

## Post-completion audit note

The original C01-F scope and test evidence remain valid, but a later synthetic red-team audit found four small gaps that must close before real C0/C1 use: caller-defined and DrvFS/symlink-aliased project-root boundaries, external path resolution before receipt validation, pair-only media SHA described as controller-observed/path-bound, and detector/tracker backend/version strings that can still carry path-like or sensitive text. These findings are routed to bounded `M0-CAM-C01-F2`.

The former `wandering_m0cam_c01_f2_integrity_fix_required / c01_f2_complete=false` value is a historical audit state. [C01-F2](../wandering_camera_c01_f2_v1/README.md) has since completed; current operational status is maintained in the task table. This report alone is not final handoff-ready evidence. `evidence_scope=synthetic_schema_contract_only`, `C0=false`, `C1=false`, `authorized_camera_data_consumed=false`, and `m0cam_d_started=false` remain unchanged.

## Closed public provenance paths

- `prepare_camera_input_pair()` is now a synthetic/schema-only pair normalizer. Its public signature has no context, provenance, authorization-status, evidence-scope, hook, or fake-runtime input. It rejects an authorized sidecar before reading tracking or creating output.
- `prepare_authorized_camera_session()` has no caller-supplied provenance input. Pair-only preparation always records `media_opened=false`, `detector_run=false`, `tracker_run=false`, `camera_qc_run=false`, `model_inference_run=false`, and `m0cam_d_started=false`.
- Passing the removed `preparation_context` keyword raises `TypeError`; there is no deprecated compatibility path.

## Private video provenance

The receipt-gated video controller is the only video path. It submits through an underscore-prefixed internal function and an exact frozen dataclass rather than an arbitrary mapping. The private carrier contains only controller schema, selected source, observed source SHA-256, fixed detector/tracker identities and parameters, raw observation count, and probed video metadata.

The internal validator rejects:

- controller or controller-schema drift;
- selected-source or observed source-hash mismatch;
- detector or tracker identity/version mismatch;
- non-fixed execution parameters;
- raw tracker count versus normalized tracking count mismatch;
- sidecar versus probed video metadata mismatch;
- extra provenance constructor fields.

`camera_collection.__all__` exports only the production controller and its error type. Internal video metadata/runtime carriers are underscore-prefixed and are not CLI or public exports.

## Atomic final bindings

`tracking.jsonl`, `media_sidecar.json`, and `preparation_summary.json` are built in the same fresh staging directory, reloaded, compared with their expected canonical bytes, and committed by one directory rename. A failed reload or comparison removes staging and leaves no final output.

Authorized summaries persist only repository-safe bindings:

- `authorization_binding`: receipt ID, canonical receipt SHA-256, approved/active validation state, purpose, dataset role, operation, validity interval, and aggregate scope counts;
- `collection_binding`: anonymous collection ID, canonical validated collection SHA-256, dataset role, and receipt reference;
- `source_binding`: anonymous video/session/group/setup/device/epoch scope and the source SHA fields implemented in this initial stage. The later audit found that pair-only output incorrectly describes the sidecar-declared SHA as observed and path-bound; F2 must separate declaration from controller observation before real use;
- canonical tracking and output-sidecar hashes and byte counts;
- for the video controller only, its schema, fixed detector/tracker execution facts, raw count, and non-path video metadata.

The final summary contains no raw receipt or collection, allowed-ID lists, direct identity, absolute media path, token/header/credential, live URL, or arbitrary caller extras. Acceptance must read the on-disk summary; CLI stdout is not an independent trust root.

## Portable input and path boundary

Before protected tracking reads or video/hash/runtime/tracker/temp/staging/final work:

- `media_ref` rejects control characters, overlength values, URLs/credentials, Windows drive or drive-relative paths, UNC/backslash paths, POSIX absolute paths, and traversal;
- timezone accepts only `UTC` or an existing safe IANA zone and rejects control/path/URL/credential/nonexistent values;
- production receipt, collection, tracking, sidecar, video, and output resolved paths must remain outside the project root, including `..` and symlink resolutions back into the repository.

The video SHA-256 is recomputed after the shared tracker returns. Any mutation, including a same-length byte change, cleans raw temporary files and produces no staging or final directory.

## Trust boundary and next step

The code validates the exact receipt/collection structures and controller-observable software facts. It does not implement PKI, signatures, owner registries, remote allowlists, or prove consent, deidentification, camera placement, participant identity, or the owner-declared mapping between `source_video_id` and an external path.

C01-F2 subsequently completed the later audit findings. Current work is maintained only in the task table; before M0-CAM-D, the project still requires the remaining software portability gate plus real `C0=true` and `C1=true` owner inputs.

See [VERIFICATION.md](VERIFICATION.md) for commands and evidence boundaries.
