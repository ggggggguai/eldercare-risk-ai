# M0-CAM B01/B02 verification

Date: 2026-08-17

## Environment

All Python and pytest commands used WSL Ubuntu 22.04 and:

```text
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python ...
```

`pip show elderly-monitoring-algorithms` reported editable project location:

```text
/mnt/c/Users/lenovo/Desktop/心理算法
```

CUDA initialization reported that the installed driver was older than the
PyTorch runtime. Real video tracking and frozen model inference completed on
CPU. No shared dependency, `environment.yml`, model, threshold or class was
changed.

## Tests

Focused S0/S1A after the policy metadata and evaluator changes:

```text
33 passed in 14.04s
```

Adjacent S0/S1A/S2A/EP1A/EP1B/importer regression:

```text
94 passed in 26.84s
```

The first adjacent run was `93 passed, 1 failed` because the S2A synthetic
fixture still used an obsolete producer ID. Updating only that fixture to the
frozen producer ID closed the failure.

The first full camera run after the policy freeze produced:

```text
609 passed, 2 failed, 1 warning in 176.67s
```

Both failures were S2B synthetic review fixtures carrying the obsolete S0
producer ID. The production review loader correctly failed closed. Updating
only that fixture to `m0cam-ep2a-s0-b01-development-frozen-v1` gave:

```text
S2B focused: 6 passed in 8.64s
full tests/test_wandering_camera_*.py: 611 passed, 1 warning in 165.99s
```

The warning is the existing portability test's intentional duplicate ZIP
entry (`candidate/candidate_manifest.json`).

## Documentation checks

`git diff --check` completed without whitespace errors. Git printed only the
repository's existing LF-to-CRLF working-copy warnings.

A read-only relative-link scan checked 257 links across the updated M0-CAM
entry points, task books and reports. Every M0-CAM link exists. The scan also
reported three pre-existing missing fall-risk JSON links in `docs/README.md`;
`git show HEAD:docs/README.md` confirmed that all three predate this work, so
they were left untouched under the module boundary.

## Real-data runs

B01 development:

```text
36 videos
58 ready boundaries
57 shape-eligible truth
69 total S0 rows = 1 proposed + 48 uncertain + 20 rejected_by_qc
S2A forward/skipped = 1/68
```

B02 frozen-video holdout, run once after `POLICY_FREEZE.md` existed:

```text
11 videos
18 ready / shape-eligible truth
21 total S0 rows = 2 proposed + 15 uncertain + 4 rejected_by_qc
S2A forward/skipped = 2/19
excluded: B02 frame-check and exposed B02-0008
```

This is the historical proposed-only v1 snapshot. On 2026-08-17 the owner
explicitly authorized reusing the inspected B02 data for a new forwarding
policy; the following v2 run is therefore post-hoc development/exploratory,
not a second holdout:

```text
policy=m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2
21 total S0 rows = 2 proposed + 15 uncertain + 4 rejected_by_qc
S2A forward/skipped = 17/4
all-shape-eligible ready = 17/18
candidate-ready binary = 17/17
endpoint/binding mismatch = 0/0
```

Focused S2A/S2B v2 tests were `8 passed in 11.21s`. The new S2A bundle and
evaluation used fresh, non-overwriting directories. See
[v2 verification](../wandering_camera_b02_candidate_inclusive_development_v1/VERIFICATION.md).

Both runs used fresh output roots and rejected overwrite. Raw videos, project
`annotations.xml` and per-video XML were not modified.

## Track 56 adjudication

The B01 runner created a derived reviewed XML and receipt under its fresh local
output. It changed 194 repeated CVAT box attributes for track 56 from
`purpose_evidence=scripted` to the owner-decided `unknown`, while preserving
`purpose_context=unknown`, `observable_pattern=unknown`,
`evaluation_role=uncertain` and `tracking_issue=out_of_frame`.

```text
raw_xml_modified=false
raw_xml_sha256=0e5e2a439158433ed280bd80d380d41279ef55f6fc4cd332aacb8165d6fd5f48
reviewed_xml_sha256=febf6133fd5f635db90d84ecebc2b7ce0a52e317fd8db55f6450793d52414dc1
```

## Preserved failed/intermediate outputs

No output was overwritten or deleted. These failed/intermediate directories
remain intentionally preserved:

```text
tmp/m0cam_b01_movement_scale_analysis_20260817_v1/
tmp/m0cam_b01_stationary_dwell_analysis_20260817_v1/
tmp/m0cam_b01_stationary_dwell_analysis_20260817_v2/
tmp/m0cam_b01_frozen_evaluation_20260817_v1/
```

The dwell v1/v2 attempts stopped before proposal scoring because the fixed
producer contract rejected non-repository configs. The successful isolated
development comparison is v3. The frozen B01 v1 attempt generated proposal
bundles but stopped before evaluation when the S1A loader still required both
development-validation flags to be false; v2 contains the completed result.
