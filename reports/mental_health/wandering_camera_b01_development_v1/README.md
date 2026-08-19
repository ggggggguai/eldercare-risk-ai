# M0-CAM B01 development / B02 historical holdout and v2 addendum

Date: 2026-08-17

Status:

```text
m0cam_ep1b_implementation_status=completed
m0cam_ep1b_human_truth_status=reviewed_manual_cvat
m0cam_ep1b_evaluation_status=completed_descriptive_pilot
m0cam_ep1b_status=completed

m0cam_ep2a_s1b_policy_status=frozen_on_b01_development
m0cam_ep2a_s1b_video_holdout_status=completed_once_on_b02
m0cam_ep2a_s2a_v1_video_holdout_status=completed_once_on_b02_historical
m0cam_ep2a_s2a_v2_status=completed_on_reused_b02_post_hoc_development
m0cam_ep2a_automatic_shape_coverage_status=17_of_18_on_reused_b02_post_hoc_development
m0cam_ep2a_status=not_completed
```

Evidence scope: `authorized_labeled_development_evaluated`, a historical
`frozen_unexposed_video_holdout` v1 snapshot, and a later
`post_hoc_development_exploratory_not_held_out` v2 addendum. B02 is the same
participant and camera setup as B01. This report is not cross-participant,
cross-setup, sealed-camera, alert, FAR or clinical evidence.

## Cohort and provenance

- B01: 36 usable development videos, participant `P-OFFICE-01`, session
  `S-20260814`, setup `C6C-OFFICE-01`.
- B02 historical v1 holdout: 11 videos that were unexposed when the old
  proposed-only policy was first scored, the same participant/setup, session
  `S-20260816`. They were later explicitly reused for v2 development and are
  not held-out for v2.
- `VID-B02-0008_window-direct-pacing-R01.mp4` was exposed before cohort freeze
  and is excluded from held-out metrics. Both frame-check videos remain
  excluded.
- The owner confirmed that first-pass CVAT annotation was blind to S0, S2A,
  S2B and frozen-model outputs.
- Old P01/L01/R01/H01/H04/H05 AI preannotations are excluded from every result
  in this report.
- B01 track 56 uses the owner adjudication `unknown + unknown` in a derived
  reviewed XML. The raw XML was not changed.

The cohort declaration is in
[COHORT_FREEZE.md](../wandering_camera_b01_b02_intake_v1/COHORT_FREEZE.md).
The exact pre-held-out policy record is in [POLICY_FREEZE.md](POLICY_FREEZE.md).

## Frozen policy

The only S0 behavioral change was
`movement_start_min_buckets: 3 -> 5`; the displacement threshold remains
`0.25` body heights. The 15-second stationary dwell, technical hard breaks,
proposal status rules and S1A matching values were retained. Producer and
matching IDs are:

```text
m0cam-ep2a-s0-b01-development-frozen-v1
m0cam-ep2a-s1b-b01-development-frozen-v1
```

The 5-bucket policy improved full-B01 boundary F1 from `0.395062` to
`0.598131`; 5/8/10-second dwell alternatives did not beat the retained
15-second dwell and produced more split/fragmentation candidates. B02 was not
used for this selection and the frozen version was not changed after B02.

## EP1B oracle-boundary shape

The binary result comes directly from the frozen independent binary head at
the unchanged `0.5` threshold. Four-class output is subtype diagnostic.

| cohort | eligible / ready | binary accuracy | binary macro-F1 | four-class accuracy | subtype macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B01 development, all eligible | 57 / 52 | 0.807018 | 0.817375 | 0.631579 | 0.316723 |
| B01 ready-only conditional | 52 / 52 | 0.884615 | 0.864583 | 0.692308 | 0.322917 |
| B02 frozen-video holdout, all eligible | 18 / 18 | 1.000000 | 1.000000 | 0.888889 | 0.666667 |
| B02 ready-only conditional | 18 / 18 | 1.000000 | 1.000000 | 0.888889 | 0.666667 |

B01 D/P/L/R support is `36/9/3/9`; B02 support is `9/2/2/5`.
B02's two subtype errors are both pacing predicted as lapping. Their binary
predictions are correct, matching the known camera projection collapse rather
than motivating label/model/threshold changes.

EP1B is complete only as a small manual-CVAT, oracle-boundary descriptive
camera shape evaluation. Its score cannot be used as automatic-boundary or
alert performance.

## EP2A boundary

The primary boundary view includes both `proposed` and `uncertain`; the
proposed-only view remains conditional.

| cohort | TP / candidates / truth | precision | recall | F1 | mean tIoU | onset abs mean (s) | offset abs mean (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 development | 32 / 49 / 58 | 0.653061 | 0.551724 | 0.598131 | 0.862593 | 1.055208 | 2.195833 |
| B02 frozen-video holdout | 17 / 17 / 18 | 1.000000 | 0.944444 | 0.971429 | 0.731254 | 0.996078 | 1.594118 |
| B02 proposed-only conditional | 2 / 2 / 18 | 1.000000 | 0.111111 | 0.200000 | 0.597823 | 0.600000 | 1.116667 |

The single B02 primary-view miss is random episode
`vid-b02-0007-cvat-11`, covered only by a `rejected_by_qc` interval. B02 also
contains three matched low-tIoU diagnostics and four large-offset diagnostics.
These observations remain frozen in the v1 snapshot. On 2026-08-17 the owner
explicitly authorized using the already inspected B02 result to establish a
new, separately identified S2A forwarding policy; that later result is
post-hoc development rather than held-out.

## Historical automatic proposal + frozen shape v1

The historical v1 S2A preserves the original S0 endpoint and only forwards `proposed`. An
`uncertain` boundary is skipped as `boundary_uncertain`; `rejected_by_qc` is
skipped as `unavailable`. All non-ready cases remain pipeline misses in the
all-eligible automatic-shape result.

| cohort | boundary matched / eligible | shape ready / eligible | binary accuracy | binary macro-F1 | subtype accuracy | subtype macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 development | 32 / 57 | 1 / 57 | 0.017544 | 0.027027 | 0.017544 | 0.013514 |
| B02 frozen-video holdout | 17 / 18 | 2 / 18 | 0.111111 | 0.181818 | 0.111111 | 0.090909 |

The two historical B02 v1 ready predictions are direct and both correct. Ready-only full
macro-F1 is `not_computable` because no wandering-like/subtype-positive sample
was allowed through `proposed`. This is a coverage failure, not evidence that
automatic shape classification is accurate. The strong oracle result and the
strong all-candidate boundary result must remain separate from this low
end-to-end shape coverage.

## Candidate-inclusive S2A v2 addendum

The owner authorized reusing B02 as development/exploratory data after the v1
result had been inspected. Policy
`m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2` keeps every uncertain
proposal's original endpoint, status, reasons and manual-review requirement,
but permits it to enter episode QC and frozen shape inference. It does not
promote uncertain to proposed or accepted boundary. `rejected_by_qc` remains
unavailable and skipped.

| view | support | ready | binary accuracy | binary macro-F1 | subtype accuracy | subtype macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all shape eligible | 18 | 17 | 0.944444 | 0.970588 | 0.833333 | 0.638889 |
| candidate-ready conditional | 17 | 17 | 1.000000 | 1.000000 | 0.882353 | 0.666667 |
| uncertain-only diagnostic | 15 | 15 | 1.000000 | 1.000000 | 0.866667 | 0.666667 |

The 21 S0 rows remain `2 proposed + 15 uncertain + 4 rejected_by_qc`;
forward/skipped is `17/4`. Ready coverage increased from historical `2/18` to
`17/18`. All 17 matched candidates are binary-correct; the one all-eligible
pipeline miss is still the rejected-by-QC boundary miss. Endpoint and binding
mismatches are `0/0`; model, threshold, classes, producer, matching and human
truth were not changed. Full evidence is in the
[candidate-inclusive v2 report](../wandering_camera_b02_candidate_inclusive_development_v1/README.md).

## Current conclusion

EP1B's manual-truth completion gate is closed. The old `2/18` forwarding
coverage blocker is also closed by v2 on reused B02. EP2A remains
`not_completed` because v2 was designed after inspecting B02 and evaluated on
the same one-participant, one-setup cohort. It has no independent
participant/setup validation.

No further v2 change should be selected on B01/B02. The next evidence step is
to run v2 unchanged on a predeclared new participant and/or camera setup. This
report does not authorize EP2B, EP2C, EP3, alert, risk or `AlgorithmEvent`
work.

## Local machine outputs

The non-overwriting machine bundles remain in ignored local paths:

```text
tmp/m0cam_b01_frozen_evaluation_20260817_v2/
tmp/m0cam_b02_frozen_video_heldout_20260817_v1/
tmp/m0cam_b02_candidate_inclusive_shape_20260817_v1/
```

Key metric SHA-256 values:

```text
B01 EP1B metrics       7e37bc41a43b0fb3e3215991d8ea038058b26b44368ec227042da59213ff2031
B01 automatic shape   687898de8de8577f9a3bf1211464195107afa88c270ec9eb8c5cd1f4ab9cc10e
B02 S1A metrics       8d2084e6d48b555d1c21d90b70ac37a90fa24e1c6ab8202dfe209e92f963bd7c
B02 EP1B metrics       066b4944e6dafebc5ec8c0263677b39f2c5d6602aaf153c0bf50c206892161d8
B02 automatic shape   4fe44b67aa87ee3a5fcb51038e74d785e2055e989bf127b02489bb55c35e72eb
B02 v2 metrics         2e8f5e966fdb03ac562b50c047504b825f21f41556b94cebea5beb95e6ac9866
```

Verification commands and intermediate-output notes are in
[VERIFICATION.md](VERIFICATION.md).
