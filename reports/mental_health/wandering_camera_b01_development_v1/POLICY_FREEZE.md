# M0-CAM B01 development policy freeze

Freeze date: 2026-08-17

This record was written after B01 development evaluation and before the first
model/proposal run over the 11 frozen B02 held-out videos listed in
`wandering_camera_b01_b02_intake_v1/COHORT_FREEZE.md`.

2026-08-17 后续说明：本文件继续作为旧 proposed-only v1 在首次 B02 运行前的
冻结记录，内容不回写。负责人在查看 v1 结果后另行明确授权复用 B02 建立
candidate-inclusive S2A v2；producer/matching/model/threshold/classes 均未改，但 v2
B02 结果固定为 post-hoc development/exploratory，不是第二次 holdout。当前状态见
[v2 报告](../wandering_camera_b02_candidate_inclusive_development_v1/README.md)。

```text
policy_freeze_status=frozen_before_b02_heldout_inference
development_cohort=B01 usable videos + previously exposed B02-0008 diagnostic
heldout_cohort=11 unexposed B02 videos
producer_config_id=m0cam-ep2a-s0-b01-development-frozen-v1
matching_policy_id=m0cam-ep2a-s1b-b01-development-frozen-v1
frozen_model_modified=false
binary_threshold_modified=false
class_definition_modified=false
human_truth_modified_from_prediction=false
```

## Frozen S0 policy

```text
bucket_seconds=0.5
minimum_track_confidence=0.25
maximum_internal_gap_buckets=3
movement_start_min_buckets=5
movement_start_min_displacement_body_heights=0.25
stationary_max_displacement_body_heights=0.08
stationary_dwell_candidate_seconds=15.0
minimum_episode_duration_seconds=2.0
```

Only `movement_start_min_buckets` changed from the original S0 development
baseline (`3 -> 5`). This extends the displacement observation window from one
second to two seconds without changing the `0.25` body-height displacement
threshold. The state machine, technical hard breaks, soft-close semantics and
proposal status rules are unchanged.

## Frozen matching policy

```text
primary_view=all_locomotion_candidates (proposed + uncertain)
conditional_view=proposed_only_conditional
minimum_temporal_iou=0.25
maximum_onset_delta_sec=10.0
optimization_order=maximum_match_count, maximum_total_temporal_iou,
                   minimum_total_onset_offset_delta,
                   stable_prediction_annotation_id
```

Matching values were not changed after synthetic S1A implementation. B01 did
not provide repeated evidence that a matching change was safer than retaining
the preregistered policy. Diagnostic failure thresholds remain unvalidated and
do not select matches.

## B01 selection evidence

The full B01 development set contains 36 usable videos, 58 ready human
locomotion boundaries and 57 shape-eligible episodes. Track 56 in
`VID-B01-0036_out-of-frame-R01.mp4` uses the owner adjudication
`purpose_context=unknown + purpose_evidence=unknown` in a derived reviewed XML;
the raw XML is unchanged.

Full-B01 movement-start comparison with 15-second dwell:

| start buckets | candidates | matches | precision | recall | F1 | mean tIoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 23 | 16 | 0.695652 | 0.275862 | 0.395062 | 0.730270 |
| 5 | 49 | 32 | 0.653061 | 0.551724 | 0.598131 | 0.862593 |

Full-B01 dwell comparison with five-bucket opening:

| dwell (s) | candidates | matches | P | R | F1 | merge | split | fragmentation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 74 | 37 | 0.500000 | 0.637931 | 0.560606 | 5 | 13 | 5 |
| 8 | 59 | 34 | 0.576271 | 0.586207 | 0.581197 | 7 | 8 | 1 |
| 10 | 53 | 33 | 0.622642 | 0.568966 | 0.594595 | 6 | 4 | 1 |
| 15 | 49 | 32 | 0.653061 | 0.551724 | 0.598131 | 7 | 2 | 1 |

Shortening dwell increased recall but created substantially more split and
fragmentation candidates and did not improve F1. The 15-second protocol anchor
was therefore retained.

The frozen B01 result is descriptive development evidence:

```text
ready_truth=58
all_locomotion_candidates=49
matched=32
precision=0.653061
recall=0.551724
f1=0.598131
mean_matched_tiou=0.862593
mean_onset_absolute_error_sec=1.055208
mean_offset_absolute_error_sec=2.195833
```

It is not B02 held-out performance. It does not make an S0 proposal an accepted
boundary, and it does not validate alert, risk, FAR or clinical behavior.

## Frozen identities

```text
wandering_camera_episode_boundary_proposal_v1.yaml
  sha256=66828e0f5fe8c960146ca5781a531431b8e22b689adaf3f7c500e2f946c4b99c

wandering_camera_episode_boundary_eval_v1.yaml
  sha256=60fe0f4a748885be039f02abaf8ba062773840545b69f5c69507a8d06d535fd3

wandering_camera_episode_proposal_shape_v1.yaml
  sha256=4a1535a12b5eb7585b943a53d0e85cb7140583289fda2b68067052b962b1658a

topowander_m0r_candidate_v3/candidate_manifest.json
  sha256=3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7

B01 frozen S1A metrics.json
  sha256=6c5093e63c2db2a248a94f313901330e098270182b0d8546427546f820f326fa
```

## Held-out rule

The 11 B02 held-out videos may now be run once with these exact producer,
matching, frozen-shape and `0.5` binary policies. Their result must be reported
without changing this version. B02-0008 remains an exposed development
diagnostic and is excluded from held-out metrics. Because B01/B02 share one
participant and camera setup, this is a frozen video-level holdout only; it is
not cross-participant, cross-setup, fully independent-session or sealed-camera
evidence.
