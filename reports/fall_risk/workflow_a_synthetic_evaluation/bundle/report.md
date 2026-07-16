# Fall-risk event evaluation

Protocol: `fall-event-eval-v1` (`development_provisional`)
Evaluation config SHA-256: `5254a066dcc3316bc7bb6f2076ff570eac323cb6a5f18cc999cd39705645330e`

## Metrics

| Metric | Value |
|---|---:|
| `precision` | `1.0` |
| `recall` | `1.0` |
| `f1` | `1.0` |
| `event_pr_auc` | `1.0` |
| `tp` | `1` |
| `fp` | `0` |
| `fn` | `0` |
| `false_discovery_rate` | `0.0` |
| `miss_rate` | `0.0` |
| `fp_per_camera_hour` | `None` |
| `mean_boundary_iou` | `0.904761904762` |
| `mean_detection_latency_sec` | `0.1` |
| `mean_lead_time_sec` | `None` |

## Metric definitions

- `Precision = TP / (TP + FP)`; `Recall = TP / (TP + FN)`.
- `false_discovery_rate = FP / (TP + FP)`; traditional FPR is only emitted when an explicit negative-observation denominator exists.
- `lead_time = reference_event_start - alert_time`; positive values are early warnings.
- Fall detection latency and near-fall onset detection latency use `alert_time - reference onset`; recovery is reported with recovery-specific metrics. A provisional input may fall back to prediction `onset_time`, and the fallback count is reported.
- `fp_per_camera_hour` uses only manifest rows explicitly marked `continuous_monitoring_eligible=true`.
- `event_pr_auc` uses step-wise average precision (`delta_recall * precision`), scans every distinct score threshold, and reruns the same one-to-one matcher.

## Matching protocol

```json
{
  "duplicate_alert_policy": "unmatched_is_false_positive",
  "event_merge_enabled": false,
  "event_merge_gap_sec": 1.0,
  "event_reset_gap_sec": 2.0,
  "event_type_compatibility": {
    "fall": [
      "fall"
    ],
    "near_fall": [
      "near_fall"
    ],
    "recovery": [
      "recovery"
    ]
  },
  "iou_threshold": 0.5,
  "matching_operator": "iou_or_onset",
  "one_to_one_assignment": "maximum_cardinality_deterministic",
  "onset_tolerance_sec": 0.5,
  "pr_auc_method": "average_precision_step",
  "reset_rule_application": "audit_only_postprocessed_predictions",
  "search_window_after_sec": 2.0,
  "search_window_before_sec": 2.0
}
```

## Interpretation

This run uses a provisional development protocol unless the protocol status is explicitly frozen. Undefined denominators are reported as `null`; short event clips are not counted as continuous camera hours.

## Reproducibility

- `code_version`: `5ab26b1ede4bc7bea25347183d72c21c5e715f3c`
- `environment_name`: `eldercare-ai`
- `environment_package_count`: `110`
- `environment_packages_sha256`: `7bbc6b88e853dfe2d6776d7763c48f5f9e1ca03aa37d343738ed825ce84d0e4a`
- `evaluation_config_file_sha256`: `b898915107ae6cc5e54509988ddfb102aeff4dbbd8c8eded700277bc3e15eb60`
- `evaluation_config_normalized_sha256`: `5254a066dcc3316bc7bb6f2076ff570eac323cb6a5f18cc999cd39705645330e`
- `evaluation_implementation_sha256`: `22bd1dbde51f56ae36db68640429449d7b23b6e1a15c6b11cc001eb7fa00015e`
- `git_dirty`: `True`
- `ground_truth_hash`: `fe81fa0a387baf25f29671678e089704e4db78f154df4cd39248f3adafa86513`
- `label_version`: `synthetic-labels-v1`
- `manifest_hash`: `d13f70410d0cd6c1e46ff92d65efb06d0dd8689df82d6c7b30a5acc08e574baf`
- `package_source`: `src/elderly_monitoring/__init__.py`
- `partition`: `validation`
- `platform`: `macOS-26.5.1-arm64-arm-64bit`
- `prediction_hash`: `0661d70ce40d1cb5963abda1542fd25fb42851e01a02ddff171348fd43f10290`
- `python_executable`: `python`
- `python_version`: `3.11.15`
- `reproduction_command`: `conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_events.py --ground-truth reports/fall_risk/workflow_a_synthetic_evaluation/input/ground_truth.jsonl --predictions reports/fall_risk/workflow_a_synthetic_evaluation/input/predictions.jsonl --manifest reports/fall_risk/workflow_a_synthetic_evaluation/input/manifest.jsonl --split reports/fall_risk/workflow_a_synthetic_evaluation/input/split.json --assignments reports/fall_risk/workflow_a_synthetic_evaluation/input/assignments.jsonl --partition validation --config configs/evaluation/fall_event_v1.provisional.yaml --output-dir reports/fall_risk/workflow_a_synthetic_evaluation/bundle --label-version synthetic-labels-v1 --allow-provisional --overwrite`
- `split_id`: `fall_event_v1_synthetic:sha256:5bb3684c004f73394b321a77c11bf3cba7c86606602801e391da1e2c618a505d`
- `split_sha256`: `5bb3684c004f73394b321a77c11bf3cba7c86606602801e391da1e2c618a505d`
- `tracked_diff_sha256`: `c38ad3a6d50fb8a560100f60eb71bb8d0b0fc9268605bb1144841c1ec60c2df8`
- `validation_report_hash`: `None`
