# W5D-02 verification

```json
{
  "automatic": {
    "B01": {
      "automatic_binary": {
        "accuracy": 0.9090909090909091,
        "class_order": [
          "direct_or_non_wandering",
          "wandering_like"
        ],
        "correct": 30,
        "macro_f1": 0.8891377379619261,
        "per_class": {
          "direct_or_non_wandering": {
            "f1": 0.9361702127659575,
            "precision": 0.88,
            "predicted_count": 25,
            "recall": 1.0,
            "support": 22,
            "true_positive": 22
          },
          "wandering_like": {
            "f1": 0.8421052631578948,
            "precision": 1.0,
            "predicted_count": 8,
            "recall": 0.7272727272727273,
            "support": 11,
            "true_positive": 8
          }
        },
        "support": 33
      },
      "automatic_binary_population": "automatic_boundary_matches_prediction_ready",
      "known_episode_support": 52,
      "pipeline_miss_count": 19,
      "pipeline_miss_reasons": {
        "matched_proposal_not_prediction_ready": 1,
        "no_automatic_boundary_match": 18
      },
      "prediction_coverage_gate_satisfied": true,
      "ready_uncertain_prediction_coverage": 0.8846153846153846,
      "ready_uncertain_prediction_overlap_count": 46
    },
    "B02": {
      "automatic_binary": {
        "accuracy": 1.0,
        "class_order": [
          "direct_or_non_wandering",
          "wandering_like"
        ],
        "correct": 18,
        "macro_f1": 1.0,
        "per_class": {
          "direct_or_non_wandering": {
            "f1": 1.0,
            "precision": 1.0,
            "predicted_count": 9,
            "recall": 1.0,
            "support": 9,
            "true_positive": 9
          },
          "wandering_like": {
            "f1": 1.0,
            "precision": 1.0,
            "predicted_count": 9,
            "recall": 1.0,
            "support": 9,
            "true_positive": 9
          }
        },
        "support": 18
      },
      "automatic_binary_population": "automatic_boundary_matches_prediction_ready",
      "known_episode_support": 20,
      "pipeline_miss_count": 2,
      "pipeline_miss_reasons": {
        "no_automatic_boundary_match": 2
      },
      "prediction_coverage_gate_satisfied": true,
      "ready_uncertain_prediction_coverage": 0.9,
      "ready_uncertain_prediction_overlap_count": 18
    },
    "pooled": {
      "automatic_binary": {
        "accuracy": 0.9411764705882353,
        "class_order": [
          "direct_or_non_wandering",
          "wandering_like"
        ],
        "correct": 48,
        "macro_f1": 0.9363825363825364,
        "per_class": {
          "direct_or_non_wandering": {
            "f1": 0.9538461538461539,
            "precision": 0.9117647058823529,
            "predicted_count": 34,
            "recall": 1.0,
            "support": 31,
            "true_positive": 31
          },
          "wandering_like": {
            "f1": 0.9189189189189189,
            "precision": 1.0,
            "predicted_count": 17,
            "recall": 0.85,
            "support": 20,
            "true_positive": 17
          }
        },
        "support": 51
      },
      "automatic_binary_population": "automatic_boundary_matches_prediction_ready",
      "known_episode_support": 72,
      "pipeline_miss_count": 21,
      "pipeline_miss_reasons": {
        "matched_proposal_not_prediction_ready": 1,
        "no_automatic_boundary_match": 20
      },
      "prediction_coverage_gate_satisfied": true,
      "ready_uncertain_prediction_coverage": 0.8888888888888888,
      "ready_uncertain_prediction_overlap_count": 64
    }
  },
  "binary_adjustment_decision": {
    "decision": "retain_fixed_primary_and_threshold",
    "model_retrained": false,
    "reason": "oracle evidence is reported separately; automatic weakness is boundary/QC-driven unless the current oracle result is insufficient",
    "threshold": 0.5
  },
  "config_sha256": "49b16834e8e0ba8335640012551492b4f692ccd106b1b2c1047d59f51ac4bd29",
  "development_video_count": 48,
  "identity": {
    "candidate_id": "topowander-m0s-seed20260731-epoch0005",
    "development_index_sha256": "bd29c218cdd78c054955c7ac51ab195003a9a5533d84c2efc7f9cc3dd01b82f8",
    "model_state_sha256": "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031",
    "segmenter_profile_id": "closing-s008-d04",
    "segmenter_profile_sha256": "ac69ceab548d263fd51f09ee36121055ea373d995353ae5f8050d33d3d5948aa"
  },
  "oracle": {
    "binary": {
      "accuracy": 0.8571428571428571,
      "class_order": [
        "direct_or_non_wandering",
        "wandering_like"
      ],
      "correct": 66,
      "macro_f1": 0.8756286266924564,
      "per_class": {
        "direct_or_non_wandering": {
          "f1": 0.9148936170212766,
          "precision": 0.8958333333333334,
          "predicted_count": 48,
          "recall": 0.9347826086956522,
          "support": 46,
          "true_positive": 43
        },
        "wandering_like": {
          "f1": 0.8363636363636364,
          "precision": 0.9583333333333334,
          "predicted_count": 24,
          "recall": 0.7419354838709677,
          "support": 31,
          "true_positive": 23
        }
      },
      "pipeline_miss_count": 5,
      "support": 77
    },
    "four_class": {
      "accuracy": 0.6883116883116883,
      "class_order": [
        "direct",
        "pacing",
        "lapping",
        "random"
      ],
      "correct": 53,
      "macro_f1": 0.4689980037976532,
      "per_class": {
        "direct": {
          "f1": 0.9148936170212766,
          "precision": 0.8958333333333334,
          "predicted_count": 48,
          "recall": 0.9347826086956522,
          "support": 46,
          "true_positive": 43
        },
        "lapping": {
          "f1": 0.4347826086956522,
          "precision": 0.2777777777777778,
          "predicted_count": 18,
          "recall": 1.0,
          "support": 5,
          "true_positive": 5
        },
        "pacing": {
          "f1": 0.0,
          "precision": 0.0,
          "predicted_count": 1,
          "recall": 0.0,
          "support": 12,
          "true_positive": 0
        },
        "random": {
          "f1": 0.5263157894736842,
          "precision": 1.0,
          "predicted_count": 5,
          "recall": 0.35714285714285715,
          "support": 14,
          "true_positive": 5
        }
      },
      "pipeline_miss_count": 5,
      "support": 77
    },
    "summary": {
      "alert_metrics_available": false,
      "automatic_boundary_inference": false,
      "batch_count": 48,
      "binary_decision_threshold": 0.5,
      "candidate_id": "topowander-m0s-seed20260731-epoch0005",
      "candidate_manifest_sha256": "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7",
      "class_order": [
        "direct",
        "pacing",
        "lapping",
        "random"
      ],
      "evaluation_name": "oracle-boundary shape classification",
      "failure_count": 25,
      "legacy_40_second_diagnostic_included": false,
      "model_state_sha256": "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031",
      "models_retrained": false,
      "prediction_status_counts_all_truth": {
        "abstention": 0,
        "boundary_uncertain": 0,
        "inference_error": 0,
        "ready": 72,
        "unavailable": 6
      },
      "prediction_status_counts_for_shape_eligible": {
        "abstention": 0,
        "boundary_uncertain": 0,
        "inference_error": 0,
        "ready": 72,
        "unavailable": 5
      },
      "prediction_truth_joined_after_inference": true,
      "probability_calibrated": false,
      "schema_version": "wandering-camera-episode-eval-summary-v1",
      "shape_eligible_count": 77,
      "shape_truth_status_counts": {
        "eligible": 77,
        "excluded": 0,
        "unknown": 1
      },
      "status": "wandering_m0cam_ep1b_oracle_boundary_evaluated",
      "study_design": "descriptive_development_pilot",
      "truth_count": 78,
      "truth_labels_consumed_by_inference": false
    }
  },
  "pipeline_id": "w5d02-b01-b02-automatic-episode-shape-v1",
  "proposal_result_count": 129,
  "run_summary": {
    "degraded_components": [
      "automatic_boundary_segmenter"
    ],
    "failure_counts": {
      "automatic_pipeline_miss": 21,
      "oracle_pipeline_miss": 0,
      "prediction_inference_error": 0,
      "prediction_unavailable": 34,
      "proposal_rejected_by_qc": 24
    },
    "finished_at": "2026-08-18T04:01:12.990069Z",
    "identity": {
      "config_id": "w5d02-b01-b02-automatic-episode-shape-v1",
      "config_sha256": "49b16834e8e0ba8335640012551492b4f692ccd106b1b2c1047d59f51ac4bd29",
      "model_id": "topowander-m0s-seed20260731-epoch0005",
      "model_sha256": "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031",
      "policy_id": "closing-s008-d04",
      "policy_sha256": "ac69ceab548d263fd51f09ee36121055ea373d995353ae5f8050d33d3d5948aa"
    },
    "input_counts": {
      "B01_videos": 36,
      "B02_videos": 12,
      "development_videos": 48,
      "known_episode_support": 72
    },
    "module": "mental_health",
    "output_counts": {
      "binary_prediction_ready": 95,
      "episode_results": 129,
      "error": 0,
      "four_class_prediction_ready": 95,
      "ready": 14,
      "unavailable": 34,
      "uncertain": 81
    },
    "person_id": "P-OFFICE-01",
    "quality_flags": [
      "automatic_boundary_not_human_accepted",
      "development_only",
      "segmenter_selection_gate_not_satisfied"
    ],
    "run_id": "w5d02-b01-b02-20260818-v1",
    "schema_version": "wandering-handoff-run-summary-v1",
    "session_id": null,
    "source_refs": [
      {
        "artifact_path": "/mnt/c/Users/lenovo/Desktop/心理算法/reports/mental_health/wandering_camera_5d_contract_v3/development_index.jsonl",
        "ref_id": "w5d00-development-index",
        "ref_type": "development_index",
        "sha256": "bd29c218cdd78c054955c7ac51ab195003a9a5533d84c2efc7f9cc3dd01b82f8"
      },
      {
        "artifact_path": "reports/mental_health/wandering_camera_segmenter_search_v2/selected_segmenter_profile.yaml",
        "ref_id": "closing-s008-d04",
        "ref_type": "segmenter_profile",
        "sha256": "ac69ceab548d263fd51f09ee36121055ea373d995353ae5f8050d33d3d5948aa"
      }
    ],
    "source_video_id": null,
    "started_at": "2026-08-18T04:00:27.044939Z",
    "status": "uncertain"
  }
}
```
