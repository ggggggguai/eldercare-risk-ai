from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.near_fall_self_collected import (
    _build_scf_windows,
    _sha256,
    _write_deterministic_npz,
    _write_json,
    _write_jsonl,
)
from elderly_monitoring.modules.fall_risk.near_fall_training import (
    NEAR_FALL_CHANNELS,
    NEAR_FALL_JOINTS,
    NearFallDatasetConfig,
    apply_normalization,
    fit_normalization_statistics,
)


POSITIVE_ACTIONS = {"C03", "C04", "C05"}
NEGATIVE_ACTIONS = {
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A10",
    "C01",
    "C02",
}
TRAIN_SCENES = {"base", "bed", "block", "dining"}
VALIDATION_SCENES = {"dark"}
CHALLENGE_SCENES = {"hall"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a scene-isolated near-fall dataset from the 115-video batch."
    )
    root = Path("data/evaluations/fall_risk/fall_self_total_20260826_home")
    parser.add_argument("--manifest", type=Path, default=root / "manifest.jsonl")
    parser.add_argument("--actions", type=Path, default=root / "action_labels.jsonl")
    parser.add_argument("--ground-truth", type=Path, default=root / "ground_truth_events.jsonl")
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def partition_for_scene(scene: str) -> str | None:
    normalized = scene.strip().lower()
    if normalized in TRAIN_SCENES:
        return "train"
    if normalized in VALIDATION_SCENES:
        return "validation"
    if normalized in CHALLENGE_SCENES:
        return "challenge"
    return None


def build_candidates(
    action_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for action in action_rows:
        action_id = str(action.get("action_id", ""))
        if action_id not in POSITIVE_ACTIONS | NEGATIVE_ACTIONS:
            continue
        video_id = str(action["video_id"])
        manifest_row = manifest.get(video_id)
        if manifest_row is None:
            continue
        scene = str(action.get("scene") or manifest_row.get("scene_region") or "")
        partition = partition_for_scene(scene)
        if partition is None:
            continue
        label_id = str(action["label_id"])
        candidates.append(
            {
                "candidate_id": f"fall_self_near_fall_{label_id}",
                "video_id": video_id,
                "start_frame": int(action["start_frame"]),
                "end_frame_exclusive": int(action["end_frame"]) + 1,
                "candidate_role": (
                    "positive_candidate"
                    if action_id in POSITIVE_ACTIONS
                    else "hard_negative_candidate"
                ),
                "source_action_id": action_id,
                "hard_negative_type": (
                    None if action_id in POSITIVE_ACTIONS else action.get("action_name")
                ),
                "subject_id": "self_collected_adult_01",
                "source_group_id": f"fall_self_scene_{scene}",
                "sample_group_id": video_id,
                "partition": partition,
                "scene": scene,
            }
        )
    return sorted(
        candidates,
        key=lambda row: (
            str(row["partition"]),
            str(row["video_id"]),
            int(row["start_frame"]),
            str(row["candidate_id"]),
        ),
    )


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.manifest, args.actions, args.ground_truth):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.pose_root.is_dir():
        raise FileNotFoundError(args.pose_root)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    manifest_rows = _read_jsonl(args.manifest)
    action_rows = _read_jsonl(args.actions)
    truth_rows = _read_jsonl(args.ground_truth)
    manifest = {
        str(row["video_id"]): row
        for row in manifest_rows
        if row.get("eligibility") is True
    }
    candidates = build_candidates(action_rows, manifest)
    development = [row for row in candidates if row["partition"] != "challenge"]
    challenge = [row for row in candidates if row["partition"] == "challenge"]
    preparation = NearFallDatasetConfig(fallback_window_secs=(2.0,))
    tensors, samples, rejected = _build_scf_windows(
        development,
        manifest=manifest,
        pose_root=args.pose_root,
        config=preparation,
    )
    if not samples:
        raise ValueError("no usable near-fall development windows")

    candidate_by_id = {str(row["candidate_id"]): row for row in development}
    for sample in samples:
        candidate = candidate_by_id[str(sample["label_id"])]
        partition = str(candidate["partition"])
        sample["partition"] = partition
        sample["dataset"] = "fall_self_total_20260826_home"
        sample["subject_id"] = "self_collected_adult_01"
        sample["source_group_id"] = f"fall_self_scene_{candidate['scene']}"
        sample["sample_group_id"] = str(sample["video_id"])
        sample["split_group_id"] = f"fall_self_video_{sample['video_id']}"

    raw = np.stack(tensors).astype(np.float32)
    partitions = np.asarray([str(row["partition"]) for row in samples])
    labels = np.asarray([int(row["label"]) for row in samples], dtype=np.int64)
    for partition in ("train", "validation"):
        if set(labels[partitions == partition].tolist()) != {0, 1}:
            raise ValueError(f"near-fall {partition} partition lacks a binary class")
    normalization = fit_normalization_statistics(raw, partitions)
    features = apply_normalization(raw, normalization)
    event_counts = Counter(str(row["event_id"]) for row in samples)
    weights = np.asarray(
        [1.0 / event_counts[str(row["event_id"])] for row in samples],
        dtype=np.float32,
    )
    loss_weights = _balanced_loss_weights(samples, weights)
    arrays = {
        "features": features,
        "labels": labels,
        "partitions": partitions,
        "sample_ids": np.asarray([str(row["sample_id"]) for row in samples]),
        "event_ids": np.asarray([str(row["event_id"]) for row in samples]),
        "subject_ids": np.asarray([str(row["subject_id"]) for row in samples]),
        "source_group_ids": np.asarray([str(row["source_group_id"]) for row in samples]),
        "sample_group_ids": np.asarray([str(row["sample_group_id"]) for row in samples]),
        "split_group_ids": np.asarray([str(row["split_group_id"]) for row in samples]),
        "sample_weights": weights,
        "loss_weights": loss_weights,
        "loss_eligible": np.ones(len(samples), dtype=np.bool_),
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = args.output_dir / "dataset.npz"
    samples_path = args.output_dir / "samples.jsonl"
    _write_deterministic_npz(dataset_path, arrays)
    _write_jsonl(samples_path, samples)
    _write_jsonl(args.output_dir / "candidates.jsonl", candidates)
    _write_jsonl(args.output_dir / "challenge_candidates.jsonl", challenge)
    challenge_video_ids = {str(row["video_id"]) for row in challenge}
    _write_jsonl(
        args.output_dir / "challenge_manifest.jsonl",
        [row for row in manifest_rows if str(row["video_id"]) in challenge_video_ids],
    )
    _write_jsonl(
        args.output_dir / "challenge_ground_truth_events.jsonl",
        [
            row
            for row in truth_rows
            if str(row["video_id"]) in challenge_video_ids
            and str(row.get("source_action_id")) in POSITIVE_ACTIONS
        ],
    )
    metadata = {
        "schema_version": "near-fall-event-dataset-v2",
        "task": "near_fall_recovery_confirmation_v1",
        "status": "development_provisional",
        "synthetic": False,
        "training_ready": True,
        "formal_training_ready": False,
        "training_scope": "single_subject_scene_isolated_development",
        "target_semantics": "C03/C04/C05 action-end recovery proxy",
        "negative_semantics": "reviewed A01-A08/A10/C01/C02 action intervals",
        "label_mapping": {"explicit_negative": 0, "near_fall": 1},
        "joint_order": list(NEAR_FALL_JOINTS),
        "channel_order": list(NEAR_FALL_CHANNELS),
        "feature_shape": list(features.shape),
        "preparation_config": preparation.__dict__,
        "normalization": {
            "fit_partition": "train",
            "mean": normalization["mean"].tolist(),
            "std": normalization["std"].tolist(),
            "valid_mask_normalized": False,
        },
        "split_policy": {
            "unit": "scene",
            "train_scenes": sorted(TRAIN_SCENES),
            "validation_scenes": sorted(VALIDATION_SCENES),
            "challenge_scenes": sorted(CHALLENGE_SCENES),
            "allow_subject_overlap": True,
            "reason": "the batch contains one subject; scene isolation is development evidence only",
        },
        "partition_counts": dict(sorted(Counter(partitions.tolist()).items())),
        "partition_event_counts": _partition_event_counts(samples),
        "loss_weighting": "partition_class_balanced_event_weight",
        "candidate_partition_counts": dict(
            sorted(Counter(str(row["partition"]) for row in candidates).items())
        ),
        "rejected_window_counts": dict(sorted(rejected.items())),
        "test_pose_read": False,
        "test_evaluated": False,
        "dataset_sha256": _sha256(dataset_path),
        "samples_sha256": _sha256(samples_path),
        "input_sha256": {
            "manifest": _sha256(args.manifest),
            "actions": _sha256(args.actions),
            "ground_truth": _sha256(args.ground_truth),
        },
        "limitations": [
            "all partitions contain the same adult and capture batch",
            "scene holdout is development evidence, not person-independent testing",
            "C03/C04/C05 are action-level proxies, not clinical near-fall ground truth",
            "the challenge scene was inspected during decoder diagnosis and is not untouched",
        ],
    }
    metadata_path = args.output_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    return {
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "sample_count": len(samples),
        "partition_counts": metadata["partition_counts"],
        "partition_event_counts": metadata["partition_event_counts"],
        "challenge_video_count": len(challenge_video_ids),
    }


def _partition_event_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition in ("train", "validation"):
        event_labels = {
            str(row["event_id"]): int(row["label"])
            for row in samples
            if row["partition"] == partition
        }
        positive = sum(event_labels.values())
        result[partition] = {
            "events": len(event_labels),
            "positive": positive,
            "negative": len(event_labels) - positive,
        }
    return result


def _balanced_loss_weights(
    samples: Sequence[Mapping[str, Any]], event_weights: np.ndarray
) -> np.ndarray:
    result = np.asarray(event_weights, dtype=np.float32).copy()
    for partition in ("train", "validation"):
        event_labels = {
            str(row["event_id"]): int(row["label"])
            for row in samples
            if row["partition"] == partition
        }
        counts = Counter(event_labels.values())
        if set(counts) != {0, 1}:
            raise ValueError(f"near-fall {partition} partition lacks a binary class")
        total = sum(counts.values())
        class_weights = {
            label: total / (2.0 * count) for label, count in counts.items()
        }
        for index, sample in enumerate(samples):
            if sample["partition"] == partition:
                result[index] *= class_weights[int(sample["label"])]
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_dataset(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
