from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.fall_event_continuous_dataset import (
    ContinuousFallDatasetConfig,
    build_continuous_fall_dataset,
)


FALL_ACTIONS = {"D01", "D02"}
IGNORED_ACTIONS = {"D04"}
DEFAULT_HOLDOUT_PREFIXES = ("fall_f05_", "fall_n07_", "fall_n08_", "fall_r01_", "fall_r02_")
DEFAULT_VALIDATION_PREFIXES = ("fall_n05_", "fall_n06_")
DEFAULT_VALIDATION_F04_SCENES = {"bed", "dining", "hall"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a video-isolated adaptation dataset from fall_nearfall_v1."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluations/fall_risk/fall_nearfall_v1/manifest.jsonl"),
    )
    parser.add_argument(
        "--actions",
        type=Path,
        default=Path("data/evaluations/fall_risk/fall_nearfall_v1/action_labels.jsonl"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/evaluations/fall_risk/fall_nearfall_v1/ground_truth_events.jsonl"),
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=Path("reports/fall_risk/fall_nearfall_v1/pose"),
    )
    parser.add_argument(
        "--base-data",
        type=Path,
        default=Path("reports/fall_risk/fall_event_continuous_dataset_v2/dataset.npz"),
    )
    parser.add_argument(
        "--base-metadata",
        type=Path,
        default=Path("reports/fall_risk/fall_event_continuous_dataset_v2/metadata.json"),
    )
    parser.add_argument(
        "--base-samples",
        type=Path,
        default=Path("reports/fall_risk/fall_event_continuous_dataset_v2/samples.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--max-gap-sec", type=float, default=0.25)
    parser.add_argument("--min-observed-frames", type=int, default=16)
    parser.add_argument("--min-partial-observed-frames", type=int, default=8)
    parser.add_argument("--min-valid-joint-ratio", type=float, default=0.50)
    parser.add_argument("--max-interpolated-joint-ratio", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_adaptation_dataset(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def prepare_adaptation_dataset(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.manifest,
        args.actions,
        args.ground_truth,
        args.base_data,
        args.base_metadata,
        args.base_samples,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input files: " + ", ".join(missing))
    if not args.pose_root.is_dir():
        raise FileNotFoundError(f"pose root does not exist: {args.pose_root}")
    if args.output_dir.exists():
        raise FileExistsError(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = _read_jsonl(args.manifest)
    actions = _read_jsonl(args.actions)
    truth = _read_jsonl(args.ground_truth)
    manifest_by_video = {str(row["video_id"]): row for row in manifest}
    actions_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in actions:
        actions_by_video.setdefault(str(row["video_id"]), []).append(row)

    split_by_video = {
        str(row["video_id"]): _partition_for_video(row)
        for row in manifest
        if bool(row.get("eligibility", False))
    }
    holdout_ids = {video_id for video_id, part in split_by_video.items() if part == "holdout"}
    train_ids = {video_id for video_id, part in split_by_video.items() if part == "train"}
    validation_ids = {video_id for video_id, part in split_by_video.items() if part == "validation"}
    if not train_ids or not validation_ids or not holdout_ids:
        raise ValueError("adaptation split must contain train, validation and holdout videos")

    governance_rows = _build_governance_rows(actions_by_video, split_by_video, manifest_by_video)
    self_dir = args.output_dir / "self_collected_dataset"
    config = ContinuousFallDatasetConfig(
        window_sec=args.window_sec,
        target_fps=args.target_fps,
        max_gap_sec=args.max_gap_sec,
        min_observed_frames=args.min_observed_frames,
        min_partial_observed_frames=args.min_partial_observed_frames,
        min_valid_joint_ratio=args.min_valid_joint_ratio,
        max_interpolated_joint_ratio=args.max_interpolated_joint_ratio,
    )
    build_result = build_continuous_fall_dataset(
        governance_rows,
        manifest,
        pose_roots=[args.pose_root],
        output_dir=self_dir,
        config=config,
    )
    merged_result = _merge_datasets(
        args.base_data,
        args.base_metadata,
        args.base_samples,
        self_dir,
        args.output_dir,
    )

    holdout_manifest = [manifest_by_video[video_id] for video_id in sorted(holdout_ids)]
    holdout_truth = [
        {**row, "split_id": "fall-nearfall-adaptation-v1-holdout"}
        for row in truth
        if str(row.get("video_id")) in holdout_ids
    ]
    _write_jsonl(args.output_dir / "holdout_manifest.jsonl", holdout_manifest)
    _write_jsonl(args.output_dir / "holdout_ground_truth_events.jsonl", holdout_truth)
    split = {
        "schema_version": "fall-nearfall-adaptation-split-v1",
        "split_id": "fall-nearfall-adaptation-v1",
        "status": "development_provisional",
        "leakage_policy": "video_isolated; holdout videos are never materialized into training dataset",
        "partition_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "holdout": len(holdout_ids),
        },
        "train_video_ids": sorted(train_ids),
        "validation_video_ids": sorted(validation_ids),
        "holdout_video_ids": sorted(holdout_ids),
        "train_sample_count": int(merged_result["adaptation_partition_counts"]["train"]),
        "validation_sample_count": int(merged_result["adaptation_partition_counts"]["validation"]),
        "adaptation_dataset": build_result,
        "merged_dataset": merged_result,
        "source_hashes": {
            "manifest": _sha256_file(args.manifest),
            "actions": _sha256_file(args.actions),
            "ground_truth": _sha256_file(args.ground_truth),
            "base_data": _sha256_file(args.base_data),
            "base_metadata": _sha256_file(args.base_metadata),
            "base_samples": _sha256_file(args.base_samples),
        },
        "holdout_manifest_sha256": _sha256_file(args.output_dir / "holdout_manifest.jsonl"),
        "holdout_ground_truth_sha256": _sha256_file(
            args.output_dir / "holdout_ground_truth_events.jsonl"
        ),
        "holdout_not_used_for": [
            "threshold_selection",
            "model_selection",
            "training",
            "normalization",
        ],
        "limitations": [
            "all self-collected videos are one adult, one household and one capture batch",
            "holdout is domain holdout, not cross-person or clinical validation",
            "D04 long static after fall is excluded from adaptation supervision",
        ],
    }
    _write_json(args.output_dir / "split.json", split)
    return {
        "output_dir": args.output_dir.as_posix(),
        "dataset_path": merged_result["dataset_path"],
        "holdout_manifest": (args.output_dir / "holdout_manifest.jsonl").as_posix(),
        "holdout_ground_truth": (args.output_dir / "holdout_ground_truth_events.jsonl").as_posix(),
        "split_path": (args.output_dir / "split.json").as_posix(),
        "partition_counts": split["partition_counts"],
        "adaptation_sample_counts": {
            "train": merged_result["adaptation_partition_counts"]["train"],
            "validation": merged_result["adaptation_partition_counts"]["validation"],
        },
        "adaptation_presence_counts": merged_result["adaptation_presence_counts"],
    }


def _partition_for_video(row: Mapping[str, Any]) -> str:
    video_id = str(row["video_id"]).lower()
    scene = str(row.get("scene_region", "")).lower()
    if video_id.startswith(DEFAULT_HOLDOUT_PREFIXES):
        return "holdout"
    if video_id.startswith(DEFAULT_VALIDATION_PREFIXES):
        return "validation"
    if video_id.startswith("fall_f04_") and scene in DEFAULT_VALIDATION_F04_SCENES:
        return "validation"
    return "train"


def _build_governance_rows(
    actions_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    split_by_video: Mapping[str, str],
    manifest_by_video: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for video_id in sorted(actions_by_video):
        partition = split_by_video.get(video_id)
        if partition not in {"train", "validation"}:
            continue
        manifest = manifest_by_video.get(video_id, {})
        for action in sorted(actions_by_video[video_id], key=lambda row: str(row.get("label_id"))):
            action_id = str(action.get("action_id", ""))
            if action_id in IGNORED_ACTIONS:
                skipped["post_fall_static"] += 1
                continue
            if not action.get("label_id"):
                skipped["missing_label_id"] += 1
                continue
            start = float(action.get("start_time", 0.0))
            end = float(action.get("end_time", 0.0))
            if end <= start:
                skipped["invalid_interval"] += 1
                continue
            positive = action_id in FALL_ACTIONS
            label_id = str(action["label_id"])
            digest = hashlib.sha256(f"fall-nearfall-adaptation-v1:{video_id}:{label_id}".encode()).hexdigest()[:24]
            rows.append(
                {
                    "schema_version": "fall-event-continuous-supervision-v1",
                    "sample_id": f"adaptv1_{digest}",
                    "source_label_id": f"adapt_action_{label_id}",
                    "source_label_kind": "self_collected_action",
                    "video_id": video_id,
                    "partition": partition,
                    "dataset": "self_collected_fall_nearfall",
                    "subject_id": manifest.get("subject_id", "self_collected_adult_01"),
                    "scene_region": manifest.get("scene_region", "unknown"),
                    "pose_root": "reports/fall_risk/fall_nearfall_v1/pose",
                    "pose_path": f"reports/fall_risk/fall_nearfall_v1/pose/{video_id}.jsonl",
                    "start_time_sec": start,
                    "end_time_exclusive_sec": end,
                    "start_frame": int(action.get("start_frame", 0)),
                    "end_frame_exclusive": int(action.get("end_frame", 0)) + 1,
                    "action_id": action_id,
                    "action_name": action.get("action_name"),
                    "target_presence": int(positive),
                    "allowed_heads": ["presence"],
                    "presence_loss_weight": 1.0 if positive else 0.8,
                    "onset_loss_weight": 0.0,
                    "sampling_weight": 1.0,
                    "supervision_family": "self_collected_fall" if positive else "self_collected_hard_negative",
                    "supervision_strength": "adaptation",
                    "boundary_precision": "manual_interval",
                    "materialization_status": "ready",
                    "source_group_id": "fall_nearfall_v1_single_subject_session",
                    "split_group_id": f"adapt_split_{partition}",
                    "normalization_group_id": f"adapt_video_{video_id}",
                    "normalization_weight_policy": "one_window_per_action_interval",
                    "hard_negative_type": None if positive else action.get("event_type"),
                    "causal_pose_policy": "mask_interpolated_coordinates_and_derived_motion",
                }
            )
    if not rows:
        raise ValueError("no eligible adaptation supervision rows")
    return rows


def _merge_datasets(
    base_data: Path,
    base_metadata: Path,
    base_samples: Path,
    adaptation_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    with np.load(base_data, allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    with np.load(adaptation_dir / "dataset.npz", allow_pickle=False) as archive:
        adaptation = {name: archive[name] for name in archive.files}
    if set(base) != set(adaptation):
        raise ValueError(f"base/adaptation array keys differ: {sorted(set(base) ^ set(adaptation))}")
    arrays: dict[str, np.ndarray] = {}
    for name in base:
        if base[name].ndim == 0 or adaptation[name].ndim == 0:
            raise ValueError(f"scalar dataset array is unsupported: {name}")
        if base[name].shape[1:] != adaptation[name].shape[1:]:
            raise ValueError(f"dataset array shapes differ for {name}")
        arrays[name] = np.concatenate((base[name], adaptation[name]), axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.npz"
    with dataset_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    base_rows = _read_jsonl(base_samples)
    adaptation_rows = _read_jsonl(adaptation_dir / "samples.jsonl")
    samples_path = output_dir / "samples.jsonl"
    _write_jsonl(samples_path, [*base_rows, *adaptation_rows])
    base_metadata_value = json.loads(base_metadata.read_text(encoding="utf-8"))
    adaptation_metadata = json.loads((adaptation_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        **base_metadata_value,
        "schema_version": "fall-event-continuous-dataset-v1",
        "status": "development_provisional_adaptation",
        "adaptation_source": "fall_nearfall_v1",
        "base_dataset_sha256": _sha256_file(base_data),
        "adaptation_dataset_sha256": _sha256_file(adaptation_dir / "dataset.npz"),
        "feature_shape": list(arrays["features"].shape),
        "sample_count": int(arrays["features"].shape[0]),
        "partition_counts": dict(Counter(arrays["partitions"].astype(str).tolist())),
        "presence_counts": dict(Counter(arrays["presence_targets"].astype(str).tolist())),
        "adaptation_partition_counts": dict(Counter(adaptation["partitions"].astype(str).tolist())),
        "adaptation_presence_counts": dict(Counter(adaptation["presence_targets"].astype(str).tolist())),
        "adaptation_metadata": adaptation_metadata,
        "test_pose_read": False,
        "test_truth_read": False,
        "main_path_replacement": False,
    }
    metadata["dataset_sha256"] = _sha256_file(dataset_path)
    metadata["samples_sha256"] = _sha256_file(samples_path)
    _write_json(output_dir / "metadata.json", metadata)
    return {
        "dataset_path": dataset_path.as_posix(),
        "samples_path": samples_path.as_posix(),
        "metadata_path": (output_dir / "metadata.json").as_posix(),
        "sample_count": int(arrays["features"].shape[0]),
        "partition_counts": metadata["partition_counts"],
        "presence_counts": metadata["presence_counts"],
        "adaptation_partition_counts": metadata["adaptation_partition_counts"],
        "adaptation_presence_counts": metadata["adaptation_presence_counts"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
