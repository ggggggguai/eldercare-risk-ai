from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.gait import (
    GAIT_STABILITY_FEATURE_NAMES,
    GaitAnalysisConfig,
    extract_gait_windows,
)
from elderly_monitoring.modules.fall_risk.gait_tensor import build_gait_tensor
from elderly_monitoring.modules.fall_risk.kinecal_gait import (
    CANONICAL_GAIT_JOINTS,
    GAIT_TCN_CHANNELS,
)


GAIT_TABULAR_FEATURE_NAMES = GAIT_STABILITY_FEATURE_NAMES + (
    "usable_frame_ratio",
    "gait_keypoint_coverage",
    "mean_core_keypoint_quality",
    "interpolated_point_ratio",
    "jump_outlier_per_frame",
)
GAIT_TARGET_MAPPING = {"normal_activity": 0, "gait_instability": 1}
_NORMAL_WALK_ACTIONS = {"A01", "normal_walk"}
_UNSTABLE_GAIT_ACTIONS = {
    "B01",
    "B02",
    "B03",
    "B04",
    "slow_walk",
    "dragging_walk",
    "shuffling_walk",
    "swaying_walk",
}
_PARTITIONS = ("train", "validation", "test")
_V3_ACTION_SCHEMA = "fall-risk-action-label-v3"


@dataclass(frozen=True)
class GaitWindowPreparationConfig:
    window_frames: int = 64
    stride_frames: int = 32
    min_observed_frames: int = 12
    require_usable_gait: bool = True
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    split_search_iterations: int = 4096

    def __post_init__(self) -> None:
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if self.stride_frames < 1:
            raise ValueError("stride_frames must be positive")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")
        if self.split_search_iterations < 1:
            raise ValueError("split_search_iterations must be positive")


def select_gait_training_labels(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        action_id = str(row.get("action_id", ""))
        is_v3 = row.get("schema_version") == _V3_ACTION_SCHEMA
        action_name = str(
            row.get("action_type", "") if is_v3 else row.get("action_name", "")
        )
        event_type = str(row.get("event_type", ""))
        if action_id in _UNSTABLE_GAIT_ACTIONS or action_name in _UNSTABLE_GAIT_ACTIONS:
            if not is_v3 and event_type != "gait_instability":
                continue
            label = 1
            target_name = "gait_instability"
        elif action_id in _NORMAL_WALK_ACTIONS or action_name in _NORMAL_WALK_ACTIONS:
            if not is_v3 and event_type != "normal_activity":
                continue
            label = 0
            target_name = "normal_activity"
        else:
            continue
        if is_v3:
            if row.get("training_tier") == "ignore":
                continue
            end_exclusive = row.get("end_frame_exclusive")
            if not isinstance(end_exclusive, int) or end_exclusive <= int(
                row.get("start_frame", -1)
            ):
                raise ValueError(
                    f"v3 gait label {row.get('label_id')} has invalid half-open frame bounds"
                )
            row["action_name"] = action_name
            row["event_type"] = target_name
            row["end_frame"] = end_exclusive - 1
            if row.get("end_time_exclusive") is not None:
                row["end_time"] = row["end_time_exclusive"]
        elif row.get("quality") not in (None, "", "clear", "partial_occlusion"):
            continue
        row["label"] = label
        row["target_name"] = target_name
        selected.append(row)
    return selected


def enrich_gait_training_labels(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select gait labels and join their authoritative video manifest metadata."""

    manifests: dict[str, dict[str, Any]] = {}
    for source in manifest_rows:
        manifest = dict(source)
        raw_video_id = manifest.get("video_id")
        if raw_video_id is None or raw_video_id == "":
            continue
        video_id = str(raw_video_id)
        if video_id in manifests:
            raise ValueError(f"duplicate gait manifest video_id: {video_id}")
        manifests[video_id] = manifest

    enriched: list[dict[str, Any]] = []
    for label in select_gait_training_labels(rows):
        video_id = str(label.get("video_id", ""))
        manifest = manifests.get(video_id)
        if manifest is None:
            raise ValueError(f"gait label references missing manifest video: {video_id}")
        if manifest.get("eligibility") is not True:
            raise ValueError(f"gait label references ineligible manifest video: {video_id}")
        label_asset = str(label.get("asset_id", ""))
        manifest_asset = str(manifest.get("asset_id", ""))
        if label_asset and manifest_asset and label_asset != manifest_asset:
            raise ValueError(f"gait label/manifest asset mismatch for {video_id}")
        path = str(manifest.get("path", ""))
        if not path:
            raise ValueError(f"gait manifest is missing video path: {video_id}")
        row = dict(label)
        row.update(
            {
                "file_path": path,
                "scene": str(manifest.get("scene_region", "unknown")),
                "dataset": str(manifest.get("dataset", "unknown")),
                "fps": manifest.get("fps"),
                "frame_count": manifest.get("frame_count"),
                "manifest_content_sha256": manifest.get("sha256"),
            }
        )
        enriched.append(row)
    return enriched


def prepare_gait_window_dataset(
    labels_path: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    *,
    config: GaitWindowPreparationConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    preparation = config or GaitWindowPreparationConfig()
    label_source = Path(labels_path)
    pose_root = Path(pose_dir)
    destination = Path(output_dir)
    labels = select_gait_training_labels(_read_jsonl(label_source))
    if not labels:
        raise ValueError("no gait_instability or normal_walk labels were selected")

    split_group = _split_group_field(labels)
    pose_cache: dict[str, list[dict[str, Any]]] = {}
    pose_paths: dict[str, Path] = {}
    samples: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    tabular_rows: list[list[float]] = []
    rule_scores: list[float] = []
    rejected = Counter()

    labels.sort(
        key=lambda row: (
            str(row.get("video_id", "")),
            int(row.get("start_frame", 0)),
            str(row.get("label_id", "")),
        )
    )
    for label in labels:
        video_id = str(label.get("video_id", ""))
        if not video_id:
            raise ValueError("gait training label is missing video_id")
        if video_id not in pose_cache:
            path = _resolve_pose_path(pose_root, video_id)
            if path is None:
                raise FileNotFoundError(
                    f"pose-quality JSONL not found for {video_id} in {pose_root}"
                )
            pose_paths[video_id] = path
            pose_cache[video_id] = _read_jsonl(path)

        selected_records = _select_labeled_track(label, pose_cache[video_id])
        if not selected_records:
            rejected["no_matching_pose_track"] += 1
            continue
        start_frame = int(label["start_frame"])
        end_frame = int(label["end_frame"])
        frame_records: list[Mapping[str, Any] | None] = [
            selected_records.get(frame_id)
            for frame_id in range(start_frame, end_frame + 1)
        ]
        for window_index, window_start in enumerate(
            _window_starts(len(frame_records), preparation)
        ):
            slots = frame_records[
                window_start : window_start + preparation.window_frames
            ]
            observed_records = [record for record in slots if record is not None]
            if len(observed_records) < preparation.min_observed_frames:
                rejected["insufficient_observed_frames"] += 1
                continue
            rule = _rule_baseline(observed_records)
            if rule is None:
                rejected["rule_window_unavailable"] += 1
                continue
            quality = rule["quality_coverage"]
            if preparation.require_usable_gait and quality["insufficient_gait_quality"]:
                rejected["insufficient_gait_quality"] += 1
                continue
            tensor = build_gait_tensor(
                slots,
                window_frames=preparation.window_frames,
            )
            if not np.any(tensor[..., -1] > 0):
                rejected["empty_quality_mask"] += 1
                continue

            sample_id = f"{label['label_id']}:window-{window_index:03d}"
            split_group_id = str(label[split_group])
            tensors.append(tensor)
            rule_scores.append(float(rule["gait_risk_score"]))
            tabular_rows.append(_tabular_features(rule))
            samples.append(
                {
                    "sample_index": len(samples),
                    "sample_id": sample_id,
                    "label_id": str(label["label_id"]),
                    "video_id": video_id,
                    "split_group_id": split_group_id,
                    "label": int(label["label"]),
                    "target_name": str(label["target_name"]),
                    "action_id": str(label.get("action_id", "")),
                    "action_name": str(label.get("action_name", "")),
                    "scene": str(label.get("scene", "unknown")),
                    "start_frame": start_frame + window_start,
                    "end_frame": min(
                        end_frame,
                        start_frame + window_start + preparation.window_frames - 1,
                    ),
                    "observed_frame_count": len(observed_records),
                    "rule_gait_risk_score": float(rule["gait_risk_score"]),
                }
            )

    if not samples:
        raise ValueError("no usable gait windows were generated")
    labels_array = np.asarray([sample["label"] for sample in samples], dtype=np.int64)
    group_ids = np.asarray([sample["split_group_id"] for sample in samples])
    assignments = _search_group_split(labels_array, group_ids, preparation)
    partitions = np.asarray([assignments[str(group)] for group in group_ids])
    for sample, partition in zip(samples, partitions, strict=True):
        sample["partition"] = str(partition)

    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dataset.npz"
    metadata_path = destination / "metadata.json"
    samples_path = destination / "samples.jsonl"
    for path in (dataset_path, metadata_path, samples_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"gait dataset output already exists: {path}")

    features_array = np.stack(tensors).astype(np.float32)
    tabular_array = np.asarray(tabular_rows, dtype=np.float32)
    rule_array = np.asarray(rule_scores, dtype=np.float32)
    sample_ids = np.asarray([sample["sample_id"] for sample in samples])
    video_ids = np.asarray([sample["video_id"] for sample in samples])
    scenes = np.asarray([sample["scene"] for sample in samples])
    _write_npz_atomic(
        dataset_path,
        features=features_array,
        labels=labels_array,
        sample_ids=sample_ids,
        split_group_ids=group_ids,
        participant_ids=group_ids,
        video_ids=video_ids,
        groups=scenes,
        partitions=partitions,
        tabular_features=tabular_array,
        rule_scores=rule_array,
    )
    _write_jsonl_atomic(samples_path, samples)

    class_counts = Counter(int(value) for value in labels_array)
    partition_counts = Counter(str(value) for value in partitions)
    group_partition_counts = Counter(assignments.values())
    metadata = {
        "schema_version": "gait-window-dataset-v1",
        "task": "gait_instability_vs_normal_activity",
        "negative_label_policy": "A01_normal_walk_only",
        "label_mapping": dict(GAIT_TARGET_MAPPING),
        "joint_order": list(CANONICAL_GAIT_JOINTS),
        "channel_order": list(GAIT_TCN_CHANNELS),
        "tabular_feature_names": list(GAIT_TABULAR_FEATURE_NAMES),
        "feature_shape": list(features_array.shape),
        "preparation_config": asdict(preparation),
        "split_group": split_group,
        "split_is_provisional": split_group != "subject_id",
        "split_limitations": (
            [
                "source labels do not provide usable subject_id values",
                "video-grouped separation cannot rule out the same actor across videos",
            ]
            if split_group != "subject_id"
            else []
        ),
        "sample_count": len(samples),
        "split_group_count": len(assignments),
        "class_counts": {str(key): class_counts.get(key, 0) for key in (0, 1)},
        "window_partition_counts": {
            partition: partition_counts.get(partition, 0) for partition in _PARTITIONS
        },
        "group_partition_counts": {
            partition: group_partition_counts.get(partition, 0)
            for partition in _PARTITIONS
        },
        "group_assignments": dict(sorted(assignments.items())),
        "rejected_window_counts": dict(sorted(rejected.items())),
        "labels_path": label_source.as_posix(),
        "labels_sha256": _sha256_file(label_source),
        "pose_inputs": {
            video_id: {"path": path.as_posix(), "sha256": _sha256_file(path)}
            for video_id, path in sorted(pose_paths.items())
        },
        "samples_sha256": _sha256_file(samples_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "limitations": [
            "labels describe observable gait actions and are not clinical diagnoses",
            "probabilities require calibration before deployment",
            "the dataset is isolated from AlgorithmEvent and the realtime main path",
        ],
    }
    _write_json_atomic(metadata_path, metadata)
    return {
        "output_dir": destination.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "sample_count": len(samples),
        "split_group": split_group,
        "split_group_count": len(assignments),
        "window_partition_counts": metadata["window_partition_counts"],
        "rejected_window_counts": metadata["rejected_window_counts"],
    }


def _split_group_field(labels: Sequence[Mapping[str, Any]]) -> str:
    subject_ids = [str(row.get("subject_id", "")).strip() for row in labels]
    if subject_ids and all(value and value.lower() != "unknown" for value in subject_ids):
        return "subject_id"
    return "video_id"


def _resolve_pose_path(root: Path, video_id: str) -> Path | None:
    candidates = (
        root / f"{video_id}.jsonl",
        root / f"{video_id}_poses_cleaned.jsonl",
    )
    return next((path for path in candidates if path.is_file()), None)


def _select_labeled_track(
    label: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    start_frame = int(label["start_frame"])
    end_frame = int(label["end_frame"])
    by_track: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        frame_id = int(record.get("frame_id", -1))
        if not start_frame <= frame_id <= end_frame:
            continue
        track_id = str(record.get("track_id", record.get("person_id", "unknown")))
        previous = by_track[track_id].get(frame_id)
        if previous is None or _record_confidence(record) > _record_confidence(previous):
            by_track[track_id][frame_id] = record
    if not by_track:
        return {}

    best_track = max(
        sorted(by_track),
        key=lambda track_id: _track_match_score(
            label, by_track[track_id], start_frame, end_frame
        ),
    )
    return by_track[best_track]


def _track_match_score(
    label: Mapping[str, Any],
    records: Mapping[int, Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
) -> tuple[float, float, float, float]:
    frame_span = max(1, end_frame - start_frame + 1)
    coverage = len(records) / frame_span
    overlaps: list[float] = []
    for frame_id, record in records.items():
        actual = record.get("bbox_pixels", record.get("bbox"))
        expected = _interpolated_bbox(label, frame_id, start_frame, end_frame)
        if _is_bbox(actual) and expected is not None:
            overlaps.append(_bbox_iou(actual, expected))
    mean_iou = float(np.mean(overlaps)) if overlaps else 0.0
    mean_confidence = float(
        np.mean([_record_confidence(record) for record in records.values()])
    )
    # CVAT 的动作框比“在片段中出现更久”更能识别目标人物。覆盖率只作为
    # 辅助项，避免背景人物凭完整轨迹压过与标注框实际重合的短轨迹。
    return mean_iou + (0.25 * coverage), mean_iou, coverage, mean_confidence


def _interpolated_bbox(
    label: Mapping[str, Any],
    frame_id: int,
    start_frame: int,
    end_frame: int,
) -> list[float] | None:
    start = label.get("bbox_start")
    end = label.get("bbox_end")
    if not _is_bbox(start) or not _is_bbox(end):
        return None
    ratio = (
        (frame_id - start_frame) / (end_frame - start_frame)
        if end_frame > start_frame
        else 0.0
    )
    return [
        float(first) + ((float(second) - float(first)) * ratio)
        for first, second in zip(start, end, strict=True)
    ]


def _bbox_iou(first: Sequence[Any], second: Sequence[Any]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 4
    )


def _record_confidence(record: Mapping[str, Any]) -> float:
    value = _optional_float(record.get("pose_confidence", record.get("keypoint_quality")))
    return value if value is not None else 0.0


def _window_starts(
    frame_count: int,
    config: GaitWindowPreparationConfig,
) -> list[int]:
    if frame_count <= config.window_frames:
        return [0]
    final_start = frame_count - config.window_frames
    starts = list(range(0, final_start + 1, config.stride_frames))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _rule_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    config = GaitAnalysisConfig(
        window_frames=len(records), min_window_frames=min(5, len(records))
    )
    windows = extract_gait_windows(records, config=config)
    return windows[0] if len(windows) == 1 else None


def _tabular_features(rule: Mapping[str, Any]) -> list[float]:
    gait = rule["gait_stability_features"]
    quality = rule["quality_coverage"]
    frame_count = max(1, int(quality.get("frame_count", 0)))
    values = [float(gait.get(name) or 0.0) for name in GAIT_STABILITY_FEATURE_NAMES]
    values.extend(
        [
            float(quality.get("usable_frame_ratio", 0.0)),
            float(quality.get("gait_keypoint_coverage", 0.0)),
            float(quality.get("mean_core_keypoint_quality", 0.0)),
            float(quality.get("interpolated_point_ratio", 0.0)),
            float(quality.get("jump_outlier_count", 0.0)) / frame_count,
        ]
    )
    return values


def _search_group_split(
    labels: np.ndarray,
    group_ids: np.ndarray,
    config: GaitWindowPreparationConfig,
) -> dict[str, str]:
    normalized_group_ids = group_ids.astype(str)
    groups = sorted(set(normalized_group_ids.tolist()))
    if len(groups) < 6:
        raise ValueError("at least 6 split groups are required for three binary partitions")
    test_fraction = 1.0 - config.train_fraction - config.validation_fraction
    fractions = (config.train_fraction, config.validation_fraction, test_fraction)
    group_counts = {
        group: np.bincount(labels[normalized_group_ids == group], minlength=2)
        for group in groups
    }
    train_count = max(1, int(round(len(groups) * config.train_fraction)))
    validation_count = max(1, int(round(len(groups) * config.validation_fraction)))
    if train_count + validation_count >= len(groups):
        validation_count = max(1, len(groups) - train_count - 1)
    sizes = (
        train_count,
        validation_count,
        len(groups) - train_count - validation_count,
    )

    best: tuple[float, dict[str, str]] | None = None
    for attempt in range(config.split_search_iterations):
        shuffled = groups.copy()
        random.Random(config.seed + (attempt * 1_000_003)).shuffle(shuffled)
        partitions = {
            "train": shuffled[: sizes[0]],
            "validation": shuffled[sizes[0] : sizes[0] + sizes[1]],
            "test": shuffled[sizes[0] + sizes[1] :],
        }
        partition_class_counts = {
            partition: sum(
                (group_counts[group] for group in partition_groups),
                start=np.zeros(2, dtype=np.int64),
            )
            for partition, partition_groups in partitions.items()
        }
        if any(np.any(counts == 0) for counts in partition_class_counts.values()):
            continue
        total_class_counts = np.bincount(labels, minlength=2)
        score = 0.0
        for partition, fraction in zip(_PARTITIONS, fractions, strict=True):
            actual = partition_class_counts[partition] / total_class_counts
            score += float(np.abs(actual - fraction).sum())
        assignments = {
            group: partition
            for partition, partition_groups in partitions.items()
            for group in partition_groups
        }
        if best is None or score < best[0]:
            best = (score, assignments)
            if score <= 1e-12:
                break
    if best is None:
        raise ValueError(
            "unable to create train/validation/test group split with both classes"
        )
    return best[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(partial, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
