from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.fall_event_continuous import (
    FALL_EVENT_CONTINUOUS_CHANNELS,
    FALL_EVENT_CONTINUOUS_JOINTS,
    build_fall_event_continuous_tensor,
    resample_causal_pose_records,
)


DEVELOPMENT_PARTITIONS = {"train", "validation"}
_EPSILON = 1e-6


@dataclass(frozen=True)
class ContinuousFallDatasetConfig:
    window_sec: float = 4.0
    target_fps: float = 8.0
    max_gap_sec: float = 0.25
    min_observed_frames: int = 16
    min_partial_observed_frames: int = 8
    min_valid_joint_ratio: float = 0.50
    max_interpolated_joint_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.window_sec <= 0 or self.target_fps <= 0 or self.max_gap_sec <= 0:
            raise ValueError("window_sec, target_fps and max_gap_sec must be positive")
        if self.window_frames < 2:
            raise ValueError("continuous fall window must contain at least two frames")
        if not 1 <= self.min_observed_frames <= self.window_frames:
            raise ValueError("min_observed_frames must be within the window")
        if not 1 <= self.min_partial_observed_frames <= self.min_observed_frames:
            raise ValueError("min_partial_observed_frames must not exceed min_observed_frames")
        if not 0 < self.min_valid_joint_ratio <= 1:
            raise ValueError("min_valid_joint_ratio must be within (0, 1]")
        if not 0 <= self.max_interpolated_joint_ratio <= 1:
            raise ValueError("max_interpolated_joint_ratio must be within [0, 1]")

    @property
    def window_frames(self) -> int:
        return int(round(self.window_sec * self.target_fps))


def clean_pose_records_for_causal_training(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Mask pose coordinates produced by the non-causal offline interpolator."""

    cleaned: list[dict[str, Any]] = []
    for source in records:
        row = json.loads(json.dumps(dict(source), ensure_ascii=False))
        points = row.get("keypoints")
        if not isinstance(points, list):
            cleaned.append(row)
            continue
        for point in points:
            if not isinstance(point, dict) or point.get("source") != "interpolated":
                continue
            point["valid"] = False
            point["causal_masked"] = True
            point["quality_weight"] = 0.0
            point["x_smooth"] = None
            point["y_smooth"] = None
        cleaned.append(row)
    return cleaned


def build_continuous_fall_dataset(
    governance_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    pose_roots: Sequence[str | Path],
    output_dir: str | Path,
    config: ContinuousFallDatasetConfig | None = None,
) -> dict[str, Any]:
    contract = config or ContinuousFallDatasetConfig()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"continuous fall dataset output already exists: {destination}")
    roots = [Path(root) for root in pose_roots]
    if not roots or any(not root.is_dir() for root in roots):
        raise FileNotFoundError("all continuous fall pose roots must exist")
    manifest_index = {
        str(row["video_id"]): dict(row)
        for row in manifest_rows
        if isinstance(row.get("video_id"), str)
    }
    rows = sorted(
        (dict(row) for row in governance_rows if row.get("partition") in DEVELOPMENT_PARTITIONS),
        key=lambda row: (str(row["partition"]), str(row["sample_id"])),
    )
    pose_cache: dict[str, list[dict[str, Any]]] = {}
    pose_hash_cache: dict[str, str] = {}
    tensors: list[np.ndarray] = []
    presence_targets: list[int] = []
    onset_targets: list[int] = []
    onset_masks: list[int] = []
    sample_weights: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        video_id = str(row["video_id"])
        manifest = manifest_index.get(video_id)
        if manifest is None or manifest.get("eligibility") is not True:
            rejected["missing_or_ineligible_manifest"] += 1
            continue
        pose_path = _resolve_pose_path(roots, video_id)
        if pose_path is None:
            rejected["missing_pose"] += 1
            continue
        if video_id not in pose_cache:
            pose_cache[video_id] = clean_pose_records_for_causal_training(
                _read_jsonl(pose_path)
            )
            pose_hash_cache[video_id] = _sha256_file(pose_path)
        records = _select_target_track(pose_cache[video_id], row)
        if not records:
            rejected["no_target_track"] += 1
            continue
        end_time = float(row["end_time_exclusive_sec"])
        duration = _optional_float(manifest.get("duration_sec"))
        cutoff = max(end_time, contract.window_sec)
        if duration is not None:
            cutoff = min(cutoff, duration)
        if cutoff <= 0:
            rejected["invalid_cutoff"] += 1
            continue
        try:
            tensor, slot_times, source_times, partial_context = _build_padded_causal_window(
                records, cutoff_time_sec=cutoff, config=contract
            )
        except ValueError as exc:
            rejected[_reason(exc)] += 1
            continue
        frame_index = {name: index for index, name in enumerate(FALL_EVENT_CONTINUOUS_CHANNELS)}
        frame_mask = tensor[:, 0, frame_index["frame_mask"]] > 0
        observed = int(np.sum(frame_mask))
        valid = tensor[..., frame_index["valid_mask"]] > 0
        valid_ratio = float(np.mean(valid[frame_mask])) if observed else 0.0
        interpolated = tensor[..., frame_index["interpolated_mask"]] > 0
        interpolated_ratio = float(np.mean(interpolated[valid])) if np.any(valid) else 0.0
        required_observed = (
            contract.min_partial_observed_frames
            if partial_context
            else contract.min_observed_frames
        )
        if observed < required_observed:
            rejected["insufficient_observed_frames"] += 1
            continue
        if valid_ratio < contract.min_valid_joint_ratio:
            rejected["insufficient_valid_joint_ratio"] += 1
            continue
        if interpolated_ratio > contract.max_interpolated_joint_ratio + _EPSILON:
            rejected["interpolated_joint_ratio"] += 1
            continue
        onset_time = float(row["start_time_sec"])
        window_start = cutoff - contract.window_sec
        onset_allowed = "onset" in list(row.get("allowed_heads", [])) and float(
            row.get("onset_loss_weight", 0.0)
        ) > 0
        onset_mask = int(onset_allowed and window_start - _EPSILON <= onset_time <= cutoff + _EPSILON)
        onset_target = 0
        if onset_mask:
            onset_target = int(round((onset_time - window_start) * contract.target_fps))
            onset_target = min(contract.window_frames - 1, max(0, onset_target))
        tensors.append(tensor)
        presence_targets.append(int(row["target_presence"]))
        onset_targets.append(onset_target)
        onset_masks.append(onset_mask)
        sample_weights.append(float(row.get("sampling_weight", 1.0)))
        future_available = int(
            sum(_timestamp(record) > cutoff + _EPSILON for record in pose_cache[video_id])
        )
        sample_rows.append(
            {
                **row,
                "dataset_schema_version": "fall-event-continuous-dataset-v1",
                "cutoff_time_sec": round(cutoff, 6),
                "window_start_time_sec": round(window_start, 6),
                "window_end_time_sec": round(cutoff, 6),
                "partial_context": partial_context,
                "observed_frame_count": observed,
                "required_observed_frame_count": required_observed,
                "valid_joint_ratio": round(valid_ratio, 6),
                "interpolated_joint_ratio": round(interpolated_ratio, 6),
                "future_pose_records_available": future_available,
                "future_observation_count": 0,
                "onset_target_frame": onset_target,
                "onset_supervision_mask": onset_mask,
                "pose_sha256": pose_hash_cache[video_id],
                "pose_cleaning_policy": "mask_interpolated_coordinates_and_derived_motion",
            }
        )
    if not sample_rows:
        raise ValueError("no valid continuous fall windows were generated")
    arrays = {
        "features": np.stack(tensors).astype(np.float32),
        "presence_targets": np.asarray(presence_targets, dtype=np.int64),
        "onset_targets": np.asarray(onset_targets, dtype=np.int64),
        "onset_masks": np.asarray(onset_masks, dtype=np.float32),
        "sample_weights": np.asarray(sample_weights, dtype=np.float32),
        "presence_loss_weights": np.asarray(
            [float(row.get("presence_loss_weight", 1.0)) for row in sample_rows],
            dtype=np.float32,
        ),
        "onset_loss_weights": np.asarray(
            [float(row.get("onset_loss_weight", 0.0)) for row in sample_rows],
            dtype=np.float32,
        ),
        "partitions": np.asarray([row["partition"] for row in sample_rows]),
        "sample_ids": np.asarray([row["sample_id"] for row in sample_rows]),
        "source_label_ids": np.asarray([row["source_label_id"] for row in sample_rows]),
        "video_ids": np.asarray([row["video_id"] for row in sample_rows]),
    }
    destination.mkdir(parents=True, exist_ok=False)
    dataset_path = destination / "dataset.npz"
    samples_path = destination / "samples.jsonl"
    metadata_path = destination / "metadata.json"
    _write_npz_atomic(dataset_path, arrays)
    _write_jsonl_atomic(samples_path, sample_rows)
    metadata = {
        "schema_version": "fall-event-continuous-dataset-v1",
        "status": "development_provisional",
        "task": "fall_event_continuous_presence_onset",
        "target_task": "fall_event_v1",
        "feature_shape": list(arrays["features"].shape),
        "joint_order": list(FALL_EVENT_CONTINUOUS_JOINTS),
        "channel_order": list(FALL_EVENT_CONTINUOUS_CHANNELS),
        "config": asdict(contract),
        "sample_count": len(sample_rows),
        "partition_counts": dict(Counter(str(value) for value in arrays["partitions"])),
        "presence_counts": dict(Counter(str(value) for value in arrays["presence_targets"])),
        "onset_supervised_count": int(np.sum(arrays["onset_masks"] > 0)),
        "rejected_counts": dict(sorted(rejected.items())),
        "test_pose_read": False,
        "test_truth_read": False,
        "causal": True,
        "partial_context_count": int(sum(row["partial_context"] for row in sample_rows)),
        "future_observation_count": 0,
        "future_pose_records_available": int(
            sum(row["future_pose_records_available"] for row in sample_rows)
        ),
        "limitations": [
            "development train/validation only; test labels and pose are not read",
            "one causal cutoff is materialized per governed interval",
            "continuous background camera-hour and elderly-domain gates remain open",
        ],
    }
    metadata["dataset_sha256"] = _sha256_file(dataset_path)
    metadata["samples_sha256"] = _sha256_file(samples_path)
    _write_json_atomic(metadata_path, metadata)
    return {
        "output_dir": destination.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "samples_path": samples_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "sample_count": len(sample_rows),
        "rejected_counts": dict(sorted(rejected.items())),
    }


def _select_target_track(records: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> list[dict[str, Any]]:
    start = float(row["start_time_sec"])
    end = float(row["end_time_exclusive_sec"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        timestamp = _optional_float(record.get("timestamp_sec"))
        if timestamp is None:
            continue
        if start <= timestamp <= end:
            groups[_target_key(record)].append(dict(record))
    if not groups:
        groups = defaultdict(list)
        for record in records:
            groups[_target_key(record)].append(dict(record))
    if not groups:
        return []
    key = max(groups, key=lambda value: (len(groups[value]), value))
    selected = [dict(record) for record in records if _target_key(record) == key]
    selected.sort(key=lambda record: (_timestamp(record), int(record.get("frame_id", 0))))
    return selected


def _build_padded_causal_window(
    records: Sequence[Mapping[str, Any]],
    *,
    cutoff_time_sec: float,
    config: ContinuousFallDatasetConfig,
) -> tuple[np.ndarray, list[float], list[float | None], bool]:
    history_frames = min(
        config.window_frames,
        int(math.floor(cutoff_time_sec * config.target_fps + _EPSILON)) + 1,
    )
    if history_frames < 2:
        raise ValueError("insufficient causal history for a two-frame partial window")
    history_sec = history_frames / config.target_fps
    slots, slot_times, source_times = resample_causal_pose_records(
        records,
        cutoff_time_sec=cutoff_time_sec,
        window_sec=history_sec,
        target_fps=config.target_fps,
        max_gap_sec=config.max_gap_sec,
    )
    tensor = build_fall_event_continuous_tensor(
        slots,
        slot_timestamps_sec=slot_times,
        cutoff_time_sec=cutoff_time_sec,
        max_gap_sec=config.max_gap_sec,
    )
    missing_prefix = config.window_frames - tensor.shape[0]
    if missing_prefix <= 0:
        return tensor, slot_times, source_times, False
    padding = np.zeros(
        (missing_prefix, len(FALL_EVENT_CONTINUOUS_JOINTS), len(FALL_EVENT_CONTINUOUS_CHANNELS)),
        dtype=np.float32,
    )
    padded_times = [
        round(cutoff_time_sec - (config.window_frames - 1 - index) / config.target_fps, 9)
        for index in range(config.window_frames)
    ]
    return (
        np.concatenate((padding, tensor), axis=0),
        padded_times,
        [None] * missing_prefix + source_times,
        True,
    )


def _target_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("person_id", "unknown")), str(record.get("track_id", "unknown"))


def _resolve_pose_path(roots: Sequence[Path], video_id: str) -> Path | None:
    for root in roots:
        for name in (f"{video_id}.jsonl", f"{video_id}_poses_cleaned.jsonl"):
            path = root / name
            if path.is_file():
                return path
    return None


def _timestamp(record: Mapping[str, Any]) -> float:
    value = _optional_float(record.get("timestamp_sec"))
    if value is None or value < 0:
        raise ValueError("pose record requires a non-negative timestamp_sec")
    return value


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _reason(exc: ValueError) -> str:
    message = str(exc)
    if "insufficient causal history" in message:
        return "insufficient_causal_history"
    if "multiple target tracks" in message:
        return "multiple_target_tracks"
    if "duplicate target timestamps" in message:
        return "duplicate_target_timestamps"
    return "tensor_build_error"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
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
    partial.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)
