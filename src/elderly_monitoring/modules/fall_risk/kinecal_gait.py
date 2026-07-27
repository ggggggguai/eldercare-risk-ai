from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from elderly_monitoring.modules.fall_risk.kinecal import (
    age_group_mismatches,
    participant_number,
    read_register_csv,
    select_risk_group_participants,
)


CANONICAL_GAIT_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "pelvis_center",
    "shoulder_center",
)
GAIT_TCN_CHANNELS = ("x", "y", "dx", "dy", "quality")
KINECAL_TO_CANONICAL = (
    ("ShoulderLeft", "left_shoulder"),
    ("ShoulderRight", "right_shoulder"),
    ("ElbowLeft", "left_elbow"),
    ("ElbowRight", "right_elbow"),
    ("WristLeft", "left_wrist"),
    ("WristRight", "right_wrist"),
    ("HipLeft", "left_hip"),
    ("HipRight", "right_hip"),
    ("KneeLeft", "left_knee"),
    ("KneeRight", "right_knee"),
    ("AnkleLeft", "left_ankle"),
    ("AnkleRight", "right_ankle"),
)
KINECAL_TRACKING_QUALITY = {
    "Tracked": 1.0,
    "Inferred": 0.5,
    "NotTracked": 0.0,
}
FALL_HISTORY_PROXY_LABELS = {"NF": 0, "FHs": 1, "FHm": 1}
PARTITIONS = ("train", "validation", "test")
MOVEMENT = "3m-walk-Front-View"
_TIMESTAMP_SCALE_SECONDS = 0.001


@dataclass(frozen=True)
class KinecalGaitPreparationConfig:
    target_fps: float = 30.0
    max_gap_sec: float = 0.10
    window_frames: int = 128
    stride_frames: int = 64
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2")
        if self.stride_frames < 1:
            raise ValueError("stride_frames must be at least 1")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")


def parse_kinecal_frame(text: str) -> tuple[np.ndarray, np.ndarray]:
    rows: dict[str, tuple[np.ndarray, float]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 7:
            raise ValueError(
                f"invalid KINECAL skeleton row {line_number}: expected 7 fields, "
                f"got {len(fields)}"
            )
        joint, state = fields[:2]
        if joint in rows:
            raise ValueError(f"duplicate KINECAL joint {joint!r} at row {line_number}")
        if state not in KINECAL_TRACKING_QUALITY:
            raise ValueError(f"unknown KINECAL tracking state: {state!r}")
        try:
            pixel_xy = np.asarray(
                [_parse_pixel_value(fields[5]), _parse_pixel_value(fields[6])],
                dtype=np.float32,
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid KINECAL pixel coordinate at row {line_number}"
            ) from exc
        quality = KINECAL_TRACKING_QUALITY[state]
        if not np.isfinite(pixel_xy).all():
            pixel_xy[:] = np.nan
            quality = 0.0
        rows[joint] = (pixel_xy, quality)

    missing = [source for source, _ in KINECAL_TO_CANONICAL if source not in rows]
    if missing:
        raise ValueError(f"KINECAL frame is missing required joints: {missing}")

    xy = np.stack([rows[source][0] for source, _ in KINECAL_TO_CANONICAL])
    quality = np.asarray(
        [rows[source][1] for source, _ in KINECAL_TO_CANONICAL],
        dtype=np.float32,
    )
    pelvis = (xy[6] + xy[7]) / 2.0
    shoulders = (xy[0] + xy[1]) / 2.0
    xy = np.concatenate([xy, pelvis[None, :], shoulders[None, :]], axis=0)
    quality = np.concatenate(
        [
            quality,
            np.asarray(
                [min(quality[6], quality[7]), min(quality[0], quality[1])],
                dtype=np.float32,
            ),
        ]
    )
    return xy.astype(np.float32, copy=False), quality


def build_canonical_sequence(
    recording_dir: str | Path,
    *,
    target_fps: float = 30.0,
    max_gap_sec: float = 0.10,
) -> np.ndarray:
    directory = Path(recording_dir)
    frame_paths = sorted(directory.glob("*.txt"), key=_frame_timestamp)
    if not frame_paths:
        raise ValueError(f"no KINECAL skeleton frames found in {directory}")

    ticks = [_frame_timestamp(path) for path in frame_paths]
    if len(set(ticks)) != len(ticks):
        raise ValueError(f"duplicate KINECAL frame timestamps in {directory}")
    if any(current <= previous for previous, current in zip(ticks, ticks[1:])):
        raise ValueError(f"KINECAL frame timestamps are not increasing in {directory}")

    parsed: list[tuple[np.ndarray, np.ndarray]] = []
    for path in frame_paths:
        try:
            parsed.append(parse_kinecal_frame(path.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    source_xy = np.stack([item[0] for item in parsed]).astype(np.float32)
    source_quality = np.stack([item[1] for item in parsed]).astype(np.float32)
    source_times = np.asarray(
        [tick - ticks[0] for tick in ticks], dtype=np.float64
    ) * _TIMESTAMP_SCALE_SECONDS
    source_xy = _interpolate_missing_joints(source_xy, source_quality, source_times)

    target_times = _target_timestamps(source_times, target_fps)
    xy = _interpolate_array(source_xy, source_times, target_times)
    quality = _interpolate_array(
        source_quality[:, :, None], source_times, target_times
    )[:, :, 0]
    nearest_distance = _nearest_source_distance(source_times, target_times)
    quality[nearest_distance > max_gap_sec] = 0.0
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)

    pelvis = xy[:, 12:13, :]
    centered = xy - pelvis
    torso_lengths = np.linalg.norm(xy[:, 13, :] - xy[:, 12, :], axis=1)
    valid_scale = (
        (quality[:, 12] > 0.0)
        & (quality[:, 13] > 0.0)
        & np.isfinite(torso_lengths)
        & (torso_lengths > 1e-6)
    )
    if not valid_scale.any():
        raise ValueError(f"cannot estimate body scale for KINECAL recording {directory}")
    body_scale = float(np.median(torso_lengths[valid_scale]))
    centered = (centered / body_scale).astype(np.float32)
    centered[quality <= 0.0] = 0.0

    delta = np.zeros_like(centered)
    if len(centered) > 1:
        delta[1:] = centered[1:] - centered[:-1]
        valid_pair = (quality[1:] > 0.0) & (quality[:-1] > 0.0)
        delta[1:][~valid_pair] = 0.0

    features = np.concatenate(
        [centered, delta, quality[:, :, None]],
        axis=2,
    ).astype(np.float32)
    if features.shape[1:] != (len(CANONICAL_GAIT_JOINTS), len(GAIT_TCN_CHANNELS)):
        raise AssertionError(f"unexpected KINECAL gait tensor shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite normalized features in {directory}")
    return features


def make_windows(
    sequence: np.ndarray,
    *,
    window_frames: int,
    stride_frames: int,
) -> list[tuple[np.ndarray, int, int, int]]:
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [T, V, C]")
    if window_frames < 2 or stride_frames < 1:
        raise ValueError("invalid window or stride length")
    frame_count = int(sequence.shape[0])
    if frame_count < 1:
        raise ValueError("cannot window an empty sequence")
    if frame_count <= window_frames:
        padded = np.zeros(
            (window_frames, sequence.shape[1], sequence.shape[2]),
            dtype=np.float32,
        )
        padded[:frame_count] = sequence
        return [(padded, frame_count, 0, frame_count)]

    final_start = frame_count - window_frames
    starts = list(range(0, final_start + 1, stride_frames))
    if starts[-1] != final_start:
        starts.append(final_start)
    return [
        (
            sequence[start : start + window_frames].astype(np.float32, copy=True),
            window_frames,
            start,
            start + window_frames,
        )
        for start in starts
    ]


def stratified_participant_split(
    participant_labels: Mapping[str, int],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    if not participant_labels:
        raise ValueError("participant_labels cannot be empty")
    test_fraction = 1.0 - train_fraction - validation_fraction
    if train_fraction <= 0 or validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("train, validation and test fractions must all be positive")

    by_label: dict[int, list[str]] = defaultdict(list)
    for participant_id, label in participant_labels.items():
        by_label[int(label)].append(str(participant_id))
    if set(by_label) != {0, 1}:
        raise ValueError("binary participant labels must contain both classes 0 and 1")

    assignments: dict[str, str] = {}
    for label in sorted(by_label):
        participants = sorted(by_label[label])
        if len(participants) < 3:
            raise ValueError(
                f"class {label} needs at least 3 participants for train/validation/test"
            )
        random.Random(seed + (label * 1_000_003)).shuffle(participants)
        validation_count = max(1, int(round(len(participants) * validation_fraction)))
        test_count = max(1, int(round(len(participants) * test_fraction)))
        while validation_count + test_count >= len(participants):
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError(f"class {label} cannot be split into three partitions")
        train_count = len(participants) - validation_count - test_count
        for participant_id in participants[:train_count]:
            assignments[participant_id] = "train"
        validation_end = train_count + validation_count
        for participant_id in participants[train_count:validation_end]:
            assignments[participant_id] = "validation"
        for participant_id in participants[validation_end:]:
            assignments[participant_id] = "test"
    return assignments


def prepare_kinecal_gait_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    config: KinecalGaitPreparationConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    preparation = config or KinecalGaitPreparationConfig()
    source_root = Path(raw_dir)
    destination = Path(output_dir)
    register_path = source_root / "register.csv"
    if not register_path.is_file():
        raise FileNotFoundError(f"KINECAL register.csv not found: {register_path}")

    participants = select_risk_group_participants(
        read_register_csv(register_path.read_text(encoding="utf-8"))
    )
    available: list[tuple[dict[str, str], Path]] = []
    unavailable: list[dict[str, str]] = []
    for participant in participants:
        number = participant_number(participant["part_id"])
        recording_dir = (
            source_root
            / "kinecal"
            / number
            / f"{number}_{MOVEMENT}"
            / "skel"
        )
        if recording_dir.is_dir() and any(recording_dir.glob("*.txt")):
            available.append((participant, recording_dir))
        else:
            unavailable.append(
                {
                    "part_id": participant["part_id"],
                    "group": participant["group"],
                    "movement": MOVEMENT,
                    "reason": "source_recording_unavailable",
                }
            )

    participant_labels = {
        participant["part_id"]: FALL_HISTORY_PROXY_LABELS[participant["group"]]
        for participant, _ in available
    }
    assignments = stratified_participant_split(
        participant_labels,
        seed=preparation.seed,
        train_fraction=preparation.train_fraction,
        validation_fraction=preparation.validation_fraction,
    )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    participant_ids: list[str] = []
    groups: list[str] = []
    partitions: list[str] = []
    sample_ids: list[str] = []
    valid_lengths: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    manifest_rows: list[dict[str, Any]] = []

    available.sort(key=lambda item: int(participant_number(item[0]["part_id"])))
    for participant, recording_dir in available:
        participant_id = participant["part_id"]
        group = participant["group"]
        label = FALL_HISTORY_PROXY_LABELS[group]
        partition = assignments[participant_id]
        sequence = build_canonical_sequence(
            recording_dir,
            target_fps=preparation.target_fps,
            max_gap_sec=preparation.max_gap_sec,
        )
        windows = make_windows(
            sequence,
            window_frames=preparation.window_frames,
            stride_frames=preparation.stride_frames,
        )
        for window_index, (window, valid_length, start, end) in enumerate(windows):
            sample_id = f"{participant_id}:{MOVEMENT}:window-{window_index:03d}"
            sample_index = len(feature_rows)
            feature_rows.append(window)
            labels.append(label)
            participant_ids.append(participant_id)
            groups.append(group)
            partitions.append(partition)
            sample_ids.append(sample_id)
            valid_lengths.append(valid_length)
            starts.append(start)
            ends.append(end)
            manifest_rows.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "part_id": participant_id,
                    "source_group": group,
                    "label": label,
                    "partition": partition,
                    "movement": MOVEMENT,
                    "start_frame_30fps": start,
                    "end_frame_30fps": end,
                    "valid_length": valid_length,
                    "source_recording": recording_dir.relative_to(source_root).as_posix(),
                }
            )

    if not feature_rows:
        raise ValueError("no KINECAL gait windows were generated")
    features = np.stack(feature_rows).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("prepared KINECAL gait dataset contains non-finite features")

    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = destination / "dataset.npz"
    metadata_path = destination / "metadata.json"
    manifest_path = destination / "samples.jsonl"
    for path in (dataset_path, metadata_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {path}")

    _write_npz_atomic(
        dataset_path,
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        participant_ids=np.asarray(participant_ids),
        groups=np.asarray(groups),
        partitions=np.asarray(partitions),
        sample_ids=np.asarray(sample_ids),
        valid_lengths=np.asarray(valid_lengths, dtype=np.int64),
        start_frames=np.asarray(starts, dtype=np.int64),
        end_frames=np.asarray(ends, dtype=np.int64),
    )
    _write_jsonl_atomic(manifest_path, manifest_rows)

    participant_partition_counts = Counter(assignments.values())
    window_partition_counts = Counter(partitions)
    participant_class_counts = Counter(participant_labels.values())
    window_class_counts = Counter(labels)
    metadata: dict[str, Any] = {
        "schema_version": "kinecal-gait-tcn-v1",
        "dataset": "KINECAL",
        "source_version": "1.0.3",
        "task": "retrospective_fall_history_proxy_binary_classification",
        "movement": MOVEMENT,
        "label_mapping": dict(sorted(FALL_HISTORY_PROXY_LABELS.items())),
        "joint_order": list(CANONICAL_GAIT_JOINTS),
        "channel_order": list(GAIT_TCN_CHANNELS),
        "feature_shape": list(features.shape),
        "preparation_config": asdict(preparation),
        "participant_count": len(available),
        "window_count": len(features),
        "participant_partition_counts": {
            partition: participant_partition_counts.get(partition, 0)
            for partition in PARTITIONS
        },
        "window_partition_counts": {
            partition: window_partition_counts.get(partition, 0)
            for partition in PARTITIONS
        },
        "participant_class_counts": {
            str(label): participant_class_counts.get(label, 0) for label in (0, 1)
        },
        "window_class_counts": {
            str(label): window_class_counts.get(label, 0) for label in (0, 1)
        },
        "participant_assignments": dict(sorted(assignments.items())),
        "source_unavailable_recordings": unavailable,
        "age_group_mismatch": age_group_mismatches(participants),
        "register_sha256": _sha256_file(register_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "samples_manifest_sha256": _sha256_file(manifest_path),
        "limitations": [
            "fall history is a retrospective proxy, not a future fall outcome",
            "all windows from one participant remain in one partition",
            "KINECAL pixel skeletons are not an end-to-end RGB deployment test",
        ],
    }
    _write_json_atomic(metadata_path, metadata)
    return {
        "output_dir": destination.as_posix(),
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "participant_count": len(available),
        "window_count": len(features),
        "participant_partition_counts": metadata["participant_partition_counts"],
        "window_partition_counts": metadata["window_partition_counts"],
        "source_unavailable_recording_count": len(unavailable),
    }


def _frame_timestamp(path: Path) -> int:
    timestamp = path.stem.rsplit("_", 1)[-1]
    try:
        return int(timestamp)
    except ValueError as exc:
        raise ValueError(f"non-numeric KINECAL frame filename: {path.name}") from exc


def _parse_pixel_value(value: str) -> float:
    if "∞" in value:
        return math.nan
    return float(value)


def _interpolate_missing_joints(
    xy: np.ndarray,
    quality: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    output = xy.astype(np.float32, copy=True)
    for joint in range(output.shape[1]):
        valid = quality[:, joint] > 0.0
        if not valid.any():
            output[:, joint, :] = 0.0
            continue
        for coordinate in range(2):
            output[:, joint, coordinate] = np.interp(
                times,
                times[valid],
                output[valid, joint, coordinate],
            )
    return output


def _target_timestamps(source_times: np.ndarray, target_fps: float) -> np.ndarray:
    if len(source_times) == 1:
        return np.asarray([0.0], dtype=np.float64)
    duration = float(source_times[-1])
    frame_count = max(1, int(math.floor(duration * target_fps)) + 1)
    return np.arange(frame_count, dtype=np.float64) / target_fps


def _interpolate_array(
    values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    output = np.empty(
        (len(target_times), values.shape[1], values.shape[2]),
        dtype=np.float32,
    )
    for joint in range(values.shape[1]):
        for coordinate in range(values.shape[2]):
            output[:, joint, coordinate] = np.interp(
                target_times,
                source_times,
                values[:, joint, coordinate],
            )
    return output


def _nearest_source_distance(
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    insertions = np.searchsorted(source_times, target_times)
    left_indices = np.clip(insertions - 1, 0, len(source_times) - 1)
    right_indices = np.clip(insertions, 0, len(source_times) - 1)
    left = np.abs(target_times - source_times[left_indices])
    right = np.abs(source_times[right_indices] - target_times)
    return np.minimum(left, right)


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
