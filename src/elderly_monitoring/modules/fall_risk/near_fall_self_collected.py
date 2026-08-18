from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .near_fall_training import (
    NEAR_FALL_CHANNELS,
    NEAR_FALL_JOINTS,
    NearFallDatasetConfig,
    _has_causal_context,
    _left_pad_causal_tensor,
    _read_jsonl,
    _resample_causal_window,
    _window_quality,
    apply_normalization,
    build_near_fall_tensor,
    fit_normalization_statistics,
)


EXPERIMENT_ACTIONS = {
    "E1": {"A02", "A03", "A05", "A06", "A08", "A10"},
    "E2": {"A02", "A03", "A05", "A06", "A08", "A10", "C03"},
    "E3": {"A02", "A03", "A05", "A06", "A08", "A10", "C03", "C04", "C05"},
}
TRAIN_SUBJECTS = {"P01", "P02", "P04"}
CHALLENGE_SUBJECTS = {"P05"}
EXCLUDED_SUBJECTS = {"P03"}


def select_scf_near_fall_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    partition: str,
) -> list[dict[str, Any]]:
    if experiment not in EXPERIMENT_ACTIONS:
        raise ValueError(f"unsupported SCF near-fall experiment: {experiment}")
    if partition not in {"train", "challenge"}:
        raise ValueError(f"unsupported SCF partition: {partition}")
    subjects = TRAIN_SUBJECTS if partition == "train" else CHALLENGE_SUBJECTS
    selected: list[dict[str, Any]] = []
    for source in candidates:
        row = dict(source)
        if str(row.get("provisional_subject")) not in subjects:
            continue
        if str(row.get("source_action_id")) not in EXPERIMENT_ACTIONS[experiment]:
            continue
        if partition == "train" and row.get("loss_eligible") is not True:
            continue
        if str(row.get("disposition", "")).startswith("quarantine"):
            continue
        selected.append(row)
    return sorted(
        selected,
        key=lambda row: (
            str(row["sample_group_id"]),
            str(row["video_id"]),
            int(row["start_frame"]),
            str(row["candidate_id"]),
        ),
    )


def build_augmented_near_fall_dataset(
    *,
    base_dataset_path: str | Path,
    base_metadata_path: str | Path,
    base_samples_path: str | Path,
    candidates_path: str | Path,
    manifest_path: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    experiment: str,
    config: NearFallDatasetConfig | None = None,
) -> dict[str, Any]:
    preparation = config or NearFallDatasetConfig()
    base_dataset_file = Path(base_dataset_path)
    base_metadata_file = Path(base_metadata_path)
    base_samples_file = Path(base_samples_path)
    candidate_file = Path(candidates_path)
    manifest_file = Path(manifest_path)
    pose_root = Path(pose_dir)
    destination = Path(output_dir)
    inputs = (
        base_dataset_file,
        base_metadata_file,
        base_samples_file,
        candidate_file,
        manifest_file,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pose_root.is_dir():
        raise FileNotFoundError(pose_root)
    if destination.exists():
        raise FileExistsError(destination)

    metadata = json.loads(base_metadata_file.read_text(encoding="utf-8"))
    if metadata.get("dataset_sha256") != _sha256(base_dataset_file):
        raise ValueError("base near-fall dataset SHA-256 mismatch")
    if metadata.get("test_pose_read") is not False or metadata.get("test_evaluated") is not False:
        raise ValueError("base near-fall dataset violates locked-test policy")
    with np.load(base_dataset_file, allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    base_samples = _read_jsonl(base_samples_file)
    if len(base_samples) != len(base["features"]):
        raise ValueError("base near-fall samples/array length mismatch")
    candidates = select_scf_near_fall_candidates(
        _read_jsonl(candidate_file), experiment=experiment, partition="train"
    )
    manifest = {str(row["video_id"]): row for row in _read_jsonl(manifest_file)}
    scf_tensors, scf_samples, rejected = _build_scf_windows(
        candidates, manifest=manifest, pose_root=pose_root, config=preparation
    )
    if not scf_samples:
        raise ValueError(f"{experiment} produced no usable SCF training windows")

    base_raw = _reverse_normalization(
        base["features"], base["normalization_mean"], base["normalization_std"]
    )
    raw = np.concatenate([base_raw, np.stack(scf_tensors).astype(np.float32)])
    base_partitions = np.asarray(base["partitions"]).astype(str)
    partitions = np.concatenate(
        [base_partitions, np.full(len(scf_samples), "train", dtype="<U10")]
    )
    normalization = fit_normalization_statistics(raw, partitions)
    features = apply_normalization(raw, normalization)

    combined_samples = [dict(row) for row in base_samples] + scf_samples
    event_counts = Counter(str(row["event_id"]) for row in combined_samples)
    weights = np.asarray(
        [1.0 / event_counts[str(row["event_id"])] for row in combined_samples],
        dtype=np.float32,
    )
    arrays = {
        "features": features,
        "labels": np.concatenate(
            [np.asarray(base["labels"], dtype=np.int64), np.asarray([row["label"] for row in scf_samples], dtype=np.int64)]
        ),
        "partitions": partitions,
        "sample_ids": np.asarray([row["sample_id"] for row in combined_samples]),
        "event_ids": np.asarray([row["event_id"] for row in combined_samples]),
        "subject_ids": np.asarray([row["subject_id"] for row in combined_samples]),
        "source_group_ids": np.asarray([row["source_group_id"] for row in combined_samples]),
        "sample_group_ids": np.asarray([row["sample_group_id"] for row in combined_samples]),
        "split_group_ids": np.asarray([row["split_group_id"] for row in combined_samples]),
        "sample_weights": weights,
        "loss_eligible": np.ones(len(combined_samples), dtype=np.bool_),
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
    }
    validation_mask = base_partitions == "validation"
    original_validation_ids = np.asarray(base["sample_ids"]).astype(str)[validation_mask]
    augmented_validation_ids = arrays["sample_ids"][arrays["partitions"] == "validation"].astype(str)
    if not np.array_equal(original_validation_ids, augmented_validation_ids):
        raise ValueError("base validation sample membership changed")

    destination.mkdir(parents=True, exist_ok=False)
    dataset_path = destination / "dataset.npz"
    samples_path = destination / "samples.jsonl"
    metadata_path = destination / "metadata.json"
    _write_deterministic_npz(dataset_path, arrays)
    _write_jsonl(samples_path, combined_samples)
    augmented_metadata = {
        "schema_version": "near-fall-event-dataset-v1",
        "task": "near_fall_recovery_confirmation_v1",
        "status": "development_provisional",
        "synthetic": False,
        "training_ready": True,
        "formal_training_ready": False,
        "training_scope": "train_validation_development_only",
        "target_semantics": "binary recovery confirmation; SCF positives use a reviewed action-interval-end proxy",
        "negative_semantics": "base explicit negatives plus reviewed SCF hard-negative intervals",
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
        "sample_count": len(combined_samples),
        "event_count": len(event_counts),
        "partition_counts": dict(sorted(Counter(partitions.tolist()).items())),
        "locked_test_label_count": int(metadata.get("locked_test_label_count", 0)),
        "test_pose_read": False,
        "test_evaluated": False,
        "experiment": experiment,
        "self_collected_split_id": "scf-mvp-v1-dev-p010204-train-p05-challenge-p03-excluded-v1",
        "self_collected_train_subjects": sorted(TRAIN_SUBJECTS),
        "self_collected_challenge_subjects": sorted(CHALLENGE_SUBJECTS),
        "self_collected_excluded_subjects": sorted(EXCLUDED_SUBJECTS),
        "self_collected_candidate_count": len(candidates),
        "self_collected_window_count": len(scf_samples),
        "self_collected_action_window_counts": dict(
            sorted(Counter(str(row["source_action_id"]) for row in scf_samples).items())
        ),
        "self_collected_rejected_window_counts": dict(sorted(rejected.items())),
        "base_validation_sample_ids_sha256": _sha256_strings(original_validation_ids.tolist()),
        "base_validation_event_ids_sha256": _sha256_strings(
            np.asarray(base["event_ids"]).astype(str)[validation_mask].tolist()
        ),
        "normalization_refit_on_combined_train": True,
        "input_sha256": {
            "base_dataset": _sha256(base_dataset_file),
            "base_metadata": _sha256(base_metadata_file),
            "base_samples": _sha256(base_samples_file),
            "candidates": _sha256(candidate_file),
            "manifest": _sha256(manifest_file),
        },
        "samples_sha256": _sha256(samples_path),
        "dataset_sha256": _sha256(dataset_path),
        "limitations": [
            "SCF positive recovery anchors are coarse reviewed_action_interval_end_proxy values, not exact recovery timing",
            "P03 is excluded because its CVAT/media frame mapping remains unresolved",
            "P05 is challenge-only and never enters loss",
            "test poses and test truth are not read or materialized",
            "rules remain the runtime main path",
        ],
    }
    _write_json(metadata_path, augmented_metadata)
    return {
        "dataset_path": dataset_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "samples_path": samples_path.as_posix(),
        "sample_count": len(combined_samples),
        "self_collected_window_count": len(scf_samples),
        "dataset_sha256": augmented_metadata["dataset_sha256"],
    }


def build_scf_challenge_windows(
    *,
    candidates_path: str | Path,
    manifest_path: str | Path,
    pose_dir: str | Path,
    experiment: str = "E3",
    config: NearFallDatasetConfig | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    preparation = config or NearFallDatasetConfig()
    candidates = select_scf_near_fall_candidates(
        _read_jsonl(Path(candidates_path)), experiment=experiment, partition="challenge"
    )
    manifest = {
        str(row["video_id"]): row for row in _read_jsonl(Path(manifest_path))
    }
    tensors, samples, rejected = _build_scf_windows(
        candidates,
        manifest=manifest,
        pose_root=Path(pose_dir),
        config=preparation,
    )
    if not samples:
        raise ValueError("SCF challenge produced no usable windows")
    for sample in samples:
        sample["partition"] = "challenge"
        sample["loss_eligible"] = False
    return np.stack(tensors).astype(np.float32), samples, dict(sorted(rejected.items()))


def _build_scf_windows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Mapping[str, Any]],
    pose_root: Path,
    config: NearFallDatasetConfig,
) -> tuple[list[np.ndarray], list[dict[str, Any]], Counter[str]]:
    tensors: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for candidate in candidates:
        video_id = str(candidate["video_id"])
        manifest_row = manifest.get(video_id)
        if manifest_row is None:
            raise ValueError(f"SCF candidate lacks manifest row: {video_id}")
        pose_path = pose_root / f"{video_id}.jsonl"
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        records = _read_jsonl(pose_path)
        track_id = _dominant_track_id(candidate, records)
        if track_id is None:
            rejected["no_pose_track_in_reviewed_interval"] += 1
            continue
        track = sorted(
            (
                dict(row)
                for row in records
                if str(row.get("track_id", row.get("person_id", "unknown"))) == track_id
            ),
            key=lambda row: (float(row.get("timestamp_sec", 0.0)), int(row.get("frame_id", -1))),
        )
        anchors = _candidate_anchors(candidate, track, config)
        if not anchors:
            rejected["insufficient_causal_context"] += 1
            continue
        positive = str(candidate["candidate_role"]) == "positive_candidate"
        for window_index, anchor in enumerate(anchors):
            anchor_frame = int(anchor["frame_id"])
            anchor_time = float(anchor["timestamp_sec"])
            selected_window = None
            attempted_quality = False
            for context_sec in config.context_window_secs:
                context_config = replace(
                    config, window_sec=context_sec, fallback_window_secs=()
                )
                if not _has_causal_context(
                    track,
                    anchor_time_sec=anchor_time,
                    window_frames=context_config.window_frames,
                    target_fps=context_config.target_fps,
                ):
                    continue
                slots = _resample_causal_window(
                    track,
                    anchor_time_sec=anchor_time,
                    anchor_frame=anchor_frame,
                    config=context_config,
                )
                quality = _window_quality(slots, context_config.window_frames)
                attempted_quality = True
                if quality["observed_frame_count"] < config.min_observed_frames:
                    continue
                if (
                    quality["usable_frame_ratio"] < config.min_usable_frame_ratio
                    or quality["joint_coverage"] < config.min_joint_coverage
                    or quality["mean_joint_quality"] < config.min_mean_joint_quality
                ):
                    continue
                tensor = build_near_fall_tensor(
                    slots,
                    window_frames=context_config.window_frames,
                    target_fps=context_config.target_fps,
                )
                selected_window = (
                    context_sec,
                    context_config,
                    slots,
                    quality,
                    _left_pad_causal_tensor(tensor, config.window_frames),
                )
                break
            if selected_window is None:
                rejected[
                    "insufficient_quality" if attempted_quality else "insufficient_context"
                ] += 1
                continue
            context_sec, context_config, slots, quality, tensor = selected_window
            if not np.any(tensor[..., 7] > 0):
                rejected["empty_valid_mask"] += 1
                continue
            candidate_id = str(candidate["candidate_id"])
            event_id = f"scf_event_{candidate_id.removeprefix('scf_nearfall_')}"
            tensors.append(tensor)
            samples.append(
                {
                    "sample_id": f"{candidate_id}:window-{window_index:03d}",
                    "event_id": event_id,
                    "label_id": candidate_id,
                    "video_id": video_id,
                    "asset_id": str(manifest_row["asset_id"]),
                    "label": int(positive),
                    "target_name": "near_fall" if positive else "explicit_near_fall_negative",
                    "hard_negative_type": candidate.get("hard_negative_type"),
                    "source_action_id": str(candidate["source_action_id"]),
                    "partition": "train",
                    "subject_id": str(candidate["subject_id"]),
                    "source_group_id": str(candidate["source_group_id"]),
                    "sample_group_id": str(candidate["sample_group_id"]),
                    "split_group_id": str(candidate["sample_group_id"]),
                    "physical_event_id": event_id if positive else None,
                    "annotation_track_id": None,
                    "pose_track_id": track_id,
                    "track_match_method": "dominant_pose_track_in_reviewed_action_interval",
                    "dataset": "self_collected_scf_mvp_v1",
                    "anchor_reason": (
                        "reviewed_action_interval_end_proxy"
                        if positive
                        else "reviewed_negative_interval_anchor"
                    ),
                    "window_end_frame": anchor_frame,
                    "window_end_time_sec": anchor_time,
                    "context_sec": context_sec,
                    "context_frames": context_config.window_frames,
                    "context_length": quality["observed_frame_count"],
                    "window_frames": config.window_frames,
                    "padding_frames": config.window_frames
                    - context_config.window_frames,
                    "max_source_frame": max(
                        int(row["frame_id"]) for row in slots if row is not None
                    ),
                    "quality": quality,
                    "loss_eligible": True,
                }
            )
    return tensors, samples, rejected


def _dominant_track_id(
    candidate: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> str | None:
    start = int(candidate["start_frame"])
    end = int(candidate["end_frame_exclusive"])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        frame = int(row.get("frame_id", -1))
        if start <= frame < end:
            grouped[str(row.get("track_id", row.get("person_id", "unknown")))].append(row)
    if not grouped:
        return None
    return max(
        sorted(grouped),
        key=lambda key: (
            len(grouped[key]),
            float(np.mean([float(row.get("pose_confidence", 0.0)) for row in grouped[key]])),
        ),
    )


def _candidate_anchors(
    candidate: Mapping[str, Any],
    track: Sequence[Mapping[str, Any]],
    config: NearFallDatasetConfig,
) -> list[dict[str, Any]]:
    start = int(candidate["start_frame"])
    end = int(candidate["end_frame_exclusive"])
    interval = [dict(row) for row in track if start <= int(row["frame_id"]) < end]
    if not interval:
        return []
    if str(candidate["candidate_role"]) == "positive_candidate":
        return [max(interval, key=lambda row: (int(row["frame_id"]), float(row["timestamp_sec"])))]
    first_track_time = min(float(row["timestamp_sec"]) for row in track)
    span = (
        int(round(config.target_fps * min(config.context_window_secs))) - 1
    ) / config.target_fps
    eligible = [row for row in interval if float(row["timestamp_sec"]) >= first_track_time + span - 1e-9]
    if not eligible:
        return []
    targets = np.arange(
        float(eligible[0]["timestamp_sec"]),
        float(eligible[-1]["timestamp_sec"]) + 1e-9,
        config.stride_sec,
    ).tolist()
    if not targets or abs(targets[-1] - float(eligible[-1]["timestamp_sec"])) > 1e-6:
        targets.append(float(eligible[-1]["timestamp_sec"]))
    anchors: list[dict[str, Any]] = []
    for target in targets:
        row = min(eligible, key=lambda item: (abs(float(item["timestamp_sec"]) - target), int(item["frame_id"])))
        if not anchors or int(anchors[-1]["frame_id"]) != int(row["frame_id"]):
            anchors.append(dict(row))
    if len(anchors) > config.max_windows_per_event:
        indices = np.linspace(0, len(anchors) - 1, config.max_windows_per_event, dtype=np.int64)
        anchors = [anchors[index] for index in sorted(set(indices.tolist()))]
    return anchors


def _reverse_normalization(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    valid = output[..., 7] > 0
    for channel in range(7):
        raw = output[..., channel] * float(std[channel]) + float(mean[channel])
        output[..., channel] = np.where(valid, raw, 0.0)
    output[..., 7] = valid.astype(np.float32)
    return output


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    _write_text(path, text)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
