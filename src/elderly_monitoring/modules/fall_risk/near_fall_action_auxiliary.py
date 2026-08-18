"""Governed auxiliary negatives from reviewed A01/A04 action intervals.

These labels are useful negative evidence for near-fall confirmation, but they are
single-annotated action labels rather than double-reviewed near-fall negatives.
The builder therefore keeps them in a separate supervision tier and applies a
small loss weight. Test rows are audited and locked without reading pose data.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .near_fall_training import (
    NEAR_FALL_CHANNELS,
    NEAR_FALL_JOINTS,
    NearFallDatasetConfig,
    _has_causal_context,
    _left_pad_causal_tensor,
    _manifest_index,
    _read_jsonl,
    _resample_causal_window,
    _select_labeled_track,
    _validate_development_samples,
    _validate_manifest_link,
    _window_anchor_frames,
    _window_quality,
    apply_normalization,
    build_near_fall_tensor,
    fit_normalization_statistics,
)


AUXILIARY_ACTION_IDS = {"A01", "A04"}
DEFAULT_AUXILIARY_LOSS_WEIGHT = 0.25


def select_action_auxiliary_labels(
    action_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select reviewed A01/A04 primary action labels and lock test rows."""

    assignments = {str(row.get("label_id")): dict(row) for row in assignment_rows}
    selected: list[dict[str, Any]] = []
    locked_test: list[dict[str, Any]] = []
    for source in action_rows:
        row = dict(source)
        label_id = str(row.get("label_id", ""))
        assignment = assignments.get(label_id)
        if row.get("schema_version") != "fall-risk-action-label-v3":
            continue
        if str(row.get("action_id")) not in AUXILIARY_ACTION_IDS:
            continue
        if row.get("training_tier") != "primary" or row.get("action_type_training_tier") != "primary":
            continue
        if row.get("review_status") != "single_annotated" or row.get("target_status") != "confirmed":
            continue
        if row.get("quality_flags"):
            continue
        if row.get("linked_event_id") is not None:
            continue
        if assignment is None:
            raise ValueError(f"auxiliary action is missing split assignment: {label_id}")
        if assignment.get("label_kind") != "action" or assignment.get("task_type") != "action":
            raise ValueError(f"auxiliary action has invalid split assignment: {label_id}")
        partition = str(assignment.get("partition"))
        if partition not in {"train", "validation", "test"}:
            raise ValueError(f"auxiliary action has invalid partition: {label_id}")
        for field in ("video_id", "subject_id", "source_group_id", "sample_group_id"):
            if str(row.get(field)) != str(assignment.get(field)):
                raise ValueError(f"auxiliary action {field} mismatch: {label_id}")
        row["partition"] = partition
        row["split_group_id"] = str(assignment.get("split_group_id") or "")
        row["asset_id"] = str(row.get("asset_id") or assignment.get("asset_id") or "")
        row["label"] = 0
        row["target_name"] = "auxiliary_action_negative"
        row["supervision_tier"] = "auxiliary"
        row["source_action_id"] = str(row["action_id"])
        row["action_label_id"] = label_id
        if partition == "test":
            locked_test.append(row)
        else:
            selected.append(row)
    return sorted(selected, key=_label_sort_key), sorted(locked_test, key=_label_sort_key)


def build_action_auxiliary_near_fall_dataset(
    *,
    base_dataset_path: str | Path,
    base_metadata_path: str | Path,
    base_samples_path: str | Path,
    action_labels_path: str | Path,
    assignments_path: str | Path,
    manifest_path: str | Path,
    pose_dir: str | Path,
    output_dir: str | Path,
    config: NearFallDatasetConfig | None = None,
    auxiliary_loss_weight: float = DEFAULT_AUXILIARY_LOSS_WEIGHT,
) -> dict[str, Any]:
    """Append train/validation A01/A04 auxiliary negative windows to a dataset."""

    if not 0 < auxiliary_loss_weight < 1:
        raise ValueError("auxiliary_loss_weight must be within (0, 1)")
    preparation = config or NearFallDatasetConfig()
    base_path = Path(base_dataset_path)
    metadata_path = Path(base_metadata_path)
    samples_path = Path(base_samples_path)
    action_path = Path(action_labels_path)
    assignments_path = Path(assignments_path)
    manifest_path = Path(manifest_path)
    pose_root = Path(pose_dir)
    destination = Path(output_dir)
    for path in (base_path, metadata_path, samples_path, action_path, assignments_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pose_root.is_dir():
        raise FileNotFoundError(pose_root)
    if destination.exists():
        raise FileExistsError(destination)

    base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if base_metadata.get("dataset_sha256") != _sha256(base_path):
        raise ValueError("base near-fall dataset SHA-256 mismatch")
    if base_metadata.get("test_pose_read") is not False or base_metadata.get("test_evaluated") is not False:
        raise ValueError("base near-fall dataset violates locked-test policy")
    with np.load(base_path, allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    base_samples = _read_jsonl(samples_path)
    if len(base_samples) != len(base["features"]):
        raise ValueError("base near-fall samples/array length mismatch")

    action_rows = _read_jsonl(action_path)
    assignment_rows = _read_jsonl(assignments_path)
    selected, locked_test = select_action_auxiliary_labels(action_rows, assignment_rows)
    manifest_index = _manifest_index(_read_jsonl(manifest_path))
    tensors, aux_samples, audits, rejected, pose_paths = _build_action_windows(
        selected,
        manifest_index=manifest_index,
        pose_root=pose_root,
        config=preparation,
        auxiliary_loss_weight=auxiliary_loss_weight,
    )
    if not aux_samples:
        raise ValueError("A01/A04 auxiliary action labels produced no usable windows")
    training_indices = [
        index for index, row in enumerate(aux_samples) if row["partition"] == "train"
    ]
    validation_indices = [
        index
        for index, row in enumerate(aux_samples)
        if row["partition"] == "validation"
    ]
    if not training_indices:
        raise ValueError("A01/A04 auxiliary actions produced no train windows")
    training_tensors = [tensors[index] for index in training_indices]
    training_aux_samples = [aux_samples[index] for index in training_indices]
    validation_tensors = [tensors[index] for index in validation_indices]
    validation_aux_samples = [aux_samples[index] for index in validation_indices]

    base_raw = _reverse_normalization(
        base["features"], base["normalization_mean"], base["normalization_std"]
    )
    raw = np.concatenate(
        [base_raw, np.stack(training_tensors).astype(np.float32)]
    )
    base_partitions = np.asarray(base["partitions"]).astype(str)
    partitions = np.concatenate(
        [base_partitions, np.full(len(training_aux_samples), "train", dtype="<U10")]
    )
    normalization = fit_normalization_statistics(raw, partitions)
    features = apply_normalization(raw, normalization)

    combined_samples = [dict(row) for row in base_samples] + training_aux_samples
    for sample in combined_samples[: len(base_samples)]:
        sample.setdefault("supervision_tier", "primary")
        sample.setdefault("loss_weight", 1.0)
    event_counts = Counter(str(row["event_id"]) for row in combined_samples)
    sample_weights = np.asarray(
        [1.0 / event_counts[str(row["event_id"])] for row in combined_samples],
        dtype=np.float32,
    )
    loss_weights = np.asarray(
        [float(row.get("loss_weight", 1.0)) * weight for row, weight in zip(combined_samples, sample_weights, strict=True)],
        dtype=np.float32,
    )
    arrays = {
        "features": features,
        "labels": np.concatenate([
            np.asarray(base["labels"], dtype=np.int64),
            np.asarray([row["label"] for row in training_aux_samples], dtype=np.int64),
        ]),
        "partitions": partitions,
        "sample_ids": np.asarray([row["sample_id"] for row in combined_samples]),
        "event_ids": np.asarray([row["event_id"] for row in combined_samples]),
        "subject_ids": np.asarray([row["subject_id"] for row in combined_samples]),
        "source_group_ids": np.asarray([row["source_group_id"] for row in combined_samples]),
        "sample_group_ids": np.asarray([row["sample_group_id"] for row in combined_samples]),
        "split_group_ids": np.asarray([row["split_group_id"] for row in combined_samples]),
        "sample_weights": sample_weights,
        "loss_weights": loss_weights,
        "loss_eligible": np.ones(len(combined_samples), dtype=np.bool_),
        "normalization_mean": normalization["mean"],
        "normalization_std": normalization["std"],
    }
    _validate_development_samples(combined_samples)
    _validate_event_weights(combined_samples, sample_weights)

    destination.mkdir(parents=True, exist_ok=False)
    dataset_file = destination / "dataset.npz"
    output_samples = destination / "samples.jsonl"
    output_audit = destination / "audit.jsonl"
    output_metadata = destination / "metadata.json"
    validation_dataset_file = destination / "auxiliary_validation_dataset.npz"
    validation_samples_file = destination / "auxiliary_validation_samples.jsonl"
    _write_deterministic_npz(dataset_file, arrays)
    _write_jsonl(output_samples, combined_samples)
    validation_features = (
        apply_normalization(
            np.stack(validation_tensors).astype(np.float32), normalization
        )
        if validation_tensors
        else np.zeros((0, preparation.window_frames, len(NEAR_FALL_JOINTS), len(NEAR_FALL_CHANNELS)), dtype=np.float32)
    )
    _write_deterministic_npz(
        validation_dataset_file,
        {
            "features": validation_features,
            "labels": np.zeros(len(validation_aux_samples), dtype=np.int64),
            "sample_ids": np.asarray(
                [row["sample_id"] for row in validation_aux_samples]
            ),
            "event_ids": np.asarray(
                [row["event_id"] for row in validation_aux_samples]
            ),
            "source_action_ids": np.asarray(
                [row["source_action_id"] for row in validation_aux_samples]
            ),
        },
    )
    _write_jsonl(validation_samples_file, validation_aux_samples)
    all_audits = [
        {"label_id": str(row.get("label_id")), "video_id": str(row.get("video_id")), "partition": "test", "status": "locked_test_not_read", "supervision_tier": "auxiliary"}
        for row in locked_test
    ] + audits
    _write_jsonl(output_audit, all_audits)
    output_metadata_payload = {
        "schema_version": "near-fall-event-dataset-v2",
        "task": "near_fall_recovery_confirmation_v1",
        "status": "development_provisional",
        "synthetic": False,
        "training_ready": True,
        "formal_training_ready": False,
        "training_scope": "train_validation_development_only",
        "target_semantics": base_metadata.get("target_semantics"),
        "negative_semantics": "primary explicit negatives plus low-weight auxiliary A01/A04 action intervals",
        "label_mapping": {"explicit_negative": 0, "near_fall": 1},
        "joint_order": list(NEAR_FALL_JOINTS),
        "channel_order": list(NEAR_FALL_CHANNELS),
        "feature_shape": list(features.shape),
        "preparation_config": asdict(preparation),
        "normalization": {"fit_partition": "train", "mean": normalization["mean"].tolist(), "std": normalization["std"].tolist(), "valid_mask_normalized": False},
        "sample_count": len(combined_samples),
        "event_count": len(event_counts),
        "partition_counts": dict(sorted(Counter(partitions.tolist()).items())),
        "locked_test_label_count": int(base_metadata.get("locked_test_label_count", 0)) + len(locked_test),
        "test_pose_read": False,
        "test_evaluated": False,
        "auxiliary_action_ids": sorted(AUXILIARY_ACTION_IDS),
        "auxiliary_label_count": len(selected) + len(locked_test),
        "auxiliary_materialized_label_count": len({str(row["action_label_id"]) for row in aux_samples}),
        "auxiliary_window_count": len(aux_samples),
        "auxiliary_training_window_count": len(training_aux_samples),
        "auxiliary_validation_challenge_window_count": len(validation_aux_samples),
        "auxiliary_validation_enters_loss": False,
        "auxiliary_validation_enters_threshold_calibration": False,
        "auxiliary_loss_weight": auxiliary_loss_weight,
        "auxiliary_partition_counts": dict(sorted(Counter(row["partition"] for row in aux_samples).items())),
        "auxiliary_action_window_counts": dict(sorted(Counter(str(row["source_action_id"]) for row in aux_samples).items())),
        "auxiliary_rejected_label_counts": dict(sorted(rejected.items())),
        "input_sha256": {"base_dataset": _sha256(base_path), "base_metadata": _sha256(metadata_path), "base_samples": _sha256(samples_path), "action_labels": _sha256(action_path), "assignments": _sha256(assignments_path), "manifest": _sha256(manifest_path)},
        "pose_inputs": {video_id: {"path": path.as_posix(), "sha256": _sha256(path)} for video_id, path in sorted(pose_paths.items())},
        "test_action_labels_locked": True,
        "limitations": ["A01/A04 are single-annotated action intervals and auxiliary only", "test action labels are locked and no test pose or truth is read", "rules remain the runtime main path"],
    }
    output_metadata_payload["samples_sha256"] = _sha256(output_samples)
    output_metadata_payload["audit_sha256"] = _sha256(output_audit)
    output_metadata_payload["auxiliary_validation_dataset_sha256"] = _sha256(
        validation_dataset_file
    )
    output_metadata_payload["auxiliary_validation_samples_sha256"] = _sha256(
        validation_samples_file
    )
    output_metadata_payload["dataset_sha256"] = _sha256(dataset_file)
    _write_json(output_metadata, output_metadata_payload)
    return {"dataset_path": dataset_file.as_posix(), "metadata_path": output_metadata.as_posix(), "samples_path": output_samples.as_posix(), "audit_path": output_audit.as_posix(), "sample_count": len(combined_samples), "auxiliary_window_count": len(aux_samples), "auxiliary_training_window_count": len(training_aux_samples), "auxiliary_validation_challenge_window_count": len(validation_aux_samples), "locked_test_label_count": len(locked_test), "dataset_sha256": output_metadata_payload["dataset_sha256"]}


def _build_action_windows(
    labels: Sequence[Mapping[str, Any]],
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    pose_root: Path,
    config: NearFallDatasetConfig,
    auxiliary_loss_weight: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]], Counter[str], dict[str, Path]]:
    tensors: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    pose_paths: dict[str, Path] = {}
    for source in labels:
        label = dict(source)
        label_id = str(label["label_id"])
        video_id = str(label["video_id"])
        audit = {"label_id": label_id, "video_id": video_id, "partition": str(label["partition"]), "supervision_tier": "auxiliary", "status": "pending", "materialized_window_count": 0, "context_sec_counts": {}, "rejection_counts": {}}
        manifest_row = manifest_index.get(video_id)
        if manifest_row is None:
            raise ValueError(f"auxiliary action references missing manifest video: {video_id}")
        _validate_manifest_link(label, manifest_row)
        pose_path = pose_root / f"{video_id}.jsonl"
        if not pose_path.is_file():
            rejected["no_pose_file"] += 1
            audit["status"] = "rejected"
            audit["rejection_counts"] = {"no_pose_file": 1}
            audits.append(audit)
            continue
        pose_paths[video_id] = pose_path
        track = _select_labeled_track({**label, "label": 0}, _read_jsonl(pose_path))
        if not track:
            rejected["no_pose_track"] += 1
            audit["status"] = "rejected"
            audit["rejection_counts"] = {"no_pose_track": 1}
            audits.append(audit)
            continue
        anchors = _window_anchor_frames({**label, "label": 0}, track, config)
        if not anchors:
            rejected["insufficient_context"] += 1
            audit["status"] = "rejected"
            audit["rejection_counts"] = {"insufficient_context": 1}
            audits.append(audit)
            continue
        context_counts: Counter[str] = Counter()
        audit_rejected: Counter[str] = Counter()
        for window_index, anchor_frame in enumerate(anchors):
            anchor = track[anchor_frame]
            anchor_time = float(anchor["timestamp_sec"])
            selected_window = None
            attempted_quality = False
            for context_sec in config.context_window_secs:
                context_config = replace(config, window_sec=context_sec, fallback_window_secs=())
                if not _has_causal_context(track.values(), anchor_time_sec=anchor_time, window_frames=context_config.window_frames, target_fps=context_config.target_fps):
                    continue
                slots = _resample_causal_window(list(track.values()), anchor_time_sec=anchor_time, anchor_frame=anchor_frame, config=context_config)
                quality = _window_quality(slots, context_config.window_frames)
                attempted_quality = True
                if quality["observed_frame_count"] < config.min_observed_frames or quality["usable_frame_ratio"] < config.min_usable_frame_ratio or quality["joint_coverage"] < config.min_joint_coverage or quality["mean_joint_quality"] < config.min_mean_joint_quality:
                    continue
                tensor = _left_pad_causal_tensor(build_near_fall_tensor(slots, window_frames=context_config.window_frames, target_fps=context_config.target_fps), config.window_frames)
                selected_window = (context_sec, context_config, slots, quality, tensor)
                break
            if selected_window is None:
                reason = "insufficient_quality" if attempted_quality else "insufficient_context"
                rejected[reason] += 1
                audit_rejected[reason] += 1
                continue
            context_sec, context_config, slots, quality, tensor = selected_window
            if not np.any(tensor[..., 7] > 0):
                rejected["empty_valid_mask"] += 1
                audit_rejected["empty_valid_mask"] += 1
                continue
            event_id = f"aux_action_event_{label_id}"
            sample = {"sample_id": f"{label_id}:window-{window_index:03d}", "event_id": event_id, "label_id": label_id, "action_label_id": label_id, "source_action_id": str(label["action_id"]), "video_id": video_id, "asset_id": str(label["asset_id"]), "label": 0, "target_name": "auxiliary_action_negative", "hard_negative_type": f"auxiliary_{label['action_id']}", "supervision_tier": "auxiliary", "loss_weight": auxiliary_loss_weight, "partition": str(label["partition"]), "subject_id": str(label["subject_id"]), "source_group_id": str(label["source_group_id"]), "sample_group_id": str(label["sample_group_id"]), "split_group_id": str(label["split_group_id"]), "physical_event_id": None, "annotation_track_id": str(label.get("track_id") or "") or None, "pose_track_id": str(next(iter({str(row.get('track_id', row.get('person_id', 'unknown'))) for row in track.values()}))), "track_match_method": "dominant_pose_track_in_action_interval", "dataset": str(manifest_row.get("dataset", "unknown")), "anchor_reason": "action_interval_window_end", "window_end_frame": int(anchor_frame), "window_end_time_sec": anchor_time, "max_source_frame": max(int(row["frame_id"]) for row in slots if row is not None), "context_sec": context_sec, "context_frames": context_config.window_frames, "context_length": quality["observed_frame_count"], "window_frames": config.window_frames, "padding_frames": config.window_frames - context_config.window_frames, "quality": quality, "loss_eligible": True}
            tensors.append(tensor)
            samples.append(sample)
            context_counts[str(context_sec)] += 1
        audit["materialized_window_count"] = sum(context_counts.values())
        audit["context_sec_counts"] = dict(sorted(context_counts.items()))
        audit["rejection_counts"] = dict(sorted(audit_rejected.items()))
        audit["status"] = "materialized" if context_counts else "rejected"
        audits.append(audit)
    return tensors, samples, audits, rejected, pose_paths


def _label_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (str(row.get("partition")), str(row.get("video_id")), int(row.get("start_frame", 0)), str(row.get("label_id")))


def _reverse_normalization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    output = np.asarray(features, dtype=np.float32).copy()
    valid = output[..., 7] > 0
    for channel in range(7):
        raw = output[..., channel] * float(std[channel]) + float(mean[channel])
        output[..., channel] = np.where(valid, raw, 0.0)
    output[..., 7] = valid.astype(np.float32)
    return output


def _validate_event_weights(samples: Sequence[Mapping[str, Any]], weights: np.ndarray) -> None:
    event_ids = np.asarray([str(row["event_id"]) for row in samples])
    for event_id in np.unique(event_ids):
        if not np.isclose(float(weights[event_ids == event_id].sum()), 1.0, atol=1e-6):
            raise ValueError(f"near-fall event weights do not sum to 1: {event_id}")


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
