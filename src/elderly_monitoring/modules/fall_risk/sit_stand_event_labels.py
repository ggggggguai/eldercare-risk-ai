from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "fall-risk-sit-stand-event-label-v1"
INTERVAL_TYPES = {"event", "explicit_background", "ignore"}
TRANSITION_TYPES = {"sit_to_stand", "stand_to_sit"}
OUTCOMES = {"completed", "interrupted", "failed", "uncertain"}
BOUNDARY_PRECISIONS = {"exact", "approximate", "unknown"}
REVIEW_STATUSES = {
    "single_annotated",
    "single_reviewed",
    "double_reviewed",
    "adjudicated",
}
ELIGIBILITY_VALUES = {"eligible", "auxiliary", "ignore"}
VISIBILITY_VALUES = {"full", "partial", "not_visible", "uncertain"}
QUEUE_ACTION_IDS = {
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A09",
    "A11",
    "B06",
    "C01",
    "D01",
    "D02",
    "D03",
    "D04",
    "D05",
}
DATASET_PRIORITY = {
    "fall_detection_2017": 0,
    "caucafall": 1,
    "le2i_imvia": 2,
    "ur_fall": 3,
    "ntu_rgbd": 9,
}


def validate_sit_stand_event_labels(
    records: Iterable[Mapping[str, Any]],
    review_log_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate reviewed continuous labels without inferring unlabelled background."""

    rows = [dict(row) for row in records]
    reviews = [dict(row) for row in review_log_records]
    review_label_ids = {
        str(row.get("label_id"))
        for row in reviews
        if _nonempty_string(row.get("label_id"))
        and _nonempty_string(row.get("reviewer_id"))
    }
    seen_label_ids: set[str] = set()
    seen_event_ids: set[str] = set()
    counts: Counter[str] = Counter()
    background_duration = 0.0
    direction_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        prefix = f"sit-stand label {index}"
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{prefix} has invalid schema_version")
        for field in (
            "label_id",
            "event_id",
            "asset_id",
            "video_id",
            "subject_id",
            "source_group_id",
            "sample_group_id",
            "split_group_id",
            "scene_region",
        ):
            if not _nonempty_string(row.get(field)):
                raise ValueError(f"{prefix} requires non-empty {field}")
        label_id = str(row["label_id"])
        event_id = str(row["event_id"])
        if label_id in seen_label_ids:
            raise ValueError(f"duplicate sit-stand label_id: {label_id}")
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate sit-stand event_id: {event_id}")
        seen_label_ids.add(label_id)
        seen_event_ids.add(event_id)
        if label_id not in review_label_ids:
            raise ValueError(f"{prefix} has no matching human review log record")

        interval_type = str(row.get("interval_type"))
        if interval_type not in INTERVAL_TYPES:
            raise ValueError(f"{prefix} has invalid interval_type")
        onset = _finite_nonnegative(row.get("onset_time"), f"{prefix}.onset_time")
        offset = _finite_nonnegative(row.get("offset_time"), f"{prefix}.offset_time")
        if offset <= onset:
            raise ValueError(f"{prefix} offset_time must be greater than onset_time")
        if row.get("boundary_precision") not in BOUNDARY_PRECISIONS:
            raise ValueError(f"{prefix} has invalid boundary_precision")
        if row.get("review_status") not in REVIEW_STATUSES:
            raise ValueError(f"{prefix} has invalid review_status")
        reviewed_by = row.get("reviewed_by")
        if not isinstance(reviewed_by, list) or not all(
            _nonempty_string(value) for value in reviewed_by
        ):
            raise ValueError(f"{prefix}.reviewed_by must be a non-empty string list")
        if row.get("eligibility") not in ELIGIBILITY_VALUES:
            raise ValueError(f"{prefix} has invalid eligibility")
        if row.get("visibility") not in VISIBILITY_VALUES:
            raise ValueError(f"{prefix} has invalid visibility")
        quality_flags = row.get("quality_flags")
        if not isinstance(quality_flags, list) or not all(
            _nonempty_string(value) for value in quality_flags
        ):
            raise ValueError(f"{prefix}.quality_flags must be a string list")
        source_ids = row.get("source_action_label_ids")
        if not isinstance(source_ids, list) or not all(
            _nonempty_string(value) for value in source_ids
        ):
            raise ValueError(
                f"{prefix}.source_action_label_ids must be a string list"
            )

        transition = row.get("transition_type")
        outcome = row.get("outcome")
        attempt_count = row.get("attempt_count")
        attempts = row.get("attempt_intervals")
        if not isinstance(attempt_count, int) or isinstance(attempt_count, bool):
            raise ValueError(f"{prefix}.attempt_count must be an integer")
        if not isinstance(attempts, list):
            raise ValueError(f"{prefix}.attempt_intervals must be a list")
        if attempt_count != len(attempts):
            raise ValueError(f"{prefix}.attempt_count does not match intervals")

        if interval_type == "event":
            if transition not in TRANSITION_TYPES:
                raise ValueError(f"{prefix} event requires transition_type")
            if outcome not in OUTCOMES:
                raise ValueError(f"{prefix} event requires a valid outcome")
            if attempt_count < 1:
                raise ValueError(f"{prefix} event requires at least one attempt")
            _validate_attempts(attempts, onset, offset, prefix)
            direction_counts[str(transition)] += 1
            outcome_counts[str(outcome)] += 1
        elif interval_type == "explicit_background":
            if transition is not None or outcome is not None:
                raise ValueError(
                    f"{prefix} explicit_background cannot carry transition or outcome"
                )
            if attempt_count != 0 or attempts:
                raise ValueError(
                    f"{prefix} explicit_background cannot carry attempt semantics"
                )
            background_duration += offset - onset
        else:
            if attempt_count < 0:
                raise ValueError(f"{prefix} ignore attempt_count cannot be negative")
            if attempts:
                _validate_attempts(attempts, onset, offset, prefix)

        counts[interval_type] += 1
        review_counts[str(row["review_status"])] += 1

    return {
        "schema_version": "sit-stand-event-label-validation-v1",
        "valid": True,
        "record_count": len(rows),
        "counts": {name: int(counts[name]) for name in sorted(INTERVAL_TYPES)},
        "direction_counts": dict(sorted(direction_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "review_status_counts": dict(sorted(review_counts.items())),
        "explicit_background_duration_sec": round(background_duration, 6),
        "gates": {
            "event_localization_supervision_available": counts["event"] > 0,
            "fp_hour_denominator_available": background_duration > 0,
            "human_review_evidence_available": bool(rows) and len(review_label_ids) > 0,
        },
        "test_access": {
            "test_pose_read": False,
            "test_features_generated": False,
            "test_evaluated": False,
        },
    }


def build_sit_stand_review_queue(
    action_labels: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    manifest: Iterable[Mapping[str, Any]],
    *,
    context_sec: float = 3.0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic train/validation review candidates without truth synthesis."""

    if context_sec < 0 or not math.isfinite(context_sec):
        raise ValueError("context_sec must be finite and non-negative")
    assignment_index = {
        str(row["label_id"]): dict(row)
        for row in assignments
        if _nonempty_string(row.get("label_id"))
    }
    manifest_index = {
        str(row["video_id"]): dict(row)
        for row in manifest
        if _nonempty_string(row.get("video_id"))
    }
    queue: list[dict[str, Any]] = []
    for source in action_labels:
        action_id = str(source.get("action_id", ""))
        if action_id not in QUEUE_ACTION_IDS:
            continue
        label_id = str(source.get("label_id", ""))
        assignment = assignment_index.get(label_id)
        if assignment is None:
            raise ValueError(f"review candidate is missing split assignment: {label_id}")
        partition = str(assignment.get("partition", ""))
        if partition == "test":
            continue
        if partition not in {"train", "validation"}:
            raise ValueError(f"review candidate has invalid partition: {label_id}")
        video_id = str(source.get("video_id", ""))
        media = manifest_index.get(video_id)
        if media is None or media.get("eligibility") is not True:
            continue
        start_time = _finite_nonnegative(
            source.get("start_time"), f"{label_id}.start_time"
        )
        end_time = _finite_nonnegative(
            source.get("end_time_exclusive"), f"{label_id}.end_time_exclusive"
        )
        if end_time <= start_time:
            raise ValueError(f"review candidate has invalid interval: {label_id}")
        duration = _optional_finite_nonnegative(media.get("duration_sec"))
        context_start = max(0.0, start_time - context_sec)
        context_end = end_time + context_sec
        if duration is not None:
            context_end = min(duration, context_end)
        material = f"{label_id}|{partition}|{context_start:.6f}|{context_end:.6f}"
        queue_id = "sitstand_review_" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()[:24]
        queue.append(
            {
                "schema_version": "sit-stand-event-review-queue-v1",
                "queue_id": queue_id,
                "source_action_label_id": label_id,
                "asset_id": str(source.get("asset_id", "")),
                "video_id": video_id,
                "dataset": str(media.get("dataset", "unknown")),
                "scene_region": str(media.get("scene_region", "unknown")),
                "action_id": action_id,
                "partition": partition,
                "subject_id": str(source.get("subject_id", "unknown")),
                "source_group_id": str(source.get("source_group_id", "unknown")),
                "sample_group_id": str(source.get("sample_group_id", "unknown")),
                "split_group_id": str(assignment.get("split_group_id", "unknown")),
                "source_start_frame": source.get("start_frame"),
                "source_end_frame_exclusive": source.get("end_frame_exclusive"),
                "source_start_time": start_time,
                "source_end_time_exclusive": end_time,
                "context_start_time": round(context_start, 6),
                "context_end_time_exclusive": round(context_end, 6),
                "review_tasks": [
                    "confirm_event_or_hard_negative_or_ignore",
                    "mark_true_onset_offset_and_direction",
                    "confirm_explicit_background_only_when_observed",
                    "record_visibility_quality_and_tracking_conflicts",
                ],
                "truth_prefilled": False,
                "test_pose_read": False,
            }
        )
    queue.sort(
        key=lambda row: (
            DATASET_PRIORITY.get(str(row["dataset"]), 8),
            0 if row["partition"] == "validation" else 1,
            str(row["video_id"]),
            float(row["source_start_time"]),
            str(row["source_action_label_id"]),
        )
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        queue = queue[:limit]
    return queue


def _validate_attempts(
    attempts: Sequence[Mapping[str, Any]], onset: float, offset: float, prefix: str
) -> None:
    previous_offset = onset
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise ValueError(f"{prefix} attempt interval {index} must be an object")
        attempt_onset = _finite_nonnegative(
            attempt.get("onset_time"), f"{prefix}.attempt[{index}].onset_time"
        )
        attempt_offset = _finite_nonnegative(
            attempt.get("offset_time"), f"{prefix}.attempt[{index}].offset_time"
        )
        if not (onset <= attempt_onset < attempt_offset <= offset):
            raise ValueError(f"{prefix} attempt interval must be inside the event")
        if attempt_onset < previous_offset:
            raise ValueError(f"{prefix} attempt intervals must not overlap")
        previous_offset = attempt_offset


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _optional_finite_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, "duration_sec")


def canonical_jsonl_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
