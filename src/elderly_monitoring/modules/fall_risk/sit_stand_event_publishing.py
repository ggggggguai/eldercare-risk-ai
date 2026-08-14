from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.sit_stand_event_labels import (
    SCHEMA_VERSION,
    validate_sit_stand_event_labels,
)


EVENT_ACTIONS = {
    "A03": ("stand_to_sit", "completed"),
    "A04": ("sit_to_stand", "completed"),
    "B06": ("sit_to_stand", "completed"),
    "C01": ("sit_to_stand", "failed"),
}
HARD_NEGATIVE_ACTIONS = {
    "A05": "controlled_squat",
    "A06": "controlled_bend",
    "A07": "bed_transfer_or_lying",
    "A09": "kneel_or_floor_activity",
    "D01": "fall_or_rapid_descent",
    "D02": "fall_or_rapid_descent",
    "D03": "fall_or_rapid_descent",
    "D05": "fall_or_rapid_descent",
}
AMBIGUOUS_IGNORE_ACTIONS = {"A11", "D04"}
SUPPORTED_ACTIONS = set(EVENT_ACTIONS) | set(HARD_NEGATIVE_ACTIONS) | AMBIGUOUS_IGNORE_ACTIONS
DEVELOPMENT_PARTITIONS = {"train", "validation"}


def publish_sit_stand_event_labels(
    action_labels: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    manifest: Iterable[Mapping[str, Any]],
    *,
    decision: Mapping[str, Any],
    source_action_labels_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish reviewed source intervals without inferring unlabelled background."""

    actions = [dict(row) for row in action_labels]
    assignment_index = {
        str(row.get("label_id")): dict(row)
        for row in assignments
        if _nonempty(row.get("label_id"))
    }
    manifest_index = {
        str(row.get("video_id")): dict(row)
        for row in manifest
        if _nonempty(row.get("video_id"))
    }
    _validate_decision(
        decision,
        actions,
        source_action_labels_sha256=source_action_labels_sha256,
    )

    labels: list[dict[str, Any]] = []
    review_log: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    locked_test_source_label_count = 0
    excluded_source_label_count = 0

    for source in sorted(actions, key=lambda row: str(row.get("label_id", ""))):
        action_id = str(source.get("action_id", ""))
        if action_id not in SUPPORTED_ACTIONS:
            continue
        source_label_id = _required_string(source, "label_id")
        assignment = assignment_index.get(source_label_id)
        if assignment is None:
            raise ValueError(f"sit-stand source label has no split assignment: {source_label_id}")
        partition = str(assignment.get("partition", ""))
        if partition == "test":
            locked_test_source_label_count += 1
            continue
        if partition not in DEVELOPMENT_PARTITIONS:
            raise ValueError(
                f"sit-stand source label has invalid partition {partition}: {source_label_id}"
            )
        video_id = _required_string(source, "video_id")
        media = manifest_index.get(video_id)
        if media is None or media.get("eligibility") is not True:
            excluded_source_label_count += 1
            continue
        annotator_id = _required_string(source, "annotator_id")
        reviewed_by = _reviewed_by(source, annotator_id)
        start_time = _finite_nonnegative(source.get("start_time"), "start_time")
        end_time = _finite_nonnegative(
            source.get("end_time_exclusive"), "end_time_exclusive"
        )
        if end_time <= start_time:
            raise ValueError(f"invalid source interval for {source_label_id}")

        tier = str(
            source.get("action_type_training_tier", source.get("training_tier", ""))
        )
        if tier not in {"primary", "auxiliary", "ignore"}:
            raise ValueError(f"invalid source training tier for {source_label_id}: {tier}")
        interval_type, transition_type, outcome, hard_negative_type = _semantics(
            action_id, tier
        )
        material = (
            f"{decision['decision_id']}|{source_label_id}|{interval_type}|"
            f"{start_time:.9f}|{end_time:.9f}"
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        attempt_count = 1 if interval_type == "event" else 0
        attempt_intervals = (
            [{"onset_time": start_time, "offset_time": end_time}]
            if interval_type == "event"
            else []
        )
        eligibility = (
            "ignore" if interval_type == "ignore" else ("eligible" if tier == "primary" else "auxiliary")
        )
        label = {
            "schema_version": SCHEMA_VERSION,
            "label_id": "sitstand_" + digest[:24],
            "event_id": "sitstand_interval_" + digest[24:48],
            "asset_id": _required_string(source, "asset_id"),
            "video_id": video_id,
            "subject_id": _required_string(source, "subject_id"),
            "source_group_id": _required_string(source, "source_group_id"),
            "sample_group_id": _required_string(source, "sample_group_id"),
            "split_group_id": _required_string(assignment, "split_group_id"),
            "interval_type": interval_type,
            "transition_type": transition_type,
            "onset_time": start_time,
            "offset_time": end_time,
            "boundary_precision": str(source.get("boundary_precision", "unknown")),
            "outcome": outcome,
            "attempt_count": attempt_count,
            "attempt_intervals": attempt_intervals,
            "seat_off_proxy": None,
            "seat_contact_proxy": None,
            "support_contact_proxy": None,
            "stabilization_time_proxy": None,
            "scene_region": str(media.get("scene_region") or "unknown"),
            "visibility": "uncertain",
            "quality_flags": _string_list(source.get("quality_flags", []), "quality_flags"),
            "review_status": str(source.get("review_status", "single_annotated")),
            "reviewed_by": reviewed_by,
            "eligibility": eligibility,
            "hard_negative_type": hard_negative_type,
            "source_action_label_ids": [source_label_id],
        }
        labels.append(label)
        for reviewer_id in reviewed_by:
            review_log.append(
                {
                    "schema_version": "sit-stand-event-review-log-v1",
                    "label_id": label["label_id"],
                    "reviewer_id": reviewer_id,
                    "role": "source_annotator" if reviewer_id == annotator_id else "source_reviewer",
                    "review_status": label["review_status"],
                    "source_action_label_id": source_label_id,
                    "decision_id": decision["decision_id"],
                    "evidence": "existing_human_action_annotation_reused_by_project_owner_confirmation",
                }
            )
        counts[interval_type] += 1
        partition_counts[partition] += 1
        if transition_type is not None:
            direction_counts[transition_type] += 1
        if hard_negative_type is not None:
            hard_negative_counts[hard_negative_type] += 1

    labels.sort(key=lambda row: str(row["label_id"]))
    review_log.sort(key=lambda row: (str(row["label_id"]), str(row["reviewer_id"])))
    validation = validate_sit_stand_event_labels(labels, review_log)
    return {
        "labels": labels,
        "review_log": review_log,
        "report": {
            "schema_version": "sit-stand-event-publication-report-v1",
            "decision_id": decision["decision_id"],
            "source_split_id": decision["source_split_id"],
            "source_action_labels_sha256": decision["source_action_labels_sha256"],
            "record_count": len(labels),
            "interval_type_counts": dict(sorted(counts.items())),
            "direction_counts": dict(sorted(direction_counts.items())),
            "hard_negative_counts": dict(sorted(hard_negative_counts.items())),
            "development_partition_counts": dict(sorted(partition_counts.items())),
            "locked_test_source_label_count": locked_test_source_label_count,
            "excluded_source_label_count": excluded_source_label_count,
            "unlabelled_background_inferred": False,
            "test_truth_published": False,
            "validation": validation,
        },
    }


def canonical_jsonl_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_decision(
    decision: Mapping[str, Any],
    actions: list[dict[str, Any]],
    *,
    source_action_labels_sha256: str | None,
) -> None:
    if decision.get("schema_version") != "sit-stand-source-label-decision-v1":
        raise ValueError("invalid sit-stand source label decision schema_version")
    for field in (
        "decision_id",
        "approved_by_role",
        "approval_source",
        "source_action_labels_sha256",
        "source_split_id",
    ):
        _required_string(decision, field)
    if decision.get("reuse_source_annotator_as_review_evidence") is not True:
        raise ValueError("decision must explicitly allow source annotator evidence reuse")
    actual_hash = source_action_labels_sha256 or canonical_jsonl_sha256(actions)
    if actual_hash != decision["source_action_labels_sha256"]:
        raise ValueError(
            "sit-stand source action label hash mismatch: "
            f"expected {decision['source_action_labels_sha256']}, got {actual_hash}"
        )


def _semantics(
    action_id: str, tier: str
) -> tuple[str, str | None, str | None, str | None]:
    if tier == "ignore" or action_id in AMBIGUOUS_IGNORE_ACTIONS:
        return "ignore", None, None, None
    if action_id in EVENT_ACTIONS:
        transition, outcome = EVENT_ACTIONS[action_id]
        return "event", transition, outcome, None
    return "explicit_background", None, None, HARD_NEGATIVE_ACTIONS[action_id]


def _reviewed_by(source: Mapping[str, Any], annotator_id: str) -> list[str]:
    reviewers = _string_list(source.get("reviewer_ids", []), "reviewer_ids")
    return list(dict.fromkeys([annotator_id, *reviewers]))


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(_nonempty(item) for item in value):
        raise ValueError(f"{field} must be a string list")
    return [str(item) for item in value]


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not _nonempty(value):
        raise ValueError(f"sit-stand publication requires non-empty {field}")
    return str(value)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not (result >= 0 and result < float("inf")):
        raise ValueError(f"{field} must be finite and non-negative")
    return result
