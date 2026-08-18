from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.ntu_rgbd_cvat import (
    load_ntu_rgbd_a043_decision,
)


ACTION_SCHEMA_VERSION = "fall-risk-action-label-v3"
EVENT_SCHEMA_VERSION = "fall-risk-event-label-v3"
SPLIT_SCHEMA_VERSION = "fall-risk-training-split-v3"
REVIEWED_DECISION_SCHEMA_VERSION = "fall-risk-reviewed-training-decision-v1"
SPLIT_PARTITIONS = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}

_UNKNOWN_IDENTIFIERS = {"", "unknown", "none", "null", "n/a", "na"}
_ASSET_REFERENCE_FIELDS = (
    "parent_asset_id",
    "derived_from_asset_id",
    "source_asset_id",
    "duplicate_of_asset_id",
)
_ASSET_REFERENCE_LIST_FIELDS = ("adjacent_asset_ids", "neighbor_asset_ids")
_RELATION_GROUP_FIELDS = (
    "derivation_group_id",
    "derived_group_id",
    "duplicate_group_id",
    "adjacent_window_group_id",
    "adjacency_group_id",
    "window_group_id",
)

ACTION_DEFINITIONS: dict[str, tuple[str | None, str | None]] = {
    "A01": ("walk", "normal_walk"),
    "A02": ("turn", "normal_turn"),
    "A03": ("sit_down", "controlled_sit_down"),
    "A04": ("sit_to_stand", "normal_sit_to_stand"),
    "A05": ("squat", "controlled_squat"),
    "A06": ("bend", "controlled_bend"),
    "A07": ("lie_down", "controlled_lie_down"),
    "A08": ("support_use", "routine_support_contact"),
    "A09": ("floor_transition", "kneel_or_floor_activity"),
    "A10": ("step_adjustment", "normal_step_adjustment"),
    "A11": ("assisted_transition", "assisted_sit_or_lowering"),
    "A12": ("exercise", "normal_hop"),
    "B01": ("walk", "slow_walk"),
    "B02": ("walk", "dragging_walk"),
    "B03": ("walk", "shuffling_walk"),
    "B04": ("walk", "swaying_walk"),
    "B05": ("turn", "unstable_turn"),
    "B06": ("sit_to_stand", "slow_sit_to_stand"),
    "C01": ("sit_to_stand", "failed_sit_to_stand"),
    "C02": ("support_use", "sustained_support_walk"),
    "C03": ("balance_loss", "stumble"),
    "C04": ("balance_loss", "rapid_support_reaction"),
    "C05": ("balance_loss", "rapid_body_drop"),
    "D01": ("fall", "forward_fall"),
    "D02": ("fall", "lateral_fall"),
    "D03": ("fall", "backward_fall"),
    "D04": ("post_fall_state", "post_fall_immobile"),
    "D05": ("fall", "seated_fall"),
    "U01": (None, None),
}

FALL_ACTION_SUBTYPES = {
    "D01": "forward",
    "D02": "lateral",
    "D03": "backward",
    "D05": "seated",
}

ACTION_ATTRIBUTES: dict[str, list[str]] = {
    "A08": ["support_used"],
    "A11": ["assisted"],
    "C02": ["support_used"],
    "C04": ["contact_proxy", "rapid", "support_used"],
    "C05": ["rapid"],
}

QUALITY_FLAG_MAP = {
    "clear": [],
    "partial_occlusion": ["partial_occlusion"],
    "heavy_occlusion": ["heavy_occlusion"],
    "low_light": ["low_light"],
    "off_screen": ["off_screen"],
    "multi_person_uncertain": ["multi_person_uncertain"],
}

SEVERE_QUALITY_FLAGS = {
    "heavy_occlusion",
    "off_screen",
    "multi_person_uncertain",
}

OFFICIAL_CLIP_ACTION_SOURCES = {
    "toaga_official_walking",
    "ntu_rgbd_clip_label",
}
MANUAL_EXACT_CLIP_ACTION_SOURCES = {"ntu_rgbd_manual_clip_label"}

FALL_HARD_NEGATIVES = {
    "controlled_sit_down",
    "controlled_lie_down",
    "squat_or_kneel",
    "bend",
    "bed_entry_or_exit",
    "assisted_lowering",
    "occlusion_or_camera_motion",
}

NEAR_FALL_HARD_NEGATIVES = {
    "normal_turn",
    "normal_step_adjustment",
    "routine_support_contact",
    "fast_but_controlled_sit",
    "controlled_squat",
    "controlled_bend",
    "exercise_or_stretch",
    "progressed_to_fall",
}

HARD_NEGATIVES_BY_TASK = {
    "fall_event": FALL_HARD_NEGATIVES,
    "near_fall_event": NEAR_FALL_HARD_NEGATIVES,
}

_TIER_RANK = {"ignore": 0, "auxiliary": 1, "primary": 2}
_NTU_RGBD_A043_SOURCE_PATTERN = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<person>\d{3})"
    r"R(?P<repetition>\d{3})A043_rgb\.(?:mp4|avi)"
)
_NTU_RGBD_HARD_NEGATIVE_TYPES = {
    "A05_controlled_squat": "squat_or_kneel",
}


@dataclass(frozen=True)
class TrainingLabelMigrationResult:
    action_labels: list[dict[str, Any]]
    event_labels: list[dict[str, Any]]
    report: dict[str, Any]


def migrate_v2_training_labels(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    manual_negative_decisions: Sequence[Mapping[str, Any]] = (),
    reviewed_decisions: Sequence[Mapping[str, Any]] = (),
) -> TrainingLabelMigrationResult:
    video_manifest_rows = [
        row
        for row in manifest_rows
        if isinstance(row.get("video_id"), str) and row.get("video_id")
    ]
    manifests = _index_unique(video_manifest_rows, "video_id", "manifest")
    v2_actions = _index_unique(action_rows, "label_id", "v2 action labels")
    _index_unique(event_rows, "label_id", "v2 event labels")

    actions_v3: list[dict[str, Any]] = []
    action_v3_by_v2: dict[str, dict[str, Any]] = {}
    for v2_action in action_rows:
        video_id = _required_string(v2_action, "video_id")
        manifest = _required_manifest(manifests, video_id)
        migrated = _migrate_action(v2_action, manifest)
        actions_v3.append(migrated)
        action_v3_by_v2[_required_string(v2_action, "label_id")] = migrated
    fall_actions_by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for action in action_rows:
        if action.get("action_id") in FALL_ACTION_SUBTYPES:
            fall_actions_by_video[_required_string(action, "video_id")].append(action)

    official_falls = [
        event
        for event in event_rows
        if event.get("event_type") == "fall" and event.get("label_source") == "le2i_txt"
    ]
    mapped_falls = [
        event
        for event in event_rows
        if event.get("event_type") == "fall"
        and event.get("label_source") == "cvat_action_mapping"
    ]

    events_v3: list[dict[str, Any]] = []
    action_to_event_id: dict[str, str] = {}
    deduplicated_official_action_falls = 0
    validated_reviewed_decisions = _validated_reviewed_training_decisions(
        reviewed_decisions
    )
    reviewed_directives = _reviewed_directives(validated_reviewed_decisions)
    matched_reviewed_directives: Counter[str] = Counter()
    reviewed_decision_matches: Counter[str] = Counter()

    for event in official_falls:
        video_id = _required_string(event, "video_id")
        manifest = _required_manifest(manifests, video_id)
        candidates = [
            action
            for action in fall_actions_by_video.get(video_id, [])
            if _inclusive_intervals_overlap(action, event)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"official fall {event.get('label_id')} overlaps multiple fall actions"
            )
        matched_action = candidates[0] if candidates else None
        reviewed_boundary = _matching_ntu_full_clip_boundary(
            reviewed_directives["ntu_full_clip_fall_boundary"],
            matched_action,
            manifest,
        )
        migrated = _migrate_fall_event(
            event,
            manifest,
            matched_action=matched_action,
            action_v3_by_v2=action_v3_by_v2,
            official=True,
            reviewed_boundary=reviewed_boundary,
        )
        if reviewed_boundary is not None:
            matched_reviewed_directives[str(reviewed_boundary["directive_key"])] += 1
            reviewed_decision_matches["ntu_full_clip_fall_boundary"] += 1
        events_v3.append(migrated)
        if matched_action is not None:
            action_to_event_id[_required_string(matched_action, "label_id")] = migrated[
                "label_id"
            ]
            deduplicated_official_action_falls += 1

    for event in mapped_falls:
        source_action_id = _required_string(event, "source_action_label_id")
        if source_action_id in action_to_event_id:
            continue
        source_action = v2_actions.get(source_action_id)
        if source_action is None:
            raise ValueError(
                f"mapped fall {event.get('label_id')} references missing action {source_action_id}"
            )
        manifest = _required_manifest(
            manifests, _required_string(event, "video_id")
        )
        reviewed_boundary = _matching_ntu_full_clip_boundary(
            reviewed_directives["ntu_full_clip_fall_boundary"],
            source_action,
            manifest,
        )
        migrated = _migrate_fall_event(
            event,
            manifest,
            matched_action=source_action,
            action_v3_by_v2=action_v3_by_v2,
            official=False,
            reviewed_boundary=reviewed_boundary,
        )
        if reviewed_boundary is not None:
            matched_reviewed_directives[str(reviewed_boundary["directive_key"])] += 1
            reviewed_decision_matches["ntu_full_clip_fall_boundary"] += 1
        events_v3.append(migrated)
        action_to_event_id[source_action_id] = migrated["label_id"]

    fall_events_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events_v3:
        fall_events_by_video[event["video_id"]].append(event)

    unresolved_post_fall_states = 0
    for v2_id, action in action_v3_by_v2.items():
        action_id = action["action_id"]
        if action_id in FALL_ACTION_SUBTYPES:
            action["linked_event_id"] = action_to_event_id.get(v2_id)
        elif action_id == "D04":
            parents = [
                event
                for event in fall_events_by_video.get(action["video_id"], [])
                if event["end_frame_exclusive"] <= action["start_frame"]
            ]
            if len(parents) == 1:
                parent = parents[0]
                action["linked_event_id"] = parent["label_id"]
                parent["linked_action_ids"] = sorted(
                    {*parent["linked_action_ids"], action["label_id"]}
                )
            else:
                action["training_tier"] = "ignore"
                unresolved_post_fall_states += 1

    _assign_action_type_training_tiers(actions_v3)

    occupied_action_targets = {
        (action_id, str(event["task_type"]))
        for event in events_v3
        for action_id in event.get("linked_action_ids") or []
    }
    for action in actions_v3:
        positive_rule = reviewed_directives["near_fall_positive"].get(
            str(action["action_id"])
        )
        if positive_rule is None:
            continue
        target = (str(action["label_id"]), "near_fall_event")
        if target in occupied_action_targets:
            raise ValueError(
                "reviewed decision would duplicate an action/task event: "
                f"{action['label_id']}:near_fall_event"
            )
        event = _near_fall_positive_from_action(action, positive_rule)
        events_v3.append(event)
        action["linked_event_id"] = event["label_id"]
        occupied_action_targets.add(target)
        matched_reviewed_directives[str(positive_rule["directive_key"])] += 1
        reviewed_decision_matches["near_fall_positive"] += 1

    for action in actions_v3:
        if action.get("training_tier") == "ignore":
            continue
        for task_type in HARD_NEGATIVES_BY_TASK:
            rule = _reviewed_action_negative_rule(
                reviewed_directives,
                action,
                task_type,
            )
            if rule is None:
                continue
            target = (str(action["label_id"]), task_type)
            if target in occupied_action_targets:
                raise ValueError(
                    "reviewed decision would duplicate an action/task event: "
                    f"{action['label_id']}:{task_type}"
                )
            event_tier_cap = _required_manifest(
                manifests, str(action["video_id"])
            ).get("training_tier_cap")
            events_v3.append(
                _event_negative_from_action(
                    action, rule, training_tier_cap=event_tier_cap
                )
            )
            occupied_action_targets.add(target)
            matched_reviewed_directives[str(rule["directive_key"])] += 1
            reviewed_decision_matches[str(rule["match_category"])] += 1

    for source_event in tuple(events_v3):
        if (
            source_event.get("label_role") != "positive"
            or source_event.get("training_tier") == "ignore"
        ):
            continue
        for rule in reviewed_directives["event_hard_negative"]:
            if (
                source_event.get("task_type") != rule["source_task_type"]
                or source_event.get("label_role") != rule["source_label_role"]
            ):
                continue
            events_v3.append(_event_negative_from_event(source_event, rule))
            matched_reviewed_directives[str(rule["directive_key"])] += 1
            reviewed_decision_matches["event_hard_negative"] += 1

    validated_manual_negative_decisions = _validated_manual_negative_decisions(
        manual_negative_decisions
    )
    matched_manual_negative_decisions: set[str] = set()
    for decision in validated_manual_negative_decisions:
        candidates = [
            action
            for action in actions_v3
            if action.get("video_id") == decision["video_id"]
            and action.get("action_id") == decision["source_action_id"]
        ]
        decision_key = str(decision["decision_key"])
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(
                "manual negative decision matches multiple actions: "
                f"{decision_key}"
            )
        action = candidates[0]
        if action.get("training_tier") == "ignore":
            raise ValueError(
                "manual negative decision targets an ignored action: "
                f"{decision_key}"
            )
        target = (str(action["label_id"]), str(decision["task_type"]))
        if target in occupied_action_targets:
            raise ValueError(
                "manual and reviewed decisions target the same action/task: "
                f"{action['label_id']}:{decision['task_type']}"
            )
        event_tier_cap = _required_manifest(
            manifests, str(action["video_id"])
        ).get("training_tier_cap")
        events_v3.append(
            _event_negative_from_action(
                action, decision, training_tier_cap=event_tier_cap
            )
        )
        occupied_action_targets.add(target)
        matched_manual_negative_decisions.add(decision_key)

    for v2_action in action_rows:
        if v2_action.get("action_id") != "U01":
            continue
        migrated_action = action_v3_by_v2[_required_string(v2_action, "label_id")]
        for task_type in ("fall_event", "near_fall_event"):
            events_v3.append(_event_ignore_from_action(migrated_action, task_type))

    actions_v3.sort(key=_label_sort_key)
    events_v3.sort(key=_label_sort_key)
    report = _migration_report(
        action_rows=action_rows,
        event_rows=event_rows,
        actions_v3=actions_v3,
        events_v3=events_v3,
        deduplicated_official_action_falls=deduplicated_official_action_falls,
        unresolved_post_fall_states=unresolved_post_fall_states,
        manual_negative_decisions=validated_manual_negative_decisions,
        matched_manual_negative_decisions=matched_manual_negative_decisions,
        reviewed_directives=reviewed_directives["all"],
        matched_reviewed_directives=matched_reviewed_directives,
        reviewed_decision_matches=reviewed_decision_matches,
    )
    return TrainingLabelMigrationResult(actions_v3, events_v3, report)


def write_training_label_migration(
    *,
    manifest_path: Path | str,
    action_labels_v2_path: Path | str,
    event_labels_v2_path: Path | str,
    action_labels_v3_path: Path | str,
    event_labels_v3_path: Path | str,
    report_path: Path | str,
    manual_negative_decision_paths: Sequence[Path | str] = (),
    reviewed_decision_paths: Sequence[Path | str] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    input_paths = {
        "manifest": Path(manifest_path),
        "action_labels_v2": Path(action_labels_v2_path),
        "event_labels_v2": Path(event_labels_v2_path),
    }
    decision_paths = [Path(path) for path in manual_negative_decision_paths]
    if len(decision_paths) != len(set(decision_paths)):
        raise ValueError("manual negative decision paths must be unique")
    for index, path in enumerate(decision_paths):
        input_paths[f"manual_negative_decision_{index}"] = path
    manual_negative_decisions = _load_manual_negative_decisions(decision_paths)
    reviewed_paths = [Path(path) for path in reviewed_decision_paths]
    if len(reviewed_paths) != len(set(reviewed_paths)):
        raise ValueError("reviewed decision paths must be unique")
    for index, path in enumerate(reviewed_paths):
        input_paths[f"reviewed_decision_{index}"] = path
    action_rows = read_jsonl_strict(input_paths["action_labels_v2"])
    reviewed_decisions = _load_reviewed_training_decisions(
        reviewed_paths,
        action_labels_sha256=_sha256_file(input_paths["action_labels_v2"]),
    )
    result = migrate_v2_training_labels(
        manifest_rows=read_jsonl_strict(input_paths["manifest"]),
        action_rows=action_rows,
        event_rows=read_jsonl_strict(input_paths["event_labels_v2"]),
        manual_negative_decisions=manual_negative_decisions,
        reviewed_decisions=reviewed_decisions,
    )
    outputs = {
        "action_labels_v3": Path(action_labels_v3_path),
        "event_labels_v3": Path(event_labels_v3_path),
        "migration_report": Path(report_path),
    }
    _ensure_outputs_available(outputs.values(), overwrite=overwrite)
    _write_jsonl_atomic(outputs["action_labels_v3"], result.action_labels)
    _write_jsonl_atomic(outputs["event_labels_v3"], result.event_labels)
    report = dict(result.report)
    report["input_sha256"] = {
        key: _sha256_file(path) for key, path in input_paths.items()
    }
    report["output_sha256"] = {
        "action_labels_v3": _sha256_file(outputs["action_labels_v3"]),
        "event_labels_v3": _sha256_file(outputs["event_labels_v3"]),
    }
    _write_json_atomic(outputs["migration_report"], report)
    return report


def validate_training_labels_v3(
    *,
    manifest_path: Path | str,
    action_labels_path: Path | str,
    event_labels_path: Path | str,
    action_schema_path: Path | str,
    event_schema_path: Path | str,
    split_assignments_path: Path | str | None = None,
    split_report_path: Path | str | None = None,
) -> dict[str, Any]:
    paths = {
        "manifest": Path(manifest_path),
        "action_labels": Path(action_labels_path),
        "event_labels": Path(event_labels_path),
        "action_schema": Path(action_schema_path),
        "event_schema": Path(event_schema_path),
    }
    if split_assignments_path is not None:
        paths["split_assignments"] = Path(split_assignments_path)
    if split_report_path is not None:
        paths["split_report"] = Path(split_report_path)
    manifest_rows = read_jsonl_strict(paths["manifest"])
    action_rows = read_jsonl_strict(paths["action_labels"])
    event_rows = read_jsonl_strict(paths["event_labels"])
    action_schema = _read_json_strict(paths["action_schema"])
    event_schema = _read_json_strict(paths["event_schema"])

    issues: list[dict[str, Any]] = []
    video_manifest_rows = [
        row
        for row in manifest_rows
        if isinstance(row.get("video_id"), str) and row.get("video_id")
    ]
    manifests = _index_for_validation(
        video_manifest_rows, "video_id", "manifest", issues
    )

    action_ids: set[str] = set()
    for row in action_rows:
        _validate_schema_node(
            row, action_schema, "action", row.get("label_id"), issues
        )
        _validate_common_semantics(row, manifests, "action", issues)
        _validate_action_semantics(row, issues)
        label_id = row.get("label_id")
        if isinstance(label_id, str):
            if label_id in action_ids:
                _issue(issues, "duplicate_label_id", "action", label_id)
            action_ids.add(label_id)
    _validate_action_type_training_tiers(action_rows, issues)

    event_ids: set[str] = set()
    physical_positive_keys: set[tuple[str, str]] = set()
    for row in event_rows:
        _validate_schema_node(
            row, event_schema, "event", row.get("label_id"), issues
        )
        _validate_common_semantics(row, manifests, "event", issues)
        _validate_event_semantics(row, issues)
        label_id = row.get("label_id")
        if isinstance(label_id, str):
            if label_id in event_ids:
                _issue(issues, "duplicate_label_id", "event", label_id)
            event_ids.add(label_id)
        if row.get("label_role") == "positive" and isinstance(
            row.get("physical_event_id"), str
        ):
            key = (str(row.get("task_type")), str(row["physical_event_id"]))
            if key in physical_positive_keys:
                _issue(
                    issues,
                    "duplicate_physical_event",
                    "event",
                    label_id,
                )
            physical_positive_keys.add(key)

    events_by_id = {
        row["label_id"]: row
        for row in event_rows
        if isinstance(row.get("label_id"), str)
    }
    actions_by_id = {
        row["label_id"]: row
        for row in action_rows
        if isinstance(row.get("label_id"), str)
    }
    _validate_cross_references(action_rows, event_rows, actions_by_id, events_by_id, issues)

    split_valid = False
    split_summary: dict[str, Any] | None = None
    if split_assignments_path is None and split_report_path is None:
        split_summary = {"valid": False, "reason": "training_split_missing"}
    elif split_assignments_path is None or split_report_path is None:
        _issue(issues, "training_split_files_incomplete", "split", None)
        split_summary = {"valid": False, "reason": "training_split_files_incomplete"}
    else:
        split_summary = _validate_training_split_v3(
            assignments_path=Path(split_assignments_path),
            report_path=Path(split_report_path),
            manifest_path=paths["manifest"],
            action_labels_path=paths["action_labels"],
            event_labels_path=paths["event_labels"],
            action_rows=action_rows,
            event_rows=event_rows,
            issues=issues,
        )
        split_valid = bool(split_summary["valid"])

    counts = _validation_counts(action_rows, event_rows)
    hard_negative_coverage = _hard_negative_coverage(event_rows)
    warnings = _training_warnings(event_rows, hard_negative_coverage)
    if not split_valid:
        warnings.append("training_split_not_ready")
    action_type_ready = _split_action_type_ready(split_summary)
    if not action_type_ready:
        warnings.append("primary_action_type_split_coverage_incomplete")
    valid = not issues
    return {
        "schema_version": "fall-risk-training-label-validation-v3",
        "valid": valid,
        "issues": issues,
        "warnings": warnings,
        "counts": counts,
        "class_counts": {
            "actions": dict(sorted(Counter(
                str(row.get("action_type")) for row in action_rows
            ).items())),
            "events": dict(sorted(Counter(
                str(row.get("event_type")) for row in event_rows
                if row.get("label_role") == "positive"
            ).items())),
        },
        "action_type_counts_by_tier": _action_type_counts_by_tier(action_rows),
        "hard_negative_coverage": hard_negative_coverage,
        "split": split_summary,
        "input_sha256": {key: _sha256_file(path) for key, path in paths.items()},
        "training_ready": {
            "action_type": valid and split_valid and action_type_ready,
            "fall_event": valid
            and split_valid
            and _split_task_ready(split_summary, "fall_event")
            and counts["primary_fall_positive"] > 0
            and counts["primary_fall_negative"] > 0
            and not hard_negative_coverage["fall_event"]["missing"],
            "near_fall_event": valid
            and split_valid
            and _split_task_ready(split_summary, "near_fall_event")
            and counts["primary_near_fall_positive"] > 0
            and counts["primary_near_fall_negative"] > 0
            and not hard_negative_coverage["near_fall_event"]["missing"],
        },
    }


def write_training_label_validation_report(
    *, report_path: Path | str, overwrite: bool = False, **kwargs: Any
) -> dict[str, Any]:
    path = Path(report_path)
    _ensure_outputs_available([path], overwrite=overwrite)
    report = validate_training_labels_v3(**kwargs)
    _write_json_atomic(path, report)
    return report


def build_training_split_v3(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    seed: str,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    inherited_assignment_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one deterministic split shared by v3 action and event labels."""

    normalized_ratios = _validate_split_ratios(ratios)
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("split seed must be a non-empty string")
    seed = seed.strip()
    manifests = {
        _required_string(row, "asset_id"): row
        for row in manifest_rows
        if isinstance(row.get("asset_id"), str)
    }
    labels: list[tuple[str, Mapping[str, Any]]] = [
        *(("action", row) for row in action_rows),
        *(("event", row) for row in event_rows),
    ]
    if not labels:
        raise ValueError("cannot build a split without v3 labels")
    labels_by_id: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for kind, row in labels:
        label_id = _required_string(row, "label_id")
        if label_id in labels_by_id:
            raise ValueError(f"duplicate v3 split label_id: {label_id}")
        labels_by_id[label_id] = (kind, row)
    inherited_by_label: dict[str, Mapping[str, Any]] = {}
    inherited_rows_by_asset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for inherited in inherited_assignment_rows:
        asset_id = _required_string(inherited, "asset_id")
        inherited_rows_by_asset[asset_id].append(inherited)
        label_id = _required_string(inherited, "label_id")
        if label_id not in labels_by_id:
            continue
        if label_id in inherited_by_label:
            raise ValueError(f"duplicate inherited split label_id: {label_id}")
        kind, label = labels_by_id[label_id]
        expected = {
            "label_kind": kind,
            "asset_id": label.get("asset_id"),
            "video_id": label.get("video_id"),
            "source_group_id": label.get("source_group_id"),
            "sample_group_id": label.get("sample_group_id"),
            "content_sha256": label.get("content_sha256"),
            "training_tier": label.get("training_tier"),
        }
        mismatched = [
            field
            for field, expected_value in expected.items()
            if inherited.get(field) != expected_value
        ]
        if mismatched:
            raise ValueError(
                f"inherited split label mismatch for {label_id}: {mismatched}"
            )
        if inherited.get("partition") not in SPLIT_PARTITIONS:
            raise ValueError(f"invalid inherited split partition for {label_id}")
        inherited_by_label[label_id] = inherited

    labels_by_asset: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for kind, row in labels:
        asset_id = _required_string(row, "asset_id")
        if asset_id not in manifests:
            raise ValueError(f"label asset_id missing from manifest: {asset_id}")
        labels_by_asset[asset_id].append((kind, row))

    inherited_partition_by_asset: dict[str, str] = {}
    inherited_group_by_asset: dict[str, str] = {}
    for asset_id, inherited_rows in inherited_rows_by_asset.items():
        if asset_id not in labels_by_asset:
            continue
        partitions = {str(row.get("partition")) for row in inherited_rows}
        group_ids = {str(row.get("split_group_id")) for row in inherited_rows}
        if len(partitions) != 1 or not partitions <= set(SPLIT_PARTITIONS):
            raise ValueError(f"inherited asset {asset_id} has conflicting partitions")
        if len(group_ids) != 1:
            raise ValueError(f"inherited asset {asset_id} has conflicting split groups")
        for field in (
            "video_id",
            "source_group_id",
            "sample_group_id",
            "content_sha256",
        ):
            inherited_values = {str(row.get(field)) for row in inherited_rows}
            current_values = {
                str(row.get(field)) for _, row in labels_by_asset[asset_id]
            }
            if inherited_values != current_values:
                raise ValueError(
                    f"inherited asset mismatch for {asset_id}: {field}"
                )
        inherited_partition_by_asset[asset_id] = next(iter(partitions))
        inherited_group_by_asset[asset_id] = next(iter(group_ids))

    union_find = _UnionFind(labels_by_asset)
    token_owner: dict[str, str] = {}
    for asset_id in sorted(labels_by_asset):
        manifest = manifests[asset_id]
        asset_labels = [row for _, row in labels_by_asset[asset_id]]
        for token in sorted(_split_tokens(asset_id, manifest, asset_labels)):
            owner = token_owner.setdefault(token, asset_id)
            union_find.union(asset_id, owner)

    component_assets: dict[str, list[str]] = defaultdict(list)
    for asset_id in sorted(labels_by_asset):
        component_assets[union_find.find(asset_id)].append(asset_id)

    components = []
    for members in component_assets.values():
        sorted_members = sorted(members)
        component_id = _stable_id("splitgrp", SPLIT_SCHEMA_VERSION, *sorted_members)
        components.append(
            (
                component_id,
                sorted_members,
                _split_component_metrics(sorted_members, labels_by_asset),
            )
        )
    partition_by_component = _balanced_component_partitions(
        components, seed=seed, ratios=normalized_ratios
    )
    for component_id, members, _ in components:
        fixed_partitions = {
            str(manifests[asset_id]["split_partition"])
            for asset_id in members
            if manifests[asset_id].get("split_partition") is not None
        }
        inherited_partitions = {
            inherited_partition_by_asset[asset_id]
            for asset_id in members
            if asset_id in inherited_partition_by_asset
        }
        inherited_group_ids = {
            inherited_group_by_asset[asset_id]
            for asset_id in members
            if asset_id in inherited_group_by_asset
        }
        if inherited_group_ids and inherited_group_ids != {component_id}:
            raise ValueError(
                f"component {component_id} does not match inherited split group"
            )
        fixed_partitions.update(inherited_partitions)
        if not fixed_partitions:
            continue
        if not fixed_partitions <= set(SPLIT_PARTITIONS):
            raise ValueError(
                f"component {component_id} has an invalid fixed split partition"
            )
        if len(fixed_partitions) != 1:
            raise ValueError(
                f"component {component_id} has conflicting fixed split partitions"
            )
        partition_by_component[component_id] = next(iter(fixed_partitions))
    component_by_asset: dict[str, str] = {}
    partition_by_asset: dict[str, str] = {}
    for component_id, members, _ in components:
        for asset_id in members:
            component_by_asset[asset_id] = component_id
            partition_by_asset[asset_id] = partition_by_component[component_id]

    assignments: list[dict[str, Any]] = []
    for kind, row in labels:
        asset_id = str(row["asset_id"])
        assignments.append(
            {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "label_id": _required_string(row, "label_id"),
                "label_kind": kind,
                "task_type": (
                    str(row.get("task_type")) if kind == "event" else "action"
                ),
                "asset_id": asset_id,
                "video_id": _required_string(row, "video_id"),
                "subject_id": _required_string(row, "subject_id"),
                "source_group_id": _required_string(row, "source_group_id"),
                "sample_group_id": _required_string(row, "sample_group_id"),
                "physical_event_id": row.get("physical_event_id"),
                "content_sha256": _required_string(row, "content_sha256"),
                "training_tier": _required_string(row, "training_tier"),
                "partition": partition_by_asset[asset_id],
                "split_group_id": component_by_asset[asset_id],
            }
        )
    assignments.sort(key=lambda row: (row["label_kind"], row["label_id"]))
    leakage_issues = _split_leakage_issues(assignments)
    if leakage_issues:
        raise ValueError(
            "generated v3 split contains leakage: "
            + json.dumps(leakage_issues, ensure_ascii=False, sort_keys=True)
        )
    report = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "ratios": normalized_ratios,
        "assignment_count": len(assignments),
        "asset_count": len(labels_by_asset),
        "group_count": len(component_assets),
        "partition_label_counts": {
            partition: sum(row["partition"] == partition for row in assignments)
            for partition in SPLIT_PARTITIONS
        },
        "partition_asset_counts": {
            partition: len(
                {row["asset_id"] for row in assignments if row["partition"] == partition}
            )
            for partition in SPLIT_PARTITIONS
        },
        "partition_supervision_counts": _partition_supervision_counts(
            assignments, _split_label_index(action_rows, event_rows)
        ),
        "partition_action_type_counts": _partition_action_type_counts(
            assignments, _split_label_index(action_rows, event_rows)
        ),
        "allocation_method": "deterministic_group_level_supervision_balance",
        "inherited_assignment_count": sum(
            len(labels_by_asset[asset_id])
            for asset_id in inherited_partition_by_asset
        ),
        "inherited_asset_count": len(inherited_partition_by_asset),
        "leakage_issues": [],
    }
    return assignments, report


def write_training_split_v3(
    *,
    manifest_path: Path | str,
    action_labels_path: Path | str,
    event_labels_path: Path | str,
    assignments_path: Path | str,
    report_path: Path | str,
    seed: str,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    inherited_assignments_path: Path | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_paths = {
        "manifest": Path(manifest_path),
        "action_labels": Path(action_labels_path),
        "event_labels": Path(event_labels_path),
    }
    assignments_output = Path(assignments_path)
    report_output = Path(report_path)
    _ensure_outputs_available([assignments_output, report_output], overwrite=overwrite)
    inherited_source = (
        Path(inherited_assignments_path)
        if inherited_assignments_path is not None
        else None
    )
    inherited_rows = (
        read_jsonl_strict(inherited_source) if inherited_source is not None else []
    )
    assignments, report = build_training_split_v3(
        manifest_rows=read_jsonl_strict(input_paths["manifest"]),
        action_rows=read_jsonl_strict(input_paths["action_labels"]),
        event_rows=read_jsonl_strict(input_paths["event_labels"]),
        seed=seed,
        ratios=ratios,
        inherited_assignment_rows=inherited_rows,
    )
    _write_jsonl_atomic(assignments_output, assignments)
    report = {
        **report,
        "input_sha256": {key: _sha256_file(path) for key, path in input_paths.items()},
        "assignments_sha256": _sha256_file(assignments_output),
    }
    if inherited_source is not None:
        report["inherited_assignments"] = {
            "sha256": _sha256_file(inherited_source),
        }
        try:
            report["inherited_assignments"]["path"] = inherited_source.resolve().relative_to(
                Path.cwd().resolve()
            ).as_posix()
        except ValueError:
            pass
    report["split_id"] = _stable_id(
        "splitv3",
        report["seed"],
        report["assignments_sha256"],
        *report["input_sha256"].values(),
    )
    _write_json_atomic(report_output, report)
    return report


def _validate_training_split_v3(
    *,
    assignments_path: Path,
    report_path: Path,
    manifest_path: Path,
    action_labels_path: Path,
    event_labels_path: Path,
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_count_before = len(issues)
    assignments = read_jsonl_strict(assignments_path)
    report = _read_json_strict(report_path)
    expected_labels = {
        **{
            str(row["label_id"]): ("action", row)
            for row in action_rows
            if isinstance(row.get("label_id"), str)
        },
        **{
            str(row["label_id"]): ("event", row)
            for row in event_rows
            if isinstance(row.get("label_id"), str)
        },
    }
    seen: set[str] = set()
    for assignment in assignments:
        label_id = assignment.get("label_id")
        if assignment.get("schema_version") != SPLIT_SCHEMA_VERSION:
            _issue(issues, "split_schema_version_invalid", "split", label_id)
        if not isinstance(label_id, str) or label_id not in expected_labels:
            _issue(issues, "split_unknown_label", "split", label_id)
            continue
        if label_id in seen:
            _issue(issues, "split_duplicate_label", "split", label_id)
        seen.add(label_id)
        label_kind, label = expected_labels[label_id]
        expected = {
            "label_kind": label_kind,
            "asset_id": label.get("asset_id"),
            "video_id": label.get("video_id"),
            "subject_id": label.get("subject_id"),
            "source_group_id": label.get("source_group_id"),
            "sample_group_id": label.get("sample_group_id"),
            "content_sha256": label.get("content_sha256"),
            "training_tier": label.get("training_tier"),
            "physical_event_id": label.get("physical_event_id"),
            "task_type": label.get("task_type") if label_kind == "event" else "action",
        }
        for field, expected_value in expected.items():
            if assignment.get(field) != expected_value:
                _issue(
                    issues,
                    "split_label_field_mismatch",
                    "split",
                    label_id,
                    field,
                )
        if assignment.get("partition") not in SPLIT_PARTITIONS:
            _issue(issues, "split_partition_invalid", "split", label_id)
        if not _known_split_identifier(assignment.get("split_group_id")):
            _issue(issues, "split_group_missing", "split", label_id)
    for missing_label_id in sorted(set(expected_labels) - seen):
        _issue(issues, "split_label_missing", "split", missing_label_id)
    for leakage_issue in _split_leakage_issues(assignments):
        _issue(
            issues,
            "split_cross_partition_leakage",
            "split",
            None,
            leakage_issue,
        )

    if report.get("schema_version") != SPLIT_SCHEMA_VERSION:
        _issue(issues, "split_report_schema_invalid", "split", None)
    expected_input_sha256 = {
        "manifest": _sha256_file(manifest_path),
        "action_labels": _sha256_file(action_labels_path),
        "event_labels": _sha256_file(event_labels_path),
    }
    if report.get("input_sha256") != expected_input_sha256:
        _issue(issues, "split_input_hash_mismatch", "split", None)
    if report.get("assignments_sha256") != _sha256_file(assignments_path):
        _issue(issues, "split_assignments_hash_mismatch", "split", None)
    expected_split_id = _stable_id(
        "splitv3",
        report.get("seed"),
        _sha256_file(assignments_path),
        expected_input_sha256["manifest"],
        expected_input_sha256["action_labels"],
        expected_input_sha256["event_labels"],
    )
    if report.get("split_id") != expected_split_id:
        _issue(issues, "split_id_mismatch", "split", None)
    if report.get("assignment_count") != len(assignments):
        _issue(issues, "split_assignment_count_mismatch", "split", None)
    supervision_counts = _partition_supervision_counts(
        assignments, {label_id: value[1] for label_id, value in expected_labels.items()}
    )
    if report.get("partition_supervision_counts") != supervision_counts:
        _issue(issues, "split_supervision_count_mismatch", "split", None)
    action_type_counts = _partition_action_type_counts(
        assignments, {label_id: value[1] for label_id, value in expected_labels.items()}
    )
    if report.get("partition_action_type_counts") != action_type_counts:
        _issue(issues, "split_action_type_count_mismatch", "split", None)
    if report.get("leakage_issues") != []:
        _issue(issues, "split_report_has_leakage", "split", None)
    return {
        "valid": len(issues) == issue_count_before,
        "split_id": report.get("split_id"),
        "assignment_count": len(assignments),
        "partition_label_counts": report.get("partition_label_counts"),
        "partition_supervision_counts": supervision_counts,
        "partition_action_type_counts": action_type_counts,
        "input_sha256": expected_input_sha256,
    }


def _split_task_ready(split_summary: Mapping[str, Any] | None, task_type: str) -> bool:
    if not isinstance(split_summary, Mapping) or split_summary.get("valid") is not True:
        return False
    supervision = split_summary.get("partition_supervision_counts")
    if not isinstance(supervision, Mapping):
        return False
    task_counts = supervision.get(task_type)
    if not isinstance(task_counts, Mapping):
        return False
    return all(
        isinstance(task_counts.get(partition), Mapping)
        and task_counts[partition].get("primary_positive", 0) > 0
        and task_counts[partition].get("primary_negative", 0) > 0
        for partition in SPLIT_PARTITIONS
    )


def _split_action_type_ready(split_summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(split_summary, Mapping) or split_summary.get("valid") is not True:
        return False
    counts = split_summary.get("partition_action_type_counts")
    if not isinstance(counts, Mapping) or not counts:
        return False
    return all(
        isinstance(partition_counts, Mapping)
        and all(partition_counts.get(partition, 0) > 0 for partition in SPLIT_PARTITIONS)
        for partition_counts in counts.values()
    )


def read_jsonl_strict(path: Path | str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise ValueError(f"blank JSONL line at {file_path}:{line_number}")
        value = json.loads(
            raw_line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {file_path}:{line_number}")
        rows.append(value)
    return rows


def _load_manual_negative_decisions(
    decision_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for path in decision_paths:
        payload = load_ntu_rgbd_a043_decision(path)
        decision_id = _required_string(payload, "decision_id")
        reviewer_id = _required_string(payload, "reviewer_id")
        decision_sha256 = _sha256_file(path)
        for adjudication in payload["adjudications"]:
            if adjudication.get("type") != "accepted_hard_negative":
                continue
            accepted_label = _required_string(adjudication, "accepted_label")
            hard_negative_type = _NTU_RGBD_HARD_NEGATIVE_TYPES.get(accepted_label)
            if hard_negative_type is None:
                raise ValueError(
                    "unsupported NTU RGB+D hard-negative label: "
                    f"{accepted_label}"
                )
            source_action_id = accepted_label.split("_", 1)[0]
            for source_name in adjudication["source_names"]:
                video_id = _ntu_rgbd_a043_video_id(str(source_name))
                decisions.append(
                    {
                        "decision_key": f"{decision_id}:{video_id}:fall_event",
                        "decision_id": decision_id,
                        "video_id": video_id,
                        "source_action_id": source_action_id,
                        "task_type": "fall_event",
                        "hard_negative_type": hard_negative_type,
                        "reviewer_id": reviewer_id,
                        "reason": _required_string(adjudication, "reason"),
                        "source_annotation_path": path.as_posix(),
                        "source_annotation_sha256": decision_sha256,
                    }
                )
    return _validated_manual_negative_decisions(decisions)


def _load_reviewed_training_decisions(
    decision_paths: Sequence[Path],
    *,
    action_labels_sha256: str,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for path in decision_paths:
        payload = _read_json_strict(path)
        pinned_hash = _required_string(payload, "action_labels_sha256")
        if pinned_hash != action_labels_sha256:
            raise ValueError(
                "reviewed decision action-label hash mismatch: "
                f"{path} pins {pinned_hash}, current is {action_labels_sha256}"
            )
        decisions.append(
            {
                **payload,
                "source_annotation_path": path.as_posix(),
                "source_annotation_sha256": _sha256_file(path),
            }
        )
    return _validated_reviewed_training_decisions(decisions)


def _validated_reviewed_training_decisions(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    required = {
        "schema_version",
        "decision_id",
        "reviewed_at",
        "reviewer_id",
        "source_task_id",
        "action_labels_sha256",
        "ntu_full_clip_fall_boundary",
        "near_fall_positive_actions",
        "action_hard_negative_mappings",
        "video_hard_negative_overrides",
        "quality_hard_negative_mappings",
        "event_hard_negative_mappings",
        "rationale",
        "source_annotation_path",
        "source_annotation_sha256",
    }
    normalized: list[dict[str, Any]] = []
    seen_decision_ids: set[str] = set()
    for index, decision in enumerate(decisions):
        if set(decision) != required:
            raise ValueError(f"reviewed decision {index} has an invalid shape")
        if decision.get("schema_version") != REVIEWED_DECISION_SCHEMA_VERSION:
            raise ValueError(f"reviewed decision {index} has an invalid schema_version")
        row = dict(decision)
        for field in (
            "decision_id",
            "reviewed_at",
            "reviewer_id",
            "source_task_id",
            "action_labels_sha256",
            "source_annotation_path",
            "source_annotation_sha256",
        ):
            _required_string(row, field)
        for hash_field in ("action_labels_sha256", "source_annotation_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row[hash_field])) is None:
                raise ValueError(
                    f"reviewed decision {index} has an invalid {hash_field}"
                )
        decision_id = str(row["decision_id"])
        if decision_id in seen_decision_ids:
            raise ValueError(f"duplicate reviewed decision_id: {decision_id}")
        seen_decision_ids.add(decision_id)

        rationale = row["rationale"]
        if (
            not isinstance(rationale, list)
            or not rationale
            or any(not isinstance(value, str) or not value.strip() for value in rationale)
        ):
            raise ValueError(f"reviewed decision {index} has invalid rationale")
        _validate_ntu_boundary_policy(row["ntu_full_clip_fall_boundary"], index)
        _validate_near_fall_positive_rules(row["near_fall_positive_actions"], index)
        _validate_action_negative_rules(
            row["action_hard_negative_mappings"], index, "action"
        )
        _validate_video_negative_overrides(
            row["video_hard_negative_overrides"], index
        )
        _validate_quality_negative_rules(
            row["quality_hard_negative_mappings"], index
        )
        _validate_event_negative_rules(row["event_hard_negative_mappings"], index)
        normalized.append(row)
    return sorted(normalized, key=lambda row: str(row["decision_id"]))


def _validate_ntu_boundary_policy(value: Any, decision_index: int) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"reviewed decision {decision_index} has invalid NTU boundary policy"
        )
    if value == {"enabled": False}:
        return
    expected = {
        "enabled",
        "video_id_prefix",
        "action_ids",
        "onset_frame",
        "offset_frame",
        "accepted_source_end_frame_gaps",
        "boundary_precision",
        "training_tier_policy",
    }
    if set(value) != expected or value.get("enabled") is not True:
        raise ValueError(
            f"reviewed decision {decision_index} has invalid NTU boundary policy"
        )
    if value.get("video_id_prefix") != "ntu_rgbd_":
        raise ValueError("NTU boundary policy must target ntu_rgbd_ video IDs")
    action_ids = value.get("action_ids")
    if (
        not isinstance(action_ids, list)
        or not action_ids
        or any(action_id not in FALL_ACTION_SUBTYPES for action_id in action_ids)
    ):
        raise ValueError("NTU boundary policy has invalid fall action_ids")
    if len(action_ids) != len(set(action_ids)):
        raise ValueError("NTU boundary policy has duplicate action_ids")
    if (
        value.get("onset_frame") != "first_frame"
        or value.get("offset_frame") != "last_frame"
        or value.get("accepted_source_end_frame_gaps") != [0, 1]
        or value.get("boundary_precision") != "exact"
        or value.get("training_tier_policy") != "preserve"
    ):
        raise ValueError("unsupported NTU full-clip boundary policy")


def _validate_near_fall_positive_rules(value: Any, decision_index: int) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"reviewed decision {decision_index} has invalid near-fall rules"
        )
    expected = {
        "action_id",
        "event_subtype",
        "recovery_frame",
        "training_tier_policy",
    }
    seen: set[str] = set()
    for rule in value:
        if not isinstance(rule, Mapping) or set(rule) != expected:
            raise ValueError("invalid reviewed near-fall positive rule")
        action_id = str(rule.get("action_id"))
        if action_id not in {"C03", "C04", "C05"} or action_id in seen:
            raise ValueError("invalid or duplicate near-fall positive action_id")
        if (
            rule.get("event_subtype")
            not in {
                "stumble_recovery",
                "rapid_support_recovery",
                "rapid_body_drop_recovery",
            }
            or rule.get("recovery_frame") != "last_frame"
            or rule.get("training_tier_policy") != "preserve"
        ):
            raise ValueError("unsupported reviewed near-fall positive rule")
        seen.add(action_id)


def _validate_action_negative_rules(
    value: Any, decision_index: int, rule_kind: str
) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"reviewed decision {decision_index} has invalid {rule_kind} negative rules"
        )
    expected = {"task_type", "action_ids", "hard_negative_type"}
    seen: set[tuple[str, str]] = set()
    for rule in value:
        if not isinstance(rule, Mapping) or set(rule) != expected:
            raise ValueError("invalid reviewed action hard-negative rule")
        task_type = str(rule.get("task_type"))
        hard_negative_type = str(rule.get("hard_negative_type"))
        action_ids = rule.get("action_ids")
        if (
            task_type not in HARD_NEGATIVES_BY_TASK
            or hard_negative_type not in HARD_NEGATIVES_BY_TASK[task_type]
            or not isinstance(action_ids, list)
            or not action_ids
        ):
            raise ValueError("invalid reviewed action hard-negative rule values")
        for action_id in action_ids:
            target = (task_type, str(action_id))
            if action_id not in ACTION_DEFINITIONS or target in seen:
                raise ValueError("duplicate or unsupported action hard-negative target")
            seen.add(target)


def _validate_video_negative_overrides(value: Any, decision_index: int) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"reviewed decision {decision_index} has invalid video overrides"
        )
    expected = {"task_type", "action_id", "video_ids", "hard_negative_type"}
    seen: set[tuple[str, str]] = set()
    for rule in value:
        if not isinstance(rule, Mapping) or set(rule) != expected:
            raise ValueError("invalid reviewed video hard-negative override")
        task_type = str(rule.get("task_type"))
        action_id = str(rule.get("action_id"))
        hard_negative_type = str(rule.get("hard_negative_type"))
        video_ids = rule.get("video_ids")
        if (
            task_type not in HARD_NEGATIVES_BY_TASK
            or action_id not in ACTION_DEFINITIONS
            or hard_negative_type not in HARD_NEGATIVES_BY_TASK[task_type]
            or not isinstance(video_ids, list)
            or not video_ids
        ):
            raise ValueError("invalid reviewed video hard-negative override values")
        for video_id in video_ids:
            target = (task_type, str(video_id))
            if not isinstance(video_id, str) or not video_id or target in seen:
                raise ValueError("duplicate or invalid video hard-negative target")
            seen.add(target)


def _validate_quality_negative_rules(value: Any, decision_index: int) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"reviewed decision {decision_index} has invalid quality rules"
        )
    expected = {
        "task_type",
        "quality_flag",
        "action_ids",
        "hard_negative_type",
        "training_tier_policy",
    }
    seen: set[tuple[str, str, str]] = set()
    for rule in value:
        if not isinstance(rule, Mapping) or set(rule) != expected:
            raise ValueError("invalid reviewed quality hard-negative rule")
        task_type = str(rule.get("task_type"))
        quality_flag = str(rule.get("quality_flag"))
        hard_negative_type = str(rule.get("hard_negative_type"))
        action_ids = rule.get("action_ids")
        if (
            task_type not in HARD_NEGATIVES_BY_TASK
            or quality_flag != "partial_occlusion"
            or hard_negative_type not in HARD_NEGATIVES_BY_TASK[task_type]
            or rule.get("training_tier_policy") != "reviewed_primary"
            or not isinstance(action_ids, list)
            or not action_ids
        ):
            raise ValueError("invalid reviewed quality hard-negative rule values")
        for action_id in action_ids:
            target = (task_type, quality_flag, str(action_id))
            if action_id not in ACTION_DEFINITIONS or target in seen:
                raise ValueError("duplicate or unsupported quality hard-negative target")
            seen.add(target)


def _validate_event_negative_rules(value: Any, decision_index: int) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"reviewed decision {decision_index} has invalid event negative rules"
        )
    expected = {
        "source_task_type",
        "source_label_role",
        "task_type",
        "hard_negative_type",
    }
    seen: set[tuple[str, str, str]] = set()
    for rule in value:
        if not isinstance(rule, Mapping) or set(rule) != expected:
            raise ValueError("invalid reviewed event hard-negative rule")
        source_task_type = str(rule.get("source_task_type"))
        source_label_role = str(rule.get("source_label_role"))
        task_type = str(rule.get("task_type"))
        hard_negative_type = str(rule.get("hard_negative_type"))
        target = (source_task_type, source_label_role, task_type)
        if (
            source_task_type != "fall_event"
            or source_label_role != "positive"
            or task_type != "near_fall_event"
            or hard_negative_type not in HARD_NEGATIVES_BY_TASK[task_type]
            or target in seen
        ):
            raise ValueError("unsupported or duplicate event hard-negative rule")
        seen.add(target)


def _reviewed_directives(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ntu_full_clip_fall_boundary": [],
        "near_fall_positive": {},
        "action_hard_negative": {},
        "video_hard_negative_override": {},
        "quality_hard_negative": {},
        "event_hard_negative": [],
        "all": [],
    }
    for decision in decisions:
        decision_id = str(decision["decision_id"])

        def add(
            rule: Mapping[str, Any], directive_key: str, *, reason: str
        ) -> dict[str, Any]:
            expanded = {
                **rule,
                "directive_key": directive_key,
                "decision_id": decision_id,
                "reviewer_id": str(decision["reviewer_id"]),
                "source_annotation_path": str(decision["source_annotation_path"]),
                "source_annotation_sha256": str(
                    decision["source_annotation_sha256"]
                ),
                "reason": reason,
            }
            result["all"].append(expanded)
            return expanded

        boundary = decision["ntu_full_clip_fall_boundary"]
        if boundary.get("enabled"):
            result["ntu_full_clip_fall_boundary"].append(
                add(
                    boundary,
                    f"{decision_id}:ntu_full_clip_fall_boundary",
                    reason=(
                        f"Reviewed decision {decision_id}: NTU full-clip fall labels "
                        "use the first frame as onset and the last media frame as offset."
                    ),
                )
            )
        for rule in decision["near_fall_positive_actions"]:
            action_id = str(rule["action_id"])
            if action_id in result["near_fall_positive"]:
                raise ValueError(f"duplicate reviewed near-fall target: {action_id}")
            result["near_fall_positive"][action_id] = add(
                rule,
                f"{decision_id}:near_fall_positive:{action_id}",
                reason=(
                    f"Reviewed decision {decision_id}: reviewed {action_id} labels are "
                    f"near-fall positives ({rule['event_subtype']}) with recovery at "
                    "their inclusive last frame."
                ),
            )
        for rule in decision["action_hard_negative_mappings"]:
            for action_id_value in rule["action_ids"]:
                action_id = str(action_id_value)
                target = (str(rule["task_type"]), action_id)
                if target in result["action_hard_negative"]:
                    raise ValueError(f"duplicate reviewed action negative target: {target}")
                result["action_hard_negative"][target] = add(
                    {
                        key: value
                        for key, value in rule.items()
                        if key != "action_ids"
                    }
                    | {"source_action_id": action_id, "match_category": "action_hard_negative"},
                    f"{decision_id}:action_hard_negative:{target[0]}:{action_id}",
                    reason=(
                        f"Reviewed decision {decision_id}: canonical action {action_id} "
                        f"is an explicit {target[0]} negative "
                        f"({rule['hard_negative_type']})."
                    ),
                )
        for rule in decision["video_hard_negative_overrides"]:
            for video_id_value in rule["video_ids"]:
                video_id = str(video_id_value)
                target = (str(rule["task_type"]), str(rule["action_id"]), video_id)
                if target in result["video_hard_negative_override"]:
                    raise ValueError(f"duplicate reviewed video negative target: {target}")
                result["video_hard_negative_override"][target] = add(
                    {
                        key: value
                        for key, value in rule.items()
                        if key != "video_ids"
                    }
                    | {
                        "video_id": video_id,
                        "source_action_id": str(rule["action_id"]),
                        "match_category": "video_hard_negative_override",
                    },
                    f"{decision_id}:video_hard_negative_override:{target[0]}:{video_id}",
                    reason=(
                        f"Reviewed decision {decision_id}: video {video_id} action "
                        f"{rule['action_id']} is an explicit {target[0]} negative "
                        f"({rule['hard_negative_type']})."
                    ),
                )
        for rule in decision["quality_hard_negative_mappings"]:
            for action_id_value in rule["action_ids"]:
                action_id = str(action_id_value)
                target = (
                    str(rule["task_type"]),
                    action_id,
                    str(rule["quality_flag"]),
                )
                if target in result["quality_hard_negative"]:
                    raise ValueError(f"duplicate reviewed quality negative target: {target}")
                result["quality_hard_negative"][target] = add(
                    {
                        key: value
                        for key, value in rule.items()
                        if key != "action_ids"
                    }
                    | {
                        "source_action_id": action_id,
                        "match_category": "quality_hard_negative",
                        "event_training_tier": "primary",
                    },
                    f"{decision_id}:quality_hard_negative:{target[0]}:{action_id}:{target[2]}",
                    reason=(
                        f"Reviewed decision {decision_id}: {target[2]} on canonical "
                        f"action {action_id} is an explicit {target[0]} negative "
                        f"({rule['hard_negative_type']})."
                    ),
                )
        for rule in decision["event_hard_negative_mappings"]:
            result["event_hard_negative"].append(
                add(
                    rule,
                    f"{decision_id}:event_hard_negative:{rule['source_task_type']}:{rule['task_type']}",
                    reason=(
                        f"Reviewed decision {decision_id}: a confirmed "
                        f"{rule['source_task_type']} positive is an explicit "
                        f"{rule['task_type']} negative ({rule['hard_negative_type']})."
                    ),
                )
            )
    return result


def _reviewed_action_negative_rule(
    directives: Mapping[str, Any],
    action: Mapping[str, Any],
    task_type: str,
) -> Mapping[str, Any] | None:
    action_id = str(action["action_id"])
    video_id = str(action["video_id"])
    override = directives["video_hard_negative_override"].get(
        (task_type, action_id, video_id)
    )
    if override is not None:
        return override
    direct = directives["action_hard_negative"].get((task_type, action_id))
    if direct is not None:
        return direct
    for quality_flag in action.get("quality_flags") or []:
        quality_rule = directives["quality_hard_negative"].get(
            (task_type, action_id, str(quality_flag))
        )
        if quality_rule is not None:
            return quality_rule
    return None


def _validated_manual_negative_decisions(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    required = {
        "decision_key",
        "decision_id",
        "video_id",
        "source_action_id",
        "task_type",
        "hard_negative_type",
        "reviewer_id",
        "reason",
        "source_annotation_path",
        "source_annotation_sha256",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    for index, decision in enumerate(decisions):
        if set(decision) != required:
            raise ValueError(f"manual negative decision {index} has an invalid shape")
        row = {key: _required_string(decision, key) for key in required}
        if row["task_type"] not in HARD_NEGATIVES_BY_TASK:
            raise ValueError(
                f"manual negative decision {index} has an invalid task_type"
            )
        if row["hard_negative_type"] not in HARD_NEGATIVES_BY_TASK[row["task_type"]]:
            raise ValueError(
                f"manual negative decision {index} has an invalid hard-negative type"
            )
        if re.fullmatch(r"[0-9a-f]{64}", row["source_annotation_sha256"]) is None:
            raise ValueError(
                f"manual negative decision {index} has an invalid source hash"
            )
        if row["decision_key"] in seen:
            raise ValueError(
                f"duplicate manual negative decision: {row['decision_key']}"
            )
        target = (row["video_id"], row["task_type"])
        if target in seen_targets:
            raise ValueError(
                "duplicate manual negative target: "
                f"{row['video_id']}:{row['task_type']}"
            )
        seen.add(row["decision_key"])
        seen_targets.add(target)
        normalized.append(row)
    return sorted(normalized, key=lambda row: row["decision_key"])


def _ntu_rgbd_a043_video_id(source_name: str) -> str:
    match = _NTU_RGBD_A043_SOURCE_PATTERN.fullmatch(Path(source_name).name)
    if match is None:
        raise ValueError(f"invalid NTU RGB+D A043 source name: {source_name}")
    return (
        f"ntu_rgbd_s{match.group('setup')}_p{match.group('person')}_"
        f"r{match.group('repetition')}_a043_c{match.group('camera')}"
    )


def _migrate_action(
    row: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    action_id = _required_string(row, "action_id")
    if action_id not in ACTION_DEFINITIONS:
        raise ValueError(f"unsupported v2 action_id: {action_id}")
    common = _common_from_v2_action(row, manifest)
    return {
        **common,
        "schema_version": ACTION_SCHEMA_VERSION,
        "label_id": _stable_id("actionv3", ACTION_SCHEMA_VERSION, row["label_id"]),
        "action_id": action_id,
        "action_family": ACTION_DEFINITIONS[action_id][0],
        "action_type": ACTION_DEFINITIONS[action_id][1],
        "action_attributes": list(ACTION_ATTRIBUTES.get(action_id, [])),
        "action_type_training_tier": "ignore",
        "linked_event_id": None,
    }


def _assign_action_type_training_tiers(
    action_rows: Sequence[dict[str, Any]],
) -> None:
    target_by_type = _action_type_tier_targets(action_rows)
    for row in action_rows:
        action_type = row.get("action_type")
        parent_tier = str(row.get("training_tier"))
        if _is_manual_ntu_action(row):
            row["action_type_training_tier"] = parent_tier
            continue
        target_tier = target_by_type.get(str(action_type), "ignore")
        if parent_tier not in _TIER_RANK:
            row["action_type_training_tier"] = "ignore"
        else:
            row["action_type_training_tier"] = min(
                (parent_tier, target_tier), key=lambda tier: _TIER_RANK[tier]
            )


def _action_type_tier_targets(
    action_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    sample_groups_by_type: dict[str, set[str]] = defaultdict(set)
    source_groups_by_type: dict[str, set[str]] = defaultdict(set)
    for row in action_rows:
        action_type = row.get("action_type")
        if not isinstance(action_type, str) or row.get("training_tier") == "ignore":
            continue
        sample_group_id = row.get("sample_group_id")
        source_group_id = row.get("source_group_id")
        if isinstance(sample_group_id, str) and sample_group_id:
            sample_groups_by_type[action_type].add(sample_group_id)
        if isinstance(source_group_id, str) and source_group_id:
            source_groups_by_type[action_type].add(source_group_id)
    targets: dict[str, str] = {}
    for action_type in sample_groups_by_type:
        episode_count = len(sample_groups_by_type[action_type])
        source_count = len(source_groups_by_type[action_type])
        if episode_count < 10:
            targets[action_type] = "ignore"
        elif episode_count < 30 or source_count < 3:
            targets[action_type] = "auxiliary"
        else:
            targets[action_type] = "primary"
    return targets


def _validate_action_type_training_tiers(
    action_rows: Sequence[Mapping[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    targets = _action_type_tier_targets(action_rows)
    for row in action_rows:
        action_type = row.get("action_type")
        parent_tier = row.get("training_tier")
        if _is_manual_ntu_action(row):
            expected = str(parent_tier) if parent_tier in _TIER_RANK else "ignore"
            if row.get("action_type_training_tier") != expected:
                _issue(
                    issues,
                    "action_type_training_tier_mismatch",
                    "action",
                    row.get("label_id"),
                    {"expected": expected, "actual": row.get("action_type_training_tier")},
                )
            continue
        expected = targets.get(str(action_type), "ignore")
        if parent_tier in _TIER_RANK:
            expected = min(
                (str(parent_tier), expected), key=lambda tier: _TIER_RANK[tier]
            )
        else:
            expected = "ignore"
        if row.get("action_type_training_tier") != expected:
            _issue(
                issues,
                "action_type_training_tier_mismatch",
                "action",
                row.get("label_id"),
                {"expected": expected, "actual": row.get("action_type_training_tier")},
            )


def _action_type_counts_by_tier(
    action_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in action_rows:
        action_type = str(row.get("action_type"))
        tier = str(row.get("action_type_training_tier"))
        counts[tier][action_type] += 1
    return {tier: dict(sorted(values.items())) for tier, values in sorted(counts.items())}


def _migrate_fall_event(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    matched_action: Mapping[str, Any] | None,
    action_v3_by_v2: Mapping[str, Mapping[str, Any]],
    official: bool,
    reviewed_boundary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    anchor_label_id = _required_string(row, "label_id")
    physical_event_id = _stable_id(
        "physical", EVENT_SCHEMA_VERSION, row["video_id"], anchor_label_id
    )
    label_id = _stable_id(
        "eventv3", EVENT_SCHEMA_VERSION, "fall_event", physical_event_id
    )
    source_refs = [_source_ref_from_event(row)]
    linked_action_ids: list[str] = []
    quality_flags: list[str] = []
    track_id: str | None = None
    subtype = "unknown"
    subtype_tier = "ignore"
    subject_id = str(manifest.get("subject_id") or "unknown")
    if matched_action is not None:
        matched_v2_id = _required_string(matched_action, "label_id")
        matched_v3 = action_v3_by_v2[matched_v2_id]
        source_refs.append(_source_ref_from_action(matched_action))
        linked_action_ids.append(str(matched_v3["label_id"]))
        subtype = FALL_ACTION_SUBTYPES[_required_string(matched_action, "action_id")]
        subtype_tier = (
            "auxiliary" if matched_v3["training_tier"] != "ignore" else "ignore"
        )
        quality_flags = list(matched_v3["quality_flags"])
        track_id = matched_v3["track_id"]
        if matched_v3["subject_id"] != "unknown":
            subject_id = str(matched_v3["subject_id"])

    interval = _interval_from_v2(row, manifest)
    boundary_precision = "exact" if official else "approximate"
    reviewer_ids: list[str] = []
    review_status = "source_verified" if official else "single_annotated"
    note = str(row.get("note") or "")
    if reviewed_boundary is not None:
        frame_count = _required_int(manifest, "frame_count")
        fps_num = _required_int(manifest, "fps_num")
        fps_den = _required_int(manifest, "fps_den")
        interval = {
            "start_frame": 0,
            "end_frame_exclusive": frame_count,
            "frame_index_base": 0,
            "start_time": 0.0,
            "end_time_exclusive": _frame_time(frame_count, fps_num, fps_den),
        }
        boundary_precision = "exact"
        reviewer_ids = [str(reviewed_boundary["reviewer_id"])]
        review_status = "adjudicated"
        source_refs.append(_manual_source_ref(reviewed_boundary))
        note = _join_notes(note, str(reviewed_boundary["reason"]))
    target_status = _target_status(subject_id, quality_flags)
    eligible = bool(manifest.get("eligibility", True))
    if not eligible or target_status == "uncertain":
        training_tier = "ignore"
    elif official:
        training_tier = "primary"
    else:
        training_tier = "auxiliary"
    if training_tier == "ignore":
        subtype_tier = "ignore"
    source_refs = _unique_source_refs(source_refs)
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "label_id": label_id,
        "asset_id": _required_string(manifest, "asset_id"),
        "video_id": _required_string(row, "video_id"),
        "content_sha256": _required_string(manifest, "sha256"),
        "subject_id": subject_id,
        "source_group_id": _source_group(manifest),
        "sample_group_id": _sample_group_id(manifest),
        "track_id": track_id,
        **interval,
        "target_status": target_status,
        "boundary_precision": boundary_precision,
        "quality_flags": quality_flags,
        "training_tier": training_tier,
        "source_refs": source_refs,
        "annotator_id": "official_source" if official else str(
            matched_action.get("labeler", "unknown") if matched_action else "unknown"
        ),
        "reviewer_ids": reviewer_ids,
        "review_status": review_status,
        "note": note,
        "physical_event_id": physical_event_id,
        "task_type": "fall_event",
        "label_role": "positive",
        "event_type": "fall",
        "event_subtype": subtype,
        "event_outcome": "fell",
        "hard_negative_type": None,
        "onset_frame": interval["start_frame"],
        "peak_frame": None,
        "impact_frame": None,
        "recovery_frame": None,
        "linked_action_ids": linked_action_ids,
        "contact_evidence": "not_applicable",
        "subtype_training_tier": subtype_tier,
    }


def _matching_ntu_full_clip_boundary(
    rules: Sequence[Mapping[str, Any]],
    action: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if action is None:
        return None
    video_id = _required_string(action, "video_id")
    frame_count = _required_int(manifest, "frame_count")
    matches = [
        rule
        for rule in rules
        if video_id.startswith(str(rule["video_id_prefix"]))
        and action.get("action_id") in rule["action_ids"]
        and action.get("start_frame") == 0
        and frame_count - 1 - int(action.get("end_frame", -1))
        in rule["accepted_source_end_frame_gaps"]
    ]
    if len(matches) > 1:
        raise ValueError(
            f"multiple reviewed NTU full-clip boundary policies match {video_id}"
        )
    return matches[0] if matches else None


def _near_fall_positive_from_action(
    action: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    physical_event_id = _stable_id(
        "physical", EVENT_SCHEMA_VERSION, "near_fall", action["label_id"]
    )
    task_type = "near_fall_event"
    return {
        **{
            key: action[key]
            for key in (
                "asset_id",
                "video_id",
                "content_sha256",
                "subject_id",
                "source_group_id",
                "sample_group_id",
                "track_id",
                "start_frame",
                "end_frame_exclusive",
                "frame_index_base",
                "start_time",
                "end_time_exclusive",
                "target_status",
                "boundary_precision",
                "quality_flags",
                "training_tier",
                "annotator_id",
            )
        },
        "schema_version": EVENT_SCHEMA_VERSION,
        "label_id": _stable_id(
            "eventv3", EVENT_SCHEMA_VERSION, task_type, physical_event_id
        ),
        "source_refs": _unique_source_refs(
            [*action["source_refs"], _manual_source_ref(decision)]
        ),
        "reviewer_ids": [str(decision["reviewer_id"])],
        "review_status": "adjudicated",
        "note": _join_notes(str(action.get("note") or ""), str(decision["reason"])),
        "physical_event_id": physical_event_id,
        "task_type": task_type,
        "label_role": "positive",
        "event_type": "near_fall",
        "event_subtype": str(decision["event_subtype"]),
        "event_outcome": "recovered_without_fall",
        "hard_negative_type": None,
        "onset_frame": int(action["start_frame"]),
        "peak_frame": None,
        "impact_frame": None,
        "recovery_frame": int(action["end_frame_exclusive"]) - 1,
        "linked_action_ids": [str(action["label_id"])],
        "contact_evidence": "unknown",
        "subtype_training_tier": str(action["training_tier"]),
    }


def _event_ignore_from_action(
    action: Mapping[str, Any], task_type: str
) -> dict[str, Any]:
    return {
        **{
            key: action[key]
            for key in (
                "asset_id",
                "video_id",
                "content_sha256",
                "subject_id",
                "source_group_id",
                "sample_group_id",
                "track_id",
                "start_frame",
                "end_frame_exclusive",
                "frame_index_base",
                "start_time",
                "end_time_exclusive",
                "target_status",
                "boundary_precision",
                "quality_flags",
                "source_refs",
                "annotator_id",
                "reviewer_ids",
                "review_status",
                "note",
            )
        },
        "schema_version": EVENT_SCHEMA_VERSION,
        "label_id": _stable_id(
            "eventv3", EVENT_SCHEMA_VERSION, task_type, "ignore", action["label_id"]
        ),
        "training_tier": "ignore",
        "physical_event_id": None,
        "task_type": task_type,
        "label_role": "ignore",
        "event_type": None,
        "event_subtype": None,
        "event_outcome": None,
        "hard_negative_type": None,
        "onset_frame": None,
        "peak_frame": None,
        "impact_frame": None,
        "recovery_frame": None,
        "linked_action_ids": [action["label_id"]],
        "contact_evidence": "not_applicable",
        "subtype_training_tier": "ignore",
    }


def _event_negative_from_action(
    action: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    training_tier_cap: Any = None,
) -> dict[str, Any]:
    common = {
        key: action[key]
        for key in (
            "asset_id",
            "video_id",
            "content_sha256",
            "subject_id",
            "source_group_id",
            "sample_group_id",
            "track_id",
            "start_frame",
            "end_frame_exclusive",
            "frame_index_base",
            "start_time",
            "end_time_exclusive",
            "target_status",
            "boundary_precision",
            "quality_flags",
            "training_tier",
            "annotator_id",
        )
    }
    if decision.get("event_training_tier") is not None:
        common["training_tier"] = str(decision["event_training_tier"])
    if training_tier_cap is not None:
        cap = str(training_tier_cap)
        if cap not in _TIER_RANK:
            raise ValueError(f"invalid event training_tier_cap: {cap}")
        common["training_tier"] = min(
            (str(common["training_tier"]), cap),
            key=lambda tier: _TIER_RANK[tier],
        )
    task_type = str(decision["task_type"])
    return {
        **common,
        "schema_version": EVENT_SCHEMA_VERSION,
        "label_id": _stable_id(
            "eventv3",
            EVENT_SCHEMA_VERSION,
            task_type,
            "negative",
            action["label_id"],
            decision["decision_id"],
        ),
        "source_refs": _unique_source_refs(
            [*action["source_refs"], _manual_source_ref(decision)]
        ),
        "reviewer_ids": [str(decision["reviewer_id"])],
        "review_status": "adjudicated",
        "note": str(decision["reason"]),
        "physical_event_id": None,
        "task_type": task_type,
        "label_role": "negative",
        "event_type": None,
        "event_subtype": None,
        "event_outcome": None,
        "hard_negative_type": str(decision["hard_negative_type"]),
        "onset_frame": None,
        "peak_frame": None,
        "impact_frame": None,
        "recovery_frame": None,
        "linked_action_ids": [str(action["label_id"])],
        "contact_evidence": "not_applicable",
        "subtype_training_tier": "ignore",
    }


def _event_negative_from_event(
    source_event: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    task_type = str(decision["task_type"])
    return {
        **{
            key: source_event[key]
            for key in (
                "asset_id",
                "video_id",
                "content_sha256",
                "subject_id",
                "source_group_id",
                "sample_group_id",
                "track_id",
                "start_frame",
                "end_frame_exclusive",
                "frame_index_base",
                "start_time",
                "end_time_exclusive",
                "target_status",
                "boundary_precision",
                "quality_flags",
                "training_tier",
                "annotator_id",
            )
        },
        "schema_version": EVENT_SCHEMA_VERSION,
        "label_id": _stable_id(
            "eventv3",
            EVENT_SCHEMA_VERSION,
            task_type,
            "negative",
            source_event["label_id"],
            decision["decision_id"],
        ),
        "source_refs": _unique_source_refs(
            [*source_event["source_refs"], _manual_source_ref(decision)]
        ),
        "reviewer_ids": [str(decision["reviewer_id"])],
        "review_status": "adjudicated",
        "note": _join_notes(
            str(source_event.get("note") or ""), str(decision["reason"])
        ),
        "physical_event_id": None,
        "task_type": task_type,
        "label_role": "negative",
        "event_type": None,
        "event_subtype": None,
        "event_outcome": None,
        "hard_negative_type": str(decision["hard_negative_type"]),
        "onset_frame": None,
        "peak_frame": None,
        "impact_frame": None,
        "recovery_frame": None,
        "linked_action_ids": list(source_event["linked_action_ids"]),
        "contact_evidence": "not_applicable",
        "subtype_training_tier": "ignore",
    }


def _manual_source_ref(decision: Mapping[str, Any]) -> dict[str, str]:
    decision_key = str(
        decision.get("decision_key") or decision.get("directive_key")
    )
    return {
        "source_type": "manual_v3",
        "source_record_id": decision_key,
        "source_annotation_path": str(decision["source_annotation_path"]),
        "source_annotation_sha256": str(decision["source_annotation_sha256"]),
        "source_label_id": decision_key,
    }


def _join_notes(*values: str) -> str:
    return " ".join(value.strip() for value in values if value.strip())


def _common_from_v2_action(
    row: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    quality = str(row.get("quality") or "clear")
    quality_flags = list(QUALITY_FLAG_MAP.get(quality, []))
    subject_id = str(row.get("subject_id") or manifest.get("subject_id") or "unknown")
    target_status = _target_status(subject_id, quality_flags)
    action_id = _required_string(row, "action_id")
    is_official_clip_source = row.get("source") in OFFICIAL_CLIP_ACTION_SOURCES
    is_manual_exact_clip_source = (
        row.get("source") in MANUAL_EXACT_CLIP_ACTION_SOURCES
    )
    eligible = bool(manifest.get("eligibility", True))
    if (
        action_id == "U01"
        or not eligible
        or target_status == "uncertain"
        or SEVERE_QUALITY_FLAGS.intersection(quality_flags)
    ):
        training_tier = "ignore"
    elif is_manual_exact_clip_source:
        training_tier = "primary"
    elif is_official_clip_source:
        training_tier = "auxiliary"
    elif action_id in {"C03", "C04", "C05", "D04"} or quality_flags:
        training_tier = "auxiliary"
    else:
        training_tier = "primary"
    tier_cap = manifest.get("training_tier_cap")
    if tier_cap is not None:
        if tier_cap not in _TIER_RANK:
            raise ValueError(f"invalid manifest training_tier_cap: {tier_cap}")
        training_tier = min(
            (training_tier, str(tier_cap)), key=lambda tier: _TIER_RANK[tier]
        )
    track = (
        None
        if is_official_clip_source or is_manual_exact_clip_source
        else row.get("cvat_track_id")
    )
    return {
        "asset_id": _required_string(manifest, "asset_id"),
        "video_id": _required_string(row, "video_id"),
        "content_sha256": _required_string(manifest, "sha256"),
        "subject_id": subject_id,
        "source_group_id": _source_group(manifest),
        "sample_group_id": _sample_group_id(manifest),
        "track_id": None if track is None else str(track),
        **_interval_from_v2(row, manifest),
        "target_status": target_status,
        "boundary_precision": (
            "unknown"
            if action_id == "U01" or is_official_clip_source
            else "exact"
            if is_manual_exact_clip_source
            else "approximate"
        ),
        "quality_flags": quality_flags,
        "training_tier": training_tier,
        "source_refs": [_source_ref_from_action(row)],
        "annotator_id": str(row.get("labeler") or "unknown"),
        "reviewer_ids": [],
        "review_status": (
            "source_verified" if is_official_clip_source else "single_annotated"
        ),
        "note": str(row.get("note") or ""),
    }


def _source_ref_from_action(row: Mapping[str, Any]) -> dict[str, str]:
    source = row.get("source")
    if source == "toaga_official_walking":
        source_type = "v2_toaga_official_walking"
    elif source == "ntu_rgbd_manual_clip_label":
        source_type = "v2_ntu_rgbd_manual_clip_label"
    elif source == "ntu_rgbd_clip_label":
        source_type = "v2_ntu_rgbd_clip_label"
    else:
        source_type = "v2_cvat_action"
    return {
        "source_type": source_type,
        "source_record_id": _required_string(row, "source_record_id"),
        "source_annotation_path": _required_string(row, "source_annotation_path"),
        "source_annotation_sha256": _required_string(
            row, "source_annotation_sha256"
        ),
        "source_label_id": _required_string(row, "label_id"),
    }


def _is_manual_ntu_action(row: Mapping[str, Any]) -> bool:
    return any(
        isinstance(source, Mapping)
        and source.get("source_type") == "v2_ntu_rgbd_manual_clip_label"
        for source in row.get("source_refs") or []
    )


def _source_ref_from_event(row: Mapping[str, Any]) -> dict[str, str]:
    label_source = row.get("label_source")
    source_type = (
        "v2_le2i_txt" if label_source == "le2i_txt" else "v2_cvat_action_mapping"
    )
    return {
        "source_type": source_type,
        "source_record_id": _required_string(row, "source_record_id"),
        "source_annotation_path": _required_string(row, "source_annotation_path"),
        "source_annotation_sha256": _required_string(
            row, "source_annotation_sha256"
        ),
        "source_label_id": _required_string(row, "label_id"),
    }


def _interval_from_v2(
    row: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, int | float]:
    start = _required_int(row, "start_frame")
    end_exclusive = _required_int(row, "end_frame") + 1
    frame_count = _required_int(manifest, "frame_count")
    if start < 0 or start >= end_exclusive or end_exclusive > frame_count:
        raise ValueError(
            f"invalid v2 interval for {row.get('label_id')}: [{start}, {end_exclusive})"
        )
    fps_num = _required_int(manifest, "fps_num")
    fps_den = _required_int(manifest, "fps_den")
    if fps_num <= 0 or fps_den <= 0:
        raise ValueError(f"invalid FPS for {manifest.get('video_id')}")
    return {
        "start_frame": start,
        "end_frame_exclusive": end_exclusive,
        "frame_index_base": 0,
        "start_time": _frame_time(start, fps_num, fps_den),
        "end_time_exclusive": _frame_time(end_exclusive, fps_num, fps_den),
    }


def _target_status(subject_id: str, quality_flags: Iterable[str]) -> str:
    flags = set(quality_flags)
    if {"multi_person_uncertain", "off_screen"}.intersection(flags):
        return "uncertain"
    return "single_person_assumed" if subject_id == "unknown" else "confirmed"


def _migration_report(
    *,
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    actions_v3: Sequence[Mapping[str, Any]],
    events_v3: Sequence[Mapping[str, Any]],
    deduplicated_official_action_falls: int,
    unresolved_post_fall_states: int,
    manual_negative_decisions: Sequence[Mapping[str, Any]],
    matched_manual_negative_decisions: set[str],
    reviewed_directives: Sequence[Mapping[str, Any]],
    matched_reviewed_directives: Mapping[str, int],
    reviewed_decision_matches: Mapping[str, int],
) -> dict[str, Any]:
    excluded_events = Counter(
        str(row.get("event_type"))
        for row in event_rows
        if row.get("event_type") != "fall"
    )
    return {
        "schema_version": "fall-risk-training-label-migration-v3",
        "input_counts": {
            "action_labels_v2": len(action_rows),
            "event_labels_v2": len(event_rows),
        },
        "output_counts": {
            "action_labels_v3": len(actions_v3),
            "event_labels_v3": len(events_v3),
        },
        "action_counts_by_id": dict(
            sorted(Counter(str(row["action_id"]) for row in actions_v3).items())
        ),
        "action_counts_by_tier": dict(
            sorted(Counter(str(row["training_tier"]) for row in actions_v3).items())
        ),
        "action_type_counts_by_tier": _action_type_counts_by_tier(actions_v3),
        "event_counts_by_role": dict(
            sorted(Counter(str(row["label_role"]) for row in events_v3).items())
        ),
        "fall_positive_count": sum(
            row.get("event_type") == "fall" for row in events_v3
        ),
        "near_fall_positive_count": sum(
            row.get("event_type") == "near_fall" for row in events_v3
        ),
        "deduplicated_official_action_falls": deduplicated_official_action_falls,
        "unresolved_post_fall_states": unresolved_post_fall_states,
        "excluded_v2_event_counts": dict(sorted(excluded_events.items())),
        "manual_negative_count": len(matched_manual_negative_decisions),
        "unmatched_manual_negative_decisions": sorted(
            str(decision["decision_key"])
            for decision in manual_negative_decisions
            if str(decision["decision_key"])
            not in matched_manual_negative_decisions
        ),
        "reviewed_decision_matches": dict(sorted(reviewed_decision_matches.items())),
        "matched_reviewed_directive_count": sum(
            str(rule["directive_key"]) in matched_reviewed_directives
            for rule in reviewed_directives
        ),
        "unmatched_reviewed_decisions": sorted(
            str(rule["directive_key"])
            for rule in reviewed_directives
            if str(rule["directive_key"]) not in matched_reviewed_directives
        ),
        "automatic_negative_count": sum(
            row.get("label_role") == "negative"
            and any(
                source.get("source_type") == "manual_v3"
                and ":" in str(source.get("source_record_id"))
                for source in row.get("source_refs") or []
                if isinstance(source, Mapping)
            )
            for row in events_v3
        )
        - len(matched_manual_negative_decisions),
    }


def _validation_counts(
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "action_labels": len(action_rows),
        "event_labels": len(event_rows),
        "action_primary": sum(row.get("training_tier") == "primary" for row in action_rows),
        "action_auxiliary": sum(row.get("training_tier") == "auxiliary" for row in action_rows),
        "action_ignore": sum(row.get("training_tier") == "ignore" for row in action_rows),
        "event_positive": sum(row.get("label_role") == "positive" for row in event_rows),
        "event_negative": sum(row.get("label_role") == "negative" for row in event_rows),
        "event_ignore": sum(row.get("label_role") == "ignore" for row in event_rows),
        "fall_positive": sum(row.get("event_type") == "fall" for row in event_rows),
        "near_fall_positive": sum(row.get("event_type") == "near_fall" for row in event_rows),
        "primary_fall_positive": sum(
            row.get("event_type") == "fall" and row.get("training_tier") == "primary"
            for row in event_rows
        ),
        "primary_near_fall_positive": sum(
            row.get("event_type") == "near_fall"
            and row.get("training_tier") == "primary"
            for row in event_rows
        ),
        "fall_negative": sum(
            row.get("task_type") == "fall_event" and row.get("label_role") == "negative"
            for row in event_rows
        ),
        "near_fall_negative": sum(
            row.get("task_type") == "near_fall_event"
            and row.get("label_role") == "negative"
            for row in event_rows
        ),
        "primary_fall_negative": sum(
            row.get("task_type") == "fall_event"
            and row.get("label_role") == "negative"
            and row.get("training_tier") == "primary"
            for row in event_rows
        ),
        "primary_near_fall_negative": sum(
            row.get("task_type") == "near_fall_event"
            and row.get("label_role") == "negative"
            and row.get("training_tier") == "primary"
            for row in event_rows
        ),
    }


def _hard_negative_coverage(
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    present_by_task: dict[str, set[str]] = defaultdict(set)
    for row in event_rows:
        if (
            row.get("label_role") == "negative"
            and row.get("training_tier") == "primary"
            and isinstance(row.get("hard_negative_type"), str)
        ):
            present_by_task[str(row.get("task_type"))].add(
                str(row["hard_negative_type"])
            )
    return {
        task_type: {
            "present": sorted(present_by_task[task_type]),
            "missing": sorted(required - present_by_task[task_type]),
        }
        for task_type, required in HARD_NEGATIVES_BY_TASK.items()
    }


def _training_warnings(
    event_rows: Sequence[Mapping[str, Any]],
    hard_negative_coverage: Mapping[str, Mapping[str, Sequence[str]]],
) -> list[str]:
    warnings: list[str] = []
    if not any(row.get("event_type") == "near_fall" for row in event_rows):
        warnings.append("near_fall_has_no_positive_labels")
    if hard_negative_coverage["fall_event"]["missing"]:
        warnings.append("fall_missing_hard_negatives")
    if hard_negative_coverage["near_fall_event"]["missing"]:
        warnings.append("near_fall_missing_hard_negatives")
    return warnings


def _validate_common_semantics(
    row: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    kind: str,
    issues: list[dict[str, Any]],
) -> None:
    label_id = row.get("label_id")
    video_id = row.get("video_id")
    manifest = manifests.get(video_id) if isinstance(video_id, str) else None
    if manifest is None:
        _issue(issues, "manifest_missing", kind, label_id)
        return
    for field, manifest_field in (
        ("asset_id", "asset_id"),
        ("content_sha256", "sha256"),
        ("source_group_id", "source_group_id"),
    ):
        expected = manifest.get(manifest_field)
        if field == "source_group_id" and not expected:
            expected = manifest.get("original_event_id") or manifest.get("video_id")
        if row.get(field) != expected:
            _issue(issues, f"{field}_mismatch", kind, label_id)
    start = row.get("start_frame")
    end = row.get("end_frame_exclusive")
    if not isinstance(start, int) or not isinstance(end, int) or start >= end:
        _issue(issues, "invalid_half_open_interval", kind, label_id)
        return
    frame_count = manifest.get("frame_count")
    if isinstance(frame_count, int) and end > frame_count:
        _issue(issues, "interval_exceeds_frame_count", kind, label_id)
    fps_num = manifest.get("fps_num")
    fps_den = manifest.get("fps_den")
    if isinstance(fps_num, int) and isinstance(fps_den, int) and fps_num > 0 and fps_den > 0:
        expected_start = _frame_time(start, fps_num, fps_den)
        expected_end = _frame_time(end, fps_num, fps_den)
        if not _close_number(row.get("start_time"), expected_start):
            _issue(issues, "start_time_mismatch", kind, label_id)
        if not _close_number(row.get("end_time_exclusive"), expected_end):
            _issue(issues, "end_time_mismatch", kind, label_id)
    flags = set(row.get("quality_flags") or [])
    if row.get("training_tier") == "primary" and (
        row.get("target_status") == "uncertain"
        or SEVERE_QUALITY_FLAGS.intersection(flags)
        or not bool(manifest.get("eligibility", True))
    ):
        _issue(issues, "primary_label_not_eligible", kind, label_id)
    _validate_source_refs(row, kind, issues)


def _validate_action_semantics(
    row: Mapping[str, Any], issues: list[dict[str, Any]]
) -> None:
    label_id = row.get("label_id")
    action_id = row.get("action_id")
    expected = ACTION_DEFINITIONS.get(str(action_id))
    if expected is None:
        _issue(issues, "unknown_action_id", "action", label_id)
        return
    if (row.get("action_family"), row.get("action_type")) != expected:
        _issue(issues, "action_taxonomy_mismatch", "action", label_id)
    if action_id == "U01" and row.get("training_tier") != "ignore":
        _issue(issues, "u01_must_be_ignore", "action", label_id)
    if action_id == "U01" and row.get("action_type_training_tier") != "ignore":
        _issue(issues, "u01_action_type_must_be_ignore", "action", label_id)
    if action_id != "U01" and row.get("action_type") is None:
        _issue(issues, "action_type_required", "action", label_id)
    parent_tier = row.get("training_tier")
    subtype_tier = row.get("action_type_training_tier")
    if (
        parent_tier in _TIER_RANK
        and subtype_tier in _TIER_RANK
        and _TIER_RANK[str(subtype_tier)] > _TIER_RANK[str(parent_tier)]
    ):
        _issue(issues, "action_type_tier_exceeds_parent", "action", label_id)


def _validate_event_semantics(
    row: Mapping[str, Any], issues: list[dict[str, Any]]
) -> None:
    label_id = row.get("label_id")
    role = row.get("label_role")
    event_fields = ("event_type", "event_subtype", "event_outcome")
    point_fields = ("onset_frame", "peak_frame", "impact_frame", "recovery_frame")
    if role == "positive":
        if any(row.get(field) is None for field in event_fields):
            _issue(issues, "positive_event_fields_required", "event", label_id)
        if row.get("physical_event_id") is None:
            _issue(issues, "physical_event_id_required", "event", label_id)
        if row.get("hard_negative_type") is not None:
            _issue(issues, "positive_has_hard_negative_type", "event", label_id)
        if row.get("event_type") == "fall":
            if row.get("task_type") != "fall_event" or row.get("event_outcome") != "fell":
                _issue(issues, "fall_task_or_outcome_invalid", "event", label_id)
            if row.get("recovery_frame") is not None:
                _issue(issues, "fall_recovery_forbidden", "event", label_id)
            if row.get("contact_evidence") != "not_applicable":
                _issue(issues, "fall_contact_evidence_invalid", "event", label_id)
        elif row.get("event_type") == "near_fall":
            if (
                row.get("task_type") != "near_fall_event"
                or row.get("event_outcome") != "recovered_without_fall"
            ):
                _issue(issues, "near_fall_task_or_outcome_invalid", "event", label_id)
            if row.get("impact_frame") is not None:
                _issue(issues, "near_fall_impact_forbidden", "event", label_id)
            if row.get("training_tier") == "primary":
                if row.get("recovery_frame") is None:
                    _issue(issues, "near_fall_recovery_required", "event", label_id)
                if row.get("review_status") not in {"double_reviewed", "adjudicated"}:
                    _issue(issues, "near_fall_double_review_required", "event", label_id)
        if row.get("onset_frame") is None:
            _issue(issues, "event_onset_required", "event", label_id)
        parent_tier = row.get("training_tier")
        subtype_tier = row.get("subtype_training_tier")
        if (
            parent_tier in _TIER_RANK
            and subtype_tier in _TIER_RANK
            and _TIER_RANK[str(subtype_tier)] > _TIER_RANK[str(parent_tier)]
        ):
            _issue(issues, "subtype_tier_exceeds_parent", "event", label_id)
        if row.get("event_subtype") == "unknown" and subtype_tier != "ignore":
            _issue(issues, "unknown_subtype_must_be_ignore", "event", label_id)
    elif role == "negative":
        if any(row.get(field) is not None for field in event_fields):
            _issue(issues, "negative_event_fields_forbidden", "event", label_id)
        if row.get("hard_negative_type") is None:
            _issue(issues, "negative_type_required", "event", label_id)
        allowed_types = HARD_NEGATIVES_BY_TASK.get(str(row.get("task_type")), set())
        if row.get("hard_negative_type") not in allowed_types | {"background"}:
            _issue(issues, "negative_type_task_mismatch", "event", label_id)
        if any(row.get(field) is not None for field in point_fields):
            _issue(issues, "negative_point_fields_forbidden", "event", label_id)
        if row.get("training_tier") == "ignore":
            _issue(issues, "negative_tier_cannot_be_ignore", "event", label_id)
        if row.get("subtype_training_tier") != "ignore":
            _issue(issues, "negative_subtype_tier_must_be_ignore", "event", label_id)
        if not any(
            source.get("source_type") == "manual_v3"
            for source in row.get("source_refs") or []
            if isinstance(source, Mapping)
        ):
            _issue(issues, "negative_manual_confirmation_required", "event", label_id)
        if row.get("review_status") in {"unreviewed", "source_verified"}:
            _issue(issues, "negative_review_required", "event", label_id)
    elif role == "ignore":
        if any(row.get(field) is not None for field in event_fields):
            _issue(issues, "ignore_event_fields_forbidden", "event", label_id)
        if row.get("hard_negative_type") is not None:
            _issue(issues, "ignore_negative_type_forbidden", "event", label_id)
        if any(row.get(field) is not None for field in point_fields):
            _issue(issues, "ignore_point_fields_forbidden", "event", label_id)
        if row.get("training_tier") != "ignore":
            _issue(issues, "ignore_tier_required", "event", label_id)
        if row.get("subtype_training_tier") != "ignore":
            _issue(issues, "ignore_subtype_tier_required", "event", label_id)
    start = row.get("start_frame")
    end = row.get("end_frame_exclusive")
    if isinstance(start, int) and isinstance(end, int):
        for field in point_fields:
            value = row.get(field)
            if isinstance(value, int) and not start <= value < end:
                _issue(issues, "event_point_outside_interval", "event", label_id, field)
        onset = row.get("onset_frame")
        recovery = row.get("recovery_frame")
        if isinstance(onset, int) and isinstance(recovery, int) and recovery < onset:
            _issue(issues, "recovery_before_onset", "event", label_id)


def _validate_cross_references(
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    events_by_id: Mapping[str, Mapping[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    for action in action_rows:
        event_id = action.get("linked_event_id")
        if event_id is None:
            continue
        event = events_by_id.get(str(event_id))
        if event is None:
            _issue(issues, "linked_event_missing", "action", action.get("label_id"))
            continue
        if action.get("video_id") != event.get("video_id"):
            _issue(issues, "linked_event_video_mismatch", "action", action.get("label_id"))
        if action.get("subject_id") != event.get("subject_id"):
            _issue(issues, "linked_event_subject_mismatch", "action", action.get("label_id"))
        if event.get("label_role") != "positive":
            _issue(issues, "action_must_link_positive_event", "action", action.get("label_id"))
        if action.get("label_id") not in (event.get("linked_action_ids") or []):
            _issue(issues, "linked_event_not_reciprocal", "action", action.get("label_id"))
        action_id = action.get("action_id")
        event_type = event.get("event_type")
        if action_id in FALL_ACTION_SUBTYPES:
            if event_type != "fall":
                _issue(issues, "fall_action_event_type_mismatch", "action", action.get("label_id"))
            expected_subtype = FALL_ACTION_SUBTYPES[str(action_id)]
            if event.get("event_subtype") != expected_subtype:
                _issue(issues, "fall_action_subtype_mismatch", "action", action.get("label_id"))
        elif action_id == "D04":
            if event_type != "fall":
                _issue(issues, "post_fall_parent_must_be_fall", "action", action.get("label_id"))
            if action.get("start_frame", -1) < event.get(
                "end_frame_exclusive", math.inf
            ):
                _issue(issues, "post_fall_state_before_event_end", "action", action.get("label_id"))
        elif action_id not in {"C03", "C04", "C05"}:
            _issue(issues, "action_type_cannot_link_event", "action", action.get("label_id"))
    for event in event_rows:
        for action_id in event.get("linked_action_ids") or []:
            action = actions_by_id.get(str(action_id))
            if action is None:
                _issue(issues, "linked_action_missing", "event", event.get("label_id"))
            elif action.get("video_id") != event.get("video_id"):
                _issue(issues, "linked_action_video_mismatch", "event", event.get("label_id"))
            elif action.get("subject_id") != event.get("subject_id"):
                _issue(issues, "linked_action_subject_mismatch", "event", event.get("label_id"))
            elif (
                event.get("label_role") == "positive"
                and action.get("linked_event_id") != event.get("label_id")
            ):
                _issue(issues, "linked_action_not_reciprocal", "event", event.get("label_id"))


def _validate_source_refs(
    row: Mapping[str, Any], kind: str, issues: list[dict[str, Any]]
) -> None:
    for source in row.get("source_refs") or []:
        if not isinstance(source, Mapping):
            continue
        path_value = source.get("source_annotation_path")
        expected_hash = source.get("source_annotation_sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            continue
        path = Path(path_value)
        if not path.is_file():
            _issue(issues, "source_file_missing", kind, row.get("label_id"), path_value)
        elif _sha256_file(path) != expected_hash:
            _issue(issues, "source_hash_mismatch", kind, row.get("label_id"), path_value)


def _validate_schema_node(
    value: Any,
    schema: Mapping[str, Any],
    kind: str,
    label_id: Any,
    issues: list[dict[str, Any]],
    path: str = "$",
) -> None:
    if not _matches_schema_type(value, schema.get("type")):
        _issue(issues, "schema_type", kind, label_id, path)
        return
    if "const" in schema and value != schema["const"]:
        _issue(issues, "schema_const", kind, label_id, path)
    if "enum" in schema and value not in schema["enum"]:
        _issue(issues, "schema_enum", kind, label_id, path)
    if isinstance(value, str):
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            _issue(issues, "schema_pattern", kind, label_id, path)
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            _issue(issues, "schema_min_length", kind, label_id, path)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            _issue(issues, "schema_minimum", kind, label_id, path)
        exclusive = schema.get("exclusiveMinimum")
        if isinstance(exclusive, (int, float)) and value <= exclusive:
            _issue(issues, "schema_exclusive_minimum", kind, label_id, path)
    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            _issue(issues, "schema_min_items", kind, label_id, path)
        if schema.get("uniqueItems"):
            normalized = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(normalized) != len(set(normalized)):
                _issue(issues, "schema_unique_items", kind, label_id, path)
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_node(
                    item, item_schema, kind, label_id, issues, f"{path}[{index}]"
                )
    if isinstance(value, Mapping):
        required = schema.get("required") or []
        for field in required:
            if field not in value:
                _issue(issues, "schema_required", kind, label_id, f"{path}.{field}")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for field in value:
                if field not in properties:
                    _issue(issues, "schema_unknown_field", kind, label_id, f"{path}.{field}")
        for field, child in value.items():
            child_schema = properties.get(field)
            if isinstance(child_schema, Mapping):
                _validate_schema_node(
                    child,
                    child_schema,
                    kind,
                    label_id,
                    issues,
                    f"{path}.{field}",
                )


def _matches_schema_type(value: Any, expected: Any) -> bool:
    if expected is None:
        return True
    types = expected if isinstance(expected, list) else [expected]
    for type_name in types:
        if type_name == "null" and value is None:
            return True
        if type_name == "object" and isinstance(value, Mapping):
            return True
        if type_name == "array" and isinstance(value, list):
            return True
        if type_name == "string" and isinstance(value, str):
            return True
        if type_name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if type_name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return math.isfinite(float(value))
        if type_name == "boolean" and isinstance(value, bool):
            return True
    return False


def _index_unique(
    rows: Sequence[Mapping[str, Any]], key: str, name: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = _required_string(row, key)
        if value in result:
            raise ValueError(f"duplicate {name} {key}: {value}")
        result[value] = row
    return result


def _index_for_validation(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    name: str,
    issues: list[dict[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str):
            _issue(issues, f"{name}_key_invalid", name, value)
        elif value in result:
            _issue(issues, f"duplicate_{name}_key", name, value)
        else:
            result[value] = row
    return result


def _required_manifest(
    manifests: Mapping[str, Mapping[str, Any]], video_id: str
) -> Mapping[str, Any]:
    manifest = manifests.get(video_id)
    if manifest is None:
        raise ValueError(f"video_id missing from manifest: {video_id}")
    return manifest


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _source_group(manifest: Mapping[str, Any]) -> str:
    return str(
        manifest.get("source_group_id")
        or manifest.get("original_event_id")
        or manifest.get("video_id")
    )


def _sample_group_id(manifest: Mapping[str, Any]) -> str:
    return _stable_id(
        "samplegrp",
        manifest.get("original_event_id") or manifest.get("video_id"),
        manifest.get("sha256"),
    )


def _validate_split_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_PARTITIONS):
        raise ValueError("split ratios must contain exactly train, validation and test")
    normalized: dict[str, float] = {}
    for partition in SPLIT_PARTITIONS:
        value = ratios[partition]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"split ratio {partition} must be numeric")
        number = float(value)
        if not 0.0 < number < 1.0:
            raise ValueError(f"split ratio {partition} must be between 0 and 1")
        normalized[partition] = number
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")
    return normalized


def _split_tokens(
    asset_id: str,
    manifest: Mapping[str, Any],
    labels: Sequence[Mapping[str, Any]],
) -> set[str]:
    tokens = {f"asset:{asset_id}"}
    sources = (manifest, *labels)
    for source in sources:
        for field in (
            "content_sha256",
            "sha256",
            "sample_group_id",
            "physical_event_id",
            "source_group_id",
            "original_event_id",
        ):
            value = source.get(field)
            if _known_split_identifier(value):
                tokens.add(f"{field}:{str(value).strip()}")
        subject_id = source.get("subject_id")
        if _known_split_identifier(subject_id):
            tokens.add(f"subject_id:{str(subject_id).strip()}")
        for field in _ASSET_REFERENCE_FIELDS:
            value = source.get(field)
            if _known_split_identifier(value):
                tokens.add(f"asset:{str(value).strip()}")
        for field in _ASSET_REFERENCE_LIST_FIELDS:
            values = source.get(field)
            if isinstance(values, list):
                for value in values:
                    if _known_split_identifier(value):
                        tokens.add(f"asset:{str(value).strip()}")
        for field in _RELATION_GROUP_FIELDS:
            value = source.get(field)
            if _known_split_identifier(value):
                tokens.add(f"relation:{str(value).strip()}")
    return tokens


def _known_split_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.strip().lower() not in _UNKNOWN_IDENTIFIERS
    )


def _split_component_metrics(
    asset_ids: Sequence[str],
    labels_by_asset: Mapping[str, Sequence[tuple[str, Mapping[str, Any]]]],
) -> dict[str, int]:
    metrics = {
        "total_labels": 0,
        "primary_actions": 0,
        "primary_action_class_count": 0,
        "primary_fall_positive": 0,
        "primary_fall_negative": 0,
        "primary_near_fall_positive": 0,
        "primary_near_fall_negative": 0,
    }
    primary_action_types: set[str] = set()
    for asset_id in asset_ids:
        for kind, row in labels_by_asset[asset_id]:
            metrics["total_labels"] += 1
            if kind == "action":
                if row.get("action_type_training_tier") != "primary":
                    continue
                metrics["primary_actions"] += 1
                action_type = row.get("action_type")
                if isinstance(action_type, str):
                    primary_action_types.add(action_type)
                continue
            if row.get("training_tier") != "primary":
                continue
            task_type = row.get("task_type")
            role = row.get("label_role")
            if task_type == "fall_event" and role == "positive":
                metrics["primary_fall_positive"] += 1
            elif task_type == "fall_event" and role == "negative":
                metrics["primary_fall_negative"] += 1
            elif task_type == "near_fall_event" and role == "positive":
                metrics["primary_near_fall_positive"] += 1
            elif task_type == "near_fall_event" and role == "negative":
                metrics["primary_near_fall_negative"] += 1
    metrics["primary_action_class_count"] = len(primary_action_types)
    return metrics


def _balanced_component_partitions(
    components: Sequence[tuple[str, Sequence[str], Mapping[str, int]]],
    *,
    seed: str,
    ratios: Mapping[str, float],
) -> dict[str, str]:
    event_keys = (
        "primary_fall_positive",
        "primary_fall_negative",
        "primary_near_fall_positive",
        "primary_near_fall_negative",
    )
    totals = {
        key: sum(int(metrics[key]) for _, _, metrics in components)
        for key in (
            "total_labels",
            "primary_actions",
            "primary_action_class_count",
            *event_keys,
        )
    }
    counts = {
        partition: {key: 0 for key in totals}
        for partition in SPLIT_PARTITIONS
    }

    def tie_rank(component_id: str, partition: str = "") -> str:
        return hashlib.sha256(
            f"{seed}\0{component_id}\0{partition}".encode("utf-8")
        ).hexdigest()

    event_components = [
        component
        for component in components
        if sum(int(component[2][key]) for key in event_keys) > 0
    ]
    other_components = [component for component in components if component not in event_components]
    event_components.sort(
        key=lambda component: (
            -sum(int(component[2][key]) for key in event_keys),
            -int(component[2]["total_labels"]),
            tie_rank(component[0]),
        )
    )
    other_components.sort(
        key=lambda component: (
            -int(component[2]["total_labels"]),
            tie_rank(component[0]),
        )
    )

    assignments: dict[str, str] = {}
    anchored_components: list[tuple[str, Sequence[str], Mapping[str, int]]] = []
    anchor_partitions: list[str] = []
    if event_components:
        anchored_components.append(event_components[0])
        anchor_partitions.append("train")
    evaluation_anchors = sorted(
        event_components[1:],
        key=lambda component: (
            -int(component[2]["primary_action_class_count"]),
            -int(component[2]["primary_actions"]),
            -sum(int(component[2][key]) for key in event_keys),
            tie_rank(component[0]),
        ),
    )[:2]
    anchored_components.extend(evaluation_anchors)
    anchor_partitions.extend(SPLIT_PARTITIONS[1 : 1 + len(evaluation_anchors)])
    missing_anchor_partitions = [
        partition for partition in SPLIT_PARTITIONS if partition not in anchor_partitions
    ]
    action_anchor_candidates = sorted(
        (
            component
            for component in other_components
            if int(component[2]["primary_actions"]) > 0
        ),
        key=lambda component: (
            -int(component[2]["primary_action_class_count"]),
            -int(component[2]["primary_actions"]),
            -int(component[2]["total_labels"]),
            tie_rank(component[0]),
        ),
    )
    for component, partition in zip(
        action_anchor_candidates, missing_anchor_partitions, strict=False
    ):
        anchored_components.append(component)
        anchor_partitions.append(partition)
    for (component_id, _, metrics), partition in zip(
        anchored_components, anchor_partitions, strict=True
    ):
        assignments[component_id] = partition
        for key in totals:
            counts[partition][key] += int(metrics[key])

    anchored_ids = {component[0] for component in anchored_components}
    remaining_event_components = [
        component for component in event_components if component[0] not in anchored_ids
    ]
    remaining_other_components = [
        component for component in other_components if component[0] not in anchored_ids
    ]
    for component_id, _, metrics in (
        *remaining_event_components,
        *remaining_other_components,
    ):
        active_keys = (
            event_keys
            if sum(int(metrics[key]) for key in event_keys) > 0
            else ("total_labels", "primary_actions", "primary_action_class_count")
        )
        candidates = []
        for partition in SPLIT_PARTITIONS:
            cost = 0.0
            for key in active_keys:
                total = totals[key]
                if total <= 0:
                    continue
                target = total * ratios[partition]
                projected = counts[partition][key] + int(metrics[key])
                cost += ((projected - target) / max(target, 1.0)) ** 2
            candidates.append((cost, tie_rank(component_id, partition), partition))
        partition = min(candidates)[2]
        assignments[component_id] = partition
        for key in totals:
            counts[partition][key] += int(metrics[key])
    return assignments


def _split_label_index(
    action_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["label_id"]): row
        for row in (*action_rows, *event_rows)
        if isinstance(row.get("label_id"), str)
    }


def _partition_supervision_counts(
    assignments: Sequence[Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    result = {
        task_type: {
            partition: {"primary_positive": 0, "primary_negative": 0}
            for partition in SPLIT_PARTITIONS
        }
        for task_type in ("fall_event", "near_fall_event")
    }
    for assignment in assignments:
        label = labels_by_id.get(str(assignment.get("label_id")))
        if label is None or label.get("training_tier") != "primary":
            continue
        task_type = label.get("task_type")
        role = label.get("label_role")
        partition = assignment.get("partition")
        if task_type in result and partition in SPLIT_PARTITIONS and role in {
            "positive",
            "negative",
        }:
            result[str(task_type)][str(partition)][f"primary_{role}"] += 1
    return result


def _partition_action_type_counts(
    assignments: Sequence[Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for assignment in assignments:
        if assignment.get("label_kind") != "action":
            continue
        label = labels_by_id.get(str(assignment.get("label_id")))
        if label is None or label.get("action_type_training_tier") != "primary":
            continue
        action_type = label.get("action_type")
        partition = assignment.get("partition")
        if isinstance(action_type, str) and partition in SPLIT_PARTITIONS:
            counts[action_type][str(partition)] += 1
    return {
        action_type: {partition: int(values[partition]) for partition in SPLIT_PARTITIONS}
        for action_type, values in sorted(counts.items())
    }


def _split_leakage_issues(
    assignments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for field in (
        "asset_id",
        "video_id",
        "content_sha256",
        "subject_id",
        "source_group_id",
        "sample_group_id",
        "physical_event_id",
        "split_group_id",
    ):
        partitions_by_value: dict[str, set[str]] = defaultdict(set)
        for row in assignments:
            value = row.get(field)
            if not _known_split_identifier(value):
                continue
            partitions_by_value[str(value).strip()].add(str(row.get("partition")))
        for value, partitions in sorted(partitions_by_value.items()):
            if len(partitions) > 1:
                issues.append(
                    {
                        "field": field,
                        "value": value,
                        "partitions": sorted(partitions),
                    }
                )
    return issues


class _UnionFind:
    def __init__(self, items: Mapping[str, Any]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self._parent[second] = first


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _frame_time(frame: int, fps_num: int, fps_den: int) -> float:
    return round(frame * fps_den / fps_num, 9)


def _inclusive_intervals_overlap(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return max(_required_int(left, "start_frame"), _required_int(right, "start_frame")) <= min(
        _required_int(left, "end_frame"), _required_int(right, "end_frame")
    )


def _label_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("video_id", "")),
        int(row.get("start_frame", 0)),
        int(row.get("end_frame_exclusive", 0)),
        str(row.get("task_type", "")),
        str(row.get("label_id", "")),
    )


def _unique_source_refs(refs: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for ref in refs:
        key = (str(ref["source_type"]), str(ref["source_label_id"]))
        result[key] = dict(ref)
    return [result[key] for key in sorted(result)]


def _close_number(actual: Any, expected: float, tolerance: float = 0.001) -> bool:
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=tolerance
    )


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    record_kind: str,
    label_id: Any,
    detail: Any | None = None,
) -> None:
    issue = {
        "code": code,
        "record_kind": record_kind,
        "label_id": label_id,
    }
    if detail is not None:
        issue["detail"] = detail
    issues.append(issue)


def _ensure_outputs_available(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output exists; pass overwrite=True: " + ", ".join(str(path) for path in existing)
        )


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary_path, path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def _read_json_strict(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
