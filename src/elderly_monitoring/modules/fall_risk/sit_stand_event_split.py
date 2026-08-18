from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from elderly_monitoring.modules.fall_risk.sit_stand_development_gate import (
    DIRECTION_MINIMUMS,
    HARD_NEGATIVE_MIN_GROUPS,
    MAX_VALIDATION_NTU_EVENT_SHARE,
    MIN_SOURCE_GROUPS,
    evaluate_development_localization_gate,
)


PARTITIONS = {"train", "validation", "test"}
PROTECTION_FIELDS = (
    "subject_id",
    "source_group_id",
    "sample_group_id",
    "split_group_id",
    "video_id",
    "event_id",
)


def build_sit_stand_event_split(
    labels: Iterable[Mapping[str, Any]],
    source_assignments: Iterable[Mapping[str, Any]],
    manifest: Iterable[Mapping[str, Any]],
    *,
    materializable_label_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic task-specific development split without pose access."""

    rows = [dict(row) for row in labels]
    rows.sort(key=lambda row: str(row.get("label_id", "")))
    if not rows:
        raise ValueError("sit-stand event split requires at least one reviewed label")
    source_index = {
        str(row["label_id"]): dict(row)
        for row in source_assignments
        if isinstance(row.get("label_id"), str)
    }
    manifest_index = {
        str(row["video_id"]): dict(row)
        for row in manifest
        if isinstance(row.get("video_id"), str)
    }
    label_ids = [str(row.get("label_id", "")) for row in rows]
    if any(not value for value in label_ids) or len(label_ids) != len(set(label_ids)):
        raise ValueError("sit-stand labels require unique non-empty label_id values")
    if materializable_label_ids is not None:
        unknown = sorted(materializable_label_ids - set(label_ids))
        if unknown:
            raise ValueError(
                "materializable label ids are absent from sit-stand labels: "
                + ", ".join(unknown[:5])
            )
        if not materializable_label_ids:
            raise ValueError("materializable label ids must not be empty")

    parent = {label_id: label_id for label_id in label_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    protection_index: dict[tuple[str, str], str] = {}
    source_action_index: dict[str, str] = {}
    for row in rows:
        label_id = str(row["label_id"])
        for field in PROTECTION_FIELDS:
            value = row.get(field)
            if value in (None, "", "unknown"):
                continue
            key = (field, str(value))
            if key in protection_index:
                union(label_id, protection_index[key])
            else:
                protection_index[key] = label_id
        source_action_ids = row.get("source_action_label_ids", [])
        if not isinstance(source_action_ids, list):
            raise ValueError(f"{label_id}.source_action_label_ids must be a list")
        for action_id in source_action_ids:
            action_key = str(action_id)
            if action_key in source_action_index:
                union(label_id, source_action_index[action_key])
            else:
                source_action_index[action_key] = label_id

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[find(str(row["label_id"]))].append(row)
    protected_groups: list[dict[str, Any]] = []
    locked_test_label_ids: set[str] = set()
    locked_test_label_count = 0
    locked_test_group_count = 0
    for _, group_rows in sorted(groups.items()):
        inherited: set[str] = set()
        for row in group_rows:
            for action_id in row.get("source_action_label_ids", []):
                source = source_index.get(str(action_id))
                if source is None:
                    raise ValueError(f"missing source split assignment for {action_id}")
                partition = str(source.get("partition", ""))
                if partition not in PARTITIONS:
                    raise ValueError(f"invalid inherited partition for {action_id}")
                inherited.add(partition)
        group_material = "|".join(sorted(str(row["label_id"]) for row in group_rows))
        split_group_id = "sitstandgrp_" + hashlib.sha256(
            group_material.encode("utf-8")
        ).hexdigest()[:24]
        if "test" in inherited:
            locked_test_label_count += len(group_rows)
            locked_test_group_count += 1
            locked_test_label_ids.update(str(row["label_id"]) for row in group_rows)
            continue
        protected_groups.append(
            _summarize_group(
                split_group_id,
                group_rows,
                manifest_index,
                materializable_label_ids=materializable_label_ids,
            )
        )

    validation_ids = _select_validation_groups(protected_groups)
    assignments: list[dict[str, Any]] = []
    group_partitions: dict[str, str] = {}
    for group in sorted(protected_groups, key=lambda item: str(item["group_id"])):
        partition = "train" if group["train_only"] else (
            "validation" if str(group["group_id"]) in validation_ids else "train"
        )
        group_partitions[str(group["group_id"])] = partition
        for row in group["rows"]:
            assignments.append(
                {
                    "schema_version": "sit-stand-event-split-assignment-v1",
                    "label_id": str(row["label_id"]),
                    "event_id": str(row.get("event_id", "")),
                    "video_id": str(row.get("video_id", "")),
                    "split_group_id": str(group["group_id"]),
                    "partition": partition,
                    "features_materialized": partition in {"train", "validation"},
                    "partition_policy": str(
                        row.get("partition_policy", "rebalanced_development")
                    ),
                }
            )
    assignments.sort(key=lambda row: str(row["label_id"]))
    gate_rows = [
        row
        for row in rows
        if str(row["label_id"]) not in locked_test_label_ids
        and (
            materializable_label_ids is None
            or str(row["label_id"]) in materializable_label_ids
        )
    ]
    development_gate = evaluate_development_localization_gate(
        gate_rows,
        assignment_index={str(row["label_id"]): row for row in assignments},
        manifest_index=manifest_index,
    )
    if not development_gate["passed"]:
        raise ValueError(
            "sit-stand development split cannot satisfy localization gate: "
            + json.dumps(development_gate, sort_keys=True)
        )
    assignment_hash = _canonical_hash(assignments)
    split_id = "sitstandsplit_" + hashlib.sha256(
        assignment_hash.encode("ascii")
    ).hexdigest()[:24]
    counts = Counter(str(row["partition"]) for row in assignments)
    group_counts = Counter(group_partitions.values())
    if locked_test_group_count:
        group_counts["test"] = locked_test_group_count
    return {
        "assignments": assignments,
        "split": {
            "schema_version": "sit-stand-event-split-v1",
            "task": "sit_stand_event_localization_v1",
            "status": "development_provisional",
            "split_id": split_id,
            "assignments_sha256": assignment_hash,
            "assignment_counts": dict(sorted(counts.items())),
            "protection_group_counts": dict(sorted(group_counts.items())),
            "locked_test_label_count": locked_test_label_count,
            "locked_test_protection_group_count": locked_test_group_count,
            "train_only_label_count": sum(
                len(group["rows"]) for group in protected_groups if group["train_only"]
            ),
            "train_only_protection_group_count": sum(
                bool(group["train_only"]) for group in protected_groups
            ),
            "protection_fields": list(PROTECTION_FIELDS)
            + ["source_action_label_ids", "adjacent_intervals_same_video"],
            "leakage_issues": [],
            "development_gate": development_gate,
            "development_rebalanced_from_source_split": True,
            "materialization_aware": materializable_label_ids is not None,
            "materializable_label_count": (
                len(materializable_label_ids)
                if materializable_label_ids is not None
                else None
            ),
            "materializable_labels_sha256": (
                _canonical_hash([{"label_id": value} for value in sorted(materializable_label_ids)])
                if materializable_label_ids is not None
                else None
            ),
            "test_access": {
                "test_pose_read": False,
                "test_features_generated": False,
                "test_evaluated": False,
            },
        },
    }


def _summarize_group(
    group_id: str,
    rows: list[dict[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    *,
    materializable_label_ids: set[str] | None,
) -> dict[str, Any]:
    directions: Counter[str] = Counter()
    event_datasets: Counter[str] = Counter()
    hard_negatives: set[str] = set()
    source_groups: set[str] = set()
    background_seconds = 0.0
    train_only = False
    for row in rows:
        train_only = train_only or row.get("partition_policy") == "train_only"
        if (
            materializable_label_ids is not None
            and str(row["label_id"]) not in materializable_label_ids
        ):
            continue
        if row.get("eligibility") == "ignore":
            continue
        source_groups.add(str(row.get("source_group_id")))
        if row.get("interval_type") == "event":
            directions[str(row.get("transition_type"))] += 1
            dataset = str(
                manifest_index.get(str(row.get("video_id")), {}).get(
                    "dataset", "missing"
                )
            )
            event_datasets[dataset] += 1
        elif row.get("interval_type") == "explicit_background":
            background_seconds += float(row["offset_time"]) - float(row["onset_time"])
            if isinstance(row.get("hard_negative_type"), str):
                hard_negatives.add(str(row["hard_negative_type"]))
    return {
        "group_id": group_id,
        "rows": sorted(rows, key=lambda row: str(row["label_id"])),
        "directions": directions,
        "event_datasets": event_datasets,
        "hard_negatives": hard_negatives,
        "source_groups": source_groups,
        "background_seconds": background_seconds,
        "train_only": train_only,
    }


def _select_validation_groups(groups: list[dict[str, Any]]) -> set[str]:
    selected: set[str] = set()
    totals: Counter[str] = Counter()
    for group in groups:
        totals.update(group["directions"])

    def can_select(group: Mapping[str, Any]) -> bool:
        if group.get("train_only") is True:
            return False
        return all(
            totals[direction]
            - sum(
                int(candidate["directions"][direction])
                for candidate in groups
                if candidate["group_id"] in selected
            )
            - int(group["directions"][direction])
            >= minimum
            for direction, minimum in DIRECTION_MINIMUMS["train"].items()
        )

    def add(group: Mapping[str, Any]) -> bool:
        group_id = str(group["group_id"])
        if group_id in selected or not can_select(group):
            return False
        selected.add(group_id)
        return True

    stable = sorted(groups, key=lambda group: _group_rank(str(group["group_id"])))
    for category, minimum in HARD_NEGATIVE_MIN_GROUPS.items():
        candidates = sorted(
            (group for group in stable if category in group["hard_negatives"]),
            key=lambda group: (
                bool(group["event_datasets"].get("ntu_rgbd", 0)),
                sum(group["directions"].values()),
                _group_rank(str(group["group_id"])),
            ),
        )
        while sum(category in group["hard_negatives"] for group in groups if group["group_id"] in selected) < minimum:
            if not candidates:
                break
            add(candidates.pop(0))

    for direction, minimum in DIRECTION_MINIMUMS["validation"].items():
        candidates = sorted(
            (group for group in stable if group["directions"][direction] > 0),
            key=lambda group: (
                bool(group["event_datasets"].get("ntu_rgbd", 0)),
                -int(group["directions"][direction]),
                sum(group["directions"].values()),
                _group_rank(str(group["group_id"])),
            ),
        )
        while sum(
            int(group["directions"][direction])
            for group in groups
            if group["group_id"] in selected
        ) < minimum:
            if not candidates:
                break
            add(candidates.pop(0))

    def validation_sources() -> set[str]:
        return set().union(
            *(
                group["source_groups"]
                for group in groups
                if group["group_id"] in selected
            )
        ) if selected else set()

    for group in stable:
        if len(validation_sources()) >= MIN_SOURCE_GROUPS:
            break
        add(group)

    while True:
        validation_events = sum(
            sum(group["directions"].values())
            for group in groups
            if group["group_id"] in selected
        )
        ntu_events = sum(
            int(group["event_datasets"].get("ntu_rgbd", 0))
            for group in groups
            if group["group_id"] in selected
        )
        share = ntu_events / validation_events if validation_events else 1.0
        if share <= MAX_VALIDATION_NTU_EVENT_SHARE:
            break
        candidates = sorted(
            (
                group
                for group in stable
                if group["group_id"] not in selected
                and sum(group["directions"].values()) > 0
                and not group["event_datasets"].get("ntu_rgbd", 0)
            ),
            key=lambda group: (
                -sum(group["directions"].values()),
                _group_rank(str(group["group_id"])),
            ),
        )
        if not candidates or not add(candidates[0]):
            break
    return selected


def _group_rank(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()


def _canonical_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
