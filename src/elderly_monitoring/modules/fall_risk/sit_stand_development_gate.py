from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


DIRECTION_MINIMUMS = {
    "train": {"sit_to_stand": 40, "stand_to_sit": 40},
    "validation": {"sit_to_stand": 20, "stand_to_sit": 20},
}
HARD_NEGATIVE_MIN_GROUPS = {
    "controlled_bend": 10,
    "controlled_squat": 10,
    "fall_or_rapid_descent": 10,
    "bed_transfer_or_lying": 10,
}
MIN_SOURCE_GROUPS = 3
MAX_VALIDATION_NTU_EVENT_SHARE = 0.5


def evaluate_development_localization_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    assignment_index: Mapping[str, Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    direction_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_groups: dict[str, set[str]] = defaultdict(set)
    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    background_seconds: Counter[str] = Counter()
    hard_negative_groups: dict[str, set[str]] = defaultdict(set)
    assignment_issues: list[dict[str, Any]] = []

    for row in rows:
        if row.get("eligibility") == "ignore":
            continue
        assignment = assignment_index.get(str(row.get("label_id")))
        if assignment is None:
            source_ids = row.get("source_action_label_ids", [])
            inherited = {
                str(assignment_index.get(str(source_id), {}).get("partition", "missing"))
                for source_id in source_ids
            }
            partition = next(iter(inherited)) if len(inherited) == 1 else "missing"
            assigned_group = str(row.get("split_group_id", "missing"))
        else:
            partition = str(assignment.get("partition", "missing"))
            assigned_group = str(assignment.get("split_group_id", "missing"))
        if partition not in {"train", "validation"}:
            assignment_issues.append(
                {"label_id": row.get("label_id"), "partition": partition}
            )
            continue

        source_groups[partition].add(str(row.get("source_group_id")))
        if row.get("interval_type") == "event":
            direction_counts[partition][str(row.get("transition_type"))] += 1
            dataset = str(
                manifest_index.get(str(row.get("video_id")), {}).get(
                    "dataset", "missing"
                )
            )
            dataset_counts[partition][dataset] += 1
        elif row.get("interval_type") == "explicit_background":
            background_seconds[partition] += float(row["offset_time"]) - float(
                row["onset_time"]
            )
            if partition == "validation" and isinstance(
                row.get("hard_negative_type"), str
            ):
                hard_negative_groups[str(row["hard_negative_type"])].add(
                    assigned_group
                )

    direction_requirements = {
        partition: {
            direction: {
                "count": int(direction_counts[partition][direction]),
                "minimum": minimum,
                "passed": direction_counts[partition][direction] >= minimum,
            }
            for direction, minimum in requirements.items()
        }
        for partition, requirements in DIRECTION_MINIMUMS.items()
    }
    source_group_requirements = {
        partition: {
            "count": len(source_groups[partition]),
            "minimum": MIN_SOURCE_GROUPS,
            "passed": len(source_groups[partition]) >= MIN_SOURCE_GROUPS,
        }
        for partition in ("train", "validation")
    }
    hard_negative_requirements = {
        category: {
            "protection_group_count": len(hard_negative_groups[category]),
            "minimum": minimum,
            "passed": len(hard_negative_groups[category]) >= minimum,
        }
        for category, minimum in HARD_NEGATIVE_MIN_GROUPS.items()
    }
    validation_event_total = sum(direction_counts["validation"].values())
    validation_ntu_share = (
        dataset_counts["validation"]["ntu_rgbd"] / validation_event_total
        if validation_event_total
        else None
    )
    dataset_requirement = {
        "validation_event_dataset_counts": dict(
            sorted(dataset_counts["validation"].items())
        ),
        "validation_ntu_share": (
            round(validation_ntu_share, 6)
            if validation_ntu_share is not None
            else None
        ),
        "maximum_validation_ntu_share": MAX_VALIDATION_NTU_EVENT_SHARE,
        "passed": validation_ntu_share is not None
        and validation_ntu_share <= MAX_VALIDATION_NTU_EVENT_SHARE,
    }
    background_requirements = {
        partition: {
            "duration_sec": round(float(background_seconds[partition]), 6),
            "passed": background_seconds[partition] > 0,
        }
        for partition in ("train", "validation")
    }
    checks = [
        *(
            item["passed"]
            for values in direction_requirements.values()
            for item in values.values()
        ),
        *(item["passed"] for item in source_group_requirements.values()),
        *(item["passed"] for item in hard_negative_requirements.values()),
        dataset_requirement["passed"],
        *(item["passed"] for item in background_requirements.values()),
        not assignment_issues,
    ]
    return {
        "passed": all(checks),
        "direction_requirements": direction_requirements,
        "source_group_requirements": source_group_requirements,
        "hard_negative_requirements": hard_negative_requirements,
        "dataset_requirement": dataset_requirement,
        "background_requirements": background_requirements,
        "assignment_issues": assignment_issues,
    }
