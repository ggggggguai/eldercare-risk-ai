from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from elderly_monitoring.modules.fall_risk.sit_stand_event_labels import (
    validate_sit_stand_event_labels,
)
from elderly_monitoring.modules.fall_risk.sit_stand_development_gate import (
    evaluate_development_localization_gate,
)


STATUS_VALUES = {
    "infrastructure_only",
    "development_provisional",
    "candidate_frozen",
    "test_released",
}
SIT_STAND_ACTION_IDS = {"A03", "A04", "B06", "C01"}
DEVELOPMENT_DIRECTION_MINIMUMS = {
    "train": {"sit_to_stand": 40, "stand_to_sit": 40},
    "validation": {"sit_to_stand": 20, "stand_to_sit": 20},
}
DEVELOPMENT_HARD_NEGATIVE_MIN_GROUPS = {
    "controlled_bend": 10,
    "controlled_squat": 10,
    "fall_or_rapid_descent": 10,
    "bed_transfer_or_lying": 10,
}


def build_sit_stand_training_audit(
    *,
    action_labels: Path,
    manifest: Path,
    assignments: Path,
    split: Path,
    validation: Path,
    training_config: Path,
    evaluation_config: Path,
    historical_report: Path,
    event_labels: Path | None = None,
    review_log: Path | None = None,
    event_assignments: Path | None = None,
    event_split: Path | None = None,
) -> dict[str, Any]:
    """Build the P0 sit-stand audit without opening test pose or truth."""

    required = {
        "action_labels": action_labels,
        "manifest": manifest,
        "assignments": assignments,
        "split": split,
        "v3_validation": validation,
        "training_config": training_config,
        "evaluation_config": evaluation_config,
        "historical_report": historical_report,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing sit-stand audit input {name}: {path}")
    hashes = {name: _sha256(path) for name, path in required.items()}
    if event_labels is not None and event_labels.is_file():
        hashes["event_labels"] = _sha256(event_labels)
    if review_log is not None and review_log.is_file():
        hashes["review_log"] = _sha256(review_log)

    actions = _read_jsonl(action_labels)
    manifests = _read_jsonl(manifest)
    assignment_rows = _read_jsonl(assignments)
    split_report = _read_json(split)
    validation_report = _read_json(validation)
    training = _read_yaml(training_config)
    evaluation = _read_yaml(evaluation_config)
    assignment_index = {
        str(row.get("label_id")): row
        for row in assignment_rows
        if isinstance(row.get("label_id"), str)
    }
    localization_assignment_index = assignment_index
    localization_split_id = split_id = split_report.get("split_id")
    event_split_report: dict[str, Any] | None = None
    if event_assignments is not None and event_assignments.is_file():
        localization_assignment_rows = _read_jsonl(event_assignments)
        localization_assignment_index = {
            str(row.get("label_id")): row
            for row in localization_assignment_rows
            if isinstance(row.get("label_id"), str)
        }
        hashes["event_assignments"] = _sha256(event_assignments)
    if event_split is not None and event_split.is_file():
        event_split_report = _read_json(event_split)
        localization_split_id = event_split_report.get("split_id")
        hashes["event_split"] = _sha256(event_split)
    manifest_index = {
        str(row.get("video_id")): row
        for row in manifests
        if isinstance(row.get("video_id"), str)
    }
    selected = [row for row in actions if row.get("action_id") in SIT_STAND_ACTION_IDS]
    action_counts = _action_counts(selected, assignment_index, manifest_index)
    leakage = _leakage(selected, assignment_index)

    event_validation: dict[str, Any] | None = None
    event_rows: list[dict[str, Any]] = []
    event_label_state = "not_created_pending_human_review"
    if event_labels is not None and event_labels.is_file():
        if review_log is None or not review_log.is_file():
            raise ValueError("sit-stand event labels exist without a review log")
        event_rows = _read_jsonl(event_labels)
        event_validation = validate_sit_stand_event_labels(
            event_rows, _read_jsonl(review_log)
        )
        event_label_state = "reviewed_labels_present"

    hash_mismatches = _hash_mismatches(
        hashes, split_report=split_report, validation_report=validation_report
    )
    validation_split_id = (
        validation_report.get("split", {}).get("split_id")
        if isinstance(validation_report.get("split"), Mapping)
        else None
    )
    if validation_split_id is not None and validation_split_id != split_id:
        hash_mismatches["split_id"] = {
            "actual": split_id,
            "validation": validation_split_id,
        }
    split_status = str(split_report.get("status") or "provisional_unfrozen")
    explicit_frozen_approval = (
        split_status == "frozen"
        and isinstance(split_report.get("approved_by"), str)
        and bool(str(split_report.get("approved_by")).strip())
    )
    background_seconds = (
        float(event_validation["explicit_background_duration_sec"])
        if event_validation is not None
        else 0.0
    )
    event_count = (
        int(event_validation["counts"]["event"])
        if event_validation is not None
        else 0
    )
    localization_gate = evaluate_development_localization_gate(
        event_rows,
        assignment_index=localization_assignment_index,
        manifest_index=manifest_index,
    )
    localization_ready = bool(localization_gate["passed"])
    action_proxy_ready = bool(
        validation_report.get("training_ready", {}).get("action_type", False)
    )
    event_review_counts = (
        event_validation.get("review_status_counts", {})
        if event_validation is not None
        else {}
    )
    phase_ready = False
    functional_ready = False
    frozen_protocol = evaluation.get("protocol_status") == "frozen"
    status = "infrastructure_only"
    if localization_ready and not leakage and not hash_mismatches:
        status = "development_provisional"
    if status == "development_provisional" and explicit_frozen_approval and frozen_protocol:
        status = "candidate_frozen"
    if status == "candidate_frozen" and split_report.get("test_release_id"):
        status = "test_released"
    if status not in STATUS_VALUES:
        raise AssertionError(f"invalid sit-stand audit status: {status}")

    historical = _historical_artifacts(historical_report)
    gates = {
        "hash_bindings": {
            "passed": not hash_mismatches,
            "mismatches": hash_mismatches,
        },
        "split_leakage": {"passed": not leakage, "issues": leakage},
        "frozen_approval": {
            "passed": explicit_frozen_approval,
            "reported_status": split_status,
        },
        "event_localization": {
            "event_count": event_count,
            "explicit_background_duration_sec": background_seconds,
            **localization_gate,
        },
        "phase_segmentation": {"passed": phase_ready},
        "action_proxy": {"passed": action_proxy_ready},
        "functional_proxy": {"passed": functional_ready},
    }
    material = json.dumps(
        {"hashes": hashes, "split_id": split_id, "status": status},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": "sit-stand-training-audit-v2",
        "manifest_id": "sit_stand_training_manifest_"
        + hashlib.sha256(material).hexdigest()[:24],
        "status": status,
        "task": "sit_stand_event_localization_v1",
        "source_split_id": split_id,
        "input_sha256": hashes,
        "facts": {
            "event_label_state": event_label_state,
            "confirmed_event_count": event_count,
            "explicit_background_duration_sec": background_seconds,
            "explicit_background_camera_hours": round(background_seconds / 3600.0, 6),
            "continuous_video_count": 0,
            "direction_counts": (
                event_validation.get("direction_counts", {})
                if event_validation is not None
                else {}
            ),
            "outcome_counts": (
                event_validation.get("outcome_counts", {})
                if event_validation is not None
                else {}
            ),
            "review_status_counts": event_review_counts,
            "hard_negative_counts": {},
            "development_localization_coverage": localization_gate,
            "action_counts": action_counts,
            "training_config_status": training.get("current_status"),
            "evaluation_protocol_status": evaluation.get("protocol_status"),
            "historical_artifacts": historical,
            "event_split_id": localization_split_id,
        },
        "gates": gates,
        "test_access": {
            "test_pose_read": False,
            "test_features_generated": False,
            "test_evaluated": False,
        },
        "authorization": {
            "schema_validator_queue_and_synthetic_infrastructure_allowed": True,
            "continuous_model_training_allowed": localization_ready,
            "formal_test_allowed": status in {"candidate_frozen", "test_released"},
        },
    }


def _development_localization_gate(
    rows: list[dict[str, Any]],
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
        source_ids = row.get("source_action_label_ids", [])
        partitions = {
            str(assignment_index.get(str(source_id), {}).get("partition", "missing"))
            for source_id in source_ids
        }
        if len(partitions) != 1 or next(iter(partitions), "missing") not in {
            "train",
            "validation",
        }:
            assignment_issues.append(
                {
                    "label_id": row.get("label_id"),
                    "partitions": sorted(partitions),
                }
            )
            continue
        partition = next(iter(partitions))
        if row.get("interval_type") == "event":
            direction_counts[partition][str(row.get("transition_type"))] += 1
            source_groups[partition].add(str(row.get("source_group_id")))
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
                    str(row.get("split_group_id"))
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
        for partition, requirements in DEVELOPMENT_DIRECTION_MINIMUMS.items()
    }
    source_group_requirements = {
        partition: {
            "count": len(source_groups[partition]),
            "minimum": 3,
            "passed": len(source_groups[partition]) >= 3,
        }
        for partition in ("train", "validation")
    }
    hard_negative_requirements = {
        category: {
            "protection_group_count": len(hard_negative_groups[category]),
            "minimum": minimum,
            "passed": len(hard_negative_groups[category]) >= minimum,
        }
        for category, minimum in DEVELOPMENT_HARD_NEGATIVE_MIN_GROUPS.items()
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
        "maximum_validation_ntu_share": 0.5,
        "passed": validation_ntu_share is not None and validation_ntu_share <= 0.5,
    }
    background_requirement = {
        partition: {
            "duration_sec": round(float(background_seconds[partition]), 6),
            "passed": background_seconds[partition] > 0,
        }
        for partition in ("train", "validation")
    }
    checks = [
        *(item["passed"] for values in direction_requirements.values() for item in values.values()),
        *(item["passed"] for item in source_group_requirements.values()),
        *(item["passed"] for item in hard_negative_requirements.values()),
        dataset_requirement["passed"],
        *(item["passed"] for item in background_requirement.values()),
        not assignment_issues,
    ]
    return {
        "passed": all(checks),
        "direction_requirements": direction_requirements,
        "source_group_requirements": source_group_requirements,
        "hard_negative_requirements": hard_negative_requirements,
        "dataset_requirement": dataset_requirement,
        "background_requirements": background_requirement,
        "assignment_issues": assignment_issues,
    }


def write_sit_stand_training_audit(
    report: Mapping[str, Any],
    *,
    manifest_output: Path,
    audit_output: Path,
    blockers_output: Path,
) -> None:
    outputs = (manifest_output, audit_output, blockers_output)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "sit-stand audit output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_output.write_text(_render_audit(report), encoding="utf-8")
    blockers_output.write_text(_render_blockers(report), encoding="utf-8")


def _action_counts(
    rows: list[dict[str, Any]],
    assignments: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for action_id in sorted(SIT_STAND_ACTION_IDS):
        selected = [row for row in rows if row.get("action_id") == action_id]
        tiers = Counter(str(row.get("action_type_training_tier", "unknown")) for row in selected)
        partitions = Counter(
            str(assignments.get(str(row.get("label_id")), {}).get("partition", "missing"))
            for row in selected
        )
        datasets = Counter(
            str(manifest.get(str(row.get("video_id")), {}).get("dataset", "missing"))
            for row in selected
        )
        tier_partition: dict[str, Counter[str]] = defaultdict(Counter)
        for row in selected:
            tier = str(row.get("action_type_training_tier", "unknown"))
            partition = str(
                assignments.get(str(row.get("label_id")), {}).get(
                    "partition", "missing"
                )
            )
            tier_partition[tier][partition] += 1
        result[action_id] = {
            "total": len(selected),
            "tier_counts": dict(sorted(tiers.items())),
            "partition_counts": dict(sorted(partitions.items())),
            "dataset_counts": dict(sorted(datasets.items())),
            "tier_partition_counts": {
                tier: dict(sorted(counts.items()))
                for tier, counts in sorted(tier_partition.items())
            },
            "subject_count": len({str(row.get("subject_id")) for row in selected}),
            "source_group_count": len(
                {str(row.get("source_group_id")) for row in selected}
            ),
            "sample_group_count": len(
                {str(row.get("sample_group_id")) for row in selected}
            ),
            "split_group_count": len(
                {
                    str(assignments.get(str(row.get("label_id")), {}).get("split_group_id"))
                    for row in selected
                }
            ),
        }
    return result


def _leakage(
    rows: list[dict[str, Any]], assignments: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for field in ("subject_id", "source_group_id", "sample_group_id"):
        partitions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.get(field)
            assignment = assignments.get(str(row.get("label_id")), {})
            partition = assignment.get("partition")
            if value not in (None, "", "unknown") and isinstance(partition, str):
                partitions[str(value)].add(partition)
        for value, found in sorted(partitions.items()):
            if len(found) > 1:
                issues.append(
                    {"field": field, "value": value, "partitions": sorted(found)}
                )
    return issues


def _hash_mismatches(
    hashes: Mapping[str, str],
    *,
    split_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> dict[str, Any]:
    split_inputs = split_report.get("input_sha256", {})
    validation_inputs = validation_report.get("input_sha256", {})
    expectations = {
        "action_labels": [
            split_inputs.get("action_labels"),
            validation_inputs.get("action_labels"),
        ],
        "manifest": [
            split_inputs.get("manifest"),
            validation_inputs.get("manifest"),
        ],
        "assignments": [
            split_report.get("assignments_sha256"),
            validation_inputs.get("split_assignments"),
        ],
        "split": [validation_inputs.get("split_report")],
    }
    mismatches: dict[str, Any] = {}
    for actual_name, expected_values in expectations.items():
        expected = [value for value in expected_values if isinstance(value, str)]
        actual = hashes[actual_name]
        if not expected or any(value != actual for value in expected):
            mismatches[actual_name] = {"actual": actual, "expected": expected}
    return mismatches


def _historical_artifacts(report: Path) -> dict[str, Any]:
    text = report.read_text(encoding="utf-8")
    candidates = []
    for token in text.replace("`", " ").split():
        stripped = token.strip(".,:;()[]{}")
        if stripped.endswith((".npz", ".pt", ".joblib")):
            candidates.append(stripped)
    return {
        "referenced_paths": sorted(set(candidates)),
        "existing_paths": sorted(path for path in set(candidates) if Path(path).is_file()),
        "metadata_hash_match_verified": False,
    }


def _render_audit(report: Mapping[str, Any]) -> str:
    facts = report["facts"]
    return (
        "# Sit-stand training P0 audit\n\n"
        f"- Status: `{report['status']}`\n"
        f"- Source split: `{report['source_split_id']}`\n"
        f"- Confirmed events: {facts['confirmed_event_count']}\n"
        f"- Explicit background: {facts['explicit_background_camera_hours']} camera-hour\n"
        "- Test pose/features/evaluation: `false/false/false`\n\n"
        "This audit is fail closed and does not treat action clip boundaries as event truth.\n"
    )


def _render_blockers(report: Mapping[str, Any]) -> str:
    lines = ["# Sit-stand training blockers", ""]
    for name, gate in report["gates"].items():
        if gate.get("passed") is not True:
            lines.append(f"- `{name}`: blocked")
    lines.extend(
        [
            "",
            "Continuous labels, explicit background, frozen protocol, and custodian-controlled test release remain human-governed gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
