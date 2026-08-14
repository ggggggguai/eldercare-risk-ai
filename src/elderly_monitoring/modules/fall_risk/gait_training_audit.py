from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from elderly_monitoring.modules.fall_risk.gait_training import _resolve_pose_path


AUDIT_SCHEMA_VERSION = "gait-training-audit-v1"
GAIT_ACTION_IDS = tuple(f"A{index:02d}" for index in range(1, 13)) + (
    "B01",
    "B02",
    "B03",
    "B04",
)
POSITIVE_ACTION_IDS = ("B02", "B03", "B04")
REQUIRED_RELEASE_FIELDS = (
    "release_id",
    "approved_by",
    "released_at",
    "split_id",
    "labels_sha256",
    "manifest_sha256",
    "assignments_sha256",
    "evaluation_config_sha256",
    "checkpoint_sha256",
    "candidate_manifest_sha256",
)


def build_gait_training_audit(
    *,
    manifest: Path,
    action_labels: Path,
    action_schema: Path,
    assignments: Path,
    split: Path,
    validation: Path,
    training: Path,
    evaluation: Path,
    pose_dir: Path,
    dataset_metadata: Path | None = None,
    historical_reports: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Audit gait development inputs without opening sealed test pose files."""

    required = {
        "manifest": manifest,
        "action_labels": action_labels,
        "action_schema": action_schema,
        "split_assignments": assignments,
        "split_report": split,
        "v3_validation": validation,
        "training_config": training,
        "evaluation_config": evaluation,
    }
    if dataset_metadata is not None:
        required["dataset_metadata"] = dataset_metadata
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"missing pose directory: {pose_dir}")

    hashes = {name: _sha256(path) for name, path in required.items()}
    split_report = _read_json(split)
    validation_report = _read_json(validation)
    training_config = _read_yaml(training)
    evaluation_config = _read_yaml(evaluation)
    labels = _read_jsonl(action_labels)
    manifest_rows = _read_jsonl(manifest)
    assignment_rows = _read_jsonl(assignments)

    hash_mismatches = _hash_mismatches(hashes, split_report, validation_report)
    split_id = str(split_report.get("split_id", ""))
    validation_split_id = str(
        _mapping(validation_report.get("split")).get("split_id", "")
    )
    if not split_id or split_id != validation_split_id:
        hash_mismatches["split_id"] = {
            "actual": split_id,
            "expected_by_validation": validation_split_id,
        }

    assignments_by_label = {
        str(row.get("label_id")): row
        for row in assignment_rows
        if row.get("label_kind") == "action" and row.get("label_id")
    }
    manifest_by_video = {
        str(row.get("video_id")): row
        for row in manifest_rows
        if row.get("video_id") not in (None, "")
    }
    selected: list[dict[str, Any]] = []
    missing_assignments: list[str] = []
    for label in labels:
        action_id = str(label.get("action_id", ""))
        if action_id not in GAIT_ACTION_IDS or label.get("training_tier") == "ignore":
            continue
        label_id = str(label.get("label_id", ""))
        assignment = assignments_by_label.get(label_id)
        if assignment is None:
            missing_assignments.append(label_id)
            continue
        row = dict(label)
        row["partition"] = str(assignment.get("partition", ""))
        row["split_group_id"] = str(assignment.get("split_group_id", ""))
        selected.append(row)

    counts = _label_counts(selected)
    leakage_issues = split_report.get("leakage_issues")
    leakage_passed = isinstance(leakage_issues, list) and not leakage_issues
    basic_binary = _basic_binary_gate(selected)
    validation_scale = _validation_scale_gate(selected)
    pose_cache = _audit_pose_cache(
        selected,
        pose_dir=pose_dir,
        manifest_by_video=manifest_by_video,
    )

    dataset_facts: dict[str, Any] | None = None
    test_tensor_generated = False
    test_evaluated = False
    test_pose_read = False
    if dataset_metadata is not None:
        dataset_facts = _read_json(dataset_metadata)
        test_tensor_generated = bool(
            dataset_facts.get("test_tensor_generated")
            or _mapping(dataset_facts.get("window_partition_counts")).get("test", 0)
        )
        test_evaluated = bool(dataset_facts.get("test_evaluated"))
        test_pose_read = bool(dataset_facts.get("test_pose_read"))
        if dataset_facts.get("source_split_id") != split_id:
            hash_mismatches["dataset_source_split_id"] = {
                "actual": dataset_facts.get("source_split_id"),
                "expected": split_id,
            }

    historical = _historical_split_audit(historical_reports, split_id)
    test_isolation_passed = not (
        test_pose_read or test_tensor_generated or test_evaluated
    )
    infrastructure_passed = all(
        (
            not hash_mismatches,
            not missing_assignments,
            leakage_passed,
            basic_binary["passed"],
            pose_cache["development_inputs_usable"],
            test_isolation_passed,
        )
    )
    split_status = str(
        split_report.get("status") or split_report.get("split_status") or "provisional"
    )
    protocol_status = str(
        evaluation_config.get("protocol_status", "development_provisional")
    )
    training_ready = _mapping(validation_report.get("training_ready"))
    frozen_gates_passed = all(
        (
            infrastructure_passed,
            split_status == "frozen",
            protocol_status == "frozen",
            training_ready.get("action_type") is True,
            validation_scale["passed"],
        )
    )
    if not infrastructure_passed:
        status = "infrastructure_only"
    elif frozen_gates_passed:
        status = "candidate_frozen"
    else:
        status = "development_provisional"

    gates = {
        "hash_bindings": {
            "passed": not hash_mismatches,
            "mismatches": hash_mismatches,
        },
        "split_leakage": {
            "passed": leakage_passed,
            "issues": leakage_issues if isinstance(leakage_issues, list) else None,
        },
        "development_binary": basic_binary,
        "validation_scale": validation_scale,
        "pose_cache": {
            "passed": pose_cache["development_inputs_usable"],
            "missing": pose_cache["missing"],
            "parse_errors": pose_cache["parse_errors"],
        },
        "test_isolation": {
            "passed": test_isolation_passed,
            "test_pose_read": test_pose_read,
            "test_tensor_generated": test_tensor_generated,
            "test_evaluated": test_evaluated,
        },
        "frozen_split": {"passed": split_status == "frozen", "status": split_status},
        "frozen_evaluation_protocol": {
            "passed": protocol_status == "frozen",
            "status": protocol_status,
        },
        "action_type_supervision": {
            "passed": training_ready.get("action_type") is True,
        },
        "test_release": {"passed": False, "status": "custodian_required"},
    }
    manifest_material = json.dumps(
        {"hashes": hashes, "split_id": split_id, "status": status},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "manifest_id": "gait_training_manifest_"
        + hashlib.sha256(manifest_material).hexdigest()[:24],
        "status": status,
        "scope": {
            "task": "gait_instability_vs_normal_activity",
            "negative_action_ids": list(GAIT_ACTION_IDS[:12]),
            "positive_action_ids": list(POSITIVE_ACTION_IDS),
            "functional_proxy_action_ids": ["B01"],
            "runtime_main_path": "rule_fallback",
        },
        "facts": {
            "split_id": split_id,
            "split_status": split_status,
            "protocol_status": protocol_status,
            "training_config_status": training_config.get("current_status"),
            "v3_valid": validation_report.get("valid") is True,
            "training_ready": training_ready,
            "selected_label_count": len(selected),
            "missing_assignment_label_ids": sorted(missing_assignments),
        },
        "counts": counts,
        "pose_cache": pose_cache,
        "dataset": dataset_facts,
        "historical_artifacts": historical,
        "test_access": {
            "test_pose_read": test_pose_read,
            "test_tensor_generated": test_tensor_generated,
            "test_evaluated": test_evaluated,
            "policy": "custodian_release_required",
        },
        "gates": gates,
        "input_sha256": hashes,
        "references": [
            "AGENTS.md",
            "README.md",
            "docs/README.md",
            "docs/architecture/算法工程骨架.md",
            "docs/interfaces/算法事件输出接口.md",
            "docs/tasks/README.md",
            "docs/modules/fall_risk/README.md",
            "docs/modules/fall_risk/plans/跌倒风险算法研发计划.md",
            "docs/modules/fall_risk/plans/步态模型训练方案.md",
            "docs/modules/fall_risk/data/跌倒风险标签字典.md",
            "docs/modules/fall_risk/data/数据集标注规范.md",
            "data/annotations/fall_risk/README.md",
        ],
        "authorization": {
            "bounded_train_validation_development_allowed": infrastructure_passed,
            "formal_training_allowed": frozen_gates_passed,
            "test_evaluation_allowed": False,
            "runtime_replacement_allowed": False,
        },
    }


def write_gait_training_audit(
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
            "gait audit output already exists: "
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


def validate_gait_test_release(
    release_path: Path,
    *,
    split_path: Path,
    labels_path: Path,
    manifest_path: Path,
    assignments_path: Path,
    evaluation_config_path: Path,
    checkpoint_path: Path,
    candidate_manifest_path: Path,
) -> dict[str, Any]:
    """Validate independent-custodian authorization before any test input is opened."""

    for path in (
        release_path,
        split_path,
        labels_path,
        manifest_path,
        assignments_path,
        evaluation_config_path,
        checkpoint_path,
        candidate_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required gait test-release input is missing: {path}")
    release = _read_json(release_path)
    missing = [field for field in REQUIRED_RELEASE_FIELDS if not release.get(field)]
    if missing:
        raise ValueError("gait test release is missing fields: " + ", ".join(missing))
    split_report = _read_json(split_path)
    candidate = _read_json(candidate_manifest_path)
    if str(split_report.get("status") or split_report.get("split_status")) != "frozen":
        raise ValueError("gait test release requires a frozen split")
    evaluation_config = _read_yaml(evaluation_config_path)
    if evaluation_config.get("protocol_status") != "frozen":
        raise ValueError("gait test release requires a frozen evaluation protocol")
    if candidate.get("protocol_status") != "candidate_frozen":
        raise ValueError("gait test candidate must have protocol_status=candidate_frozen")
    expected = {
        "split_id": split_report.get("split_id"),
        "labels_sha256": _sha256(labels_path),
        "manifest_sha256": _sha256(manifest_path),
        "assignments_sha256": _sha256(assignments_path),
        "evaluation_config_sha256": _sha256(evaluation_config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "candidate_manifest_sha256": _sha256(candidate_manifest_path),
    }
    for field, actual in expected.items():
        if release.get(field) != actual:
            raise ValueError(f"gait test release {field} mismatch")
    if candidate.get("source_split_id") != expected["split_id"]:
        raise ValueError("gait candidate source_split_id mismatch")
    if candidate.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
        raise ValueError("gait candidate checkpoint_sha256 mismatch")
    return release


def _hash_mismatches(
    hashes: Mapping[str, str],
    split_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    split_inputs = _mapping(split_report.get("input_sha256"))
    validation_inputs = _mapping(validation_report.get("input_sha256"))
    expectations = {
        "action_schema": [validation_inputs.get("action_schema")],
        "manifest": [split_inputs.get("manifest"), validation_inputs.get("manifest")],
        "action_labels": [
            split_inputs.get("action_labels"),
            validation_inputs.get("action_labels"),
        ],
        "split_assignments": [
            split_report.get("assignments_sha256"),
            validation_inputs.get("split_assignments"),
        ],
        "split_report": [validation_inputs.get("split_report")],
    }
    mismatches: dict[str, dict[str, Any]] = {}
    for name, values in expectations.items():
        expected = [value for value in values if isinstance(value, str)]
        if not expected or any(value != hashes[name] for value in expected):
            mismatches[name] = {"actual": hashes[name], "expected": expected}
    return mismatches


def _label_counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    action: dict[str, dict[str, int]] = {action_id: {} for action_id in GAIT_ACTION_IDS}
    by_tier: Counter[str] = Counter()
    by_action_tier: Counter[str] = Counter()
    by_partition: Counter[str] = Counter()
    by_dataset: Counter[str] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        action_id = str(row.get("action_id"))
        partition = str(row.get("partition"))
        action[action_id][partition] = action[action_id].get(partition, 0) + 1
        by_tier[f"{action_id}:{row.get('training_tier')}"] += 1
        by_action_tier[f"{action_id}:{row.get('action_type_training_tier')}"] += 1
        by_partition[partition] += 1
        by_dataset[str(row.get("dataset", "unknown"))] += 1
        for field in ("source_group_id", "sample_group_id", "split_group_id"):
            value = str(row.get(field, ""))
            if value:
                groups[f"{partition}:{field}"].add(value)
    return {
        "action_id": action,
        "training_tier": dict(sorted(by_tier.items())),
        "action_type_training_tier": dict(sorted(by_action_tier.items())),
        "partition": dict(sorted(by_partition.items())),
        "dataset": dict(sorted(by_dataset.items())),
        "independent_groups": {
            key: len(value) for key, value in sorted(groups.items())
        },
    }


def _basic_binary_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    per_partition: dict[str, dict[str, int]] = {}
    for partition in ("train", "validation"):
        selected = [row for row in rows if row.get("partition") == partition]
        negative = sum(str(row.get("action_id")) in GAIT_ACTION_IDS[:12] for row in selected)
        positive = sum(str(row.get("action_id")) in POSITIVE_ACTION_IDS for row in selected)
        per_partition[partition] = {"negative": negative, "positive": positive}
    return {
        "passed": all(
            values["negative"] > 0 and values["positive"] > 0
            for values in per_partition.values()
        ),
        "partition_counts": per_partition,
    }


def _validation_scale_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    positive = [
        row
        for row in rows
        if row.get("partition") == "validation"
        and row.get("training_tier") == "primary"
        and row.get("action_id") in POSITIVE_ACTION_IDS
    ]
    subtype_counts = {
        action_id: len({str(row.get("label_id")) for row in positive if row.get("action_id") == action_id})
        for action_id in POSITIVE_ACTION_IDS
    }
    source_count = len({str(row.get("source_group_id")) for row in positive})
    segment_count = len({str(row.get("label_id")) for row in positive})
    return {
        "passed": segment_count >= 30
        and all(value >= 10 for value in subtype_counts.values())
        and source_count >= 3,
        "positive_segment_count": segment_count,
        "subtype_segment_counts": subtype_counts,
        "positive_source_group_count": source_count,
        "requirements": {
            "positive_segments": 30,
            "segments_per_subtype": 10,
            "source_groups": 3,
        },
    }


def _audit_pose_cache(
    rows: list[Mapping[str, Any]],
    *,
    pose_dir: Path,
    manifest_by_video: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relevant_by_video: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        relevant_by_video[str(row.get("video_id"))].add(str(row.get("partition")))
    result: dict[str, Any] = {
        "eligible_manifest_video_count": sum(
            row.get("eligibility") is True and row.get("media_type") == "video"
            for row in manifest_by_video.values()
        ),
        "relevant_video_count": len(relevant_by_video),
        "hit": 0,
        "missing": 0,
        "empty": 0,
        "hashed": 0,
        "parsed": 0,
        "parse_errors": 0,
        "target_track_available": 0,
        "test_files_sealed": 0,
        "test_pose_read": False,
    }
    digest = hashlib.sha256()
    for video_id, partitions in sorted(relevant_by_video.items()):
        path = _resolve_pose_path(pose_dir, video_id)
        if path is None:
            result["missing"] += 1
            continue
        result["hit"] += 1
        if path.stat().st_size == 0:
            result["empty"] += 1
        if partitions == {"test"}:
            result["test_files_sealed"] += 1
            continue
        try:
            file_hash = _sha256(path)
            parsed = _read_jsonl(path)
        except (OSError, ValueError):
            result["parse_errors"] += 1
            continue
        result["hashed"] += 1
        result["parsed"] += 1
        if any(row.get("track_id") not in (None, "") for row in parsed):
            result["target_track_available"] += 1
        digest.update(path.name.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    result["development_pose_sha256"] = digest.hexdigest()
    result["development_inputs_usable"] = (
        result["missing"] == 0
        and result["empty"] == 0
        and result["parse_errors"] == 0
    )
    return result


def _historical_split_audit(paths: tuple[Path, ...], current_split_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            output.append({"path": path.as_posix(), "status": "missing"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        split_ids = sorted(set(re.findall(r"splitv3_[0-9a-f]{24}", text)))
        historical_only = bool(re.search(r"旧 split|历史实验|不得作为当前", text))
        output.append(
            {
                "path": path.as_posix(),
                "source_split_ids": split_ids,
                "matches_current_split": current_split_id in split_ids and not historical_only,
                "historical_only": historical_only,
            }
        )
    return output


def _render_audit(report: Mapping[str, Any]) -> str:
    facts = _mapping(report.get("facts"))
    pose = _mapping(report.get("pose_cache"))
    validation = _mapping(_mapping(report.get("gates")).get("validation_scale"))
    return (
        "# 步态模型训练审计\n\n"
        f"- 状态：`{report.get('status')}`\n"
        f"- split：`{facts.get('split_id')}`（`{facts.get('split_status')}`）\n"
        f"- 协议：`{facts.get('protocol_status')}`\n"
        f"- train/validation 姿态命中：{pose.get('hit')}；test sealed：{pose.get('test_files_sealed')}\n"
        f"- validation 独立正动作段：{validation.get('positive_segment_count')}\n"
        "- test pose/tensor/metrics：均未读取或生成。\n\n"
        "本报告只授权有界 train/validation 开发；规则步态分支继续作为主路径和 fallback。\n"
    )


def _render_blockers(report: Mapping[str, Any]) -> str:
    gates = _mapping(report.get("gates"))
    blocked = [name for name, gate in gates.items() if _mapping(gate).get("passed") is not True]
    lines = ["# 步态模型训练阻塞清单", ""]
    lines.extend(f"- `{name}`：未通过。" for name in blocked)
    lines.extend(
        [
            "",
            "正式训练仍依赖人工补数、B03 primary 监督、split/协议冻结和独立 test 保管人签发。",
        ]
    )
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
