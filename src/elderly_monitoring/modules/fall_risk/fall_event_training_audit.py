from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


AUDIT_SCHEMA_VERSION = "fall-event-training-audit-v1"
FROZEN_PROTOCOL_STATUS = "frozen"
MIN_VALIDATION_FALL_EVENTS = 30
MIN_HARD_NEGATIVE_GROUPS_PER_TYPE = 20
MIN_CONTINUOUS_CAMERA_HOURS = 100.0
MIN_ELDERLY_SUBJECTS = 20
MIN_ELDERLY_ADL_CAMERA_HOURS = 30.0
REQUIRED_PROTOCOL_FIELDS = (
    "score_threshold",
    "iou_threshold",
    "onset_tolerance_sec",
    "matching_operator",
    "one_to_one_assignment",
    "event_merge_enabled",
    "event_reset_gap_sec",
    "bootstrap_iterations",
    "bootstrap_grouping",
    "rejection_count_policy",
    "test_custodian_role",
    "test_release_policy",
)


def build_fall_event_training_audit(
    *,
    manifest: Path,
    action_labels: Path,
    event_labels: Path,
    assignments: Path,
    split: Path,
    formal: Path,
    validation: Path,
    evaluation: Path,
    training: Path,
    candidate_report: Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed P0 audit without parsing sealed test labels."""

    required_paths = {
        "manifest": manifest,
        "action_labels": action_labels,
        "event_labels": event_labels,
        "split_assignments": assignments,
        "split_report": split,
        "formal_validation": formal,
        "v3_validation": validation,
        "evaluation_config": evaluation,
        "training_config": training,
    }
    if candidate_report is not None:
        required_paths["candidate_report"] = candidate_report
    for name, path in required_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    hashes = {name: _sha256(path) for name, path in required_paths.items()}
    split_report = _read_json_object(split)
    formal_report = _read_json_object(formal)
    validation_report = _read_json_object(validation)
    evaluation_config = _read_yaml_object(evaluation)
    training_config = _read_yaml_object(training)

    hash_mismatches = _hash_mismatches(
        hashes=hashes,
        split_report=split_report,
        validation_report=validation_report,
    )
    current_split_id = split_report.get("split_id")
    validation_split = _mapping(validation_report.get("split"))
    validation_split_id = validation_split.get("split_id")
    if current_split_id != validation_split_id:
        hash_mismatches["split_id"] = {
            "actual": current_split_id,
            "expected_by_validation": validation_split_id,
        }

    manifest_facts = _manifest_facts(manifest)
    formal_counts = _mapping(formal_report.get("counts"))
    issue_counts = _mapping(
        _mapping(formal_report.get("distributions")).get("issue_code")
    )
    formal_ready = (
        formal_report.get("formal_ready") is True
        and formal_counts.get("errors") == 0
        and formal_counts.get("blockers") == 0
    )

    leakage_issues = split_report.get("leakage_issues")
    split_leakage_passed = (
        isinstance(leakage_issues, list)
        and not leakage_issues
        and validation_split.get("valid") is True
    )
    training_ready = _mapping(validation_report.get("training_ready"))
    fall_hard_negatives = _mapping(
        _mapping(validation_report.get("hard_negative_coverage")).get(
            "fall_event"
        )
    )
    missing_hard_negatives = fall_hard_negatives.get("missing")
    supervision_passed = (
        validation_report.get("valid") is True
        and training_ready.get("fall_event") is True
        and isinstance(missing_hard_negatives, list)
        and not missing_hard_negatives
    )

    supervision_counts = _mapping(
        _mapping(validation_split.get("partition_supervision_counts")).get(
            "fall_event"
        )
    )
    development_supervision = {
        partition: _mapping(supervision_counts.get(partition))
        for partition in ("train", "validation")
    }
    validation_positive_records = _optional_nonnegative_int(
        development_supervision["validation"].get("primary_positive")
    )
    validation_record_floor_met = (
        validation_positive_records is not None
        and validation_positive_records >= MIN_VALIDATION_FALL_EVENTS
    )

    protocol_status = evaluation_config.get("protocol_status")
    missing_protocol_fields = [
        field for field in REQUIRED_PROTOCOL_FIELDS if field not in evaluation_config
    ]
    protocol_passed = (
        protocol_status == FROZEN_PROTOCOL_STATUS and not missing_protocol_fields
    )

    split_status = split_report.get("status") or split_report.get("split_status")
    split_frozen = split_status == "frozen"
    candidate_split_ids = _candidate_split_ids(candidate_report)
    candidate_matches_current_split = (
        not candidate_split_ids or current_split_id in candidate_split_ids
    )

    gates = {
        "hash_bindings": {
            "passed": not hash_mismatches,
            "status": "passed" if not hash_mismatches else "blocked",
            "mismatches": hash_mismatches,
        },
        "v2_formal": {
            "passed": formal_ready,
            "status": "passed" if formal_ready else "blocked",
            "errors": formal_counts.get("errors"),
            "blockers": formal_counts.get("blockers"),
        },
        "fall_event_supervision": {
            "passed": supervision_passed,
            "status": "passed" if supervision_passed else "blocked",
            "missing_hard_negative_types": missing_hard_negatives,
        },
        "split_leakage": {
            "passed": split_leakage_passed,
            "status": "passed" if split_leakage_passed else "blocked",
            "reported_issue_count": (
                len(leakage_issues) if isinstance(leakage_issues, list) else None
            ),
        },
        "frozen_split": {
            "passed": split_frozen,
            "status": "passed" if split_frozen else "blocked",
            "reported_status": split_status or "provisional_unfrozen",
        },
        "validation_sample_scale": {
            "passed": False,
            "status": (
                "independent_group_audit_required"
                if validation_record_floor_met
                else "blocked"
            ),
            "observed_primary_label_records": validation_positive_records,
            "required_independent_events": MIN_VALIDATION_FALL_EVENTS,
            "independent_event_count_verified": False,
        },
        "hard_negative_group_scale": {
            "passed": False,
            "status": "independent_group_audit_required",
            "required_groups_per_type": MIN_HARD_NEGATIVE_GROUPS_PER_TYPE,
            "label_type_coverage_passed": supervision_passed,
        },
        "continuous_background": {
            "passed": (
                manifest_facts["continuous_camera_hours"]
                >= MIN_CONTINUOUS_CAMERA_HOURS
            ),
            "status": (
                "passed"
                if manifest_facts["continuous_camera_hours"]
                >= MIN_CONTINUOUS_CAMERA_HOURS
                else "blocked"
            ),
            "observed_camera_hours": manifest_facts["continuous_camera_hours"],
            "required_camera_hours": MIN_CONTINUOUS_CAMERA_HOURS,
        },
        "elderly_adl_domain": {
            "passed": (
                manifest_facts["elderly_adl_subject_count"] >= MIN_ELDERLY_SUBJECTS
                and manifest_facts["elderly_adl_camera_hours"]
                >= MIN_ELDERLY_ADL_CAMERA_HOURS
            ),
            "status": (
                "passed"
                if manifest_facts["elderly_adl_subject_count"]
                >= MIN_ELDERLY_SUBJECTS
                and manifest_facts["elderly_adl_camera_hours"]
                >= MIN_ELDERLY_ADL_CAMERA_HOURS
                else "blocked"
            ),
            "observed_subjects": manifest_facts["elderly_adl_subject_count"],
            "observed_camera_hours": manifest_facts["elderly_adl_camera_hours"],
            "required_subjects": MIN_ELDERLY_SUBJECTS,
            "required_camera_hours": MIN_ELDERLY_ADL_CAMERA_HOURS,
            "explicit_evidence_only": True,
        },
        "evaluation_protocol": {
            "passed": protocol_passed,
            "status": "passed" if protocol_passed else "blocked",
            "protocol_status": protocol_status,
            "missing_required_fields": missing_protocol_fields,
        },
        "test_release": {
            "passed": False,
            "status": "custodian_required",
            "test_truth_opened_by_audit": False,
        },
        "candidate_report_current_split": {
            "passed": candidate_matches_current_split,
            "status": (
                "passed" if candidate_matches_current_split else "rebuild_required"
            ),
            "candidate_split_ids": candidate_split_ids,
            "current_split_id": current_split_id,
        },
    }
    p0_passed = all(gate.get("passed") is True for gate in gates.values())
    status = "p0_ready" if p0_passed else "infrastructure_only"
    manifest_id_material = json.dumps(
        {"hashes": hashes, "split_id": current_split_id, "status": status},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_id = "fall_event_training_manifest_" + hashlib.sha256(
        manifest_id_material
    ).hexdigest()[:24]

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "status": status,
        "p0_passed": p0_passed,
        "scope": {
            "task": "fall_event",
            "near_fall_event_separate": True,
            "runtime_main_path": "FallStateDetector",
            "candidate_runtime_status": "provisional_shadow",
            "test_truth_policy": "custodian_only_not_opened_by_audit",
        },
        "facts": {
            "split_id": current_split_id,
            "split_assignment_count": split_report.get("assignment_count"),
            "split_group_count": split_report.get("group_count"),
            "v3_valid": validation_report.get("valid") is True,
            "training_ready": training_ready,
            "fall_event_supervision_visible_to_development": development_supervision,
            "formal_issue_counts": dict(sorted(issue_counts.items())),
            "continuous_video_count": manifest_facts["continuous_video_count"],
            "continuous_camera_hours": manifest_facts["continuous_camera_hours"],
            "elderly_adl_subject_count": manifest_facts[
                "elderly_adl_subject_count"
            ],
            "elderly_adl_camera_hours": manifest_facts[
                "elderly_adl_camera_hours"
            ],
            "evaluation_protocol": {
                "version": evaluation_config.get("protocol_version"),
                "status": protocol_status,
            },
            "training_config_status": training_config.get("current_status")
            or training_config.get("status"),
        },
        "gates": gates,
        "input_sha256": hashes,
        "authorization": {
            "formal_training_allowed": p0_passed,
            "formal_test_allowed": False,
            "audit_tools_tests_synthetic_and_short_smoke_allowed": True,
            "frozen_evaluation_config_generation_allowed": False,
        },
    }


def write_fall_event_training_audit(
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
            "fall-event audit output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_output.write_text(_render_audit_markdown(report), encoding="utf-8")
    blockers_output.write_text(_render_blockers_markdown(report), encoding="utf-8")


def _hash_mismatches(
    *,
    hashes: Mapping[str, str],
    split_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    split_inputs = _mapping(split_report.get("input_sha256"))
    validation_inputs = _mapping(validation_report.get("input_sha256"))
    expectations = {
        "manifest": (
            hashes["manifest"],
            [split_inputs.get("manifest"), validation_inputs.get("manifest")],
        ),
        "action_labels": (
            hashes["action_labels"],
            [
                split_inputs.get("action_labels"),
                validation_inputs.get("action_labels"),
            ],
        ),
        "event_labels": (
            hashes["event_labels"],
            [
                split_inputs.get("event_labels"),
                validation_inputs.get("event_labels"),
            ],
        ),
        "split_assignments": (
            hashes["split_assignments"],
            [
                split_report.get("assignments_sha256"),
                validation_inputs.get("split_assignments"),
            ],
        ),
        "split_report": (
            hashes["split_report"],
            [validation_inputs.get("split_report")],
        ),
    }
    for name, (actual, expected_values) in expectations.items():
        expected = [value for value in expected_values if isinstance(value, str)]
        if not expected or any(value != actual for value in expected):
            mismatches[name] = {"actual": actual, "expected": expected}
    return mismatches


def _manifest_facts(path: Path) -> dict[str, Any]:
    continuous_duration_by_video: dict[str, float] = {}
    elderly_adl_duration_by_video: dict[str, float] = {}
    elderly_adl_subjects: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
        if not isinstance(row, dict) or row.get("media_type") != "video":
            continue
        video_id = row.get("video_id")
        duration = _optional_positive_float(row.get("duration_sec"))
        if not isinstance(video_id, str) or not video_id or duration is None:
            continue
        if row.get("continuous_monitoring_eligible") is True:
            _set_consistent_duration(continuous_duration_by_video, video_id, duration)
        if (
            row.get("elderly_domain_eligible") is True
            and row.get("adl_monitoring_eligible") is True
        ):
            _set_consistent_duration(elderly_adl_duration_by_video, video_id, duration)
            subject_id = row.get("subject_id")
            if isinstance(subject_id, str) and subject_id.lower() not in {
                "",
                "unknown",
                "none",
                "null",
            }:
                elderly_adl_subjects.add(subject_id)
    return {
        "continuous_video_count": len(continuous_duration_by_video),
        "continuous_camera_hours": round(
            sum(continuous_duration_by_video.values()) / 3600.0, 6
        ),
        "elderly_adl_subject_count": len(elderly_adl_subjects),
        "elderly_adl_camera_hours": round(
            sum(elderly_adl_duration_by_video.values()) / 3600.0, 6
        ),
    }


def _set_consistent_duration(
    values: dict[str, float], video_id: str, duration: float
) -> None:
    existing = values.get(video_id)
    if existing is not None and not math.isclose(existing, duration, abs_tol=1e-6):
        raise ValueError(f"conflicting duration for video {video_id}")
    values[video_id] = duration


def _candidate_split_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8")
    declared = re.findall(
        r"source\s+split[^`\n]*`(splitv3_[0-9a-f]{24})`",
        text,
        flags=re.IGNORECASE,
    )
    if declared:
        return sorted(set(declared))
    return sorted(set(re.findall(r"splitv3_[0-9a-f]{24}", text)))


def _render_audit_markdown(report: Mapping[str, Any]) -> str:
    facts = _mapping(report.get("facts"))
    gates = _mapping(report.get("gates"))
    lines = [
        "# 跌倒事件训练 P0 审计",
        "",
        f"状态：`{report.get('status')}`。本报告不是训练结果、冻结数据发布或 test 评估。",
        "",
        "## 当前事实",
        "",
        f"- 当前 v3 split：`{facts.get('split_id')}`。",
        f"- v3 结构校验：`valid={str(facts.get('v3_valid')).lower()}`。",
        f"- 连续背景：`{facts.get('continuous_camera_hours')}` camera-hour。",
        f"- 显式老人 ADL：`{facts.get('elderly_adl_subject_count')}` 人、`{facts.get('elderly_adl_camera_hours')}` camera-hour。",
        f"- 评估协议：`{_mapping(facts.get('evaluation_protocol')).get('status')}`。",
        "- 本审计不解析 test 标签内容；test 只允许独立保管人执行一次性发布。",
        "",
        "## 门禁结果",
        "",
        "| 门禁 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    for name, raw_gate in gates.items():
        gate = _mapping(raw_gate)
        detail = _gate_detail(name, gate)
        lines.append(f"| `{name}` | `{gate.get('status')}` | {detail} |")
    lines.extend(
        [
            "",
            "## 推断",
            "",
            "P0 未通过，因此当前只允许审计、工具、测试、合成数据和短 smoke。不得启动正式训练、生成 frozen 协议、读取 test 真值或替换 `FallStateDetector` 主路径。",
            "",
            "## 建议",
            "",
            "由数据与协议负责人依次解除 formal、独立保护组规模、连续背景、老人域、frozen split/协议和 test 保管门禁；解除前保持候选状态为 `provisional_shadow`。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_blockers_markdown(report: Mapping[str, Any]) -> str:
    facts = _mapping(report.get("facts"))
    issue_counts = _mapping(facts.get("formal_issue_counts"))
    gates = _mapping(report.get("gates"))
    lines = [
        "# 跌倒事件训练阻塞清单",
        "",
        f"当前状态：`{report.get('status')}`。仅列出未通过或仍需独立复核的门禁。",
        "",
        "## Formal blockers",
        "",
    ]
    if issue_counts:
        for code, count in issue_counts.items():
            lines.append(f"- `{code}`：{count}")
    else:
        lines.append("- 无机器可读 issue 计数。")
    lines.extend(["", "## P0 blockers", ""])
    for name, raw_gate in gates.items():
        gate = _mapping(raw_gate)
        if gate.get("passed") is True:
            continue
        lines.append(f"- `{name}`：{_gate_detail(name, gate)}")
    lines.extend(
        [
            "",
            "## 停止边界",
            "",
            "- 不创建 `fall_event_v2.frozen.yaml` 或 frozen split。",
            "- 不读取 test 真值，不训练、选模、调阈值或拟合校准器。",
            "- 不把 candidate-clip 指标解释为连续事件指标。",
            "- 不修改规则主路径或 `AlgorithmEvent` 契约。",
            "",
        ]
    )
    return "\n".join(lines)


def _gate_detail(name: str, gate: Mapping[str, Any]) -> str:
    if name == "v2_formal":
        return f"errors={gate.get('errors')}，blockers={gate.get('blockers')}"
    if name == "hash_bindings":
        mismatches = _mapping(gate.get("mismatches"))
        return "输入 hash 一致" if not mismatches else "漂移：" + ", ".join(mismatches)
    if name == "validation_sample_scale":
        return (
            f"validation primary 标签记录={gate.get('observed_primary_label_records')}；"
            f"要求至少 {gate.get('required_independent_events')} 个独立事件且仍需保护组复核"
        )
    if name == "continuous_background":
        return (
            f"{gate.get('observed_camera_hours')}/{gate.get('required_camera_hours')} camera-hour"
        )
    if name == "elderly_adl_domain":
        return (
            f"{gate.get('observed_subjects')}/{gate.get('required_subjects')} 人，"
            f"{gate.get('observed_camera_hours')}/{gate.get('required_camera_hours')} camera-hour"
        )
    if name == "evaluation_protocol":
        missing = gate.get("missing_required_fields") or []
        return f"status={gate.get('protocol_status')}，缺失预注册字段={','.join(missing) or '无'}"
    if name == "candidate_report_current_split":
        return (
            f"候选报告 split={gate.get('candidate_split_ids')}，"
            f"当前 split={gate.get('current_split_id')}"
        )
    if name == "hard_negative_group_scale":
        return f"每类需至少 {gate.get('required_groups_per_type')} 个独立保护组"
    if name == "test_release":
        return "需要独立保管人和一次性发布授权"
    if name == "frozen_split":
        return f"split 状态={gate.get('reported_status')}"
    if name == "fall_event_supervision":
        return f"缺失 hard-negative 类型={gate.get('missing_hard_negative_types')}"
    if name == "split_leakage":
        return f"机器报告 leakage issue={gate.get('reported_issue_count')}"
    return str(gate.get("status"))


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0 else None
