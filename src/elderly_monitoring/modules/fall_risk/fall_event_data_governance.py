from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEVELOPMENT_PARTITIONS = {"train", "validation"}
ALL_PARTITIONS = {*DEVELOPMENT_PARTITIONS, "test"}


def govern_fall_event_training_data(
    *,
    action_labels: str | Path,
    event_labels: str | Path,
    manifest: str | Path,
    assignments: str | Path,
    split: str | Path,
    validation: str | Path,
    governance_config: str | Path,
    pose_roots: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a hash-bound train/validation supervision manifest without test access."""

    paths = {
        "action_labels": Path(action_labels),
        "event_labels": Path(event_labels),
        "manifest": Path(manifest),
        "split_assignments": Path(assignments),
        "split_report": Path(split),
        "validation_report": Path(validation),
        "governance_config": Path(governance_config),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"fall-event governance {name} not found: {path}")
    roots = [Path(value) for value in pose_roots]
    if not roots:
        raise ValueError("fall-event governance requires at least one pose root")
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"fall-event governance pose root not found: {root}")

    destination = Path(output_dir)
    output_paths = {
        "manifest": destination / "training_manifest.jsonl",
        "audit": destination / "audit.json",
        "report": destination / "README.md",
    }
    if destination.exists():
        raise FileExistsError(f"fall-event governance output already exists: {destination}")

    config = _read_json(paths["governance_config"])
    _validate_config(config)
    actual_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    expected_hashes = _mapping(config.get("input_sha256"), "input_sha256")
    for name in (
        "action_labels",
        "event_labels",
        "manifest",
        "split_assignments",
        "split_report",
        "validation_report",
    ):
        if expected_hashes.get(name) != actual_hashes[name]:
            raise ValueError(f"governance {name} SHA-256 mismatch")

    split_report = _read_json(paths["split_report"])
    validation_report = _read_json(paths["validation_report"])
    _validate_source_reports(
        split_report=split_report,
        validation_report=validation_report,
        actual_hashes=actual_hashes,
    )

    assignment_rows = _read_jsonl(paths["split_assignments"])
    assignment_index = _assignment_index(assignment_rows)
    manifest_index = _manifest_index(_read_jsonl(paths["manifest"]))
    action_rows = _read_jsonl(paths["action_labels"])
    event_rows = _read_jsonl(paths["event_labels"])

    locked_test_labels = 0
    candidates: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for row in event_rows:
        label_id = _label_id(row)
        assignment = _required_assignment(label_id, assignment_index)
        partition = str(assignment["partition"])
        if partition == "test":
            locked_test_labels += 1
            continue
        if row.get("schema_version") != "fall-risk-event-label-v3":
            raise ValueError(f"invalid development event schema: {label_id}")
        if row.get("task_type") != "fall_event":
            exclusions["non_fall_event_task"] += 1
            continue
        role = str(row.get("label_role", ""))
        tier = str(row.get("training_tier", ""))
        if role == "ignore" or tier == "ignore":
            exclusions["ignored_event_supervision"] += 1
            continue
        if role not in {"positive", "negative"} or tier not in {
            "primary",
            "auxiliary",
        }:
            raise ValueError(f"invalid development fall-event supervision: {label_id}")
        family = (
            "primary_fall"
            if role == "positive" and tier == "primary"
            else "auxiliary_fall"
            if role == "positive"
            else "explicit_hard_negative"
        )
        candidates.append(
            _candidate_from_label(
                row=row,
                assignment=assignment,
                manifest_index=manifest_index,
                family=family,
                source_label_kind="event",
                target_presence=int(role == "positive"),
                config=config,
            )
        )

    action_families = _mapping(
        config.get("action_negative_families"), "action_negative_families"
    )
    action_to_family: dict[str, str] = {}
    for family in ("ordinary_background", "recovered_near_fall"):
        action_ids = action_families.get(family)
        if not isinstance(action_ids, list) or not all(
            isinstance(value, str) for value in action_ids
        ):
            raise ValueError(f"invalid action list for {family}")
        for action_id in action_ids:
            if action_id in action_to_family:
                raise ValueError(f"action appears in multiple governance families: {action_id}")
            action_to_family[action_id] = family

    for row in action_rows:
        label_id = _label_id(row)
        assignment = _required_assignment(label_id, assignment_index)
        partition = str(assignment["partition"])
        if partition == "test":
            locked_test_labels += 1
            continue
        if row.get("schema_version") != "fall-risk-action-label-v3":
            raise ValueError(f"invalid development action schema: {label_id}")
        family = action_to_family.get(str(row.get("action_id", "")))
        if family is None:
            exclusions["action_not_authorized_for_fall_negative"] += 1
            continue
        tier = str(row.get("training_tier", ""))
        if tier == "ignore":
            exclusions["ignored_action_supervision"] += 1
            continue
        if tier not in {"primary", "auxiliary"}:
            raise ValueError(f"invalid development action training tier: {label_id}")
        candidates.append(
            _candidate_from_label(
                row=row,
                assignment=assignment,
                manifest_index=manifest_index,
                family=family,
                source_label_kind="action",
                target_presence=0,
                config=config,
            )
        )

    candidates.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["video_id"]),
            float(row["start_time_sec"]),
            str(row["source_label_id"]),
        )
    )
    materialized: list[dict[str, Any]] = []
    missing_pose_ids: set[str] = set()
    pose_root_counts: Counter[str] = Counter()
    for row in candidates:
        video_id = str(row["video_id"])
        pose_path, root = _resolve_pose_path(roots, video_id)
        if pose_path is None or root is None:
            exclusions["missing_development_pose"] += 1
            missing_pose_ids.add(video_id)
            continue
        governed = dict(row)
        governed["pose_path"] = pose_path.as_posix()
        governed["pose_root"] = root.as_posix()
        governed["pose_sha256"] = _sha256_file(pose_path)
        governed["materialization_status"] = "ready"
        materialized.append(governed)
        pose_root_counts[root.as_posix()] += 1

    if not materialized:
        raise ValueError("no governed train/validation fall-event samples are materializable")
    _assign_sampling_weights(materialized, config)
    _validate_partition_isolation(materialized)

    family_counts = _nested_counts(materialized, "partition", "supervision_family")
    dataset_counts = _nested_counts(materialized, "partition", "dataset")
    tier_counts = _nested_counts(materialized, "partition", "supervision_strength")
    sampling_mass = _sampling_mass(materialized)
    train_sampling_weights = [
        float(row["sampling_weight"])
        for row in materialized
        if row["partition"] == "train"
    ]
    split_status = str(split_report.get("status", "provisional_unfrozen"))
    audit = {
        "schema_version": "fall-event-continuous-training-governance-audit-v1",
        "governance_id": str(config["governance_id"]),
        "status": "provisional_training_data_governed",
        "source_split_id": split_report.get("split_id"),
        "source_split_status": split_status,
        "input_sha256": actual_hashes,
        "output_sha256": {},
        "counts": {
            "development_candidates": len(candidates),
            "materialized_samples": len(materialized),
            "locked_test_labels": locked_test_labels,
            "unique_development_videos": len(
                {str(row["video_id"]) for row in materialized}
            ),
            "unique_development_subjects": len(
                {str(row["subject_id"]) for row in materialized}
            ),
            "partition_family": family_counts,
            "partition_dataset": dataset_counts,
            "partition_supervision_strength": tier_counts,
            "excluded": dict(sorted(exclusions.items())),
        },
        "sampling": {
            "method": "bucket_then_dataset_then_subject_inverse_frequency",
            "training_weight_mean": round(
                sum(train_sampling_weights) / max(1, len(train_sampling_weights)),
                8,
            ),
            "training_weight_min": round(min(train_sampling_weights), 8),
            "training_weight_max": round(max(train_sampling_weights), 8),
            "mass_by_family": sampling_mass,
            "window_expansion_policy": (
                "divide each expanded window loss by windows in normalization_group_id"
            ),
        },
        "pose_coverage": {
            "candidate_count": len(candidates),
            "available_count": len(materialized),
            "missing_count": len(candidates) - len(materialized),
            "missing_video_count": len(missing_pose_ids),
            "missing_video_ids": sorted(missing_pose_ids),
            "resolved_root_counts": dict(sorted(pose_root_counts.items())),
        },
        "data_access": {
            "development_pose_roots": [root.as_posix() for root in roots],
            "test_label_semantics_parsed": False,
            "test_pose_read": False,
            "test_policy": "partition is checked before label semantics or pose resolution",
        },
        "causality": {
            "interpolated_coordinates": _mapping(
                config.get("pose_policy"), "pose_policy"
            ).get("interpolated_coordinates"),
            "current_cleaned_pose_online_causality_proven": False,
        },
        "gates": {
            "hash_bindings": "passed",
            "v3_fall_event_supervision": "passed",
            "test_isolation": "passed",
            "development_pose_coverage": (
                "passed" if len(candidates) == len(materialized) else "partial"
            ),
            "frozen_split": (
                "passed" if split_status in {"frozen", "formal_frozen"} else "blocked"
            ),
            "continuous_background": "blocked",
            "elderly_adl_domain": "blocked",
            "online_causal_pose_cleaning": "blocked",
            "main_path_replacement": "blocked",
        },
        "limitations": [
            "this artifact governs supervision and sampling; it is not a trained model result",
            "explicit action intervals increase background diversity but do not replace camera-hour continuous background",
            "SCF_MVP_V1 has three released training subjects and cannot establish elderly-domain generalization",
            "interpolated cleaned-pose coordinates must be masked before causal feature construction",
            "the split and evaluation protocol remain unfrozen, so rule replacement is not authorized",
        ],
    }

    destination.mkdir(parents=True, exist_ok=False)
    _write_jsonl_atomic(output_paths["manifest"], materialized)
    audit["output_sha256"]["training_manifest"] = _sha256_file(
        output_paths["manifest"]
    )
    _write_json_atomic(output_paths["audit"], audit)
    _write_text_atomic(output_paths["report"], _render_markdown(audit))
    return {
        "output_dir": destination.as_posix(),
        "manifest_path": output_paths["manifest"].as_posix(),
        "audit_path": output_paths["audit"].as_posix(),
        "report_path": output_paths["report"].as_posix(),
        "sample_count": len(materialized),
        "locked_test_label_count": locked_test_labels,
        "missing_pose_video_count": len(missing_pose_ids),
    }


def _candidate_from_label(
    *,
    row: Mapping[str, Any],
    assignment: Mapping[str, Any],
    manifest_index: Mapping[str, Mapping[str, Any]],
    family: str,
    source_label_kind: str,
    target_presence: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    label_id = str(row["label_id"])
    video_id = str(row.get("video_id", ""))
    if str(assignment.get("video_id", "")) != video_id:
        raise ValueError(f"label/assignment video mismatch: {label_id}")
    if str(assignment.get("label_kind", "")) != source_label_kind:
        raise ValueError(f"label/assignment kind mismatch: {label_id}")
    partition = str(assignment.get("partition", ""))
    if partition not in DEVELOPMENT_PARTITIONS:
        raise ValueError(f"invalid development partition: {label_id}")
    manifest_row = manifest_index.get(video_id)
    if manifest_row is None:
        raise ValueError(f"development label references missing manifest video: {video_id}")
    if manifest_row.get("eligibility") is not True:
        raise ValueError(f"development label references ineligible video: {video_id}")
    if str(manifest_row.get("asset_id", "")) != str(row.get("asset_id", "")):
        raise ValueError(f"label/manifest asset mismatch: {label_id}")

    start_frame = _nonnegative_int(row.get("start_frame"), f"{label_id}.start_frame")
    end_frame = _positive_int(
        row.get("end_frame_exclusive"), f"{label_id}.end_frame_exclusive"
    )
    start_time = _finite_nonnegative(row.get("start_time"), f"{label_id}.start_time")
    end_time = _finite_positive(
        row.get("end_time_exclusive"), f"{label_id}.end_time_exclusive"
    )
    if end_frame <= start_frame or end_time <= start_time:
        raise ValueError(f"invalid governed interval: {label_id}")

    tier = str(row.get("training_tier", ""))
    presence_weight, onset_weight = _loss_weights(
        row=row, family=family, tier=tier, config=config
    )
    allowed_heads = ["presence"]
    if onset_weight > 0:
        allowed_heads.append("onset")
    sample_key = f"{source_label_kind}:{label_id}:{family}"
    sample_id = "fallgovv1_" + hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:24]
    subject_id = str(row.get("subject_id", assignment.get("subject_id", "unknown")))
    return {
        "schema_version": "fall-event-continuous-supervision-v1",
        "sample_id": sample_id,
        "source_label_kind": source_label_kind,
        "source_label_id": label_id,
        "video_id": video_id,
        "asset_id": str(row.get("asset_id", "")),
        "subject_id": subject_id,
        "source_group_id": str(
            row.get("source_group_id", assignment.get("source_group_id", "unknown"))
        ),
        "sample_group_id": str(
            row.get("sample_group_id", assignment.get("sample_group_id", "unknown"))
        ),
        "split_group_id": str(assignment.get("split_group_id", "unknown")),
        "partition": partition,
        "dataset": str(manifest_row.get("dataset", "unknown")),
        "scene_region": str(manifest_row.get("scene_region", "unknown")),
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "start_time_sec": start_time,
        "end_time_exclusive_sec": end_time,
        "target_presence": target_presence,
        "supervision_family": family,
        "supervision_strength": tier,
        "boundary_precision": str(row.get("boundary_precision", "unknown")),
        "action_id": row.get("action_id"),
        "hard_negative_type": row.get("hard_negative_type"),
        "allowed_heads": allowed_heads,
        "presence_loss_weight": presence_weight,
        "onset_loss_weight": onset_weight,
        "normalization_group_id": label_id,
        "normalization_weight_policy": "inverse_expanded_window_count",
        "sampling_bucket": family,
        "sampling_weight": 0.0,
        "causal_pose_policy": "mask_interpolated_coordinates_and_derived_motion",
    }


def _loss_weights(
    *,
    row: Mapping[str, Any],
    family: str,
    tier: str,
    config: Mapping[str, Any],
) -> tuple[float, float]:
    policies = _mapping(config.get("family_policy"), "family_policy")
    policy = _mapping(policies.get(family), f"family_policy.{family}")
    if family in {"primary_fall", "auxiliary_fall", "explicit_hard_negative"}:
        presence = _positive_number(
            policy.get("presence_loss_weight"),
            f"family_policy.{family}.presence_loss_weight",
        )
    else:
        presence = _positive_number(
            policy.get(f"{tier}_presence_loss_weight"),
            f"family_policy.{family}.{tier}_presence_loss_weight",
        )
    onset = 0.0
    if family == "primary_fall" and row.get("boundary_precision") == "exact":
        source_types = {
            str(ref.get("source_type", ""))
            for ref in row.get("source_refs", [])
            if isinstance(ref, Mapping)
        }
        onset_sources = set(config.get("onset_source_types", []))
        if source_types & onset_sources:
            onset = 1.0
    return round(presence, 8), onset


def _assign_sampling_weights(
    rows: list[dict[str, Any]], config: Mapping[str, Any]
) -> None:
    policies = _mapping(config.get("family_policy"), "family_policy")
    train_rows = [row for row in rows if row["partition"] == "train"]
    hierarchy: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in train_rows:
        family = str(row["supervision_family"])
        dataset = str(row["dataset"])
        subject = str(row["subject_id"])
        if not subject or subject.lower() == "unknown":
            subject = str(row["source_group_id"])
        hierarchy[family][dataset][subject].append(row)
    for family, datasets in hierarchy.items():
        policy = _mapping(policies.get(family), f"family_policy.{family}")
        bucket_mass = _positive_number(
            policy.get("bucket_mass"), f"family_policy.{family}.bucket_mass"
        )
        for subjects in datasets.values():
            for subject_rows in subjects.values():
                raw = (
                    bucket_mass
                    / len(datasets)
                    / len(subjects)
                    / len(subject_rows)
                )
                for row in subject_rows:
                    row["sampling_weight"] = raw
    sampling_policy = _mapping(config.get("sampling_policy"), "sampling_policy")
    minimum = _positive_number(
        sampling_policy.get("min_sampling_weight"),
        "sampling_policy.min_sampling_weight",
    )
    maximum = _positive_number(
        sampling_policy.get("max_sampling_weight"),
        "sampling_policy.max_sampling_weight",
    )
    if maximum <= minimum or not minimum < 1.0 < maximum:
        raise ValueError("sampling weight bounds must contain 1.0")
    raw_weights = [float(row["sampling_weight"]) for row in train_rows]
    if not raw_weights or min(raw_weights) <= 0:
        raise ValueError("training sampling weights have zero mass")
    scale = _mean_preserving_clip_scale(raw_weights, minimum, maximum)
    for row in train_rows:
        row["sampling_weight"] = round(
            min(maximum, max(minimum, scale * float(row["sampling_weight"]))), 8
        )
    for row in rows:
        if row["partition"] != "train":
            row["sampling_weight"] = 1.0


def _mean_preserving_clip_scale(
    raw_weights: Sequence[float], minimum: float, maximum: float
) -> float:
    """Find a deterministic scale whose clipped weights have mean one."""

    lower = 0.0
    upper = 1.0

    def clipped_mean(scale: float) -> float:
        return sum(
            min(maximum, max(minimum, scale * value)) for value in raw_weights
        ) / len(raw_weights)

    while clipped_mean(upper) < 1.0:
        upper *= 2.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if clipped_mean(middle) < 1.0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def _validate_source_reports(
    *,
    split_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    actual_hashes: Mapping[str, str],
) -> None:
    if split_report.get("leakage_issues"):
        raise ValueError("source fall-risk split contains leakage issues")
    if validation_report.get("valid") is not True:
        raise ValueError("source fall-risk v3 validation is not valid")
    if _mapping(validation_report.get("training_ready"), "training_ready").get(
        "fall_event"
    ) is not True:
        raise ValueError("source fall-event supervision gate is not ready")
    split_inputs = _mapping(split_report.get("input_sha256"), "split.input_sha256")
    for name in ("action_labels", "event_labels", "manifest"):
        expected = split_inputs.get(name)
        if expected is not None and expected != actual_hashes[name]:
            raise ValueError(f"source split {name} SHA-256 mismatch")
    expected_assignments = split_report.get("assignments_sha256")
    if (
        expected_assignments is not None
        and expected_assignments != actual_hashes["split_assignments"]
    ):
        raise ValueError("source split assignments SHA-256 mismatch")
    validation_inputs = _mapping(
        validation_report.get("input_sha256"), "validation.input_sha256"
    )
    for name in (
        "action_labels",
        "event_labels",
        "manifest",
        "split_assignments",
        "split_report",
    ):
        expected = validation_inputs.get(name)
        if expected is not None and expected != actual_hashes[name]:
            raise ValueError(f"source validation {name} SHA-256 mismatch")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "fall-event-continuous-training-governance-v1":
        raise ValueError("unsupported fall-event training governance schema")
    if not isinstance(config.get("governance_id"), str):
        raise ValueError("fall-event governance_id is required")
    onset_sources = config.get("onset_source_types")
    if not isinstance(onset_sources, list) or not all(
        isinstance(value, str) for value in onset_sources
    ):
        raise ValueError("fall-event onset_source_types must be a string list")
    pose_policy = _mapping(config.get("pose_policy"), "pose_policy")
    if (
        pose_policy.get("interpolated_coordinates")
        != "mask_coordinates_and_derived_motion"
    ):
        raise ValueError("governance must mask non-causal interpolated coordinates")
    if pose_policy.get("test_pose_access") != "forbidden":
        raise ValueError("governance must forbid test pose access")


def _assignment_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        label_id = _label_id(row)
        if label_id in output:
            raise ValueError(f"duplicate split assignment: {label_id}")
        if row.get("partition") not in ALL_PARTITIONS:
            raise ValueError(f"invalid split assignment partition: {label_id}")
        output[label_id] = dict(row)
    return output


def _required_assignment(
    label_id: str, assignments: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    assignment = assignments.get(label_id)
    if assignment is None:
        raise ValueError(f"label has no split assignment: {label_id}")
    return assignment


def _manifest_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_video_id = row.get("video_id")
        if not isinstance(raw_video_id, str) or not raw_video_id:
            continue
        if raw_video_id in output:
            raise ValueError(f"duplicate manifest video_id: {raw_video_id}")
        output[raw_video_id] = dict(row)
    return output


def _resolve_pose_path(roots: Sequence[Path], video_id: str) -> tuple[Path | None, Path | None]:
    for root in roots:
        for name in (f"{video_id}.jsonl", f"{video_id}_poses_cleaned.jsonl"):
            candidate = root / name
            if candidate.is_file():
                return candidate, root
    return None, None


def _validate_partition_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    for field in (
        "subject_id",
        "source_group_id",
        "sample_group_id",
        "video_id",
        "split_group_id",
    ):
        partitions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = str(row.get(field, "")).strip()
            if value and value.lower() != "unknown":
                partitions[value].add(str(row["partition"]))
        leaked = sorted(value for value, values in partitions.items() if len(values) > 1)
        if leaked:
            raise ValueError(f"governed fall-event {field} crosses partitions: {leaked[0]}")


def _nested_counts(
    rows: Sequence[Mapping[str, Any]], outer: str, inner: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row[outer])][str(row[inner])] += 1
    return {
        key: dict(sorted(counter.items())) for key, counter in sorted(counts.items())
    }


def _sampling_mass(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    mass: Counter[str] = Counter()
    for row in rows:
        if row["partition"] == "train":
            mass[str(row["supervision_family"])] += float(row["sampling_weight"])
    return {key: round(value, 6) for key, value in sorted(mass.items())}


def _render_markdown(audit: Mapping[str, Any]) -> str:
    counts = _mapping(audit["counts"], "counts")
    pose = _mapping(audit["pose_coverage"], "pose_coverage")
    gates = _mapping(audit["gates"], "gates")
    lines = [
        "# 跌倒连续模型训练数据治理 v1",
        "",
        f"状态：`{audit['status']}`。本产物不是模型效果、冻结发布或主链路替换证据。",
        "",
        "## 数据结果",
        "",
        f"- 开发候选监督：{counts['development_candidates']} 条。",
        f"- 可物化训练/验证监督：{counts['materialized_samples']} 条。",
        f"- 唯一开发视频：{counts['unique_development_videos']} 个。",
        f"- 姿态缺失视频：{pose['missing_video_count']} 个。",
        f"- 封锁 test 标签：{counts['locked_test_labels']} 条；未解析语义、未读取姿态。",
        "",
        "## 采样与监督",
        "",
        "- 事件标签优先；动作标签只补充普通背景与恢复型近跌倒负监督。",
        "- train 权重按监督桶、数据来源、主体三级逆频率归一。",
        "- 只有 primary LE2I 官方精确边界正例训练 onset；辅助整段边界只训练 presence。",
        "- 窗口扩增后必须按 `normalization_group_id` 做事件内归一，防止长事件支配 loss。",
        "- 清洗姿态中的插值坐标及其派生运动必须屏蔽，不能作为在线因果输入。",
        "",
        "## 门禁",
        "",
        "| 门禁 | 状态 |",
        "|---|---|",
    ]
    lines.extend(f"| `{key}` | `{value}` |" for key, value in gates.items())
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本治理扩大了可用显式监督，但没有补齐连续背景 camera-hour、老人域、冻结 split/协议或在线因果姿态清洗，因此不能据此替换规则主路径。",
            "",
        ]
    )
    return "\n".join(lines)


def _label_id(row: Mapping[str, Any]) -> str:
    value = row.get("label_id")
    if not isinstance(value, str) or not value:
        raise ValueError("label_id is required")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    os.replace(partial, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, value: str) -> None:
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)
