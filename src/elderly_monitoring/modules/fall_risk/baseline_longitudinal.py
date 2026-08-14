"""Leakage-safe longitudinal evaluation infrastructure for personal baselines.

The split builder never uses outcome values for allocation.  Each identity is
assigned to one outer partition by its conservative source group, then its own
timeline is divided into reference-only and scoring periods.  Outcome labels
remain outside split assignments and are joined only by the evaluator.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from elderly_monitoring.modules.fall_risk.baseline import (
    BaselineModelConfig,
    PersonalBaselineTracker,
    build_personal_baselines,
    load_baseline_config,
    score_baseline_deviation,
)
from elderly_monitoring.modules.fall_risk.features import FALL_RISK_FEATURE_SPECS


PARTITIONS = ("train", "validation", "test")
PROTOCOL_STATUSES = {"provisional", "frozen"}
QUALITY_STATES = {"valid", "reduced", "unavailable"}
BASELINE_STATES = {
    "unavailable",
    "none",
    "cold",
    "initial",
    "stable",
    "drift_suspected",
    "recovery",
}
REQUIRED_VARIANTS = (
    "no_personal_baseline",
    "mean_std_personal_baseline",
    "median_mad_personal_baseline",
    "robust_ewma_cusum_guarded",
)
_UNKNOWN = {"", "unknown", "none", "null", "n/a", "na", "missing", "unavailable"}
_OBSERVATION_REQUIRED_FIELDS = {
    "observation_id",
    "record_type",
    "schema_version",
    "asset_id",
    "person_id",
    "device_id",
    "camera_profile_id",
    "source_group_id",
    "period_id",
    "period_start",
    "period_end",
    "timezone",
    "completed",
    "aggregation_version",
    "upstream_versions",
    "input_summary",
    "valid_monitoring_hours",
    "metric_quality",
    "quality_state",
    "no_personal_baseline_score",
    "no_personal_baseline_available_weight",
    "fusion_config_version",
    "outcome_label_id",
}
_LABEL_REQUIRED_FIELDS = {
    "label_id",
    "asset_id",
    "task_type",
    "subject_id",
    "start_time",
    "end_time",
    "risk_level",
    "risk_factors",
    "label_source",
}
_REVIEW_REQUIRED_FIELDS = {
    "review_id",
    "label_id",
    "reviewer_id",
    "decision",
    "reviewed_at",
    "evidence_reference",
}
_PREDICTION_REQUIRED_FIELDS = {
    "observation_id",
    "variant",
    "score",
    "baseline_state",
    "quality_state",
    "model_version",
    "config_hash",
    "split_id",
}


class LongitudinalConfigError(ValueError):
    pass


class LongitudinalDataError(ValueError):
    pass


@dataclass(frozen=True)
class LongitudinalEvaluationResult:
    partition: str
    protocol_version: str
    protocol_status: str
    protocol_hash: str
    split_id: str
    metrics_by_variant: dict[str, dict[str, Any]]
    confidence_intervals: dict[str, dict[str, Any]]
    stratified_metrics: dict[str, dict[str, Any]]
    failure_cases: list[dict[str, Any]]
    sample_count: int


def generate_longitudinal_ablation_predictions(
    observations: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    split_metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    partition: str,
    baseline_config: BaselineModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Replay the four fixed statistical candidates without opening outcomes."""
    config = _validate_protocol(protocol)
    if partition not in PARTITIONS:
        raise LongitudinalDataError("partition must be train, validation or test")
    if partition == "test":
        raise LongitudinalDataError(
            "test prediction generation must run inside the custodian release workflow"
        )
    assignment_rows = [dict(row) for row in assignments]
    split_id = _validate_evaluation_split(
        split_metadata,
        assignment_rows,
        config,
        partition=partition,
    )
    observation_index = _unique_index(observations, "observation_id", "observation")
    allowed_ids = {
        str(row["observation_id"])
        for row in assignment_rows
        if row.get("partition") == partition
    }
    if set(observation_index) != allowed_ids:
        raise LongitudinalDataError(
            "observation input must contain exactly one partition; test features must remain sealed"
        )
    expected_hashes = split_metadata.get("partition_input_sha256", {}).get(partition, {})
    if expected_hashes.get("observations") != _rows_hash(observation_index.values()):
        raise LongitudinalDataError("observation hash does not match split partition")

    model_config = baseline_config or load_baseline_config()
    if model_config.aggregation_period != config["aggregation_period"]:
        raise LongitudinalDataError("baseline and longitudinal aggregation periods differ")
    required_reference = max(model_config.min_history_records, model_config.stable_history_days)
    if config["min_reference_periods"] < required_reference:
        raise LongitudinalDataError(
            "longitudinal reference minimum is below the baseline stable-state requirement"
        )
    baseline_config_hash = _value_hash(asdict(model_config))
    fusion_versions = {
        str(row.get("fusion_config_version", "")) for row in observation_index.values()
    }
    if len(fusion_versions) != 1 or not _known(next(iter(fusion_versions), "")):
        raise LongitudinalDataError("one fusion_config_version is required per replay")
    replay_config_hash = _value_hash(
        {
            "algorithm": "fall-baseline-longitudinal-causal-replay-v1",
            "baseline_config_sha256": baseline_config_hash,
            "fusion_config_version": next(iter(fusion_versions)),
            "protocol_sha256": _value_hash(config),
        }
    )
    baseline_weight = next(
        spec.weight
        for spec in FALL_RISK_FEATURE_SPECS
        if spec.name == "baseline_deviation_score"
    )

    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    assignment_by_id = {
        str(row["observation_id"]): row
        for row in assignment_rows
        if row.get("partition") == partition
    }
    for observation_id, observation in observation_index.items():
        assignment = assignment_by_id[observation_id]
        by_identity[(str(assignment["person_id"]), str(assignment["camera_profile_id"]))].append(
            observation
        )

    predictions: list[dict[str, Any]] = []
    for identity in sorted(by_identity):
        timeline = sorted(
            by_identity[identity],
            key=lambda row: (_time(row["period_start"]), str(row["observation_id"])),
        )
        static_history: list[dict[str, Any]] = []
        guarded_tracker = PersonalBaselineTracker(config=model_config)
        for current in timeline:
            role = assignment_by_id[str(current["observation_id"])]["role"]
            robust = _static_baseline_result(static_history, current, model_config)
            guarded = guarded_tracker.update(current)
            if role == "scoring":
                legacy_score = _aggregate_metric_scores(
                    {
                        **robust["legacy_mean_std_metric_scores"],
                        "__scene_deviation_score": robust["scene_deviation_score"],
                    },
                    robust["available_metric_mask"],
                    {
                        **current,
                        "quality_mean": robust["baseline_features"].get(
                            "quality_mean"
                        ),
                    },
                    model_config,
                )
                if legacy_score is not None and robust["baseline_quality"].get(
                    "reduced_baseline_quality"
                ):
                    legacy_score = min(
                        legacy_score, model_config.reduced_quality_score_cap
                    )
                robust_score = _number(robust.get("baseline_deviation_score"))
                guarded_score = _number(guarded.get("baseline_deviation_score"))
                for variant, personal_score, state in (
                    ("no_personal_baseline", None, "unavailable"),
                    ("mean_std_personal_baseline", legacy_score, str(robust["baseline_state"])),
                    ("median_mad_personal_baseline", robust_score, str(robust["baseline_state"])),
                    ("robust_ewma_cusum_guarded", guarded_score, str(guarded["baseline_state"])),
                ):
                    predictions.append(
                        {
                            "observation_id": current["observation_id"],
                            "variant": variant,
                            "score": _fuse_personal_baseline(
                                current,
                                personal_score,
                                baseline_weight=baseline_weight,
                            ),
                            "baseline_state": state,
                            "quality_state": current["quality_state"],
                            "model_version": f"{variant}-causal-replay-v1",
                            "config_hash": replay_config_hash,
                            "split_id": split_id,
                        }
                    )
            static_history.append(current)
    return sorted(predictions, key=lambda row: (row["variant"], row["observation_id"]))


def build_longitudinal_split(
    observations: Iterable[Mapping[str, Any]],
    risk_labels: Iterable[Mapping[str, Any]],
    subject_profiles: Mapping[str, Any],
    review_log: Iterable[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    formal_validation_report: Mapping[str, Any] | None = None,
    formal_input_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic outer-group and within-person forward split.

    ``formal_input_sha256`` is a trusted-boundary input. File-backed callers
    must compute byte hashes from the materialized inputs, as the split CLI
    does, before requesting a frozen artifact.
    """
    config = _validate_protocol(protocol)
    raw_observations = [dict(row) for row in observations]
    raw_labels = [dict(row) for row in risk_labels]
    longitudinal_labels = [
        row for row in raw_labels if row.get("task_type") == "longitudinal_baseline"
    ]
    raw_reviews = [dict(row) for row in review_log]
    profiles_payload = dict(subject_profiles)
    blockers: list[dict[str, Any]] = []

    if not raw_observations:
        _block(blockers, "no_longitudinal_observations", "No longitudinal period observations exist.")
    if not longitudinal_labels:
        _block(blockers, "no_risk_labels", "No independently reviewed longitudinal risk labels exist.")
    subjects = profiles_payload.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        _block(blockers, "no_subject_profiles", "No stable, consent-linked subject profiles exist.")

    if config["protocol_status"] == "frozen":
        _validate_formal_attestation(
            formal_validation_report,
            formal_input_sha256,
            blockers,
        )

    profile_index = _validate_profiles(profiles_payload, blockers, frozen=config["protocol_status"] == "frozen")
    label_index = _validate_labels(longitudinal_labels, blockers)
    review_index = _validate_reviews(raw_reviews, blockers)
    for label_id in sorted(set(review_index) - set(label_index)):
        _block(
            blockers,
            "review_for_unknown_label",
            f"Review log references an unknown longitudinal label: {label_id}.",
            label_id=label_id,
        )
    normalized = _validate_observations(raw_observations, profile_index, blockers, config)
    _validate_outcome_links(normalized, label_index, review_index, config, blockers)

    assignments: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_identity[(row["person_id"], row["camera_profile_id"])].append(row)

    eligible_subjects: set[str] = set()
    for identity in sorted(by_identity):
        timeline = sorted(
            by_identity[identity],
            key=lambda row: (row["period_start_epoch"], row["period_end_epoch"], row["observation_id"]),
        )
        if any(
            earlier["period_end_epoch"] >= later["period_start_epoch"]
            for earlier, later in zip(timeline, timeline[1:])
        ):
            _block(
                blockers,
                "overlapping_identity_periods",
                f"Periods overlap for {identity[0]}/{identity[1]}.",
                person_id=identity[0],
                camera_profile_id=identity[1],
            )
            continue
        minimum = config["min_reference_periods"] + config["min_scoring_periods_per_identity"]
        if len(timeline) < minimum:
            _block(
                blockers,
                "insufficient_identity_timeline",
                f"{identity[0]}/{identity[1]} has {len(timeline)} periods; requires at least {minimum}.",
                person_id=identity[0],
                camera_profile_id=identity[1],
            )
            continue
        reference = timeline[: config["min_reference_periods"]]
        scoring = timeline[config["min_reference_periods"] :]
        first_scoring_start = scoring[0]["period_start_epoch"]
        if any(row["period_end_epoch"] >= first_scoring_start for row in reference):
            _block(
                blockers,
                "reference_scoring_overlap",
                f"Reference periods overlap scoring for {identity[0]}/{identity[1]}.",
                person_id=identity[0],
                camera_profile_id=identity[1],
            )
            continue
        if any(not _known(row.get("outcome_label_id")) for row in scoring):
            _block(
                blockers,
                "scoring_period_missing_outcome",
                f"Every scoring period requires an independent outcome for {identity[0]}/{identity[1]}.",
                person_id=identity[0],
                camera_profile_id=identity[1],
            )
            continue

        partition = _allocate_partition(
            str(timeline[0]["source_group_id"]), config["seed"], config["outer_partitions"]
        )
        eligible_subjects.add(identity[0])
        for role, rows in (("reference", reference), ("scoring", scoring)):
            for row in rows:
                assignments.append(
                    {
                        "observation_id": row["observation_id"],
                        "person_id": row["person_id"],
                        "device_id": row["device_id"],
                        "camera_profile_id": row["camera_profile_id"],
                        "source_group_id": row["source_group_id"],
                        "period_id": row["period_id"],
                        "period_start_epoch": row["period_start_epoch"],
                        "period_end_epoch": row["period_end_epoch"],
                        "valid_monitoring_hours": row["valid_monitoring_hours"],
                        "quality_state": row["quality_state"],
                        "partition": partition,
                        "role": role,
                        "protocol_sha256": _value_hash(config),
                    }
                )

    assignments.sort(key=_assignment_sort_key)
    blockers.extend(audit_longitudinal_split(assignments))
    counts = Counter(row["partition"] for row in assignments if row["role"] == "scoring")
    subject_counts = {
        partition: len(
            {
                row["person_id"]
                for row in assignments
                if row["partition"] == partition and row["role"] == "scoring"
            }
        )
        for partition in PARTITIONS
    }
    if raw_observations and len(eligible_subjects) < config["min_total_subjects"]:
        _block(
            blockers,
            "insufficient_total_subjects",
            f"Only {len(eligible_subjects)} eligible subjects; requires {config['min_total_subjects']}.",
        )
    if raw_observations:
        for partition in PARTITIONS:
            required = config["min_subjects_per_partition"]
            if config["outer_partitions"][partition] > 0 and subject_counts[partition] < required:
                _block(
                    blockers,
                    "insufficient_partition_subjects",
                    f"{partition} has {subject_counts[partition]} subjects; requires {required}.",
                    partition=partition,
                )

    blockers = _dedupe_blockers(blockers)
    raw_observations_by_id = {
        str(row.get("observation_id", "")): row for row in raw_observations
    }
    partition_input_sha256 = _partition_input_hashes(
        assignments,
        raw_observations_by_id,
        label_index,
    )
    metadata = {
        "schema_version": "fall-baseline-longitudinal-split-v1",
        "split_name": "longitudinal_baseline_v1",
        "task_type": "longitudinal_baseline",
        "protocol_version": config["protocol_version"],
        "protocol_status": config["protocol_status"],
        "status": "blocked" if blockers else ("frozen" if config["protocol_status"] == "frozen" else "ready"),
        "split_id": None,
        "split_sha256": None,
        "allocation_method": "sha256(seed+source_group_id), then chronological reference/scoring cut",
        "outcome_blind_allocation": True,
        "test_governance": dict(config["test_governance"]),
        "protocol_sha256": _value_hash(config),
        "observations_sha256": _rows_hash(raw_observations),
        "risk_labels_sha256": _rows_hash(raw_labels),
        "subject_profiles_sha256": _value_hash(profiles_payload),
        "review_log_sha256": _rows_hash(raw_reviews),
        "formal_validation_report_sha256": (
            _value_hash(formal_validation_report)
            if formal_validation_report is not None
            else None
        ),
        "partition_input_sha256": partition_input_sha256,
        "assignments_sha256": _rows_hash(assignments),
        "counts": {partition: counts[partition] for partition in PARTITIONS},
        "subject_counts": subject_counts,
        "reference_period_count": sum(row["role"] == "reference" for row in assignments),
        "scoring_period_count": sum(row["role"] == "scoring" for row in assignments),
        "eligible_subject_count": len(eligible_subjects),
        "blockers": blockers,
    }
    if not blockers and assignments:
        split_payload = {**metadata, "split_id": None, "split_sha256": None, "blockers": []}
        split_sha256 = _value_hash({"metadata": split_payload, "assignments": assignments})
        split_id = f"longitudinal_baseline_v1:sha256:{split_sha256[:16]}"
        metadata["split_id"] = split_id
        metadata["split_sha256"] = split_sha256
        assignments = [{**row, "split_id": split_id} for row in assignments]
        metadata["assignments_sha256"] = _rows_hash(assignments)
    elif blockers:
        assignments = []
        metadata["assignments_sha256"] = _rows_hash([])
        metadata["reference_period_count"] = 0
        metadata["scoring_period_count"] = 0
        metadata["counts"] = {partition: 0 for partition in PARTITIONS}
        metadata["subject_counts"] = {partition: 0 for partition in PARTITIONS}
        metadata["partition_input_sha256"] = {
            partition: {"observations": _rows_hash([]), "risk_labels": _rows_hash([])}
            for partition in PARTITIONS
        }
    return {"metadata": metadata, "assignments": assignments}


def audit_longitudinal_split(assignments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic, machine-readable split and chronology findings."""
    rows = [dict(row) for row in assignments]
    findings: list[dict[str, Any]] = []
    for key in ("person_id", "source_group_id"):
        partitions: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if _known(row.get(key)):
                partitions[str(row[key])].add(str(row.get("partition", "")))
        for value, values in sorted(partitions.items()):
            if len(values) > 1:
                _block(
                    findings,
                    "cross_partition_leakage",
                    f"{key}={value} appears in multiple partitions: {sorted(values)}.",
                    key=key,
                    value=value,
                    partitions=sorted(values),
                )

    seen_ids: set[str] = set()
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        observation_id = str(row.get("observation_id", ""))
        if not observation_id or observation_id in seen_ids:
            _block(findings, "duplicate_assignment", f"Duplicate assignment: {observation_id}.")
        seen_ids.add(observation_id)
        by_identity[(str(row.get("person_id", "")), str(row.get("camera_profile_id", "")))].append(row)

    for identity, timeline in sorted(by_identity.items()):
        reference = [row for row in timeline if row.get("role") == "reference"]
        scoring = [row for row in timeline if row.get("role") == "scoring"]
        if not reference or not scoring:
            _block(
                findings,
                "missing_timeline_role",
                f"{identity[0]}/{identity[1]} requires reference and scoring periods.",
            )
            continue
        first_scoring = min(float(row["period_start_epoch"]) for row in scoring)
        if any(float(row["period_end_epoch"]) >= first_scoring for row in reference):
            _block(
                findings,
                "reference_not_strictly_earlier",
                f"Reference is not strictly earlier for {identity[0]}/{identity[1]}.",
            )
    return _dedupe_blockers(findings)


def evaluate_longitudinal_ablation(
    observations: Iterable[Mapping[str, Any]],
    risk_labels: Iterable[Mapping[str, Any]],
    assignments: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    partition: str,
    split_metadata: Mapping[str, Any],
    test_release_ack: Mapping[str, Any] | None = None,
    code_commit: str | None = None,
) -> LongitudinalEvaluationResult:
    """Evaluate ablation predictions on one forward-scoring partition."""
    config = _validate_protocol(protocol)
    if partition not in {"validation", "test"}:
        raise LongitudinalDataError("partition must be validation or test")
    if partition == "test":
        if config["protocol_status"] != "frozen":
            raise LongitudinalDataError("test evaluation requires a frozen protocol")
        if not config["test_governance"]["custodian"] or not config["test_governance"]["frozen_at"]:
            raise LongitudinalDataError("frozen test governance is incomplete")
        if test_release_ack is None:
            raise LongitudinalDataError(
                "test evaluation requires explicit test release acknowledgement"
            )

    observation_index = _unique_index(observations, "observation_id", "observation")
    label_index = _unique_index(risk_labels, "label_id", "risk label")
    assignment_rows = [dict(row) for row in assignments]
    split_id = _validate_evaluation_split(
        split_metadata,
        assignment_rows,
        config,
        partition=partition,
    )
    leakage = audit_longitudinal_split(assignment_rows)
    if leakage:
        raise LongitudinalDataError(f"longitudinal split leakage: {leakage[0]['code']}")
    selected_assignments = [
        row for row in assignment_rows if row.get("partition") == partition and row.get("role") == "scoring"
    ]
    if not selected_assignments:
        raise LongitudinalDataError(f"no scoring observations in {partition}")
    allowed_observation_ids = {
        str(row["observation_id"])
        for row in assignment_rows
        if row.get("partition") == partition
    }
    if set(observation_index) != allowed_observation_ids:
        raise LongitudinalDataError(
            "observation input must contain exactly one partition; test features must remain sealed"
        )

    examples: dict[str, dict[str, Any]] = {}
    for assignment in selected_assignments:
        observation_id = str(assignment["observation_id"])
        observation = observation_index.get(observation_id)
        if observation is None:
            raise LongitudinalDataError(f"missing observation: {observation_id}")
        label_id = observation.get("outcome_label_id")
        label = label_index.get(str(label_id))
        if label is None:
            raise LongitudinalDataError(f"missing outcome label for {observation_id}")
        if label.get("task_type") != "longitudinal_baseline" or label.get("subject_id") != observation.get("person_id"):
            raise LongitudinalDataError(f"invalid outcome linkage for {observation_id}")
        examples[observation_id] = _evaluation_example(observation, label, assignment, config)

    expected_label_ids = {
        str(observation_index[observation_id]["outcome_label_id"])
        for observation_id in examples
    }
    if set(label_index) != expected_label_ids:
        raise LongitudinalDataError(
            "risk label input must contain exactly the selected scoring partition; test truth must remain sealed"
        )
    partition_hashes = split_metadata.get("partition_input_sha256")
    expected_partition_hashes = (
        partition_hashes.get(partition) if isinstance(partition_hashes, Mapping) else None
    )
    if not isinstance(expected_partition_hashes, Mapping):
        raise LongitudinalDataError("split metadata is missing partition input hashes")
    if expected_partition_hashes.get("observations") != _rows_hash(observation_index.values()):
        raise LongitudinalDataError("observation hash does not match split partition")
    if expected_partition_hashes.get("risk_labels") != _rows_hash(label_index.values()):
        raise LongitudinalDataError("risk label hash does not match split partition")

    prediction_rows = [dict(row) for row in predictions]
    protocol_hash = _value_hash(config)
    assignment_protocol_hashes = {
        str(row.get("protocol_sha256", "")) for row in assignment_rows
    }
    if assignment_protocol_hashes != {protocol_hash}:
        raise LongitudinalDataError("assignment protocol hash does not match evaluation protocol")
    if partition == "test":
        _validate_test_release_ack(
            test_release_ack,
            split_id=split_id,
            config=config,
            protocol_hash=protocol_hash,
            predictions_hash=_rows_hash(prediction_rows),
            labels_hash=_rows_hash(label_index.values()),
            code_commit=code_commit,
        )
    _validate_prediction_rows(prediction_rows, split_id, config)
    required_variants = tuple(config["required_variants"])
    expected_ids = set(examples)
    by_variant: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in required_variants}
    for row in prediction_rows:
        if row["variant"] not in by_variant:
            raise LongitudinalDataError(f"unexpected prediction variant: {row['variant']}")
        if row["observation_id"] not in expected_ids:
            raise LongitudinalDataError(
                "prediction input must contain exactly the selected scoring partition"
            )
        variant_rows = by_variant[row["variant"]]
        if row["observation_id"] in variant_rows:
            raise LongitudinalDataError(
                f"duplicate prediction for {row['variant']}/{row['observation_id']}"
            )
        variant_rows[row["observation_id"]] = row
    for variant, variant_rows in by_variant.items():
        if set(variant_rows) != expected_ids:
            missing = sorted(expected_ids - set(variant_rows))
            extra = sorted(set(variant_rows) - expected_ids)
            raise LongitudinalDataError(
                f"prediction coverage differs for {variant}: missing={missing}, extra={extra}"
            )

    metrics_by_variant: dict[str, dict[str, Any]] = {}
    intervals: dict[str, dict[str, Any]] = {}
    strata: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    ordered_ids = sorted(expected_ids)
    for variant in required_variants:
        rows = [
            {
                **examples[observation_id],
                "score": float(by_variant[variant][observation_id]["score"]),
                "baseline_state": str(
                    by_variant[variant][observation_id]["baseline_state"]
                ),
            }
            for observation_id in ordered_ids
        ]
        metrics_by_variant[variant] = _binary_metrics(rows, config)
        intervals[variant] = _cluster_bootstrap(rows, config)
        strata[variant] = _stratified_metrics(rows, config)
        failures.extend(_failure_rows(rows, variant, config))

    failures.sort(key=lambda row: (row["variant"], row["failure_type"], row["person_id"], row["observation_id"]))
    return LongitudinalEvaluationResult(
        partition=partition,
        protocol_version=config["protocol_version"],
        protocol_status=config["protocol_status"],
        protocol_hash=protocol_hash,
        split_id=split_id,
        metrics_by_variant=metrics_by_variant,
        confidence_intervals=intervals,
        stratified_metrics=strata,
        failure_cases=failures,
        sample_count=len(examples),
    )


def write_longitudinal_split(artifact: Mapping[str, Any], output_dir: Path | str, *, overwrite: bool = False) -> None:
    directory = Path(output_dir)
    _atomic_bundle_write(
        directory,
        {
            "split.json": _json_text(artifact["metadata"]),
            "assignments.jsonl": _jsonl_text(artifact["assignments"]),
        },
        overwrite=overwrite,
    )


def write_longitudinal_evaluation_bundle(
    result: LongitudinalEvaluationResult,
    output_dir: Path | str,
    *,
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    payload = {
        "partition": result.partition,
        "protocol_version": result.protocol_version,
        "protocol_status": result.protocol_status,
        "protocol_sha256": result.protocol_hash,
        "split_id": result.split_id,
        "sample_count": result.sample_count,
        "metrics_by_variant": result.metrics_by_variant,
        "confidence_intervals_95": result.confidence_intervals,
        "reproducibility": dict(metadata),
    }
    files = {
        "metrics.json": _json_text(payload),
        "stratified_metrics.json": _json_text(result.stratified_metrics),
        "failure_cases.jsonl": _jsonl_text(result.failure_cases),
        "report.md": _evaluation_report(result, metadata),
    }
    _atomic_bundle_write(Path(output_dir), files, overwrite=overwrite)


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(protocol, Mapping):
        raise LongitudinalConfigError("protocol must be an object")
    required = {
        "schema_version",
        "protocol_version",
        "protocol_status",
        "seed",
        "aggregation_period",
        "outer_partitions",
        "min_reference_periods",
        "min_scoring_periods_per_identity",
        "min_subjects_per_partition",
        "min_total_subjects",
        "min_valid_monitoring_hours",
        "prediction_horizon_hours",
        "high_risk_level_threshold",
        "score_threshold",
        "bootstrap_iterations",
        "bootstrap_seed",
        "required_variants",
        "review",
        "test_governance",
    }
    unknown = set(protocol) - required
    missing = required - set(protocol)
    if missing or unknown:
        raise LongitudinalConfigError(
            f"invalid protocol fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    config = json.loads(json.dumps(protocol))
    if config["schema_version"] != "fall-baseline-longitudinal-protocol-v1":
        raise LongitudinalConfigError("unsupported schema_version")
    if config["protocol_status"] not in PROTOCOL_STATUSES:
        raise LongitudinalConfigError("protocol_status must be provisional or frozen")
    if config["aggregation_period"] not in {"day", "hour"}:
        raise LongitudinalConfigError("aggregation_period must be day or hour")
    partitions = config["outer_partitions"]
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
        raise LongitudinalConfigError("outer_partitions must define train, validation and test")
    if any(not _finite(value) or float(value) < 0 for value in partitions.values()):
        raise LongitudinalConfigError("outer partition ratios must be finite and non-negative")
    if not math.isclose(sum(float(value) for value in partitions.values()), 1.0, abs_tol=1e-9):
        raise LongitudinalConfigError("outer partition ratios must sum to one")
    config["outer_partitions"] = {key: float(partitions[key]) for key in PARTITIONS}
    for field in (
        "min_reference_periods",
        "min_scoring_periods_per_identity",
        "min_subjects_per_partition",
        "min_total_subjects",
        "bootstrap_iterations",
        "bootstrap_seed",
    ):
        if not isinstance(config[field], int) or isinstance(config[field], bool) or config[field] < 0:
            raise LongitudinalConfigError(f"{field} must be a non-negative integer")
    if config["min_reference_periods"] < 1 or config["min_scoring_periods_per_identity"] < 1:
        raise LongitudinalConfigError("reference and scoring minimums must be positive")
    if not _finite(config["min_valid_monitoring_hours"]) or float(
        config["min_valid_monitoring_hours"]
    ) <= 0:
        raise LongitudinalConfigError(
            "min_valid_monitoring_hours must be finite and positive"
        )
    config["min_valid_monitoring_hours"] = float(config["min_valid_monitoring_hours"])
    if not _finite(config["score_threshold"]) or not 0 <= float(
        config["score_threshold"]
    ) <= 1:
        raise LongitudinalConfigError("score_threshold must be between zero and one")
    config["score_threshold"] = float(config["score_threshold"])
    if not isinstance(config["high_risk_level_threshold"], int) or not 0 <= config["high_risk_level_threshold"] <= 4:
        raise LongitudinalConfigError("high_risk_level_threshold must be an integer from 0 to 4")
    horizon = config["prediction_horizon_hours"]
    if not isinstance(horizon, dict) or set(horizon) != {"min", "max"}:
        raise LongitudinalConfigError("prediction_horizon_hours must define min and max")
    if not all(_finite(value) for value in horizon.values()) or float(horizon["min"]) < 0 or float(horizon["max"]) < float(horizon["min"]):
        raise LongitudinalConfigError("invalid prediction_horizon_hours")
    config["prediction_horizon_hours"] = {"min": float(horizon["min"]), "max": float(horizon["max"])}
    variants = config["required_variants"]
    if not isinstance(variants, list) or tuple(variants) != REQUIRED_VARIANTS:
        raise LongitudinalConfigError(
            "required_variants must contain the four baseline ablations in fixed order"
        )
    review = config["review"]
    if not isinstance(review, dict) or set(review) != {
        "accepted_decision", "manual_consensus_min_reviewers", "clinical_proxy_min_reviewers"
    }:
        raise LongitudinalConfigError("invalid review policy")
    if review["accepted_decision"] != "accepted" or any(
        not isinstance(review[field], int)
        or isinstance(review[field], bool)
        or review[field] < 1
        for field in (
            "manual_consensus_min_reviewers",
            "clinical_proxy_min_reviewers",
        )
    ):
        raise LongitudinalConfigError("invalid review policy")
    governance = config["test_governance"]
    if not isinstance(governance, dict) or set(governance) != {"custodian", "frozen_at"}:
        raise LongitudinalConfigError("invalid test_governance")
    if config["protocol_status"] == "frozen" and (not _known(governance["custodian"]) or _time(governance["frozen_at"]) is None):
        raise LongitudinalConfigError("frozen protocol requires custodian and frozen_at")
    if config["protocol_status"] == "frozen" and config["bootstrap_iterations"] < 10_000:
        raise LongitudinalConfigError(
            "frozen protocol requires at least 10000 bootstrap iterations"
        )
    return config


def _static_baseline_result(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any],
    config: BaselineModelConfig,
) -> dict[str, Any]:
    if not history:
        return score_baseline_deviation([current], {}, config=config)[0]
    baselines = build_personal_baselines(history, config=config)
    return score_baseline_deviation([current], baselines, config=config)[0]


def _aggregate_metric_scores(
    scores: Mapping[str, Any],
    mask: Mapping[str, Any],
    current: Mapping[str, Any],
    config: BaselineModelConfig,
) -> float | None:
    metric_quality = current.get("metric_quality")
    if not isinstance(metric_quality, Mapping):
        return None
    weighted = 0.0
    total_weight = 0.0
    for metric, weight in config.metric_weights.items():
        score = _number(scores.get(metric))
        quality_payload = metric_quality.get(metric)
        quality = (
            _number(quality_payload.get("quality"))
            if isinstance(quality_payload, Mapping)
            else None
        )
        if mask.get(metric) is not True or score is None or quality is None:
            continue
        weighted += float(weight) * quality * score
        total_weight += float(weight) * quality
    scene_score = _number(scores.get("__scene_deviation_score"))
    current_quality = _number(current.get("quality_mean"))
    if scene_score is not None and current_quality is not None:
        weighted += 0.06 * current_quality * scene_score
        total_weight += 0.06 * current_quality
    if total_weight <= 0:
        return None
    score = weighted / total_weight
    return round(max(0.0, min(1.0, score)), 6)


def _fuse_personal_baseline(
    observation: Mapping[str, Any],
    personal_score: float | None,
    *,
    baseline_weight: float,
) -> float:
    base_score = _number(observation.get("no_personal_baseline_score"))
    base_weight = _number(observation.get("no_personal_baseline_available_weight"))
    if base_score is None or not 0 <= base_score <= 1 or base_weight is None or base_weight <= 0:
        raise LongitudinalDataError(
            f"invalid no-personal-baseline fusion inputs: {observation.get('observation_id')}"
        )
    if not _known(observation.get("fusion_config_version")):
        raise LongitudinalDataError(
            f"missing fusion_config_version: {observation.get('observation_id')}"
        )
    if personal_score is None:
        return round(base_score, 6)
    return round(
        max(
            0.0,
            min(
                1.0,
                (base_score * base_weight + personal_score * baseline_weight)
                / (base_weight + baseline_weight),
            ),
        ),
        6,
    )


def _validate_profiles(payload: Mapping[str, Any], blockers: list[dict[str, Any]], *, frozen: bool) -> dict[str, dict[str, Any]]:
    if set(payload) != {"schema_version", "subjects"} or payload.get("schema_version") != "fall-risk-subject-profiles-v2":
        if payload:
            _block(blockers, "invalid_subject_profile_schema", "Subject profiles must use fall-risk-subject-profiles-v2.")
        return {}
    subjects = payload.get("subjects")
    if not isinstance(subjects, list):
        _block(blockers, "invalid_subject_profile_schema", "subjects must be a list.")
        return {}
    output: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(subjects, 1):
        if not isinstance(raw, dict):
            _block(blockers, "invalid_subject_profile", f"Profile {position} is not an object.")
            continue
        person = str(raw.get("subject_id", "")).strip()
        features = raw.get("features")
        if not _known(person) or person in output or not isinstance(features, dict):
            _block(blockers, "invalid_subject_profile", f"Invalid or duplicate profile at {position}.", person_id=person)
            continue
        if frozen and not _known(raw.get("consent_id")):
            _block(blockers, "missing_formal_consent", f"Frozen profile lacks consent: {person}.", person_id=person)
        source_group = str(features.get("source_group_id", "")).strip()
        devices = features.get("authorized_device_ids")
        cameras = features.get("authorized_camera_profile_ids")
        if not _known(source_group) or not _string_list(devices) or not _string_list(cameras):
            _block(
                blockers,
                "missing_longitudinal_identity_scope",
                f"Profile lacks source group or authorized device/camera scope: {person}.",
                person_id=person,
            )
            continue
        output[person] = raw
    return output


def _validate_formal_attestation(
    report: Mapping[str, Any] | None,
    input_sha256: Mapping[str, str] | None,
    blockers: list[dict[str, Any]],
) -> None:
    if report is None or input_sha256 is None:
        _block(
            blockers,
            "missing_formal_validation_attestation",
            "Frozen longitudinal split requires a formal v2 label validation report.",
        )
        return
    if (
        report.get("schema_version") != "fall-risk-label-validation-report-v2"
        or report.get("mode") != "formal"
        or report.get("valid") is not True
        or report.get("formal_ready") is not True
    ):
        _block(
            blockers,
            "formal_validation_not_ready",
            "Formal v2 label validation report is not ready.",
        )
        return
    report_inputs = report.get("input_sha256")
    if not isinstance(report_inputs, Mapping):
        _block(blockers, "invalid_formal_validation_attestation", "Formal report input hashes are missing.")
        return
    for field in ("risk_labels", "subject_profiles"):
        if report_inputs.get(field) != input_sha256.get(field):
            _block(
                blockers,
                "formal_validation_input_mismatch",
                f"Formal validation report does not bind current {field}.",
                field=field,
            )


def _validate_labels(rows: Sequence[dict[str, Any]], blockers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, 1):
        missing = _LABEL_REQUIRED_FIELDS - set(row)
        label_id = str(row.get("label_id", "")).strip()
        if missing or not _known(label_id) or label_id in output:
            _block(blockers, "invalid_risk_label", f"Invalid risk label at {position}.", label_id=label_id)
            continue
        if row.get("task_type") != "longitudinal_baseline" or row.get("label_source") not in {"manual_consensus", "clinical_proxy"}:
            _block(blockers, "invalid_risk_label", f"Invalid longitudinal task/source for {label_id}.", label_id=label_id)
            continue
        risk_level = row.get("risk_level")
        start = _time(row.get("start_time"))
        end = _time(row.get("end_time"))
        if not isinstance(risk_level, int) or isinstance(risk_level, bool) or not 0 <= risk_level <= 4 or start is None or end is None or end < start:
            _block(blockers, "invalid_risk_label", f"Invalid level or time for {label_id}.", label_id=label_id)
            continue
        output[label_id] = row
    return output


def _validate_reviews(rows: Sequence[dict[str, Any]], blockers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for position, row in enumerate(rows, 1):
        missing = _REVIEW_REQUIRED_FIELDS - set(row)
        review_id = str(row.get("review_id", "")).strip()
        if (
            missing
            or set(row) != _REVIEW_REQUIRED_FIELDS
            or not _known(review_id)
            or review_id in seen
            or not _known(row.get("label_id"))
            or not _known(row.get("reviewer_id"))
            or row.get("decision") not in {"accepted", "rejected", "needs_revision"}
            or _time(row.get("reviewed_at")) is None
            or not _known(row.get("evidence_reference"))
        ):
            _block(blockers, "invalid_label_review", f"Invalid review at {position}.", review_id=review_id)
            continue
        seen.add(review_id)
        output[str(row["label_id"])].append(row)
    return output


def _validate_observations(
    rows: Sequence[dict[str, Any]],
    profiles: Mapping[str, dict[str, Any]],
    blockers: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    periods: set[tuple[str, str, str]] = set()
    for position, row in enumerate(rows, 1):
        missing = _OBSERVATION_REQUIRED_FIELDS - set(row)
        observation_id = str(row.get("observation_id", "")).strip()
        if missing or not _known(observation_id) or observation_id in seen:
            _block(blockers, "invalid_longitudinal_observation", f"Invalid observation at {position}.", observation_id=observation_id)
            continue
        seen.add(observation_id)
        person = str(row.get("person_id", "")).strip()
        device = str(row.get("device_id", "")).strip()
        camera = str(row.get("camera_profile_id", "")).strip()
        source_group = str(row.get("source_group_id", "")).strip()
        period_id = str(row.get("period_id", "")).strip()
        start = _time(row.get("period_start"))
        end = _time(row.get("period_end"))
        exposure = _number(row.get("valid_monitoring_hours"))
        no_baseline_score = _number(row.get("no_personal_baseline_score"))
        no_baseline_weight = _number(row.get("no_personal_baseline_available_weight"))
        if (
            row.get("record_type") != "fall_baseline_period_features"
            or row.get("schema_version") != "fall-baseline-period-features-v1"
            or row.get("completed") is not True
            or not all(_known(value) for value in (person, device, camera, source_group, period_id))
            or start is None
            or end is None
            or not isinstance(row.get("period_start"), str)
            or not isinstance(row.get("period_end"), str)
            or end <= start
            or exposure is None
            or exposure < config["min_valid_monitoring_hours"]
            or row.get("quality_state") not in QUALITY_STATES
            or no_baseline_score is None
            or not 0 <= no_baseline_score <= 1
            or no_baseline_weight is None
            or not 0 < no_baseline_weight <= 0.84
            or not _known(row.get("fusion_config_version"))
            or not isinstance(row.get("metric_quality"), dict)
            or not row["metric_quality"]
        ):
            _block(blockers, "invalid_longitudinal_observation", f"Observation contract failed: {observation_id}.", observation_id=observation_id)
            continue
        period_key = (person, camera, period_id)
        if period_key in periods:
            _block(blockers, "duplicate_identity_period", f"Duplicate period: {period_key}.", observation_id=observation_id)
            continue
        periods.add(period_key)
        profile = profiles.get(person)
        if profile is None:
            _block(blockers, "missing_subject_profile", f"Missing profile for {person}.", person_id=person)
            continue
        features = profile["features"]
        if (
            source_group != str(features["source_group_id"])
            or device not in features["authorized_device_ids"]
            or camera not in features["authorized_camera_profile_ids"]
        ):
            _block(blockers, "identity_scope_mismatch", f"Observation is outside profile scope: {observation_id}.", observation_id=observation_id)
            continue
        output.append(
            {
                **row,
                "person_id": person,
                "device_id": device,
                "camera_profile_id": camera,
                "source_group_id": source_group,
                "period_id": period_id,
                "period_start_epoch": start,
                "period_end_epoch": end,
                "valid_monitoring_hours": exposure,
            }
        )
    return output


def _validate_outcome_links(
    observations: Sequence[dict[str, Any]],
    labels: Mapping[str, dict[str, Any]],
    reviews: Mapping[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
    blockers: list[dict[str, Any]],
) -> None:
    used: set[str] = set()
    for row in observations:
        if not _known(row.get("outcome_label_id")):
            continue
        label_id = str(row["outcome_label_id"])
        label = labels.get(label_id)
        if label is None:
            _block(blockers, "missing_outcome_label", f"Missing linked label: {label_id}.", observation_id=row["observation_id"])
            continue
        used.add(label_id)
        if label.get("subject_id") != row["person_id"] or label.get("asset_id") != row["asset_id"]:
            _block(blockers, "outcome_identity_mismatch", f"Outcome identity mismatch: {label_id}.", label_id=label_id)
        label_start = _time(label["start_time"])
        horizon = None if label_start is None else (label_start - row["period_end_epoch"]) / 3600.0
        minimum = config["prediction_horizon_hours"]["min"]
        maximum = config["prediction_horizon_hours"]["max"]
        if horizon is None or horizon <= minimum or horizon > maximum:
            _block(
                blockers,
                "outcome_not_after_observation" if horizon is not None and horizon <= minimum else "outcome_outside_prediction_horizon",
                f"Outcome {label_id} is outside the strictly future prediction horizon.",
                label_id=label_id,
            )
        accepted = config["review"]["accepted_decision"]
        accepted_reviewers = {
            str(review["reviewer_id"])
            for review in reviews.get(label_id, [])
            if review.get("decision") == accepted and _known(review.get("evidence_reference"))
        }
        required = config["review"][f"{label['label_source']}_min_reviewers"]
        if len(accepted_reviewers) < required:
            _block(
                blockers,
                "insufficient_label_reviews",
                f"Label {label_id} has {len(accepted_reviewers)} accepted reviewers; requires {required}.",
                label_id=label_id,
            )
    for label_id in sorted(set(labels) - used):
        _block(blockers, "unlinked_risk_label", f"Risk label is not linked by an observation: {label_id}.", label_id=label_id)


def _evaluation_example(observation: Mapping[str, Any], label: Mapping[str, Any], assignment: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    exposure = float(observation["valid_monitoring_hours"])
    return {
        "observation_id": observation["observation_id"],
        "person_id": observation["person_id"],
        "device_id": observation["device_id"],
        "camera_profile_id": observation["camera_profile_id"],
        "source_group_id": observation["source_group_id"],
        "scene_region": observation.get("dominant_scene_region") or observation.get("scene_region") or "unknown",
        "quality_state": observation["quality_state"],
        "valid_monitoring_hours": exposure,
        "exposure_bucket": _exposure_bucket(exposure),
        "household_day_key": (
            str(observation["source_group_id"]),
            str(observation["period_start"])[:10],
        ),
        "camera_period_key": (
            str(observation["source_group_id"]),
            str(observation["device_id"]),
            str(observation["camera_profile_id"]),
            float(assignment["period_start_epoch"]),
            float(assignment["period_end_epoch"]),
        ),
        "risk_level": int(label["risk_level"]),
        "target": int(label["risk_level"] >= config["high_risk_level_threshold"]),
        "outcome_start_epoch": _time(label["start_time"]),
        "observation_end_epoch": float(assignment["period_end_epoch"]),
    }


def _validate_prediction_rows(rows: Sequence[dict[str, Any]], split_id: str, config: Mapping[str, Any]) -> None:
    config_hashes: set[str] = set()
    for position, row in enumerate(rows, 1):
        missing = _PREDICTION_REQUIRED_FIELDS - set(row)
        score = _number(row.get("score"))
        if missing or score is None or not 0 <= score <= 1:
            raise LongitudinalDataError(f"invalid prediction at row {position}")
        if row["split_id"] != split_id:
            raise LongitudinalDataError(f"prediction split_id mismatch at row {position}")
        if row["quality_state"] not in QUALITY_STATES:
            raise LongitudinalDataError(f"invalid prediction quality_state at row {position}")
        if row["baseline_state"] not in BASELINE_STATES or not _known(
            row.get("model_version")
        ):
            raise LongitudinalDataError(f"invalid prediction baseline state or model at row {position}")
        config_hash = str(row["config_hash"])
        if len(config_hash) != 64 or any(character not in "0123456789abcdef" for character in config_hash.lower()):
            raise LongitudinalDataError(f"invalid prediction config_hash at row {position}")
        config_hashes.add(config_hash.lower())
    if len(config_hashes) != 1:
        raise LongitudinalDataError("all ablation predictions must share one config_hash")


def _validate_evaluation_split(
    metadata: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    partition: str,
) -> str:
    protocol_hash = _value_hash(config)
    split_id = str(metadata.get("split_id", ""))
    status = metadata.get("status")
    if (
        metadata.get("schema_version") != "fall-baseline-longitudinal-split-v1"
        or metadata.get("split_name") != "longitudinal_baseline_v1"
        or metadata.get("task_type") != "longitudinal_baseline"
        or not split_id
        or metadata.get("protocol_sha256") != protocol_hash
        or metadata.get("protocol_status") != config["protocol_status"]
        or status not in {"ready", "frozen"}
    ):
        raise LongitudinalDataError("split metadata does not match evaluation protocol")
    expected_status = "frozen" if config["protocol_status"] == "frozen" else "ready"
    if status != expected_status:
        raise LongitudinalDataError(
            "split metadata status does not match protocol freeze state"
        )
    split_ids = {str(row.get("split_id", "")) for row in assignments}
    if split_ids != {split_id}:
        raise LongitudinalDataError("assignments do not match split_id")
    if metadata.get("assignments_sha256") != _rows_hash(assignments):
        raise LongitudinalDataError("assignments hash does not match split metadata")
    root_assignments = [
        {key: value for key, value in row.items() if key != "split_id"}
        for row in assignments
    ]
    root_assignments.sort(key=_assignment_sort_key)
    root_metadata = {
        **dict(metadata),
        "split_id": None,
        "split_sha256": None,
        "assignments_sha256": _rows_hash(root_assignments),
    }
    expected_root = _value_hash(
        {"metadata": root_metadata, "assignments": root_assignments}
    )
    if (
        metadata.get("split_sha256") != expected_root
        or split_id != f"longitudinal_baseline_v1:sha256:{expected_root[:16]}"
    ):
        raise LongitudinalDataError("split root hash does not match metadata and assignments")
    return split_id


def _partition_input_hashes(
    assignments: Sequence[Mapping[str, Any]],
    observations_by_id: Mapping[str, Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for partition in PARTITIONS:
        partition_assignments = [
            row for row in assignments if row.get("partition") == partition
        ]
        observation_rows = [
            observations_by_id[str(row["observation_id"])]
            for row in partition_assignments
            if str(row["observation_id"]) in observations_by_id
        ]
        scoring_ids = {
            str(row["observation_id"])
            for row in partition_assignments
            if row.get("role") == "scoring"
        }
        label_ids = {
            str(observations_by_id[observation_id].get("outcome_label_id", ""))
            for observation_id in scoring_ids
            if observation_id in observations_by_id
        }
        label_rows = [labels_by_id[label_id] for label_id in sorted(label_ids) if label_id in labels_by_id]
        output[partition] = {
            "observations": _rows_hash(observation_rows),
            "risk_labels": _rows_hash(label_rows),
        }
    return output


def _validate_test_release_ack(
    acknowledgement: Mapping[str, Any] | None,
    *,
    split_id: str,
    config: Mapping[str, Any],
    protocol_hash: str,
    predictions_hash: str,
    labels_hash: str,
    code_commit: str | None,
) -> None:
    if acknowledgement is None:
        raise LongitudinalDataError("test evaluation requires explicit test release acknowledgement")
    expected = {
        "partition": "test",
        "split_id": split_id,
        "protocol_version": config["protocol_version"],
        "protocol_sha256": protocol_hash,
        "blind_test_governance_confirmed": True,
        "prediction_sha256": predictions_hash,
        "labels_sha256": labels_hash,
        "code_commit": code_commit,
    }
    for field, value in expected.items():
        if acknowledgement.get(field) != value:
            raise LongitudinalDataError(f"test release acknowledgement mismatch: {field}")
    for field in ("authorization_id", "authorized_by_role", "evaluation_run_id"):
        if not _known(acknowledgement.get(field)):
            raise LongitudinalDataError(f"test release acknowledgement requires {field}")
    if not isinstance(code_commit, str) or len(code_commit) != 40:
        raise LongitudinalDataError("test evaluation requires the current Git commit")


def _binary_metrics(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    threshold = config["score_threshold"]
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = float(row["score"]) >= threshold
        actual = bool(row["target"])
        tp += int(predicted and actual)
        fp += int(predicted and not actual)
        tn += int(not predicted and not actual)
        fn += int(not predicted and actual)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    household_days = {
        (row.get("_bootstrap_draw_index"), *row["household_day_key"])
        for row in rows
    }
    camera_exposure: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = (row.get("_bootstrap_draw_index"), *row["camera_period_key"])
        camera_exposure[key] = max(
            camera_exposure.get(key, 0.0), float(row["valid_monitoring_hours"])
        )
    total_hours = sum(camera_exposure.values())
    persons = {str(row["person_id"]) for row in rows}
    observed_days = len(rows)
    positive_lead_hours = [
        max(0.0, (float(row["outcome_start_epoch"]) - float(row["observation_end_epoch"])) / 3600.0)
        for row in rows
        if row["target"] and float(row["score"]) >= threshold
    ]
    return {
        "sample_count": len(rows),
        "subject_count": len(persons),
        "positive_count": sum(int(row["target"]) for row in rows),
        "negative_count": sum(1 - int(row["target"]) for row in rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "pr_auc": round(_average_precision(rows), 6),
        "brier_score": round(sum((float(row["score"]) - int(row["target"])) ** 2 for row in rows) / len(rows), 6),
        "expected_calibration_error": round(_ece(rows), 6),
        "false_positives_per_observed_day": round(_ratio(fp, observed_days), 6),
        "household_day_count": len(household_days),
        "camera_hour_count": round(total_hours, 6),
        "false_positives_per_household_day": round(
            _ratio(fp, len(household_days)), 6
        ),
        "false_positives_per_camera_hour": round(_ratio(fp, total_hours), 6),
        "mean_detected_lead_hours": round(sum(positive_lead_hours) / len(positive_lead_hours), 6) if positive_lead_hours else None,
    }


def _average_precision(rows: Sequence[Mapping[str, Any]]) -> float:
    positives = sum(int(row["target"]) for row in rows)
    if positives == 0:
        return 0.0
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["score"])].append(row)
    true_positives = 0
    total = 0
    previous_recall = 0.0
    area = 0.0
    for score in sorted(grouped, reverse=True):
        bucket = grouped[score]
        total += len(bucket)
        true_positives += sum(int(row["target"]) for row in bucket)
        recall = true_positives / positives
        precision = true_positives / total
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _ece(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            row for row in rows
            if lower <= float(row["score"]) < upper or (index == bins - 1 and float(row["score"]) == 1.0)
        ]
        if not bucket:
            continue
        confidence = sum(float(row["score"]) for row in bucket) / len(bucket)
        outcome = sum(int(row["target"]) for row in bucket) / len(bucket)
        error += len(bucket) / len(rows) * abs(confidence - outcome)
    return error


def _cluster_bootstrap(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["person_id"])].append(row)
    cluster_ids = sorted(clusters)
    metrics = (
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
        "false_positives_per_household_day",
        "false_positives_per_camera_hour",
        "mean_detected_lead_hours",
    )
    if not cluster_ids or config["bootstrap_iterations"] == 0:
        return {metric: {"low": None, "high": None} for metric in metrics}
    randomizer = random.Random(config["bootstrap_seed"])
    values: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(config["bootstrap_iterations"]):
        sampled: list[Mapping[str, Any]] = []
        draw_counts: Counter[str] = Counter()
        for _cluster_index in cluster_ids:
            selected_cluster = randomizer.choice(cluster_ids)
            copy_index = draw_counts[selected_cluster]
            draw_counts[selected_cluster] += 1
            sampled.extend(
                {**row, "_bootstrap_draw_index": copy_index}
                for row in clusters[selected_cluster]
            )
        result = _binary_metrics(sampled, {**config, "bootstrap_iterations": 0})
        for metric in metrics:
            if result[metric] is not None:
                values[metric].append(float(result[metric]))
    return {
        metric: (
            {
                "low": round(_percentile(items, 0.025), 6),
                "high": round(_percentile(items, 0.975), 6),
            }
            if items
            else {"low": None, "high": None}
        )
        for metric, items in values.items()
    }


def _stratified_metrics(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = (
        "person_id", "device_id", "camera_profile_id", "source_group_id", "scene_region", "quality_state", "exposure_bucket", "baseline_state"
    )
    output: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        output[dimension] = {
            value: _binary_metrics(group_rows, config)
            for value, group_rows in sorted(groups.items())
        }
    return output


def _failure_rows(rows: Sequence[Mapping[str, Any]], variant: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        predicted = float(row["score"]) >= config["score_threshold"]
        actual = bool(row["target"])
        if predicted == actual:
            continue
        failures.append(
            {
                "variant": variant,
                "failure_type": "false_positive" if predicted else "false_negative",
                "observation_id": row["observation_id"],
                "person_id": row["person_id"],
                "device_id": row["device_id"],
                "camera_profile_id": row["camera_profile_id"],
                "source_group_id": row["source_group_id"],
                "scene_region": row["scene_region"],
                "quality_state": row["quality_state"],
                "valid_monitoring_hours": row["valid_monitoring_hours"],
                "risk_level": row["risk_level"],
                "score": row["score"],
            }
        )
    return failures


def _evaluation_report(result: LongitudinalEvaluationResult, metadata: Mapping[str, Any]) -> str:
    data_status = str(metadata.get("data_status", "unspecified"))
    lines = [
        "# Longitudinal Personal Baseline Ablation",
        "",
        f"- Partition: `{result.partition}`",
        f"- Protocol: `{result.protocol_version}` (`{result.protocol_status}`)",
        f"- Split: `{result.split_id}`",
        f"- Scoring observations: {result.sample_count}",
        f"- Data status: `{data_status}`",
        "",
    ]
    if data_status.lower() == "synthetic":
        lines.extend([
            "> Synthetic results only verify evaluator behaviour. They are not evidence of real-person effectiveness.",
            "",
        ])
    lines.extend([
        "| Variant | Precision | Recall | F1 | PR-AUC | FP/household day | FP/camera hour |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, metrics in result.metrics_by_variant.items():
        lines.append(
            f"| `{variant}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['pr_auc']:.4f} | {metrics['false_positives_per_household_day']:.4f} | "
            f"{metrics['false_positives_per_camera_hour']:.4f} |"
        )
    lines.extend(["", f"Failure cases: {len(result.failure_cases)}", ""])
    return "\n".join(lines)


def _allocate_partition(source_group_id: str, seed: str, ratios: Mapping[str, float]) -> str:
    digest = hashlib.sha256(f"{seed}\0{source_group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for partition in PARTITIONS:
        cumulative += float(ratios[partition])
        if value < cumulative or partition == PARTITIONS[-1]:
            return partition
    raise AssertionError("unreachable partition allocation")


def _unique_index(rows: Iterable[Mapping[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(rows, 1):
        row = dict(raw)
        value = str(row.get(key, ""))
        if not value or value in output:
            raise LongitudinalDataError(f"invalid or duplicate {name} {key} at row {position}")
        output[value] = row
    return output


def _atomic_bundle_write(directory: Path, files: Mapping[str, str], *, overwrite: bool) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.staging.", dir=directory.parent))
    backup: Path | None = None
    try:
        for relative, content in files.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if os.path.lexists(directory):
            if not overwrite:
                raise FileExistsError(f"output exists: {directory}")
            backup = Path(tempfile.mkdtemp(prefix=f".{directory.name}.backup.", dir=directory.parent))
            backup.rmdir()
            os.replace(directory, backup)
        os.replace(staging, directory)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except BaseException:
        if backup is not None and not os.path.lexists(directory):
            os.replace(backup, directory)
            backup = None
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def _block(blockers: list[dict[str, Any]], code: str, message: str, **context: Any) -> None:
    blockers.append({"code": code, "message": message, **context})


def _dedupe_blockers(blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for blocker in blockers:
        row = dict(blocker)
        unique[json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))] = row
    return sorted(unique.values(), key=lambda row: (row["code"], row.get("person_id", ""), row.get("label_id", ""), row["message"]))


def _known(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in _UNKNOWN


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_known(item) for item in value)


def _finite(value: Any) -> bool:
    return _number(value) is not None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _time(value: Any) -> float | None:
    number = _number(value)
    if number is not None:
        return number
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _exposure_bucket(hours: float) -> str:
    if hours < 4:
        return "lt_4h"
    if hours < 8:
        return "4_to_8h"
    return "gte_8h"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _value_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _rows_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    canonical_rows = sorted(_canonical(dict(row)) for row in rows)
    return hashlib.sha256(b"\n".join(canonical_rows)).hexdigest()


def _assignment_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["partition"],
        row["source_group_id"],
        row["person_id"],
        row["camera_profile_id"],
        row["period_start_epoch"],
        row["observation_id"],
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    materialized = [json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in rows]
    return "\n".join(materialized) + ("\n" if materialized else "")
