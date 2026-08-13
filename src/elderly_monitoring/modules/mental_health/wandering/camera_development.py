"""Authorized M0-CAM development control path and pure evaluator primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterInput,
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    group_observations,
    load_camera_inputs,
    validate_media_sidecar,
    validation_scope_for_authorization_status,
)
from elderly_monitoring.modules.mental_health.wandering.camera_dataset import (
    CameraDatasetError,
    load_camera_collection_config,
    validate_authorization_receipt,
    validate_collection_manifest,
    validate_episode_annotations,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode import (
    aggregate_episode_candidates,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
    prepare_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    PrimaryCameraInferenceError,
    _camera_tensors,
    _configure_cpu_runtime,
    _load_feature_stats,
    _preflight_active_source_identity,
    _verify_preprocessing_roots,
    _verify_project_descriptor,
    load_primary_camera_runtime,
    predict_primary_camera_window,
)
from elderly_monitoring.modules.mental_health.wandering.camera_qc import (
    CameraQCError,
    run_camera_qc,
)
from elderly_monitoring.modules.mental_health.wandering.preprocessing import (
    load_preprocessing_config,
)
from elderly_monitoring.modules.mental_health.wandering.release import (
    CandidateRuntime,
    load_candidate_from_manifest,
)


DEVELOPMENT_CONFIG_SCHEMA_VERSION = "wandering-camera-development-config-v1"
DEVELOPMENT_EXECUTION_SCHEMA_VERSION = "wandering-camera-development-execution-v1"
DEVELOPMENT_STATUS = "wandering_m0cam_development_entry_hardened_waiting_c0_c1"
FOUR_CLASSES = ("direct", "pacing", "lapping", "random")
RUNTIME_STATUSES = ("ready", "unavailable", "inference_error", "uncertain")
EPISODE_SCOPE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
    "track_id",
    "participant_id",
    "session_id",
    "camera_setup_id",
    "clock_domain_id",
)


class CameraDevelopmentError(ValueError):
    """Authorized camera-development contract failed closed."""


def load_camera_development_config(path: str | Path) -> dict[str, Any]:
    """Load exact identities while keeping all scientific policies unfrozen."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraDevelopmentError("cannot read camera development config") from exc
    if value != _expected_development_config():
        raise CameraDevelopmentError("camera development config fields or identities have drifted")
    return value


def match_episodes_one_to_one(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    matching_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Deterministic temporal one-to-one matching with an explicit policy."""

    policy = _matching_policy(matching_policy)
    prediction_ids = _unique_ids(predictions, "prediction_id")
    annotation_ids = _unique_ids(annotations, "annotation_id")
    candidates: list[dict[str, Any]] = []
    for prediction in predictions:
        p_start, p_end = _interval(prediction)
        for annotation in annotations:
            if not _same_episode_scope(prediction, annotation):
                continue
            a_start, a_end = _interval(annotation)
            overlap = max(0.0, min(p_end, a_end) - max(p_start, a_start))
            union = max(p_end, a_end) - min(p_start, a_start)
            temporal_iou = overlap / union if union > 0 else 0.0
            onset_delta = abs(p_start - a_start)
            boundary_delta = onset_delta + abs(p_end - a_end)
            if (
                overlap > 0.0
                and
                temporal_iou >= policy["minimum_temporal_iou"]
                and onset_delta <= policy["maximum_onset_delta_sec"]
            ):
                candidates.append(
                    {
                        "temporal_iou": temporal_iou,
                        "boundary_delta": boundary_delta,
                        "prediction_id": str(prediction["prediction_id"]),
                        "annotation_id": str(annotation["annotation_id"]),
                    }
                )
    selected = _minimum_cost_maximum_matching(candidates)
    used_predictions = {row["prediction_id"] for row in selected}
    used_annotations = {row["annotation_id"] for row in selected}
    matches = [
        {
            "prediction_id": row["prediction_id"],
            "annotation_id": row["annotation_id"],
            "temporal_iou": float(row["temporal_iou"]),
        }
        for row in selected
    ]
    matches.sort(key=lambda row: (row["prediction_id"], row["annotation_id"]))
    return {
        "matching_policy": policy,
        "optimization_order": [
            "maximum_match_count",
            "maximum_total_temporal_iou",
            "minimum_total_onset_offset_delta",
            "stable_prediction_annotation_id",
        ],
        "matches": matches,
        "unmatched_prediction_ids": sorted(prediction_ids - used_predictions),
        "unmatched_annotation_ids": sorted(annotation_ids - used_annotations),
    }


def _minimum_cost_maximum_matching(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Solve bipartite matching with additive lexicographic costs.

    Flow continues until no augmenting path remains, which fixes maximum
    cardinality first.  Each shortest augmenting path then minimizes the tuple
    ``(-IoU, onset+offset delta, stable pair rank)`` across that cardinality.
    Residual edges allow earlier local choices to be replaced by a globally
    better matching.
    """

    ordered = sorted(
        (dict(row) for row in candidates),
        key=lambda row: (row["prediction_id"], row["annotation_id"]),
    )
    if not ordered:
        return []
    source = ("source",)
    sink = ("sink",)
    prediction_nodes = {
        row["prediction_id"]: ("prediction", row["prediction_id"]) for row in ordered
    }
    annotation_nodes = {
        row["annotation_id"]: ("annotation", row["annotation_id"]) for row in ordered
    }
    graph: dict[tuple[str, ...], list[dict[str, Any]]] = {
        source: [],
        sink: [],
        **{node: [] for node in prediction_nodes.values()},
        **{node: [] for node in annotation_nodes.values()},
    }
    zero_cost = (0.0, 0.0, 0)
    for prediction_id, node in sorted(prediction_nodes.items()):
        _add_flow_edge(graph, source, node, zero_cost)
    for annotation_id, node in sorted(annotation_nodes.items()):
        _add_flow_edge(graph, node, sink, zero_cost)

    candidate_edges: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rank, row in enumerate(ordered):
        edge = _add_flow_edge(
            graph,
            prediction_nodes[row["prediction_id"]],
            annotation_nodes[row["annotation_id"]],
            (-float(row["temporal_iou"]), float(row["boundary_delta"]), rank),
        )
        candidate_edges.append((row, edge))

    nodes = sorted(graph, key=repr)
    while True:
        distances: dict[tuple[str, ...], tuple[float, float, int]] = {
            source: zero_cost
        }
        previous: dict[tuple[str, ...], tuple[tuple[str, ...], int]] = {}
        for _ in range(max(0, len(nodes) - 1)):
            changed = False
            for node in nodes:
                if node not in distances:
                    continue
                for edge_index, edge in enumerate(graph[node]):
                    if edge["capacity"] <= 0:
                        continue
                    destination = edge["to"]
                    proposed = _add_cost(distances[node], edge["cost"])
                    if destination not in distances or proposed < distances[destination]:
                        distances[destination] = proposed
                        previous[destination] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if sink not in previous:
            break
        node = sink
        while node != source:
            prior, edge_index = previous[node]
            edge = graph[prior][edge_index]
            edge["capacity"] -= 1
            graph[node][edge["reverse"]]["capacity"] += 1
            node = prior

    return [dict(row) for row, edge in candidate_edges if edge["capacity"] == 0]


def _add_flow_edge(
    graph: dict[tuple[str, ...], list[dict[str, Any]]],
    source: tuple[str, ...],
    destination: tuple[str, ...],
    cost: tuple[float, float, int],
) -> dict[str, Any]:
    forward = {
        "to": destination,
        "reverse": len(graph[destination]),
        "capacity": 1,
        "cost": cost,
    }
    reverse = {
        "to": source,
        "reverse": len(graph[source]),
        "capacity": 0,
        "cost": (-cost[0], -cost[1], -cost[2]),
    }
    graph[source].append(forward)
    graph[destination].append(reverse)
    return forward


def _add_cost(
    left: tuple[float, float, int], right: tuple[float, float, int]
) -> tuple[float, float, int]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def evaluate_shape_counts(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return raw shape confusion/counts; do not derive F1 or policy claims."""

    four_confusion = {truth: {pred: 0 for pred in FOUR_CLASSES} for truth in FOUR_CLASSES}
    four_abstention = {truth: 0 for truth in FOUR_CLASSES}
    four_miss = {truth: 0 for truth in FOUR_CLASSES}
    binary_classes = ("direct_or_non_wandering", "wandering_like")
    binary_confusion = {truth: {pred: 0 for pred in binary_classes} for truth in binary_classes}
    binary_abstention = {truth: 0 for truth in binary_classes}
    binary_miss = {truth: 0 for truth in binary_classes}
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != {"truth", "prediction"}:
            raise CameraDevelopmentError("shape pair fields are invalid")
        truth = pair["truth"]
        prediction = pair["prediction"]
        if truth not in FOUR_CLASSES or prediction not in (*FOUR_CLASSES, "uncertain", "miss"):
            raise CameraDevelopmentError("shape pair class is invalid")
        binary_truth = "direct_or_non_wandering" if truth == "direct" else "wandering_like"
        if prediction == "uncertain":
            four_abstention[truth] += 1
            binary_abstention[binary_truth] += 1
            continue
        if prediction == "miss":
            four_miss[truth] += 1
            binary_miss[binary_truth] += 1
            continue
        four_confusion[truth][prediction] += 1
        binary_prediction = (
            "direct_or_non_wandering" if prediction == "direct" else "wandering_like"
        )
        binary_confusion[binary_truth][binary_prediction] += 1
    return {
        "four_class": {
            "class_order": list(FOUR_CLASSES),
            "count": len(pairs),
            "confusion": four_confusion,
            "abstention_by_truth": four_abstention,
            "miss_by_truth": four_miss,
        },
        "binary": {
            "class_order": list(binary_classes),
            "count": len(pairs),
            "confusion": binary_confusion,
            "abstention_by_truth": binary_abstention,
            "miss_by_truth": binary_miss,
        },
    }


def evaluate_purposeful_hard_negatives(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report diagnostic shape activations without inventing alert metrics."""

    eligible = [
        row
        for row in rows
        if row.get("evaluation_role") == "purposeful_hard_negative"
        and row.get("purpose_context") == "purposeful"
    ]
    if any(row.get("observable_pattern") not in {"pacing", "lapping", "random"} for row in eligible):
        raise CameraDevelopmentError("purposeful hard negative lost its wandering-like shape")
    return {
        "eligible_count": len(eligible),
        "shape_pacing_prediction_count": sum(row.get("prediction") == "pacing" for row in eligible),
        "shape_wandering_like_prediction_count": sum(
            row.get("prediction") in {"pacing", "lapping", "random"} for row in eligible
        ),
        "alert_metrics_available": False,
    }


def build_coverage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count ready/unavailable/inference_error/uncertain without collapsing states."""

    counts = Counter(row.get("prediction_status") for row in rows)
    unknown = set(counts) - set(RUNTIME_STATUSES)
    if unknown:
        raise CameraDevelopmentError(f"unknown prediction status: {sorted(unknown)}")
    return {"total": len(rows), **{status: counts[status] for status in RUNTIME_STATUSES}}


def union_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return a deterministic union of finite half-open intervals."""

    normalized = sorted(_numeric_interval(item) for item in intervals)
    output: list[tuple[float, float]] = []
    for start, end in normalized:
        if not output or start > output[-1][1]:
            output.append((start, end))
        else:
            output[-1] = (output[-1][0], max(output[-1][1], end))
    return output


def subtract_intervals(
    bases: Iterable[tuple[float, float]], cuts: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Subtract the union of cuts from the union of base intervals."""

    remaining: list[tuple[float, float]] = []
    normalized_cuts = union_intervals(cuts)
    for start, end in union_intervals(bases):
        pieces = [(start, end)]
        for cut_start, cut_end in normalized_cuts:
            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if cut_start > piece_start:
                    next_pieces.append((piece_start, cut_start))
                if cut_end < piece_end:
                    next_pieces.append((cut_end, piece_end))
            pieces = next_pieces
        remaining.extend(pieces)
    return remaining


def compute_eligible_negative_person_hours(
    *,
    participant_present: Sequence[Mapping[str, Any]],
    ordinary_negative: Sequence[Mapping[str, Any]],
    positive: Sequence[Mapping[str, Any]],
    truth_uncertain_or_excluded: Sequence[Mapping[str, Any]],
    qc_unavailable: Sequence[Mapping[str, Any]],
    tracking_unavailable: Sequence[Mapping[str, Any]],
    inference_error: Sequence[Mapping[str, Any]],
    tracklet_participant_bindings: Sequence[Mapping[str, Any]],
    clock_alignments: Sequence[Mapping[str, Any]],
    prediction_uncertain: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute only explicitly truth-decidable negative person-time.

    Prediction uncertainty is intentionally accepted for coverage bookkeeping
    but never subtracted from the denominator.
    """

    if not tracklet_participant_bindings:
        return _negative_not_computable("missing_explicit_tracklet_participant_binding")
    if not participant_present or not ordinary_negative:
        return _negative_not_computable("missing_presence_or_truth_decidable_negative_intervals")
    group_fields = ("participant_id", "session_id", "clock_domain_id")
    try:
        bound_groups = {
            tuple(_required_group_token(row, field) for field in group_fields)
            for row in tracklet_participant_bindings
        }
    except CameraDevelopmentError:
        return _negative_not_computable("missing_explicit_tracklet_participant_binding")
    if not bound_groups:
        return _negative_not_computable("missing_explicit_tracklet_participant_binding")

    domains_by_session: dict[tuple[str, str], set[str]] = {}
    for participant_id, session_id, clock_domain_id in bound_groups:
        domains_by_session.setdefault((participant_id, session_id), set()).add(clock_domain_id)
    if any(len(domains) != 1 for domains in domains_by_session.values()):
        return _negative_not_computable("session_clock_domains_not_explicitly_aligned")

    alignment_by_key: dict[tuple[str, str], str] = {}
    for row in clock_alignments:
        try:
            key = (
                _required_group_token(row, "session_id"),
                _required_group_token(row, "clock_domain_id"),
            )
        except CameraDevelopmentError:
            return _negative_not_computable("camera_clock_not_aligned")
        status = row.get("status")
        if key in alignment_by_key or status not in {"aligned", "single_camera"}:
            return _negative_not_computable("camera_clock_not_aligned")
        alignment_by_key[key] = str(status)
    if any(
        (session_id, clock_domain_id) not in alignment_by_key
        for _, session_id, clock_domain_id in bound_groups
    ):
        return _negative_not_computable("camera_clock_not_aligned")

    total_seconds = 0.0
    group_results: list[dict[str, Any]] = []
    for group in sorted(bound_groups):
        participant_id, session_id, clock_domain_id = group
        presence = _rows_to_group_intervals(participant_present, group)
        ordinary = _rows_to_group_intervals(ordinary_negative, group)
        if not presence or not ordinary:
            return _negative_not_computable("participant_missing_presence_or_ordinary_negative")
        truth_decidable = _intersect_intervals(presence, ordinary)
        removal_groups = {
            "positive": _rows_to_group_intervals(positive, group),
            "truth_uncertain_or_excluded": _rows_to_group_intervals(
                truth_uncertain_or_excluded, group
            ),
            "qc_unavailable": _rows_to_group_intervals(qc_unavailable, group),
            "tracking_unavailable": _rows_to_group_intervals(
                tracking_unavailable, group
            ),
            "inference_error": _rows_to_group_intervals(inference_error, group),
        }
        removals = [item for values in removal_groups.values() for item in values]
        eligible = subtract_intervals(truth_decidable, removals)
        seconds = sum(end - start for start, end in eligible)
        total_seconds += seconds
        group_results.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "clock_domain_id": clock_domain_id,
                "eligible_negative_seconds": float(seconds),
                "excluded_seconds_by_reason": {
                    name: float(sum(end - start for start, end in union_intervals(values)))
                    for name, values in removal_groups.items()
                },
            }
        )
    return {
        "status": "computed",
        "eligible_negative_seconds": float(total_seconds),
        "eligible_negative_person_hours": float(total_seconds / 3600.0),
        "participant_session_clock_domain_count": len(group_results),
        "groups": group_results,
        "prediction_uncertain_interval_count": len(prediction_uncertain),
        "prediction_uncertain_subtracted": False,
        "unlabeled_remainder_treated_as_negative": False,
        "identity_inferred_from_track_id": False,
    }


def build_stage_timings(**values: float) -> dict[str, Any]:
    """Validate and total the fixed end-to-end stage timing fields."""

    ordered = (
        "receipt",
        "collection",
        "source_preflight",
        "tracking_annotation_cohort",
        "candidate_load",
        "qc_preprocess",
        "forward",
        "evaluator",
        "artifact_write",
    )
    if set(values) != {f"{name}_ms" for name in ordered}:
        raise CameraDevelopmentError("stage timing fields are incomplete")
    stages: dict[str, float] = {}
    for name in ordered:
        value = values[f"{name}_ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CameraDevelopmentError("stage timing must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise CameraDevelopmentError("stage timing must be finite and nonnegative")
        stages[name] = number
    return {"stages": stages, "total_ms": float(sum(stages.values()))}


def run_authorized_camera_development(
    *,
    project_root: str | Path,
    config_path: str | Path,
    receipt_path: str | Path,
    collection_path: str | Path,
    tracking_path: str | Path,
    media_sidecar_path: str | Path,
    candidate_manifest_path: str | Path,
    expected_manifest_sha256: str,
    output_dir: str | Path,
    mode: str = "engineering_smoke",
    annotations_path: str | Path | None = None,
    evaluation_policy: Mapping[str, Any] | None = None,
    evaluation_policy_path: str | Path | None = None,
    _test_hooks: Mapping[str, Callable[..., Any]] | None = None,
    _candidate_loader: Callable[..., CandidateRuntime] | None = None,
    _test_fixture: bool = False,
) -> dict[str, Any]:
    """Run the independent fail-closed development entry.

    Private hooks are limited to zero-call failure assertions.  Successful
    temporary fixture runs execute every production validator and may replace
    only the manifest-bound candidate loader.  The production CLI exposes none
    of these controls.
    """

    hooks = dict(_test_hooks or {})
    output = Path(output_dir)
    _preflight_fresh_output(output)
    config = load_camera_development_config(config_path)
    root = Path(project_root).resolve(strict=True)
    _bind_repository_path(root, config_path, "configs/modules/wandering_camera_development_v1.yaml")
    if mode not in {"engineering_smoke", "labeled_evaluation"}:
        raise CameraDevelopmentError("development mode is invalid")
    if _candidate_loader is not None and not _test_fixture:
        raise CameraDevelopmentError("candidate loader injection is test-fixture-only")
    if _test_fixture:
        try:
            output.resolve(strict=False).relative_to(root)
        except ValueError:
            pass
        else:
            raise CameraDevelopmentError("test fixture output must stay outside the repository")

    receipt_started = time.perf_counter_ns()
    receipt = _read_json(Path(receipt_path), "authorization receipt")
    try:
        collection_config = load_camera_collection_config(
            root / config["collection_config"]["path"]
        )
        receipt_summary = validate_authorization_receipt(
            receipt,
            collection_config,
            operation="run_development",
        )
    except CameraDatasetError as exc:
        raise CameraDevelopmentError("authorization receipt preflight failed closed") from exc
    receipt_ms = _elapsed_ms(receipt_started)
    if evaluation_policy is not None and evaluation_policy_path is not None:
        raise CameraDevelopmentError("evaluation policy must use one input source")
    if evaluation_policy_path is not None:
        evaluation_policy = _read_json(Path(evaluation_policy_path), "evaluation policy")
    if mode == "labeled_evaluation":
        _validate_explicit_evaluation_policy(evaluation_policy)

    collection_started = time.perf_counter_ns()
    try:
        raw_collection = _call_or_default(
            hooks,
            "read_collection",
            lambda: _read_json(Path(collection_path), "collection"),
        )
        collection = validate_collection_manifest(raw_collection, collection_config)
        scope = _collection_scope(collection)
        try:
            receipt_summary = validate_authorization_receipt(
                receipt,
                collection_config,
                operation="run_development",
                **scope,
            )
        except CameraDatasetError as exc:
            raise CameraDevelopmentError(
                "collection authorization scope mismatch"
            ) from exc
        if collection["authorization_receipt_id"] != receipt_summary["receipt_id"]:
            raise CameraDevelopmentError("collection receipt reference mismatch")
        if collection["dataset_role"] != "development":
            raise CameraDevelopmentError("sealed camera collection is forbidden in development")
        camera_config_path = _verify_project_descriptor(root, config["camera_chain"], "camera chain config")
        camera_config = load_camera_config(camera_config_path)
        raw_sidecar = _call_or_default(
            hooks,
            "read_sidecar",
            lambda: _read_json(Path(media_sidecar_path), "media sidecar"),
        )
        media = validate_media_sidecar(raw_sidecar, camera_config)
        _bind_media_to_collection(media, collection)
        if media["authorization_status"] not in {
            "authorized_camera_engineering_smoke",
            "authorized_camera_labeled_evaluation",
        }:
            raise CameraDevelopmentError("production development rejects synthetic media")
        if (
            mode == "labeled_evaluation"
            and media["authorization_status"] != "authorized_camera_labeled_evaluation"
        ):
            raise CameraDevelopmentError(
                "labeled evaluation requires labeled-evaluation authorization"
            )
        validation_scope = validation_scope_for_authorization_status(
            media["authorization_status"]
        )
    except (CameraDatasetError, CameraAdapterError, CameraInferenceError, OSError, ValueError) as exc:
        if isinstance(exc, CameraDevelopmentError):
            raise
        raise CameraDevelopmentError("collection/session authorization binding failed closed") from exc
    collection_ms = _elapsed_ms(collection_started)

    source_started = time.perf_counter_ns()
    manifest_path = Path(candidate_manifest_path).resolve(strict=False)
    expected_manifest_path = (root / config["candidate"]["manifest_path"]).resolve(strict=False)
    if manifest_path != expected_manifest_path:
        raise CameraDevelopmentError("candidate manifest must use the fixed repository path")
    if expected_manifest_sha256 != config["candidate"]["manifest_sha256"]:
        raise CameraDevelopmentError("external candidate manifest SHA-256 does not match fixed config")
    try:
        source_identity = _call_or_default(
            hooks,
            "source_preflight",
            lambda: _preflight_active_source_identity(
                root=root,
                manifest_path=manifest_path,
                expected_manifest_sha256=expected_manifest_sha256,
                config=config,
            ),
        )
        preprocessing_paths = None
        preprocessing_config = None
        feature_stats = None
        preprocessing_paths = _verify_preprocessing_roots(root, camera_config)
        preprocessing_config = load_preprocessing_config(
            preprocessing_paths["preprocessing_config"]
        )
        feature_stats = _load_feature_stats(preprocessing_paths, camera_config)
    except (PrimaryCameraInferenceError, CameraInferenceError, OSError, ValueError) as exc:
        raise CameraDevelopmentError("source/candidate identity preflight failed closed") from exc
    source_preflight_ms = _elapsed_ms(source_started)

    # Tracking, annotations, and grouped cohort are intentionally unread before
    # receipt, collection/sidecar binding, and source identity all pass.
    tracking_started = time.perf_counter_ns()
    try:
        adapter = _call_or_default(
            hooks,
            "read_tracking",
            lambda: load_camera_inputs(tracking_path, media_sidecar_path, camera_config),
        )
        annotations: Sequence[Mapping[str, Any]] = []
        c3_bindings: Mapping[tuple[str, str, str, str, str, int], Mapping[str, Any]] = {}
        if mode == "labeled_evaluation":
            if annotations_path is None:
                raise CameraDevelopmentError("labeled evaluation requires annotations")
            raw_annotations = _call_or_default(
                hooks,
                "read_annotations",
                lambda: _read_jsonl(Path(annotations_path), "annotations"),
            )
            annotations = validate_episode_annotations(raw_annotations, collection)
            _validate_labeled_c3(collection, annotations)
            c3_bindings = _bind_actual_tracking_to_c3(adapter, collection, annotations)
    except (CameraDatasetError, CameraAdapterError, OSError, ValueError) as exc:
        if isinstance(exc, CameraDevelopmentError):
            raise
        raise CameraDevelopmentError("tracking/annotation/cohort validation failed closed") from exc
    tracking_annotation_cohort_ms = _elapsed_ms(tracking_started)

    candidate_started = time.perf_counter_ns()
    try:
        runtime = _call_or_default(
            hooks,
            "load_candidate",
            lambda: load_primary_camera_runtime(
                config=config,
                manifest_path=manifest_path,
                expected_manifest_sha256=expected_manifest_sha256,
                candidate_loader=_candidate_loader or load_candidate_from_manifest,
            ),
        )
    except (PrimaryCameraInferenceError, OSError, ValueError) as exc:
        raise CameraDevelopmentError("fixed candidate loader failed closed") from exc
    candidate_load_ms = _elapsed_ms(candidate_started)

    qc_started = time.perf_counter_ns()
    try:
        qc = run_camera_qc(adapter, camera_config)
        prepared_by_id: dict[str, Mapping[str, Any]] = {}
        for item in qc.ready_inputs:
            row = prepare_camera_window(
                item, camera_config, preprocessing_config, feature_stats
            )
            _camera_tensors(row)
            prepared_by_id[str(row["window_id"])] = row
        prepared = [
            dict(prepared_by_id.get(str(row["window_id"]), row))
            for row in qc.window_records
        ]
    except (CameraQCError, CameraInferenceError, KeyError, TypeError, ValueError) as exc:
        raise CameraDevelopmentError("camera QC/preprocessing failed closed") from exc
    qc_preprocess_ms = _elapsed_ms(qc_started)

    authorized_evidence_scope = (
        "authorized_labeled_development_evaluated"
        if mode == "labeled_evaluation"
        else "authorized_development_smoke"
    )
    evidence_scope = "test_fixture_only" if _test_fixture else authorized_evidence_scope
    forward_started = time.perf_counter_ns()
    try:
        predictions = []
        runtime_observation = _configure_cpu_runtime(config["runtime"])
        try:
            for row in prepared:
                prediction, _ = predict_primary_camera_window(
                    row,
                    runtime,
                    validation_scope=validation_scope,
                    evidence_scope=evidence_scope,
                )
                if mode == "labeled_evaluation":
                    prediction = _enrich_prediction_with_c3(prediction, c3_bindings)
                predictions.append(prediction)
        finally:
            import torch

            torch.set_num_threads(int(runtime_observation["previous_intra_op"]))
        merge_policy = (evaluation_policy or {}).get("episode_merge_policy")
        episodes = (
            aggregate_episode_candidates(
                predictions,
                merge_gap_seconds=float(merge_policy["merge_gap_seconds"]),
            )
            if merge_policy is not None
            else []
        )
    except (PrimaryCameraInferenceError, CameraInferenceError, ValueError, TypeError) as exc:
        raise CameraDevelopmentError("camera forward failed closed") from exc
    forward_ms = _elapsed_ms(forward_started)

    evaluator_started = time.perf_counter_ns()
    evaluation: Mapping[str, Any] | None = None
    if mode == "labeled_evaluation":
        evaluation = _call_or_default(
            hooks,
            "evaluator",
            lambda: evaluate_labeled_episodes(
                episodes,
                annotations,
                predictions,
                evaluation_policy,
                collection=collection,
            ),
        )
    evaluator_ms = _elapsed_ms(evaluator_started)

    if mode == "labeled_evaluation" and evaluation is None:
        raise CameraDevelopmentError("labeled evaluation did not produce evaluator evidence")
    execution = {
        "schema_version": DEVELOPMENT_EXECUTION_SCHEMA_VERSION,
        "status": DEVELOPMENT_STATUS,
        "evidence_scope": evidence_scope,
        "test_fixture_only": _test_fixture,
        "authorized_scope_if_non_fixture": authorized_evidence_scope,
        "mode": mode,
        "authorization_summary": receipt_summary,
        "validation_scope": validation_scope,
        "active_source_identity_preflight": source_identity,
        "prediction_count": len(predictions),
        "episode_count": len(episodes),
        "coverage": build_coverage_counts(
            [
                {
                    "prediction_status": (
                        row.get("prediction_status")
                        if row.get("prediction_status") in RUNTIME_STATUSES
                        else row.get("window_status")
                    )
                }
                for row in predictions
            ]
        ),
        "models_retrained": False,
        "authorized_camera_data_consumed": not _test_fixture,
        "sealed_camera_accessed": False,
        "wp_or_smartcare_accessed": False,
    }
    files = {
        "execution.json": canonical_json_bytes(execution),
        "predictions.jsonl": canonical_jsonl_bytes(predictions),
        "episodes.jsonl": canonical_jsonl_bytes(episodes),
    }
    if evaluation is not None:
        files["evaluation.json"] = canonical_json_bytes(dict(evaluation))
    artifact_started = time.perf_counter_ns()
    files["manifest.json"] = canonical_json_bytes(
        {
            "schema_version": "wandering-camera-development-manifest-v1",
            "status": DEVELOPMENT_STATUS,
            "evidence_scope": evidence_scope,
            "test_fixture_only": _test_fixture,
            "artifacts": {
                name: {
                    "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(files.items())
            },
        }
    )
    _commit_new_output(output, files)
    artifact_write_ms = _elapsed_ms(artifact_started)
    timings = build_stage_timings(
        receipt_ms=receipt_ms,
        collection_ms=collection_ms,
        source_preflight_ms=source_preflight_ms,
        tracking_annotation_cohort_ms=tracking_annotation_cohort_ms,
        candidate_load_ms=candidate_load_ms,
        qc_preprocess_ms=qc_preprocess_ms,
        forward_ms=forward_ms,
        evaluator_ms=evaluator_ms,
        artifact_write_ms=artifact_write_ms,
    )
    return {
        "output_dir": str(output),
        "status": DEVELOPMENT_STATUS,
        "evidence_scope": evidence_scope,
        "timings": timings,
    }


def evaluate_labeled_episodes(
    episodes: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    window_predictions: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any] | None,
    *,
    collection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validated = _validate_explicit_evaluation_policy(policy)
    normalized_predictions = [
        {
            "prediction_id": row["episode_candidate_id"],
            **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
            "start_sec": row["episode_start_sec"],
            "end_sec_exclusive": row["episode_end_sec_exclusive"],
            "predicted_pattern": _episode_pattern_with_uncertainty(
                row, validated["uncertain_policy"]
            ),
        }
        for row in episodes
    ]
    truth_masks = [
        {
            "annotation_id": row["annotation_id"],
            **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
            "start_sec": row["start_sec"],
            "end_sec_exclusive": row["end_sec_exclusive"],
        }
        for row in annotations
        if row.get("annotation_status") in {"uncertain", "excluded"}
        or row.get("evaluation_role") in {"uncertain", "excluded"}
    ]
    masked_prediction_ids = {
        row["prediction_id"]
        for row in normalized_predictions
        if any(_positive_scoped_overlap(row, mask) for mask in truth_masks)
    }
    scoreable_predictions = [
        row for row in normalized_predictions if row["prediction_id"] not in masked_prediction_ids
    ]
    eligible_annotations = {
        row["annotation_id"]: row
        for row in annotations
        if row.get("annotation_status") == "accepted"
        and row.get("observable_pattern") in FOUR_CLASSES
    }
    normalized_annotations = [
        {
            "annotation_id": row["annotation_id"],
            **{field: row[field] for field in EPISODE_SCOPE_FIELDS},
            "start_sec": row["start_sec"],
            "end_sec_exclusive": row["end_sec_exclusive"],
        }
        for row in eligible_annotations.values()
    ]
    matching = match_episodes_one_to_one(
        scoreable_predictions,
        normalized_annotations,
        matching_policy=validated["matching_policy"],
    )
    prediction_by_id = {row["prediction_id"]: row for row in normalized_predictions}
    prediction_for_annotation = {
        row["annotation_id"]: prediction_by_id[row["prediction_id"]] for row in matching["matches"]
    }
    dispositions = {
        "matched_scored": 0,
        "explicit_model_abstention": 0,
        "unavailable": 0,
        "inference_error": 0,
        "no_prediction_miss": 0,
    }
    shape_pairs: list[dict[str, Any]] = []
    disposition_by_annotation: dict[str, str] = {}
    for annotation_id, row in sorted(eligible_annotations.items()):
        matched = prediction_for_annotation.get(annotation_id)
        if matched is not None:
            predicted_pattern = matched["predicted_pattern"]
            if predicted_pattern == "uncertain":
                disposition = "explicit_model_abstention"
            else:
                disposition = "matched_scored"
            shape_pairs.append(
                {"truth": row["observable_pattern"], "prediction": predicted_pattern}
            )
        else:
            overlapping_statuses = {
                prediction.get("window_status")
                for prediction in window_predictions
                if _window_overlaps_annotation(prediction, row)
            }
            if "inference_error" in overlapping_statuses:
                disposition = "inference_error"
            elif "unavailable" in overlapping_statuses:
                disposition = "unavailable"
            else:
                disposition = "no_prediction_miss"
                shape_pairs.append(
                    {"truth": row["observable_pattern"], "prediction": "miss"}
                )
        dispositions[disposition] += 1
        disposition_by_annotation[annotation_id] = disposition
    purposeful_rows = [
        {
            **row,
            "prediction": prediction_for_annotation.get(annotation_id, {}).get(
                "predicted_pattern", "uncertain"
            ),
        }
        for annotation_id, row in sorted(eligible_annotations.items())
        if row.get("evaluation_role") == "purposeful_hard_negative"
    ]
    result = {
        "matching": matching,
        "shape_counts": evaluate_shape_counts(shape_pairs),
        "purposeful_hard_negative_diagnostics": evaluate_purposeful_hard_negatives(
            purposeful_rows
        ),
        "error_counts": {
            "unmatched_predictions": len(matching["unmatched_prediction_ids"]),
            "unmatched_annotations": len(matching["unmatched_annotation_ids"]),
        },
        "truth_masking": {
            "prediction_count": len(masked_prediction_ids),
            "prediction_ids": sorted(masked_prediction_ids),
            "truth_interval_count": len(truth_masks),
            "excluded_from_main_error_counts": True,
        },
        "accepted_truth_dispositions": dispositions,
        "accepted_truth_disposition_by_annotation": disposition_by_annotation,
        "episode_coverage": build_coverage_counts(
            [
                {
                    "prediction_status": (
                        "uncertain" if row["predicted_pattern"] == "uncertain" else "ready"
                    )
                }
                for row in scoreable_predictions
            ]
        ),
        "scientific_metrics_computed": False,
        "policy_scope": "explicit_development_only",
    }
    if collection is not None:
        result["eligible_negative_person_hours"] = _evaluate_negative_person_hours(
            collection=collection,
            annotations=annotations,
            window_predictions=window_predictions,
            normalized_episode_predictions=normalized_predictions,
        )
    return result


def _evaluate_negative_person_hours(
    *,
    collection: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    window_predictions: Sequence[Mapping[str, Any]],
    normalized_episode_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordinary_negative = [
        _annotation_interval(row)
        for row in annotations
        if row.get("annotation_status") == "accepted"
        and row.get("evaluation_role") == "ordinary_negative"
    ]
    positive = [
        _annotation_interval(row)
        for row in annotations
        if row.get("annotation_status") == "accepted"
        and row.get("evaluation_role") == "wandering_like_positive"
    ]
    truth_masks = [
        _annotation_interval(row)
        for row in annotations
        if row.get("annotation_status") in {"uncertain", "excluded"}
        or row.get("evaluation_role") in {"uncertain", "excluded"}
    ]
    unavailable = [
        _window_interval(row)
        for row in window_predictions
        if row.get("window_status") == "unavailable"
    ]
    tracking_unavailable: list[dict[str, Any]] = []
    qc_unavailable: list[dict[str, Any]] = []
    for row in unavailable:
        tracking_reason = bool(row.pop("_tracking_unavailable"))
        (tracking_unavailable if tracking_reason else qc_unavailable).append(row)
    inference_error = [
        _window_interval(row)
        for row in window_predictions
        if row.get("window_status") == "inference_error"
    ]
    prediction_uncertain = [
        {
            **{field: row[field] for field in ("participant_id", "session_id", "clock_domain_id")},
            "start_sec": row["start_sec"],
            "end_sec_exclusive": row["end_sec_exclusive"],
        }
        for row in normalized_episode_predictions
        if row.get("predicted_pattern") == "uncertain"
    ]
    return compute_eligible_negative_person_hours(
        participant_present=collection["participant_present_intervals"],
        ordinary_negative=ordinary_negative,
        positive=positive,
        truth_uncertain_or_excluded=truth_masks,
        qc_unavailable=qc_unavailable,
        tracking_unavailable=tracking_unavailable,
        inference_error=inference_error,
        tracklet_participant_bindings=collection["tracklet_participant_bindings"],
        clock_alignments=collection["clock_alignments"],
        prediction_uncertain=prediction_uncertain,
    )


def _annotation_interval(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{
            field: row[field]
            for field in ("participant_id", "session_id", "clock_domain_id")
        },
        "start_sec": row["start_sec"],
        "end_sec_exclusive": row["end_sec_exclusive"],
    }


def _window_interval(row: Mapping[str, Any]) -> dict[str, Any]:
    reasons = row.get("reason_codes", [])
    tracking_reason = any(
        str(reason).startswith("tracking_")
        or str(reason) in {"no_valid_observations", "track_fragment_unusable"}
        for reason in reasons
    )
    return {
        **{
            field: row[field]
            for field in ("participant_id", "session_id", "clock_domain_id")
        },
        "start_sec": row["window_start_sec"],
        "end_sec_exclusive": row["window_end_sec"],
        "_tracking_unavailable": tracking_reason,
    }


def _validate_explicit_evaluation_policy(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "matching_policy",
        "uncertain_policy",
        "episode_merge_policy",
    }:
        raise CameraDevelopmentError(
            "labeled production evaluation requires explicit matching/uncertain/episode merge policies"
        )
    _matching_policy(value["matching_policy"])
    uncertain = value["uncertain_policy"]
    merge = value["episode_merge_policy"]
    if not isinstance(uncertain, Mapping) or set(uncertain) != {"policy_id", "threshold"}:
        raise CameraDevelopmentError("uncertain policy is invalid")
    _nonempty_policy_id(uncertain["policy_id"], "uncertain policy_id")
    threshold = uncertain["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise CameraDevelopmentError("uncertain threshold is invalid")
    if not isinstance(merge, Mapping) or set(merge) != {"policy_id", "merge_gap_seconds"}:
        raise CameraDevelopmentError("episode merge policy is invalid")
    _nonempty_policy_id(merge["policy_id"], "episode merge policy_id")
    _finite_nonnegative(merge["merge_gap_seconds"], "merge_gap_seconds")
    return value


def _episode_pattern_with_uncertainty(
    row: Mapping[str, Any], uncertain_policy: Mapping[str, Any]
) -> str:
    summary = row.get("four_class_probability_summary")
    if not isinstance(summary, Mapping):
        raise CameraDevelopmentError("episode probability summary is missing")
    probabilities = summary.get("mean_probabilities")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != len(FOUR_CLASSES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in probabilities
        )
    ):
        raise CameraDevelopmentError("episode probability summary is invalid")
    threshold = float(uncertain_policy["threshold"])
    return "uncertain" if max(float(value) for value in probabilities) < threshold else str(
        row["predicted_pattern"]
    )


def _validate_labeled_c3(
    collection: Mapping[str, Any], annotations: Sequence[Mapping[str, Any]]
) -> None:
    if not annotations:
        raise CameraDevelopmentError("labeled evaluation requires non-empty C2 annotations")
    for field in (
        "tracklet_participant_bindings",
        "participant_present_intervals",
        "clock_alignments",
    ):
        if not collection.get(field):
            raise CameraDevelopmentError(f"labeled evaluation requires C3 {field}")
    if any(row.get("status") not in {"aligned", "single_camera"} for row in collection["clock_alignments"]):
        raise CameraDevelopmentError("labeled evaluation requires aligned C3 camera clocks")


def _bind_actual_tracking_to_c3(
    adapter: CameraAdapterInput,
    collection: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, str, int], Mapping[str, Any]]:
    actual_keys = set(group_observations(adapter.observations))
    binding_by_key: dict[tuple[str, str, str, str, str, int], Mapping[str, Any]] = {}
    for row in collection["tracklet_participant_bindings"]:
        key = _tracking_scope_key(row)
        if key in binding_by_key:
            raise CameraDevelopmentError("duplicate C3 binding for actual tracking scope")
        binding_by_key[key] = row
    missing = actual_keys - set(binding_by_key)
    if missing:
        raise CameraDevelopmentError("actual tracking contains an unbound C3 tracklet")
    actual_prefixes = {key[:5] for key in actual_keys}
    stale = {
        key for key in binding_by_key if key[:5] in actual_prefixes and key not in actual_keys
    }
    if stale:
        raise CameraDevelopmentError("C3 contains a stale or wrong tracklet binding")
    for row in annotations:
        if _tracking_scope_key(row) not in actual_keys:
            raise CameraDevelopmentError("annotation is bound to a non-observed tracklet")
    return {key: binding_by_key[key] for key in sorted(actual_keys)}


def _enrich_prediction_with_c3(
    prediction: Mapping[str, Any],
    bindings: Mapping[tuple[str, str, str, str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    key = _tracking_scope_key(prediction)
    binding = bindings.get(key)
    if binding is None:
        raise CameraDevelopmentError("prediction tracklet has no unique C3 binding")
    return {
        **dict(prediction),
        "participant_id": binding["participant_id"],
        "session_id": binding["session_id"],
        "camera_setup_id": binding["camera_setup_id"],
        "clock_domain_id": binding["clock_domain_id"],
    }


def _tracking_scope_key(
    row: Mapping[str, Any],
) -> tuple[str, str, str, str, str, int]:
    fields = (
        "source_group_id",
        "source_video_id",
        "device_id",
        "setup_id",
        "stream_epoch",
    )
    values: list[str] = []
    for field in fields:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise CameraDevelopmentError(f"C3 tracking scope field is invalid: {field}")
        values.append(value)
    track_id = row.get("track_id")
    if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id < 0:
        raise CameraDevelopmentError("C3 tracking scope track_id is invalid")
    return (*values, track_id)


def _matching_policy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "policy_id",
        "minimum_temporal_iou",
        "maximum_onset_delta_sec",
    }:
        raise CameraDevelopmentError("explicit matching policy is required")
    policy_id = _nonempty_policy_id(value["policy_id"], "matching policy_id")
    minimum_iou = _finite_nonnegative(value["minimum_temporal_iou"], "minimum_temporal_iou")
    maximum_onset = _finite_nonnegative(
        value["maximum_onset_delta_sec"], "maximum_onset_delta_sec"
    )
    if minimum_iou > 1:
        raise CameraDevelopmentError("minimum_temporal_iou must be within [0,1]")
    return {
        "policy_id": policy_id,
        "minimum_temporal_iou": minimum_iou,
        "maximum_onset_delta_sec": maximum_onset,
    }


def _expected_development_config() -> dict[str, Any]:
    return {
        "schema_version": DEVELOPMENT_CONFIG_SCHEMA_VERSION,
        "purpose": "topowander_m0cam_authorized_development",
        "dataset_role": "development",
        "collection_config": {"path": "configs/data/wandering_camera_collection_v1.yaml"},
        "camera_chain": {
            "path": "configs/modules/wandering_camera_v1.yaml",
            "sha256": "08bbec6ef263dd45fef3262c407ce675107f87ea584584f7ab61142cd734ae8e",
            "size_bytes": 2500,
        },
        "candidate": {
            "candidate_id": "topowander-m0s-seed20260731-epoch0005",
            "manifest_path": "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/artifacts/topowander_m0r_candidate_v3/candidate_manifest.json",
            "manifest_sha256": "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7",
            "manifest_size_bytes": 12642,
            "model_state_sha256": "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031",
            "primary_seed": 20260731,
            "best_epoch": 5,
            "retrained": False,
            "ensemble": False,
        },
        "runtime": {
            "device": "cpu",
            "intra_op_threads": 8,
            "inter_op_threads": 1,
            "maximum_batch_size": 64,
        },
        "inference": {"binary_decision_threshold": 0.5},
        "matching_policy": None,
        "uncertain_policy": None,
        "episode_merge_policy": None,
        "data_access": {
            "wp_raw": False,
            "smartcare_official_or_raw": False,
            "sealed_camera": False,
            "human_camera_requires_c0": True,
        },
    }


def _read_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraDevelopmentError(f"cannot parse {role}") from exc
    if not isinstance(value, dict):
        raise CameraDevelopmentError(f"{role} must be a JSON object")
    return value


def _read_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        output = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraDevelopmentError(f"cannot parse {role} JSONL") from exc
    if any(not isinstance(row, dict) for row in output):
        raise CameraDevelopmentError(f"{role} JSONL rows must be objects")
    return output


def _call_or_default(
    hooks: Mapping[str, Callable[..., Any]], name: str, default: Callable[[], Any]
) -> Any:
    return hooks[name]() if name in hooks else default()


def _collection_scope(collection: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        "participant_ids": {str(row["participant_id"]) for row in collection["participants"]},
        "session_ids": {str(row["session_id"]) for row in collection["sessions"]},
        "camera_setup_ids": {
            str(row["camera_setup_id"]) for row in collection["camera_setups"]
        },
        "source_group_ids": {str(row["source_group_id"]) for row in collection["sessions"]},
    }


def _same_episode_scope(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for field in EPISODE_SCOPE_FIELDS:
        if field not in left or field not in right or left[field] != right[field]:
            return False
    return True


def _positive_scoped_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not _same_episode_scope(left, right):
        return False
    left_start, left_end = _interval(left)
    right_start, right_end = _interval(right)
    return min(left_end, right_end) > max(left_start, right_start)


def _window_overlaps_annotation(
    prediction: Mapping[str, Any], annotation: Mapping[str, Any]
) -> bool:
    if any(
        field not in prediction
        or field not in annotation
        or prediction[field] != annotation[field]
        for field in EPISODE_SCOPE_FIELDS
    ):
        return False
    try:
        start = _finite_nonnegative(prediction.get("window_start_sec"), "window_start_sec")
        end = _finite_nonnegative(prediction.get("window_end_sec"), "window_end_sec")
        annotation_start, annotation_end = _interval(annotation)
    except CameraDevelopmentError:
        return False
    return end > start and min(end, annotation_end) > max(start, annotation_start)


def _bind_media_to_collection(media: Mapping[str, Any], collection: Mapping[str, Any]) -> None:
    matches = [
        source
        for source in collection["sources"]
        if source["source_video_id"] == media["source_video_id"]
    ]
    if len(matches) != 1:
        raise CameraDevelopmentError("media source_video_id is outside collection")
    source = matches[0]
    if (
        source["source_group_id"] != media["source_group_id"]
        or source["device_id"] != media["device_id"]
        or source["setup_id"] != media["setup_id"]
        or source["stream_epoch"] != media["stream_epoch"]
    ):
        raise CameraDevelopmentError("media sidecar authorization binding mismatch")


def _bind_repository_path(root: Path, path: str | Path, relative: str) -> None:
    if Path(path).resolve(strict=False) != (root / relative).resolve(strict=False):
        raise CameraDevelopmentError(f"must use fixed repository path: {relative}")


def _preflight_fresh_output(output: Path) -> None:
    if output.exists():
        raise CameraDevelopmentError(f"final output already exists: {output}")
    if output.parent.exists() and list(output.parent.glob(f"{output.name}.staging-*")):
        raise CameraDevelopmentError("same-prefix staging output already exists")


def _commit_new_output(output: Path, files: Mapping[str, bytes]) -> None:
    _preflight_fresh_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.staging-", dir=output.parent))
    try:
        for name, payload in sorted(files.items()):
            destination = temporary / name
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output.exists():
            raise CameraDevelopmentError(f"final output already exists: {output}")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _unique_ids(rows: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    output: list[str] = []
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise CameraDevelopmentError(f"{field} is invalid")
        output.append(value)
    if len(output) != len(set(output)):
        raise CameraDevelopmentError(f"duplicate {field}")
    return set(output)


def _interval(row: Mapping[str, Any]) -> tuple[float, float]:
    start = _finite_nonnegative(row.get("start_sec"), "start_sec")
    end = _finite_nonnegative(row.get("end_sec_exclusive"), "end_sec_exclusive")
    if end <= start:
        raise CameraDevelopmentError("interval must be non-empty")
    return start, end


def _numeric_interval(value: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise CameraDevelopmentError("interval must contain two values")
    start = _finite_nonnegative(value[0], "interval start")
    end = _finite_nonnegative(value[1], "interval end")
    if end <= start:
        raise CameraDevelopmentError("interval must be non-empty")
    return start, end


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraDevelopmentError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise CameraDevelopmentError(f"{field} must be finite and nonnegative")
    return number


def _nonempty_policy_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraDevelopmentError(f"{field} must be a non-empty string")
    return value


def _rows_to_group_intervals(
    rows: Sequence[Mapping[str, Any]], group: tuple[str, str, str]
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for row in rows:
        observed = tuple(
            _required_group_token(row, field)
            for field in ("participant_id", "session_id", "clock_domain_id")
        )
        if observed == group:
            output.append(_interval(row))
    return output


def _required_group_token(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise CameraDevelopmentError(f"{field} is required for person-hours")
    return value


def _intersect_intervals(
    left: Iterable[tuple[float, float]], right: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for left_start, left_end in union_intervals(left):
        for right_start, right_end in union_intervals(right):
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end > start:
                output.append((start, end))
    return union_intervals(output)


def _negative_not_computable(reason: str) -> dict[str, Any]:
    return {
        "status": "not_computable",
        "reason": reason,
        "eligible_negative_seconds": None,
        "eligible_negative_person_hours": None,
        "prediction_uncertain_subtracted": False,
        "unlabeled_remainder_treated_as_negative": False,
        "identity_inferred_from_track_id": False,
    }
