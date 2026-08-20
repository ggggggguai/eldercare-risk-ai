"""Batch evaluation for truth-separated oracle-boundary camera episodes."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    CameraEpisodeImportError,
    load_episode_truth,
)


CAMERA_EPISODE_EVAL_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-episode-eval-config-v1"
)
CAMERA_EPISODE_EVAL_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-episode-eval-summary-v1"
)
CAMERA_EPISODE_EVAL_RESULT_SCHEMA_VERSION = (
    "wandering-camera-episode-eval-result-v1"
)
CAMERA_EPISODE_EVAL_INDEX_FIELDS = frozenset(
    {
        "bundle_id",
        "prediction_bundle_dir",
        "truth_jsonl",
        "participant_id",
        "session_id",
        "camera_setup_id",
    }
)
_PREDICTION_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "source_video_id",
        "track_id",
        "start_sec",
        "end_sec_exclusive",
        "duration_sec",
        "prediction_status",
        "prediction_reason_codes",
        "qc_status",
        "qc_reason_codes",
        "predicted_pattern",
        "binary",
        "four_class",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "binary_decision_threshold",
        "probability_calibrated",
    }
)
_SUMMARY_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "source_video_id",
        "episode_count",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "binary_decision_threshold",
        "probability_calibrated",
        "models_retrained",
        "truth_labels_consumed_by_inference",
        "automatic_boundary_inference",
        "legacy_40_second_diagnostic_included",
    }
)
_NOT_COMPUTABLE = "not_computable"


class CameraEpisodeEvaluationError(ValueError):
    """Batch index, identity join, or evaluation semantics are invalid."""

    def __init__(self, message: str, *, code: str = "invalid_evaluation_input"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CameraEpisodeEvaluationBuildResult:
    output_dir: Path
    truth_count: int
    shape_eligible_count: int
    ready_count: int
    failure_count: int


def load_camera_episode_evaluation_config(path: str | Path) -> dict[str, Any]:
    """Load the small semantic config that fixes EP1B metric denominators."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeEvaluationError("cannot read episode evaluation config") from exc
    if not isinstance(value, Mapping):
        raise CameraEpisodeEvaluationError("episode evaluation config must be a mapping")
    config = dict(value)
    required = {
        "schema_version",
        "evaluation_name",
        "class_order",
        "binary_mapping",
        "shape_eligibility",
        "runtime_statuses",
        "pipeline_miss_statuses",
        "duration_bands",
        "identity_time_tolerance_sec",
    }
    if set(config) != required:
        raise CameraEpisodeEvaluationError("episode evaluation config fields are invalid")
    if config["schema_version"] != CAMERA_EPISODE_EVAL_CONFIG_SCHEMA_VERSION:
        raise CameraEpisodeEvaluationError("episode evaluation config version is invalid")
    class_order = config["class_order"]
    if class_order != ["direct", "pacing", "lapping", "random"]:
        raise CameraEpisodeEvaluationError("episode evaluation class order is invalid")
    if config["binary_mapping"] != {
        "direct": "direct_or_non_wandering",
        "pacing": "wandering_like",
        "lapping": "wandering_like",
        "random": "wandering_like",
    }:
        raise CameraEpisodeEvaluationError("episode evaluation binary mapping is invalid")
    statuses = config["runtime_statuses"]
    if statuses != [
        "ready",
        "unavailable",
        "boundary_uncertain",
        "inference_error",
        "abstention",
    ]:
        raise CameraEpisodeEvaluationError("episode evaluation runtime statuses are invalid")
    if config["pipeline_miss_statuses"] != statuses[1:]:
        raise CameraEpisodeEvaluationError("pipeline miss statuses are invalid")
    eligibility = config["shape_eligibility"]
    if eligibility != {
        "explicit_exclusion_statuses": ["excluded"],
        "explicit_exclusion_roles": ["excluded"],
        "explicit_exclusion_tracking_issues": ["wrong_target"],
    }:
        raise CameraEpisodeEvaluationError("shape eligibility config is invalid")
    bands = config["duration_bands"]
    if bands != [
        {"name": "short", "minimum_sec": 0.0, "maximum_sec_exclusive": 15.0},
        {"name": "medium", "minimum_sec": 15.0, "maximum_sec_exclusive": 40.0},
        {"name": "long", "minimum_sec": 40.0, "maximum_sec_exclusive": None},
    ]:
        raise CameraEpisodeEvaluationError("duration bands are invalid")
    tolerance = _finite_number(config["identity_time_tolerance_sec"], "tolerance")
    if tolerance < 0.0 or tolerance > 1e-3:
        raise CameraEpisodeEvaluationError("identity tolerance is invalid")
    config["identity_time_tolerance_sec"] = tolerance
    return config


def build_camera_episode_evaluation_bundle(
    *,
    config_path: str | Path,
    batch_index_path: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeEvaluationBuildResult:
    """Join multiple EP1A bundles to independent truth and write a fresh report."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera episode evaluation output already exists: {output}")
    config = load_camera_episode_evaluation_config(config_path)
    try:
        index_path = Path(batch_index_path).resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeEvaluationError(
            "episode batch index is missing or inaccessible",
            code="batch_index_inaccessible",
        ) from exc
    index_rows = _load_batch_index(index_path)
    results, input_identity = _join_batch(index_path, index_rows, config)
    eligible = [row for row in results if row["shape_truth_status"] == "eligible"]
    main_metrics, main_confusion = _metric_bundle(eligible, config)
    ready_only = [row for row in eligible if row["prediction_status"] == "ready"]
    ready_metrics, ready_confusion = _metric_bundle(ready_only, config)
    group_support = _group_support(results, config)
    metrics = {
        "evaluation_name": config["evaluation_name"],
        "primary_metric_population": "all_shape_eligible",
        "conditional_metric_population": "ready_only_conditional",
        "all_shape_eligible": main_metrics,
        "ready_only_conditional": ready_metrics,
        "qc_coverage": _coverage_metrics(eligible, config),
        "duration_bands": _duration_metrics(eligible, config),
        "group_counts": {
            group_name: len(groups) for group_name, groups in group_support.items()
        },
        "group_support": group_support,
        "purposeful_hard_negative_diagnostics": _purposeful_diagnostics(
            eligible, config
        ),
        "probability_calibrated": False,
        "alert_metrics_available": False,
    }
    confusion = {
        "all_shape_eligible": main_confusion,
        "ready_only_conditional": ready_confusion,
    }
    failures = _failure_rows(results)
    shape_status_counts = Counter(row["shape_truth_status"] for row in results)
    status_counts = Counter(row["prediction_status"] for row in eligible)
    all_status_counts = Counter(row["prediction_status"] for row in results)
    summary = {
        "schema_version": CAMERA_EPISODE_EVAL_SUMMARY_SCHEMA_VERSION,
        "status": "wandering_m0cam_ep1b_oracle_boundary_evaluated",
        "study_design": "descriptive_development_pilot",
        "evaluation_name": config["evaluation_name"],
        "batch_count": len(index_rows),
        "truth_count": len(results),
        "shape_truth_status_counts": {
            name: shape_status_counts.get(name, 0)
            for name in ("eligible", "unknown", "excluded")
        },
        "shape_eligible_count": len(eligible),
        "prediction_status_counts_all_truth": {
            name: all_status_counts.get(name, 0) for name in config["runtime_statuses"]
        },
        "prediction_status_counts_for_shape_eligible": {
            name: status_counts.get(name, 0) for name in config["runtime_statuses"]
        },
        "candidate_id": input_identity["candidate_id"],
        "candidate_manifest_sha256": input_identity["candidate_manifest_sha256"],
        "model_state_sha256": input_identity["model_state_sha256"],
        "class_order": list(config["class_order"]),
        "binary_decision_threshold": 0.5,
        "probability_calibrated": False,
        "truth_labels_consumed_by_inference": False,
        "prediction_truth_joined_after_inference": True,
        "automatic_boundary_inference": False,
        "legacy_40_second_diagnostic_included": False,
        "alert_metrics_available": False,
        "models_retrained": False,
        "failure_count": len(failures),
    }
    files = {
        "summary.json": canonical_json_bytes(summary),
        "metrics.json": canonical_json_bytes(metrics),
        "confusion.json": canonical_json_bytes(confusion),
        "episode_results.jsonl": canonical_jsonl_bytes(results),
        "failures.jsonl": canonical_jsonl_bytes(failures),
        "README.md": _readme_bytes(summary, metrics),
    }
    _commit_new_directory(output, files)
    return CameraEpisodeEvaluationBuildResult(
        output_dir=output,
        truth_count=len(results),
        shape_eligible_count=len(eligible),
        ready_count=status_counts.get("ready", 0),
        failure_count=len(failures),
    )


def _load_batch_index(path: Path) -> list[dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CameraEpisodeEvaluationError("cannot read episode batch index") from exc
    rows: list[dict[str, str]] = []
    bundle_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, CameraEpisodeEvaluationError) as exc:
            raise CameraEpisodeEvaluationError(
                f"invalid batch index JSON at line {line_number}"
            ) from exc
        if not isinstance(value, Mapping) or frozenset(value) != CAMERA_EPISODE_EVAL_INDEX_FIELDS:
            raise CameraEpisodeEvaluationError("batch index fields are invalid")
        row = {name: _nonempty_text(value[name], name) for name in value}
        if row["bundle_id"] in bundle_ids:
            raise CameraEpisodeEvaluationError("duplicate bundle_id in batch index")
        bundle_ids.add(row["bundle_id"])
        rows.append(row)
    if not rows:
        raise CameraEpisodeEvaluationError("episode batch index is empty")
    return sorted(rows, key=lambda row: row["bundle_id"])


def _join_batch(
    index_path: Path,
    index_rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    all_results: list[dict[str, Any]] = []
    global_truth_ids: set[str] = set()
    global_prediction_ids: set[str] = set()
    global_truth_identities: list[tuple[str, int, float, float]] = []
    global_prediction_identities: list[tuple[str, int, float, float]] = []
    global_source_contexts: dict[str, tuple[str, str]] = {}
    global_track_participants: dict[tuple[str, int], str] = {}
    common_identity: dict[str, str] | None = None
    for index_row in index_rows:
        bundle = _resolve_relative(index_path.parent, index_row["prediction_bundle_dir"])
        truth_path = _resolve_relative(index_path.parent, index_row["truth_jsonl"])
        if not bundle.is_dir():
            raise CameraEpisodeEvaluationError("prediction bundle directory is missing")
        if not truth_path.is_file():
            raise CameraEpisodeEvaluationError("episode truth file is missing")
        summary = _load_summary(bundle / "summary.json")
        predictions = _load_predictions(bundle / "episode_predictions.jsonl", config)
        try:
            truths = load_episode_truth(truth_path)
        except CameraEpisodeImportError as exc:
            raise CameraEpisodeEvaluationError(str(exc)) from exc
        if summary["episode_count"] != len(predictions):
            raise CameraEpisodeEvaluationError("prediction summary episode count mismatch")
        if any(row["source_video_id"] != summary["source_video_id"] for row in predictions):
            raise CameraEpisodeEvaluationError(
                "prediction source_video_id identity differs from summary"
            )
        identity = {
            "candidate_id": summary["candidate_id"],
            "candidate_manifest_sha256": summary["candidate_manifest_sha256"],
            "model_state_sha256": summary["model_state_sha256"],
        }
        if any(
            any(prediction[field] != value for field, value in identity.items())
            for prediction in predictions
        ):
            raise CameraEpisodeEvaluationError(
                "prediction candidate/model identity differs from summary"
            )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise CameraEpisodeEvaluationError("candidate/model identity differs across bundles")
        truth_by_id = _unique_by_episode_id(truths, "truth")
        prediction_by_id = _unique_by_episode_id(predictions, "prediction")
        duplicate_truth = global_truth_ids.intersection(truth_by_id)
        duplicate_prediction = global_prediction_ids.intersection(prediction_by_id)
        if duplicate_truth or duplicate_prediction:
            raise CameraEpisodeEvaluationError("duplicate episode_id across evaluation batch")
        global_truth_ids.update(truth_by_id)
        global_prediction_ids.update(prediction_by_id)
        _extend_unique_physical_identities(
            global_truth_identities,
            truths,
            track_field="target_track_id",
            tolerance=float(config["identity_time_tolerance_sec"]),
            role="truth",
        )
        _extend_unique_physical_identities(
            global_prediction_identities,
            predictions,
            track_field="track_id",
            tolerance=float(config["identity_time_tolerance_sec"]),
            role="prediction",
        )
        if set(truth_by_id) != set(prediction_by_id):
            missing_prediction = sorted(set(truth_by_id) - set(prediction_by_id))
            missing_truth = sorted(set(prediction_by_id) - set(truth_by_id))
            raise CameraEpisodeEvaluationError(
                "one-to-one episode join has missing prediction or truth: "
                f"missing_prediction={missing_prediction}, missing_truth={missing_truth}"
            )
        for episode_id in sorted(truth_by_id):
            truth = truth_by_id[episode_id]
            prediction = prediction_by_id[episode_id]
            _validate_identity_pair(truth, prediction, config)
            _validate_group_identity(
                truth,
                index_row,
                source_contexts=global_source_contexts,
                track_participants=global_track_participants,
            )
            status, reason = _derive_shape_truth_status(truth, config)
            duration = float(truth["end_sec_exclusive"]) - float(truth["start_sec"])
            row = {
                "bundle_id": index_row["bundle_id"],
                "participant_id": index_row["participant_id"],
                "session_id": index_row["session_id"],
                "camera_setup_id": index_row["camera_setup_id"],
                **truth,
                "schema_version": CAMERA_EPISODE_EVAL_RESULT_SCHEMA_VERSION,
                "shape_truth_status": status,
                "shape_truth_reason": reason,
                "duration_sec": duration,
                "duration_band": _duration_band(duration, config),
                "prediction_status": prediction["prediction_status"],
                "prediction_reason_codes": list(prediction["prediction_reason_codes"]),
                "predicted_pattern": prediction["predicted_pattern"],
                "binary": prediction["binary"],
                "binary_truth": config["binary_mapping"].get(
                    truth["observable_pattern"]
                ),
                "predicted_binary_label": (
                    prediction["binary"]["predicted_label"]
                    if prediction["prediction_status"] == "ready"
                    else None
                ),
                "qc_status": prediction["qc_status"],
                "qc_reason_codes": list(prediction["qc_reason_codes"]),
                "four_class": prediction["four_class"],
                "candidate_id": prediction["candidate_id"],
                "candidate_manifest_sha256": prediction[
                    "candidate_manifest_sha256"
                ],
                "model_state_sha256": prediction["model_state_sha256"],
                "probability_calibrated": prediction["probability_calibrated"],
            }
            row["four_class_metric_disposition"] = _metric_disposition(
                row,
                truth_field="observable_pattern",
                prediction_field="predicted_pattern",
            )
            row["shape_binary_metric_disposition"] = _metric_disposition(
                row,
                truth_field="binary_truth",
                prediction_field="predicted_binary_label",
            )
            if row["shape_truth_status"] != "eligible":
                row["main_metric_disposition"] = "not_shape_eligible"
            elif row["prediction_status"] != "ready":
                row["main_metric_disposition"] = "pipeline_miss"
            elif (
                row["four_class_metric_disposition"] == "correct"
                and row["shape_binary_metric_disposition"] == "correct"
            ):
                row["main_metric_disposition"] = "correct"
            else:
                row["main_metric_disposition"] = "misclassified"
            all_results.append(row)
    if common_identity is None:
        raise CameraEpisodeEvaluationError("evaluation batch has no candidate identity")
    return sorted(all_results, key=_result_sort_key), common_identity


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, CameraEpisodeEvaluationError) as exc:
        raise CameraEpisodeEvaluationError("cannot read prediction bundle summary") from exc
    if not isinstance(value, Mapping):
        raise CameraEpisodeEvaluationError("prediction bundle summary is incomplete")
    missing_fields = sorted(_SUMMARY_REQUIRED_FIELDS - frozenset(value))
    if missing_fields:
        raise CameraEpisodeEvaluationError(
            "prediction bundle summary is missing required fields: "
            + ", ".join(missing_fields)
        )
    row = dict(value)
    if row["schema_version"] != "wandering-camera-episode-run-summary-v1":
        raise CameraEpisodeEvaluationError("prediction bundle summary version is invalid")
    if row["truth_labels_consumed_by_inference"] is not False:
        raise CameraEpisodeEvaluationError("prediction bundle consumed truth labels")
    if row["automatic_boundary_inference"] is not False:
        raise CameraEpisodeEvaluationError("prediction bundle is not oracle-boundary")
    if row["probability_calibrated"] is not False:
        raise CameraEpisodeEvaluationError("prediction probabilities claim calibration")
    if row["models_retrained"] is not False:
        raise CameraEpisodeEvaluationError("prediction summary models_retrained must be false")
    if row["legacy_40_second_diagnostic_included"] is not False:
        raise CameraEpisodeEvaluationError(
            "prediction summary legacy_40_second_diagnostic_included must be false"
        )
    if row["binary_decision_threshold"] != 0.5:
        raise CameraEpisodeEvaluationError("prediction threshold differs from 0.5")
    for field in (
        "source_video_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
    ):
        row[field] = _nonempty_text(row[field], field)
    if isinstance(row["episode_count"], bool) or not isinstance(row["episode_count"], int) or row["episode_count"] < 1:
        raise CameraEpisodeEvaluationError("prediction summary episode_count is invalid")
    return row


def _load_predictions(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CameraEpisodeEvaluationError("cannot read episode predictions") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_json_constant)
            row = _validate_prediction(value, config)
        except json.JSONDecodeError as exc:
            raise CameraEpisodeEvaluationError(
                f"invalid episode prediction at line {line_number}"
            ) from exc
        except CameraEpisodeEvaluationError as exc:
            raise CameraEpisodeEvaluationError(
                f"invalid episode prediction at line {line_number}: {exc}",
                code=exc.code,
            ) from exc
        rows.append(row)
    if not rows:
        raise CameraEpisodeEvaluationError("episode prediction JSONL is empty")
    _unique_by_episode_id(rows, "prediction")
    return rows


def _validate_prediction(value: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _PREDICTION_REQUIRED_FIELDS.issubset(value):
        raise CameraEpisodeEvaluationError("episode prediction fields are incomplete")
    row = dict(value)
    if row["schema_version"] != "wandering-camera-episode-prediction-v1":
        raise CameraEpisodeEvaluationError("episode prediction version is invalid")
    for field in (
        "episode_id",
        "source_video_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
    ):
        row[field] = _nonempty_text(row[field], field)
    if isinstance(row["track_id"], bool) or not isinstance(row["track_id"], int) or row["track_id"] < 0:
        raise CameraEpisodeEvaluationError("episode prediction track_id is invalid")
    for field in ("start_sec", "end_sec_exclusive", "duration_sec"):
        row[field] = _finite_number(row[field], field)
    if row["start_sec"] < 0.0 or row["start_sec"] >= row["end_sec_exclusive"]:
        raise CameraEpisodeEvaluationError("episode prediction interval is invalid")
    if not math.isclose(
        row["duration_sec"],
        row["end_sec_exclusive"] - row["start_sec"],
        rel_tol=0.0,
        abs_tol=float(config["identity_time_tolerance_sec"]),
    ):
        raise CameraEpisodeEvaluationError("episode prediction duration is inconsistent")
    status = row["prediction_status"]
    if not isinstance(status, str) or status not in config["runtime_statuses"]:
        raise CameraEpisodeEvaluationError("episode prediction status is invalid")
    reasons = row["prediction_reason_codes"]
    if not isinstance(reasons, list) or any(
        not isinstance(item, str) or not item for item in reasons
    ):
        raise CameraEpisodeEvaluationError("prediction reason codes are invalid")
    if (status == "ready") != (not reasons):
        raise CameraEpisodeEvaluationError(
            "ready predictions need no reasons and non-ready predictions need reasons"
        )
    qc_status = row["qc_status"]
    if not isinstance(qc_status, str) or qc_status not in {
        "ready",
        "unavailable",
        "boundary_uncertain",
    }:
        raise CameraEpisodeEvaluationError("episode prediction QC status is invalid")
    qc_reasons = row["qc_reason_codes"]
    if not isinstance(qc_reasons, list) or any(
        not isinstance(item, str) or not item for item in qc_reasons
    ):
        raise CameraEpisodeEvaluationError("QC reason codes are invalid")
    if (qc_status == "ready") != (not qc_reasons):
        raise CameraEpisodeEvaluationError(
            "ready QC needs no reasons and non-ready QC needs reasons"
        )
    allowed_prediction_statuses = {
        "ready": {"ready", "inference_error", "abstention"},
        "unavailable": {"unavailable"},
        "boundary_uncertain": {"boundary_uncertain"},
    }
    if status not in allowed_prediction_statuses[qc_status]:
        raise CameraEpisodeEvaluationError(
            "QC/prediction status combination is invalid"
        )
    if row["binary_decision_threshold"] != 0.5:
        raise CameraEpisodeEvaluationError("episode prediction threshold differs from 0.5")
    if row["probability_calibrated"] is not False:
        raise CameraEpisodeEvaluationError("episode prediction claims calibrated probability")
    if status == "ready":
        if row["predicted_pattern"] not in config["class_order"]:
            raise CameraEpisodeEvaluationError("ready prediction has no valid shape")
        binary = row["binary"]
        binary_order = ["direct_or_non_wandering", "wandering_like"]
        if (
            not isinstance(binary, Mapping)
            or binary.get("class_order") != binary_order
            or binary.get("predicted_label") not in binary_order
        ):
            raise CameraEpisodeEvaluationError(
                "ready binary prediction is inconsistent"
            )
        probabilities = binary.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != 2:
            raise CameraEpisodeEvaluationError(
                "ready binary probabilities are invalid"
            )
        binary_probabilities = [
            _finite_number(value, "binary probability") for value in probabilities
        ]
        if any(value < 0.0 or value > 1.0 for value in binary_probabilities) or not math.isclose(
            sum(binary_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise CameraEpisodeEvaluationError(
                "ready binary probabilities are invalid"
            )
        expected_binary_label = (
            "wandering_like"
            if binary_probabilities[1] >= 0.5
            else "direct_or_non_wandering"
        )
        if binary["predicted_label"] != expected_binary_label:
            raise CameraEpisodeEvaluationError(
                "binary predicted label violates the frozen 0.5 threshold"
            )
        four_class = row["four_class"]
        if (
            not isinstance(four_class, Mapping)
            or four_class.get("class_order") != config["class_order"]
            or four_class.get("predicted_label") != row["predicted_pattern"]
        ):
            raise CameraEpisodeEvaluationError("ready four-class prediction is inconsistent")
        probabilities = four_class.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(
            config["class_order"]
        ):
            raise CameraEpisodeEvaluationError(
                "ready four-class probabilities are invalid"
            )
        four_class_probabilities = [
            _finite_number(value, "four-class probability")
            for value in probabilities
        ]
        if any(
            value < 0.0 or value > 1.0 for value in four_class_probabilities
        ) or not math.isclose(
            sum(four_class_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6
        ):
            raise CameraEpisodeEvaluationError(
                "ready four-class probabilities are invalid"
            )
        if not math.isclose(
            four_class_probabilities[0],
            binary_probabilities[0],
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            sum(four_class_probabilities[1:]),
            binary_probabilities[1],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CameraEpisodeEvaluationError(
                "binary and four-class probabilities violate the frozen hierarchical relation"
            )
        expected_four_class_label = config["class_order"][
            max(
                range(len(four_class_probabilities)),
                key=four_class_probabilities.__getitem__,
            )
        ]
        if four_class["predicted_label"] != expected_four_class_label:
            raise CameraEpisodeEvaluationError(
                "four-class predicted label violates fixed-order argmax"
            )
    elif (
        row["predicted_pattern"] is not None
        or row["binary"] is not None
        or row["four_class"] is not None
    ):
        raise CameraEpisodeEvaluationError(
            "non-ready prediction must not claim a shape"
        )
    return row


def _validate_identity_pair(
    truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if truth["episode_id"] != prediction["episode_id"]:
        raise CameraEpisodeEvaluationError(
            "episode identity mismatch", code="episode_identity_mismatch"
        )
    if truth["source_video_id"] != prediction["source_video_id"]:
        raise CameraEpisodeEvaluationError(
            "source video identity mismatch", code="source_video_identity_mismatch"
        )
    if truth["target_track_id"] != prediction["track_id"]:
        raise CameraEpisodeEvaluationError(
            "track identity mismatch", code="track_identity_mismatch"
        )
    tolerance = float(config["identity_time_tolerance_sec"])
    for truth_field, prediction_field in (
        ("start_sec", "start_sec"),
        ("end_sec_exclusive", "end_sec_exclusive"),
    ):
        if not math.isclose(
            float(truth[truth_field]),
            float(prediction[prediction_field]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise CameraEpisodeEvaluationError(
                "episode interval identity drift", code="episode_interval_identity_drift"
            )


def _derive_shape_truth_status(
    truth: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, str]:
    eligibility = config["shape_eligibility"]
    if truth["annotation_status"] in eligibility["explicit_exclusion_statuses"]:
        return "excluded", "annotation_status_excluded"
    if truth["evaluation_role"] in eligibility["explicit_exclusion_roles"]:
        return "excluded", "evaluation_role_excluded"
    if truth["tracking_issue"] in eligibility["explicit_exclusion_tracking_issues"]:
        return "excluded", "wrong_target"
    if truth["observable_pattern"] not in config["class_order"]:
        return "unknown", "observable_pattern_unknown"
    return "eligible", "clear_shape"


def _metric_bundle(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    four_order = list(config["class_order"])
    binary_order = ["direct_or_non_wandering", "wandering_like"]
    four_confusion = {
        truth: {prediction: 0 for prediction in [*four_order, "pipeline_miss"]}
        for truth in four_order
    }
    binary_confusion = {
        truth: {prediction: 0 for prediction in [*binary_order, "pipeline_miss"]}
        for truth in binary_order
    }
    for row in rows:
        truth = row["observable_pattern"]
        prediction = (
            row["predicted_pattern"]
            if row["prediction_status"] == "ready"
            else "pipeline_miss"
        )
        four_confusion[truth][prediction] += 1
        binary_truth = config["binary_mapping"][truth]
        binary_prediction = (
            row["predicted_binary_label"]
            if row["prediction_status"] == "ready"
            else "pipeline_miss"
        )
        binary_confusion[binary_truth][binary_prediction] += 1
    metrics = {
        "four_class": _classification_metrics(four_confusion, four_order),
        "shape_binary": _classification_metrics(binary_confusion, binary_order),
    }
    confusion = {
        "four_class": four_confusion,
        "shape_binary": binary_confusion,
    }
    return metrics, confusion


def _classification_metrics(
    confusion: Mapping[str, Mapping[str, int]], class_order: Sequence[str]
) -> dict[str, Any]:
    support_total = sum(sum(confusion[truth].values()) for truth in class_order)
    correct = sum(confusion[label][label] for label in class_order)
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    all_supported = True
    for label in class_order:
        true_positive = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in class_order)
        precision = true_positive / predicted if predicted else 0.0
        recall: float | str
        f1: float | str
        if support:
            recall = true_positive / support
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            f1_values.append(float(f1))
        else:
            recall = _NOT_COMPUTABLE
            f1 = _NOT_COMPUTABLE
            all_supported = False
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "true_positive": true_positive,
            "predicted_count": predicted,
        }
    return {
        "class_order": list(class_order),
        "support": support_total,
        "correct": correct,
        "accuracy": correct / support_total if support_total else _NOT_COMPUTABLE,
        "macro_f1": (
            sum(f1_values) / len(class_order)
            if all_supported and len(f1_values) == len(class_order)
            else _NOT_COMPUTABLE
        ),
        "per_class": per_class,
        "pipeline_miss_count": sum(
            confusion[truth]["pipeline_miss"] for truth in class_order
        ),
    }


def _coverage_metrics(
    eligible: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "definition": "qc_status_ready",
        "overall": _coverage_row(eligible, config),
        "by_truth_class": {
            label: _coverage_row(
                [row for row in eligible if row["observable_pattern"] == label],
                config,
            )
            for label in config["class_order"]
        },
    }


def _coverage_row(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    prediction_counts = Counter(row["prediction_status"] for row in rows)
    qc_counts = Counter(row["qc_status"] for row in rows)
    total = len(rows)
    qc_ready = qc_counts.get("ready", 0)
    prediction_ready = prediction_counts.get("ready", 0)
    return {
        "total": total,
        "qc_ready": qc_ready,
        "qc_unavailable": qc_counts.get("unavailable", 0),
        "qc_boundary_uncertain": qc_counts.get("boundary_uncertain", 0),
        "coverage": qc_ready / total if total else _NOT_COMPUTABLE,
        "prediction_ready": prediction_ready,
        "prediction_ready_coverage": (
            prediction_ready / total if total else _NOT_COMPUTABLE
        ),
        "prediction_status_counts": {
            name: prediction_counts.get(name, 0)
            for name in config["runtime_statuses"]
        },
    }


def _duration_metrics(
    eligible: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for band in config["duration_bands"]:
        rows = [row for row in eligible if row["duration_band"] == band["name"]]
        metric, _confusion = _metric_bundle(rows, config)
        output[band["name"]] = {
            "minimum_sec": band["minimum_sec"],
            "maximum_sec_exclusive": band["maximum_sec_exclusive"],
            "support": len(rows),
            "coverage": _coverage_row(rows, config),
            "four_class": metric["four_class"],
            "shape_binary": metric["shape_binary"],
        }
    return output


def _group_support(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        name: {
            group_id: _support_row(group_rows, config)
            for group_id, group_rows in _group_rows(rows, field).items()
        }
        for name, field in (
            ("participant", "participant_id"),
            ("session", "session_id"),
            ("camera_setup", "camera_setup_id"),
        )
    }


def _group_rows(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        output.setdefault(str(row[field]), []).append(row)
    return dict(sorted(output.items()))


def _support_row(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    eligible = [row for row in rows if row["shape_truth_status"] == "eligible"]
    coverage = _coverage_row(eligible, config)
    return {
        "truth_total": len(rows),
        "eligible": len(eligible),
        "qc_ready": coverage["qc_ready"],
        "qc_coverage": coverage["coverage"],
        "ready": coverage["prediction_ready"],
        "prediction_ready_coverage": coverage["prediction_ready_coverage"],
        "eligible_by_truth_class": {
            label: sum(row["observable_pattern"] == label for row in eligible)
            for label in config["class_order"]
        },
        "prediction_status_counts_for_shape_eligible": coverage[
            "prediction_status_counts"
        ],
        "purposeful_hard_negative": sum(
            row["purpose_context"] == "purposeful"
            and row["observable_pattern"] in {"pacing", "lapping", "random"}
            for row in eligible
        ),
    }


def _purposeful_diagnostics(
    eligible: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [
        row
        for row in eligible
        if row["purpose_context"] == "purposeful"
        and row["observable_pattern"] in {"pacing", "lapping", "random"}
    ]
    declared_role_count = sum(
        row["evaluation_role"] == "purposeful_hard_negative" for row in rows
    )
    distribution = Counter(
        row["predicted_pattern"]
        if row["prediction_status"] == "ready"
        else "pipeline_miss"
        for row in rows
    )
    binary_distribution = Counter(
        row["predicted_binary_label"]
        if row["prediction_status"] == "ready"
        else "pipeline_miss"
        for row in rows
    )
    return {
        "eligible_count": len(rows),
        "declared_role_count": declared_role_count,
        "role_mismatch_count": len(rows) - declared_role_count,
        "evaluation_role_distribution": dict(
            sorted(Counter(row["evaluation_role"] for row in rows).items())
        ),
        "truth_shape_distribution": dict(
            sorted(Counter(row["observable_pattern"] for row in rows).items())
        ),
        "shape_prediction_distribution": {
            name: distribution.get(name, 0)
            for name in [*config["class_order"], "pipeline_miss"]
        },
        "shape_binary_prediction_distribution": {
            name: binary_distribution.get(name, 0)
            for name in [
                "direct_or_non_wandering",
                "wandering_like",
                "pipeline_miss",
            ]
        },
        "shape_wandering_like_prediction_count": sum(
            row["prediction_status"] == "ready"
            and row["predicted_binary_label"] == "wandering_like"
            for row in rows
        ),
        "alert_metrics_available": False,
        "diagnostic_only": True,
    }


def _failure_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        failure_type: str | None = None
        classification_error_types: list[str] = []
        if row["shape_truth_status"] != "eligible":
            failure_type = "truth_not_shape_eligible"
        elif row["prediction_status"] != "ready":
            failure_type = "pipeline_miss"
        else:
            if row["four_class_metric_disposition"] == "misclassified":
                classification_error_types.append("four_class_misclassified")
            if row["shape_binary_metric_disposition"] == "misclassified":
                classification_error_types.append("shape_binary_misclassified")
            if classification_error_types:
                failure_type = "misclassified"
        if failure_type is None:
            continue
        failures.append(
            {
                "failure_type": failure_type,
                "bundle_id": row["bundle_id"],
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "camera_setup_id": row["camera_setup_id"],
                "episode_id": row["episode_id"],
                "source_video_id": row["source_video_id"],
                "target_track_id": row["target_track_id"],
                "start_sec": row["start_sec"],
                "end_sec_exclusive": row["end_sec_exclusive"],
                "duration_sec": row["duration_sec"],
                "duration_band": row["duration_band"],
                "shape_truth_status": row["shape_truth_status"],
                "shape_truth_reason": row["shape_truth_reason"],
                "observable_pattern": row["observable_pattern"],
                "binary_truth": row["binary_truth"],
                "purpose_context": row["purpose_context"],
                "purpose_evidence": row["purpose_evidence"],
                "evaluation_role": row["evaluation_role"],
                "annotation_status": row["annotation_status"],
                "visibility_quality": row["visibility_quality"],
                "tracking_issue": row["tracking_issue"],
                "prediction_status": row["prediction_status"],
                "predicted_pattern": row["predicted_pattern"],
                "predicted_binary_label": row["predicted_binary_label"],
                "classification_error_types": classification_error_types,
                "qc_status": row["qc_status"],
                "prediction_reason_codes": row["prediction_reason_codes"],
                "qc_reason_codes": row["qc_reason_codes"],
            }
        )
    return sorted(failures, key=lambda row: (row["source_video_id"], row["start_sec"], row["episode_id"]))


def _metric_disposition(
    row: Mapping[str, Any], *, truth_field: str, prediction_field: str
) -> str:
    if row["shape_truth_status"] != "eligible":
        return "not_shape_eligible"
    if row["prediction_status"] != "ready":
        return "pipeline_miss"
    if row[prediction_field] == row[truth_field]:
        return "correct"
    return "misclassified"


def _duration_band(duration: float, config: Mapping[str, Any]) -> str:
    for band in config["duration_bands"]:
        maximum = band["maximum_sec_exclusive"]
        if duration >= float(band["minimum_sec"]) and (
            maximum is None or duration < float(maximum)
        ):
            return str(band["name"])
    raise CameraEpisodeEvaluationError("episode duration does not fit a configured band")


def _unique_by_episode_id(
    rows: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in output:
            raise CameraEpisodeEvaluationError(f"duplicate episode_id in {role}")
        output[episode_id] = row
    return output


def _extend_unique_physical_identities(
    existing: list[tuple[str, int, float, float]],
    rows: Sequence[Mapping[str, Any]],
    *,
    track_field: str,
    tolerance: float,
    role: str,
) -> None:
    for row in rows:
        identity = (
            str(row["source_video_id"]),
            int(row[track_field]),
            float(row["start_sec"]),
            float(row["end_sec_exclusive"]),
        )
        for prior in existing:
            if identity[:2] != prior[:2]:
                continue
            if math.isclose(
                identity[2], prior[2], rel_tol=0.0, abs_tol=tolerance
            ) and math.isclose(
                identity[3], prior[3], rel_tol=0.0, abs_tol=tolerance
            ):
                raise CameraEpisodeEvaluationError(
                    f"duplicate physical episode identity in {role}"
                )
            if (
                identity[2] < prior[3] - tolerance
                and identity[3] > prior[2] + tolerance
            ):
                raise CameraEpisodeEvaluationError(
                    f"overlapping physical episode identity in {role}"
                )
        existing.append(identity)


def _validate_group_identity(
    truth: Mapping[str, Any],
    index_row: Mapping[str, str],
    *,
    source_contexts: dict[str, tuple[str, str]],
    track_participants: dict[tuple[str, int], str],
) -> None:
    source_video_id = str(truth["source_video_id"])
    source_context = (
        index_row["session_id"],
        index_row["camera_setup_id"],
    )
    prior_context = source_contexts.setdefault(source_video_id, source_context)
    if prior_context != source_context:
        raise CameraEpisodeEvaluationError(
            "source video group identity is inconsistent across bundles",
            code="source_group_identity_mismatch",
        )

    track_key = (source_video_id, int(truth["target_track_id"]))
    participant_id = index_row["participant_id"]
    prior_participant = track_participants.setdefault(track_key, participant_id)
    if prior_participant != participant_id:
        raise CameraEpisodeEvaluationError(
            "source track group identity is inconsistent across bundles",
            code="source_group_identity_mismatch",
        )


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve(strict=False)


def _result_sort_key(row: Mapping[str, Any]) -> tuple[str, str, float, str]:
    return (
        str(row["bundle_id"]),
        str(row["source_video_id"]),
        float(row["start_sec"]),
        str(row["episode_id"]),
    )


def _readme_bytes(summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> bytes:
    four = metrics["all_shape_eligible"]["four_class"]
    binary = metrics["all_shape_eligible"]["shape_binary"]
    ready_four = metrics["ready_only_conditional"]["four_class"]
    lines = [
        "# M0-CAM-EP1B oracle-boundary episode evaluation",
        "",
        "This is a descriptive development pilot. It does not evaluate automatic boundaries, alerts, real older adults, or clinical outcomes.",
        "",
        f"- Truth episodes: {summary['truth_count']}",
        f"- Shape-eligible episodes: {summary['shape_eligible_count']}",
        f"- All-eligible four-class macro-F1: {four['macro_f1']}",
        f"- All-eligible shape-binary macro-F1: {binary['macro_f1']}",
        f"- Ready-only conditional four-class macro-F1: {ready_four['macro_f1']}",
        f"- QC coverage: {metrics['qc_coverage']['overall']['coverage']}",
        "- Prediction-ready coverage: "
        f"{metrics['qc_coverage']['overall']['prediction_ready_coverage']}",
        "- Probability calibrated: false",
        "- Alert metrics available: false",
        "",
        "Non-ready eligible episodes remain in the primary denominator as pipeline misses. Purposeful pacing/lapping/random truth remains unchanged in shape metrics.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera episode evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = temporary / relative
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
        raise CameraEpisodeEvaluationError(f"{field} must be a non-empty string")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CameraEpisodeEvaluationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise CameraEpisodeEvaluationError(f"{field} must be a finite number")
    return number


def _reject_json_constant(value: str) -> None:
    raise CameraEpisodeEvaluationError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "CAMERA_EPISODE_EVAL_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_EVAL_RESULT_SCHEMA_VERSION",
    "CAMERA_EPISODE_EVAL_SUMMARY_SCHEMA_VERSION",
    "CameraEpisodeEvaluationBuildResult",
    "CameraEpisodeEvaluationError",
    "build_camera_episode_evaluation_bundle",
    "load_camera_episode_evaluation_config",
]
