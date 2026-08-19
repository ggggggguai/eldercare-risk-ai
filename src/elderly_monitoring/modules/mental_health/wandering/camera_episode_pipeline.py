"""Integrated W5D-02 automatic proposal-to-shape pipeline."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    build_camera_episode_boundary_development_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_evaluation import (
    build_camera_episode_boundary_evaluation_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_import import (
    load_episode_boundaries,
    load_episode_truth,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_inference import (
    build_camera_episode_inference_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_evaluation import (
    build_camera_episode_evaluation_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_inference import (
    build_camera_episode_proposal_inference_bundle,
)


CAMERA_EPISODE_PIPELINE_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-episode-pipeline-config-v1"
)
CAMERA_EPISODE_PIPELINE_SUMMARY_SCHEMA_VERSION = (
    "wandering-camera-episode-pipeline-summary-v1"
)
CAMERA_EPISODE_PIPELINE_MANIFEST_SCHEMA_VERSION = (
    "wandering-camera-episode-pipeline-handoff-manifest-partial-v1"
)
EPISODE_RESULT_SCHEMA_VERSION = "wandering-handoff-episode-result-v1"
MODULE = "mental_health"
_SHA256 = 64
_STATUS_VALUES = {"ready", "uncertain", "unavailable", "error"}
_PREDICTION_STATUSES = {"ready", "unavailable", "boundary_uncertain", "inference_error"}
_PROPOSAL_STATUSES = {"proposed", "uncertain", "rejected_by_qc"}
_BOUNDARY_EVAL_VIEW = "all_locomotion_candidates"
_EPISODE_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "module",
        "record_id",
        "episode_id",
        "person_id",
        "session_id",
        "source_video_id",
        "start_time",
        "end_time",
        "technical_segment_index",
        "start_sec",
        "end_sec_exclusive",
        "duration_seconds",
        "status",
        "proposal_status",
        "run_status",
        "qc_status",
        "reason_codes",
        "binary",
        "four_class",
        "quality_flags",
        "identity",
        "source_refs",
    }
)


class CameraEpisodePipelineError(ValueError):
    """The W5D-02 pipeline input, identity, or output contract is invalid."""


@dataclass(frozen=True)
class CameraEpisodePipelineBuildResult:
    output_dir: Path
    video_count: int
    proposal_count: int
    result_count: int
    status: str
    prediction_coverage: float


def load_camera_episode_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable W5D-02 production configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodePipelineError("cannot read W5D-02 pipeline config") from exc
    if not isinstance(value, Mapping):
        raise CameraEpisodePipelineError("W5D-02 pipeline config must be a mapping")
    config = dict(value)
    expected = {
        "schema_version",
        "pipeline_id",
        "development",
        "segmenter",
        "shape_bridge",
        "fixed_primary",
        "evaluation",
        "gates",
        "outputs",
    }
    if set(config) != expected:
        raise CameraEpisodePipelineError("W5D-02 pipeline config fields drifted")
    if config["schema_version"] != CAMERA_EPISODE_PIPELINE_CONFIG_SCHEMA_VERSION:
        raise CameraEpisodePipelineError("W5D-02 pipeline config schema drifted")
    if config["pipeline_id"] != "w5d02-b01-b02-automatic-episode-shape-v1":
        raise CameraEpisodePipelineError("W5D-02 pipeline identity drifted")
    development = _mapping(config, "development")
    if development.get("expected_batch_counts") not in ({"B01": 36, "B02": 12},):
        raise CameraEpisodePipelineError("W5D-02 development batch counts drifted")
    _descriptor_config(development, "index")
    _nonempty(development.get("index_schema_id"), "index_schema_id")
    segmenter = _mapping(config, "segmenter")
    if segmenter.get("profile_id") != "closing-s008-d04":
        raise CameraEpisodePipelineError("W5D-01 selected profile drifted")
    _descriptor_config(segmenter, "profile")
    bridge = _mapping(config, "shape_bridge")
    _descriptor_config(bridge, "config")
    if bridge.get("policy_id") != "m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2":
        raise CameraEpisodePipelineError("shape bridge policy drifted")
    primary = _mapping(config, "fixed_primary")
    if primary.get("candidate_id") != "topowander-m0s-seed20260731-epoch0005":
        raise CameraEpisodePipelineError("fixed primary candidate drifted")
    if primary.get("binary_decision_threshold") != 0.5:
        raise CameraEpisodePipelineError("fixed primary binary threshold drifted")
    _digest(primary.get("model_state_sha256"), "model_state_sha256")
    _descriptor_config(primary, "candidate_manifest")
    _descriptor_config(primary, "episode_config")
    _descriptor_config(primary, "primary_config")
    evaluation = _mapping(config, "evaluation")
    _descriptor_config(evaluation, "boundary_config")
    _descriptor_config(evaluation, "episode_config")
    _descriptor_config(evaluation, "episode_result_schema")
    gates = _mapping(config, "gates")
    minimum = gates.get("known_episode_prediction_coverage_minimum")
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool) or not 0 < float(minimum) <= 1:
        raise CameraEpisodePipelineError("prediction coverage gate is invalid")
    outputs = _mapping(config, "outputs")
    if outputs.get("required_files") != [
        "episode_results.jsonl",
        "run_summary.json",
        "handoff_manifest.partial.json",
        "README.md",
        "VERIFICATION.md",
    ]:
        raise CameraEpisodePipelineError("W5D-02 output file contract drifted")
    return config


def build_camera_episode_pipeline(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> CameraEpisodePipelineBuildResult:
    """Run proposal, QC, 80-point preprocessing, shape inference, and reports."""

    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"W5D-02 output directory already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    if not config_file.is_relative_to(root):
        raise CameraEpisodePipelineError("pipeline config must be inside project root")
    config = load_camera_episode_pipeline_config(config_file)
    config_sha256 = _sha256_file(config_file)
    files = _load_and_verify_bindings(root, config)
    rows = _load_development_rows(files["index"], config)
    files["boundaries"] = {
        str(row["source_video_id"]): Path(row["_boundary_path"])
        for row in rows
    }
    if run_id is None:
        run_id = "w5d02-" + hashlib.sha256(
            canonical_jsonl_bytes(rows)
        ).hexdigest()[:16]
    _token(run_id, "run_id")
    started_at = (now or _utc_now)()
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)

    work_parent = output.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.work-", dir=work_parent))
    try:
        proposal_root = work / "proposals"
        proposal_root.mkdir()
        proposal_index_rows: list[dict[str, str]] = []
        boundary_index_rows: list[dict[str, str]] = []
        oracle_index_rows: list[dict[str, str]] = []
        row_context: dict[str, dict[str, Any]] = {}
        for row in rows:
            source_video_id = str(row["source_video_id"])
            bundle_dir = proposal_root / source_video_id
            build_camera_episode_boundary_development_bundle(
                project_root=root,
                development_profile_path=files["profile"],
                tracking_jsonl_path=row["artifacts"]["tracking"]["path"],
                media_sidecar_path=row["artifacts"]["media_sidecar"]["path"],
                output_dir=bundle_dir,
            )
            sidecar = _load_json(row["artifacts"]["media_sidecar"]["path"], "media sidecar")
            sidecar_setup = str(sidecar["setup_id"])
            proposal_binding = {
                "bundle_id": f"{config['segmenter']['profile_id']}-{source_video_id}",
                "proposal_bundle_dir": bundle_dir.as_posix(),
                "tracking_jsonl": row["artifacts"]["tracking"]["path"].as_posix(),
                "media_sidecar": row["artifacts"]["media_sidecar"]["path"].as_posix(),
                "participant_id": str(row["participant_id"]),
                "session_id": str(row["session_id"]),
                # The bridge validates against sidecar identity. The canonical index
                # setup remains in row_context for output and evaluation binding.
                "camera_setup_id": sidecar_setup,
                "clock_domain_id": str(row["clock_domain_id"]),
            }
            proposal_index_rows.append(proposal_binding)
            boundary_index_rows.append(
                {
                    "bundle_id": proposal_binding["bundle_id"],
                    "proposal_bundle_dir": bundle_dir.as_posix(),
                    "human_boundary_jsonl": files["boundaries"][source_video_id].as_posix(),
                    "media_sidecar": row["artifacts"]["media_sidecar"]["path"].as_posix(),
                    "participant_id": str(row["participant_id"]),
                    "session_id": str(row["session_id"]),
                    "camera_setup_id": str(row["camera_setup_id"]),
                    "clock_domain_id": str(row["clock_domain_id"]),
                    "truth_source": "independent_human",
                }
            )
            row_context[source_video_id] = {
                "row": row,
                "sidecar": sidecar,
                "boundary_path": files["boundaries"][source_video_id],
            }

        proposal_index = work / "proposal_batch_index.jsonl"
        proposal_index.write_bytes(canonical_jsonl_bytes(proposal_index_rows))
        shape_work = work / "shape_bridge"
        build_camera_episode_proposal_inference_bundle(
            project_root=root,
            config_path=files["shape_config"],
            batch_index_path=proposal_index,
            output_dir=shape_work,
        )
        shape_rows = _load_jsonl(shape_work / "proposal_shape_predictions.jsonl", "shape predictions")
        expected_proposal_count = sum(
            int(json.loads((Path(binding["proposal_bundle_dir"]) / "summary.json").read_text(encoding="utf-8"))["proposal_count"])
            for binding in proposal_index_rows
        )
        if len(shape_rows) != expected_proposal_count:
            raise CameraEpisodePipelineError("shape result count differs from proposal count")

        boundary_index = work / "automatic_boundary_batch_index.jsonl"
        boundary_index.write_bytes(canonical_jsonl_bytes(boundary_index_rows))
        boundary_eval = work / "automatic_boundary_evaluation"
        build_camera_episode_boundary_evaluation_bundle(
            project_root=root,
            config_path=files["boundary_config"],
            batch_index_path=boundary_index,
            output_dir=boundary_eval,
        )
        automatic_metrics = _evaluate_automatic_binary(
            truth_rows=_load_truth_rows(row_context),
            proposal_shape_rows=shape_rows,
            match_rows=_load_jsonl(boundary_eval / "matches.jsonl", "automatic matches"),
            minimum_prediction_coverage=float(config["gates"]["known_episode_prediction_coverage_minimum"]),
        )

        oracle_root = work / "oracle"
        oracle_root.mkdir()
        for row in rows:
            source_video_id = str(row["source_video_id"])
            prediction_dir = oracle_root / source_video_id
            build_camera_episode_inference_bundle(
                project_root=root,
                episode_config_path=files["episode_config"],
                tracking_jsonl_path=row["artifacts"]["tracking"]["path"],
                media_sidecar_path=row["artifacts"]["media_sidecar"]["path"],
                candidate_manifest_path=files["candidate_manifest"],
                expected_manifest_sha256=str(config["fixed_primary"]["candidate_manifest"]["sha256"]),
                output_dir=prediction_dir,
                episode_boundaries_path=files["boundaries"][source_video_id],
            )
            oracle_index_rows.append(
                {
                    "bundle_id": f"oracle-{source_video_id}",
                    "prediction_bundle_dir": prediction_dir.as_posix(),
                    "truth_jsonl": row["artifacts"]["truth"]["path"].as_posix(),
                    "participant_id": str(row["participant_id"]),
                    "session_id": str(row["session_id"]),
                    "camera_setup_id": str(row["camera_setup_id"]),
                }
            )
        oracle_index = work / "oracle_batch_index.jsonl"
        oracle_index.write_bytes(canonical_jsonl_bytes(oracle_index_rows))
        oracle_eval = work / "oracle_evaluation"
        build_camera_episode_evaluation_bundle(
            config_path=files["episode_eval_config"],
            batch_index_path=oracle_index,
            output_dir=oracle_eval,
        )
        oracle_metrics = _load_json(oracle_eval / "metrics.json", "oracle metrics")
        oracle_summary = _load_json(oracle_eval / "summary.json", "oracle summary")

        episode_results = [
            _build_episode_result(
                prediction=row,
                index_row=row_context[str(row["source_video_id"])] ["row"],
                config_id=config["pipeline_id"],
                config_sha256=config_sha256,
                profile_id=str(config["segmenter"]["profile_id"]),
                profile_sha256=str(config["segmenter"]["profile"]["sha256"]),
                context=row_context[str(row["source_video_id"])],
                index_descriptor=files["index_descriptor"],
            )
            for row in shape_rows
        ]
        episode_results.sort(
            key=lambda row: (
                str(row["source_video_id"]),
                int(row["technical_segment_index"]),
                float(row["start_sec"]),
                str(row["episode_id"]),
            )
        )
        for episode_result in episode_results:
            _validate_episode_result(episode_result)
        finished_at = (now or _utc_now)()
        if finished_at.tzinfo is None:
            finished_at = finished_at.replace(tzinfo=timezone.utc)
        status = _run_status(episode_results, automatic_metrics, config)
        run_summary = _build_run_summary(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            rows=rows,
            episode_results=episode_results,
            automatic_metrics=automatic_metrics,
            oracle_summary=oracle_summary,
            oracle_metrics=oracle_metrics,
            config=config,
            config_sha256=config_sha256,
            profile_sha256=str(config["segmenter"]["profile"]["sha256"]),
            index_descriptor=files["index_descriptor"],
        )
        episode_payload = canonical_jsonl_bytes(episode_results)
        summary_payload = canonical_json_bytes(run_summary)
        verification_text = _verification_text(
            config=config,
            config_sha256=config_sha256,
            rows=rows,
            shape_rows=shape_rows,
            automatic_metrics=automatic_metrics,
            oracle_metrics=oracle_metrics,
            oracle_summary=oracle_summary,
            run_summary=run_summary,
            files=files,
        ).encode("utf-8")
        readme_text = _readme_text(
            status=status,
            run_summary=run_summary,
            automatic_metrics=automatic_metrics,
            oracle_metrics=oracle_metrics,
        ).encode("utf-8")
        artifact_payloads = {
            "episode_results.jsonl": episode_payload,
            "run_summary.json": summary_payload,
            "README.md": readme_text,
            "VERIFICATION.md": verification_text,
        }
        manifest = _build_partial_manifest(
            run_id=run_id,
            status=status,
            artifact_payloads=artifact_payloads,
            config=config,
            config_sha256=config_sha256,
            profile_sha256=str(config["segmenter"]["profile"]["sha256"]),
            index_descriptor=files["index_descriptor"],
            automatic_metrics=automatic_metrics,
        )
        artifact_payloads["handoff_manifest.partial.json"] = canonical_json_bytes(manifest)
        _commit_new_directory(output, artifact_payloads)
        return CameraEpisodePipelineBuildResult(
            output_dir=output,
            video_count=len(rows),
            proposal_count=len(shape_rows),
            result_count=len(episode_results),
            status=status,
            prediction_coverage=float(
                automatic_metrics["pooled"]["ready_uncertain_prediction_coverage"]
            ),
        )
    except CameraEpisodePipelineError:
        raise
    except Exception as exc:
        raise CameraEpisodePipelineError("W5D-02 pipeline failed before atomic commit") from exc
    finally:
        if work.exists():
            shutil.rmtree(work)


def _build_episode_result(
    *,
    prediction: Mapping[str, Any],
    index_row: Mapping[str, Any],
    config_id: str,
    config_sha256: str,
    profile_id: str,
    profile_sha256: str,
    context: Mapping[str, Any] | None = None,
    index_descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one historical bridge row into the frozen W5D episode contract."""

    proposal_status = str(prediction.get("proposal_status"))
    if proposal_status not in _PROPOSAL_STATUSES:
        raise CameraEpisodePipelineError("proposal status is invalid")
    prediction_status = str(prediction.get("prediction_status"))
    if prediction_status not in _PREDICTION_STATUSES:
        raise CameraEpisodePipelineError("prediction status is invalid")
    qc_status = _normalize_status(prediction.get("qc_status"), fallback=prediction_status)
    if proposal_status == "rejected_by_qc":
        status = "unavailable"
        run_status = "rejected"
    elif prediction_status == "ready":
        status = "uncertain" if proposal_status == "uncertain" else "ready"
        run_status = "uncertain" if proposal_status == "uncertain" else "auto_accepted"
    elif prediction_status == "inference_error":
        status = "error"
        run_status = "rejected"
    elif prediction_status == "boundary_uncertain":
        status = "uncertain"
        run_status = "rejected"
    else:
        status = "unavailable"
        run_status = "rejected"
    if context is None:
        context = {"row": index_row, "sidecar": {}}
    row = context["row"]
    sidecar = context.get("sidecar", {})
    binary = _prediction_with_threshold(prediction.get("binary"), threshold=0.5)
    four_class = _prediction_with_threshold(prediction.get("four_class"), threshold=None)
    reason_codes = _unique_strings(
        [
            *list(prediction.get("reason_codes", [])),
            *list(prediction.get("hard_break_reasons", [])),
            *list(prediction.get("qc_reason_codes", [])),
            *list(prediction.get("prediction_reason_codes", [])),
        ]
    )
    quality_flags = _unique_strings(
        [
            *list(prediction.get("quality_flags", [])),
            "automatic_boundary_not_human_accepted",
            "binary_primary_result",
            "four_class_diagnostic",
        ]
    )
    if proposal_status == "uncertain":
        quality_flags = _unique_strings([*quality_flags, "boundary_uncertain_candidate"])
    if prediction.get("model_invocation_skipped"):
        quality_flags = _unique_strings([*quality_flags, "model_invocation_skipped"])
    source_refs = [
        _source_ref("development_index", "w5d00-development-index", index_descriptor),
        _source_ref_from_artifact(
            "tracking", str(row["source_video_id"]), row["artifacts"]["tracking"]
        ),
        _source_ref_from_artifact(
            "media_sidecar", str(row["source_video_id"]), row["artifacts"]["media_sidecar"]
        ),
        _source_ref_from_artifact(
            "truth", str(row["source_video_id"]), row["artifacts"]["truth"]
        ),
    ]
    if context.get("boundary_path") is not None:
        boundary_path = Path(context["boundary_path"])
        source_refs.append(
            {
                "ref_type": "boundary_truth",
                "ref_id": str(row["source_video_id"]),
                "artifact_path": boundary_path.as_posix(),
                "sha256": _sha256_file(boundary_path),
            }
        )
    start = float(prediction["start_sec"])
    end = float(prediction["end_sec_exclusive"])
    return {
        "schema_version": EPISODE_RESULT_SCHEMA_VERSION,
        "module": MODULE,
        "record_id": f"w5d02-{prediction['proposal_id']}",
        "episode_id": str(prediction["proposal_id"]),
        "person_id": str(row["participant_id"]),
        "session_id": str(row["session_id"]),
        "source_video_id": str(row["source_video_id"]),
        "start_time": _timestamp_for_interval(sidecar.get("capture_started_at"), start),
        "end_time": _timestamp_for_interval(sidecar.get("capture_started_at"), end),
        "technical_segment_index": int(prediction["technical_segment_index"]),
        "start_sec": start,
        "end_sec_exclusive": end,
        "duration_seconds": float(prediction["duration_sec"]),
        "status": status,
        "proposal_status": proposal_status,
        "run_status": run_status,
        "qc_status": qc_status,
        "reason_codes": reason_codes,
        "binary": binary,
        "four_class": four_class,
        "quality_flags": quality_flags,
        "identity": {
            "model_id": prediction.get("candidate_id"),
            "model_sha256": prediction.get("model_state_sha256"),
            "config_id": config_id,
            "config_sha256": config_sha256,
            "policy_id": profile_id,
            "policy_sha256": profile_sha256,
        },
        "source_refs": source_refs,
    }


def _evaluate_automatic_binary(
    *,
    truth_rows: Sequence[Mapping[str, Any]],
    proposal_shape_rows: Sequence[Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, Any]],
    minimum_prediction_coverage: float,
) -> dict[str, Any]:
    """Report automatic binary quality, ready+uncertain overlap coverage, and misses."""

    by_proposal = {str(row["proposal_id"]): row for row in proposal_shape_rows}
    known = [
        row
        for row in truth_rows
        if row.get("annotation_status", "accepted") != "excluded"
        and row.get("evaluation_role", "ordinary_negative") != "excluded"
        and row.get("tracking_issue") != "wrong_target"
    ]
    match_by_truth: dict[str, Mapping[str, Any]] = {}
    for match in match_rows:
        if match.get("view") != _BOUNDARY_EVAL_VIEW:
            continue
        match_by_truth.setdefault(str(match["episode_id"]), match)
    outputs: dict[str, Any] = {}
    for cohort in ("B01", "B02", "pooled"):
        cohort_truth = [
            row for row in known if cohort == "pooled" or row.get("batch_id") == cohort
        ]
        ready_overlap = 0
        classification: list[tuple[str, str]] = []
        matched_prediction_ready = 0
        miss_reasons: Counter[str] = Counter()
        for truth in cohort_truth:
            candidates = [
                row
                for row in proposal_shape_rows
                if str(row.get("source_video_id")) == str(truth["source_video_id"])
                and int(row.get("track_id", -1)) == int(truth["target_track_id"])
                and str(row.get("proposal_status")) in {"proposed", "uncertain"}
                and _positive_overlap(row, truth)
            ]
            ready_candidates = [
                row
                for row in candidates
                if str(row.get("prediction_status")) == "ready"
                and isinstance(row.get("binary"), Mapping)
            ]
            if ready_candidates:
                ready_overlap += 1
            match = match_by_truth.get(str(truth["episode_id"]))
            if match is None:
                miss_reasons["no_automatic_boundary_match"] += 1
                continue
            proposal = by_proposal.get(str(match["proposal_id"]))
            if proposal is None or str(proposal.get("prediction_status")) != "ready":
                miss_reasons["matched_proposal_not_prediction_ready"] += 1
                continue
            binary = proposal.get("binary")
            if not isinstance(binary, Mapping):
                miss_reasons["matched_proposal_missing_binary"] += 1
                continue
            matched_prediction_ready += 1
            if truth.get("annotation_status", "accepted") != "accepted":
                continue
            truth_label = (
                "direct_or_non_wandering"
                if str(truth["observable_pattern"]) == "direct"
                else "wandering_like"
            )
            classification.append((truth_label, str(binary["predicted_label"])))
        support = len(cohort_truth)
        coverage = ready_overlap / support if support else 0.0
        accepted_support = sum(
            truth.get("annotation_status", "accepted") == "accepted"
            for truth in cohort_truth
        )
        outputs[cohort] = {
            "known_episode_support": support,
            "ready_uncertain_prediction_overlap_count": ready_overlap,
            "ready_uncertain_prediction_coverage": coverage,
            "prediction_coverage_gate_satisfied": coverage >= minimum_prediction_coverage,
            "known_episode_prediction_coverage_miss_count": support - ready_overlap,
            "known_episode_match_pipeline_miss_count": support - matched_prediction_ready,
            "shape_binary_eligible_support": accepted_support,
            "pipeline_miss_count": accepted_support - len(classification),
            "pipeline_miss_reasons": dict(sorted(miss_reasons.items())),
            "automatic_binary": _binary_metrics(classification),
            "automatic_binary_population": "automatic_boundary_matches_prediction_ready",
        }
    return outputs


def _validate_episode_result(value: Mapping[str, Any]) -> None:
    """Validate the W5D episode result subset without adding a runtime dependency."""

    if not isinstance(value, Mapping) or frozenset(value) != _EPISODE_RESULT_FIELDS:
        raise CameraEpisodePipelineError("episode result fields drift from W5D schema")
    if value["schema_version"] != EPISODE_RESULT_SCHEMA_VERSION or value["module"] != MODULE:
        raise CameraEpisodePipelineError("episode result schema identity is invalid")
    for name in ("record_id", "episode_id", "person_id", "session_id", "source_video_id"):
        _nonempty(value[name], f"episode result {name}")
    if not isinstance(value["technical_segment_index"], int) or value["technical_segment_index"] < 0:
        raise CameraEpisodePipelineError("episode result technical segment is invalid")
    start = _finite_nonnegative(value["start_sec"], "episode result start")
    end = _finite_positive(value["end_sec_exclusive"], "episode result end")
    duration = _finite_positive(value["duration_seconds"], "episode result duration")
    if start >= end or not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-9):
        raise CameraEpisodePipelineError("episode result endpoints are inconsistent")
    if value["status"] not in _STATUS_VALUES or value["qc_status"] not in _STATUS_VALUES:
        raise CameraEpisodePipelineError("episode result status is invalid")
    if value["proposal_status"] not in _PROPOSAL_STATUSES:
        raise CameraEpisodePipelineError("episode result proposal status is invalid")
    if value["run_status"] not in {"auto_accepted", "uncertain", "rejected"}:
        raise CameraEpisodePipelineError("episode result run status is invalid")
    _validate_prediction(value["binary"], threshold=0.5, role="binary")
    _validate_prediction(value["four_class"], threshold=None, role="four class")
    for name in ("reason_codes", "quality_flags"):
        values = value[name]
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item for item in values
        ) or len(values) != len(set(values)):
            raise CameraEpisodePipelineError(f"episode result {name} is invalid")
    identity = value["identity"]
    expected_identity = {
        "model_id",
        "model_sha256",
        "config_id",
        "config_sha256",
        "policy_id",
        "policy_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_identity:
        raise CameraEpisodePipelineError("episode result identity is invalid")
    for name in ("config_sha256", "policy_sha256"):
        _digest(identity[name], f"episode result identity {name}")
    if identity["model_sha256"] is not None:
        _digest(identity["model_sha256"], "episode result model sha256")
    source_refs = value["source_refs"]
    if not isinstance(source_refs, list) or not source_refs:
        raise CameraEpisodePipelineError("episode result source refs are invalid")
    for ref in source_refs:
        if not isinstance(ref, Mapping) or set(ref) != {
            "ref_type", "ref_id", "artifact_path", "sha256"
        }:
            raise CameraEpisodePipelineError("episode result source ref fields are invalid")
        _nonempty(ref["ref_type"], "episode result source ref type")
        _nonempty(ref["ref_id"], "episode result source ref ID")
        if ref["sha256"] is not None:
            _digest(ref["sha256"], "episode result source ref sha256")
    if value["run_status"] == "auto_accepted" and (
        value["status"] != "ready"
        or value["proposal_status"] != "proposed"
        or value["binary"] is None
    ):
        raise CameraEpisodePipelineError("auto accepted result is semantically invalid")
    if value["proposal_status"] == "rejected_by_qc" and (
        value["status"] != "unavailable" or value["run_status"] != "rejected"
    ):
        raise CameraEpisodePipelineError("rejected proposal result is semantically invalid")


def _validate_prediction(value: Any, *, threshold: float | None, role: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) != {
        "class_order", "probabilities", "predicted_label", "decision_threshold"
    }:
        raise CameraEpisodePipelineError(f"episode result {role} prediction is invalid")
    classes = value["class_order"]
    probabilities = value["probabilities"]
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(item, str) and item for item in classes)
        or not isinstance(probabilities, list)
        or len(probabilities) != len(classes)
        or any(not isinstance(item, (int, float)) or not 0 <= float(item) <= 1 for item in probabilities)
        or value["predicted_label"] not in classes
        or value["decision_threshold"] != threshold
    ):
        raise CameraEpisodePipelineError(f"episode result {role} prediction values are invalid")


def _finite_nonnegative(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise CameraEpisodePipelineError(f"{role} is invalid")
    return float(value)


def _finite_positive(value: Any, role: str) -> float:
    number = _finite_nonnegative(value, role)
    if number <= 0:
        raise CameraEpisodePipelineError(f"{role} must be positive")
    return number


def _build_run_summary(
    *,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    rows: Sequence[Mapping[str, Any]],
    episode_results: Sequence[Mapping[str, Any]],
    automatic_metrics: Mapping[str, Any],
    oracle_summary: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    config_sha256: str,
    profile_sha256: str,
    index_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in episode_results)
    proposal_counts = Counter(str(row["proposal_status"]) for row in episode_results)
    failures = {
        "proposal_rejected_by_qc": proposal_counts.get("rejected_by_qc", 0),
        "prediction_unavailable": counts.get("unavailable", 0),
        "prediction_inference_error": counts.get("error", 0),
        "automatic_binary_pipeline_miss": int(automatic_metrics["pooled"]["pipeline_miss_count"]),
        "automatic_known_episode_coverage_miss": int(
            automatic_metrics["pooled"]["known_episode_prediction_coverage_miss_count"]
        ),
        "automatic_known_episode_match_miss": int(
            automatic_metrics["pooled"]["known_episode_match_pipeline_miss_count"]
        ),
        "oracle_binary_pipeline_miss": int(
            oracle_metrics["all_shape_eligible"]["shape_binary"]["pipeline_miss_count"]
        ),
    }
    quality_flags = ["automatic_boundary_not_human_accepted", "development_only"]
    if not all(
        bool(automatic_metrics[cohort]["prediction_coverage_gate_satisfied"])
        for cohort in ("B01", "B02", "pooled")
    ):
        quality_flags.append("known_episode_prediction_coverage_below_gate")
    if not bool(config["segmenter"].get("selection_gate_satisfied", False)):
        quality_flags.append("segmenter_selection_gate_not_satisfied")
    return {
        "schema_version": "wandering-handoff-run-summary-v1",
        "module": MODULE,
        "run_id": run_id,
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "person_id": _single_or_none(row["participant_id"] for row in rows),
        "session_id": _single_or_none(row["session_id"] for row in rows),
        "source_video_id": None,
        "status": status,
        "input_counts": {
            "development_videos": len(rows),
            "B01_videos": sum(row["batch_id"] == "B01" for row in rows),
            "B02_videos": sum(row["batch_id"] == "B02" for row in rows),
            "known_episode_support": int(automatic_metrics["pooled"]["known_episode_support"]),
        },
        "output_counts": {
            "episode_results": len(episode_results),
            "proposal_results": len(episode_results),
            "proposal_proposed": proposal_counts.get("proposed", 0),
            "proposal_uncertain": proposal_counts.get("uncertain", 0),
            "proposal_rejected_by_qc": proposal_counts.get("rejected_by_qc", 0),
            "model_forward_invocations": sum(
                row["binary"] is not None for row in episode_results
            ),
            "model_invocation_skipped": sum(
                row["binary"] is None for row in episode_results
            ),
            "ready": counts.get("ready", 0),
            "uncertain": counts.get("uncertain", 0),
            "unavailable": counts.get("unavailable", 0),
            "error": counts.get("error", 0),
            "binary_prediction_ready": sum(row["binary"] is not None for row in episode_results),
            "four_class_prediction_ready": sum(row["four_class"] is not None for row in episode_results),
        },
        "failure_counts": failures,
        "degraded_components": [
            "automatic_boundary_segmenter"
            if not bool(config["segmenter"].get("selection_gate_satisfied", False))
            else "none",
        ],
        "quality_flags": _unique_strings(quality_flags),
        "identity": {
            "model_id": config["fixed_primary"]["candidate_id"],
            "model_sha256": config["fixed_primary"]["model_state_sha256"],
            "config_id": config["pipeline_id"],
            "config_sha256": config_sha256,
            "policy_id": config["segmenter"]["profile_id"],
            "policy_sha256": profile_sha256,
        },
        "source_refs": [
            _source_ref("development_index", "w5d00-development-index", index_descriptor),
            _source_ref("segmenter_profile", config["segmenter"]["profile_id"], config["segmenter"]["profile"]),
        ],
    }


def _build_partial_manifest(
    *,
    run_id: str,
    status: str,
    artifact_payloads: Mapping[str, bytes],
    config: Mapping[str, Any],
    config_sha256: str,
    profile_sha256: str,
    index_descriptor: Mapping[str, Any],
    automatic_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = {
        name: {
            "schema_version": (
                EPISODE_RESULT_SCHEMA_VERSION
                if name == "episode_results.jsonl"
                else "wandering-handoff-run-summary-v1"
                if name == "run_summary.json"
                else None
            ),
            "record_count": (
                len(artifact_payloads[name].splitlines())
                if name == "episode_results.jsonl"
                else 1
                if name == "run_summary.json"
                else None
            ),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(artifact_payloads.items())
    }
    return {
        "schema_version": CAMERA_EPISODE_PIPELINE_MANIFEST_SCHEMA_VERSION,
        "module": MODULE,
        "handoff_id": run_id,
        "stage": "W5D-02",
        "status": status,
        "algorithm_event_emitted": False,
        "pending_files": [
            "context_reviews.jsonl",
            "daily_reports.jsonl",
            "baseline_profiles.jsonl",
            "baseline_deviations.jsonl",
            "handoff_manifest.json",
        ],
        "artifacts": artifacts,
        "evaluation": {
            "automatic_binary": automatic_metrics["pooled"]["automatic_binary"],
            "ready_uncertain_prediction_coverage": automatic_metrics["pooled"][
                "ready_uncertain_prediction_coverage"
            ],
            "known_episode_prediction_coverage_gate_satisfied": automatic_metrics["pooled"][
                "prediction_coverage_gate_satisfied"
            ],
            "oracle_boundary_reported_separately": True,
            "pipeline_miss_reported_separately": True,
        },
        "identity": {
            "model_id": config["fixed_primary"]["candidate_id"],
            "model_sha256": config["fixed_primary"]["model_state_sha256"],
            "config_id": config["pipeline_id"],
            "config_sha256": config_sha256,
            "policy_id": config["segmenter"]["profile_id"],
            "policy_sha256": profile_sha256,
        },
        "source_refs": [_source_ref("development_index", "w5d00-development-index", index_descriptor)],
        "quality_flags": ["automatic_boundary_not_human_accepted", "development_only"],
    }


def _load_and_verify_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    development = config["development"]
    index_path = _verify_descriptor(root, development["index"], "development index")
    index_descriptor = {**development["index"], "path": index_path}
    profile = _verify_descriptor(root, config["segmenter"]["profile"], "selected segmenter profile")
    shape_config = _verify_descriptor(root, config["shape_bridge"]["config"], "shape bridge config")
    candidate_manifest = _verify_descriptor(root, config["fixed_primary"]["candidate_manifest"], "candidate manifest")
    episode_config = _verify_descriptor(root, config["fixed_primary"]["episode_config"], "episode config")
    primary_config = _verify_descriptor(root, config["fixed_primary"]["primary_config"], "primary config")
    boundary_config = _verify_descriptor(root, config["evaluation"]["boundary_config"], "boundary evaluation config")
    episode_eval_config = _verify_descriptor(root, config["evaluation"]["episode_config"], "episode evaluation config")
    episode_result_schema = _verify_descriptor(root, config["evaluation"]["episode_result_schema"], "episode result schema")
    schema = _load_json(episode_result_schema, "episode result schema")
    if schema.get("$id") != EPISODE_RESULT_SCHEMA_VERSION:
        raise CameraEpisodePipelineError("episode result schema identity drifted")
    profile_value = _load_yaml(profile, "selected segmenter profile")
    if profile_value.get("profile_id") != config["segmenter"]["profile_id"]:
        raise CameraEpisodePipelineError("selected segmenter profile ID drifted")
    return {
        "index": index_path,
        "index_descriptor": index_descriptor,
        "profile": profile,
        "profile_descriptor": profile,
        "shape_config": shape_config,
        "candidate_manifest": candidate_manifest,
        "episode_config": episode_config,
        "primary_config": primary_config,
        "boundary_config": boundary_config,
        "episode_eval_config": episode_eval_config,
        "episode_result_schema": episode_result_schema,
        "profile_descriptor_raw": config["segmenter"]["profile"],
        "boundaries": {},
    }


def _load_development_rows(index_path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _load_jsonl(index_path, "development index")
    expected = config["development"]["expected_batch_counts"]
    counts = Counter(str(row.get("batch_id")) for row in rows)
    if dict(sorted(counts.items())) != dict(sorted(expected.items())):
        raise CameraEpisodePipelineError("development index batch counts drifted")
    ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "wandering-camera-development-index-v1":
            raise CameraEpisodePipelineError("development index schema drifted")
        if row.get("dataset_role") != "development" or row.get("development_reuse_allowed") is not True:
            raise CameraEpisodePipelineError("development index role drifted")
        source_video_id = _nonempty(row.get("source_video_id"), "source_video_id")
        if source_video_id in ids:
            raise CameraEpisodePipelineError("development source_video_id is duplicated")
        ids.add(source_video_id)
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"video", "tracking", "media_sidecar", "truth", "cvat_xml"}:
            raise CameraEpisodePipelineError("development artifact set drifted")
        for name in artifacts:
            descriptor = artifacts[name]
            path = _verify_descriptor_value(descriptor, f"development {name}")
            artifacts[name] = {**descriptor, "path": path}
        truth_path = Path(artifacts["truth"]["path"])
        boundary_path = truth_path.with_name("episode_boundaries.jsonl").resolve(strict=True)
        if not boundary_path.is_file():
            raise CameraEpisodePipelineError(f"boundary truth is missing for {source_video_id}")
        row["artifacts"] = dict(artifacts)
        row["_boundary_path"] = boundary_path
    return sorted(rows, key=lambda row: str(row["source_video_id"]))


def _load_truth_rows(contexts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_video_id, context in contexts.items():
        truth_rows = load_episode_truth(context["row"]["artifacts"]["truth"]["path"])
        for truth in truth_rows:
            row = dict(truth)
            row["batch_id"] = context["row"]["batch_id"]
            rows.append(row)
    return rows


def _prediction_with_threshold(value: Any, *, threshold: float | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CameraEpisodePipelineError("model prediction object is invalid")
    result = dict(value)
    result["decision_threshold"] = threshold
    return result


def _run_status(
    episode_results: Sequence[Mapping[str, Any]],
    automatic_metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    if any(row["status"] == "error" for row in episode_results):
        return "error"
    if any(row["status"] == "unavailable" for row in episode_results):
        return "uncertain"
    if not bool(automatic_metrics["pooled"]["prediction_coverage_gate_satisfied"]):
        return "uncertain"
    if not bool(config["segmenter"].get("selection_gate_satisfied", False)):
        return "uncertain"
    return "ready"


def _binary_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    labels = ["direct_or_non_wandering", "wandering_like"]
    support = len(pairs)
    correct = sum(true == pred for true, pred in pairs)
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(true == label and pred == label for true, pred in pairs)
        predicted = sum(pred == label for _true, pred in pairs)
        true_count = sum(true == label for true, _pred in pairs)
        precision = tp / predicted if predicted else 0.0
        recall = tp / true_count if true_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if true_count:
            f1_values.append(f1)
        per_class[label] = {
            "support": true_count,
            "predicted_count": predicted,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "class_order": labels,
        "support": support,
        "correct": correct,
        "accuracy": correct / support if support else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
    }


def _positive_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        float(left["start_sec"]) < float(right["end_sec_exclusive"])
        and float(left["end_sec_exclusive"]) > float(right["start_sec"])
    )


def _source_ref(ref_type: str, ref_id: str, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    if descriptor is None:
        return {
            "ref_type": ref_type,
            "ref_id": ref_id,
            "artifact_path": None,
            "sha256": None,
        }
    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "artifact_path": str(descriptor["path"]),
        "sha256": str(descriptor["sha256"]),
    }


def _source_ref_from_artifact(ref_type: str, ref_id: str, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return _source_ref(ref_type, ref_id, descriptor)


def _verify_descriptor(root: Path, descriptor: Mapping[str, Any], role: str) -> Path:
    if not isinstance(descriptor, Mapping):
        raise CameraEpisodePipelineError(f"{role} descriptor is invalid")
    path_value = _nonempty(descriptor.get("path"), f"{role} path")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if _sha256_file(path) != descriptor.get("sha256"):
        raise CameraEpisodePipelineError(f"{role} hash drifted")
    return path


def _verify_descriptor_value(value: Mapping[str, Any], role: str) -> Path:
    if not isinstance(value, Mapping):
        raise CameraEpisodePipelineError(f"{role} descriptor is invalid")
    path = Path(_nonempty(value.get("path"), f"{role} path")).resolve(strict=True)
    if _sha256_file(path) != value.get("sha256"):
        raise CameraEpisodePipelineError(f"{role} hash drifted")
    return path


def _descriptor_config(mapping: Mapping[str, Any], name: str) -> None:
    value = mapping.get(name)
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not value.get("path"):
        raise CameraEpisodePipelineError(f"{name} descriptor is missing")
    _digest(value.get("sha256"), f"{name}.sha256")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraEpisodePipelineError(f"{role} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CameraEpisodePipelineError(f"{role} must be a JSON object")
    return value


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CameraEpisodePipelineError(f"cannot read {role}") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CameraEpisodePipelineError(f"{role} contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise CameraEpisodePipelineError(f"{role} row must be an object")
        rows.append(value)
    return rows


def _load_yaml(path: Path, role: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodePipelineError(f"{role} is invalid YAML") from exc
    if not isinstance(value, dict):
        raise CameraEpisodePipelineError(f"{role} must be a mapping")
    return value


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        raise CameraEpisodePipelineError(f"{name} must be a mapping")
    return dict(result)


def _nonempty(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise CameraEpisodePipelineError(f"{role} must be a non-empty string")
    return value


def _token(value: Any, role: str) -> str:
    text = _nonempty(value, role)
    if any(char in text for char in "\r\n\0"):
        raise CameraEpisodePipelineError(f"{role} contains a control character")
    return text


def _digest(value: Any, role: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256 or any(char not in "0123456789abcdef" for char in value):
        raise CameraEpisodePipelineError(f"{role} must be a lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CameraEpisodePipelineError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _normalize_status(value: Any, *, fallback: str) -> str:
    mapping = {"ready": "ready", "uncertain": "uncertain", "boundary_uncertain": "uncertain", "unavailable": "unavailable", "inference_error": "error", "error": "error"}
    status = mapping.get(str(value), mapping.get(fallback, "error"))
    if status not in _STATUS_VALUES:
        raise CameraEpisodePipelineError("normalized status is invalid")
    return status


def _unique_strings(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _timestamp_for_interval(started: Any, seconds: float) -> str | None:
    if started is None:
        return None
    if not isinstance(started, str) or not started:
        return None
    try:
        value = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _iso(value + timedelta(seconds=seconds))


def _single_or_none(values: Sequence[Any] | Any) -> str | None:
    items = list(values) if not isinstance(values, str) else [values]
    unique = sorted({str(value) for value in items})
    return unique[0] if len(unique) == 1 else None


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _readme_text(
    *,
    status: str,
    run_summary: Mapping[str, Any],
    automatic_metrics: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
) -> str:
    pooled = automatic_metrics["pooled"]
    oracle_binary = oracle_metrics["all_shape_eligible"]["shape_binary"]
    return "\n".join(
        [
            "# W5D-02 automatic episode-to-shape pipeline",
            "",
            f"status: `{status}`",
            f"development videos: `{run_summary['input_counts']['development_videos']}`",
            f"episode results: `{run_summary['output_counts']['episode_results']}`",
            f"ready+uncertain prediction coverage: `{pooled['ready_uncertain_prediction_coverage']:.6f}`",
            f"coverage gate (>= 0.85): `{pooled['prediction_coverage_gate_satisfied']}`",
            f"automatic binary support: `{pooled['automatic_binary']['support']}`",
            f"automatic binary macro-F1: `{pooled['automatic_binary']['macro_f1']}`",
            f"oracle binary macro-F1: `{oracle_binary['macro_f1']}`",
            "",
            "Binary is the primary result; four-class output is diagnostic. Every automatic boundary is marked as not human accepted. This is development evidence and does not emit AlgorithmEvent.",
            "",
        ]
    )


def _verification_text(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    shape_rows: Sequence[Mapping[str, Any]],
    automatic_metrics: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
    oracle_summary: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    files: Mapping[str, Any],
) -> str:
    payload = {
        "pipeline_id": config["pipeline_id"],
        "config_sha256": config_sha256,
        "development_video_count": len(rows),
        "proposal_result_count": len(shape_rows),
        "automatic": automatic_metrics,
        "oracle": {
            "summary": oracle_summary,
            "binary": oracle_metrics["all_shape_eligible"]["shape_binary"],
            "four_class": oracle_metrics["all_shape_eligible"]["four_class"],
        },
        "run_summary": run_summary,
        "identity": {
            "candidate_id": config["fixed_primary"]["candidate_id"],
            "model_state_sha256": config["fixed_primary"]["model_state_sha256"],
            "segmenter_profile_id": config["segmenter"]["profile_id"],
            "segmenter_profile_sha256": config["segmenter"]["profile"]["sha256"],
            "development_index_sha256": files["index_descriptor"]["sha256"],
        },
        "binary_adjustment_decision": {
            "decision": "retain_fixed_primary_and_threshold",
            "reason": (
                "fresh oracle binary macro-F1="
                f"{oracle_metrics['all_shape_eligible']['shape_binary']['macro_f1']} "
                "with fixed threshold 0.5; automatic conditional binary macro-F1="
                f"{automatic_metrics['pooled']['automatic_binary']['macro_f1']} and "
                f"{automatic_metrics['pooled']['known_episode_match_pipeline_miss_count']} "
                "known-episode boundary/QC match misses. The evidence does not support "
                "a camera binary adjustment."
            ),
            "threshold": 0.5,
            "model_retrained": False,
        },
    }
    return "# W5D-02 verification\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n```\n"


def _commit_new_directory(output: Path, payloads: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"W5D-02 output directory already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for name, payload in sorted(payloads.items()):
            destination = temporary / name
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "CAMERA_EPISODE_PIPELINE_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_PIPELINE_MANIFEST_SCHEMA_VERSION",
    "CAMERA_EPISODE_PIPELINE_SUMMARY_SCHEMA_VERSION",
    "CameraEpisodePipelineBuildResult",
    "CameraEpisodePipelineError",
    "_build_episode_result",
    "_evaluate_automatic_binary",
    "build_camera_episode_pipeline",
    "load_camera_episode_pipeline_config",
]
