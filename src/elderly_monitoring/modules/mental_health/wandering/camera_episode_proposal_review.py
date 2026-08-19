"""Truth-free review pack for S0 proposals and S2A shape results."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    CameraAdapterInput,
    CameraObservation,
    load_camera_inputs,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
    CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION,
    PRODUCER_CONFIG_ID,
    PROPOSAL_STATUSES,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_proposal_inference import (
    CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION,
    CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION,
)
from elderly_monitoring.modules.mental_health.wandering.camera_inference import (
    CameraInferenceError,
    load_camera_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
    BINARY_CLASS_ORDER,
    FOUR_CLASS_ORDER,
)


CAMERA_EPISODE_PROPOSAL_REVIEW_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-episode-proposal-review-config-v1"
)
CAMERA_EPISODE_PROPOSAL_REVIEW_INDEX_SCHEMA_VERSION = (
    "wandering-camera-episode-proposal-review-index-v1"
)
_CONFIG_RELATIVE = Path(
    "configs/modules/wandering_camera_episode_proposal_review_v1.yaml"
)
_BATCH_FIELDS = frozenset(
    {
        "bundle_id",
        "proposal_bundle_dir",
        "proposal_shape_bundle_dir",
        "tracking_jsonl",
        "media_sidecar",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
    }
)
_SCOPE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
    "track_id",
)
_PROPOSAL_LINK_FIELDS = (
    "proposal_id",
    *_SCOPE_FIELDS,
    "technical_segment_index",
    "start_sec",
    "end_sec_exclusive",
    "duration_sec",
    "proposal_status",
    "start_reason",
    "end_reason",
    "reason_codes",
    "hard_break_reasons",
    "uncertain",
    "manual_review_required",
    "producer_config_id",
)
_HUMAN_REVIEW_FIELDS = (
    "human_boundary_decision",
    "corrected_start_sec",
    "corrected_end_sec",
    "observable_pattern",
    "purpose_context",
    "review_notes",
)
_REVIEW_INDEX_FIELDS = (
    "schema_version",
    "bundle_id",
    "participant_id",
    "session_id",
    "camera_setup_id",
    "clock_domain_id",
    "source_group_id",
    "source_video_id",
    "media_ref",
    "device_id",
    "setup_id",
    "stream_epoch",
    "track_id",
    "technical_segment_index",
    "proposal_id",
    "start_sec",
    "end_sec_exclusive",
    "duration_sec",
    "proposal_status",
    "reason_codes",
    "prediction_status",
    "model_invocation_skipped",
    "candidate_id",
    "predicted_binary",
    "binary_class_order",
    "binary_probabilities",
    "predicted_pattern",
    "pattern_class_order",
    "pattern_probabilities",
    "trajectory_point_count",
    "trajectory_plot_path",
    "needs_human_review",
)
_TEMPLATE_FIELDS = (
    "source_video_id",
    "proposal_id",
    "original_start_sec",
    "original_end_sec_exclusive",
    "proposal_status",
    "prediction_status",
    "trajectory_plot_path",
    *_HUMAN_REVIEW_FIELDS,
)
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class CameraEpisodeProposalReviewError(ValueError):
    """Review inputs cannot be joined without ambiguity."""


@dataclass(frozen=True)
class CameraEpisodeProposalReviewBuildResult:
    output_dir: Path
    source_video_count: int
    proposal_count: int
    trajectory_plot_count: int
    model_forward_invocation_count: int
    model_invocation_skipped_count: int


@dataclass(frozen=True)
class _SourceInput:
    binding: Mapping[str, str]
    adapter: CameraAdapterInput
    proposals: tuple[dict[str, Any], ...]
    proposal_shape_bundle_dir: Path


def load_camera_episode_proposal_review_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load the fixed thin S2B review configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeProposalReviewError(
            "cannot read camera episode proposal review config"
        ) from exc
    if value != _expected_config():
        raise CameraEpisodeProposalReviewError(
            "camera episode proposal review config fields or values drifted"
        )
    return value


def build_camera_episode_proposal_review_pack(
    *,
    project_root: str | Path,
    config_path: str | Path,
    batch_index_path: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeProposalReviewBuildResult:
    """Build review rows and trajectory plots without creating truth."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera proposal review output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    fixed_config_path = (root / _CONFIG_RELATIVE).resolve(strict=False)
    if Path(config_path).resolve(strict=False) != fixed_config_path:
        raise CameraEpisodeProposalReviewError(
            "proposal review config must use the fixed repository path"
        )
    config = load_camera_episode_proposal_review_config(fixed_config_path)
    try:
        camera_config_path = (root / config["camera_config_path"]).resolve(strict=True)
        camera_config = load_camera_config(camera_config_path)
    except (OSError, CameraInferenceError, ValueError) as exc:
        raise CameraEpisodeProposalReviewError(
            "proposal review camera config binding failed"
        ) from exc

    sources = _load_sources(Path(batch_index_path), camera_config=camera_config)
    proposal_by_id: dict[str, tuple[_SourceInput, dict[str, Any]]] = {}
    for source in sources:
        for proposal in source.proposals:
            proposal_id = str(proposal["proposal_id"])
            if proposal_id in proposal_by_id:
                raise CameraEpisodeProposalReviewError("duplicate S0 proposal ID")
            proposal_by_id[proposal_id] = (source, proposal)
    if not proposal_by_id:
        raise CameraEpisodeProposalReviewError("proposal review batch has no proposals")

    prediction_by_id = _load_prediction_bundles(
        {source.proposal_shape_bundle_dir for source in sources}
    )
    if set(prediction_by_id) != set(proposal_by_id):
        raise CameraEpisodeProposalReviewError(
            "S0 proposal and S2A result ID sets differ"
        )

    assembled: list[tuple[_SourceInput, dict[str, Any], dict[str, Any]]] = []
    for proposal_id, (source, proposal) in proposal_by_id.items():
        prediction = prediction_by_id[proposal_id]
        _validate_join(source, proposal, prediction)
        assembled.append((source, proposal, prediction))
    assembled.sort(key=_assembled_sort_key)

    files: dict[str, bytes] = {}
    review_rows: list[dict[str, str]] = []
    template_rows: list[dict[str, str]] = []
    for source, proposal, prediction in assembled:
        observations = _proposal_observations(source.adapter, proposal)
        plot_path = _plot_path(proposal)
        files[plot_path] = _trajectory_svg(
            proposal=proposal,
            prediction=prediction,
            media=source.adapter.media_sidecar,
            observations=observations,
        )
        review_row = _review_row(
            source=source,
            proposal=proposal,
            prediction=prediction,
            plot_path=plot_path,
            point_count=len(observations),
        )
        review_rows.append(review_row)
        template_rows.append(
            {
                "source_video_id": review_row["source_video_id"],
                "proposal_id": review_row["proposal_id"],
                "original_start_sec": review_row["start_sec"],
                "original_end_sec_exclusive": review_row["end_sec_exclusive"],
                "proposal_status": review_row["proposal_status"],
                "prediction_status": review_row["prediction_status"],
                "trajectory_plot_path": review_row["trajectory_plot_path"],
                **{name: "" for name in _HUMAN_REVIEW_FIELDS},
            }
        )

    invoked_count = sum(
        row["model_invocation_skipped"] == "false" for row in review_rows
    )
    skipped_count = len(review_rows) - invoked_count
    files["review_index.csv"] = _csv_bytes(_REVIEW_INDEX_FIELDS, review_rows)
    files["human_review_template.csv"] = _csv_bytes(
        _TEMPLATE_FIELDS, template_rows
    )
    files["per_video_summary.md"] = _per_video_summary(review_rows).encode("utf-8")
    files["README.md"] = _bundle_readme(
        source_video_count=len({row["source_video_id"] for row in review_rows}),
        proposal_count=len(review_rows),
        invoked_count=invoked_count,
        skipped_count=skipped_count,
    ).encode("utf-8")
    _commit_new_directory(output, files)
    return CameraEpisodeProposalReviewBuildResult(
        output_dir=output,
        source_video_count=len({row["source_video_id"] for row in review_rows}),
        proposal_count=len(review_rows),
        trajectory_plot_count=len(review_rows),
        model_forward_invocation_count=invoked_count,
        model_invocation_skipped_count=skipped_count,
    )


def _load_sources(
    batch_index_path: Path,
    *,
    camera_config: Mapping[str, Any],
) -> tuple[_SourceInput, ...]:
    try:
        index_path = batch_index_path.resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeProposalReviewError(
            "proposal review batch index is inaccessible"
        ) from exc
    rows = _load_jsonl(index_path, "proposal review batch index")
    if not rows:
        raise CameraEpisodeProposalReviewError("proposal review batch index is empty")
    bundle_ids: set[str] = set()
    physical_sources: set[tuple[str, str, str, str, str]] = set()
    output: list[_SourceInput] = []
    for raw_binding in rows:
        if set(raw_binding) != _BATCH_FIELDS or not all(
            isinstance(raw_binding[name], str) and raw_binding[name]
            for name in _BATCH_FIELDS
        ):
            raise CameraEpisodeProposalReviewError(
                "proposal review batch index fields are invalid"
            )
        binding = {name: str(raw_binding[name]) for name in _BATCH_FIELDS}
        if binding["bundle_id"] in bundle_ids:
            raise CameraEpisodeProposalReviewError("duplicate proposal review bundle ID")
        bundle_ids.add(binding["bundle_id"])
        proposal_dir = _resolve_index_path(
            index_path, binding["proposal_bundle_dir"], directory=True
        )
        shape_dir = _resolve_index_path(
            index_path, binding["proposal_shape_bundle_dir"], directory=True
        )
        tracking = _resolve_index_path(index_path, binding["tracking_jsonl"])
        sidecar = _resolve_index_path(index_path, binding["media_sidecar"])
        try:
            adapter = load_camera_inputs(tracking, sidecar, camera_config)
        except (CameraAdapterError, OSError, ValueError) as exc:
            raise CameraEpisodeProposalReviewError(
                "proposal review tracking or sidecar is invalid"
            ) from exc
        if binding["camera_setup_id"] != adapter.media_sidecar["setup_id"]:
            raise CameraEpisodeProposalReviewError(
                "proposal review camera setup binding differs from sidecar"
            )
        physical_key = tuple(
            str(adapter.media_sidecar[name])
            for name in (
                "source_group_id",
                "source_video_id",
                "device_id",
                "setup_id",
                "stream_epoch",
            )
        )
        if physical_key in physical_sources:
            raise CameraEpisodeProposalReviewError(
                "physical camera source is bound more than once"
            )
        physical_sources.add(physical_key)
        summary = _load_json(proposal_dir / "summary.json", "S0 proposal summary")
        proposals = _load_jsonl(proposal_dir / "proposals.jsonl", "S0 proposals")
        if not (proposal_dir / "diagnostics.jsonl").is_file():
            raise CameraEpisodeProposalReviewError("S0 proposal diagnostics are missing")
        _validate_proposal_summary(summary, adapter, proposals)
        for proposal in proposals:
            _validate_proposal(proposal, adapter)
        output.append(
            _SourceInput(
                binding=binding,
                adapter=adapter,
                proposals=tuple(proposals),
                proposal_shape_bundle_dir=shape_dir,
            )
        )
    return tuple(output)


def _load_prediction_bundles(
    bundle_dirs: Iterable[Path],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for bundle_dir in sorted(bundle_dirs, key=str):
        summary = _load_json(bundle_dir / "summary.json", "S2A proposal-shape summary")
        predictions = _load_jsonl(
            bundle_dir / "proposal_shape_predictions.jsonl",
            "S2A proposal-shape predictions",
        )
        if not (bundle_dir / "README.md").is_file():
            raise CameraEpisodeProposalReviewError(
                "S2A proposal-shape README is missing"
            )
        _validate_prediction_summary(summary, predictions)
        for prediction in predictions:
            _validate_prediction(prediction)
            proposal_id = str(prediction["proposal_id"])
            if proposal_id in output:
                raise CameraEpisodeProposalReviewError("duplicate S2A proposal ID")
            output[proposal_id] = prediction
    return output


def _validate_proposal_summary(
    summary: Mapping[str, Any],
    adapter: CameraAdapterInput,
    proposals: Sequence[Mapping[str, Any]],
) -> None:
    if (
        summary.get("schema_version")
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SUMMARY_SCHEMA_VERSION
        or summary.get("proposal_schema_version")
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION
        or summary.get("proposal_is_accepted_boundary") is not False
        or summary.get("proposal_count") != len(proposals)
    ):
        raise CameraEpisodeProposalReviewError("S0 proposal summary is invalid")
    for name in _SCOPE_FIELDS[:-1]:
        if summary.get(name) != adapter.media_sidecar[name]:
            raise CameraEpisodeProposalReviewError(
                "S0 proposal summary scope differs from sidecar"
            )


def _validate_proposal(
    proposal: Mapping[str, Any], adapter: CameraAdapterInput
) -> None:
    required = set(_PROPOSAL_LINK_FIELDS) | {"schema_version"}
    if not isinstance(proposal, Mapping) or not required.issubset(proposal):
        raise CameraEpisodeProposalReviewError("S0 proposal fields are incomplete")
    if (
        proposal["schema_version"]
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION
        or proposal["producer_config_id"] != PRODUCER_CONFIG_ID
        or proposal["proposal_status"] not in PROPOSAL_STATUSES
        or proposal["manual_review_required"] is not True
        or not isinstance(proposal["proposal_id"], str)
        or not proposal["proposal_id"]
        or not isinstance(proposal["track_id"], int)
        or isinstance(proposal["track_id"], bool)
        or not isinstance(proposal["technical_segment_index"], int)
        or isinstance(proposal["technical_segment_index"], bool)
        or proposal["technical_segment_index"] < 0
    ):
        raise CameraEpisodeProposalReviewError("S0 proposal identity is invalid")
    for name in _SCOPE_FIELDS[:-1]:
        if proposal[name] != adapter.media_sidecar[name]:
            raise CameraEpisodeProposalReviewError(
                "S0 proposal scope differs from sidecar"
            )
    start = _finite_float(proposal["start_sec"], "proposal start")
    end = _finite_float(proposal["end_sec_exclusive"], "proposal end")
    duration = _finite_float(proposal["duration_sec"], "proposal duration")
    if (
        start < 0.0
        or start >= end
        or end > float(adapter.media_sidecar["duration_sec"]) + 1e-9
        or not math.isclose(duration, end - start, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise CameraEpisodeProposalReviewError("S0 proposal endpoints are invalid")
    expected_uncertain = proposal["proposal_status"] != "proposed"
    if proposal.get("uncertain") is not expected_uncertain:
        raise CameraEpisodeProposalReviewError("S0 proposal uncertainty is invalid")
    for name in ("reason_codes", "hard_break_reasons"):
        _string_list(proposal[name], f"S0 proposal {name}")


def _validate_prediction_summary(
    summary: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> None:
    invoked = sum(
        isinstance(row, Mapping) and row.get("model_invocation_skipped") is False
        for row in predictions
    )
    if (
        summary.get("schema_version")
        != CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION
        or summary.get("input_boundary_kind") != "automatic_proposal"
        or summary.get("proposal_schema_version")
        != CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION
        or summary.get("proposal_count") != len(predictions)
        or summary.get("result_count") != len(predictions)
        or summary.get("model_forward_invocation_count") != invoked
        or summary.get("model_invocation_skipped_count")
        != len(predictions) - invoked
        or summary.get("proposal_endpoint_modified") is not False
        or summary.get("accepted_boundary_emitted") is not False
        or summary.get("truth_labels_consumed") is not False
        or summary.get("shape_performance_metrics_available") is not False
    ):
        raise CameraEpisodeProposalReviewError(
            "S2A proposal-shape summary is invalid"
        )


def _validate_prediction(prediction: Mapping[str, Any]) -> None:
    required = set(_PROPOSAL_LINK_FIELDS) | {
        "schema_version",
        "bundle_id",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
        "prediction_status",
        "prediction_reason_codes",
        "binary",
        "four_class",
        "predicted_binary",
        "predicted_pattern",
        "candidate_id",
        "model_invocation_skipped",
    }
    if not isinstance(prediction, Mapping) or not required.issubset(prediction):
        raise CameraEpisodeProposalReviewError("S2A prediction fields are incomplete")
    if (
        prediction["schema_version"]
        != CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION
        or not all(
            isinstance(prediction[name], str) and prediction[name]
            for name in (
                "bundle_id",
                "participant_id",
                "session_id",
                "camera_setup_id",
                "clock_domain_id",
                "candidate_id",
            )
        )
        or prediction["prediction_status"]
        not in {"ready", "unavailable", "boundary_uncertain", "inference_error"}
        or not isinstance(prediction["model_invocation_skipped"], bool)
    ):
        raise CameraEpisodeProposalReviewError("S2A prediction identity is invalid")
    _string_list(prediction["prediction_reason_codes"], "S2A prediction reasons")
    status = str(prediction["prediction_status"])
    skipped = bool(prediction["model_invocation_skipped"])
    if status == "ready":
        if skipped:
            raise CameraEpisodeProposalReviewError("ready S2A prediction was skipped")
        binary = _probability_head(
            prediction["binary"], BINARY_CLASS_ORDER, "S2A binary"
        )
        pattern = _probability_head(
            prediction["four_class"], FOUR_CLASS_ORDER, "S2A four-class"
        )
        if (
            prediction["predicted_binary"] != binary["predicted_label"]
            or prediction["predicted_pattern"] != pattern["predicted_label"]
        ):
            raise CameraEpisodeProposalReviewError(
                "S2A predicted labels differ from probability heads"
            )
    else:
        if status in {"unavailable", "boundary_uncertain"} and not skipped:
            raise CameraEpisodeProposalReviewError(
                "non-ready S2A prediction skip status is invalid"
            )
        if status == "inference_error" and skipped:
            raise CameraEpisodeProposalReviewError(
                "S2A inference error must record attempted invocation"
            )
        if any(
            prediction.get(name) is not None
            for name in ("binary", "four_class", "predicted_binary", "predicted_pattern")
        ):
            raise CameraEpisodeProposalReviewError(
                "non-ready S2A prediction must not contain probabilities"
            )


def _validate_join(
    source: _SourceInput,
    proposal: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> None:
    for name in _PROPOSAL_LINK_FIELDS:
        if prediction.get(name) != proposal[name]:
            raise CameraEpisodeProposalReviewError(
                f"S0/S2A proposal field differs: {name}"
            )
    for name in (
        "bundle_id",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
    ):
        if prediction.get(name) != source.binding[name]:
            raise CameraEpisodeProposalReviewError(
                f"S2A result differs from review batch binding: {name}"
            )
    proposal_status = str(proposal["proposal_status"])
    prediction_status = str(prediction["prediction_status"])
    if (
        proposal_status == "uncertain"
        and prediction_status
        not in {"ready", "unavailable", "boundary_uncertain", "inference_error"}
    ) or (
        proposal_status == "rejected_by_qc"
        and prediction_status != "unavailable"
    ) or (
        proposal_status == "proposed"
        and prediction_status not in {"ready", "unavailable", "inference_error"}
    ):
        raise CameraEpisodeProposalReviewError(
            "S0 proposal and S2A prediction statuses differ"
        )


def _proposal_observations(
    adapter: CameraAdapterInput, proposal: Mapping[str, Any]
) -> tuple[CameraObservation, ...]:
    scope = tuple(proposal[name] for name in _SCOPE_FIELDS)
    start = float(proposal["start_sec"])
    end = float(proposal["end_sec_exclusive"])
    observations = tuple(
        observation
        for observation in adapter.observations
        if observation.scope_key == scope
        and observation.timestamp_sec >= start - 1e-9
        and observation.timestamp_sec < end
    )
    if not observations:
        raise CameraEpisodeProposalReviewError(
            "proposal has no matching tracking observations"
        )
    return observations


def _review_row(
    *,
    source: _SourceInput,
    proposal: Mapping[str, Any],
    prediction: Mapping[str, Any],
    plot_path: str,
    point_count: int,
) -> dict[str, str]:
    binary = prediction["binary"]
    pattern = prediction["four_class"]
    return {
        "schema_version": CAMERA_EPISODE_PROPOSAL_REVIEW_INDEX_SCHEMA_VERSION,
        "bundle_id": source.binding["bundle_id"],
        "participant_id": source.binding["participant_id"],
        "session_id": source.binding["session_id"],
        "camera_setup_id": source.binding["camera_setup_id"],
        "clock_domain_id": source.binding["clock_domain_id"],
        "source_group_id": str(proposal["source_group_id"]),
        "source_video_id": str(proposal["source_video_id"]),
        "media_ref": str(source.adapter.media_sidecar["media_ref"]),
        "device_id": str(proposal["device_id"]),
        "setup_id": str(proposal["setup_id"]),
        "stream_epoch": str(proposal["stream_epoch"]),
        "track_id": str(proposal["track_id"]),
        "technical_segment_index": str(proposal["technical_segment_index"]),
        "proposal_id": str(proposal["proposal_id"]),
        "start_sec": _format_seconds(proposal["start_sec"]),
        "end_sec_exclusive": _format_seconds(proposal["end_sec_exclusive"]),
        "duration_sec": _format_seconds(proposal["duration_sec"]),
        "proposal_status": str(proposal["proposal_status"]),
        "reason_codes": _json_cell(proposal["reason_codes"]),
        "prediction_status": str(prediction["prediction_status"]),
        "model_invocation_skipped": _bool_cell(
            bool(prediction["model_invocation_skipped"])
        ),
        "candidate_id": str(prediction["candidate_id"]),
        "predicted_binary": (
            str(prediction["predicted_binary"])
            if prediction["predicted_binary"] is not None
            else ""
        ),
        "binary_class_order": _json_cell(binary["class_order"]) if binary else "",
        "binary_probabilities": (
            _json_cell(binary["probabilities"]) if binary else ""
        ),
        "predicted_pattern": (
            str(prediction["predicted_pattern"])
            if prediction["predicted_pattern"] is not None
            else ""
        ),
        "pattern_class_order": (
            _json_cell(pattern["class_order"]) if pattern else ""
        ),
        "pattern_probabilities": (
            _json_cell(pattern["probabilities"]) if pattern else ""
        ),
        "trajectory_point_count": str(point_count),
        "trajectory_plot_path": plot_path,
        "needs_human_review": "true",
    }


def _trajectory_svg(
    *,
    proposal: Mapping[str, Any],
    prediction: Mapping[str, Any],
    media: Mapping[str, Any],
    observations: Sequence[CameraObservation],
) -> bytes:
    width = 900
    height = 650
    left = 80.0
    top = 145.0
    plot_width = 740.0
    plot_height = 410.0
    coordinates = [
        (
            left + observation.bbox_bottom_point[0] * plot_width,
            top + observation.bbox_bottom_point[1] * plot_height,
        )
        for observation in observations
    ]
    points = " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates)
    start_x, start_y = coordinates[0]
    end_x, end_y = coordinates[-1]
    source_video_id = html.escape(str(proposal["source_video_id"]))
    proposal_id = html.escape(str(proposal["proposal_id"]))
    interval = (
        f"{float(proposal['start_sec']):.3f} to "
        f"{float(proposal['end_sec_exclusive']):.3f} sec"
    )
    status = html.escape(str(proposal["proposal_status"]))
    prediction_status = html.escape(str(prediction["prediction_status"]))
    reasons = html.escape(", ".join(str(item) for item in proposal["reason_codes"]))
    media_ref = html.escape(str(media["media_ref"]))
    grid = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * plot_width
        y = top + fraction * plot_height
        grid.append(
            f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
            f'y2="{top + plot_height:.1f}" class="grid"/>'
        )
        grid.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{y:.1f}" class="grid"/>'
        )
    point_marks = "".join(
        f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.2" class="point"/>'
        for x, y in coordinates
    )
    document = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="direction-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f5f99"/>
    </marker>
  </defs>
  <style>
    text {{ font-family: Arial, sans-serif; fill: #17202a; letter-spacing: 0; }}
    .title {{ font-size: 18px; font-weight: 700; }}
    .meta {{ font-size: 13px; }}
    .grid {{ stroke: #d9dee3; stroke-width: 1; }}
    .border {{ fill: #f8fafb; stroke: #66737f; stroke-width: 1.5; }}
    .track {{ fill: none; stroke: #1f5f99; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }}
    .point {{ fill: #2f78b7; opacity: 0.72; }}
    .legend {{ font-size: 12px; }}
  </style>
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="40" y="32" class="title">Proposal trajectory review</text>
  <text x="40" y="56" class="meta">source_video_id: {source_video_id} | proposal_id: {proposal_id}</text>
  <text x="40" y="78" class="meta">media_ref: {media_ref} | track_id: {proposal['track_id']} | interval: {interval}</text>
  <text x="40" y="100" class="meta">proposal_status: {status} | prediction_status: {prediction_status}</text>
  <text x="40" y="122" class="meta">reason_codes: {reasons}</text>
  <rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" class="border"/>
  {''.join(grid)}
  <polyline points="{points}" class="track" marker-end="url(#direction-arrow)"/>
  {point_marks}
  <circle cx="{start_x:.3f}" cy="{start_y:.3f}" r="7" fill="#16834c" stroke="#ffffff" stroke-width="2"/>
  <circle cx="{end_x:.3f}" cy="{end_y:.3f}" r="7" fill="#bd2c35" stroke="#ffffff" stroke-width="2"/>
  <text x="{left:.1f}" y="{top + plot_height + 28:.1f}" class="legend">Normalized image coordinates, top-left origin</text>
  <circle cx="{left + 350:.1f}" cy="{top + plot_height + 23:.1f}" r="6" fill="#16834c"/>
  <text x="{left + 362:.1f}" y="{top + plot_height + 28:.1f}" class="legend">start</text>
  <circle cx="{left + 425:.1f}" cy="{top + plot_height + 23:.1f}" r="6" fill="#bd2c35"/>
  <text x="{left + 437:.1f}" y="{top + plot_height + 28:.1f}" class="legend">end</text>
  <text x="{left:.1f}" y="{top + plot_height + 55:.1f}" class="legend">Direction follows the blue path toward the arrow. Tracking points: {len(observations)}.</text>
</svg>
'''
    return document.encode("utf-8")


def _per_video_summary(rows: Sequence[Mapping[str, str]]) -> str:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_video_id"]].append(row)
    lines = [
        "# Per-video proposal review summary",
        "",
        "Truth-free triage only. Proposal and model states are review context, not human labels.",
        "",
    ]
    for source_video_id in sorted(grouped):
        items = grouped[source_video_id]
        proposal_counts = Counter(row["proposal_status"] for row in items)
        prediction_counts = Counter(row["prediction_status"] for row in items)
        lines.extend(
            [
                f"## {source_video_id}",
                "",
                f"- media_ref: `{items[0]['media_ref']}`",
                f"- proposal_count: `{len(items)}`",
                f"- proposal_status_counts: `{_counter_text(proposal_counts)}`",
                f"- prediction_status_counts: `{_counter_text(prediction_counts)}`",
                "- performance metrics: not computed",
                "",
                "| proposal_id | interval_sec | proposal_status | prediction_status | reason_codes | trajectory |",
                "| --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in items:
            reason = row["reason_codes"].replace("|", "\\|")
            lines.append(
                f"| `{row['proposal_id']}` | {row['start_sec']}–{row['end_sec_exclusive']} "
                f"| `{row['proposal_status']}` | `{row['prediction_status']}` "
                f"| `{reason}` | [plot]({row['trajectory_plot_path']}) |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _bundle_readme(
    *,
    source_video_count: int,
    proposal_count: int,
    invoked_count: int,
    skipped_count: int,
) -> str:
    return (
        "# Proposal-shape review pack\n\n"
        "This bundle connects original S0 proposals, S2A frozen-shape results, "
        "tracking trajectories, and media identity for human review. It does not "
        "create truth or accepted boundaries.\n\n"
        f"- source videos: `{source_video_count}`\n"
        f"- proposals/review rows/trajectory plots: `{proposal_count}/{proposal_count}/{proposal_count}`\n"
        f"- S2A model invocations/skips represented: `{invoked_count}/{skipped_count}`\n"
        "- human review columns: blank\n"
        "- performance metrics: not computed\n"
        "- alert, FAR, and clinical metrics: not computed\n\n"
        "Use `review_index.csv` to locate each video, interval, track, prediction, "
        "and plot. Enter independent review only in the blank columns of "
        "`human_review_template.csv`; model output is not truth.\n"
    )


def _probability_head(
    value: Any, expected_order: Sequence[str], role: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "class_order",
        "probabilities",
        "predicted_label",
    }:
        raise CameraEpisodeProposalReviewError(f"{role} fields are invalid")
    if value["class_order"] != list(expected_order):
        raise CameraEpisodeProposalReviewError(f"{role} class order is invalid")
    probabilities = value["probabilities"]
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != len(expected_order)
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in probabilities
        )
        or not math.isclose(
            sum(float(item) for item in probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        or value["predicted_label"] not in expected_order
    ):
        raise CameraEpisodeProposalReviewError(f"{role} probabilities are invalid")
    return value


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraEpisodeProposalReviewError(f"cannot read {role}") from exc
    if not isinstance(value, dict):
        raise CameraEpisodeProposalReviewError(f"{role} must be a JSON object")
    return value


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CameraEpisodeProposalReviewError(f"cannot read {role}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise CameraEpisodeProposalReviewError(f"{role} rows must be JSON objects")
    return values


def _resolve_index_path(
    index_path: Path, value: str, *, directory: bool = False
) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = index_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CameraEpisodeProposalReviewError(
            "proposal review input path is inaccessible"
        ) from exc
    if directory and not resolved.is_dir():
        raise CameraEpisodeProposalReviewError(
            "proposal review bundle path is not a directory"
        )
    if not directory and not resolved.is_file():
        raise CameraEpisodeProposalReviewError(
            "proposal review input path is not a file"
        )
    return resolved


def _csv_bytes(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"camera proposal review output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise CameraEpisodeProposalReviewError(
                    "proposal review output path is unsafe"
                )
            destination = temporary.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _plot_path(proposal: Mapping[str, Any]) -> str:
    identity = f"{proposal['source_video_id']}|{proposal['proposal_id']}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    video = _filename_token(str(proposal["source_video_id"]))
    proposal_id = _filename_token(str(proposal["proposal_id"]))
    return f"per_proposal_trajectory_plots/{video}__{proposal_id}__{digest}.svg"


def _filename_token(value: str) -> str:
    token = _SAFE_FILENAME.sub("_", value).strip("._")
    return token[:80] or "identity"


def _assembled_sort_key(
    item: tuple[_SourceInput, Mapping[str, Any], Mapping[str, Any]],
) -> tuple[Any, ...]:
    _source, proposal, _prediction = item
    return (
        str(proposal["source_group_id"]),
        str(proposal["source_video_id"]),
        str(proposal["device_id"]),
        str(proposal["setup_id"]),
        str(proposal["stream_epoch"]),
        int(proposal["track_id"]),
        float(proposal["start_sec"]),
        str(proposal["proposal_id"]),
    )


def _finite_float(value: Any, role: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CameraEpisodeProposalReviewError(f"{role} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CameraEpisodeProposalReviewError(f"{role} is not finite")
    return result


def _string_list(value: Any, role: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise CameraEpisodeProposalReviewError(f"{role} is invalid")
    return value


def _format_seconds(value: Any) -> str:
    return f"{float(value):.6f}"


def _json_cell(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _bool_cell(value: bool) -> str:
    return "true" if value else "false"


def _counter_text(counter: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={counter[name]}" for name in sorted(counter))


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CAMERA_EPISODE_PROPOSAL_REVIEW_CONFIG_SCHEMA_VERSION,
        "purpose": "topowander_proposal_shape_truth_free_review_pack",
        "camera_config_path": "configs/modules/wandering_camera_v1.yaml",
        "proposal_schema_version": CAMERA_EPISODE_BOUNDARY_PROPOSAL_SCHEMA_VERSION,
        "proposal_shape_prediction_schema_version": (
            CAMERA_EPISODE_PROPOSAL_SHAPE_PREDICTION_SCHEMA_VERSION
        ),
        "proposal_shape_summary_schema_version": (
            CAMERA_EPISODE_PROPOSAL_SHAPE_SUMMARY_SCHEMA_VERSION
        ),
        "review_index_schema_version": (
            CAMERA_EPISODE_PROPOSAL_REVIEW_INDEX_SCHEMA_VERSION
        ),
        "trajectory_plot_format": "svg",
        "human_review_fields": list(_HUMAN_REVIEW_FIELDS),
    }


__all__ = [
    "CAMERA_EPISODE_PROPOSAL_REVIEW_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_PROPOSAL_REVIEW_INDEX_SCHEMA_VERSION",
    "CameraEpisodeProposalReviewBuildResult",
    "CameraEpisodeProposalReviewError",
    "build_camera_episode_proposal_review_pack",
    "load_camera_episode_proposal_review_config",
]
