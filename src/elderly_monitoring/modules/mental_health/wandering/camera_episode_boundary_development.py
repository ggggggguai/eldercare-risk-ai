"""B01+B02 recall-first camera episode segmenter development search."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary import (
    CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION,
    PRODUCER_CONFIG_ID,
    build_camera_episode_boundary_development_bundle,
    load_camera_episode_boundary_development_profile,
    load_camera_episode_boundary_proposal_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_boundary_evaluation import (
    build_camera_episode_boundary_evaluation_bundle,
)


CAMERA_EPISODE_SEGMENTER_SEARCH_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-segmenter-search-config-v1"
)
CAMERA_EPISODE_SEGMENTER_SEARCH_MANIFEST_SCHEMA_VERSION = (
    "wandering-camera-segmenter-search-manifest-v1"
)
CAMERA_EPISODE_SEGMENTER_RUN_METRICS_SCHEMA_VERSION = (
    "wandering-camera-segmenter-run-metrics-v1"
)
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "search_id",
        "development_contract",
        "proposal_baseline",
        "evaluation_config",
        "expected_batches",
        "selection_policy",
        "candidates",
    }
)
_DESCRIPTOR_FIELDS = frozenset({"path", "sha256"})
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "stage",
        "parameter_group",
        "state_machine",
        "boundary_refinement",
    }
)
_STATE_FIELDS = frozenset(
    {
        "movement_start_min_buckets",
        "movement_start_min_displacement_body_heights",
        "stationary_max_displacement_body_heights",
        "stationary_dwell_candidate_seconds",
        "minimum_episode_duration_seconds",
    }
)
_REFINEMENT_FIELDS = frozenset(
    {
        "enabled",
        "bucket_seconds",
        "search_radius_seconds",
        "movement_step_min_body_heights",
        "stationary_step_max_body_heights",
        "stationary_confirmation_buckets",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "policy_id",
        "pooled_recall_minimum",
        "pooled_f1_minimum",
        "matched_mean_tiou_minimum",
        "known_episode_candidate_coverage_minimum",
        "technical_hard_break_crossing_maximum",
        "finalist_count",
        "ranking_order",
    }
)
_RANKING_ORDER = (
    "gate_satisfied",
    "pooled_recall",
    "minimum_batch_recall",
    "known_episode_candidate_coverage",
    "pooled_f1",
    "minimum_batch_f1",
    "matched_mean_tiou",
    "serious_split_merge_count",
    "candidate_support",
    "candidate_id",
)


class CameraEpisodeBoundaryDevelopmentError(ValueError):
    """The development search config, input identity, or result audit failed."""


@dataclass(frozen=True)
class CameraEpisodeSegmenterSearchResult:
    output_dir: Path
    selected_candidate_id: str
    candidate_count: int
    selected_gate_satisfied: bool
    manifest_sha256: str


def load_camera_episode_segmenter_search_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load and fail closed on an undeclared W5D-01 search change."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraEpisodeBoundaryDevelopmentError(
            "cannot read camera segmenter search config"
        ) from exc
    if not isinstance(value, dict) or frozenset(value) != _CONFIG_FIELDS:
        raise CameraEpisodeBoundaryDevelopmentError("search config fields drifted")
    if value.get("schema_version") != CAMERA_EPISODE_SEGMENTER_SEARCH_CONFIG_SCHEMA_VERSION:
        raise CameraEpisodeBoundaryDevelopmentError("search config schema drifted")
    _stable_token(value.get("search_id"), "search_id")
    contract = value.get("development_contract")
    if not isinstance(contract, dict) or set(contract) != {
        "development_index",
        "run_manifest",
    }:
        raise CameraEpisodeBoundaryDevelopmentError(
            "development_contract fields drifted"
        )
    _validate_descriptor(contract["development_index"], "development_index")
    _validate_descriptor(contract["run_manifest"], "run_manifest")
    _validate_descriptor(value.get("proposal_baseline"), "proposal_baseline")
    _validate_descriptor(value.get("evaluation_config"), "evaluation_config")
    if value.get("expected_batches") != {"B01": 36, "B02": 12}:
        raise CameraEpisodeBoundaryDevelopmentError("expected B01/B02 counts drifted")
    _validate_selection_policy(value.get("selection_policy"))
    _validate_candidate_grid(value.get("candidates"))
    return value


def rank_segmenter_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic gate-aware, recall-first candidate ordering."""

    required = {
        "candidate_id",
        "gate_satisfied",
        "pooled_recall",
        "minimum_batch_recall",
        "known_episode_candidate_coverage",
        "pooled_f1",
        "minimum_batch_f1",
        "matched_mean_tiou",
        "technical_hard_break_crossing_count",
        "serious_split_merge_count",
        "candidate_support",
    }
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not required.issubset(candidate):
            raise CameraEpisodeBoundaryDevelopmentError(
                "candidate ranking fields are incomplete"
            )
        row = dict(candidate)
        candidate_id = _stable_token(row["candidate_id"], "candidate_id")
        if candidate_id in ids:
            raise CameraEpisodeBoundaryDevelopmentError(
                "duplicate candidate_id in ranking"
            )
        ids.add(candidate_id)
        if not isinstance(row["gate_satisfied"], bool):
            raise CameraEpisodeBoundaryDevelopmentError(
                "candidate gate_satisfied must be boolean"
            )
        for field in (
            "pooled_recall",
            "minimum_batch_recall",
            "known_episode_candidate_coverage",
            "pooled_f1",
            "minimum_batch_f1",
            "matched_mean_tiou",
        ):
            row[field] = _finite_ratio(row[field], field)
        for field in (
            "technical_hard_break_crossing_count",
            "serious_split_merge_count",
            "candidate_support",
        ):
            row[field] = _nonnegative_int(row[field], field)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["gate_satisfied"]),
            int(row["technical_hard_break_crossing_count"]),
            -float(row["pooled_recall"]),
            -float(row["minimum_batch_recall"]),
            -float(row["known_episode_candidate_coverage"]),
            -float(row["pooled_f1"]),
            -float(row["minimum_batch_f1"]),
            -float(row["matched_mean_tiou"]),
            int(row["serious_split_merge_count"]),
            int(row["candidate_support"]),
            str(row["candidate_id"]),
        ),
    )


def build_camera_episode_segmenter_search(
    *,
    project_root: str | Path,
    config_path: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
) -> CameraEpisodeSegmenterSearchResult:
    """Run the declared full-cohort grid, replay finalists, and publish a report."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    _require_under(root, config_file, "config_path")
    config = load_camera_episode_segmenter_search_config(config_file)
    work = Path(work_dir).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    _require_under(root, work, "work_dir")
    _require_under(root, output, "output_dir")
    if work.exists():
        raise FileExistsError(f"segmenter search work directory exists: {work}")
    if output.exists():
        raise FileExistsError(f"segmenter search output directory exists: {output}")

    contract = config["development_contract"]
    index_path = _resolve_descriptor(root, contract["development_index"], "development_index")
    contract_manifest_path = _resolve_descriptor(
        root, contract["run_manifest"], "run_manifest"
    )
    baseline_path = _resolve_descriptor(
        root, config["proposal_baseline"], "proposal_baseline"
    )
    evaluation_path = _resolve_descriptor(
        root, config["evaluation_config"], "evaluation_config"
    )
    baseline = load_camera_episode_boundary_proposal_config(baseline_path)
    _validate_baseline_candidate_binding(config["candidates"], baseline)
    inputs, boundary_manifest = _load_development_inputs(
        index_path,
        expected_batches=config["expected_batches"],
    )
    contract_manifest = _load_json(contract_manifest_path, "contract manifest")
    if (
        contract_manifest.get("status") != "ready"
        or contract_manifest.get("development_index", {}).get("record_count") != len(inputs)
        or contract_manifest.get("artifacts", {})
        .get("development_index.jsonl", {})
        .get("sha256")
        != _sha256_file(index_path)
    ):
        raise CameraEpisodeBoundaryDevelopmentError(
            "W5D-00 contract manifest does not bind the development index"
        )

    work.mkdir(parents=True)
    generated_profiles = work / "candidate_configs"
    generated_profiles.mkdir()
    run_results: list[dict[str, Any]] = []
    profile_payloads: dict[str, bytes] = {}
    for candidate in config["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        profile = _candidate_profile(candidate, baseline=baseline)
        profile_payload = _yaml_bytes(profile)
        profile_path = generated_profiles / f"{candidate_id}.yaml"
        _write_new_file(profile_path, profile_payload)
        load_camera_episode_boundary_development_profile(profile_path)
        profile_payloads[candidate_id] = profile_payload
        result = _execute_candidate(
            root=root,
            evaluation_config_path=evaluation_path,
            inputs=inputs,
            candidate=candidate,
            profile_path=profile_path,
            run_root=work / "screening" / candidate_id,
            selection_policy=config["selection_policy"],
        )
        run_results.append(result)

    ranked = rank_segmenter_candidates(
        [result["selection_metrics"] for result in run_results]
    )
    by_id = {str(result["candidate_id"]): result for result in run_results}
    finalist_count = int(config["selection_policy"]["finalist_count"])
    finalist_ids = [str(row["candidate_id"]) for row in ranked[:finalist_count]]
    for candidate_id in finalist_ids:
        original = by_id[candidate_id]
        candidate = next(
            row for row in config["candidates"] if row["candidate_id"] == candidate_id
        )
        replay = _execute_candidate(
            root=root,
            evaluation_config_path=evaluation_path,
            inputs=inputs,
            candidate=candidate,
            profile_path=generated_profiles / f"{candidate_id}.yaml",
            run_root=work / "replays" / candidate_id,
            selection_policy=config["selection_policy"],
        )
        if (
            original["comparison_payload"] != replay["comparison_payload"]
            or original["failures_payload"] != replay["failures_payload"]
            or original["proposal_tree_sha256"] != replay["proposal_tree_sha256"]
        ):
            raise CameraEpisodeBoundaryDevelopmentError(
                f"full replay drifted for candidate {candidate_id}"
            )
        original["metrics"]["full_replay"] = {
            "performed": True,
            "metrics_and_failures_identical": True,
            "proposal_tree_identical": True,
            "replay_proposal_tree_sha256": replay["proposal_tree_sha256"],
        }
    for result in run_results:
        result["metrics"].setdefault(
            "full_replay",
            {
                "performed": False,
                "metrics_and_failures_identical": False,
                "proposal_tree_identical": False,
                "replay_proposal_tree_sha256": None,
            },
        )

    selected_id = finalist_ids[0]
    selected = by_id[selected_id]
    report_payloads: dict[str, bytes] = {}
    for candidate in config["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        result = by_id[candidate_id]
        report_payloads[f"candidate_configs/{candidate_id}.yaml"] = profile_payloads[
            candidate_id
        ]
        report_payloads[f"runs/{candidate_id}/metrics.json"] = canonical_json_bytes(
            result["metrics"]
        )
        report_payloads[f"runs/{candidate_id}/failures.jsonl"] = result[
            "failures_payload"
        ]
    report_payloads["selected_segmenter_profile.yaml"] = profile_payloads[selected_id]
    report_payloads["README.md"] = _search_readme(
        config=config,
        ranked=ranked,
        selected=selected,
        finalist_ids=finalist_ids,
    ).encode("utf-8")
    report_payloads["VERIFICATION.md"] = _verification_markdown(
        config=config,
        selected=selected,
        finalist_ids=finalist_ids,
        boundary_manifest=boundary_manifest,
    ).encode("utf-8")
    manifest = {
        "schema_version": CAMERA_EPISODE_SEGMENTER_SEARCH_MANIFEST_SCHEMA_VERSION,
        "search_id": config["search_id"],
        "status": "selected",
        "selection_policy": deepcopy(config["selection_policy"]),
        "inputs": {
            "search_config": _file_descriptor(config_file),
            "development_index": _file_descriptor(index_path),
            "contract_manifest": _file_descriptor(contract_manifest_path),
            "proposal_baseline": _file_descriptor(baseline_path),
            "evaluation_config": _file_descriptor(evaluation_path),
            "development_record_count": len(inputs),
            "batch_counts": dict(sorted(Counter(row["batch_id"] for row in inputs).items())),
            "boundary_truth_manifest_sha256": hashlib.sha256(
                canonical_json_bytes(boundary_manifest)
            ).hexdigest(),
        },
        "candidate_count": len(run_results),
        "ranked_candidates": [
            {
                "rank": index + 1,
                **row,
            }
            for index, row in enumerate(ranked)
        ],
        "full_replay_candidate_ids": finalist_ids,
        "selected_candidate_id": selected_id,
        "selected_gate_satisfied": bool(
            selected["selection_metrics"]["gate_satisfied"]
        ),
        "selection_mode": (
            "all_gates_satisfied"
            if selected["selection_metrics"]["gate_satisfied"]
            else "stable_recall_first_fallback"
        ),
        "historical_reports_overwritten": False,
        "truth_or_labels_modified": False,
        "artifacts": {
            relative: _payload_descriptor(payload)
            for relative, payload in sorted(report_payloads.items())
        },
    }
    manifest_payload = canonical_json_bytes(manifest)
    report_payloads["segmenter_search_manifest.json"] = manifest_payload
    _commit_new_directory(output, report_payloads)
    return CameraEpisodeSegmenterSearchResult(
        output_dir=output,
        selected_candidate_id=selected_id,
        candidate_count=len(run_results),
        selected_gate_satisfied=bool(selected["selection_metrics"]["gate_satisfied"]),
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )


def _execute_candidate(
    *,
    root: Path,
    evaluation_config_path: Path,
    inputs: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    profile_path: Path,
    run_root: Path,
    selection_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if run_root.exists():
        raise FileExistsError(f"candidate run directory exists: {run_root}")
    run_root.mkdir(parents=True)
    proposals_root = run_root / "proposals"
    proposals_root.mkdir()
    index_rows: dict[str, list[dict[str, str]]] = {
        "pooled": [],
        "B01": [],
        "B02": [],
    }
    for item in inputs:
        source_video_id = str(item["source_video_id"])
        bundle_dir = proposals_root / source_video_id
        build_camera_episode_boundary_development_bundle(
            project_root=root,
            development_profile_path=profile_path,
            tracking_jsonl_path=item["tracking_path"],
            media_sidecar_path=item["media_sidecar_path"],
            output_dir=bundle_dir,
        )
        index_row = {
            "bundle_id": f"{candidate['candidate_id']}-{source_video_id}",
            "proposal_bundle_dir": bundle_dir.as_posix(),
            "human_boundary_jsonl": Path(item["boundary_path"]).as_posix(),
            "media_sidecar": Path(item["media_sidecar_path"]).as_posix(),
            "participant_id": str(item["participant_id"]),
            "session_id": str(item["session_id"]),
            "camera_setup_id": str(item["camera_setup_id"]),
            "clock_domain_id": str(item["clock_domain_id"]),
            "truth_source": "independent_human",
        }
        index_rows["pooled"].append(index_row)
        index_rows[str(item["batch_id"])].append(index_row)

    evaluations = run_root / "evaluations"
    evaluations.mkdir()
    cohort_summaries: dict[str, dict[str, Any]] = {}
    pooled_failures = b""
    for cohort in ("pooled", "B01", "B02"):
        index_path = run_root / f"{cohort}_batch_index.jsonl"
        _write_new_file(index_path, canonical_jsonl_bytes(index_rows[cohort]))
        evaluation_dir = evaluations / cohort
        build_camera_episode_boundary_evaluation_bundle(
            project_root=root,
            config_path=evaluation_config_path,
            batch_index_path=index_path,
            output_dir=evaluation_dir,
        )
        cohort_summaries[cohort] = _summarize_evaluation(evaluation_dir)
        if cohort == "pooled":
            pooled_failures = (evaluation_dir / "failures.jsonl").read_bytes()

    hard_break_audit = _audit_technical_hard_breaks(proposals_root)
    _audit_failure_completeness(evaluations / "pooled")
    pooled = cohort_summaries["pooled"]
    minimum_batch_recall = min(
        cohort_summaries[batch]["recall"] for batch in ("B01", "B02")
    )
    minimum_batch_f1 = min(
        cohort_summaries[batch]["f1"] for batch in ("B01", "B02")
    )
    gate_checks = {
        "pooled_recall": pooled["recall"]
        >= float(selection_policy["pooled_recall_minimum"]),
        "pooled_f1": pooled["f1"] >= float(selection_policy["pooled_f1_minimum"]),
        "matched_mean_tiou": pooled["matched_mean_tiou"]
        >= float(selection_policy["matched_mean_tiou_minimum"]),
        "known_episode_candidate_coverage": pooled[
            "known_episode_candidate_coverage"
        ]
        >= float(selection_policy["known_episode_candidate_coverage_minimum"]),
        "technical_hard_break_crossing_count": hard_break_audit["crossing_count"]
        <= int(selection_policy["technical_hard_break_crossing_maximum"]),
    }
    selection_metrics = {
        "candidate_id": candidate["candidate_id"],
        "gate_satisfied": all(gate_checks.values()),
        "pooled_recall": pooled["recall"],
        "minimum_batch_recall": minimum_batch_recall,
        "known_episode_candidate_coverage": pooled[
            "known_episode_candidate_coverage"
        ],
        "pooled_f1": pooled["f1"],
        "minimum_batch_f1": minimum_batch_f1,
        "matched_mean_tiou": pooled["matched_mean_tiou"],
        "technical_hard_break_crossing_count": hard_break_audit["crossing_count"],
        "serious_split_merge_count": pooled["split_candidate_count"]
        + pooled["merge_candidate_count"],
        "candidate_support": pooled["candidate_support"],
    }
    profile_sha256 = _sha256_file(profile_path)
    metrics = {
        "schema_version": CAMERA_EPISODE_SEGMENTER_RUN_METRICS_SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "stage": candidate["stage"],
        "parameter_group": candidate["parameter_group"],
        "profile_sha256": profile_sha256,
        "cohorts": cohort_summaries,
        "selection_metrics": selection_metrics,
        "gate_checks": gate_checks,
        "technical_hard_break_audit": hard_break_audit,
        "failure_completeness_audit": {
            "all_primary_misses_retained": True,
            "all_split_merge_diagnostics_retained": True,
        },
        "screening_scope": "full_b01_b02_development_48_videos",
    }
    comparison_payload = canonical_json_bytes(
        {
            "cohorts": cohort_summaries,
            "selection_metrics": selection_metrics,
            "gate_checks": gate_checks,
            "technical_hard_break_audit": hard_break_audit,
        }
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "metrics": metrics,
        "selection_metrics": selection_metrics,
        "comparison_payload": comparison_payload,
        "failures_payload": pooled_failures,
        "proposal_tree_sha256": _tree_sha256(proposals_root),
    }


def _summarize_evaluation(evaluation_dir: Path) -> dict[str, Any]:
    metrics = _load_json(evaluation_dir / "metrics.json", "evaluation metrics")
    primary = metrics["views"]["all_locomotion_candidates"]
    detection = primary["detection"]
    localization = primary["localization"]
    coverage = metrics["coverage"]["ready_truth_overlap_status_counts"]
    ready_truth = sum(int(value) for value in coverage.values())
    covered = int(coverage["proposed_overlap"]) + int(coverage["uncertain_only_overlap"])
    diagnostics = _load_jsonl(
        evaluation_dir / "split_merge_diagnostics.jsonl",
        "split/merge diagnostics",
    )
    failures = _load_jsonl(evaluation_dir / "failures.jsonl", "evaluation failures")
    failure_counts = Counter(str(row["failure_type"]) for row in failures)
    tiou = localization["temporal_iou"]["mean"]
    return {
        "ready_truth_support": int(detection["ready_truth_support"]),
        "candidate_support": int(detection["candidate_support"]),
        "matched_count": int(detection["matched_count"]),
        "unmatched_truth_count": int(detection["unmatched_truth_count"]),
        "unmatched_candidate_count": int(detection["unmatched_candidate_count"]),
        "precision": _metric_number(detection["precision"]),
        "recall": _metric_number(detection["recall"]),
        "f1": _metric_number(detection["f1"]),
        "matched_mean_tiou": _metric_number(tiou),
        "mean_onset_absolute_error_sec": _metric_number(
            localization["onset_absolute_error_sec"]["mean"]
        ),
        "mean_offset_absolute_error_sec": _metric_number(
            localization["offset_absolute_error_sec"]["mean"]
        ),
        "known_episode_candidate_coverage": covered / ready_truth if ready_truth else 0.0,
        "coverage_counts": dict(coverage),
        "split_candidate_count": sum(
            row["diagnostic_type"] == "split_candidate" for row in diagnostics
        ),
        "merge_candidate_count": sum(
            row["diagnostic_type"] == "merge_candidate" for row in diagnostics
        ),
        "fragmentation_candidate_count": sum(
            row["diagnostic_type"] == "fragmentation_candidate"
            for row in diagnostics
        ),
        "failure_count": len(failures),
        "failure_type_counts": dict(sorted(failure_counts.items())),
    }


def _audit_failure_completeness(evaluation_dir: Path) -> None:
    unmatched = _load_jsonl(
        evaluation_dir / "unmatched_truth.jsonl", "unmatched truth"
    )
    diagnostics = _load_jsonl(
        evaluation_dir / "split_merge_diagnostics.jsonl", "split/merge diagnostics"
    )
    failures = _load_jsonl(evaluation_dir / "failures.jsonl", "failures")
    primary_misses = {
        str(row["episode_id"])
        for row in unmatched
        if row["view"] == "all_locomotion_candidates"
    }
    retained_misses = {
        str(row["episode_id"])
        for row in failures
        if row["failure_type"] == "unmatched_ready_truth"
        and row["view"] == "all_locomotion_candidates"
    }
    if primary_misses != retained_misses:
        raise CameraEpisodeBoundaryDevelopmentError(
            "primary-view misses are not fully retained in failures"
        )
    expected_serious = {
        (
            str(row["diagnostic_type"]),
            tuple(row["proposal_ids"]),
            tuple(row["episode_ids"]),
        )
        for row in diagnostics
        if row["diagnostic_type"] in {"split_candidate", "merge_candidate"}
    }
    retained_serious = {
        (
            str(row["failure_type"]),
            tuple(row["proposal_ids"]),
            tuple(row["episode_ids"]),
        )
        for row in failures
        if row["failure_type"] in {"split_candidate", "merge_candidate"}
    }
    if expected_serious != retained_serious:
        raise CameraEpisodeBoundaryDevelopmentError(
            "split/merge diagnostics are not fully retained in failures"
        )


def _audit_technical_hard_breaks(proposals_root: Path) -> dict[str, Any]:
    crossing_rows: list[dict[str, Any]] = []
    split_count = 0
    for bundle in sorted(path for path in proposals_root.iterdir() if path.is_dir()):
        proposals = _load_jsonl(bundle / "proposals.jsonl", "proposal rows")
        diagnostics = _load_jsonl(bundle / "diagnostics.jsonl", "proposal diagnostics")
        by_scope: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in diagnostics:
            key = (
                row["source_group_id"],
                row["source_video_id"],
                row["device_id"],
                row["setup_id"],
                row["stream_epoch"],
                int(row["track_id"]),
            )
            by_scope.setdefault(key, []).append(row)
        for scope_rows in by_scope.values():
            ordered = sorted(scope_rows, key=lambda row: int(row["technical_segment_index"]))
            for left, right in zip(ordered, ordered[1:]):
                if not (left["hard_break_reasons"] or right["hard_break_reasons"]):
                    continue
                split_count += 1
                split_time = (
                    float(left["segment_end_observation_sec"])
                    + float(right["segment_start_observation_sec"])
                ) / 2.0
                left_index = int(left["technical_segment_index"])
                right_index = int(right["technical_segment_index"])
                for proposal in proposals:
                    if int(proposal["track_id"]) != int(left["track_id"]):
                        continue
                    segment_index = int(proposal["technical_segment_index"])
                    crosses = (
                        segment_index == left_index
                        and float(proposal["end_sec_exclusive"]) > split_time + 1e-9
                    ) or (
                        segment_index == right_index
                        and float(proposal["start_sec"]) < split_time - 1e-9
                    )
                    if crosses:
                        crossing_rows.append(
                            {
                                "source_video_id": proposal["source_video_id"],
                                "proposal_id": proposal["proposal_id"],
                                "technical_split_sec": split_time,
                            }
                        )
    return {
        "technical_hard_break_split_count": split_count,
        "crossing_count": len(crossing_rows),
        "crossings": crossing_rows,
    }


def _load_development_inputs(
    index_path: Path,
    *,
    expected_batches: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_jsonl(index_path, "development index")
    output: list[dict[str, Any]] = []
    boundary_manifest: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != "wandering-camera-development-index-v1":
            raise CameraEpisodeBoundaryDevelopmentError(
                "development index schema drifted"
            )
        source_video_id = _stable_token(row.get("source_video_id"), "source_video_id")
        if source_video_id in source_ids:
            raise CameraEpisodeBoundaryDevelopmentError(
                "duplicate source_video_id in development index"
            )
        source_ids.add(source_video_id)
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise CameraEpisodeBoundaryDevelopmentError(
                "development index artifacts are invalid"
            )
        resolved: dict[str, Path] = {}
        for name in ("tracking", "media_sidecar", "truth"):
            locator = artifacts.get(name)
            if not isinstance(locator, Mapping):
                raise CameraEpisodeBoundaryDevelopmentError(
                    f"development {name} locator is invalid"
                )
            path = Path(str(locator.get("path"))).resolve(strict=True)
            if not path.is_file() or _sha256_file(path) != locator.get("sha256"):
                raise CameraEpisodeBoundaryDevelopmentError(
                    f"development {name} identity drifted for {source_video_id}"
                )
            resolved[name] = path
        boundary_path = resolved["truth"].with_name("episode_boundaries.jsonl")
        if not boundary_path.is_file():
            raise CameraEpisodeBoundaryDevelopmentError(
                f"boundary truth is missing for {source_video_id}"
            )
        boundary_descriptor = _file_descriptor(boundary_path)
        boundary_manifest.append(
            {
                "source_video_id": source_video_id,
                **boundary_descriptor,
            }
        )
        output.append(
            {
                "batch_id": str(row["batch_id"]),
                "source_video_id": source_video_id,
                "participant_id": str(row["participant_id"]),
                "session_id": str(row["session_id"]),
                "camera_setup_id": str(row["camera_setup_id"]),
                "clock_domain_id": str(row["clock_domain_id"]),
                "tracking_path": resolved["tracking"],
                "media_sidecar_path": resolved["media_sidecar"],
                "boundary_path": boundary_path,
            }
        )
    counts = Counter(row["batch_id"] for row in output)
    if dict(sorted(counts.items())) != dict(expected_batches):
        raise CameraEpisodeBoundaryDevelopmentError(
            "development index B01/B02 counts drifted"
        )
    return (
        sorted(output, key=lambda row: row["source_video_id"]),
        sorted(boundary_manifest, key=lambda row: row["source_video_id"]),
    )


def _candidate_profile(
    candidate: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    profile = deepcopy(dict(baseline))
    profile["schema_version"] = (
        CAMERA_EPISODE_BOUNDARY_DEVELOPMENT_PROFILE_SCHEMA_VERSION
    )
    profile["purpose"] = "b01_b02_development_parameter_search"
    profile["profile_id"] = str(candidate["candidate_id"])
    profile["base_producer_config_id"] = PRODUCER_CONFIG_ID
    profile["state_machine"] = deepcopy(dict(candidate["state_machine"]))
    profile["boundary_refinement"] = deepcopy(
        dict(candidate["boundary_refinement"])
    )
    profile["protocol"] = {
        "movement_start_candidate_basis": (
            f"w5d01_{candidate['parameter_group']}_{candidate['candidate_id']}"
        ),
        "movement_start_candidate_validated": False,
        "stationary_dwell_candidate_basis": (
            f"w5d01_{candidate['parameter_group']}_{candidate['candidate_id']}"
        ),
        "stationary_dwell_candidate_validated": False,
    }
    return profile


def _validate_baseline_candidate_binding(
    candidates: Sequence[Mapping[str, Any]],
    frozen_baseline: Mapping[str, Any],
) -> None:
    baseline = next(candidate for candidate in candidates if candidate["stage"] == "baseline")
    if dict(baseline["state_machine"]) != dict(frozen_baseline["state_machine"]):
        raise CameraEpisodeBoundaryDevelopmentError(
            "declared baseline candidate drifted from the frozen proposal baseline"
        )
    if baseline["boundary_refinement"].get("enabled") is not False:
        raise CameraEpisodeBoundaryDevelopmentError(
            "declared baseline candidate unexpectedly enables boundary refinement"
        )


def _validate_candidate_grid(value: Any) -> None:
    if not isinstance(value, list) or len(value) < 4:
        raise CameraEpisodeBoundaryDevelopmentError(
            "candidate grid must contain baseline and a small comparison grid"
        )
    ids: set[str] = set()
    baseline_rows: list[Mapping[str, Any]] = []
    for candidate in value:
        if not isinstance(candidate, Mapping) or frozenset(candidate) != _CANDIDATE_FIELDS:
            raise CameraEpisodeBoundaryDevelopmentError("candidate fields drifted")
        candidate_id = _stable_token(candidate["candidate_id"], "candidate_id")
        if candidate_id in ids:
            raise CameraEpisodeBoundaryDevelopmentError("duplicate candidate_id")
        ids.add(candidate_id)
        if candidate.get("stage") not in {"baseline", "one_group", "combined"}:
            raise CameraEpisodeBoundaryDevelopmentError("candidate stage is invalid")
        if candidate.get("parameter_group") not in {
            "baseline",
            "movement_opening",
            "stationary_closing",
            "boundary_refinement",
            "movement_stationary_refinement",
        }:
            raise CameraEpisodeBoundaryDevelopmentError(
                "candidate parameter_group is invalid"
            )
        if not isinstance(candidate.get("state_machine"), Mapping) or frozenset(
            candidate["state_machine"]
        ) != _STATE_FIELDS:
            raise CameraEpisodeBoundaryDevelopmentError(
                "candidate state_machine fields drifted"
            )
        if not isinstance(candidate.get("boundary_refinement"), Mapping) or frozenset(
            candidate["boundary_refinement"]
        ) != _REFINEMENT_FIELDS:
            raise CameraEpisodeBoundaryDevelopmentError(
                "candidate boundary_refinement fields drifted"
            )
        if candidate["stage"] == "baseline":
            baseline_rows.append(candidate)
    if len(baseline_rows) != 1 or baseline_rows[0]["parameter_group"] != "baseline":
        raise CameraEpisodeBoundaryDevelopmentError(
            "candidate grid must declare exactly one baseline"
        )
    baseline = baseline_rows[0]
    baseline_state = dict(baseline["state_machine"])
    baseline_refinement = dict(baseline["boundary_refinement"])
    if baseline_refinement.get("enabled") is not False:
        raise CameraEpisodeBoundaryDevelopmentError(
            "baseline refinement must remain disabled"
        )
    for candidate in value:
        if candidate["stage"] != "one_group":
            continue
        state = dict(candidate["state_machine"])
        refinement = dict(candidate["boundary_refinement"])
        group = candidate["parameter_group"]
        if group == "movement_opening":
            unchanged = {
                key for key in _STATE_FIELDS if not key.startswith("movement_start_")
            }
            valid = all(state[key] == baseline_state[key] for key in unchanged) and (
                refinement == baseline_refinement
            )
        elif group == "stationary_closing":
            unchanged = {
                "movement_start_min_buckets",
                "movement_start_min_displacement_body_heights",
                "minimum_episode_duration_seconds",
            }
            valid = all(state[key] == baseline_state[key] for key in unchanged) and (
                refinement == baseline_refinement
            )
        elif group == "boundary_refinement":
            valid = state == baseline_state and refinement != baseline_refinement
        else:
            valid = False
        if not valid:
            raise CameraEpisodeBoundaryDevelopmentError(
                "one-group candidate changed undeclared parameters"
            )


def _validate_selection_policy(value: Any) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != _SELECTION_FIELDS:
        raise CameraEpisodeBoundaryDevelopmentError(
            "selection_policy fields drifted"
        )
    _stable_token(value.get("policy_id"), "policy_id")
    for field in (
        "pooled_recall_minimum",
        "pooled_f1_minimum",
        "matched_mean_tiou_minimum",
        "known_episode_candidate_coverage_minimum",
    ):
        _finite_ratio(value.get(field), field)
    if value.get("technical_hard_break_crossing_maximum") != 0:
        raise CameraEpisodeBoundaryDevelopmentError(
            "technical hard-break crossing maximum must remain zero"
        )
    finalists = value.get("finalist_count")
    if isinstance(finalists, bool) or not isinstance(finalists, int) or not 2 <= finalists <= 3:
        raise CameraEpisodeBoundaryDevelopmentError(
            "finalist_count must be two or three"
        )
    if value.get("ranking_order") != list(_RANKING_ORDER):
        raise CameraEpisodeBoundaryDevelopmentError(
            "recall-first ranking order drifted"
        )


def _resolve_descriptor(root: Path, value: Mapping[str, Any], role: str) -> Path:
    _validate_descriptor(value, role)
    relative = PurePosixPath(str(value["path"]).replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} path is unsafe")
    path = (root / relative).resolve(strict=True)
    _require_under(root, path, role)
    if not path.is_file() or _sha256_file(path) != value["sha256"]:
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} identity drifted")
    return path


def _validate_descriptor(value: Any, role: str) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != _DESCRIPTOR_FIELDS:
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} descriptor fields drifted")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} path is invalid")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} sha256 is invalid")


def _search_readme(
    *,
    config: Mapping[str, Any],
    ranked: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    finalist_ids: Sequence[str],
) -> str:
    metric = selected["selection_metrics"]
    return (
        "# W5D-01 B01+B02 segmenter search\n\n"
        "The search uses all 48 W5D-00 development videos. B01 and B02 are "
        "reported separately; these are development metrics, not held-out or "
        "cross-person evidence.\n\n"
        f"- search_id: `{config['search_id']}`\n"
        f"- selected_candidate_id: `{selected['candidate_id']}`\n"
        f"- selection_mode: `{'all_gates_satisfied' if metric['gate_satisfied'] else 'stable_recall_first_fallback'}`\n"
        f"- pooled recall/F1/mean-tIoU/coverage: "
        f"`{metric['pooled_recall']:.6f}/{metric['pooled_f1']:.6f}/"
        f"{metric['matched_mean_tiou']:.6f}/"
        f"{metric['known_episode_candidate_coverage']:.6f}`\n"
        f"- finalist full replays: `{', '.join(finalist_ids)}`\n"
        f"- technical hard-break crossings: "
        f"`{metric['technical_hard_break_crossing_count']}`\n"
        f"- ranked candidate count: `{len(ranked)}`\n\n"
        "The selected YAML is the only downstream W5D-02 development profile. "
        "Historical v1/v2 reports were not overwritten.\n"
    )


def _verification_markdown(
    *,
    config: Mapping[str, Any],
    selected: Mapping[str, Any],
    finalist_ids: Sequence[str],
    boundary_manifest: Sequence[Mapping[str, Any]],
) -> str:
    metric = selected["selection_metrics"]
    cohorts = selected["metrics"]["cohorts"]
    return (
        "# W5D-01 verification\n\n"
        f"- development records: `48` (`B01=36`, `B02=12`)\n"
        f"- boundary truth files hashed: `{len(boundary_manifest)}`\n"
        f"- selected: `{selected['candidate_id']}`\n"
        f"- pooled gate satisfied: `{str(metric['gate_satisfied']).lower()}`\n"
        f"- B01 recall/F1/tIoU/coverage: "
        f"`{cohorts['B01']['recall']:.6f}/{cohorts['B01']['f1']:.6f}/"
        f"{cohorts['B01']['matched_mean_tiou']:.6f}/"
        f"{cohorts['B01']['known_episode_candidate_coverage']:.6f}`\n"
        f"- B02 recall/F1/tIoU/coverage: "
        f"`{cohorts['B02']['recall']:.6f}/{cohorts['B02']['f1']:.6f}/"
        f"{cohorts['B02']['matched_mean_tiou']:.6f}/"
        f"{cohorts['B02']['known_episode_candidate_coverage']:.6f}`\n"
        f"- hard-break crossing count: "
        f"`{metric['technical_hard_break_crossing_count']}`\n"
        f"- finalist deterministic replays: `{', '.join(finalist_ids)}`\n"
        "- all primary misses retained in failures: `true`\n"
        "- all split/merge diagnostics retained in failures: `true`\n"
        "- human truth or labels modified: `false`\n"
        "- historical evidence overwritten: `false`\n\n"
        "Run with the project `eldercare-ai` conda environment using "
        "`scripts/wandering/tune_camera_episode_boundaries.py`.\n"
    )


def _metric_number(value: Any) -> float:
    if isinstance(value, bool):
        raise CameraEpisodeBoundaryDevelopmentError("metric cannot be boolean")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if value == "not_computable":
        return 0.0
    raise CameraEpisodeBoundaryDevelopmentError("metric is invalid")


def _finite_ratio(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} must be a finite ratio")
    return float(value)


def _nonnegative_int(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} must be nonnegative")
    return value


def _stable_token(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(not (character.isalnum() or character in "._-") for character in value)
    ):
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} is not a stable token")
    return value


def _require_under(root: Path, path: Path, role: str) -> None:
    if not path.is_relative_to(root):
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} escapes the project root")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, _NonFiniteJsonError) as exc:
        raise CameraEpisodeBoundaryDevelopmentError(f"cannot read {role}") from exc
    if not isinstance(value, dict):
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} must be an object")
    return value


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CameraEpisodeBoundaryDevelopmentError(f"{role} is missing")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise CameraEpisodeBoundaryDevelopmentError(
                    f"{role} rows must be objects"
                )
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, _NonFiniteJsonError) as exc:
        raise CameraEpisodeBoundaryDevelopmentError(f"cannot read {role}") from exc
    return rows


class _NonFiniteJsonError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise _NonFiniteJsonError(f"non-finite JSON constant: {value}")


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _payload_descriptor(payload: bytes) -> dict[str, Any]:
    return {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _tree_sha256(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "byte_count": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"segmenter search output directory exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            destination = staging / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for relative, payload in files.items():
            if (staging / PurePosixPath(relative)).read_bytes() != payload:
                raise CameraEpisodeBoundaryDevelopmentError(
                    f"report staging verification failed: {relative}"
                )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "CAMERA_EPISODE_SEGMENTER_RUN_METRICS_SCHEMA_VERSION",
    "CAMERA_EPISODE_SEGMENTER_SEARCH_CONFIG_SCHEMA_VERSION",
    "CAMERA_EPISODE_SEGMENTER_SEARCH_MANIFEST_SCHEMA_VERSION",
    "CameraEpisodeBoundaryDevelopmentError",
    "CameraEpisodeSegmenterSearchResult",
    "build_camera_episode_segmenter_search",
    "load_camera_episode_segmenter_search_config",
    "rank_segmenter_candidates",
]
