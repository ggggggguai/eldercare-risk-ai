"""W5D-05 backend-consumable camera algorithm handoff assembler."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
    CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION,
    _validate_context_review,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_baseline import (
    CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION,
    CAMERA_BASELINE_PROFILE_SCHEMA_VERSION,
    CAMERA_DAILY_REPORT_SCHEMA_VERSION,
    ValidatedCameraDailyBaselineBundle,
    _validate_baseline_deviation,
    _validate_baseline_profile,
    _validate_daily_report,
    load_validated_camera_daily_baseline_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    EPISODE_RESULT_SCHEMA_VERSION,
    _validate_episode_result,
)


CAMERA_HANDOFF_CONFIG_SCHEMA_VERSION = "wandering-camera-handoff-config-v1"
CAMERA_HANDOFF_MANIFEST_SCHEMA_VERSION = "wandering-handoff-manifest-v2"
CAMERA_HANDOFF_RUN_SUMMARY_SCHEMA_VERSION = "wandering-handoff-run-summary-v2"
MODULE = "mental_health"

HANDOFF_FILES = (
    "handoff_manifest.json",
    "episode_results.jsonl",
    "context_reviews.jsonl",
    "daily_reports.jsonl",
    "baseline_profiles.jsonl",
    "baseline_deviations.jsonl",
    "run_summary.json",
    "README.md",
)
_DATA_FILES = (
    "episode_results.jsonl",
    "context_reviews.jsonl",
    "daily_reports.jsonl",
    "baseline_profiles.jsonl",
    "baseline_deviations.jsonl",
)
_DESCRIBED_FILES = frozenset(HANDOFF_FILES) - {"handoff_manifest.json"}
_SCHEMA_IDS = {
    "episode_results.jsonl": EPISODE_RESULT_SCHEMA_VERSION,
    "context_reviews.jsonl": CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION,
    "daily_reports.jsonl": CAMERA_DAILY_REPORT_SCHEMA_VERSION,
    "baseline_profiles.jsonl": CAMERA_BASELINE_PROFILE_SCHEMA_VERSION,
    "baseline_deviations.jsonl": CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION,
    "handoff_manifest.json": CAMERA_HANDOFF_MANIFEST_SCHEMA_VERSION,
    "run_summary.json": CAMERA_HANDOFF_RUN_SUMMARY_SCHEMA_VERSION,
}
_STAGE_SPECS = {
    "episode": (
        "W5D-02",
        "wandering-camera-episode-pipeline-handoff-manifest-partial-v1",
        "episode_results.jsonl",
    ),
    "context": (
        "W5D-03A",
        "wandering-camera-context-core-handoff-manifest-partial-v1",
        "context_reviews.jsonl",
    ),
    "daily": (
        "W5D-04",
        "wandering-camera-daily-baseline-handoff-manifest-partial-v1",
        "daily_reports.jsonl",
    ),
}
_STATUS_VALUES = frozenset({"ready", "uncertain", "unavailable", "error"})
_DELIVERY_VALUES = frozenset(
    {
        "algorithm_ready_with_home_smoke_pending",
        "algorithm_ready_after_home_smoke",
    }
)
_SHA256_LENGTH = 64
_FORBIDDEN_ROW_FIELDS = frozenset(
    {
        "algorithm_event",
        "diagnosis",
        "medical_diagnosis",
        "risk_decision",
        "alert_decision",
    }
)


class CameraHandoffError(ValueError):
    """An upstream stage or final W5D-05 handoff contract is invalid."""


@dataclass(frozen=True)
class CameraHandoffBuildResult:
    output_dir: Path
    episode_result_count: int
    context_review_count: int
    daily_report_count: int
    baseline_profile_count: int
    baseline_deviation_count: int
    status: str
    delivery_status: str


@dataclass(frozen=True)
class ValidatedCameraHandoffBundle:
    output_dir: Path
    manifest: Mapping[str, Any]
    episode_results: tuple[Mapping[str, Any], ...]
    context_reviews: tuple[Mapping[str, Any], ...]
    daily_reports: tuple[Mapping[str, Any], ...]
    baseline_profiles: tuple[Mapping[str, Any], ...]
    baseline_deviations: tuple[Mapping[str, Any], ...]
    run_summary: Mapping[str, Any]


@dataclass(frozen=True)
class _StageBundle:
    name: str
    root: Path
    manifest_path: Path
    manifest_bytes: bytes
    manifest: Mapping[str, Any]
    artifact_name: str
    artifact_bytes: bytes
    artifact_rows: tuple[Mapping[str, Any], ...]


def load_camera_handoff_config(path: str | Path) -> dict[str, Any]:
    """Load the pinned W5D-05 assembly configuration."""

    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CameraHandoffError(f"cannot read camera handoff config: {config_path}") from exc
    _validate_config(value)
    return dict(value)


def build_camera_handoff_bundle(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    episode_bundle_dir: str | Path | None = None,
    context_bundle_dir: str | Path | None = None,
    daily_bundle_dir: str | Path | None = None,
    allow_unpinned_stage_overrides: bool = False,
) -> CameraHandoffBuildResult:
    """Assemble verified W5D-02/03A/04 artifacts into one fresh directory."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"camera handoff output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config_bytes = _read_bytes(config_file, "camera handoff config")
    config = load_camera_handoff_config(config_file)
    schema_refs = _load_schema_refs(root, config["schemas"])

    overrides = {
        "episode": episode_bundle_dir,
        "context": context_bundle_dir,
        "daily": daily_bundle_dir,
    }
    stages = {
        name: _load_stage_bundle(
            root=root,
            name=name,
            config=config["inputs"][name],
            override=overrides[name],
            allow_unpinned_override=allow_unpinned_stage_overrides,
        )
        for name in ("episode", "context", "daily")
    }

    episodes = [dict(row) for row in stages["episode"].artifact_rows]
    contexts = [dict(row) for row in stages["context"].artifact_rows]
    daily_validated = _load_daily_stage(stages["daily"])
    daily = [dict(row) for row in daily_validated.daily_reports]
    profiles = [dict(row) for row in daily_validated.baseline_profiles]
    deviations = [dict(row) for row in daily_validated.baseline_deviations]
    _validate_all_rows(episodes, contexts, daily, profiles, deviations)
    _validate_upstream_bindings(stages, daily_validated, episodes, contexts)
    conservation = _validate_cross_stage(
        episodes, contexts, daily, profiles, deviations
    )

    home = _validated_home_state(config, stages)
    model_identity = _common_episode_identity(episodes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    policy_sha256 = hashlib.sha256(
        canonical_json_bytes(config["policy"])
    ).hexdigest()
    identity = {
        "model_id": model_identity["model_id"],
        "model_sha256": model_identity["model_sha256"],
        "config_id": str(config["identity"]["config_id"]),
        "config_sha256": config_sha256,
        "policy_id": str(config["identity"]["policy_id"]),
        "policy_sha256": policy_sha256,
    }
    _validate_identity(identity)

    effective_run_id = run_id or str(config["handoff_id"])
    _nonempty(effective_run_id, "run_id")
    started_at = datetime.now(timezone.utc)
    status = _aggregate_status(episodes, contexts, daily, profiles, deviations)
    source_refs = [_stage_source_ref(stage) for stage in stages.values()]
    quality_flags = _quality_flags(stages, home)
    degraded_components = _degraded_components(
        episodes, contexts, daily, profiles, stages, home
    )
    failure_counts = _failure_counts(episodes, contexts, daily, profiles, deviations)
    counts = {
        "episode_results": len(episodes),
        "context_reviews": len(contexts),
        "daily_reports": len(daily),
        "baseline_profiles": len(profiles),
        "baseline_deviations": len(deviations),
    }
    run_summary = {
        "schema_version": CAMERA_HANDOFF_RUN_SUMMARY_SCHEMA_VERSION,
        "module": MODULE,
        "run_id": effective_run_id,
        "started_at": _iso(started_at),
        "finished_at": _iso(datetime.now(timezone.utc)),
        "person_id": _single_or_none(row["person_id"] for row in daily),
        "session_id": None,
        "source_video_id": None,
        "status": status,
        "delivery_status": str(config["delivery_status"]),
        "validation_scope": str(config["validation_scope"]),
        **home,
        "algorithm_event_emitted": False,
        "medical_diagnosis_emitted": False,
        "backend_or_frontend_implemented": False,
        "input_counts": {
            "source_stages": 3,
            "episode_results": len(episodes),
            "context_reviews": len(contexts),
            "daily_reports": len(daily),
            "baseline_profiles": len(profiles),
            "baseline_deviations": len(deviations),
        },
        "output_counts": counts,
        "failure_counts": failure_counts,
        "degraded_components": degraded_components,
        "quality_flags": quality_flags,
        "conservation": conservation,
        "identity": identity,
        "source_refs": source_refs,
    }
    _validate_run_summary(run_summary)

    payloads = {
        "episode_results.jsonl": stages["episode"].artifact_bytes,
        "context_reviews.jsonl": stages["context"].artifact_bytes,
        "daily_reports.jsonl": _read_bytes(
            stages["daily"].root / "daily_reports.jsonl", "daily reports"
        ),
        "baseline_profiles.jsonl": _read_bytes(
            stages["daily"].root / "baseline_profiles.jsonl", "baseline profiles"
        ),
        "baseline_deviations.jsonl": _read_bytes(
            stages["daily"].root / "baseline_deviations.jsonl",
            "baseline deviations",
        ),
        "run_summary.json": canonical_json_bytes(run_summary),
        "README.md": _readme(effective_run_id).encode("utf-8"),
    }
    descriptors = {
        name: _descriptor(
            payload,
            _SCHEMA_IDS.get(name),
            counts.get(name.removesuffix(".jsonl"))
            if name.endswith(".jsonl")
            else (1 if name == "run_summary.json" else None),
        )
        for name, payload in payloads.items()
    }
    stage_descriptors = {
        name: _stage_descriptor(stage) for name, stage in stages.items()
    }
    manifest = {
        "schema_version": CAMERA_HANDOFF_MANIFEST_SCHEMA_VERSION,
        "module": MODULE,
        "handoff_id": effective_run_id,
        "created_at": _iso(datetime.now(timezone.utc)),
        "person_id": run_summary["person_id"],
        "session_id": None,
        "source_video_id": None,
        "status": status,
        "delivery_status": str(config["delivery_status"]),
        "validation_scope": str(config["validation_scope"]),
        **home,
        "algorithm_event_emitted": False,
        "medical_diagnosis_emitted": False,
        "backend_or_frontend_implemented": False,
        "artifacts": descriptors,
        "schemas": schema_refs,
        "source_stages": stage_descriptors,
        "quality_flags": quality_flags,
        "identity": identity,
        "source_refs": source_refs,
    }
    _validate_manifest(manifest)
    final_payloads = {
        "handoff_manifest.json": canonical_json_bytes(manifest),
        **payloads,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for name, payload in final_payloads.items():
            _write_new(staging / name, payload)
        loaded = load_validated_camera_handoff_bundle(staging)
        if (
            list(loaded.episode_results) != episodes
            or list(loaded.context_reviews) != contexts
            or list(loaded.daily_reports) != daily
            or list(loaded.baseline_profiles) != profiles
            or list(loaded.baseline_deviations) != deviations
        ):
            raise CameraHandoffError("public loader readback differs from source rows")
        if output.exists():
            raise FileExistsError(f"camera handoff output already exists: {output}")
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return CameraHandoffBuildResult(
        output_dir=output,
        episode_result_count=len(episodes),
        context_review_count=len(contexts),
        daily_report_count=len(daily),
        baseline_profile_count=len(profiles),
        baseline_deviation_count=len(deviations),
        status=status,
        delivery_status=str(config["delivery_status"]),
    )


def load_validated_camera_handoff_bundle(
    output_dir: str | Path,
) -> ValidatedCameraHandoffBundle:
    """Load and independently validate a final eight-file handoff bundle."""

    output = Path(output_dir)
    if not output.is_dir():
        raise CameraHandoffError("camera handoff bundle is not a directory")
    entries = {entry.name: entry for entry in output.iterdir()}
    if set(entries) != set(HANDOFF_FILES) or any(
        not entry.is_file() for entry in entries.values()
    ):
        raise CameraHandoffError("camera handoff bundle file set drifted")

    _, manifest = _load_json(output / "handoff_manifest.json", "handoff manifest")
    _validate_manifest(manifest)
    for name, descriptor in manifest["artifacts"].items():
        _verify_descriptor(output / name, descriptor, f"handoff {name} descriptor")

    episode_bytes, episodes = _load_jsonl(
        output / "episode_results.jsonl", "episode results"
    )
    context_bytes, contexts = _load_jsonl(
        output / "context_reviews.jsonl", "context reviews", allow_empty=True
    )
    daily_bytes, daily = _load_jsonl(output / "daily_reports.jsonl", "daily reports")
    profile_bytes, profiles = _load_jsonl(
        output / "baseline_profiles.jsonl", "baseline profiles"
    )
    deviation_bytes, deviations = _load_jsonl(
        output / "baseline_deviations.jsonl", "baseline deviations"
    )
    _, run_summary = _load_json(output / "run_summary.json", "run summary")
    _validate_run_summary(run_summary)
    _validate_all_rows(episodes, contexts, daily, profiles, deviations)
    conservation = _validate_cross_stage(
        episodes, contexts, daily, profiles, deviations
    )
    expected_payloads = {
        "episode_results.jsonl": episode_bytes,
        "context_reviews.jsonl": context_bytes,
        "daily_reports.jsonl": daily_bytes,
        "baseline_profiles.jsonl": profile_bytes,
        "baseline_deviations.jsonl": deviation_bytes,
    }
    for name, payload in expected_payloads.items():
        stage_key = {
            "episode_results.jsonl": "episode",
            "context_reviews.jsonl": "context",
        }.get(name, "daily")
        source_stage = manifest["source_stages"][stage_key]
        if source_stage["artifacts"][name] != manifest["artifacts"][name]:
            raise CameraHandoffError(f"{name} source-stage descriptor mismatch")
        if source_stage["artifacts"][name]["sha256"] != hashlib.sha256(payload).hexdigest():
            raise CameraHandoffError(f"{name} source-stage sha256 mismatch")

    if manifest["handoff_id"] != run_summary["run_id"]:
        raise CameraHandoffError("manifest and run summary stable ID mismatch")
    for name in (
        "module",
        "status",
        "delivery_status",
        "validation_scope",
        "home_input_status",
        "home_annotation_status",
        "home_smoke_status",
        "algorithm_event_emitted",
        "medical_diagnosis_emitted",
        "backend_or_frontend_implemented",
        "identity",
        "source_refs",
    ):
        if manifest[name] != run_summary[name]:
            raise CameraHandoffError(f"manifest/run summary {name} mismatch")
    if run_summary["conservation"] != conservation:
        raise CameraHandoffError("run summary conservation values drifted")
    expected_counts = {
        "episode_results": len(episodes),
        "context_reviews": len(contexts),
        "daily_reports": len(daily),
        "baseline_profiles": len(profiles),
        "baseline_deviations": len(deviations),
    }
    if run_summary["output_counts"] != expected_counts:
        raise CameraHandoffError("run summary output counts drifted")

    return ValidatedCameraHandoffBundle(
        output_dir=output,
        manifest=manifest,
        episode_results=tuple(episodes),
        context_reviews=tuple(contexts),
        daily_reports=tuple(daily),
        baseline_profiles=tuple(profiles),
        baseline_deviations=tuple(deviations),
        run_summary=run_summary,
    )


def _load_stage_bundle(
    *,
    root: Path,
    name: str,
    config: Mapping[str, Any],
    override: str | Path | None,
    allow_unpinned_override: bool,
) -> _StageBundle:
    stage_name, manifest_schema, artifact_name = _STAGE_SPECS[name]
    configured_path = _resolve(root, config["bundle_path"])
    stage_root = Path(override).resolve() if override is not None else configured_path
    if not stage_root.is_dir():
        raise CameraHandoffError(f"{name} stage bundle is not a directory: {stage_root}")
    manifest_path = stage_root / "handoff_manifest.partial.json"
    manifest_bytes, manifest = _load_json(manifest_path, f"{name} stage manifest")
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected_manifest_sha = str(config["manifest_sha256"])
    if not (override is not None and allow_unpinned_override):
        if actual_manifest_sha != expected_manifest_sha:
            raise CameraHandoffError(f"{name} stage manifest sha256 mismatch")
    if (
        manifest.get("schema_version") != manifest_schema
        or manifest.get("module") != MODULE
        or manifest.get("stage") != stage_name
    ):
        raise CameraHandoffError(f"{name} stage manifest identity drifted")
    if manifest.get("status") not in _STATUS_VALUES:
        raise CameraHandoffError(f"{name} stage status is invalid")
    if manifest.get("algorithm_event_emitted") is not False:
        raise CameraHandoffError(f"{name} stage emitted an AlgorithmEvent")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifact_name not in artifacts:
        raise CameraHandoffError(f"{name} stage artifact descriptor is missing")
    for filename, descriptor in artifacts.items():
        if not isinstance(filename, str) or not filename:
            raise CameraHandoffError(f"{name} stage artifact name is invalid")
        _verify_descriptor(stage_root / filename, descriptor, f"{name} stage descriptor")
    artifact_path = stage_root / artifact_name
    artifact_bytes, artifact_rows = _load_jsonl(
        artifact_path,
        f"{name} stage {artifact_name}",
        allow_empty=name == "context",
    )
    descriptor = artifacts[artifact_name]
    if descriptor.get("record_count") != len(artifact_rows):
        raise CameraHandoffError(f"{name} stage record count mismatch")
    return _StageBundle(
        name=name,
        root=stage_root,
        manifest_path=manifest_path,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        artifact_name=artifact_name,
        artifact_bytes=artifact_bytes,
        artifact_rows=tuple(artifact_rows),
    )


def _load_daily_stage(stage: _StageBundle) -> ValidatedCameraDailyBaselineBundle:
    try:
        loaded = load_validated_camera_daily_baseline_bundle(stage.root)
    except Exception as exc:
        raise CameraHandoffError("daily stage public validation failed") from exc
    if dict(loaded.manifest) != dict(stage.manifest):
        raise CameraHandoffError("daily stage manifest changed during validation")
    return loaded


def _validate_upstream_bindings(
    stages: Mapping[str, _StageBundle],
    daily: ValidatedCameraDailyBaselineBundle,
    episodes: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
) -> None:
    episode_sha = hashlib.sha256(stages["episode"].artifact_bytes).hexdigest()
    context_sha = hashlib.sha256(stages["context"].artifact_bytes).hexdigest()
    context_manifest = stages["context"].manifest
    passthrough = context_manifest.get("episode_results_passthrough")
    if not isinstance(passthrough, Mapping) or passthrough.get("sha256") != episode_sha:
        raise CameraHandoffError("context stage episode passthrough hash mismatch")
    if passthrough.get("mutated") is not False:
        raise CameraHandoffError("context stage reports episode mutation")
    refs = daily.manifest.get("baseline_source_refs")
    if not isinstance(refs, list):
        raise CameraHandoffError("daily stage source refs are missing")
    _require_source_hash(refs, "episode_results", episode_sha)
    _require_source_hash(refs, "context_reviews", context_sha)
    episode_manifest_identity = stages["episode"].manifest.get("identity")
    if not isinstance(episode_manifest_identity, Mapping):
        raise CameraHandoffError("episode stage identity is missing")
    if episodes and any(row["identity"] != episode_manifest_identity for row in episodes):
        raise CameraHandoffError("episode row identity differs from stage manifest")
    context_manifest_identity = context_manifest.get("identity")
    if not isinstance(context_manifest_identity, Mapping):
        raise CameraHandoffError("context stage identity is missing")
    if set(context_manifest_identity) != {
        "config_id",
        "config_sha256",
        "policy_id",
        "policy_sha256",
    } or any(
        any(row["identity"].get(key) != expected for key, expected in context_manifest_identity.items())
        for row in contexts
    ):
        raise CameraHandoffError("context row identity differs from stage manifest")
    if any(
        row["identity"] != daily.manifest.get("identity")
        for rows in (daily.daily_reports, daily.baseline_profiles, daily.baseline_deviations)
        for row in rows
    ):
        raise CameraHandoffError("daily row identity differs from stage manifest")


def _validate_all_rows(
    episodes: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    deviations: Sequence[Mapping[str, Any]],
) -> None:
    if not episodes or not daily or not profiles or not deviations:
        raise CameraHandoffError("required handoff row set is empty")
    validators = (
        (episodes, _validate_episode_result, "episode"),
        (contexts, _validate_context_review, "context"),
        (daily, _validate_daily_report, "daily"),
        (profiles, _validate_baseline_profile, "profile"),
        (deviations, _validate_baseline_deviation, "deviation"),
    )
    for rows, validator, role in validators:
        record_ids: set[str] = set()
        for row in rows:
            try:
                validator(row)
            except Exception as exc:
                raise CameraHandoffError(f"{role} row schema validation failed") from exc
            record_id = str(row["record_id"])
            if record_id in record_ids:
                raise CameraHandoffError(f"duplicate {role} record_id")
            record_ids.add(record_id)
            if row.get("module") != MODULE:
                raise CameraHandoffError(f"{role} row module drifted")
            if _FORBIDDEN_ROW_FIELDS.intersection(row):
                raise CameraHandoffError(f"{role} row contains a medical/event field")
            if "start_time" in row:
                _validate_time_pair(row.get("start_time"), row.get("end_time"), role)


def _validate_cross_stage(
    episodes: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    deviations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    episodes_by_id: dict[str, Mapping[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        if episode_id in episodes_by_id:
            raise CameraHandoffError("duplicate episode_id")
        episodes_by_id[episode_id] = episode
    context_episode_ids: set[str] = set()
    for context in contexts:
        episode_id = str(context["episode_id"])
        if episode_id in context_episode_ids:
            raise CameraHandoffError("duplicate context episode_id")
        context_episode_ids.add(episode_id)
        episode = episodes_by_id.get(episode_id)
        if episode is None:
            raise CameraHandoffError("context references an unknown episode")
        expected = {
            "episode_record_id": episode["record_id"],
            "person_id": episode["person_id"],
            "session_id": episode["session_id"],
            "source_video_id": episode["source_video_id"],
            "start_time": episode["start_time"],
            "end_time": episode["end_time"],
            "start_sec": episode["start_sec"],
            "end_sec_exclusive": episode["end_sec_exclusive"],
            "episode_identity": episode["identity"],
            "episode_snapshot_sha256": hashlib.sha256(
                canonical_json_bytes(episode)
            ).hexdigest(),
        }
        if any(context[name] != value for name, value in expected.items()):
            raise CameraHandoffError("context and episode identity/snapshot mismatch")

    daily_by_key = {
        (str(row["person_id"]), str(row["local_date"]), str(row["timezone"])): row
        for row in daily
    }
    if len(daily_by_key) != len(daily):
        raise CameraHandoffError("duplicate daily person/date/timezone")
    profiles_by_key = _rows_by_day(profiles, "profile")
    deviations_by_key = _rows_by_day(deviations, "deviation")
    if set(daily_by_key) != set(profiles_by_key) or set(daily_by_key) != set(deviations_by_key):
        raise CameraHandoffError("daily/profile/deviation day identities differ")
    for key, day in daily_by_key.items():
        profile = profiles_by_key[key]
        deviation = deviations_by_key[key]
        if deviation["profile_id"] != profile["profile_id"]:
            raise CameraHandoffError("deviation profile_id does not match profile")
        for row in (profile, deviation):
            for name in ("person_id", "local_date", "timezone", "session_id", "source_video_id"):
                if row[name] != day[name]:
                    raise CameraHandoffError("daily/profile/deviation identity mismatch")
        if day["baseline_readiness"] != profile["readiness_status"]:
            raise CameraHandoffError("daily/profile readiness mismatch")
        if profile["readiness_status"] != deviation["readiness_status"]:
            raise CameraHandoffError("profile/deviation readiness mismatch")

    valid_shape = [
        row
        for row in episodes
        if row["status"] in {"ready", "uncertain"}
        and isinstance(row.get("binary"), Mapping)
        and row["binary"].get("predicted_label")
        in {"direct_or_non_wandering", "wandering_like"}
        and isinstance(row.get("four_class"), Mapping)
        and row["four_class"].get("predicted_label")
        in {"direct", "pacing", "lapping", "random"}
    ]
    expected_shape_counts = Counter(
        str(row["four_class"]["predicted_label"]) for row in valid_shape
    )
    expected_shape_durations = Counter()
    for row in valid_shape:
        shape = str(row["four_class"]["predicted_label"])
        expected_shape_durations[shape] += float(row["duration_seconds"])
        if row["binary"]["predicted_label"] == "wandering_like":
            expected_shape_counts["wandering_like"] += 1
            expected_shape_durations["wandering_like"] += float(row["duration_seconds"])
    actual_shape_counts = Counter()
    actual_shape_durations = Counter()
    for row in daily:
        actual_shape_counts.update(row["episode_counts"])
        actual_shape_durations.update(row["episode_duration_seconds"])
    for name in ("direct", "pacing", "lapping", "random", "wandering_like"):
        if actual_shape_counts[name] != expected_shape_counts[name]:
            raise CameraHandoffError("episode/daily shape count conservation failed")
        if not math.isclose(
            float(actual_shape_durations[name]),
            float(expected_shape_durations[name]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CameraHandoffError("episode/daily duration conservation failed")
    context_bucket_count = sum(sum(row["context_counts"].values()) for row in daily)
    if context_bucket_count != len(episodes):
        raise CameraHandoffError("episode/daily context bucket conservation failed")
    unavailable = sum(row["status"] == "unavailable" for row in episodes)
    errors = sum(row["status"] == "error" for row in episodes)
    if sum(row["unavailable_count"] for row in daily) != unavailable:
        raise CameraHandoffError("episode/daily unavailable conservation failed")
    if sum(row["error_count"] for row in daily) != errors:
        raise CameraHandoffError("episode/daily error conservation failed")
    return {
        "episode_result_count": len(episodes),
        "context_review_count": len(contexts),
        "daily_context_bucket_count": context_bucket_count,
        "valid_shape_count": len(valid_shape),
        "unavailable_episode_count": unavailable,
        "error_episode_count": errors,
        "wandering_like_count": expected_shape_counts["wandering_like"],
        "wandering_like_duration_seconds": round(
            expected_shape_durations["wandering_like"], 6
        ),
    }


def _rows_by_day(
    rows: Sequence[Mapping[str, Any]], role: str
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["person_id"]), str(row["local_date"]), str(row["timezone"]))
        if key in output:
            raise CameraHandoffError(f"duplicate {role} person/date/timezone")
        output[key] = row
    return output


def _validated_home_state(
    config: Mapping[str, Any], stages: Mapping[str, _StageBundle]
) -> dict[str, str]:
    state = {
        "home_input_status": str(config["home_input_status"]),
        "home_annotation_status": str(config["home_annotation_status"]),
        "home_smoke_status": str(config["home_smoke_status"]),
    }
    for stage_name in ("context", "daily"):
        manifest = stages[stage_name].manifest
        for name in ("home_input_status", "home_smoke_status"):
            if manifest.get(name) != state[name]:
                raise CameraHandoffError(f"{stage_name} stage {name} mismatch")
    context_annotation = stages["context"].manifest.get("home_annotation_status")
    if context_annotation != state["home_annotation_status"]:
        raise CameraHandoffError("context stage home_annotation_status mismatch")
    delivery = str(config["delivery_status"])
    if state["home_input_status"] == "awaiting_input":
        if (
            state["home_smoke_status"] != "not_run_input_unavailable"
            or delivery != "algorithm_ready_with_home_smoke_pending"
        ):
            raise CameraHandoffError("pending home state has an invalid delivery status")
    if delivery == "algorithm_ready_after_home_smoke" and (
        state["home_input_status"] != "available"
        or state["home_smoke_status"] not in {"ready", "uncertain"}
    ):
        raise CameraHandoffError("completed home-smoke delivery state is invalid")
    return state


def _aggregate_status(*row_groups: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(row["status"]) for rows in row_groups for row in rows]
    if "error" in statuses:
        return "error"
    if any(value != "ready" for value in statuses):
        return "uncertain"
    return "ready"


def _failure_counts(
    episodes: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    deviations: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    output: dict[str, int] = {}
    for name, rows in (
        ("episode", episodes),
        ("context", contexts),
        ("daily", daily),
        ("baseline_profile", profiles),
        ("baseline_deviation", deviations),
    ):
        counts = Counter(str(row["status"]) for row in rows)
        output[f"{name}_unavailable"] = counts["unavailable"]
        output[f"{name}_error"] = counts["error"]
    return output


def _quality_flags(
    stages: Mapping[str, _StageBundle], home: Mapping[str, str]
) -> list[str]:
    flags = {"development_only", "algorithm_evidence_not_medical_diagnosis"}
    for stage in stages.values():
        values = stage.manifest.get("quality_flags")
        if isinstance(values, list):
            flags.update(str(value) for value in values)
    if home["home_smoke_status"] == "not_run_input_unavailable":
        flags.add("home_smoke_pending")
    return sorted(flags)


def _degraded_components(
    episodes: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    daily: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    stages: Mapping[str, _StageBundle],
    home: Mapping[str, str],
) -> list[str]:
    values: set[str] = set()
    if any(row["status"] != "ready" for row in episodes):
        values.add("episode_boundary_or_shape_uncertain")
    providers = {str(row["provider"]) for row in contexts}
    if "provider-disabled" in providers:
        values.add("context_provider_disabled")
    if "deterministic-fake" in providers:
        values.add("context_provider_fake_engineering_only")
    if any(row["status"] in {"unavailable", "error"} for row in contexts):
        values.add("context_review_unavailable")
    if any("capture_clock_unavailable" in flag for row in daily for flag in row["quality_flags"]):
        values.add("capture_clock_declared_schedule")
    if any(row["readiness_status"] == "warming_up" for row in profiles):
        values.add("personal_baseline_warming_up")
    if home["home_smoke_status"] == "not_run_input_unavailable":
        values.add("home_smoke_pending")
    episode_evaluation = stages["episode"].manifest.get("evaluation")
    if isinstance(episode_evaluation, Mapping) and episode_evaluation.get(
        "known_episode_prediction_coverage_gate_satisfied"
    ) is not True:
        values.add("episode_prediction_coverage_gate_not_satisfied")
    return sorted(values)


def _stage_descriptor(stage: _StageBundle) -> dict[str, Any]:
    artifacts = stage.manifest["artifacts"]
    names = {
        "episode": ("episode_results.jsonl",),
        "context": ("context_reviews.jsonl",),
        "daily": (
            "daily_reports.jsonl",
            "baseline_profiles.jsonl",
            "baseline_deviations.jsonl",
        ),
    }[stage.name]
    return {
        "stage": str(stage.manifest["stage"]),
        "handoff_id": str(stage.manifest["handoff_id"]),
        "status": str(stage.manifest["status"]),
        "manifest_schema_version": str(stage.manifest["schema_version"]),
        "manifest_path": stage.manifest_path.resolve().as_posix(),
        "manifest_sha256": hashlib.sha256(stage.manifest_bytes).hexdigest(),
        "artifacts": {name: dict(artifacts[name]) for name in names},
    }


def _stage_source_ref(stage: _StageBundle) -> dict[str, Any]:
    return {
        "ref_type": f"{stage.name}_stage_manifest",
        "ref_id": str(stage.manifest["handoff_id"]),
        "artifact_path": stage.manifest_path.resolve().as_posix(),
        "sha256": hashlib.sha256(stage.manifest_bytes).hexdigest(),
    }


def _load_schema_refs(
    root: Path, value: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(_SCHEMA_IDS):
        raise CameraHandoffError("handoff schema config set drifted")
    refs: dict[str, dict[str, Any]] = {}
    for name, descriptor in value.items():
        _exact_fields(descriptor, {"path", "sha256", "schema_version"}, f"{name} schema config")
        path = _resolve(root, descriptor["path"])
        payload = _read_bytes(path, f"{name} schema")
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != descriptor["sha256"]:
            raise CameraHandoffError(f"{name} schema sha256 mismatch")
        try:
            schema = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CameraHandoffError(f"{name} schema is invalid JSON") from exc
        if (
            not isinstance(schema, Mapping)
            or schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
            or schema.get("$id") != _SCHEMA_IDS[name]
            or descriptor["schema_version"] != _SCHEMA_IDS[name]
        ):
            raise CameraHandoffError(f"{name} schema identity drifted")
        refs[name] = {
            "schema_version": str(descriptor["schema_version"]),
            "artifact_path": path.as_posix(),
            "sha256": actual_sha,
        }
    return refs


def _validate_manifest(value: Any) -> None:
    fields = {
        "schema_version",
        "module",
        "handoff_id",
        "created_at",
        "person_id",
        "session_id",
        "source_video_id",
        "status",
        "delivery_status",
        "validation_scope",
        "home_input_status",
        "home_annotation_status",
        "home_smoke_status",
        "algorithm_event_emitted",
        "medical_diagnosis_emitted",
        "backend_or_frontend_implemented",
        "artifacts",
        "schemas",
        "source_stages",
        "quality_flags",
        "identity",
        "source_refs",
    }
    _exact_fields(value, fields, "handoff manifest")
    if value["schema_version"] != CAMERA_HANDOFF_MANIFEST_SCHEMA_VERSION:
        raise CameraHandoffError("handoff manifest schema drifted")
    _validate_common_header(value, "handoff manifest")
    _iso_string(value["created_at"], "handoff created_at")
    if not isinstance(value["artifacts"], Mapping) or set(value["artifacts"]) != _DESCRIBED_FILES:
        raise CameraHandoffError("handoff manifest artifact set drifted")
    for name, descriptor in value["artifacts"].items():
        _validate_descriptor_syntax(descriptor)
        expected_schema = _SCHEMA_IDS.get(name)
        if descriptor["schema_version"] != expected_schema:
            raise CameraHandoffError(f"{name} artifact schema identity drifted")
    schemas = value["schemas"]
    if not isinstance(schemas, Mapping) or set(schemas) != set(_SCHEMA_IDS):
        raise CameraHandoffError("handoff manifest schema refs drifted")
    for name, ref in schemas.items():
        _exact_fields(ref, {"schema_version", "artifact_path", "sha256"}, f"{name} schema ref")
        if ref["schema_version"] != _SCHEMA_IDS[name]:
            raise CameraHandoffError(f"{name} schema ref identity drifted")
        _digest(ref["sha256"], f"{name} schema ref sha256")
    source_stages = value["source_stages"]
    if not isinstance(source_stages, Mapping) or set(source_stages) != set(_STAGE_SPECS):
        raise CameraHandoffError("handoff source stage set drifted")
    for name, stage in source_stages.items():
        _validate_stage_descriptor(name, stage)
    _validate_identity(value["identity"])
    _validate_string_list(value["quality_flags"], "manifest quality_flags")
    _validate_source_refs(value["source_refs"])


def _validate_run_summary(value: Any) -> None:
    fields = {
        "schema_version",
        "module",
        "run_id",
        "started_at",
        "finished_at",
        "person_id",
        "session_id",
        "source_video_id",
        "status",
        "delivery_status",
        "validation_scope",
        "home_input_status",
        "home_annotation_status",
        "home_smoke_status",
        "algorithm_event_emitted",
        "medical_diagnosis_emitted",
        "backend_or_frontend_implemented",
        "input_counts",
        "output_counts",
        "failure_counts",
        "degraded_components",
        "quality_flags",
        "conservation",
        "identity",
        "source_refs",
    }
    _exact_fields(value, fields, "handoff run summary")
    if value["schema_version"] != CAMERA_HANDOFF_RUN_SUMMARY_SCHEMA_VERSION:
        raise CameraHandoffError("handoff run summary schema drifted")
    _validate_common_header(value, "handoff run summary", id_name="run_id")
    started = _iso_string(value["started_at"], "run started_at")
    finished = _iso_string(value["finished_at"], "run finished_at")
    if finished < started:
        raise CameraHandoffError("run finished_at precedes started_at")
    for name in ("input_counts", "output_counts", "failure_counts"):
        counts = value[name]
        if not isinstance(counts, Mapping) or any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in counts.items()
        ):
            raise CameraHandoffError(f"run summary {name} is invalid")
    _validate_string_list(value["degraded_components"], "degraded_components")
    _validate_string_list(value["quality_flags"], "run quality_flags")
    conservation = value["conservation"]
    expected_conservation = {
        "episode_result_count",
        "context_review_count",
        "daily_context_bucket_count",
        "valid_shape_count",
        "unavailable_episode_count",
        "error_episode_count",
        "wandering_like_count",
        "wandering_like_duration_seconds",
    }
    _exact_fields(conservation, expected_conservation, "conservation")
    for name, number in conservation.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number < 0:
            raise CameraHandoffError(f"conservation {name} is invalid")
    _validate_identity(value["identity"])
    _validate_source_refs(value["source_refs"])


def _validate_common_header(
    value: Mapping[str, Any], role: str, *, id_name: str = "handoff_id"
) -> None:
    if value["module"] != MODULE:
        raise CameraHandoffError(f"{role} module drifted")
    _nonempty(value[id_name], f"{role} {id_name}")
    if value["person_id"] is not None:
        _nonempty(value["person_id"], f"{role} person_id")
    if value["session_id"] is not None or value["source_video_id"] is not None:
        raise CameraHandoffError(f"{role} aggregate session/video must be null")
    if value["status"] not in _STATUS_VALUES:
        raise CameraHandoffError(f"{role} status is invalid")
    if value["delivery_status"] not in _DELIVERY_VALUES:
        raise CameraHandoffError(f"{role} delivery status is invalid")
    _nonempty(value["validation_scope"], f"{role} validation_scope")
    if value["home_input_status"] not in {"awaiting_input", "available"}:
        raise CameraHandoffError(f"{role} home input status is invalid")
    if value["home_annotation_status"] not in {"not_provided", "unlabeled", "available"}:
        raise CameraHandoffError(f"{role} home annotation status is invalid")
    if value["home_smoke_status"] not in {
        "not_run_input_unavailable",
        "ready",
        "uncertain",
        "error",
    }:
        raise CameraHandoffError(f"{role} home smoke status is invalid")
    if (
        value["algorithm_event_emitted"] is not False
        or value["medical_diagnosis_emitted"] is not False
        or value["backend_or_frontend_implemented"] is not False
    ):
        raise CameraHandoffError(f"{role} boundary flags drifted")
    if value["home_input_status"] == "awaiting_input" and (
        value["home_smoke_status"] != "not_run_input_unavailable"
        or value["delivery_status"] != "algorithm_ready_with_home_smoke_pending"
    ):
        raise CameraHandoffError(f"{role} pending home state drifted")


def _validate_stage_descriptor(name: str, value: Any) -> None:
    fields = {
        "stage",
        "handoff_id",
        "status",
        "manifest_schema_version",
        "manifest_path",
        "manifest_sha256",
        "artifacts",
    }
    _exact_fields(value, fields, f"{name} source stage")
    expected_stage, expected_schema, _ = _STAGE_SPECS[name]
    if value["stage"] != expected_stage or value["manifest_schema_version"] != expected_schema:
        raise CameraHandoffError(f"{name} source stage identity drifted")
    _nonempty(value["handoff_id"], f"{name} source handoff_id")
    _nonempty(value["manifest_path"], f"{name} source manifest_path")
    _digest(value["manifest_sha256"], f"{name} source manifest_sha256")
    if value["status"] not in _STATUS_VALUES:
        raise CameraHandoffError(f"{name} source stage status is invalid")
    expected_artifacts = {
        "episode": {"episode_results.jsonl"},
        "context": {"context_reviews.jsonl"},
        "daily": {
            "daily_reports.jsonl",
            "baseline_profiles.jsonl",
            "baseline_deviations.jsonl",
        },
    }[name]
    if not isinstance(value["artifacts"], Mapping) or set(value["artifacts"]) != expected_artifacts:
        raise CameraHandoffError(f"{name} source artifact set drifted")
    for descriptor in value["artifacts"].values():
        _validate_descriptor_syntax(descriptor)


def _validate_config(value: Any) -> None:
    fields = {
        "schema_version",
        "handoff_id",
        "delivery_status",
        "validation_scope",
        "home_input_status",
        "home_annotation_status",
        "home_smoke_status",
        "inputs",
        "schemas",
        "identity",
        "policy",
        "output",
    }
    _exact_fields(value, fields, "camera handoff config")
    if value["schema_version"] != CAMERA_HANDOFF_CONFIG_SCHEMA_VERSION:
        raise CameraHandoffError("camera handoff config schema drifted")
    _nonempty(value["handoff_id"], "handoff_id")
    if value["delivery_status"] not in _DELIVERY_VALUES:
        raise CameraHandoffError("config delivery_status is invalid")
    _nonempty(value["validation_scope"], "validation_scope")
    if value["home_input_status"] not in {"awaiting_input", "available"}:
        raise CameraHandoffError("config home_input_status is invalid")
    if value["home_annotation_status"] not in {"not_provided", "unlabeled", "available"}:
        raise CameraHandoffError("config home_annotation_status is invalid")
    if value["home_smoke_status"] not in {
        "not_run_input_unavailable",
        "ready",
        "uncertain",
        "error",
    }:
        raise CameraHandoffError("config home_smoke_status is invalid")
    inputs = value["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != set(_STAGE_SPECS):
        raise CameraHandoffError("config input stage set drifted")
    for name, descriptor in inputs.items():
        _exact_fields(descriptor, {"bundle_path", "manifest_sha256"}, f"{name} input")
        _nonempty(descriptor["bundle_path"], f"{name} bundle_path")
        _digest(descriptor["manifest_sha256"], f"{name} manifest_sha256")
    schemas = value["schemas"]
    if not isinstance(schemas, Mapping) or set(schemas) != set(_SCHEMA_IDS):
        raise CameraHandoffError("config schema set drifted")
    for name, descriptor in schemas.items():
        _exact_fields(descriptor, {"path", "sha256", "schema_version"}, f"{name} schema")
        _nonempty(descriptor["path"], f"{name} schema path")
        _digest(descriptor["sha256"], f"{name} schema sha256")
        if descriptor["schema_version"] != _SCHEMA_IDS[name]:
            raise CameraHandoffError(f"{name} configured schema identity drifted")
    _exact_fields(value["identity"], {"config_id", "policy_id"}, "config identity")
    _nonempty(value["identity"]["config_id"], "config_id")
    _nonempty(value["identity"]["policy_id"], "policy_id")
    policy = value["policy"]
    _exact_fields(
        policy,
        {
            "passthrough_rows_without_mutation",
            "validate_cross_stage_identity",
            "preserve_uncertain_unavailable_error",
            "provider_failure_nonblocking",
            "algorithm_event_emitted",
            "medical_diagnosis_emitted",
            "backend_or_frontend_implemented",
        },
        "handoff policy",
    )
    if any(
        policy[name] is not expected
        for name, expected in {
            "passthrough_rows_without_mutation": True,
            "validate_cross_stage_identity": True,
            "preserve_uncertain_unavailable_error": True,
            "provider_failure_nonblocking": True,
            "algorithm_event_emitted": False,
            "medical_diagnosis_emitted": False,
            "backend_or_frontend_implemented": False,
        }.items()
    ):
        raise CameraHandoffError("handoff policy boundary drifted")
    output = value["output"]
    _exact_fields(
        output,
        {"required_files", "fresh_output_directory_must_not_exist", "atomic_commit"},
        "handoff output",
    )
    if output["required_files"] != list(HANDOFF_FILES):
        raise CameraHandoffError("handoff required file order/set drifted")
    if output["fresh_output_directory_must_not_exist"] is not True or output["atomic_commit"] is not True:
        raise CameraHandoffError("handoff fresh/atomic output policy drifted")


def _common_episode_identity(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    first = rows[0]["identity"]
    if any(row["identity"] != first for row in rows[1:]):
        raise CameraHandoffError("episode model identity is not common")
    return first


def _validate_identity(value: Any) -> None:
    fields = {
        "model_id",
        "model_sha256",
        "config_id",
        "config_sha256",
        "policy_id",
        "policy_sha256",
    }
    _exact_fields(value, fields, "identity")
    if value["model_id"] is not None:
        _nonempty(value["model_id"], "identity model_id")
    if value["model_sha256"] is not None:
        _digest(value["model_sha256"], "identity model_sha256")
    _nonempty(value["config_id"], "identity config_id")
    _digest(value["config_sha256"], "identity config_sha256")
    _nonempty(value["policy_id"], "identity policy_id")
    _digest(value["policy_sha256"], "identity policy_sha256")


def _validate_source_refs(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise CameraHandoffError("source_refs must be non-empty")
    seen: set[tuple[Any, ...]] = set()
    for ref in value:
        _exact_fields(ref, {"ref_type", "ref_id", "artifact_path", "sha256"}, "source ref")
        _nonempty(ref["ref_type"], "source ref type")
        _nonempty(ref["ref_id"], "source ref id")
        if ref["artifact_path"] is not None:
            _nonempty(ref["artifact_path"], "source ref path")
        if ref["sha256"] is not None:
            _digest(ref["sha256"], "source ref sha256")
        key = (ref["ref_type"], ref["ref_id"], ref["artifact_path"], ref["sha256"])
        if key in seen:
            raise CameraHandoffError("duplicate source ref")
        seen.add(key)


def _require_source_hash(
    refs: Sequence[Mapping[str, Any]], ref_type: str, sha256: str
) -> None:
    matches = [ref for ref in refs if ref.get("ref_type") == ref_type]
    if len(matches) != 1 or matches[0].get("sha256") != sha256:
        raise CameraHandoffError(f"daily stage {ref_type} source hash mismatch")


def _descriptor(
    payload: bytes, schema_version: str | None, record_count: int | None
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "record_count": record_count,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_descriptor_syntax(value: Any) -> None:
    _exact_fields(
        value,
        {"schema_version", "record_count", "byte_count", "sha256"},
        "artifact descriptor",
    )
    if value["schema_version"] is not None:
        _nonempty(value["schema_version"], "descriptor schema_version")
    if value["record_count"] is not None and (
        isinstance(value["record_count"], bool)
        or not isinstance(value["record_count"], int)
        or value["record_count"] < 0
    ):
        raise CameraHandoffError("descriptor record_count is invalid")
    if (
        isinstance(value["byte_count"], bool)
        or not isinstance(value["byte_count"], int)
        or value["byte_count"] < 0
    ):
        raise CameraHandoffError("descriptor byte_count is invalid")
    _digest(value["sha256"], "descriptor sha256")


def _verify_descriptor(path: Path, value: Any, role: str) -> None:
    _validate_descriptor_syntax(value)
    payload = _read_bytes(path, role)
    if value["byte_count"] != len(payload) or value["sha256"] != hashlib.sha256(payload).hexdigest():
        raise CameraHandoffError(f"{role} mismatch")
    if path.suffix == ".jsonl":
        count = len(payload.decode("utf-8").splitlines()) if payload else 0
        if value["record_count"] != count:
            raise CameraHandoffError(f"{role} record count mismatch")


def _load_json(path: Path, role: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read_bytes(path, role)
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CameraHandoffError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CameraHandoffError(f"{role} must be a JSON object")
    if canonical_json_bytes(value) != payload:
        raise CameraHandoffError(f"{role} is not canonical JSON")
    return payload, value


def _load_jsonl(
    path: Path, role: str, *, allow_empty: bool = False
) -> tuple[bytes, list[dict[str, Any]]]:
    payload = _read_bytes(path, role)
    if not payload and not allow_empty:
        raise CameraHandoffError(f"{role} must be non-empty")
    rows: list[dict[str, Any]] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            row = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise CameraHandoffError(f"{role} rows must be objects")
            rows.append(row)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CameraHandoffError(f"{role} is not valid JSONL") from exc
    if canonical_jsonl_bytes(rows) != payload:
        raise CameraHandoffError(f"{role} is not canonical JSONL")
    return payload, rows


def _read_bytes(path: Path, role: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CameraHandoffError(f"cannot read {role}: {path}") from exc


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _resolve(root: Path, value: Any) -> Path:
    path = Path(_nonempty(value, "path"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _exact_fields(value: Any, fields: set[str], role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise CameraHandoffError(
            f"{role} fields drifted: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )


def _nonempty(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraHandoffError(f"{role} must be a non-empty string")
    return value


def _digest(value: Any, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CameraHandoffError(f"{role} must be lowercase SHA-256")
    return value


def _validate_string_list(value: Any, role: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
        or value != sorted(value)
    ):
        raise CameraHandoffError(f"{role} must be a sorted unique string list")


def _validate_time_pair(start: Any, end: Any, role: str) -> None:
    if (start is None) != (end is None):
        raise CameraHandoffError(f"{role} start/end time availability differs")
    if start is not None and _iso_string(end, f"{role} end_time") <= _iso_string(
        start, f"{role} start_time"
    ):
        raise CameraHandoffError(f"{role} end_time does not follow start_time")


def _iso_string(value: Any, role: str) -> datetime:
    text = _nonempty(value, role)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CameraHandoffError(f"{role} is not ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CameraHandoffError(f"{role} must include a timezone")
    return parsed


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _single_or_none(values: Sequence[Any] | Any) -> str | None:
    unique = {str(value) for value in values}
    return next(iter(unique)) if len(unique) == 1 else None


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _readme(run_id: str) -> str:
    return f"""# W5D-05 camera algorithm handoff

Run ID: `{run_id}`

This directory is the backend-consumable `module=mental_health` evidence bundle.
It preserves episode shape, optional context, person-day aggregation, rolling
baseline readiness, unavailable/error states, source identities, and hashes.

## Reproduce the complete cached-tracking flow

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_handoff_e2e.py --output-root tmp/wandering_camera_5d_runs/{run_id}-replay --run-id {run_id}-replay
```

The command starts from the registered B01+B02 tracking/sidecar index, then runs
automatic episode/shape, optional three-frame context, daily/baseline, and final
handoff assembly without manual JSON edits. The output root must not exist.

## Contract

Every JSONL row carries `schema_version`, `module`, stable record identity,
status/quality, model/config/policy identity, and source references. The manifest
binds schema IDs, counts, byte sizes, SHA-256 values, source stages, and home-smoke
state. `algorithm_event_emitted=false`; no diagnosis, business risk decision,
backend, or frontend is implemented here.

## Current evidence limit

This is B01+B02 development evidence. Capture wall-clock timestamps were absent,
so the daily binding uses an explicitly declared schedule. Home input is still
awaiting input and this bundle is not `home_validated`; deterministic fake context
only verifies the adapter contract and is not context-label accuracy evidence.
"""
