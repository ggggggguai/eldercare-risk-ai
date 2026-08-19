"""W5D-04 real-development person-day and rolling baseline adapter."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from elderly_monitoring.modules.mental_health.config import load_mental_health_config
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_baseline_preview import (
    _quantile,
)
from elderly_monitoring.modules.mental_health.wandering.camera_context_review import (
    CAMERA_CONTEXT_REVIEW_SCHEMA_VERSION,
    _validate_context_review,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_summary import (
    _duration,
    _intersect,
    _measure,
    _merge_intervals,
    _night_intervals,
    _split_days,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode_pipeline import (
    _validate_episode_result,
)


CAMERA_DAILY_BASELINE_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-daily-baseline-config-v1"
)
CAMERA_DAILY_BINDING_SCHEMA_VERSION = "wandering-camera-real-development-binding-v1"
CAMERA_DAILY_REPORT_SCHEMA_VERSION = "wandering-handoff-daily-report-v2"
CAMERA_BASELINE_PROFILE_SCHEMA_VERSION = "wandering-handoff-baseline-profile-v1"
CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION = "wandering-handoff-baseline-deviation-v1"
CAMERA_DAILY_BASELINE_MANIFEST_SCHEMA_VERSION = (
    "wandering-camera-daily-baseline-handoff-manifest-partial-v1"
)
RUN_SUMMARY_SCHEMA_VERSION = "wandering-handoff-run-summary-v1"

SHAPE_ORDER = ("direct", "pacing", "lapping", "random")
TIER_ORDER = ("high_confidence", "uncertain", "unavailable", "error")
METRIC_ORDER = (
    "tracking_coverage",
    "presence_hours",
    "direct_count_per_presence_hour",
    "pacing_count_per_presence_hour",
    "lapping_count_per_presence_hour",
    "random_count_per_presence_hour",
    "wandering_like_count_per_presence_hour",
    "wandering_like_duration_seconds_per_presence_hour",
    "uncertain_wandering_like_count_per_presence_hour",
    "night_wandering_like_ratio",
)
_CONTEXT_LABELS = frozenset(
    {
        "phone_call",
        "searching",
        "cleaning",
        "exercise",
        "social",
        "other",
        "unknown",
        "not_reviewed",
        "not_triggered",
    }
)
_STATUSES = frozenset({"ready", "uncertain", "unavailable", "error"})
_READINESS = frozenset({"warming_up", "initial_ready", "stable_ready", "unavailable"})
_DIGEST_LENGTH = 64
_FINAL_FILES = frozenset(
    {
        "binding_manifest.json",
        "daily_reports.jsonl",
        "baseline_profiles.jsonl",
        "baseline_deviations.jsonl",
        "run_summary.json",
        "README.md",
        "VERIFICATION.md",
        "handoff_manifest.partial.json",
    }
)


class WanderingCameraDailyBaselineError(ValueError):
    """A W5D-04 input, binding, statistic, or artifact is invalid."""


@dataclass(frozen=True)
class DailyVideoBinding:
    source_video_id: str
    session_id: str
    person_id: str
    timezone: str
    started_at: datetime
    presence_intervals_sec: tuple[tuple[float, float], ...]
    tracking_intervals_sec: tuple[tuple[float, float], ...]
    time_basis: str
    presence_basis: str
    source_refs: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CameraDailyBaselineBuildResult:
    output_dir: Path
    daily_report_count: int
    baseline_profile_count: int
    baseline_deviation_count: int
    person_count: int
    local_date_count: int
    status: str


@dataclass(frozen=True)
class ValidatedCameraDailyBaselineBundle:
    output_dir: Path
    manifest: Mapping[str, Any]
    binding_manifest: Mapping[str, Any]
    daily_reports: tuple[Mapping[str, Any], ...]
    baseline_profiles: tuple[Mapping[str, Any], ...]
    baseline_deviations: tuple[Mapping[str, Any], ...]
    run_summary: Mapping[str, Any]


@dataclass
class _Day:
    person_id: str
    local_date: date
    timezone_name: str
    timezone: ZoneInfo
    presence: list[tuple[float, float]] = field(default_factory=list)
    tracking: list[tuple[float, float]] = field(default_factory=list)
    proposal_intervals: list[tuple[float, float]] = field(default_factory=list)
    qc_ready_intervals: list[tuple[float, float]] = field(default_factory=list)
    wandering_intervals: list[tuple[float, float]] = field(default_factory=list)
    sessions: set[str] = field(default_factory=set)
    videos: set[str] = field(default_factory=set)
    episode_counts: Counter[str] = field(default_factory=Counter)
    episode_durations: Counter[str] = field(default_factory=Counter)
    tier_counts: Counter[str] = field(default_factory=Counter)
    tier_durations: Counter[str] = field(default_factory=Counter)
    context_counts: Counter[str] = field(default_factory=Counter)
    quality_flags: set[str] = field(default_factory=set)
    source_refs: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = field(
        default_factory=dict
    )


def load_camera_daily_baseline_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise WanderingCameraDailyBaselineError(
            f"cannot load camera daily/baseline config: {config_path}"
        ) from exc
    _validate_config(value)
    return dict(value)


def aggregate_wandering_daily_reports(
    episode_rows: Sequence[Mapping[str, Any]],
    context_rows_by_episode: Mapping[str, Mapping[str, Any]],
    video_bindings: Sequence[DailyVideoBinding],
    *,
    identity: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    night_start: str,
    night_end: str,
    minimum_usable_tracking_coverage: float,
) -> list[dict[str, Any]]:
    """Aggregate explicitly bound camera evidence into real person/local-day rows."""

    _validate_identity(identity)
    common_refs = tuple(_validated_source_ref(value) for value in source_refs)
    night_start_time = _clock(night_start, "night_start")
    night_end_time = _clock(night_end, "night_end")
    if night_start_time == night_end_time:
        raise WanderingCameraDailyBaselineError("night window cannot be empty")
    minimum_coverage = _bounded_number(
        minimum_usable_tracking_coverage,
        "minimum_usable_tracking_coverage",
        minimum=0.0,
        maximum=1.0,
    )

    bindings: dict[str, DailyVideoBinding] = {}
    days: dict[tuple[str, date, str], _Day] = {}
    person_date_zones: dict[tuple[str, date], str] = {}
    for binding in video_bindings:
        _validate_binding(binding)
        if binding.source_video_id in bindings:
            raise WanderingCameraDailyBaselineError("duplicate source_video_id binding")
        bindings[binding.source_video_id] = binding
        zone = ZoneInfo(binding.timezone)
        start_epoch = binding.started_at.timestamp()
        presence_absolute = [
            (start_epoch + left, start_epoch + right)
            for left, right in _validated_intervals(
                binding.presence_intervals_sec, "presence"
            )
        ]
        tracking_absolute = [
            (start_epoch + left, start_epoch + right)
            for left, right in _validated_intervals(
                binding.tracking_intervals_sec, "tracking", allow_empty=True
            )
        ]
        tracking_absolute = _intersect(tracking_absolute, presence_absolute)
        for interval, role in (
            *((value, "presence") for value in presence_absolute),
            *((value, "tracking") for value in tracking_absolute),
        ):
            for local_day, segment_start, segment_end in _split_days(
                interval[0], interval[1], zone
            ):
                day = _day_for(
                    days,
                    person_date_zones,
                    binding.person_id,
                    local_day,
                    binding.timezone,
                    zone,
                )
                target = day.presence if role == "presence" else day.tracking
                target.append((segment_start, segment_end))
                day.sessions.add(binding.session_id)
                day.videos.add(binding.source_video_id)
                day.quality_flags.add("real_development_camera_evidence")
                if "declared" in binding.time_basis:
                    day.quality_flags.add("capture_clock_unavailable_declared_schedule")
                for ref in (*common_refs, *binding.source_refs):
                    _add_source_ref(day, ref)

    seen_episode_ids: set[str] = set()
    for source_episode in episode_rows:
        episode = dict(source_episode)
        episode_id = _token(episode.get("episode_id"), "episode_id")
        if episode_id in seen_episode_ids:
            raise WanderingCameraDailyBaselineError("duplicate episode_id")
        seen_episode_ids.add(episode_id)
        binding = bindings.get(_token(episode.get("source_video_id"), "source_video_id"))
        if binding is None:
            raise WanderingCameraDailyBaselineError("episode has no explicit video binding")
        if episode.get("person_id") != binding.person_id or episode.get("session_id") != binding.session_id:
            raise WanderingCameraDailyBaselineError(
                "episode person/session does not match explicit binding"
            )
        start_sec, end_sec = _interval(
            episode.get("start_sec"), episode.get("end_sec_exclusive"), "episode"
        )
        if not _covered_by((start_sec, end_sec), binding.presence_intervals_sec):
            raise WanderingCameraDailyBaselineError(
                "episode interval is outside explicit presence"
            )
        zone = ZoneInfo(binding.timezone)
        absolute_start = binding.started_at.timestamp() + start_sec
        absolute_end = binding.started_at.timestamp() + end_sec
        status = _status(episode.get("status"), "episode status")
        binary = episode.get("binary")
        four_class = episode.get("four_class")
        binary_label = (
            str(binary.get("predicted_label")) if isinstance(binary, Mapping) else None
        )
        shape = (
            str(four_class.get("predicted_label"))
            if isinstance(four_class, Mapping)
            else None
        )
        valid_shape = (
            status in {"ready", "uncertain"}
            and binary_label in {"direct_or_non_wandering", "wandering_like"}
            and shape in SHAPE_ORDER
        )
        wandering_like = valid_shape and binary_label == "wandering_like"
        tier = _episode_tier(episode, wandering_like=wandering_like)
        start_day = datetime.fromtimestamp(absolute_start, tz=zone).date()
        start_accumulator = _day_for(
            days,
            person_date_zones,
            binding.person_id,
            start_day,
            binding.timezone,
            zone,
        )
        start_accumulator.sessions.add(binding.session_id)
        start_accumulator.videos.add(binding.source_video_id)
        if tier is not None:
            start_accumulator.tier_counts[tier] += 1
        if valid_shape:
            start_accumulator.episode_counts[str(shape)] += 1
            if wandering_like:
                start_accumulator.episode_counts["wandering_like"] += 1
        context_label, context_flag = _context_label(
            episode,
            context_rows_by_episode.get(episode_id),
            wandering_like=wandering_like,
        )
        start_accumulator.context_counts[context_label] += 1
        if context_flag is not None:
            start_accumulator.quality_flags.add(context_flag)
        snapshot = hashlib.sha256(canonical_json_bytes(episode)).hexdigest()
        _add_source_ref(
            start_accumulator,
            {
                "ref_type": "episode_result",
                "ref_id": str(episode["record_id"]),
                "artifact_path": None,
                "sha256": snapshot,
            },
        )
        for ref in (*common_refs, *binding.source_refs):
            _add_source_ref(start_accumulator, ref)
        for local_day, segment_start, segment_end in _split_days(
            absolute_start, absolute_end, zone
        ):
            day = _day_for(
                days,
                person_date_zones,
                binding.person_id,
                local_day,
                binding.timezone,
                zone,
            )
            segment = (segment_start, segment_end)
            segment_duration = segment_end - segment_start
            day.sessions.add(binding.session_id)
            day.videos.add(binding.source_video_id)
            day.proposal_intervals.append(segment)
            if tier is not None:
                day.tier_durations[tier] += segment_duration
            if episode.get("qc_status") == "ready":
                day.qc_ready_intervals.append(segment)
            if valid_shape:
                day.episode_durations[str(shape)] += segment_duration
                if wandering_like:
                    day.episode_durations["wandering_like"] += segment_duration
                    day.wandering_intervals.append(segment)
            for ref in (*common_refs, *binding.source_refs):
                _add_source_ref(day, ref)

    rows = [
        _finalize_day(
            day,
            identity=identity,
            night_start=night_start_time,
            night_end=night_end_time,
            minimum_usable_tracking_coverage=minimum_coverage,
        )
        for day in days.values()
    ]
    rows.sort(key=lambda row: (row["local_date"], row["person_id"], row["timezone"]))
    for row in rows:
        _validate_daily_report(row)
    return rows


def build_rolling_baseline_rows(
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    initial_days: int,
    stable_days: int,
    max_window_days: int,
    upper_quantile: float,
    minimum_usable_tracking_coverage: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-observation profiles and deviations from strictly prior usable days."""

    _validate_identity(identity)
    if not 1 <= initial_days <= stable_days <= max_window_days:
        raise WanderingCameraDailyBaselineError("baseline day thresholds are invalid")
    quantile = _bounded_number(
        upper_quantile, "upper_quantile", minimum=0.5, maximum=1.0
    )
    if quantile >= 1.0:
        raise WanderingCameraDailyBaselineError("upper_quantile must be less than 1")
    minimum_coverage = _bounded_number(
        minimum_usable_tracking_coverage,
        "minimum_usable_tracking_coverage",
        minimum=0.0,
        maximum=1.0,
    )
    refs = tuple(_validated_source_ref(value) for value in source_refs)
    updated = [dict(row) for row in daily_rows]
    by_person: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in updated:
        _validate_daily_report(row)
        key = (str(row["person_id"]), str(row["timezone"]), str(row["local_date"]))
        if key in seen:
            raise WanderingCameraDailyBaselineError("duplicate person/timezone/local_date")
        seen.add(key)
        by_person[(key[0], key[1])].append(row)

    profiles: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []
    for (person_id, timezone_name), person_rows in sorted(by_person.items()):
        ordered = sorted(person_rows, key=lambda row: str(row["local_date"]))
        for current_index, current in enumerate(ordered):
            prior = [
                row
                for row in ordered[:current_index]
                if _daily_usable(row, minimum_coverage)
            ]
            selected = prior[-max_window_days:]
            readiness = _readiness(len(selected), initial_days, stable_days)
            current["baseline_readiness"] = readiness
            profile = _build_profile(
                current=current,
                selected=selected,
                person_id=person_id,
                timezone_name=timezone_name,
                readiness=readiness,
                max_window_days=max_window_days,
                upper_quantile=quantile,
                identity=identity,
                source_refs=refs,
            )
            deviation = _build_deviation(
                current=current,
                profile=profile,
                identity=identity,
                source_refs=refs,
                minimum_usable_tracking_coverage=minimum_coverage,
            )
            profiles.append(profile)
            deviations.append(deviation)

    updated.sort(key=lambda row: (row["local_date"], row["person_id"], row["timezone"]))
    profiles.sort(key=lambda row: (row["local_date"], row["person_id"], row["timezone"]))
    deviations.sort(key=lambda row: (row["local_date"], row["person_id"], row["timezone"]))
    for row in updated:
        _validate_daily_report(row)
    for row in profiles:
        _validate_baseline_profile(row)
    for row in deviations:
        _validate_baseline_deviation(row)
    return profiles, deviations, updated


def build_camera_daily_baseline_bundle(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    replay_days: int | None = None,
    episode_results_path: str | Path | None = None,
    context_reviews_path: str | Path | None = None,
) -> CameraDailyBaselineBuildResult:
    """Build a fresh W5D-04 real-development or deterministic replay bundle."""

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"camera daily/baseline output already exists: {output}")
    config_bytes = _read_bytes(config_file, "daily/baseline config")
    config = load_camera_daily_baseline_config(config_file)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    schema_refs = _validate_schema_files(root, config["schemas"])
    inputs = config["inputs"]
    index_path = _resolve(root, inputs["development_index"])
    episode_path = _resolve(
        root,
        str(episode_results_path)
        if episode_results_path is not None
        else inputs["episode_results"],
    )
    context_path = _resolve(
        root,
        str(context_reviews_path)
        if context_reviews_path is not None
        else inputs["context_reviews"],
    )
    mental_config_path = _resolve(root, inputs["mental_health_config"])
    index_bytes, index_rows = _load_jsonl(index_path, "development index")
    episode_bytes, episode_rows = _load_jsonl(episode_path, "episode results")
    context_bytes, context_rows = _load_jsonl(
        context_path, "context reviews", allow_empty=True
    )
    for row in episode_rows:
        try:
            _validate_episode_result(row)
        except Exception as exc:
            raise WanderingCameraDailyBaselineError(
                "episode result failed W5D-02 validation"
            ) from exc
    for row in context_rows:
        try:
            _validate_context_review(row)
        except Exception as exc:
            raise WanderingCameraDailyBaselineError(
                "context review failed W5D-03A validation"
            ) from exc
    common_episode_identity = _common_episode_identity(episode_rows)
    policy_payload = {
        "aggregation": config["aggregation"],
        "baseline": _baseline_config_payload(mental_config_path),
        "session_bindings": config["session_bindings"],
    }
    identity = {
        "model_id": common_episode_identity["model_id"],
        "model_sha256": common_episode_identity["model_sha256"],
        "config_id": str(config["identity"]["config_id"]),
        "config_sha256": config_sha256,
        "policy_id": str(config["identity"]["policy_id"]),
        "policy_sha256": hashlib.sha256(canonical_json_bytes(policy_payload)).hexdigest(),
    }
    mental_config = load_mental_health_config(mental_config_path)
    bindings, binding_manifest = _build_bindings(
        root=root,
        config=config,
        config_sha256=config_sha256,
        index_path=index_path,
        index_bytes=index_bytes,
        index_rows=index_rows,
    )
    binding_bytes = canonical_json_bytes(binding_manifest)
    binding_ref = {
        "ref_type": "daily_binding_manifest",
        "ref_id": str(config["daily_baseline_id"]),
        "artifact_path": (output / "binding_manifest.json").resolve().as_posix(),
        "sha256": hashlib.sha256(binding_bytes).hexdigest(),
    }
    input_refs = (
        binding_ref,
        _file_ref("development_index", "w5d00-development-index", index_path, index_bytes),
        _file_ref("episode_results", "w5d02-episode-results", episode_path, episode_bytes),
        _file_ref("context_reviews", "w5d03a-context-reviews", context_path, context_bytes),
    )
    contexts_by_episode: dict[str, Mapping[str, Any]] = {}
    for row in context_rows:
        episode_id = _token(row.get("episode_id"), "context episode_id")
        if episode_id in contexts_by_episode:
            raise WanderingCameraDailyBaselineError("duplicate context episode_id")
        contexts_by_episode[episode_id] = row
    daily_rows = aggregate_wandering_daily_reports(
        episode_rows,
        contexts_by_episode,
        bindings,
        identity=identity,
        source_refs=input_refs,
        night_start=mental_config.aggregation.night_start,
        night_end=mental_config.aggregation.night_end,
        minimum_usable_tracking_coverage=float(
            config["aggregation"]["minimum_usable_tracking_coverage"]
        ),
    )
    validation_scope = str(config["validation_scope"])
    if replay_days is not None:
        if isinstance(replay_days, bool) or not isinstance(replay_days, int) or replay_days < 1:
            raise WanderingCameraDailyBaselineError("replay_days must be a positive integer")
        daily_rows = _build_replay_rows(daily_rows, config["replay"], replay_days)
        validation_scope = "deterministic_replay"

    baseline_config = mental_config.baseline
    baseline_source_refs = tuple(input_refs)
    profiles, deviations, daily_rows = build_rolling_baseline_rows(
        daily_rows,
        identity=identity,
        source_refs=baseline_source_refs,
        initial_days=baseline_config.initial_days,
        stable_days=baseline_config.stable_days,
        max_window_days=baseline_config.max_window_days,
        upper_quantile=baseline_config.upper_quantile,
        minimum_usable_tracking_coverage=float(
            config["aggregation"]["minimum_usable_tracking_coverage"]
        ),
    )
    daily_bytes = canonical_jsonl_bytes(daily_rows)
    profile_bytes = canonical_jsonl_bytes(profiles)
    deviation_bytes = canonical_jsonl_bytes(deviations)
    started_at = datetime.now(timezone.utc)
    effective_run_id = run_id or (
        f"w5d04-{validation_scope}-{started_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    _token(effective_run_id, "run_id")
    status = "uncertain" if any(row["status"] != "ready" for row in daily_rows) else "ready"
    quality_flags = {
        "development_only",
        "daily_baseline_only_no_medical_diagnosis",
        "home_smoke_pending",
    }
    if validation_scope == "deterministic_replay":
        quality_flags.add("deterministic_replay_not_real_longitudinal_observation")
    else:
        quality_flags.add("capture_clock_unavailable_declared_schedule")
    run_summary = {
        "schema_version": RUN_SUMMARY_SCHEMA_VERSION,
        "module": "mental_health",
        "run_id": effective_run_id,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "person_id": next(iter({str(row["person_id"]) for row in daily_rows}), None)
        if len({str(row["person_id"]) for row in daily_rows}) == 1
        else None,
        "session_id": None,
        "source_video_id": None,
        "status": status,
        "input_counts": {
            "development_index_rows": len(index_rows),
            "episode_results": len(episode_rows),
            "context_reviews": len(context_rows),
            "video_bindings": len(bindings),
        },
        "output_counts": {
            "daily_reports": len(daily_rows),
            "baseline_profiles": len(profiles),
            "baseline_deviations": len(deviations),
            "warming_up": sum(row["readiness_status"] == "warming_up" for row in profiles),
            "initial_ready": sum(row["readiness_status"] == "initial_ready" for row in profiles),
            "stable_ready": sum(row["readiness_status"] == "stable_ready" for row in profiles),
        },
        "failure_counts": {},
        "degraded_components": (
            ["capture_clock_declared_schedule"]
            if validation_scope != "deterministic_replay"
            else ["deterministic_replay"]
        ),
        "quality_flags": sorted(quality_flags),
        "identity": dict(identity),
        "source_refs": [dict(ref) for ref in input_refs],
    }
    run_summary_bytes = canonical_json_bytes(run_summary)
    readme_bytes = _readme_text(
        validation_scope=validation_scope,
        daily_rows=daily_rows,
        profiles=profiles,
        home_input_status=str(config["home_input_status"]),
        home_smoke_status=str(config["home_smoke_status"]),
    ).encode("utf-8")
    verification_bytes = _verification_text(
        bindings=bindings,
        daily_rows=daily_rows,
        profiles=profiles,
        deviations=deviations,
        validation_scope=validation_scope,
    ).encode("utf-8")
    artifacts = {
        "binding_manifest.json": _descriptor(
            binding_bytes, CAMERA_DAILY_BINDING_SCHEMA_VERSION, 1
        ),
        "daily_reports.jsonl": _descriptor(
            daily_bytes, CAMERA_DAILY_REPORT_SCHEMA_VERSION, len(daily_rows)
        ),
        "baseline_profiles.jsonl": _descriptor(
            profile_bytes, CAMERA_BASELINE_PROFILE_SCHEMA_VERSION, len(profiles)
        ),
        "baseline_deviations.jsonl": _descriptor(
            deviation_bytes, CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION, len(deviations)
        ),
        "run_summary.json": _descriptor(
            run_summary_bytes, RUN_SUMMARY_SCHEMA_VERSION, 1
        ),
        "README.md": _descriptor(readme_bytes, None, None),
        "VERIFICATION.md": _descriptor(verification_bytes, None, None),
    }
    effective_baseline = {
        "initial_days": baseline_config.initial_days,
        "stable_days": baseline_config.stable_days,
        "max_window_days": baseline_config.max_window_days,
        "upper_quantile": baseline_config.upper_quantile,
        "minimum_usable_tracking_coverage": float(
            config["aggregation"]["minimum_usable_tracking_coverage"]
        ),
    }
    manifest = {
        "schema_version": CAMERA_DAILY_BASELINE_MANIFEST_SCHEMA_VERSION,
        "module": "mental_health",
        "handoff_id": effective_run_id,
        "stage": "W5D-04",
        "status": status,
        "validation_scope": validation_scope,
        "home_input_status": str(config["home_input_status"]),
        "home_smoke_status": str(config["home_smoke_status"]),
        "algorithm_event_emitted": False,
        "medical_diagnosis_emitted": False,
        "public_loader_recomputed": True,
        "artifacts": artifacts,
        "identity": dict(identity),
        "effective_baseline": effective_baseline,
        "schemas": schema_refs,
        "baseline_source_refs": [dict(ref) for ref in baseline_source_refs],
        "quality_flags": sorted(quality_flags),
    }
    manifest_bytes = canonical_json_bytes(manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    payloads = {
        "binding_manifest.json": binding_bytes,
        "daily_reports.jsonl": daily_bytes,
        "baseline_profiles.jsonl": profile_bytes,
        "baseline_deviations.jsonl": deviation_bytes,
        "run_summary.json": run_summary_bytes,
        "README.md": readme_bytes,
        "VERIFICATION.md": verification_bytes,
        "handoff_manifest.partial.json": manifest_bytes,
    }
    try:
        for name, payload in payloads.items():
            _write_new_file(staging / name, payload)
        loaded = load_validated_camera_daily_baseline_bundle(staging)
        if (
            list(loaded.daily_reports) != daily_rows
            or list(loaded.baseline_profiles) != profiles
            or list(loaded.baseline_deviations) != deviations
        ):
            raise WanderingCameraDailyBaselineError(
                "public loader readback does not match producer rows"
            )
        if output.exists():
            raise FileExistsError(
                f"camera daily/baseline output already exists: {output}"
            )
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return CameraDailyBaselineBuildResult(
        output_dir=output,
        daily_report_count=len(daily_rows),
        baseline_profile_count=len(profiles),
        baseline_deviation_count=len(deviations),
        person_count=len({str(row["person_id"]) for row in daily_rows}),
        local_date_count=len({str(row["local_date"]) for row in daily_rows}),
        status=status,
    )


def load_validated_camera_daily_baseline_bundle(
    output_dir: str | Path,
) -> ValidatedCameraDailyBaselineBundle:
    """Load, hash-check, and independently recompute a W5D-04 bundle."""

    output = Path(output_dir)
    if not output.is_dir():
        raise WanderingCameraDailyBaselineError("daily/baseline bundle is not a directory")
    entries = {entry.name: entry for entry in output.iterdir()}
    if set(entries) != _FINAL_FILES or any(not entry.is_file() for entry in entries.values()):
        raise WanderingCameraDailyBaselineError("daily/baseline bundle file set drifted")
    manifest = _load_json(
        output / "handoff_manifest.partial.json", "daily/baseline manifest"
    )
    if manifest.get("schema_version") != CAMERA_DAILY_BASELINE_MANIFEST_SCHEMA_VERSION:
        raise WanderingCameraDailyBaselineError("daily/baseline manifest schema drifted")
    if manifest.get("public_loader_recomputed") is not True:
        raise WanderingCameraDailyBaselineError("public loader recomputation flag drifted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _FINAL_FILES - {
        "handoff_manifest.partial.json"
    }:
        raise WanderingCameraDailyBaselineError("manifest artifact set drifted")
    for name, descriptor in artifacts.items():
        _verify_descriptor(output / name, descriptor)
    binding = _load_json(output / "binding_manifest.json", "binding manifest")
    daily_bytes, daily = _load_jsonl(output / "daily_reports.jsonl", "daily reports")
    profile_bytes, profiles = _load_jsonl(
        output / "baseline_profiles.jsonl", "baseline profiles"
    )
    deviation_bytes, deviations = _load_jsonl(
        output / "baseline_deviations.jsonl", "baseline deviations"
    )
    run_summary = _load_json(output / "run_summary.json", "run summary")
    if binding.get("schema_version") != CAMERA_DAILY_BINDING_SCHEMA_VERSION:
        raise WanderingCameraDailyBaselineError("binding manifest schema drifted")
    _validate_binding_manifest(binding)
    if run_summary.get("schema_version") != RUN_SUMMARY_SCHEMA_VERSION:
        raise WanderingCameraDailyBaselineError("run summary schema drifted")
    for row in daily:
        _validate_daily_report(row)
    for row in profiles:
        _validate_baseline_profile(row)
    for row in deviations:
        _validate_baseline_deviation(row)
    effective = manifest.get("effective_baseline")
    if not isinstance(effective, Mapping):
        raise WanderingCameraDailyBaselineError("effective_baseline is missing")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise WanderingCameraDailyBaselineError("manifest identity is missing")
    source_refs = manifest.get("baseline_source_refs")
    if not isinstance(source_refs, list) or not source_refs:
        raise WanderingCameraDailyBaselineError("baseline_source_refs are missing")
    schemas = manifest.get("schemas")
    if not isinstance(schemas, Mapping):
        raise WanderingCameraDailyBaselineError("manifest schemas are missing")
    _verify_schema_refs(schemas)
    recomputed_profiles, recomputed_deviations, recomputed_daily = (
        build_rolling_baseline_rows(
            daily,
            identity=identity,
            source_refs=source_refs,
            initial_days=int(effective["initial_days"]),
            stable_days=int(effective["stable_days"]),
            max_window_days=int(effective["max_window_days"]),
            upper_quantile=float(effective["upper_quantile"]),
            minimum_usable_tracking_coverage=float(
                effective["minimum_usable_tracking_coverage"]
            ),
        )
    )
    if (
        recomputed_daily != daily
        or recomputed_profiles != profiles
        or recomputed_deviations != deviations
    ):
        raise WanderingCameraDailyBaselineError(
            "public loader baseline recomputation mismatch"
        )
    if canonical_jsonl_bytes(daily) != daily_bytes:
        raise WanderingCameraDailyBaselineError("daily report canonical bytes drifted")
    if canonical_jsonl_bytes(profiles) != profile_bytes:
        raise WanderingCameraDailyBaselineError("baseline profile canonical bytes drifted")
    if canonical_jsonl_bytes(deviations) != deviation_bytes:
        raise WanderingCameraDailyBaselineError("baseline deviation canonical bytes drifted")
    return ValidatedCameraDailyBaselineBundle(
        output_dir=output,
        manifest=manifest,
        binding_manifest=binding,
        daily_reports=tuple(daily),
        baseline_profiles=tuple(profiles),
        baseline_deviations=tuple(deviations),
        run_summary=run_summary,
    )


def _day_for(
    days: dict[tuple[str, date, str], _Day],
    person_date_zones: dict[tuple[str, date], str],
    person_id: str,
    local_day: date,
    timezone_name: str,
    zone: ZoneInfo,
) -> _Day:
    person_day = (person_id, local_day)
    observed_zone = person_date_zones.get(person_day)
    if observed_zone is not None and observed_zone != timezone_name:
        raise WanderingCameraDailyBaselineError(
            "one person/local_date cannot mix timezones"
        )
    person_date_zones[person_day] = timezone_name
    key = (person_id, local_day, timezone_name)
    if key not in days:
        days[key] = _Day(person_id, local_day, timezone_name, zone)
    return days[key]


def _finalize_day(
    day: _Day,
    *,
    identity: Mapping[str, Any],
    night_start: time,
    night_end: time,
    minimum_usable_tracking_coverage: float,
) -> dict[str, Any]:
    presence = _merge_intervals(day.presence)
    presence_seconds = _duration(presence)
    tracking = _intersect(_merge_intervals(day.tracking), presence)
    tracking_seconds = _duration(tracking)
    tracking_coverage = (
        _ratio(tracking_seconds, presence_seconds) if presence_seconds > 0.0 else None
    )
    proposal = _intersect(_merge_intervals(day.proposal_intervals), presence)
    qc_ready = _intersect(_merge_intervals(day.qc_ready_intervals), proposal)
    qc_coverage = _ratio(_duration(qc_ready), _duration(proposal)) if proposal else None
    wandering = _intersect(_merge_intervals(day.wandering_intervals), presence)
    wandering_seconds = _duration(wandering)
    night = _night_intervals(
        day.local_date, day.timezone, night_start, night_end
    )
    night_ratio = (
        _ratio(_duration(_intersect(wandering, night)), wandering_seconds)
        if wandering_seconds > 0.0
        else None
    )
    episode_counts = {
        name: int(day.episode_counts[name]) for name in (*SHAPE_ORDER, "wandering_like")
    }
    episode_durations = {
        name: _measure(day.episode_durations[name])
        for name in (*SHAPE_ORDER, "wandering_like")
    }
    tiers = {
        name: {
            "count": int(day.tier_counts[name]),
            "duration_seconds": _measure(day.tier_durations[name]),
        }
        for name in TIER_ORDER
    }
    presence_hours = presence_seconds / 3600.0 if presence_seconds > 0.0 else None
    rates = {
        "wandering_like_count": (
            _measure(episode_counts["wandering_like"] / presence_hours)
            if presence_hours is not None
            else None
        ),
        "wandering_like_duration_seconds": (
            _measure(episode_durations["wandering_like"] / presence_hours)
            if presence_hours is not None
            else None
        ),
    }
    unavailable_count = tiers["unavailable"]["count"]
    error_count = tiers["error"]["count"]
    quality_flags = set(day.quality_flags)
    if presence_seconds <= 0.0:
        quality_flags.add("presence_unavailable")
        status = "unavailable"
        measured_presence: float | None = None
        measured_tracking: float | None = None
    elif tracking_coverage is None or tracking_coverage < minimum_usable_tracking_coverage:
        quality_flags.add("tracking_coverage_insufficient")
        status = "unavailable"
        measured_presence = _measure(presence_seconds)
        measured_tracking = _measure(tracking_seconds)
    else:
        measured_presence = _measure(presence_seconds)
        measured_tracking = _measure(tracking_seconds)
        if (
            tiers["uncertain"]["count"] > 0
            or unavailable_count > 0
            or error_count > 0
            or "capture_clock_unavailable_declared_schedule" in quality_flags
        ):
            status = "uncertain"
        else:
            status = "ready"
    if unavailable_count:
        quality_flags.add("episode_unavailable_retained")
    if error_count:
        quality_flags.add("episode_error_retained")
    if tiers["uncertain"]["count"]:
        quality_flags.add("uncertain_candidate_retained")
    source_refs = sorted(
        day.source_refs.values(),
        key=lambda ref: (
            str(ref["ref_type"]),
            str(ref["ref_id"]),
            str(ref["artifact_path"]),
            str(ref["sha256"]),
        ),
    )
    row_payload = {
        "person_id": day.person_id,
        "local_date": day.local_date.isoformat(),
        "timezone": day.timezone_name,
        "sessions": sorted(day.sessions),
        "videos": sorted(day.videos),
        "identity": dict(identity),
    }
    record_id = "w5d04-daily-" + hashlib.sha256(
        canonical_json_bytes(row_payload)
    ).hexdigest()
    return {
        "schema_version": CAMERA_DAILY_REPORT_SCHEMA_VERSION,
        "module": "mental_health",
        "record_id": record_id,
        "person_id": day.person_id,
        "session_id": next(iter(day.sessions)) if len(day.sessions) == 1 else None,
        "source_video_id": next(iter(day.videos)) if len(day.videos) == 1 else None,
        "local_date": day.local_date.isoformat(),
        "timezone": day.timezone_name,
        "status": status,
        "presence_seconds": measured_presence,
        "tracking_coverage_seconds": measured_tracking,
        "tracking_coverage": tracking_coverage,
        "qc_coverage": qc_coverage,
        "episode_counts": episode_counts,
        "episode_duration_seconds": episode_durations,
        "confidence_tiers": tiers,
        "rates_per_presence_hour": rates,
        "night_wandering_like_ratio": night_ratio,
        "context_counts": {
            name: int(day.context_counts[name]) for name in sorted(day.context_counts)
        },
        "unavailable_count": int(unavailable_count),
        "error_count": int(error_count),
        "baseline_readiness": "warming_up",
        "quality_flags": sorted(quality_flags),
        "identity": dict(identity),
        "source_refs": source_refs,
    }


def _build_profile(
    *,
    current: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    person_id: str,
    timezone_name: str,
    readiness: str,
    max_window_days: int,
    upper_quantile: float,
    identity: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values_by_metric: dict[str, list[float]] = {name: [] for name in METRIC_ORDER}
    for row in selected:
        values = _daily_metric_values(row)
        for name, value in values.items():
            if value is not None:
                values_by_metric[name].append(value)
    metrics = {
        name: _metric_stats(values_by_metric[name], upper_quantile)
        for name in METRIC_ORDER
    }
    profile_payload = {
        "person_id": person_id,
        "timezone": timezone_name,
        "local_date": current["local_date"],
        "reference_dates": [row["local_date"] for row in selected],
        "identity": dict(identity),
    }
    profile_id = "w5d04-profile-" + hashlib.sha256(
        canonical_json_bytes(profile_payload)
    ).hexdigest()
    flags: set[str] = set()
    if not selected:
        flags.add("baseline_reference_empty")
    if readiness == "warming_up":
        flags.add("baseline_warming_up")
    if "deterministic_replay_not_real_longitudinal_observation" in current.get(
        "quality_flags", []
    ):
        flags.add("deterministic_replay_not_real_longitudinal_observation")
    return {
        "schema_version": CAMERA_BASELINE_PROFILE_SCHEMA_VERSION,
        "module": "mental_health",
        "record_id": profile_id,
        "profile_id": profile_id,
        "person_id": person_id,
        "session_id": current.get("session_id"),
        "source_video_id": current.get("source_video_id"),
        "local_date": current["local_date"],
        "timezone": timezone_name,
        "status": "uncertain" if readiness == "warming_up" else "ready",
        "readiness_status": readiness,
        "usable_day_count": len(selected),
        "reference_window_days": max_window_days,
        "reference_first_local_date": selected[0]["local_date"] if selected else None,
        "reference_last_local_date": selected[-1]["local_date"] if selected else None,
        "metrics": metrics,
        "quality_flags": sorted(flags),
        "identity": dict(identity),
        "source_refs": [dict(ref) for ref in source_refs],
    }


def _build_deviation(
    *,
    current: Mapping[str, Any],
    profile: Mapping[str, Any],
    identity: Mapping[str, Any],
    source_refs: Sequence[Mapping[str, Any]],
    minimum_usable_tracking_coverage: float,
) -> dict[str, Any]:
    observation_usable = _daily_usable(current, minimum_usable_tracking_coverage)
    values = _daily_metric_values(current)
    readiness = str(profile["readiness_status"])
    metrics: dict[str, dict[str, Any]] = {}
    for name in METRIC_ORDER:
        observation = values[name]
        reference = profile["metrics"][name]
        reference_count = int(reference["count"])
        median = reference["median"]
        p90 = reference["p90"]
        if not observation_usable or observation is None:
            metric_status = "observation_unavailable"
        elif readiness == "warming_up":
            metric_status = "warming_up"
        elif reference_count == 0 or median is None or p90 is None:
            metric_status = "reference_unavailable"
        else:
            metric_status = "ready"
        ready = metric_status == "ready"
        metrics[name] = {
            "status": metric_status,
            "observation_value": observation,
            "reference_count": reference_count,
            "reference_median": median,
            "reference_p90": p90,
            "delta_from_median": (
                _measure(float(observation) - float(median)) if ready else None
            ),
            "delta_from_p90": (
                _measure(float(observation) - float(p90)) if ready else None
            ),
        }
    deviation_payload = {
        "profile_id": profile["profile_id"],
        "daily_record_id": current["record_id"],
        "identity": dict(identity),
    }
    deviation_id = "w5d04-deviation-" + hashlib.sha256(
        canonical_json_bytes(deviation_payload)
    ).hexdigest()
    flags: set[str] = set()
    if not observation_usable:
        flags.add("observation_unavailable")
        status = "unavailable"
    elif readiness == "warming_up":
        flags.add("baseline_warming_up")
        status = "uncertain"
    else:
        status = "ready"
    if "deterministic_replay_not_real_longitudinal_observation" in current.get(
        "quality_flags", []
    ):
        flags.add("deterministic_replay_not_real_longitudinal_observation")
    return {
        "schema_version": CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION,
        "module": "mental_health",
        "record_id": deviation_id,
        "deviation_id": deviation_id,
        "profile_id": profile["profile_id"],
        "person_id": current["person_id"],
        "session_id": current.get("session_id"),
        "source_video_id": current.get("source_video_id"),
        "local_date": current["local_date"],
        "timezone": current["timezone"],
        "status": status,
        "readiness_status": readiness,
        "metrics": metrics,
        "quality_flags": sorted(flags),
        "identity": dict(identity),
        "source_refs": [dict(ref) for ref in source_refs],
    }


def _daily_metric_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    presence = row.get("presence_seconds")
    coverage = row.get("tracking_coverage")
    if presence is None or coverage is None or float(presence) <= 0.0:
        return {name: None for name in METRIC_ORDER}
    presence_hours = float(presence) / 3600.0
    counts = row["episode_counts"]
    durations = row["episode_duration_seconds"]
    tiers = row["confidence_tiers"]
    return {
        "tracking_coverage": float(coverage),
        "presence_hours": _measure(presence_hours),
        "direct_count_per_presence_hour": _measure(
            float(counts["direct"]) / presence_hours
        ),
        "pacing_count_per_presence_hour": _measure(
            float(counts["pacing"]) / presence_hours
        ),
        "lapping_count_per_presence_hour": _measure(
            float(counts["lapping"]) / presence_hours
        ),
        "random_count_per_presence_hour": _measure(
            float(counts["random"]) / presence_hours
        ),
        "wandering_like_count_per_presence_hour": _measure(
            float(counts["wandering_like"]) / presence_hours
        ),
        "wandering_like_duration_seconds_per_presence_hour": _measure(
            float(durations["wandering_like"]) / presence_hours
        ),
        "uncertain_wandering_like_count_per_presence_hour": _measure(
            float(tiers["uncertain"]["count"]) / presence_hours
        ),
        "night_wandering_like_ratio": (
            None
            if row["night_wandering_like_ratio"] is None
            else float(row["night_wandering_like_ratio"])
        ),
    }


def _metric_stats(values: Sequence[float], upper_quantile: float) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "median": _measure(_quantile(ordered, 0.5)),
        "p90": _measure(_quantile(ordered, upper_quantile)),
    }


def _readiness(count: int, initial_days: int, stable_days: int) -> str:
    if count < initial_days:
        return "warming_up"
    if count < stable_days:
        return "initial_ready"
    return "stable_ready"


def _daily_usable(row: Mapping[str, Any], minimum_coverage: float) -> bool:
    presence = row.get("presence_seconds")
    coverage = row.get("tracking_coverage")
    return (
        row.get("status") in {"ready", "uncertain"}
        and _finite_number(presence)
        and float(presence) > 0.0
        and _finite_number(coverage)
        and float(coverage) >= minimum_coverage
    )


def _build_bindings(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    index_path: Path,
    index_bytes: bytes,
    index_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[DailyVideoBinding, ...], dict[str, Any]]:
    index_by_video: dict[str, Mapping[str, Any]] = {}
    for row in index_rows:
        _validate_index_row(row)
        video_id = str(row["source_video_id"])
        if video_id in index_by_video:
            raise WanderingCameraDailyBaselineError("duplicate development source_video_id")
        index_by_video[video_id] = row
    bindings: list[DailyVideoBinding] = []
    manifest_rows: list[dict[str, Any]] = []
    bound_videos: set[str] = set()
    max_gap = float(config["aggregation"]["tracking_max_gap_seconds"])
    for session_config in config["session_bindings"]:
        session_id = str(session_config["session_id"])
        person_id = str(session_config["person_id"])
        timezone_name = str(session_config["timezone"])
        zone = ZoneInfo(timezone_name)
        session_start = _aware_datetime(
            session_config["session_started_at"],
            timezone_name,
            "session_started_at",
        )
        clip_gap = _nonnegative_number(
            session_config["clip_gap_seconds"], "clip_gap_seconds"
        )
        clip_order = session_config["clip_order"]
        cursor = 0.0
        for video_id in clip_order:
            row = index_by_video.get(str(video_id))
            if row is None:
                raise WanderingCameraDailyBaselineError(
                    "session clip_order references an unknown development video"
                )
            if str(video_id) in bound_videos:
                raise WanderingCameraDailyBaselineError("video bound more than once")
            if (
                row["session_id"] != session_id
                or row["participant_id"] != person_id
                or row["timezone"] != timezone_name
                or row["batch_id"] != session_config["batch_id"]
            ):
                raise WanderingCameraDailyBaselineError(
                    "session binding does not match development index identity"
                )
            sidecar_descriptor = row["artifacts"]["media_sidecar"]
            tracking_descriptor = row["artifacts"]["tracking"]
            sidecar_path = _artifact_path(root, sidecar_descriptor, "media_sidecar")
            tracking_path = _artifact_path(root, tracking_descriptor, "tracking")
            _verify_external_descriptor(sidecar_path, sidecar_descriptor, "media_sidecar")
            _verify_external_descriptor(tracking_path, tracking_descriptor, "tracking")
            sidecar = _load_json(sidecar_path, "media sidecar")
            duration = _positive_number(sidecar.get("duration_sec"), "duration_sec")
            nominal_fps = _positive_number(
                sidecar.get("nominal_fps"), "nominal_fps"
            )
            if sidecar.get("capture_started_at") is not None:
                raise WanderingCameraDailyBaselineError(
                    "production schedule expects capture_started_at to be unavailable"
                )
            tracking_bytes, tracking_rows = _load_jsonl(
                tracking_path,
                "tracking input",
                allow_empty=True,
                require_canonical=False,
            )
            timestamps = [
                _nonnegative_number(value.get("timestamp_sec"), "tracking timestamp")
                for value in tracking_rows
            ]
            tracking_intervals = _tracking_intervals(
                timestamps,
                duration=duration,
                nominal_fps=nominal_fps,
                maximum_gap=max_gap,
            )
            clip_start = session_start + timedelta(seconds=cursor)
            binding_refs = (
                _file_ref(
                    "tracking",
                    str(video_id),
                    tracking_path,
                    tracking_bytes,
                ),
                _file_ref(
                    "media_sidecar",
                    str(video_id),
                    sidecar_path,
                    _read_bytes(sidecar_path, "media sidecar"),
                ),
            )
            binding = DailyVideoBinding(
                source_video_id=str(video_id),
                session_id=session_id,
                person_id=person_id,
                timezone=timezone_name,
                started_at=clip_start,
                presence_intervals_sec=((0.0, duration),),
                tracking_intervals_sec=tuple(tracking_intervals),
                time_basis=str(session_config["time_basis"]),
                presence_basis=str(session_config["presence_basis"]),
                source_refs=tuple(binding_refs),
            )
            _validate_binding(binding)
            bindings.append(binding)
            manifest_rows.append(
                {
                    "source_video_id": str(video_id),
                    "session_id": session_id,
                    "person_id": person_id,
                    "timezone": timezone_name,
                    "started_at": clip_start.isoformat(),
                    "end_at": (clip_start + timedelta(seconds=duration)).isoformat(),
                    "time_basis": str(session_config["time_basis"]),
                    "presence_basis": str(session_config["presence_basis"]),
                    "presence_intervals_sec": [
                        {"start_sec": 0.0, "end_sec_exclusive": _measure(duration)}
                    ],
                    "tracking_intervals_sec": [
                        {
                            "start_sec": _measure(left),
                            "end_sec_exclusive": _measure(right),
                        }
                        for left, right in tracking_intervals
                    ],
                    "source_refs": [dict(ref) for ref in binding_refs],
                }
            )
            cursor += duration + clip_gap
            bound_videos.add(str(video_id))
    if bound_videos != set(index_by_video):
        missing = sorted(set(index_by_video) - bound_videos)
        extra = sorted(bound_videos - set(index_by_video))
        raise WanderingCameraDailyBaselineError(
            f"session bindings must exactly cover development index: missing={missing}, extra={extra}"
        )
    manifest = {
        "schema_version": CAMERA_DAILY_BINDING_SCHEMA_VERSION,
        "binding_id": str(config["daily_baseline_id"]),
        "validation_scope": str(config["validation_scope"]),
        "capture_clock_status": "unavailable_declared_development_schedule",
        "person_identity_source": "development_index_owner_binding_not_track_id",
        "development_index": _file_ref(
            "development_index", "w5d00-development-index", index_path, index_bytes
        ),
        "config_sha256": config_sha256,
        "video_binding_count": len(manifest_rows),
        "session_count": len({row["session_id"] for row in manifest_rows}),
        "person_count": len({row["person_id"] for row in manifest_rows}),
        "video_bindings": sorted(
            manifest_rows, key=lambda row: (row["started_at"], row["source_video_id"])
        ),
    }
    _validate_binding_manifest(manifest)
    return tuple(bindings), manifest


def _build_replay_rows(
    source_rows: Sequence[Mapping[str, Any]],
    replay_config: Mapping[str, Any],
    replay_days: int,
) -> list[dict[str, Any]]:
    if not source_rows:
        raise WanderingCameraDailyBaselineError("cannot replay an empty daily source")
    template = next(
        (row for row in source_rows if _daily_usable(row, 0.0)), source_rows[0]
    )
    try:
        start = date.fromisoformat(str(replay_config["start_local_date"]))
    except ValueError as exc:
        raise WanderingCameraDailyBaselineError("replay start_local_date is invalid") from exc
    person_id = _token(replay_config["person_id"], "replay person_id")
    template_sha = hashlib.sha256(canonical_json_bytes(template)).hexdigest()
    rows: list[dict[str, Any]] = []
    for offset in range(replay_days):
        row = json.loads(json.dumps(template))
        local_date = (start + timedelta(days=offset)).isoformat()
        row["person_id"] = person_id
        row["session_id"] = None
        row["source_video_id"] = None
        row["local_date"] = local_date
        row["status"] = "uncertain"
        row["baseline_readiness"] = "warming_up"
        row["quality_flags"] = sorted(
            set(row["quality_flags"])
            | {
                "deterministic_replay_not_real_longitudinal_observation",
                "replayed_from_real_development_day",
            }
        )
        row["source_refs"] = sorted(
            [
                *row["source_refs"],
                {
                    "ref_type": "replay_template_daily_report",
                    "ref_id": str(template["record_id"]),
                    "artifact_path": None,
                    "sha256": template_sha,
                },
            ],
            key=lambda ref: (
                str(ref["ref_type"]),
                str(ref["ref_id"]),
                str(ref["artifact_path"]),
                str(ref["sha256"]),
            ),
        )
        payload = {
            "person_id": person_id,
            "local_date": local_date,
            "timezone": row["timezone"],
            "template_sha256": template_sha,
            "identity": row["identity"],
        }
        row["record_id"] = "w5d04-replay-daily-" + hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        rows.append(row)
    return rows


def _tracking_intervals(
    timestamps: Sequence[float],
    *,
    duration: float,
    nominal_fps: float,
    maximum_gap: float,
) -> list[tuple[float, float]]:
    ordered = sorted(set(value for value in timestamps if 0.0 <= value <= duration))
    if not ordered:
        return []
    frame_period = 1.0 / nominal_fps
    intervals: list[tuple[float, float]] = []
    start = previous = ordered[0]
    for timestamp in ordered[1:]:
        if timestamp - previous > maximum_gap:
            intervals.append((start, min(duration, previous + frame_period)))
            start = timestamp
        previous = timestamp
    intervals.append((start, min(duration, previous + frame_period)))
    return _merge_intervals(
        [(left, right) for left, right in intervals if right > left]
    )


def _common_episode_identity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise WanderingCameraDailyBaselineError("episode results must be non-empty")
    identities = [row.get("identity") for row in rows]
    first = identities[0]
    if not isinstance(first, Mapping) or any(identity != first for identity in identities[1:]):
        raise WanderingCameraDailyBaselineError("episode result identity drift")
    _validate_identity(first)
    return dict(first)


def _baseline_config_payload(path: Path) -> dict[str, Any]:
    config = load_mental_health_config(path).baseline
    return {
        "initial_days": config.initial_days,
        "stable_days": config.stable_days,
        "max_window_days": config.max_window_days,
        "upper_quantile": config.upper_quantile,
    }


def _validate_schema_files(
    root: Path, values: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    expected = {
        "binding_manifest": CAMERA_DAILY_BINDING_SCHEMA_VERSION,
        "daily_reports": CAMERA_DAILY_REPORT_SCHEMA_VERSION,
        "baseline_profiles": CAMERA_BASELINE_PROFILE_SCHEMA_VERSION,
        "baseline_deviations": CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION,
        "run_summary": RUN_SUMMARY_SCHEMA_VERSION,
    }
    if set(values) != set(expected):
        raise WanderingCameraDailyBaselineError("schema config names drifted")
    refs: dict[str, dict[str, Any]] = {}
    for name, schema_id in expected.items():
        path = _resolve(root, values[name])
        payload = _read_bytes(path, f"schema {name}")
        try:
            schema = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WanderingCameraDailyBaselineError(
                f"schema {name} is invalid JSON"
            ) from exc
        if not isinstance(schema, Mapping) or schema.get("$id") != schema_id:
            raise WanderingCameraDailyBaselineError(f"schema {name} id drifted")
        refs[name] = {
            "schema_version": schema_id,
            "artifact_path": path.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return refs


def _verify_schema_refs(values: Mapping[str, Any]) -> None:
    expected = {
        "binding_manifest": CAMERA_DAILY_BINDING_SCHEMA_VERSION,
        "daily_reports": CAMERA_DAILY_REPORT_SCHEMA_VERSION,
        "baseline_profiles": CAMERA_BASELINE_PROFILE_SCHEMA_VERSION,
        "baseline_deviations": CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION,
        "run_summary": RUN_SUMMARY_SCHEMA_VERSION,
    }
    if set(values) != set(expected):
        raise WanderingCameraDailyBaselineError("manifest schema names drifted")
    for name, schema_id in expected.items():
        ref = values[name]
        _exact_fields(
            ref,
            {"schema_version", "artifact_path", "sha256"},
            f"schema ref {name}",
        )
        if ref["schema_version"] != schema_id:
            raise WanderingCameraDailyBaselineError(f"schema ref {name} id drifted")
        _validate_digest(ref["sha256"], f"schema ref {name} sha256")
        path = Path(str(ref["artifact_path"]))
        payload = _read_bytes(path, f"schema ref {name}")
        if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
            raise WanderingCameraDailyBaselineError(f"schema ref {name} hash drifted")


def _episode_tier(
    episode: Mapping[str, Any], *, wandering_like: bool
) -> str | None:
    status = str(episode["status"])
    if status == "error":
        return "error"
    if status == "unavailable":
        return "unavailable"
    if wandering_like and status == "ready" and episode.get("run_status") == "auto_accepted":
        return "high_confidence"
    if status == "uncertain":
        return "uncertain"
    return None


def _context_label(
    episode: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    *,
    wandering_like: bool,
) -> tuple[str, str | None]:
    eligible = wandering_like or episode.get("status") == "uncertain"
    if context is None:
        return ("not_reviewed", "eligible_context_not_reviewed") if eligible else (
            "not_triggered",
            None,
        )
    if context.get("episode_id") not in {None, episode.get("episode_id")}:
        raise WanderingCameraDailyBaselineError("context episode_id mismatch")
    provider = context.get("provider")
    if provider == "deterministic-fake":
        return "unknown", "context_fake_excluded_from_semantic_counts"
    label = context.get("context_label")
    if context.get("status") != "ready" or label not in _CONTEXT_LABELS - {
        "not_reviewed",
        "not_triggered",
    }:
        return "unknown", "context_unavailable_retained"
    return str(label), None


def _add_source_ref(day: _Day, value: Mapping[str, Any]) -> None:
    ref = _validated_source_ref(value)
    key = (
        str(ref["ref_type"]),
        str(ref["ref_id"]),
        ref["artifact_path"],
        ref["sha256"],
    )
    day.source_refs[key] = ref


def _validate_binding(binding: DailyVideoBinding) -> None:
    _token(binding.source_video_id, "source_video_id")
    _token(binding.session_id, "session_id")
    _token(binding.person_id, "person_id")
    try:
        zone = ZoneInfo(binding.timezone)
    except ZoneInfoNotFoundError as exc:
        raise WanderingCameraDailyBaselineError("binding timezone is unknown") from exc
    if binding.started_at.tzinfo is None or binding.started_at.utcoffset() is None:
        raise WanderingCameraDailyBaselineError("binding started_at must be timezone-aware")
    local = binding.started_at.astimezone(zone)
    if (
        local.replace(tzinfo=None) != binding.started_at.replace(tzinfo=None)
        or local.utcoffset() != binding.started_at.utcoffset()
    ):
        raise WanderingCameraDailyBaselineError(
            "binding timezone and started_at offset disagree"
        )
    _validated_intervals(binding.presence_intervals_sec, "presence")
    _validated_intervals(binding.tracking_intervals_sec, "tracking", allow_empty=True)
    _nonempty_string(binding.time_basis, "time_basis")
    _nonempty_string(binding.presence_basis, "presence_basis")
    if not binding.source_refs:
        raise WanderingCameraDailyBaselineError("binding source_refs must be non-empty")
    for ref in binding.source_refs:
        _validated_source_ref(ref)


def _validate_config(value: Any) -> None:
    fields = {
        "schema_version",
        "daily_baseline_id",
        "validation_scope",
        "home_input_status",
        "home_smoke_status",
        "inputs",
        "schemas",
        "aggregation",
        "session_bindings",
        "replay",
        "identity",
    }
    _exact_fields(value, fields, "daily/baseline config")
    if value["schema_version"] != CAMERA_DAILY_BASELINE_CONFIG_SCHEMA_VERSION:
        raise WanderingCameraDailyBaselineError("daily/baseline config schema drifted")
    _token(value["daily_baseline_id"], "daily_baseline_id")
    if value["validation_scope"] != "b01_b02_development":
        raise WanderingCameraDailyBaselineError("production validation scope drifted")
    if value["home_input_status"] not in {"awaiting_input", "available"}:
        raise WanderingCameraDailyBaselineError("home_input_status is invalid")
    _nonempty_string(value["home_smoke_status"], "home_smoke_status")
    _exact_fields(
        value["inputs"],
        {
            "development_index",
            "episode_results",
            "context_reviews",
            "mental_health_config",
        },
        "config inputs",
    )
    for path in value["inputs"].values():
        _nonempty_string(path, "input path")
    _exact_fields(
        value["schemas"],
        {
            "binding_manifest",
            "daily_reports",
            "baseline_profiles",
            "baseline_deviations",
            "run_summary",
        },
        "config schemas",
    )
    for path in value["schemas"].values():
        _nonempty_string(path, "schema path")
    _exact_fields(
        value["aggregation"],
        {"minimum_usable_tracking_coverage", "tracking_max_gap_seconds"},
        "config aggregation",
    )
    _bounded_number(
        value["aggregation"]["minimum_usable_tracking_coverage"],
        "minimum_usable_tracking_coverage",
        minimum=0.0,
        maximum=1.0,
    )
    _positive_number(
        value["aggregation"]["tracking_max_gap_seconds"],
        "tracking_max_gap_seconds",
    )
    sessions = value["session_bindings"]
    if not isinstance(sessions, list) or not sessions:
        raise WanderingCameraDailyBaselineError("session_bindings must be non-empty")
    session_ids: set[str] = set()
    clip_ids: set[str] = set()
    session_fields = {
        "batch_id",
        "session_id",
        "person_id",
        "timezone",
        "session_started_at",
        "clip_gap_seconds",
        "clip_order",
        "time_basis",
        "presence_basis",
    }
    for session in sessions:
        _exact_fields(session, session_fields, "session binding")
        session_id = _token(session["session_id"], "session_id")
        if session_id in session_ids:
            raise WanderingCameraDailyBaselineError("duplicate session binding")
        session_ids.add(session_id)
        _token(session["batch_id"], "batch_id")
        _token(session["person_id"], "person_id")
        _aware_datetime(
            session["session_started_at"], session["timezone"], "session_started_at"
        )
        _nonnegative_number(session["clip_gap_seconds"], "clip_gap_seconds")
        clips = session["clip_order"]
        if not isinstance(clips, list) or not clips:
            raise WanderingCameraDailyBaselineError("clip_order must be non-empty")
        for clip in clips:
            clip_id = _token(clip, "clip_order source_video_id")
            if clip_id in clip_ids:
                raise WanderingCameraDailyBaselineError("clip_order duplicates a video")
            clip_ids.add(clip_id)
        _nonempty_string(session["time_basis"], "time_basis")
        _nonempty_string(session["presence_basis"], "presence_basis")
    _exact_fields(
        value["replay"],
        {"person_id", "start_local_date", "default_days"},
        "replay config",
    )
    _token(value["replay"]["person_id"], "replay person_id")
    try:
        date.fromisoformat(str(value["replay"]["start_local_date"]))
    except ValueError as exc:
        raise WanderingCameraDailyBaselineError("replay start date is invalid") from exc
    if (
        isinstance(value["replay"]["default_days"], bool)
        or not isinstance(value["replay"]["default_days"], int)
        or value["replay"]["default_days"] < 1
    ):
        raise WanderingCameraDailyBaselineError("replay default_days is invalid")
    _exact_fields(value["identity"], {"config_id", "policy_id"}, "config identity")
    _token(value["identity"]["config_id"], "config_id")
    _token(value["identity"]["policy_id"], "policy_id")


def _validate_index_row(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "batch_id",
        "dataset_role",
        "development_reuse_allowed",
        "participant_id",
        "session_id",
        "source_video_id",
        "timezone",
        "artifacts",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise WanderingCameraDailyBaselineError("development index row is incomplete")
    if value["schema_version"] != "wandering-camera-development-index-v1":
        raise WanderingCameraDailyBaselineError("development index schema drifted")
    if value["dataset_role"] != "development" or value["development_reuse_allowed"] is not True:
        raise WanderingCameraDailyBaselineError("development row is not reusable development")
    for name in ("batch_id", "participant_id", "session_id", "source_video_id"):
        _token(value[name], f"development {name}")
    try:
        ZoneInfo(str(value["timezone"]))
    except ZoneInfoNotFoundError as exc:
        raise WanderingCameraDailyBaselineError("development timezone is unknown") from exc
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or not {
        "tracking",
        "media_sidecar",
    }.issubset(artifacts):
        raise WanderingCameraDailyBaselineError("development artifacts are incomplete")


def _validate_binding_manifest(value: Mapping[str, Any]) -> None:
    _exact_fields(
        value,
        {
            "schema_version",
            "binding_id",
            "validation_scope",
            "capture_clock_status",
            "person_identity_source",
            "development_index",
            "config_sha256",
            "video_binding_count",
            "session_count",
            "person_count",
            "video_bindings",
        },
        "binding manifest",
    )
    if value["schema_version"] != CAMERA_DAILY_BINDING_SCHEMA_VERSION:
        raise WanderingCameraDailyBaselineError("binding manifest schema drifted")
    _token(value["binding_id"], "binding manifest binding_id")
    if value["validation_scope"] != "b01_b02_development":
        raise WanderingCameraDailyBaselineError("binding validation_scope drifted")
    if value["capture_clock_status"] != "unavailable_declared_development_schedule":
        raise WanderingCameraDailyBaselineError("binding capture_clock_status drifted")
    if value["person_identity_source"] != "development_index_owner_binding_not_track_id":
        raise WanderingCameraDailyBaselineError("binding person identity source drifted")
    _validate_digest(value["config_sha256"], "binding config_sha256")
    _validated_source_ref(value["development_index"])
    rows = value["video_bindings"]
    if not isinstance(rows, list) or not rows:
        raise WanderingCameraDailyBaselineError("binding video rows are empty")
    if value["video_binding_count"] != len(rows):
        raise WanderingCameraDailyBaselineError("binding video count mismatch")
    if value["session_count"] != len({row["session_id"] for row in rows}):
        raise WanderingCameraDailyBaselineError("binding session count mismatch")
    if value["person_count"] != len({row["person_id"] for row in rows}):
        raise WanderingCameraDailyBaselineError("binding person count mismatch")
    row_fields = {
        "source_video_id",
        "session_id",
        "person_id",
        "timezone",
        "started_at",
        "end_at",
        "time_basis",
        "presence_basis",
        "presence_intervals_sec",
        "tracking_intervals_sec",
        "source_refs",
    }
    seen_videos: set[str] = set()
    for row in rows:
        _exact_fields(row, row_fields, "binding video row")
        video_id = _token(row["source_video_id"], "binding source_video_id")
        if video_id in seen_videos:
            raise WanderingCameraDailyBaselineError("binding video row is duplicated")
        seen_videos.add(video_id)
        _token(row["session_id"], "binding session_id")
        _token(row["person_id"], "binding person_id")
        started = _aware_datetime(row["started_at"], row["timezone"], "binding started_at")
        ended = _aware_datetime(row["end_at"], row["timezone"], "binding end_at")
        if ended <= started:
            raise WanderingCameraDailyBaselineError("binding end_at must follow started_at")
        _nonempty_string(row["time_basis"], "binding time_basis")
        _nonempty_string(row["presence_basis"], "binding presence_basis")
        presence = _manifest_intervals(row["presence_intervals_sec"], "binding presence")
        tracking = _manifest_intervals(
            row["tracking_intervals_sec"], "binding tracking", allow_empty=True
        )
        duration = (ended - started).total_seconds()
        if any(right > duration + 1e-6 for _, right in (*presence, *tracking)):
            raise WanderingCameraDailyBaselineError("binding interval exceeds clip duration")
        if any(not _covered_by(interval, presence) for interval in tracking):
            raise WanderingCameraDailyBaselineError("binding tracking is outside presence")
        _validate_source_refs(row["source_refs"], "binding source_refs")


def _manifest_intervals(
    value: Any,
    role: str,
    *,
    allow_empty: bool = False,
) -> list[tuple[float, float]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise WanderingCameraDailyBaselineError(f"{role} intervals are invalid")
    intervals: list[tuple[float, float]] = []
    for item in value:
        _exact_fields(
            item,
            {"start_sec", "end_sec_exclusive"},
            f"{role} interval",
        )
        intervals.append(
            _interval(item["start_sec"], item["end_sec_exclusive"], role)
        )
    if _merge_intervals(intervals) != intervals:
        raise WanderingCameraDailyBaselineError(
            f"{role} intervals must be sorted and non-overlapping"
        )
    return intervals


def _validate_daily_report(value: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "module",
        "record_id",
        "person_id",
        "session_id",
        "source_video_id",
        "local_date",
        "timezone",
        "status",
        "presence_seconds",
        "tracking_coverage_seconds",
        "tracking_coverage",
        "qc_coverage",
        "episode_counts",
        "episode_duration_seconds",
        "confidence_tiers",
        "rates_per_presence_hour",
        "night_wandering_like_ratio",
        "context_counts",
        "unavailable_count",
        "error_count",
        "baseline_readiness",
        "quality_flags",
        "identity",
        "source_refs",
    }
    _exact_fields(value, fields, "daily report")
    if (
        value["schema_version"] != CAMERA_DAILY_REPORT_SCHEMA_VERSION
        or value["module"] != "mental_health"
    ):
        raise WanderingCameraDailyBaselineError("daily report schema/module drifted")
    _token(value["record_id"], "daily record_id")
    _token(value["person_id"], "daily person_id")
    _nullable_token(value["session_id"], "daily session_id")
    _nullable_token(value["source_video_id"], "daily source_video_id")
    _date_string(value["local_date"], "daily local_date")
    try:
        ZoneInfo(str(value["timezone"]))
    except ZoneInfoNotFoundError as exc:
        raise WanderingCameraDailyBaselineError("daily timezone is unknown") from exc
    _status(value["status"], "daily status")
    presence = _nullable_nonnegative(value["presence_seconds"], "presence_seconds")
    tracking_seconds = _nullable_nonnegative(
        value["tracking_coverage_seconds"], "tracking_coverage_seconds"
    )
    tracking_coverage = _nullable_bounded(
        value["tracking_coverage"], "tracking_coverage"
    )
    _nullable_bounded(value["qc_coverage"], "qc_coverage")
    if presence is None:
        if tracking_seconds is not None or tracking_coverage is not None:
            raise WanderingCameraDailyBaselineError(
                "daily coverage must be null when presence is unavailable"
            )
    elif tracking_seconds is not None and tracking_seconds > presence + 1e-6:
        raise WanderingCameraDailyBaselineError("tracking seconds exceed presence")
    shape_fields = set((*SHAPE_ORDER, "wandering_like"))
    _exact_fields(value["episode_counts"], shape_fields, "episode_counts")
    _exact_fields(
        value["episode_duration_seconds"], shape_fields, "episode durations"
    )
    for name in shape_fields:
        _nonnegative_integer(value["episode_counts"][name], f"episode_counts.{name}")
        _nonnegative_number(
            value["episode_duration_seconds"][name],
            f"episode_duration_seconds.{name}",
        )
    _exact_fields(value["confidence_tiers"], set(TIER_ORDER), "confidence tiers")
    for name in TIER_ORDER:
        tier = value["confidence_tiers"][name]
        _exact_fields(tier, {"count", "duration_seconds"}, f"tier {name}")
        _nonnegative_integer(tier["count"], f"tier {name} count")
        _nonnegative_number(tier["duration_seconds"], f"tier {name} duration")
    if value["unavailable_count"] != value["confidence_tiers"]["unavailable"]["count"]:
        raise WanderingCameraDailyBaselineError("daily unavailable_count drifted")
    if value["error_count"] != value["confidence_tiers"]["error"]["count"]:
        raise WanderingCameraDailyBaselineError("daily error_count drifted")
    _exact_fields(
        value["rates_per_presence_hour"],
        {"wandering_like_count", "wandering_like_duration_seconds"},
        "daily rates",
    )
    for rate in value["rates_per_presence_hour"].values():
        _nullable_nonnegative(rate, "daily rate")
    _nullable_bounded(value["night_wandering_like_ratio"], "night ratio")
    contexts = value["context_counts"]
    if not isinstance(contexts, Mapping):
        raise WanderingCameraDailyBaselineError("context_counts must be a mapping")
    for name, count in contexts.items():
        if name not in _CONTEXT_LABELS:
            raise WanderingCameraDailyBaselineError("context_counts label is invalid")
        _nonnegative_integer(count, f"context_counts.{name}")
    if value["baseline_readiness"] not in _READINESS:
        raise WanderingCameraDailyBaselineError("daily baseline_readiness is invalid")
    _validate_string_list(value["quality_flags"], "daily quality_flags")
    _validate_identity(value["identity"])
    _validate_source_refs(value["source_refs"], "daily source_refs")
    _reject_medical_fields(value)


def _validate_baseline_profile(value: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "module",
        "record_id",
        "profile_id",
        "person_id",
        "session_id",
        "source_video_id",
        "local_date",
        "timezone",
        "status",
        "readiness_status",
        "usable_day_count",
        "reference_window_days",
        "reference_first_local_date",
        "reference_last_local_date",
        "metrics",
        "quality_flags",
        "identity",
        "source_refs",
    }
    _exact_fields(value, fields, "baseline profile")
    if (
        value["schema_version"] != CAMERA_BASELINE_PROFILE_SCHEMA_VERSION
        or value["module"] != "mental_health"
    ):
        raise WanderingCameraDailyBaselineError("baseline profile schema/module drifted")
    if value["record_id"] != value["profile_id"]:
        raise WanderingCameraDailyBaselineError("profile record/profile id mismatch")
    _token(value["profile_id"], "profile_id")
    _token(value["person_id"], "profile person_id")
    _nullable_token(value["session_id"], "profile session_id")
    _nullable_token(value["source_video_id"], "profile source_video_id")
    current_date = _date_string(value["local_date"], "profile local_date")
    _status(value["status"], "profile status")
    if value["readiness_status"] not in _READINESS:
        raise WanderingCameraDailyBaselineError("profile readiness is invalid")
    count = _nonnegative_integer(value["usable_day_count"], "usable_day_count")
    window = _positive_integer(value["reference_window_days"], "reference_window_days")
    if count > window:
        raise WanderingCameraDailyBaselineError("profile usable days exceed window")
    first = _nullable_date(value["reference_first_local_date"], "profile first date")
    last = _nullable_date(value["reference_last_local_date"], "profile last date")
    if count == 0:
        if first is not None or last is not None:
            raise WanderingCameraDailyBaselineError("empty profile has reference dates")
    elif first is None or last is None or not first <= last < current_date:
        raise WanderingCameraDailyBaselineError(
            "profile reference dates are not strictly prior"
        )
    metrics = value["metrics"]
    _exact_fields(metrics, set(METRIC_ORDER), "baseline metrics")
    for name, stats in metrics.items():
        _exact_fields(stats, {"count", "median", "p90"}, f"profile metric {name}")
        metric_count = _nonnegative_integer(stats["count"], f"profile metric {name} count")
        median = _nullable_number(stats["median"], f"profile metric {name} median")
        p90 = _nullable_number(stats["p90"], f"profile metric {name} p90")
        if metric_count == 0 and (median is not None or p90 is not None):
            raise WanderingCameraDailyBaselineError("empty metric has statistics")
        if metric_count > 0 and (median is None or p90 is None or p90 < median):
            raise WanderingCameraDailyBaselineError("profile metric statistics are invalid")
    _validate_string_list(value["quality_flags"], "profile quality_flags")
    _validate_identity(value["identity"])
    _validate_source_refs(value["source_refs"], "profile source_refs")
    _reject_medical_fields(value)


def _validate_baseline_deviation(value: Mapping[str, Any]) -> None:
    fields = {
        "schema_version",
        "module",
        "record_id",
        "deviation_id",
        "profile_id",
        "person_id",
        "session_id",
        "source_video_id",
        "local_date",
        "timezone",
        "status",
        "readiness_status",
        "metrics",
        "quality_flags",
        "identity",
        "source_refs",
    }
    _exact_fields(value, fields, "baseline deviation")
    if (
        value["schema_version"] != CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION
        or value["module"] != "mental_health"
    ):
        raise WanderingCameraDailyBaselineError("baseline deviation schema/module drifted")
    if value["record_id"] != value["deviation_id"]:
        raise WanderingCameraDailyBaselineError("deviation record/deviation id mismatch")
    _token(value["deviation_id"], "deviation_id")
    _token(value["profile_id"], "deviation profile_id")
    _token(value["person_id"], "deviation person_id")
    _nullable_token(value["session_id"], "deviation session_id")
    _nullable_token(value["source_video_id"], "deviation source_video_id")
    _date_string(value["local_date"], "deviation local_date")
    _status(value["status"], "deviation status")
    if value["readiness_status"] not in _READINESS:
        raise WanderingCameraDailyBaselineError("deviation readiness is invalid")
    _exact_fields(value["metrics"], set(METRIC_ORDER), "deviation metrics")
    metric_fields = {
        "status",
        "observation_value",
        "reference_count",
        "reference_median",
        "reference_p90",
        "delta_from_median",
        "delta_from_p90",
    }
    for name, metric in value["metrics"].items():
        _exact_fields(metric, metric_fields, f"deviation metric {name}")
        if metric["status"] not in {
            "ready",
            "warming_up",
            "observation_unavailable",
            "reference_unavailable",
        }:
            raise WanderingCameraDailyBaselineError("deviation metric status is invalid")
        _nullable_number(metric["observation_value"], "deviation observation")
        _nonnegative_integer(metric["reference_count"], "deviation reference_count")
        for field_name in (
            "reference_median",
            "reference_p90",
            "delta_from_median",
            "delta_from_p90",
        ):
            _nullable_number(metric[field_name], f"deviation {field_name}")
        if metric["status"] != "ready" and (
            metric["delta_from_median"] is not None
            or metric["delta_from_p90"] is not None
        ):
            raise WanderingCameraDailyBaselineError(
                "non-ready deviation metric cannot contain deltas"
            )
    _validate_string_list(value["quality_flags"], "deviation quality_flags")
    _validate_identity(value["identity"])
    _validate_source_refs(value["source_refs"], "deviation source_refs")
    _reject_medical_fields(value)


def _validate_identity(value: Any) -> None:
    _exact_fields(
        value,
        {
            "model_id",
            "model_sha256",
            "config_id",
            "config_sha256",
            "policy_id",
            "policy_sha256",
        },
        "identity",
    )
    if value["model_id"] is not None:
        _nonempty_string(value["model_id"], "identity model_id")
    if value["model_sha256"] is not None:
        _validate_digest(value["model_sha256"], "identity model_sha256")
    _nonempty_string(value["config_id"], "identity config_id")
    _validate_digest(value["config_sha256"], "identity config_sha256")
    _nonempty_string(value["policy_id"], "identity policy_id")
    if value["policy_sha256"] is not None:
        _validate_digest(value["policy_sha256"], "identity policy_sha256")


def _validated_source_ref(value: Any) -> dict[str, Any]:
    _exact_fields(
        value,
        {"ref_type", "ref_id", "artifact_path", "sha256"},
        "source ref",
    )
    _nonempty_string(value["ref_type"], "source ref type")
    _nonempty_string(value["ref_id"], "source ref id")
    if value["artifact_path"] is not None:
        _nonempty_string(value["artifact_path"], "source ref artifact_path")
    if value["sha256"] is not None:
        _validate_digest(value["sha256"], "source ref sha256")
    return {
        "ref_type": str(value["ref_type"]),
        "ref_id": str(value["ref_id"]),
        "artifact_path": value["artifact_path"],
        "sha256": value["sha256"],
    }


def _validate_source_refs(value: Any, role: str) -> None:
    if not isinstance(value, list) or not value:
        raise WanderingCameraDailyBaselineError(f"{role} must be non-empty")
    refs = [_validated_source_ref(ref) for ref in value]
    keys = [
        (ref["ref_type"], ref["ref_id"], ref["artifact_path"], ref["sha256"])
        for ref in refs
    ]
    if len(keys) != len(set(keys)):
        raise WanderingCameraDailyBaselineError(f"{role} contains duplicates")


def _validate_string_list(value: Any, role: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise WanderingCameraDailyBaselineError(
            f"{role} must be a sorted unique string list"
        )


def _reject_medical_fields(value: Mapping[str, Any]) -> None:
    forbidden = {
        "medical_diagnosis",
        "diagnosis",
        "risk_level",
        "risk_score",
        "alert_decision",
        "recommended_action",
        "algorithm_event",
    }
    if forbidden.intersection(value):
        raise WanderingCameraDailyBaselineError(
            "W5D-04 evidence cannot contain diagnosis/risk/alert fields"
        )


def _validated_intervals(
    value: Sequence[tuple[float, float]],
    role: str,
    *,
    allow_empty: bool = False,
) -> list[tuple[float, float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WanderingCameraDailyBaselineError(f"{role} intervals are invalid")
    if not value and not allow_empty:
        raise WanderingCameraDailyBaselineError(f"{role} intervals must be non-empty")
    intervals = [_interval(left, right, role) for left, right in value]
    merged = _merge_intervals(intervals)
    if not allow_empty and _duration(merged) <= 0.0:
        raise WanderingCameraDailyBaselineError(f"{role} duration must be positive")
    return merged


def _covered_by(
    interval: tuple[float, float],
    coverage: Sequence[tuple[float, float]],
) -> bool:
    covered = _intersect([interval], coverage)
    return math.isclose(
        _duration(covered), interval[1] - interval[0], rel_tol=0.0, abs_tol=1e-6
    )


def _interval(start: Any, end: Any, role: str) -> tuple[float, float]:
    left = _nonnegative_number(start, f"{role} start")
    right = _nonnegative_number(end, f"{role} end")
    if right <= left:
        raise WanderingCameraDailyBaselineError(f"{role} interval is invalid")
    return left, right


def _clock(value: Any, role: str) -> time:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise WanderingCameraDailyBaselineError(f"{role} must use HH:MM")
    try:
        parsed = time(hour=int(value[:2]), minute=int(value[3:]))
    except ValueError as exc:
        raise WanderingCameraDailyBaselineError(f"{role} is invalid") from exc
    if parsed.strftime("%H:%M") != value:
        raise WanderingCameraDailyBaselineError(f"{role} must be zero-padded")
    return parsed


def _aware_datetime(value: Any, timezone_name: Any, role: str) -> datetime:
    if not isinstance(value, str) or not isinstance(timezone_name, str):
        raise WanderingCameraDailyBaselineError(f"{role}/timezone is invalid")
    try:
        parsed = datetime.fromisoformat(value)
        zone = ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise WanderingCameraDailyBaselineError(f"{role}/timezone is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WanderingCameraDailyBaselineError(f"{role} must include a UTC offset")
    observed = parsed.astimezone(zone)
    if (
        observed.replace(tzinfo=None) != parsed.replace(tzinfo=None)
        or observed.utcoffset() != parsed.utcoffset()
    ):
        raise WanderingCameraDailyBaselineError(
            f"{role} offset and timezone disagree"
        )
    return parsed


def _status(value: Any, role: str) -> str:
    if value not in _STATUSES:
        raise WanderingCameraDailyBaselineError(f"{role} is invalid")
    return str(value)


def _date_string(value: Any, role: str) -> date:
    if not isinstance(value, str):
        raise WanderingCameraDailyBaselineError(f"{role} is invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise WanderingCameraDailyBaselineError(f"{role} is invalid") from exc


def _nullable_date(value: Any, role: str) -> date | None:
    return None if value is None else _date_string(value, role)


def _token(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise WanderingCameraDailyBaselineError(f"{role} must be a non-empty token")
    if any(character.isspace() for character in value):
        raise WanderingCameraDailyBaselineError(f"{role} cannot contain whitespace")
    return value


def _nullable_token(value: Any, role: str) -> str | None:
    return None if value is None else _token(value, role)


def _nonempty_string(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WanderingCameraDailyBaselineError(f"{role} must be a non-empty string")
    return value


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _nonnegative_number(value: Any, role: str) -> float:
    if not _finite_number(value) or float(value) < 0.0:
        raise WanderingCameraDailyBaselineError(f"{role} must be non-negative")
    return float(value)


def _positive_number(value: Any, role: str) -> float:
    number = _nonnegative_number(value, role)
    if number <= 0.0:
        raise WanderingCameraDailyBaselineError(f"{role} must be positive")
    return number


def _bounded_number(
    value: Any,
    role: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if not _finite_number(value) or not minimum <= float(value) <= maximum:
        raise WanderingCameraDailyBaselineError(
            f"{role} must be in [{minimum}, {maximum}]"
        )
    return float(value)


def _nullable_number(value: Any, role: str) -> float | None:
    if value is None:
        return None
    if not _finite_number(value):
        raise WanderingCameraDailyBaselineError(f"{role} must be finite or null")
    return float(value)


def _nullable_nonnegative(value: Any, role: str) -> float | None:
    return None if value is None else _nonnegative_number(value, role)


def _nullable_bounded(value: Any, role: str) -> float | None:
    return (
        None
        if value is None
        else _bounded_number(value, role, minimum=0.0, maximum=1.0)
    )


def _nonnegative_integer(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WanderingCameraDailyBaselineError(f"{role} must be non-negative integer")
    return value


def _positive_integer(value: Any, role: str) -> int:
    result = _nonnegative_integer(value, role)
    if result <= 0:
        raise WanderingCameraDailyBaselineError(f"{role} must be positive")
    return result


def _validate_digest(value: Any, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WanderingCameraDailyBaselineError(f"{role} must be lowercase SHA-256")


def _exact_fields(value: Any, fields: set[str], role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise WanderingCameraDailyBaselineError(
            f"{role} fields drifted: missing={sorted(fields - actual)}, extra={sorted(actual - fields)}"
        )


def _resolve(root: Path, value: Any) -> Path:
    path = Path(_nonempty_string(value, "path"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _artifact_path(root: Path, descriptor: Mapping[str, Any], role: str) -> Path:
    path_value = descriptor.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise WanderingCameraDailyBaselineError(f"{role} artifact path is missing")
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_bytes(path: Path, role: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WanderingCameraDailyBaselineError(f"cannot read {role}: {path}") from exc


def _load_json(path: Path, role: str) -> dict[str, Any]:
    payload = _read_bytes(path, role)
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingCameraDailyBaselineError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise WanderingCameraDailyBaselineError(f"{role} must be a JSON object")
    if canonical_json_bytes(value) != payload:
        raise WanderingCameraDailyBaselineError(f"{role} is not canonical JSON")
    return value


def _load_jsonl(
    path: Path,
    role: str,
    *,
    allow_empty: bool = False,
    require_canonical: bool = True,
) -> tuple[bytes, list[dict[str, Any]]]:
    payload = _read_bytes(path, role)
    if not payload:
        if allow_empty:
            return payload, []
        raise WanderingCameraDailyBaselineError(f"{role} must be non-empty")
    rows: list[dict[str, Any]] = []
    try:
        for line in payload.decode("utf-8").splitlines():
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise WanderingCameraDailyBaselineError(
                    f"{role} rows must be objects"
                )
            rows.append(value)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingCameraDailyBaselineError(f"{role} is not valid JSONL") from exc
    if require_canonical and canonical_jsonl_bytes(rows) != payload:
        raise WanderingCameraDailyBaselineError(f"{role} is not canonical JSONL")
    return payload, rows


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _verify_external_descriptor(
    path: Path,
    descriptor: Mapping[str, Any],
    role: str,
) -> None:
    payload = _read_bytes(path, role)
    if descriptor.get("byte_count") != len(payload):
        raise WanderingCameraDailyBaselineError(f"{role} byte_count mismatch")
    if descriptor.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise WanderingCameraDailyBaselineError(f"{role} sha256 mismatch")


def _descriptor(
    payload: bytes,
    schema_version: str | None,
    record_count: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "record_count": record_count,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _verify_descriptor(path: Path, value: Any) -> None:
    _exact_fields(
        value,
        {"schema_version", "record_count", "byte_count", "sha256"},
        "artifact descriptor",
    )
    payload = _read_bytes(path, f"artifact {path.name}")
    if value["byte_count"] != len(payload):
        raise WanderingCameraDailyBaselineError(
            f"artifact descriptor byte_count mismatch: {path.name}"
        )
    if value["sha256"] != hashlib.sha256(payload).hexdigest():
        raise WanderingCameraDailyBaselineError(
            f"artifact descriptor sha256 mismatch: {path.name}"
        )


def _file_ref(
    ref_type: str,
    ref_id: str,
    path: Path,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "artifact_path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_new_file(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.write_bytes(payload)


def _readme_text(
    *,
    validation_scope: str,
    daily_rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    home_input_status: str,
    home_smoke_status: str,
) -> str:
    readiness = Counter(str(row["readiness_status"]) for row in profiles)
    return "\n".join(
        [
            "# W5D-04 real daily reports and rolling personal baseline",
            "",
            f"- validation scope: `{validation_scope}`",
            f"- daily reports: `{len(daily_rows)}`",
            f"- persons: `{len({str(row['person_id']) for row in daily_rows})}`",
            f"- local dates: `{len({str(row['local_date']) for row in daily_rows})}`",
            f"- warming_up / initial_ready / stable_ready: `{readiness['warming_up']} / {readiness['initial_ready']} / {readiness['stable_ready']}`",
            f"- home input: `{home_input_status}`",
            f"- home smoke: `{home_smoke_status}`",
            "",
            "B01+B02 rows use real development episode/tracking evidence with explicit person/session/timezone/presence binding. Capture clocks were absent, so the binding manifest uses a declared development replay schedule and does not claim observed wall-clock time.",
            "",
            "Deterministic replay bundles prove only the 3/7/14-day state machine. They are not real longitudinal observations, clinical validation, risk decisions, or AlgorithmEvent output.",
            "",
        ]
    )


def _verification_text(
    *,
    bindings: Sequence[DailyVideoBinding],
    daily_rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    deviations: Sequence[Mapping[str, Any]],
    validation_scope: str,
) -> str:
    readiness = Counter(str(row["readiness_status"]) for row in profiles)
    strict_prior = all(
        row["reference_last_local_date"] is None
        or str(row["reference_last_local_date"]) < str(row["local_date"])
        for row in profiles
    )
    maximum_window = max((int(row["usable_day_count"]) for row in profiles), default=0)
    return "\n".join(
        [
            "# W5D-04 verification",
            "",
            f"- validation_scope: `{validation_scope}`",
            f"- video_bindings: `{len(bindings)}`",
            f"- daily_reports: `{len(daily_rows)}`",
            f"- baseline_profiles: `{len(profiles)}`",
            f"- baseline_deviations: `{len(deviations)}`",
            f"- strict_prior_reference: `{str(strict_prior).lower()}`",
            f"- maximum_reference_day_count: `{maximum_window}`",
            f"- readiness_counts: `warming_up={readiness['warming_up']}, initial_ready={readiness['initial_ready']}, stable_ready={readiness['stable_ready']}`",
            "- public_loader_recomputed: `true`",
            "- algorithm_event_emitted: `false`",
            "- medical_diagnosis_emitted: `false`",
            "",
        ]
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise WanderingCameraDailyBaselineError("ratio denominator must be positive")
    return _measure(numerator / denominator)


__all__ = [
    "CAMERA_DAILY_BASELINE_CONFIG_SCHEMA_VERSION",
    "CAMERA_DAILY_BINDING_SCHEMA_VERSION",
    "CAMERA_DAILY_REPORT_SCHEMA_VERSION",
    "CAMERA_BASELINE_PROFILE_SCHEMA_VERSION",
    "CAMERA_BASELINE_DEVIATION_SCHEMA_VERSION",
    "DailyVideoBinding",
    "CameraDailyBaselineBuildResult",
    "ValidatedCameraDailyBaselineBundle",
    "WanderingCameraDailyBaselineError",
    "aggregate_wandering_daily_reports",
    "build_rolling_baseline_rows",
    "build_camera_daily_baseline_bundle",
    "load_camera_daily_baseline_config",
    "load_validated_camera_daily_baseline_bundle",
]
