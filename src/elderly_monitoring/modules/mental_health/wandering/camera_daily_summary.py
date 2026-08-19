"""Synthetic person-day summaries from fully validated MVP-1 camera products."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from elderly_monitoring.modules.mental_health.config import (
    DEFAULT_CONFIG_PATH,
    load_aggregation_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_product import (
    ValidatedWanderingCameraProduct,
    WanderingCameraProductError,
    load_validated_wandering_camera_product,
)


DAILY_BINDING_SCHEMA_VERSION = "wandering-daily-binding-v1"
DAILY_SUMMARY_SCHEMA_VERSION = "wandering-daily-summary-v1"
DAILY_SUMMARY_MANIFEST_SCHEMA_VERSION = "wandering-daily-summary-manifest-v1"
PRODUCT_NAME = "WanderingDailySummary"
PRODUCT_STAGE = "synthetic_daily_summary_prototype"
STATUS = "wandering_m0cam_synthetic_daily_summary_ready"
EVIDENCE_SCOPE = "synthetic_contract_only"
VALIDATION_SCOPE = "synthetic_camera_contract"
BINDING_EVIDENCE_SCOPE = "synthetic_binding_fixture"
DURATION_SEMANTICS = "candidate_interval_sum_not_presence_time"
FOUR_CLASS_ORDER = ("direct", "pacing", "lapping", "random")
BINARY_CLASS_ORDER = ("direct_or_non_wandering", "wandering_like")
SUBTYPE_ORDER = ("pacing", "lapping", "random")
WINDOW_STATUSES = ("ready", "unavailable", "inference_error")
_SOURCE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
)
_BINDING_FIELDS = frozenset({"schema_version", "evidence_scope", "sessions"})
_BINDING_SESSION_FIELDS = frozenset(
    {
        "binding_id",
        "product_manifest_sha256",
        "session_id",
        "session_started_at",
        "timezone",
        "source_scope",
        "person_tracks",
    }
)
_BINDING_PERSON_FIELDS = frozenset(
    {"synthetic_person_id", "track_ids", "presence_intervals_sec"}
)
_PRESENCE_FIELDS = frozenset({"start_sec", "end_sec_exclusive"})
_SOURCE_FIELD_SET = frozenset(_SOURCE_FIELDS)
_DESCRIPTOR_FIELDS = frozenset({"byte_count", "sha256"})
_NIGHT_WINDOW_FIELDS = frozenset({"start", "end"})
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "product_stage",
        "status",
        "evidence_scope",
        "validation_scope",
        "synthetic_person_id",
        "person_binding_verified",
        "local_date",
        "timezone",
        "night_start",
        "night_end",
        "product_count",
        "session_count",
        "source_scope_count",
        "setup_count",
        "track_count",
        "window_count",
        "presence_seconds",
        "any_window_covered_seconds",
        "ready_covered_seconds",
        "unavailable_covered_seconds",
        "inference_error_covered_seconds",
        "trajectory_coverage",
        "ready_coverage",
        "unavailable_coverage",
        "inference_error_coverage",
        "status_overlap_seconds",
        "qc_reason_counts",
        "quality_flag_counts",
        "quality_flags",
        "direct_episode_count",
        "pacing_episode_count",
        "lapping_episode_count",
        "random_episode_count",
        "direct_duration_sum_seconds",
        "pacing_duration_sum_seconds",
        "lapping_duration_sum_seconds",
        "random_duration_sum_seconds",
        "wandering_like_episode_count",
        "wandering_like_duration_sum_seconds",
        "night_wandering_like_duration_sum_seconds",
        "night_wandering_like_ratio",
        "uncertain_episode_count",
        "uncertain_ratio",
        "duration_semantics",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "primary_seed",
        "best_epoch",
        "four_class_order",
        "subtype_order",
        "binary_class_order",
        "binary_decision_threshold",
        "probability_calibrated",
        "episode_policy_status",
        "episode_merge_gap_seconds",
        "eligible_for_baseline",
        "baseline_status",
        "baseline_deviation",
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
        "algorithm_event_status",
        "algorithm_event_emitted",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "product_stage",
        "status",
        "evidence_scope",
        "validation_scope",
        "binding_evidence_scope",
        "artifacts",
        "binding_manifest",
        "input_product_manifests",
        "input_product_count",
        "input_session_count",
        "synthetic_person_count",
        "local_date_count",
        "daily_summary_count",
        "night_window",
        "mental_health_config_sha256",
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "primary_seed",
        "best_epoch",
        "four_class_order",
        "subtype_order",
        "binary_class_order",
        "binary_decision_threshold",
        "probability_calibrated",
        "episode_policy_status",
        "episode_merge_gap_seconds",
        "duration_semantics",
        "builder_source_sha256",
        "person_binding_verified",
        "eligible_for_baseline",
        "baseline_emitted",
        "risk_or_alert_decision_emitted",
        "algorithm_event_emitted",
        "real_human_media_consumed",
        "m0cam_d_started",
    }
)
_FINAL_TOP_LEVEL = frozenset({"daily_summary.jsonl", "manifest.json"})
_FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "risk",
        "risk_level",
        "risk_score",
        "diagnosis",
        "medical_diagnosis",
        "action",
        "recommended_action",
        "alert_decision",
        "algorithm_event",
        "AlgorithmEvent",
    }
)
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SYNTHETIC_PERSON_TOKEN = re.compile(r"^SYN-[A-Z0-9]+(?:-[A-Z0-9]+)*$")


class WanderingDailySummaryError(ValueError):
    """A product, binding, time, or daily-summary contract failed closed."""


@dataclass(frozen=True)
class WanderingDailySummaryBuildResult:
    output_dir: Path
    manifest_sha256: str
    input_product_count: int
    session_count: int
    person_count: int
    local_date_count: int
    summary_count: int


@dataclass(frozen=True)
class _Session:
    product: ValidatedWanderingCameraProduct
    binding_id: str
    session_id: str
    start_epoch: float
    timezone_name: str
    timezone: ZoneInfo
    source_scope: tuple[str, ...]
    people: tuple[tuple[str, tuple[int, ...], tuple[tuple[float, float], ...]], ...]


@dataclass
class _Day:
    synthetic_person_id: str
    local_date: date
    timezone_name: str
    timezone: ZoneInfo
    presence: list[tuple[float, float]] = field(default_factory=list)
    windows: list[tuple[float, float]] = field(default_factory=list)
    windows_by_status: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: {name: [] for name in WINDOW_STATUSES}
    )
    product_ids: set[str] = field(default_factory=set)
    session_ids: set[str] = field(default_factory=set)
    source_scopes: set[tuple[str, ...]] = field(default_factory=set)
    setup_ids: set[tuple[str, str]] = field(default_factory=set)
    track_ids: set[tuple[str, int]] = field(default_factory=set)
    window_ids: set[tuple[str, str]] = field(default_factory=set)
    reason_counts: Counter[str] = field(default_factory=Counter)
    quality_flag_counts: Counter[str] = field(default_factory=Counter)
    episode_counts: Counter[str] = field(default_factory=Counter)
    episode_durations: Counter[str] = field(default_factory=Counter)
    night_wandering_seconds: float = 0.0
    identities: dict[str, dict[str, Any]] = field(default_factory=dict)


def build_wandering_daily_summary(
    product_bundles: Sequence[str | Path],
    binding_manifest: str | Path,
    output_dir: str | Path,
) -> WanderingDailySummaryBuildResult:
    """Build a fresh canonical synthetic daily-summary bundle."""

    if isinstance(product_bundles, (str, bytes, Path)) or not product_bundles:
        raise WanderingDailySummaryError("at least one product bundle is required")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"daily summary output already exists: {output}")

    binding_path = Path(binding_manifest)
    binding_bytes = _read_bytes(binding_path, "binding manifest")
    binding = _load_canonical_json(binding_path, "binding manifest")
    products = _load_products(product_bundles)
    sessions = _validate_binding(binding, products)
    aggregation_config = load_aggregation_config(DEFAULT_CONFIG_PATH)
    config_bytes = _read_bytes(DEFAULT_CONFIG_PATH, "mental health config")
    days = _aggregate_days(
        sessions,
        night_start=aggregation_config.night_start_time,
        night_end=aggregation_config.night_end_time,
    )
    rows = [
        _finalize_day(
            day,
            night_start=aggregation_config.night_start,
            night_end=aggregation_config.night_end,
        )
        for day in days.values()
    ]
    rows.sort(key=lambda row: (row["local_date"], row["synthetic_person_id"], row["timezone"]))
    summary_bytes = _canonical_jsonl(rows, "daily summary")
    global_identity = _common_identity(products)
    product_descriptors = sorted(
        (
            {
                "byte_count": len(product.manifest_bytes),
                "sha256": product.manifest_sha256,
            }
            for product in products.values()
        ),
        key=lambda descriptor: str(descriptor["sha256"]),
    )
    manifest = {
        "schema_version": DAILY_SUMMARY_MANIFEST_SCHEMA_VERSION,
        "product_name": PRODUCT_NAME,
        "product_stage": PRODUCT_STAGE,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "binding_evidence_scope": BINDING_EVIDENCE_SCOPE,
        "artifacts": {"daily_summary.jsonl": _descriptor(summary_bytes)},
        "binding_manifest": _descriptor(binding_bytes),
        "input_product_manifests": product_descriptors,
        "input_product_count": len(products),
        "input_session_count": len(sessions),
        "synthetic_person_count": len({row["synthetic_person_id"] for row in rows}),
        "local_date_count": len({row["local_date"] for row in rows}),
        "daily_summary_count": len(rows),
        "night_window": {
            "start": aggregation_config.night_start,
            "end": aggregation_config.night_end,
        },
        "mental_health_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        **global_identity,
        "duration_semantics": DURATION_SEMANTICS,
        "builder_source_sha256": _sha256_file(Path(__file__)),
        "person_binding_verified": False,
        "eligible_for_baseline": False,
        "baseline_emitted": False,
        "risk_or_alert_decision_emitted": False,
        "algorithm_event_emitted": False,
        "real_human_media_consumed": False,
        "m0cam_d_started": False,
    }
    manifest_bytes = _canonical_json(manifest, "daily summary manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _write_new_file(staging / "daily_summary.jsonl", summary_bytes)
        _write_new_file(staging / "manifest.json", manifest_bytes)
        _verify_staging(staging, summary_bytes, manifest_bytes, manifest)
        if output.exists():
            raise FileExistsError(f"daily summary output already exists: {output}")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return WanderingDailySummaryBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        input_product_count=len(products),
        session_count=len(sessions),
        person_count=len({row["synthetic_person_id"] for row in rows}),
        local_date_count=len({row["local_date"] for row in rows}),
        summary_count=len(rows),
    )


def _load_products(
    product_bundles: Sequence[str | Path],
) -> dict[str, ValidatedWanderingCameraProduct]:
    products: dict[str, ValidatedWanderingCameraProduct] = {}
    for path in product_bundles:
        try:
            product = load_validated_wandering_camera_product(path)
        except (WanderingCameraProductError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WanderingDailySummaryError(f"product bundle failed full MVP-1 validation: {path}") from exc
        if product.manifest_sha256 in products:
            raise WanderingDailySummaryError("duplicate product manifest input")
        products[product.manifest_sha256] = product
    return products


def _validate_binding(
    binding: Mapping[str, Any],
    products: Mapping[str, ValidatedWanderingCameraProduct],
) -> tuple[_Session, ...]:
    _require_exact_fields(binding, _BINDING_FIELDS, "binding manifest")
    if binding.get("schema_version") != DAILY_BINDING_SCHEMA_VERSION:
        raise WanderingDailySummaryError("binding manifest schema has drifted")
    if binding.get("evidence_scope") != BINDING_EVIDENCE_SCOPE:
        raise WanderingDailySummaryError("binding manifest evidence scope has drifted")
    raw_sessions = binding.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise WanderingDailySummaryError("binding manifest sessions must be non-empty")

    sessions: list[_Session] = []
    binding_ids: set[str] = set()
    session_ids: set[str] = set()
    product_ids: set[str] = set()
    for raw in raw_sessions:
        _require_exact_fields(raw, _BINDING_SESSION_FIELDS, "binding session")
        binding_id = _token(raw.get("binding_id"), "binding_id")
        session_id = _token(raw.get("session_id"), "session_id")
        if binding_id in binding_ids or session_id in session_ids:
            raise WanderingDailySummaryError("duplicate binding_id or session_id")
        binding_ids.add(binding_id)
        session_ids.add(session_id)
        product_digest = raw.get("product_manifest_sha256")
        _validate_digest(product_digest, "binding product manifest")
        if product_digest in product_ids:
            raise WanderingDailySummaryError("duplicate product binding")
        product_ids.add(str(product_digest))
        product = products.get(str(product_digest))
        if product is None:
            raise WanderingDailySummaryError("binding product hash does not match an input product")

        source_scope_value = raw.get("source_scope")
        _require_exact_fields(source_scope_value, _SOURCE_FIELD_SET, "binding source_scope")
        source_scope = tuple(
            _nonempty_string(source_scope_value.get(name), f"source_scope.{name}")
            for name in _SOURCE_FIELDS
        )
        expected_scope = tuple(str(product.evidence["source_scope"][name]) for name in _SOURCE_FIELDS)
        if source_scope != expected_scope:
            raise WanderingDailySummaryError("binding source_scope does not match product")

        start_epoch, zone_name, zone = _session_time(raw)
        raw_people = raw.get("person_tracks")
        if not isinstance(raw_people, list) or not raw_people:
            raise WanderingDailySummaryError("binding person_tracks must be non-empty")
        people: list[tuple[str, tuple[int, ...], tuple[tuple[float, float], ...]]] = []
        track_owner: dict[int, str] = {}
        person_ids: set[str] = set()
        for raw_person in raw_people:
            _require_exact_fields(raw_person, _BINDING_PERSON_FIELDS, "binding person track")
            person_id = raw_person.get("synthetic_person_id")
            if not isinstance(person_id, str) or not _SYNTHETIC_PERSON_TOKEN.fullmatch(person_id):
                raise WanderingDailySummaryError("synthetic_person_id is not a safe synthetic token")
            if person_id in person_ids:
                raise WanderingDailySummaryError("duplicate synthetic_person_id in one session")
            person_ids.add(person_id)
            raw_track_ids = raw_person.get("track_ids")
            if (
                not isinstance(raw_track_ids, list)
                or not raw_track_ids
                or any(
                    not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0
                    for track_id in raw_track_ids
                )
                or len(set(raw_track_ids)) != len(raw_track_ids)
            ):
                raise WanderingDailySummaryError("binding track_ids are invalid or duplicated")
            for track_id in raw_track_ids:
                if track_id in track_owner:
                    raise WanderingDailySummaryError("track is bound across synthetic people")
                track_owner[track_id] = person_id
            presence = _presence_intervals(raw_person.get("presence_intervals_sec"))
            people.append((person_id, tuple(raw_track_ids), tuple(presence)))

        product_tracks = {
            int(row["track_id"])
            for rows in (product.tracking, product.tracklets, product.windows, product.predictions, product.episodes)
            for row in rows
        }
        if not product_tracks or set(track_owner) != product_tracks:
            raise WanderingDailySummaryError("every product track must be bound exactly once")
        presence_by_track = {
            track_id: presence
            for _, track_ids, presence in people
            for track_id in track_ids
        }
        for role, rows, start_name, end_name in (
            ("window", product.predictions, "window_start_sec", "window_end_sec"),
            ("episode", product.episodes, "episode_start_sec", "episode_end_sec_exclusive"),
        ):
            for row in rows:
                interval = _relative_interval(row.get(start_name), row.get(end_name), role)
                if not _fully_covered(interval, presence_by_track[int(row["track_id"])]):
                    raise WanderingDailySummaryError(f"{role} interval is outside explicit presence")
        sessions.append(
            _Session(
                product=product,
                binding_id=binding_id,
                session_id=session_id,
                start_epoch=start_epoch,
                timezone_name=zone_name,
                timezone=zone,
                source_scope=source_scope,
                people=tuple(people),
            )
        )

    if product_ids != set(products):
        raise WanderingDailySummaryError("binding sessions do not exactly cover input products")
    return tuple(sessions)


def _session_time(raw: Mapping[str, Any]) -> tuple[float, str, ZoneInfo]:
    started_at = raw.get("session_started_at")
    if not isinstance(started_at, str) or not started_at:
        raise WanderingDailySummaryError("session_started_at is required")
    try:
        parsed = datetime.fromisoformat(started_at)
    except ValueError as exc:
        raise WanderingDailySummaryError("session_started_at is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WanderingDailySummaryError("session_started_at must include a UTC offset")
    zone_name = raw.get("timezone")
    if not isinstance(zone_name, str) or not zone_name:
        raise WanderingDailySummaryError("timezone is required")
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError as exc:
        raise WanderingDailySummaryError("timezone is not a known IANA zone") from exc
    observed = parsed.astimezone(zone)
    if (
        observed.replace(tzinfo=None) != parsed.replace(tzinfo=None)
        or observed.utcoffset() != parsed.utcoffset()
    ):
        raise WanderingDailySummaryError("session offset and timezone do not describe the same local instant")
    return parsed.timestamp(), zone_name, zone


def _presence_intervals(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not value:
        raise WanderingDailySummaryError("presence intervals must be non-empty")
    intervals: list[tuple[float, float]] = []
    for raw in value:
        _require_exact_fields(raw, _PRESENCE_FIELDS, "binding presence interval")
        intervals.append(
            _relative_interval(raw.get("start_sec"), raw.get("end_sec_exclusive"), "presence")
        )
    merged = _merge_intervals(intervals)
    if _duration(merged) <= 0.0:
        raise WanderingDailySummaryError("presence_seconds must be positive")
    return merged


def _aggregate_days(
    sessions: Sequence[_Session],
    *,
    night_start: time,
    night_end: time,
) -> dict[tuple[str, date, str], _Day]:
    days: dict[tuple[str, date, str], _Day] = {}
    person_date_zones: dict[tuple[str, date], str] = {}
    for session in sessions:
        product = session.product
        identity = _identity(product)
        person_by_track = {
            track_id: person_id
            for person_id, track_ids, _ in session.people
            for track_id in track_ids
        }
        for person_id, track_ids, presence_relative in session.people:
            presence_absolute = [
                (session.start_epoch + start, session.start_epoch + end)
                for start, end in presence_relative
            ]
            for start, end in presence_absolute:
                for local_day, segment_start, segment_end in _split_days(
                    start, end, session.timezone
                ):
                    day = _day_for(
                        days,
                        person_date_zones,
                        person_id,
                        local_day,
                        session.timezone_name,
                        session.timezone,
                    )
                    day.presence.append((segment_start, segment_end))
                    _include_session(day, session, track_ids, identity)

        for prediction in product.predictions:
            track_id = int(prediction["track_id"])
            person_id = person_by_track[track_id]
            start = session.start_epoch + float(prediction["window_start_sec"])
            end = session.start_epoch + float(prediction["window_end_sec"])
            for local_day, segment_start, segment_end in _split_days(start, end, session.timezone):
                day = _day_for(
                    days,
                    person_date_zones,
                    person_id,
                    local_day,
                    session.timezone_name,
                    session.timezone,
                )
                day.windows.append((segment_start, segment_end))
                status = str(prediction["window_status"])
                day.windows_by_status[status].append((segment_start, segment_end))
                day.window_ids.add((product.manifest_sha256, str(prediction["window_id"])))
                day.reason_counts.update(str(value) for value in prediction["reason_codes"])
                day.quality_flag_counts.update(str(value) for value in prediction["quality_flags"])
                _include_session(day, session, (track_id,), identity)

        for episode in product.episodes:
            track_id = int(episode["track_id"])
            person_id = person_by_track[track_id]
            pattern = str(episode["predicted_pattern"])
            start = session.start_epoch + float(episode["episode_start_sec"])
            end = session.start_epoch + float(episode["episode_end_sec_exclusive"])
            start_day = datetime.fromtimestamp(start, tz=session.timezone).date()
            count_day = _day_for(
                days,
                person_date_zones,
                person_id,
                start_day,
                session.timezone_name,
                session.timezone,
            )
            count_day.episode_counts[pattern] += 1
            _include_session(count_day, session, (track_id,), identity)
            for local_day, segment_start, segment_end in _split_days(start, end, session.timezone):
                day = _day_for(
                    days,
                    person_date_zones,
                    person_id,
                    local_day,
                    session.timezone_name,
                    session.timezone,
                )
                duration = segment_end - segment_start
                day.episode_durations[pattern] += duration
                if pattern != "direct":
                    night = _night_intervals(local_day, session.timezone, night_start, night_end)
                    day.night_wandering_seconds += _duration(
                        _intersect([(segment_start, segment_end)], night)
                    )
                _include_session(day, session, (track_id,), identity)
    if not days:
        raise WanderingDailySummaryError("binding produced no positive person-day presence")
    return days


def _day_for(
    days: dict[tuple[str, date, str], _Day],
    person_date_zones: dict[tuple[str, date], str],
    person_id: str,
    local_day: date,
    timezone_name: str,
    timezone: ZoneInfo,
) -> _Day:
    person_date = (person_id, local_day)
    previous_zone = person_date_zones.get(person_date)
    if previous_zone is not None and previous_zone != timezone_name:
        raise WanderingDailySummaryError("one synthetic person-day cannot drift across timezones")
    person_date_zones[person_date] = timezone_name
    key = (person_id, local_day, timezone_name)
    if key not in days:
        days[key] = _Day(person_id, local_day, timezone_name, timezone)
    return days[key]


def _include_session(
    day: _Day,
    session: _Session,
    track_ids: Sequence[int],
    identity: dict[str, Any],
) -> None:
    day.product_ids.add(session.product.manifest_sha256)
    day.session_ids.add(session.session_id)
    day.source_scopes.add(session.source_scope)
    day.setup_ids.add((session.source_scope[2], session.source_scope[3]))
    day.track_ids.update((session.session_id, track_id) for track_id in track_ids)
    day.identities[session.product.manifest_sha256] = identity


def _finalize_day(day: _Day, *, night_start: str, night_end: str) -> dict[str, Any]:
    identities = list(day.identities.values())
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise WanderingDailySummaryError("candidate/model/class/episode identity drift within person-day")
    identity = identities[0]
    presence = _merge_intervals(day.presence)
    presence_seconds = _duration(presence)
    if presence_seconds <= 0.0:
        raise WanderingDailySummaryError("presence_seconds must be positive")
    any_windows = _intersect(_merge_intervals(day.windows), presence)
    status_intervals = {
        status: _intersect(_merge_intervals(day.windows_by_status[status]), presence)
        for status in WINDOW_STATUSES
    }
    overlap_parts: list[tuple[float, float]] = []
    for index, left in enumerate(WINDOW_STATUSES):
        for right in WINDOW_STATUSES[index + 1 :]:
            overlap_parts.extend(_intersect(status_intervals[left], status_intervals[right]))
    status_overlap = _duration(_merge_intervals(overlap_parts))
    quality_flags: set[str] = set()
    if status_overlap > 0.0:
        quality_flags.add("status_coverage_overlap")
    if not any_windows:
        quality_flags.add("no_window_coverage")
    quality_flags.update(day.quality_flag_counts)

    episode_counts = {name: int(day.episode_counts[name]) for name in FOUR_CLASS_ORDER}
    episode_durations = {name: float(day.episode_durations[name]) for name in FOUR_CLASS_ORDER}
    wandering_count = sum(episode_counts[name] for name in SUBTYPE_ORDER)
    wandering_duration = sum(episode_durations[name] for name in SUBTYPE_ORDER)
    night_ratio = day.night_wandering_seconds / wandering_duration if wandering_duration > 0 else None
    return {
        "schema_version": DAILY_SUMMARY_SCHEMA_VERSION,
        "product_name": PRODUCT_NAME,
        "product_stage": PRODUCT_STAGE,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "synthetic_person_id": day.synthetic_person_id,
        "person_binding_verified": False,
        "local_date": day.local_date.isoformat(),
        "timezone": day.timezone_name,
        "night_start": night_start,
        "night_end": night_end,
        "product_count": len(day.product_ids),
        "session_count": len(day.session_ids),
        "source_scope_count": len(day.source_scopes),
        "setup_count": len(day.setup_ids),
        "track_count": len(day.track_ids),
        "window_count": len(day.window_ids),
        "presence_seconds": _measure(presence_seconds),
        "any_window_covered_seconds": _measure(_duration(any_windows)),
        **{
            f"{status}_covered_seconds": _measure(_duration(status_intervals[status]))
            for status in WINDOW_STATUSES
        },
        "trajectory_coverage": _ratio(_duration(any_windows), presence_seconds),
        **{
            f"{status}_coverage": _ratio(_duration(status_intervals[status]), presence_seconds)
            for status in WINDOW_STATUSES
        },
        "status_overlap_seconds": _measure(status_overlap),
        "qc_reason_counts": {name: day.reason_counts[name] for name in sorted(day.reason_counts)},
        "quality_flag_counts": {
            name: day.quality_flag_counts[name] for name in sorted(day.quality_flag_counts)
        },
        "quality_flags": sorted(quality_flags),
        **{f"{name}_episode_count": episode_counts[name] for name in FOUR_CLASS_ORDER},
        **{
            f"{name}_duration_sum_seconds": _measure(episode_durations[name])
            for name in FOUR_CLASS_ORDER
        },
        "wandering_like_episode_count": wandering_count,
        "wandering_like_duration_sum_seconds": _measure(wandering_duration),
        "night_wandering_like_duration_sum_seconds": _measure(day.night_wandering_seconds),
        "night_wandering_like_ratio": _ratio(day.night_wandering_seconds, wandering_duration)
        if night_ratio is not None
        else None,
        "uncertain_episode_count": None,
        "uncertain_ratio": None,
        "duration_semantics": DURATION_SEMANTICS,
        **identity,
        "eligible_for_baseline": False,
        "baseline_status": "not_ready_synthetic_binding",
        "baseline_deviation": None,
        "risk_level": None,
        "risk_score": None,
        "recommended_action": None,
        "alert_decision": None,
        "medical_diagnosis": None,
        "algorithm_event": None,
        "algorithm_event_status": "not_ready_daily_summary_only",
        "algorithm_event_emitted": False,
    }


def _identity(product: ValidatedWanderingCameraProduct) -> dict[str, Any]:
    bindings = product.model_bindings
    execution = product.execution
    return {
        "candidate_id": bindings["candidate_id"],
        "candidate_manifest_sha256": bindings["candidate_manifest_sha256"],
        "model_state_sha256": bindings["model_state_sha256"],
        "primary_seed": bindings["primary_seed"],
        "best_epoch": bindings["best_epoch"],
        "four_class_order": list(bindings["four_class_order"]),
        "subtype_order": list(bindings["subtype_order"]),
        "binary_class_order": list(bindings["binary_class_order"]),
        "binary_decision_threshold": bindings["binary_decision_threshold"],
        "probability_calibrated": bindings["probability_calibrated"],
        "episode_policy_status": execution["episode_policy_status"],
        "episode_merge_gap_seconds": execution["episode_merge_gap_seconds"],
    }


def _common_identity(
    products: Mapping[str, ValidatedWanderingCameraProduct],
) -> dict[str, Any]:
    identities = [_identity(product) for product in products.values()]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise WanderingDailySummaryError("candidate/model/class/episode identity drift across products")
    return identities[0]


def _verify_staging(
    staging: Path,
    summary_bytes: bytes,
    manifest_bytes: bytes,
    expected_manifest: Mapping[str, Any],
) -> None:
    entries = {path.name: path for path in staging.iterdir()}
    if set(entries) != _FINAL_TOP_LEVEL or any(not path.is_file() for path in entries.values()):
        raise WanderingDailySummaryError("daily summary top-level collection has drifted")
    if (staging / "daily_summary.jsonl").read_bytes() != summary_bytes:
        raise WanderingDailySummaryError("daily summary staging bytes changed before commit")
    if (staging / "manifest.json").read_bytes() != manifest_bytes:
        raise WanderingDailySummaryError("daily summary manifest bytes changed before commit")
    rows = _load_canonical_jsonl(staging / "daily_summary.jsonl", "daily summary")
    manifest = _load_canonical_json(staging / "manifest.json", "daily summary manifest")
    for row in rows:
        _validate_summary_row(row)
    if list(rows) != sorted(
        rows, key=lambda row: (row["local_date"], row["synthetic_person_id"], row["timezone"])
    ):
        raise WanderingDailySummaryError("daily summary rows are not stably sorted")
    keys = [(row["synthetic_person_id"], row["local_date"], row["timezone"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise WanderingDailySummaryError("daily summary person-date-timezone key is duplicated")
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "daily summary manifest")
    if manifest != expected_manifest:
        raise WanderingDailySummaryError("daily summary manifest semantics have drifted")
    _require_exact_fields(manifest.get("night_window"), _NIGHT_WINDOW_FIELDS, "night window")
    _require_exact_fields(
        manifest.get("binding_manifest"),
        _DESCRIPTOR_FIELDS,
        "binding manifest descriptor",
    )
    product_descriptors = manifest.get("input_product_manifests")
    if not isinstance(product_descriptors, list) or any(
        not isinstance(descriptor, Mapping) or set(descriptor) != _DESCRIPTOR_FIELDS
        for descriptor in product_descriptors
    ):
        raise WanderingDailySummaryError("input product manifest descriptors are invalid")
    if product_descriptors != sorted(
        product_descriptors, key=lambda descriptor: str(descriptor["sha256"])
    ):
        raise WanderingDailySummaryError("input product manifest descriptors are not sorted")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"daily_summary.jsonl"}:
        raise WanderingDailySummaryError("daily summary manifest artifact set has drifted")
    _verify_descriptor(staging / "daily_summary.jsonl", artifacts["daily_summary.jsonl"])
    _reject_nonempty_decisions(rows, "daily summary")
    _reject_nonempty_decisions(manifest, "daily summary manifest")


def _validate_summary_row(row: Mapping[str, Any]) -> None:
    _require_exact_fields(row, _SUMMARY_FIELDS, "daily summary row")
    if (
        row.get("schema_version") != DAILY_SUMMARY_SCHEMA_VERSION
        or row.get("product_name") != PRODUCT_NAME
        or row.get("product_stage") != PRODUCT_STAGE
        or row.get("status") != STATUS
        or row.get("evidence_scope") != EVIDENCE_SCOPE
        or row.get("validation_scope") != VALIDATION_SCOPE
        or row.get("person_binding_verified") is not False
        or row.get("eligible_for_baseline") is not False
        or row.get("baseline_status") != "not_ready_synthetic_binding"
        or row.get("algorithm_event_status") != "not_ready_daily_summary_only"
        or row.get("algorithm_event_emitted") is not False
        or row.get("duration_semantics") != DURATION_SEMANTICS
        or row.get("four_class_order") != list(FOUR_CLASS_ORDER)
        or row.get("subtype_order") != list(SUBTYPE_ORDER)
        or row.get("binary_class_order") != list(BINARY_CLASS_ORDER)
        or row.get("probability_calibrated") is not False
        or row.get("uncertain_episode_count") is not None
        or row.get("uncertain_ratio") is not None
        or row.get("baseline_deviation") is not None
        or row.get("risk_level") is not None
        or row.get("risk_score") is not None
        or row.get("recommended_action") is not None
        or row.get("alert_decision") is not None
        or row.get("medical_diagnosis") is not None
        or row.get("algorithm_event") is not None
    ):
        raise WanderingDailySummaryError("daily summary row fixed contract has drifted")
    for name in ("qc_reason_counts", "quality_flag_counts"):
        counts = row.get(name)
        if (
            not isinstance(counts, Mapping)
            or list(counts) != sorted(counts)
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in counts.items()
            )
        ):
            raise WanderingDailySummaryError(f"daily summary {name} is invalid")
    flags = row.get("quality_flags")
    if not isinstance(flags, list) or flags != sorted(set(flags)) or any(
        not isinstance(flag, str) or not flag for flag in flags
    ):
        raise WanderingDailySummaryError("daily summary quality_flags are invalid")


def _relative_interval(start: Any, end: Any, role: str) -> tuple[float, float]:
    if not _finite_number(start) or not _finite_number(end):
        raise WanderingDailySummaryError(f"{role} interval must be finite")
    left = float(start)
    right = float(end)
    if left < 0.0 or right <= left:
        raise WanderingDailySummaryError(f"{role} interval is invalid")
    return left, right


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _intersect(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    left_merged = _merge_intervals(left)
    right_merged = _merge_intervals(right)
    output: list[tuple[float, float]] = []
    first = second = 0
    while first < len(left_merged) and second < len(right_merged):
        start = max(left_merged[first][0], right_merged[second][0])
        end = min(left_merged[first][1], right_merged[second][1])
        if end > start:
            output.append((start, end))
        if left_merged[first][1] <= right_merged[second][1]:
            first += 1
        else:
            second += 1
    return output


def _fully_covered(
    interval: tuple[float, float],
    coverage: Sequence[tuple[float, float]],
) -> bool:
    start, end = interval
    return any(left <= start and right >= end for left, right in coverage)


def _split_days(start: float, end: float, timezone: ZoneInfo) -> list[tuple[date, float, float]]:
    output: list[tuple[date, float, float]] = []
    cursor = start
    while cursor < end:
        local = datetime.fromtimestamp(cursor, tz=timezone)
        next_midnight = datetime.combine(
            local.date() + timedelta(days=1), time.min, tzinfo=timezone
        ).timestamp()
        segment_end = min(end, next_midnight)
        if segment_end <= cursor:
            raise WanderingDailySummaryError("cannot advance across timezone day boundary")
        output.append((local.date(), cursor, segment_end))
        cursor = segment_end
    return output


def _night_intervals(
    local_day: date,
    timezone: ZoneInfo,
    night_start: time,
    night_end: time,
) -> list[tuple[float, float]]:
    day_start = datetime.combine(local_day, time.min, tzinfo=timezone).timestamp()
    day_end = datetime.combine(local_day + timedelta(days=1), time.min, tzinfo=timezone).timestamp()
    if night_start < night_end:
        return [
            (
                datetime.combine(local_day, night_start, tzinfo=timezone).timestamp(),
                datetime.combine(local_day, night_end, tzinfo=timezone).timestamp(),
            )
        ]
    return [
        (day_start, datetime.combine(local_day, night_end, tzinfo=timezone).timestamp()),
        (datetime.combine(local_day, night_start, tzinfo=timezone).timestamp(), day_end),
    ]


def _duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _measure(value: float) -> float:
    return round(float(value), 6)


def _ratio(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6)


def _token(value: Any, role: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise WanderingDailySummaryError(f"{role} is not a safe token")
    return value


def _nonempty_string(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise WanderingDailySummaryError(f"{role} must be a non-empty string")
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_exact_fields(value: Any, fields: frozenset[str], role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WanderingDailySummaryError(f"{role} field set has drifted")


def _validate_digest(value: Any, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WanderingDailySummaryError(f"{role} is not a SHA-256 digest")


def _descriptor(payload: bytes) -> dict[str, Any]:
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _verify_descriptor(path: Path, value: Any) -> None:
    _require_exact_fields(value, _DESCRIPTOR_FIELDS, f"{path.name} descriptor")
    payload = _read_bytes(path, path.name)
    if value != _descriptor(payload):
        raise WanderingDailySummaryError(f"{path.name} descriptor does not match bytes")


def _load_canonical_json(path: Path, role: str) -> dict[str, Any]:
    raw = _read_bytes(path, role)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingDailySummaryError(f"cannot parse {role}") from exc
    if not isinstance(value, dict) or raw != _canonical_json(value, role):
        raise WanderingDailySummaryError(f"{role} is not canonical JSON")
    return value


def _load_canonical_jsonl(path: Path, role: str) -> tuple[dict[str, Any], ...]:
    raw = _read_bytes(path, role)
    try:
        rows = tuple(json.loads(line.decode("utf-8")) for line in raw.splitlines())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingDailySummaryError(f"cannot parse {role}") from exc
    if any(not isinstance(row, dict) for row in rows) or raw != _canonical_jsonl(rows, role):
        raise WanderingDailySummaryError(f"{role} is not canonical JSONL")
    return rows


def _canonical_json(value: Any, role: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except CameraAdapterError as exc:
        raise WanderingDailySummaryError(f"{role} is not finite JSON") from exc


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]], role: str) -> bytes:
    try:
        return canonical_jsonl_bytes(rows)
    except CameraAdapterError as exc:
        raise WanderingDailySummaryError(f"{role} is not finite JSONL") from exc


def _read_bytes(path: Path, role: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WanderingDailySummaryError(f"cannot read {role}") from exc


def _reject_nonempty_decisions(value: Any, role: str) -> None:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            if name in _FORBIDDEN_DECISION_FIELDS and nested is not None:
                raise WanderingDailySummaryError(f"{role} contains non-empty decision field {name}")
            _reject_nonempty_decisions(nested, role)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonempty_decisions(nested, role)


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DAILY_BINDING_SCHEMA_VERSION",
    "DAILY_SUMMARY_MANIFEST_SCHEMA_VERSION",
    "DAILY_SUMMARY_SCHEMA_VERSION",
    "WanderingDailySummaryBuildResult",
    "WanderingDailySummaryError",
    "build_wandering_daily_summary",
]

# === M0-CAM-MVP-3S READ-ONLY LOADER EXTENSION ===
# The MVP-2S producer manifest historically binds the SHA-256 of the source bytes
# before this consumer-only extension.  Preserve that frozen producer identity so
# adding full final readback does not change any fresh MVP-2S output bytes.
MVP2S_PRODUCER_SOURCE_SHA256 = (
    "f9b2b29e560c557dfa7d356fbccbd907113476e22e2fe7b67013ab1aa80e9ca3"
)
_MVP3S_LOADER_MARKER = b"# === M0-CAM-MVP-3S READ-ONLY LOADER EXTENSION ===\n"
_MVP3S_LOADER_BOUNDARY = b"]\n\n"
_MVP2S_VERIFY_STAGING = _verify_staging


@dataclass(frozen=True)
class ValidatedWanderingDailySummary:
    bundle_dir: Path
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    daily_summary_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str


def load_validated_wandering_daily_summary(
    path: str | Path,
) -> ValidatedWanderingDailySummary:
    """Fully validate and load one canonical MVP-2S final without mutating it."""

    bundle = Path(path)
    try:
        if not bundle.is_dir():
            raise WanderingDailySummaryError("daily summary bundle is not a directory")
        entries = {entry.name: entry for entry in bundle.iterdir()}
    except OSError as exc:
        raise WanderingDailySummaryError("cannot inspect daily summary bundle") from exc
    if set(entries) != _FINAL_TOP_LEVEL or any(
        not entry.is_file() for entry in entries.values()
    ):
        raise WanderingDailySummaryError("daily summary top-level collection has drifted")

    summary_path = bundle / "daily_summary.jsonl"
    manifest_path = bundle / "manifest.json"
    rows = _load_canonical_jsonl(summary_path, "daily summary")
    manifest = _load_canonical_json(manifest_path, "daily summary manifest")
    summary_bytes = _read_bytes(summary_path, "daily summary")
    manifest_bytes = _read_bytes(manifest_path, "daily summary manifest")
    _validate_loaded_daily_summary(bundle, rows, manifest)
    return ValidatedWanderingDailySummary(
        bundle_dir=bundle.resolve(),
        rows=rows,
        manifest=manifest,
        daily_summary_bytes=summary_bytes,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _validate_loaded_daily_summary(
    bundle: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if not rows:
        raise WanderingDailySummaryError("daily summary must contain at least one row")
    for row in rows:
        if (
            row.get("four_class_order") != list(FOUR_CLASS_ORDER)
            or row.get("subtype_order") != list(SUBTYPE_ORDER)
            or row.get("binary_class_order") != list(BINARY_CLASS_ORDER)
        ):
            raise WanderingDailySummaryError(
                "baseline_version_reset_required: daily class order has drifted"
            )
        if row.get("probability_calibrated") is not False:
            raise WanderingDailySummaryError(
                "baseline_version_reset_required: daily calibration status has drifted"
            )
        if row.get("duration_semantics") != DURATION_SEMANTICS:
            raise WanderingDailySummaryError(
                "baseline_version_reset_required: daily duration semantics has drifted"
            )
        _validate_summary_row(row)
        _validate_summary_row_semantics(row)
    expected_order = sorted(
        rows,
        key=lambda row: (
            str(row["local_date"]),
            str(row["synthetic_person_id"]),
            str(row["timezone"]),
        ),
    )
    if list(rows) != expected_order:
        raise WanderingDailySummaryError("daily summary rows are not stably sorted")
    keys = [
        (str(row["synthetic_person_id"]), str(row["local_date"]), str(row["timezone"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise WanderingDailySummaryError("daily summary person-date-timezone key is duplicated")

    _require_exact_fields(manifest, _MANIFEST_FIELDS, "daily summary manifest")
    if (
        manifest.get("schema_version") != DAILY_SUMMARY_MANIFEST_SCHEMA_VERSION
        or manifest.get("product_name") != PRODUCT_NAME
        or manifest.get("product_stage") != PRODUCT_STAGE
        or manifest.get("status") != STATUS
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("validation_scope") != VALIDATION_SCOPE
        or manifest.get("binding_evidence_scope") != BINDING_EVIDENCE_SCOPE
        or manifest.get("person_binding_verified") is not False
        or manifest.get("eligible_for_baseline") is not False
        or manifest.get("baseline_emitted") is not False
        or manifest.get("risk_or_alert_decision_emitted") is not False
        or manifest.get("algorithm_event_emitted") is not False
        or manifest.get("real_human_media_consumed") is not False
        or manifest.get("m0cam_d_started") is not False
        or manifest.get("duration_semantics") != DURATION_SEMANTICS
        or manifest.get("four_class_order") != list(FOUR_CLASS_ORDER)
        or manifest.get("subtype_order") != list(SUBTYPE_ORDER)
        or manifest.get("binary_class_order") != list(BINARY_CLASS_ORDER)
        or manifest.get("probability_calibrated") is not False
        or manifest.get("builder_source_sha256") != MVP2S_PRODUCER_SOURCE_SHA256
    ):
        raise WanderingDailySummaryError("daily summary manifest fixed contract has drifted")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"daily_summary.jsonl"}:
        raise WanderingDailySummaryError("daily summary manifest artifact set has drifted")
    _verify_descriptor(bundle / "daily_summary.jsonl", artifacts["daily_summary.jsonl"])
    _validate_descriptor_value(manifest.get("binding_manifest"), "binding manifest")
    inputs = manifest.get("input_product_manifests")
    if not isinstance(inputs, list) or not inputs:
        raise WanderingDailySummaryError("input product manifest descriptors are invalid")
    for descriptor in inputs:
        _validate_descriptor_value(descriptor, "input product manifest")
    if inputs != sorted(inputs, key=lambda descriptor: str(descriptor["sha256"])):
        raise WanderingDailySummaryError("input product manifest descriptors are not sorted")
    input_digests = [str(descriptor["sha256"]) for descriptor in inputs]
    if len(set(input_digests)) != len(input_digests):
        raise WanderingDailySummaryError("input product manifest descriptor is duplicated")

    count_fields = (
        "input_product_count",
        "input_session_count",
        "synthetic_person_count",
        "local_date_count",
        "daily_summary_count",
    )
    for field_name in count_fields:
        _nonnegative_integer(manifest.get(field_name), f"daily summary manifest {field_name}")
    if (
        manifest["input_product_count"] != len(inputs)
        or manifest["daily_summary_count"] != len(rows)
        or manifest["synthetic_person_count"]
        != len({str(row["synthetic_person_id"]) for row in rows})
        or manifest["local_date_count"] != len({str(row["local_date"]) for row in rows})
    ):
        raise WanderingDailySummaryError("daily summary manifest count has drifted")

    night_window = manifest.get("night_window")
    _require_exact_fields(night_window, _NIGHT_WINDOW_FIELDS, "night window")
    _clock_string(night_window.get("start"), "night window start")
    _clock_string(night_window.get("end"), "night window end")
    if night_window["start"] == night_window["end"]:
        raise WanderingDailySummaryError("night window boundaries cannot match")
    for row in rows:
        if (
            row["night_start"] != night_window["start"]
            or row["night_end"] != night_window["end"]
        ):
            raise WanderingDailySummaryError("daily summary night window has drifted")

    identity_fields = (
        "candidate_id",
        "candidate_manifest_sha256",
        "model_state_sha256",
        "primary_seed",
        "best_epoch",
        "four_class_order",
        "subtype_order",
        "binary_class_order",
        "binary_decision_threshold",
        "probability_calibrated",
        "episode_policy_status",
        "episode_merge_gap_seconds",
    )
    for row in rows:
        if any(row[field_name] != manifest[field_name] for field_name in identity_fields):
            raise WanderingDailySummaryError("daily summary model or policy identity has drifted")
    _validate_digest(manifest.get("mental_health_config_sha256"), "mental health config")
    _validate_digest(manifest.get("builder_source_sha256"), "daily summary builder source")
    active_producer_sha256 = _sha256_file(Path(__file__))
    if manifest.get("builder_source_sha256") != active_producer_sha256:
        raise WanderingDailySummaryError(
            "producer_source_identity_mismatch: daily summary builder source has drifted"
        )
    _reject_nonempty_decisions(rows, "daily summary")
    _reject_nonempty_decisions(manifest, "daily summary manifest")


def _validate_summary_row_semantics(row: Mapping[str, Any]) -> None:
    person_id = row.get("synthetic_person_id")
    if not isinstance(person_id, str) or not _SYNTHETIC_PERSON_TOKEN.fullmatch(person_id):
        raise WanderingDailySummaryError("daily summary synthetic_person_id is invalid")
    local_date = row.get("local_date")
    if not isinstance(local_date, str):
        raise WanderingDailySummaryError("daily summary local_date is invalid")
    try:
        if date.fromisoformat(local_date).isoformat() != local_date:
            raise ValueError
    except ValueError as exc:
        raise WanderingDailySummaryError("daily summary local_date is invalid") from exc
    timezone_name = row.get("timezone")
    if not isinstance(timezone_name, str):
        raise WanderingDailySummaryError("daily summary timezone is invalid")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise WanderingDailySummaryError("daily summary timezone is invalid") from exc
    _clock_string(row.get("night_start"), "daily summary night_start")
    _clock_string(row.get("night_end"), "daily summary night_end")

    for field_name in (
        "product_count",
        "session_count",
        "source_scope_count",
        "setup_count",
        "track_count",
        "window_count",
        "direct_episode_count",
        "pacing_episode_count",
        "lapping_episode_count",
        "random_episode_count",
        "wandering_like_episode_count",
    ):
        _nonnegative_integer(row.get(field_name), f"daily summary {field_name}")
    if any(row[field_name] <= 0 for field_name in ("product_count", "session_count", "track_count")):
        raise WanderingDailySummaryError("daily summary source counts must be positive")

    numeric_fields = (
        "presence_seconds",
        "any_window_covered_seconds",
        "ready_covered_seconds",
        "unavailable_covered_seconds",
        "inference_error_covered_seconds",
        "status_overlap_seconds",
        "direct_duration_sum_seconds",
        "pacing_duration_sum_seconds",
        "lapping_duration_sum_seconds",
        "random_duration_sum_seconds",
        "wandering_like_duration_sum_seconds",
        "night_wandering_like_duration_sum_seconds",
        "binary_decision_threshold",
        "episode_merge_gap_seconds",
    )
    for field_name in numeric_fields:
        if not _finite_number(row.get(field_name)) or float(row[field_name]) < 0.0:
            raise WanderingDailySummaryError(f"daily summary {field_name} is invalid")
    presence = float(row["presence_seconds"])
    if presence <= 0.0:
        raise WanderingDailySummaryError("daily summary presence_seconds must be positive")
    for field_name in (
        "any_window_covered_seconds",
        "ready_covered_seconds",
        "unavailable_covered_seconds",
        "inference_error_covered_seconds",
    ):
        if float(row[field_name]) > presence:
            raise WanderingDailySummaryError(f"daily summary {field_name} exceeds presence")

    expected_ratios = {
        "trajectory_coverage": _ratio(float(row["any_window_covered_seconds"]), presence),
        "ready_coverage": _ratio(float(row["ready_covered_seconds"]), presence),
        "unavailable_coverage": _ratio(float(row["unavailable_covered_seconds"]), presence),
        "inference_error_coverage": _ratio(
            float(row["inference_error_covered_seconds"]), presence
        ),
    }
    for field_name, expected in expected_ratios.items():
        if not _finite_number(row.get(field_name)) or float(row[field_name]) != expected:
            raise WanderingDailySummaryError(f"daily summary {field_name} has drifted")

    wandering_count = sum(int(row[f"{name}_episode_count"]) for name in SUBTYPE_ORDER)
    wandering_duration = sum(float(row[f"{name}_duration_sum_seconds"]) for name in SUBTYPE_ORDER)
    if row["wandering_like_episode_count"] != wandering_count or float(
        row["wandering_like_duration_sum_seconds"]
    ) != _measure(wandering_duration):
        raise WanderingDailySummaryError("daily summary wandering-like aggregate has drifted")
    night_duration = float(row["night_wandering_like_duration_sum_seconds"])
    if night_duration > wandering_duration:
        raise WanderingDailySummaryError("daily summary night duration exceeds wandering duration")
    expected_night_ratio = (
        _ratio(night_duration, wandering_duration) if wandering_duration > 0.0 else None
    )
    if row.get("night_wandering_like_ratio") != expected_night_ratio:
        raise WanderingDailySummaryError("daily summary night ratio has drifted")

    _validate_digest(row.get("candidate_manifest_sha256"), "candidate manifest")
    _validate_digest(row.get("model_state_sha256"), "model state")
    _nonnegative_integer(row.get("primary_seed"), "daily summary primary_seed")
    _nonnegative_integer(row.get("best_epoch"), "daily summary best_epoch")
    if not isinstance(row.get("candidate_id"), str) or not row["candidate_id"]:
        raise WanderingDailySummaryError("daily summary candidate_id is invalid")
    if not isinstance(row.get("episode_policy_status"), str) or not row[
        "episode_policy_status"
    ]:
        raise WanderingDailySummaryError("daily summary episode policy is invalid")
    threshold = float(row["binary_decision_threshold"])
    if threshold > 1.0:
        raise WanderingDailySummaryError("daily summary binary threshold is invalid")


def _validate_descriptor_value(value: Any, role: str) -> None:
    _require_exact_fields(value, _DESCRIPTOR_FIELDS, f"{role} descriptor")
    byte_count = value.get("byte_count")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise WanderingDailySummaryError(f"{role} byte_count is invalid")
    _validate_digest(value.get("sha256"), role)


def _nonnegative_integer(value: Any, role: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WanderingDailySummaryError(f"{role} must be a non-negative integer")
    return value


def _clock_string(value: Any, role: str) -> str:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise WanderingDailySummaryError(f"{role} must use HH:MM")
    try:
        if time.fromisoformat(value).strftime("%H:%M") != value:
            raise ValueError
    except ValueError as exc:
        raise WanderingDailySummaryError(f"{role} must use HH:MM") from exc
    return value


def _sha256_file(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise WanderingDailySummaryError(f"cannot hash source file: {path}") from exc
    if path.resolve() == Path(__file__).resolve():
        return _validated_mvp2s_producer_source_sha256(payload)
    return hashlib.sha256(payload).hexdigest()


def _validated_mvp2s_producer_source_sha256(payload: bytes) -> str:
    if payload.count(_MVP3S_LOADER_MARKER) != 1:
        raise WanderingDailySummaryError(
            "producer_source_identity_mismatch: MVP-2S producer marker count has drifted"
        )
    marker_index = payload.index(_MVP3S_LOADER_MARKER)
    if payload[marker_index - len(_MVP3S_LOADER_BOUNDARY) : marker_index] != (
        _MVP3S_LOADER_BOUNDARY
    ):
        raise WanderingDailySummaryError(
            "producer_source_identity_mismatch: MVP-2S producer boundary has drifted"
        )
    producer_prefix = payload[: marker_index - 1]
    producer_sha256 = hashlib.sha256(producer_prefix).hexdigest()
    if producer_sha256 != MVP2S_PRODUCER_SOURCE_SHA256:
        raise WanderingDailySummaryError(
            "producer_source_identity_mismatch: MVP-2S producer prefix bytes have drifted"
        )
    return producer_sha256


def _verify_staging(
    staging: Path,
    summary_bytes: bytes,
    manifest_bytes: bytes,
    expected_manifest: Mapping[str, Any],
) -> None:
    _MVP2S_VERIFY_STAGING(staging, summary_bytes, manifest_bytes, expected_manifest)
    rows = _load_canonical_jsonl(staging / "daily_summary.jsonl", "daily summary")
    manifest = _load_canonical_json(staging / "manifest.json", "daily summary manifest")
    _validate_loaded_daily_summary(staging, rows, manifest)


__all__.extend(
    [
        "MVP2S_PRODUCER_SOURCE_SHA256",
        "ValidatedWanderingDailySummary",
        "load_validated_wandering_daily_summary",
    ]
)
