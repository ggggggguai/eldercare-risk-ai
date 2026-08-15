"""Synthetic baseline-profile previews from fully validated MVP-2S bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from elderly_monitoring.modules.mental_health.config import (
    DEFAULT_CONFIG_PATH,
    BaselineConfig,
    load_mental_health_config,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_summary import (
    ValidatedWanderingDailySummary,
    WanderingDailySummaryError,
    load_validated_wandering_daily_summary,
)


BASELINE_PREVIEW_SCHEMA_VERSION = "wandering-baseline-preview-v2"
BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION = "wandering-baseline-preview-manifest-v2"
REFERENCE_DAY_CONTRACT_VERSION = "wandering-baseline-reference-days-v1"
PRODUCT_NAME = "WanderingBaselineProfilePreview"
PRODUCT_STAGE = "synthetic_baseline_profile_preview"
STATUS = "wandering_m0cam_synthetic_baseline_profile_preview_ready"
EVIDENCE_SCOPE = "synthetic_contract_only"
VALIDATION_SCOPE = "synthetic_camera_contract"
QUANTILE_METHOD = "linear_interpolation_position_(n-1)*p"
METRIC_ORDER = (
    "trajectory_coverage",
    "presence_hours",
    "direct_episode_count_per_presence_hour",
    "pacing_episode_count_per_presence_hour",
    "lapping_episode_count_per_presence_hour",
    "random_episode_count_per_presence_hour",
    "wandering_like_episode_count_per_presence_hour",
    "wandering_like_candidate_duration_seconds_per_presence_hour",
    "night_wandering_like_ratio",
)
_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "product_stage",
        "status",
        "evidence_scope",
        "validation_scope",
        "synthetic_person_id",
        "timezone",
        "person_binding_verified",
        "eligible_for_baseline",
        "profile_contract_id",
        "profile_identity",
        "source_day_count",
        "usable_day_count",
        "excluded_day_count",
        "excluded_day_reason_counts",
        "profile_window_first_local_date",
        "profile_window_last_local_date",
        "profile_window_day_count",
        "profile_readiness_status",
        "metric_order",
        "metric_stats",
        "reference_days",
        "real_baseline_ready",
        "eligible_for_risk",
        "baseline_deviation",
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
        "algorithm_event_emitted",
    }
)
_PROFILE_IDENTITY_FIELDS = frozenset(
    {
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
        "timezone",
        "night_window",
        "source_daily_config_sha256",
        "duration_semantics",
        "baseline_config_sha256",
        "effective_baseline_config",
    }
)
_EFFECTIVE_BASELINE_CONFIG_FIELDS = frozenset(
    {"initial_days", "stable_days", "max_window_days", "upper_quantile"}
)
_NIGHT_WINDOW_FIELDS = frozenset({"start", "end"})
_STATS_FIELDS = frozenset({"count", "median", "p90", "min", "max"})
_REFERENCE_DAY_FIELDS = frozenset({"local_date", "metric_values"})
_DESCRIPTOR_FIELDS = frozenset({"byte_count", "sha256"})
_PROFILE_CONTRACT_FIELDS = frozenset({"profile_contract_id", "profile_identity"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "product_stage",
        "status",
        "evidence_scope",
        "validation_scope",
        "artifacts",
        "input_daily_manifests",
        "input_bundle_count",
        "input_daily_row_count",
        "synthetic_person_count",
        "baseline_profile_count",
        "metric_order",
        "quantile_method",
        "effective_baseline_config",
        "baseline_config_sha256",
        "mental_health_config_sha256",
        "builder_source_sha256",
        "profile_contracts",
        "person_binding_verified",
        "eligible_for_baseline",
        "real_baseline_emitted",
        "baseline_deviation_emitted",
        "risk_or_alert_decision_emitted",
        "algorithm_event_emitted",
        "real_human_media_consumed",
        "m0cam_d_started",
    }
)
_FINAL_TOP_LEVEL = frozenset({"baseline_profiles.jsonl", "manifest.json"})
_SYNTHETIC_PERSON_TOKEN = re.compile(r"^SYN-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "baseline_deviation",
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


class WanderingBaselinePreviewError(ValueError):
    """A validated input, version, or baseline-preview contract failed closed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class WanderingBaselinePreviewBuildResult:
    output_dir: Path
    manifest_sha256: str
    input_bundle_count: int
    input_daily_row_count: int
    person_count: int
    profile_count: int


@dataclass(frozen=True)
class ValidatedWanderingBaselinePreview:
    bundle_dir: Path
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    baseline_profiles_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def profile_count(self) -> int:
        return len(self.rows)


def build_wandering_baseline_preview(
    daily_bundles: Sequence[str | Path],
    output_dir: str | Path,
) -> WanderingBaselinePreviewBuildResult:
    """Build a fresh canonical profile preview without emitting real baseline or risk."""

    if isinstance(daily_bundles, (str, bytes, Path)) or not daily_bundles:
        raise WanderingBaselinePreviewError(
            "invalid_daily_bundle", "at least one daily bundle is required"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"baseline preview output already exists: {output}")

    config_bytes = _read_bytes(DEFAULT_CONFIG_PATH, "mental health config")
    baseline_config = load_mental_health_config(DEFAULT_CONFIG_PATH).baseline
    effective_config = _effective_baseline_config(baseline_config)
    baseline_config_bytes = _canonical_json(effective_config, "effective baseline config")
    baseline_config_sha256 = hashlib.sha256(baseline_config_bytes).hexdigest()
    mental_health_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    validated = _load_daily_bundles(daily_bundles)

    rows_by_person: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    person_timezones: dict[str, str] = {}
    person_identities: dict[str, dict[str, Any]] = {}
    observed_keys: set[tuple[str, str, str]] = set()
    for daily in validated:
        for source_row in daily.rows:
            row = dict(source_row)
            person_id = str(row["synthetic_person_id"])
            timezone = str(row["timezone"])
            key = (person_id, str(row["local_date"]), timezone)
            if key in observed_keys:
                raise WanderingBaselinePreviewError(
                    "duplicate_person_day",
                    "duplicate synthetic person/date/timezone input",
                )
            observed_keys.add(key)
            prior_timezone = person_timezones.get(person_id)
            if prior_timezone is not None and prior_timezone != timezone:
                raise WanderingBaselinePreviewError(
                    "baseline_version_reset_required",
                    "one synthetic person cannot mix IANA timezones",
                )
            person_timezones[person_id] = timezone
            identity = _profile_identity(
                row,
                daily.manifest,
                effective_config=effective_config,
                baseline_config_sha256=baseline_config_sha256,
            )
            prior_identity = person_identities.get(person_id)
            if prior_identity is not None and prior_identity != identity:
                raise WanderingBaselinePreviewError(
                    "baseline_version_reset_required",
                    "candidate/model/class/threshold/calibration/episode/night/config/"
                    "duration identity drift",
                )
            person_identities[person_id] = identity
            rows_by_person[person_id].append((row, identity))

    profile_rows = [
        _build_profile(
            person_id,
            person_timezones[person_id],
            rows_by_person[person_id],
            baseline_config,
        )
        for person_id in sorted(rows_by_person)
    ]
    profile_rows.sort(key=lambda row: (row["synthetic_person_id"], row["timezone"]))
    profile_bytes = _canonical_jsonl(profile_rows, "baseline profiles")
    input_descriptors = sorted(
        (
            {
                "byte_count": len(daily.manifest_bytes),
                "sha256": daily.manifest_sha256,
            }
            for daily in validated
        ),
        key=lambda descriptor: str(descriptor["sha256"]),
    )
    contracts_by_id = {
        str(row["profile_contract_id"]): {
            "profile_contract_id": row["profile_contract_id"],
            "profile_identity": row["profile_identity"],
        }
        for row in profile_rows
    }
    profile_contracts = [contracts_by_id[key] for key in sorted(contracts_by_id)]
    manifest = {
        "schema_version": BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION,
        "product_name": PRODUCT_NAME,
        "product_stage": PRODUCT_STAGE,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "artifacts": {"baseline_profiles.jsonl": _descriptor(profile_bytes)},
        "input_daily_manifests": input_descriptors,
        "input_bundle_count": len(validated),
        "input_daily_row_count": sum(len(daily.rows) for daily in validated),
        "synthetic_person_count": len(profile_rows),
        "baseline_profile_count": len(profile_rows),
        "metric_order": list(METRIC_ORDER),
        "quantile_method": QUANTILE_METHOD,
        "effective_baseline_config": effective_config,
        "baseline_config_sha256": baseline_config_sha256,
        "mental_health_config_sha256": mental_health_config_sha256,
        "builder_source_sha256": _sha256_file(Path(__file__)),
        "profile_contracts": profile_contracts,
        "person_binding_verified": False,
        "eligible_for_baseline": False,
        "real_baseline_emitted": False,
        "baseline_deviation_emitted": False,
        "risk_or_alert_decision_emitted": False,
        "algorithm_event_emitted": False,
        "real_human_media_consumed": False,
        "m0cam_d_started": False,
    }
    manifest_bytes = _canonical_json(manifest, "baseline preview manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _write_new_file(staging / "baseline_profiles.jsonl", profile_bytes)
        _write_new_file(staging / "manifest.json", manifest_bytes)
        _verify_staging(staging, profile_bytes, manifest_bytes, profile_rows, manifest)
        if output.exists():
            raise FileExistsError(f"baseline preview output already exists: {output}")
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return WanderingBaselinePreviewBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        input_bundle_count=len(validated),
        input_daily_row_count=sum(len(daily.rows) for daily in validated),
        person_count=len(profile_rows),
        profile_count=len(profile_rows),
    )


def load_validated_wandering_baseline_preview(
    path: str | Path,
) -> ValidatedWanderingBaselinePreview:
    """Read back one canonical exact-schema MVP-3S final."""

    bundle = Path(path)
    try:
        if not bundle.is_dir():
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", "baseline preview bundle is not a directory"
            )
        entries = {entry.name: entry for entry in bundle.iterdir()}
    except OSError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "cannot inspect baseline preview bundle"
        ) from exc
    if set(entries) != _FINAL_TOP_LEVEL or any(
        not entry.is_file() for entry in entries.values()
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview top-level collection has drifted"
        )
    profiles_path = bundle / "baseline_profiles.jsonl"
    manifest_path = bundle / "manifest.json"
    rows = _load_canonical_jsonl(profiles_path, "baseline profiles")
    manifest = _load_canonical_json(manifest_path, "baseline preview manifest")
    profile_bytes = _read_bytes(profiles_path, "baseline profiles")
    manifest_bytes = _read_bytes(manifest_path, "baseline preview manifest")
    _validate_preview_bundle(bundle, rows, manifest)
    return ValidatedWanderingBaselinePreview(
        bundle_dir=bundle.resolve(),
        rows=rows,
        manifest=manifest,
        baseline_profiles_bytes=profile_bytes,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _load_daily_bundles(
    paths: Sequence[str | Path],
) -> tuple[ValidatedWanderingDailySummary, ...]:
    loaded: dict[str, ValidatedWanderingDailySummary] = {}
    for path in paths:
        try:
            daily = load_validated_wandering_daily_summary(path)
        except (WanderingDailySummaryError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            if "baseline_version_reset_required" in str(exc):
                raise WanderingBaselinePreviewError(
                    "baseline_version_reset_required",
                    "daily class/calibration/duration identity has drifted",
                ) from exc
            raise WanderingBaselinePreviewError(
                "invalid_daily_bundle", f"daily bundle failed full MVP-2S validation: {path}"
            ) from exc
        if daily.manifest_sha256 in loaded:
            raise WanderingBaselinePreviewError(
                "duplicate_daily_manifest", "duplicate daily manifest input"
            )
        loaded[daily.manifest_sha256] = daily
    return tuple(loaded[digest] for digest in sorted(loaded))


def _effective_baseline_config(config: BaselineConfig) -> dict[str, Any]:
    return {
        "initial_days": config.initial_days,
        "stable_days": config.stable_days,
        "max_window_days": config.max_window_days,
        "upper_quantile": config.upper_quantile,
    }


def _profile_identity(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    effective_config: Mapping[str, Any],
    baseline_config_sha256: str,
) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "candidate_manifest_sha256": row["candidate_manifest_sha256"],
        "model_state_sha256": row["model_state_sha256"],
        "primary_seed": row["primary_seed"],
        "best_epoch": row["best_epoch"],
        "four_class_order": list(row["four_class_order"]),
        "subtype_order": list(row["subtype_order"]),
        "binary_class_order": list(row["binary_class_order"]),
        "binary_decision_threshold": row["binary_decision_threshold"],
        "probability_calibrated": row["probability_calibrated"],
        "episode_policy_status": row["episode_policy_status"],
        "episode_merge_gap_seconds": row["episode_merge_gap_seconds"],
        "timezone": row["timezone"],
        "night_window": {"start": row["night_start"], "end": row["night_end"]},
        "source_daily_config_sha256": manifest["mental_health_config_sha256"],
        "duration_semantics": row["duration_semantics"],
        "baseline_config_sha256": baseline_config_sha256,
        "effective_baseline_config": dict(effective_config),
    }


def _build_profile(
    person_id: str,
    timezone: str,
    source_rows: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    config: BaselineConfig,
) -> dict[str, Any]:
    ordered = sorted(source_rows, key=lambda item: str(item[0]["local_date"]))
    identities = [identity for _, identity in ordered]
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        raise WanderingBaselinePreviewError(
            "baseline_version_reset_required", "profile identity drift"
        )
    identity = identities[0]
    usable: list[dict[str, Any]] = []
    excluded_reasons: Counter[str] = Counter()
    for row, _ in ordered:
        reasons: list[str] = []
        if float(row["presence_seconds"]) <= 0.0:
            reasons.append("no_presence")
        if float(row["any_window_covered_seconds"]) <= 0.0:
            reasons.append("no_window_coverage")
        if reasons:
            excluded_reasons.update(reasons)
        else:
            usable.append(row)
    selected = usable[-config.max_window_days :]
    window_count = len(selected)
    if window_count < config.initial_days:
        readiness = "warming_up"
    elif window_count < config.stable_days:
        readiness = "initial_ready_preview"
    else:
        readiness = "stable_ready_preview"
    values_by_metric: dict[str, list[float]] = {name: [] for name in METRIC_ORDER}
    reference_days: list[dict[str, Any]] = []
    for row in selected:
        metric_values = _daily_metric_values(row)
        reference_days.append(
            {"local_date": row["local_date"], "metric_values": metric_values}
        )
        for name, value in metric_values.items():
            if value is not None:
                values_by_metric[name].append(value)
    contract_payload = {
        "schema_version": BASELINE_PREVIEW_SCHEMA_VERSION,
        "metric_order": list(METRIC_ORDER),
        "profile_identity": identity,
        "reference_day_contract_version": REFERENCE_DAY_CONTRACT_VERSION,
    }
    profile_contract_id = hashlib.sha256(
        _canonical_json(contract_payload, "profile contract")
    ).hexdigest()
    return {
        "schema_version": BASELINE_PREVIEW_SCHEMA_VERSION,
        "product_name": PRODUCT_NAME,
        "product_stage": PRODUCT_STAGE,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "synthetic_person_id": person_id,
        "timezone": timezone,
        "person_binding_verified": False,
        "eligible_for_baseline": False,
        "profile_contract_id": profile_contract_id,
        "profile_identity": identity,
        "source_day_count": len(ordered),
        "usable_day_count": len(usable),
        "excluded_day_count": len(ordered) - len(usable),
        "excluded_day_reason_counts": {
            name: excluded_reasons[name] for name in sorted(excluded_reasons)
        },
        "profile_window_first_local_date": selected[0]["local_date"] if selected else None,
        "profile_window_last_local_date": selected[-1]["local_date"] if selected else None,
        "profile_window_day_count": window_count,
        "profile_readiness_status": readiness,
        "metric_order": list(METRIC_ORDER),
        "metric_stats": {
            name: _metric_stats(values_by_metric[name], config.upper_quantile)
            for name in METRIC_ORDER
        },
        "reference_days": reference_days,
        "real_baseline_ready": False,
        "eligible_for_risk": False,
        "baseline_deviation": None,
        "risk_level": None,
        "risk_score": None,
        "recommended_action": None,
        "alert_decision": None,
        "medical_diagnosis": None,
        "algorithm_event": None,
        "algorithm_event_emitted": False,
    }


def _daily_metric_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    presence_hours = float(row["presence_seconds"]) / 3600.0
    return {
        "trajectory_coverage": float(row["trajectory_coverage"]),
        "presence_hours": presence_hours,
        "direct_episode_count_per_presence_hour": float(row["direct_episode_count"])
        / presence_hours,
        "pacing_episode_count_per_presence_hour": float(row["pacing_episode_count"])
        / presence_hours,
        "lapping_episode_count_per_presence_hour": float(row["lapping_episode_count"])
        / presence_hours,
        "random_episode_count_per_presence_hour": float(row["random_episode_count"])
        / presence_hours,
        "wandering_like_episode_count_per_presence_hour": float(
            row["wandering_like_episode_count"]
        )
        / presence_hours,
        "wandering_like_candidate_duration_seconds_per_presence_hour": float(
            row["wandering_like_duration_sum_seconds"]
        )
        / presence_hours,
        "night_wandering_like_ratio": (
            None
            if row["night_wandering_like_ratio"] is None
            else float(row["night_wandering_like_ratio"])
        ),
    }


def _metric_stats(values: Sequence[float], upper_quantile: float) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "median": _measure(_quantile(ordered, 0.5)),
        "p90": _measure(_quantile(ordered, upper_quantile)),
        "min": _measure(ordered[0]),
        "max": _measure(ordered[-1]),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(values[lower_index])
    fraction = position - lower_index
    return float(values[lower_index]) + (
        float(values[upper_index]) - float(values[lower_index])
    ) * fraction


def _verify_staging(
    staging: Path,
    expected_profile_bytes: bytes,
    expected_manifest_bytes: bytes,
    expected_rows: Sequence[Mapping[str, Any]],
    expected_manifest: Mapping[str, Any],
) -> None:
    entries = {entry.name: entry for entry in staging.iterdir()}
    if set(entries) != _FINAL_TOP_LEVEL or any(
        not entry.is_file() for entry in entries.values()
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview top-level collection has drifted"
        )
    if (staging / "baseline_profiles.jsonl").read_bytes() != expected_profile_bytes:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline profile staging bytes changed before commit"
        )
    if (staging / "manifest.json").read_bytes() != expected_manifest_bytes:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview manifest bytes changed before commit"
        )
    rows = _load_canonical_jsonl(staging / "baseline_profiles.jsonl", "baseline profiles")
    manifest = _load_canonical_json(staging / "manifest.json", "baseline preview manifest")
    if list(rows) != list(expected_rows) or manifest != expected_manifest:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview staging semantics have drifted"
        )
    _validate_preview_bundle(staging, rows, manifest)


def _validate_preview_bundle(
    bundle: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") == "wandering-baseline-preview-manifest-v1" or any(
        row.get("schema_version") == "wandering-baseline-preview-v1" for row in rows
    ):
        raise WanderingBaselinePreviewError(
            "mvp3s_v1_reaudit_required",
            "MVP-3S v1 lacks reference-day evidence and cannot satisfy v2 readback",
        )
    if not rows:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview must contain at least one profile"
        )
    for row in rows:
        _validate_profile(row)
    if list(rows) != sorted(
        rows, key=lambda row: (str(row["synthetic_person_id"]), str(row["timezone"]))
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline profiles are not stably sorted"
        )
    keys = [(str(row["synthetic_person_id"]), str(row["timezone"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline profile person/timezone key is duplicated"
        )

    _require_exact_fields(manifest, _MANIFEST_FIELDS, "baseline preview manifest")
    if (
        manifest.get("schema_version") != BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION
        or manifest.get("product_name") != PRODUCT_NAME
        or manifest.get("product_stage") != PRODUCT_STAGE
        or manifest.get("status") != STATUS
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("validation_scope") != VALIDATION_SCOPE
        or manifest.get("metric_order") != list(METRIC_ORDER)
        or manifest.get("quantile_method") != QUANTILE_METHOD
        or manifest.get("person_binding_verified") is not False
        or manifest.get("eligible_for_baseline") is not False
        or manifest.get("real_baseline_emitted") is not False
        or manifest.get("baseline_deviation_emitted") is not False
        or manifest.get("risk_or_alert_decision_emitted") is not False
        or manifest.get("algorithm_event_emitted") is not False
        or manifest.get("real_human_media_consumed") is not False
        or manifest.get("m0cam_d_started") is not False
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview manifest fixed contract has drifted"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"baseline_profiles.jsonl"}:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview artifact set has drifted"
        )
    _verify_descriptor(bundle / "baseline_profiles.jsonl", artifacts["baseline_profiles.jsonl"])
    input_descriptors = manifest.get("input_daily_manifests")
    if not isinstance(input_descriptors, list) or not input_descriptors:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "input daily manifest descriptors are invalid"
        )
    for descriptor in input_descriptors:
        _validate_descriptor(descriptor, "input daily manifest")
    if input_descriptors != sorted(
        input_descriptors, key=lambda descriptor: str(descriptor["sha256"])
    ) or len({descriptor["sha256"] for descriptor in input_descriptors}) != len(
        input_descriptors
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "input daily manifest descriptors have drifted"
        )
    for field_name in (
        "input_bundle_count",
        "input_daily_row_count",
        "synthetic_person_count",
        "baseline_profile_count",
    ):
        _nonnegative_integer(manifest.get(field_name), f"manifest {field_name}")
    if (
        manifest["input_bundle_count"] != len(input_descriptors)
        or manifest["synthetic_person_count"] != len(rows)
        or manifest["baseline_profile_count"] != len(rows)
        or manifest["input_daily_row_count"]
        != sum(int(row["source_day_count"]) for row in rows)
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline preview manifest count has drifted"
        )
    effective = manifest.get("effective_baseline_config")
    _validate_effective_config(effective)
    expected_config_sha = hashlib.sha256(
        _canonical_json(effective, "effective baseline config")
    ).hexdigest()
    if manifest.get("baseline_config_sha256") != expected_config_sha:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline config descriptor has drifted"
        )
    _validate_digest(manifest.get("mental_health_config_sha256"), "mental health config")
    _validate_digest(manifest.get("builder_source_sha256"), "baseline builder source")
    if manifest.get("builder_source_sha256") != _sha256_file(Path(__file__)):
        raise WanderingBaselinePreviewError(
            "active_builder_source_mismatch",
            "baseline preview manifest is not bound to the active builder source",
        )
    for row in rows:
        identity = row["profile_identity"]
        if (
            identity["effective_baseline_config"] != effective
            or identity["baseline_config_sha256"] != manifest["baseline_config_sha256"]
        ):
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", "profile and manifest baseline config have drifted"
            )

    contracts = manifest.get("profile_contracts")
    if not isinstance(contracts, list) or not contracts:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile contracts are invalid"
        )
    for contract in contracts:
        _require_exact_fields(contract, _PROFILE_CONTRACT_FIELDS, "profile contract")
        _validate_digest(contract.get("profile_contract_id"), "profile contract ID")
        _validate_profile_identity(contract.get("profile_identity"))
    if contracts != sorted(contracts, key=lambda item: str(item["profile_contract_id"])):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile contracts are not sorted"
        )
    expected_contracts = {
        str(row["profile_contract_id"]): row["profile_identity"] for row in rows
    }
    observed_contracts = {
        str(contract["profile_contract_id"]): contract["profile_identity"]
        for contract in contracts
    }
    if expected_contracts != observed_contracts or len(observed_contracts) != len(contracts):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile contracts do not match profiles"
        )
    _reject_nonempty_decisions(rows, "baseline profiles")
    _reject_nonempty_decisions(manifest, "baseline preview manifest")


def _validate_profile(row: Mapping[str, Any]) -> None:
    _require_exact_fields(row, _PROFILE_FIELDS, "baseline profile")
    if (
        row.get("schema_version") != BASELINE_PREVIEW_SCHEMA_VERSION
        or row.get("product_name") != PRODUCT_NAME
        or row.get("product_stage") != PRODUCT_STAGE
        or row.get("status") != STATUS
        or row.get("evidence_scope") != EVIDENCE_SCOPE
        or row.get("validation_scope") != VALIDATION_SCOPE
        or row.get("person_binding_verified") is not False
        or row.get("eligible_for_baseline") is not False
        or row.get("metric_order") != list(METRIC_ORDER)
        or row.get("real_baseline_ready") is not False
        or row.get("eligible_for_risk") is not False
        or row.get("baseline_deviation") is not None
        or row.get("risk_level") is not None
        or row.get("risk_score") is not None
        or row.get("recommended_action") is not None
        or row.get("alert_decision") is not None
        or row.get("medical_diagnosis") is not None
        or row.get("algorithm_event") is not None
        or row.get("algorithm_event_emitted") is not False
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline profile fixed contract has drifted"
        )
    if not isinstance(
        row.get("synthetic_person_id"), str
    ) or not _SYNTHETIC_PERSON_TOKEN.fullmatch(row["synthetic_person_id"]):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "synthetic person ID is invalid"
        )
    if not isinstance(row.get("timezone"), str) or not row["timezone"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile timezone is invalid"
        )
    try:
        ZoneInfo(row["timezone"])
    except ZoneInfoNotFoundError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile timezone is invalid"
        ) from exc
    identity = row.get("profile_identity")
    _validate_profile_identity(identity)
    if identity["timezone"] != row["timezone"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile timezone and identity have drifted"
        )
    contract_payload = {
        "schema_version": BASELINE_PREVIEW_SCHEMA_VERSION,
        "metric_order": list(METRIC_ORDER),
        "profile_identity": identity,
        "reference_day_contract_version": REFERENCE_DAY_CONTRACT_VERSION,
    }
    expected_contract_id = hashlib.sha256(
        _canonical_json(contract_payload, "profile contract")
    ).hexdigest()
    if row.get("profile_contract_id") != expected_contract_id:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile contract ID has drifted"
        )
    for field_name in (
        "source_day_count",
        "usable_day_count",
        "excluded_day_count",
        "profile_window_day_count",
    ):
        _nonnegative_integer(row.get(field_name), f"profile {field_name}")
    if (
        row["source_day_count"] != row["usable_day_count"] + row["excluded_day_count"]
        or row["profile_window_day_count"] > row["usable_day_count"]
        or row["profile_window_day_count"]
        > identity["effective_baseline_config"]["max_window_days"]
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile day counts have drifted"
        )
    excluded = row.get("excluded_day_reason_counts")
    if not isinstance(excluded, Mapping) or list(excluded) != sorted(excluded):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "excluded day reasons are invalid"
        )
    for name, count in excluded.items():
        if name not in {"no_presence", "no_window_coverage"}:
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", "excluded day reason has drifted"
            )
        _nonnegative_integer(count, "excluded day reason count")
    if sum(int(count) for count in excluded.values()) < int(row["excluded_day_count"]):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "excluded day reason count is incomplete"
        )
    window_count = int(row["profile_window_day_count"])
    first_date = row.get("profile_window_first_local_date")
    last_date = row.get("profile_window_last_local_date")
    if window_count == 0:
        if first_date is not None or last_date is not None:
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", "empty profile window dates must be null"
            )
    else:
        _date_string(first_date, "profile first date")
        _date_string(last_date, "profile last date")
        if str(first_date) > str(last_date):
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", "profile date range is invalid"
            )
    stats = row.get("metric_stats")
    if not isinstance(stats, Mapping) or set(stats) != set(METRIC_ORDER):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile metric set has drifted"
        )
    _validate_reference_stats(row, identity, stats)
    reference_count = len(row["reference_days"])
    for name in METRIC_ORDER:
        _validate_metric_stats(stats[name], reference_count, name)


def _validate_reference_stats(
    row: Mapping[str, Any],
    identity: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> None:
    effective = identity["effective_baseline_config"]
    reference_days = row.get("reference_days")
    if not isinstance(reference_days, list) or len(reference_days) > effective[
        "max_window_days"
    ]:
        raise WanderingBaselinePreviewError(
            "invalid_reference_day_evidence", "reference-day collection is invalid"
        )

    dates: list[str] = []
    values_by_metric: dict[str, list[float]] = {name: [] for name in METRIC_ORDER}
    for reference_day in reference_days:
        _require_reference_day_fields(reference_day)
        local_date = _reference_date(reference_day.get("local_date"))
        if dates and local_date <= dates[-1]:
            raise WanderingBaselinePreviewError(
                "invalid_reference_day_evidence",
                "reference-day dates must be unique and strictly increasing",
            )
        dates.append(local_date)
        metric_values = reference_day.get("metric_values")
        if not isinstance(metric_values, Mapping) or set(metric_values) != set(METRIC_ORDER):
            raise WanderingBaselinePreviewError(
                "invalid_reference_day_evidence",
                "reference-day metric field set has drifted",
            )
        for name in METRIC_ORDER:
            value = _validated_reference_metric(name, metric_values[name])
            if value is not None:
                values_by_metric[name].append(value)

    window_count = len(reference_days)
    initial_days = int(effective["initial_days"])
    stable_days = int(effective["stable_days"])
    expected_readiness = (
        "warming_up"
        if window_count < initial_days
        else "initial_ready_preview"
        if window_count < stable_days
        else "stable_ready_preview"
    )
    expected_stats = {
        name: _metric_stats(values_by_metric[name], float(effective["upper_quantile"]))
        for name in METRIC_ORDER
    }
    if (
        row.get("profile_window_day_count") != window_count
        or row.get("profile_window_first_local_date") != (dates[0] if dates else None)
        or row.get("profile_window_last_local_date") != (dates[-1] if dates else None)
        or row.get("profile_readiness_status") != expected_readiness
        or stats != expected_stats
    ):
        raise WanderingBaselinePreviewError(
            "reference_stats_mismatch",
            "profile window, readiness, or statistics do not match reference days",
        )


def _require_reference_day_fields(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _REFERENCE_DAY_FIELDS:
        raise WanderingBaselinePreviewError(
            "invalid_reference_day_evidence", "reference-day field set has drifted"
        )


def _reference_date(value: Any) -> str:
    try:
        return _date_string(value, "reference-day local date")
    except WanderingBaselinePreviewError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_reference_day_evidence", "reference-day local date is invalid"
        ) from exc


def _validated_reference_metric(name: str, value: Any) -> float | None:
    if name == "night_wandering_like_ratio" and value is None:
        return None
    if not _finite_number(value):
        raise WanderingBaselinePreviewError(
            "invalid_reference_day_evidence", f"reference-day metric {name} is invalid"
        )
    number = float(value)
    if name in {"trajectory_coverage", "night_wandering_like_ratio"}:
        valid = 0.0 <= number <= 1.0
    elif name == "presence_hours":
        valid = number > 0.0
    else:
        valid = number >= 0.0
    if not valid:
        raise WanderingBaselinePreviewError(
            "invalid_reference_day_evidence", f"reference-day metric {name} is out of range"
        )
    return number


def _validate_profile_identity(value: Any) -> None:
    _require_exact_fields(value, _PROFILE_IDENTITY_FIELDS, "profile identity")
    for field_name in (
        "candidate_manifest_sha256",
        "model_state_sha256",
        "source_daily_config_sha256",
        "baseline_config_sha256",
    ):
        _validate_digest(value.get(field_name), field_name)
    if not isinstance(value.get("candidate_id"), str) or not value["candidate_id"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile candidate ID is invalid"
        )
    for field_name in ("primary_seed", "best_epoch"):
        _nonnegative_integer(value.get(field_name), f"profile identity {field_name}")
    if value.get("four_class_order") != ["direct", "pacing", "lapping", "random"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile four-class order has drifted"
        )
    if value.get("subtype_order") != ["pacing", "lapping", "random"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile subtype order has drifted"
        )
    if value.get("binary_class_order") != ["direct_or_non_wandering", "wandering_like"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile binary class order has drifted"
        )
    for field_name in ("binary_decision_threshold", "episode_merge_gap_seconds"):
        number = value.get(field_name)
        if not _finite_number(number) or float(number) < 0.0:
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", f"profile identity {field_name} is invalid"
            )
    if float(value["binary_decision_threshold"]) > 1.0:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile binary threshold is invalid"
        )
    if value.get("probability_calibrated") is not False:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile calibration status has drifted"
        )
    if not isinstance(value.get("episode_policy_status"), str) or not value[
        "episode_policy_status"
    ]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile episode policy is invalid"
        )
    if not isinstance(value.get("timezone"), str):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile identity timezone is invalid"
        )
    try:
        ZoneInfo(value["timezone"])
    except ZoneInfoNotFoundError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile identity timezone is invalid"
        ) from exc
    if value.get("duration_semantics") != "candidate_interval_sum_not_presence_time":
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile duration semantics has drifted"
        )
    night = value.get("night_window")
    _require_exact_fields(night, _NIGHT_WINDOW_FIELDS, "profile night window")
    if not all(_clock_string(night.get(name)) for name in ("start", "end")):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile night window is invalid"
        )
    if night["start"] == night["end"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile night window boundaries cannot match"
        )
    effective = value.get("effective_baseline_config")
    _validate_effective_config(effective)
    if value["baseline_config_sha256"] != hashlib.sha256(
        _canonical_json(effective, "effective baseline config")
    ).hexdigest():
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "profile baseline config digest has drifted"
        )


def _validate_effective_config(value: Any) -> None:
    _require_exact_fields(value, _EFFECTIVE_BASELINE_CONFIG_FIELDS, "effective baseline config")
    for field_name in ("initial_days", "stable_days", "max_window_days"):
        count = _nonnegative_integer(value.get(field_name), f"baseline config {field_name}")
        if count <= 0:
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", f"baseline config {field_name} must be positive"
            )
    if not value["initial_days"] <= value["stable_days"] <= value["max_window_days"]:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline config day ordering is invalid"
        )
    quantile = value.get("upper_quantile")
    if not _finite_number(quantile) or not 0.0 < float(quantile) < 1.0:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", "baseline upper quantile is invalid"
        )


def _validate_metric_stats(value: Any, window_count: int, name: str) -> None:
    _require_exact_fields(value, _STATS_FIELDS, f"metric stats {name}")
    count = _nonnegative_integer(value.get("count"), f"metric stats {name} count")
    if count > window_count:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"metric stats {name} count exceeds window"
        )
    numbers = [value.get(field_name) for field_name in ("median", "p90", "min", "max")]
    if count == 0:
        if any(number is not None for number in numbers):
            raise WanderingBaselinePreviewError(
                "invalid_preview_bundle", f"empty metric stats {name} must be null"
            )
        return
    if any(not _finite_number(number) for number in numbers):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"metric stats {name} is invalid"
        )
    median, p90, minimum, maximum = (float(number) for number in numbers)
    if not minimum <= median <= p90 <= maximum:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"metric stats {name} ordering is invalid"
        )


def _require_exact_fields(value: Any, fields: frozenset[str], role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} field set has drifted"
        )


def _validate_digest(value: Any, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} is not a SHA-256 digest"
        )


def _nonnegative_integer(value: Any, role: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} must be a non-negative integer"
        )
    return value


def _date_string(value: Any, role: str) -> str:
    if not isinstance(value, str):
        raise WanderingBaselinePreviewError("invalid_preview_bundle", f"{role} is invalid")
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} is invalid"
        ) from exc
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _clock_string(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        return False
    try:
        hour = int(value[:2])
        minute = int(value[3:])
    except ValueError:
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59 and f"{hour:02d}:{minute:02d}" == value


def _measure(value: float) -> float:
    return round(float(value), 6)


def _descriptor(payload: bytes) -> dict[str, Any]:
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _validate_descriptor(value: Any, role: str) -> None:
    _require_exact_fields(value, _DESCRIPTOR_FIELDS, f"{role} descriptor")
    _nonnegative_integer(value.get("byte_count"), f"{role} byte_count")
    _validate_digest(value.get("sha256"), role)


def _verify_descriptor(path: Path, value: Any) -> None:
    _validate_descriptor(value, path.name)
    payload = _read_bytes(path, path.name)
    if value != _descriptor(payload):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{path.name} descriptor does not match bytes"
        )


def _canonical_json(value: Any, role: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except CameraAdapterError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} is not finite JSON"
        ) from exc


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]], role: str) -> bytes:
    try:
        return canonical_jsonl_bytes(rows)
    except CameraAdapterError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} is not finite JSONL"
        ) from exc


def _load_canonical_json(path: Path, role: str) -> dict[str, Any]:
    raw = _read_bytes(path, role)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"cannot parse {role}"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_json(value, role):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} is not canonical JSON"
        )
    return value


def _load_canonical_jsonl(path: Path, role: str) -> tuple[dict[str, Any], ...]:
    raw = _read_bytes(path, role)
    try:
        rows = tuple(json.loads(line.decode("utf-8")) for line in raw.splitlines())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"cannot parse {role}"
        ) from exc
    if any(not isinstance(row, dict) for row in rows) or raw != _canonical_jsonl(rows, role):
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"{role} is not canonical JSONL"
        )
    return rows


def _read_bytes(path: Path, role: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"cannot read {role}"
        ) from exc


def _reject_nonempty_decisions(value: Any, role: str) -> None:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            if name in _FORBIDDEN_DECISION_FIELDS and nested is not None:
                raise WanderingBaselinePreviewError(
                    "invalid_preview_bundle", f"{role} contains non-empty {name}"
                )
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
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WanderingBaselinePreviewError(
            "invalid_preview_bundle", f"cannot hash source file: {path}"
        ) from exc
    return digest.hexdigest()


__all__ = [
    "BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION",
    "BASELINE_PREVIEW_SCHEMA_VERSION",
    "METRIC_ORDER",
    "ValidatedWanderingBaselinePreview",
    "WanderingBaselinePreviewBuildResult",
    "WanderingBaselinePreviewError",
    "build_wandering_baseline_preview",
    "load_validated_wandering_baseline_preview",
]
