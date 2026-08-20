"""Synthetic baseline-deviation previews from validated MVP-3S and MVP-2S bundles."""

from __future__ import annotations

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

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_baseline_preview import (
    BASELINE_PREVIEW_SCHEMA_VERSION,
    METRIC_ORDER,
    REFERENCE_DAY_CONTRACT_VERSION,
    ValidatedWanderingBaselinePreview,
    WanderingBaselinePreviewError,
    load_validated_wandering_baseline_preview,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_summary import (
    ValidatedWanderingDailySummary,
    WanderingDailySummaryError,
    load_validated_wandering_daily_summary,
)


BASELINE_DEVIATION_PREVIEW_SCHEMA_VERSION = "wandering-baseline-deviation-preview-v1"
BASELINE_DEVIATION_PREVIEW_MANIFEST_SCHEMA_VERSION = (
    "wandering-baseline-deviation-preview-manifest-v1"
)
PRODUCT_NAME = "WanderingBaselineDeviationPreview"
PRODUCT_STAGE = "synthetic_baseline_deviation_preview"
STATUS = "wandering_m0cam_synthetic_baseline_deviation_preview_ready"
EVIDENCE_SCOPE = "synthetic_contract_only"
VALIDATION_SCOPE = "synthetic_camera_contract"
DELTA_SEMANTICS = {
    "delta_from_median": "observation_value_minus_reference_median",
    "delta_from_p90": "observation_value_minus_reference_p90",
}
_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "product_stage",
        "status",
        "evidence_scope",
        "validation_scope",
        "synthetic_person_id",
        "timezone",
        "observation_local_date",
        "person_binding_verified",
        "profile_contract_id",
        "profile_identity",
        "profile_readiness_status",
        "profile_window_last_local_date",
        "baseline_manifest_sha256",
        "observation_daily_manifest_sha256",
        "observation_any_window_covered_seconds",
        "deviation_preview_status",
        "metric_order",
        "baseline_deviation_preview",
        "real_baseline_ready",
        "baseline_deviation",
        "eligible_for_risk",
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
        "algorithm_event_emitted",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "status",
        "observation_value",
        "reference_count",
        "reference_median",
        "reference_p90",
        "delta_from_median",
        "delta_from_p90",
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
_OBSERVATION_IDENTITY_FIELDS = (
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
)
_EFFECTIVE_CONFIG_FIELDS = frozenset(
    {"initial_days", "stable_days", "max_window_days", "upper_quantile"}
)
_NIGHT_WINDOW_FIELDS = frozenset({"start", "end"})
_DESCRIPTOR_FIELDS = frozenset({"byte_count", "sha256"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "product_name",
        "product_stage",
        "status",
        "evidence_scope",
        "validation_scope",
        "artifacts",
        "baseline_manifest",
        "observation_daily_manifests",
        "observation_bundle_count",
        "observation_row_count",
        "synthetic_person_count",
        "observation_local_date_count",
        "deviation_preview_count",
        "baseline_profile_contract_ids",
        "metric_order",
        "delta_semantics",
        "builder_source_sha256",
        "person_binding_verified",
        "real_baseline_emitted",
        "baseline_deviation_emitted",
        "risk_or_alert_decision_emitted",
        "algorithm_event_emitted",
        "real_human_media_consumed",
        "m0cam_d_started",
    }
)
_FINAL_TOP_LEVEL = frozenset({"baseline_deviation_preview.jsonl", "manifest.json"})
_SYNTHETIC_PERSON_TOKEN = re.compile(r"^SYN-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_READINESS_STATUSES = frozenset(
    {"warming_up", "initial_ready_preview", "stable_ready_preview"}
)
_PREVIEW_STATUSES = frozenset(
    {"observation_unavailable", "profile_warming_up", "ready_preview"}
)
_METRIC_STATUSES = frozenset(
    {
        "ready",
        "observation_unavailable",
        "profile_warming_up",
        "observation_value_unavailable",
        "reference_unavailable",
    }
)
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


class WanderingBaselineDeviationPreviewError(ValueError):
    """A baseline-deviation preview contract failed closed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class WanderingBaselineDeviationPreviewBuildResult:
    output_dir: Path
    manifest_sha256: str
    observation_bundle_count: int
    observation_row_count: int
    person_count: int
    preview_count: int


def build_wandering_baseline_deviation_preview(
    baseline_bundle: str | Path,
    observation_daily_bundles: Sequence[str | Path],
    output_dir: str | Path,
) -> WanderingBaselineDeviationPreviewBuildResult:
    """Build a fresh canonical descriptive preview without risk or alert decisions."""

    if isinstance(observation_daily_bundles, (str, bytes, Path)) or not (
        observation_daily_bundles
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_observation_daily_bundle",
            "at least one observation daily bundle is required",
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"baseline deviation preview output already exists: {output}")

    baseline = _load_baseline_bundle(baseline_bundle)
    observations = _load_observation_bundles(observation_daily_bundles)
    reference_manifest_sha256s = {
        str(descriptor["sha256"])
        for descriptor in baseline.manifest["input_daily_manifests"]
    }
    if any(
        daily.manifest_sha256 in reference_manifest_sha256s for daily in observations
    ):
        raise WanderingBaselineDeviationPreviewError(
            "observation_manifest_reused_as_reference",
            "an observation daily manifest is already bound as a baseline reference",
        )

    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for source_profile in baseline.rows:
        profile = dict(source_profile)
        key = (str(profile["synthetic_person_id"]), str(profile["timezone"]))
        if key in profiles:
            raise WanderingBaselineDeviationPreviewError(
                "invalid_baseline_bundle", "baseline profile key is not unique"
            )
        profiles[key] = profile

    observed_keys: set[tuple[str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for daily in observations:
        for source_row in daily.rows:
            observation = dict(source_row)
            person_id = str(observation["synthetic_person_id"])
            timezone_name = str(observation["timezone"])
            local_date = str(observation["local_date"])
            key = (person_id, timezone_name, local_date)
            if key in observed_keys:
                raise WanderingBaselineDeviationPreviewError(
                    "duplicate_observation_day",
                    "observation synthetic person/timezone/day is duplicated",
                )
            observed_keys.add(key)
            profile = profiles.get((person_id, timezone_name))
            if profile is None:
                if any(profile_person == person_id for profile_person, _ in profiles):
                    raise WanderingBaselineDeviationPreviewError(
                        "baseline_version_reset_required",
                        "observation timezone has drifted from the synthetic person profile",
                    )
                raise WanderingBaselineDeviationPreviewError(
                    "baseline_profile_not_found",
                    "observation does not match one unique synthetic person/timezone profile",
                )
            reference_last = profile["profile_window_last_local_date"]
            if reference_last is None or str(reference_last) >= local_date:
                raise WanderingBaselineDeviationPreviewError(
                    "reference_window_not_prior",
                    "baseline reference window must be non-empty and strictly prior",
                )
            _require_observation_identity(profile["profile_identity"], observation, daily.manifest)
            rows.append(
                _build_preview_row(
                    baseline_manifest_sha256=baseline.manifest_sha256,
                    observation_manifest_sha256=daily.manifest_sha256,
                    profile=profile,
                    observation=observation,
                )
            )

    rows.sort(
        key=lambda row: (
            str(row["observation_local_date"]),
            str(row["synthetic_person_id"]),
            str(row["timezone"]),
        )
    )
    row_bytes = _canonical_jsonl(rows, "baseline deviation previews")
    observation_descriptors = sorted(
        (
            {"byte_count": len(daily.manifest_bytes), "sha256": daily.manifest_sha256}
            for daily in observations
        ),
        key=lambda descriptor: str(descriptor["sha256"]),
    )
    manifest = {
        "schema_version": BASELINE_DEVIATION_PREVIEW_MANIFEST_SCHEMA_VERSION,
        "product_name": PRODUCT_NAME,
        "product_stage": PRODUCT_STAGE,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "artifacts": {"baseline_deviation_preview.jsonl": _descriptor(row_bytes)},
        "baseline_manifest": {
            "byte_count": len(baseline.manifest_bytes),
            "sha256": baseline.manifest_sha256,
        },
        "observation_daily_manifests": observation_descriptors,
        "observation_bundle_count": len(observations),
        "observation_row_count": sum(len(daily.rows) for daily in observations),
        "synthetic_person_count": len({str(row["synthetic_person_id"]) for row in rows}),
        "observation_local_date_count": len(
            {str(row["observation_local_date"]) for row in rows}
        ),
        "deviation_preview_count": len(rows),
        "baseline_profile_contract_ids": sorted(
            {str(row["profile_contract_id"]) for row in rows}
        ),
        "metric_order": list(METRIC_ORDER),
        "delta_semantics": dict(DELTA_SEMANTICS),
        "builder_source_sha256": _producer_source_sha256(Path(__file__)),
        "person_binding_verified": False,
        "real_baseline_emitted": False,
        "baseline_deviation_emitted": False,
        "risk_or_alert_decision_emitted": False,
        "algorithm_event_emitted": False,
        "real_human_media_consumed": False,
        "m0cam_d_started": False,
    }
    manifest_bytes = _canonical_json(manifest, "baseline deviation preview manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _write_new_file(staging / "baseline_deviation_preview.jsonl", row_bytes)
        _write_new_file(staging / "manifest.json", manifest_bytes)
        validated = load_validated_wandering_baseline_deviation_preview(staging)
        if (
            validated.baseline_deviation_preview_bytes != row_bytes
            or validated.manifest_bytes != manifest_bytes
            or list(validated.rows) != rows
            or validated.manifest != manifest
        ):
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle",
                "staging public readback does not match producer semantics",
            )
        if output.exists():
            raise FileExistsError(
                f"baseline deviation preview output already exists: {output}"
            )
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return WanderingBaselineDeviationPreviewBuildResult(
        output_dir=output,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        observation_bundle_count=len(observations),
        observation_row_count=len(rows),
        person_count=len({str(row["synthetic_person_id"]) for row in rows}),
        preview_count=len(rows),
    )


def _load_baseline_bundle(path: str | Path) -> ValidatedWanderingBaselinePreview:
    try:
        return load_validated_wandering_baseline_preview(path)
    except (WanderingBaselinePreviewError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_baseline_bundle",
            "baseline bundle failed full MVP-3S v2 validation",
        ) from exc


def _load_observation_bundles(
    paths: Sequence[str | Path],
) -> tuple[ValidatedWanderingDailySummary, ...]:
    loaded: list[ValidatedWanderingDailySummary] = []
    for path in paths:
        try:
            loaded.append(load_validated_wandering_daily_summary(path))
        except (WanderingDailySummaryError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            if "baseline_version_reset_required" in str(exc):
                raise WanderingBaselineDeviationPreviewError(
                    "baseline_version_reset_required",
                    "observation daily identity has drifted",
                ) from exc
            raise WanderingBaselineDeviationPreviewError(
                "invalid_observation_daily_bundle",
                f"observation daily bundle failed full MVP-2S validation: {path}",
            ) from exc
    return tuple(sorted(loaded, key=lambda daily: daily.manifest_sha256))


def _require_observation_identity(
    profile_identity: Mapping[str, Any],
    observation: Mapping[str, Any],
    daily_manifest: Mapping[str, Any],
) -> None:
    observed = {
        "candidate_id": observation["candidate_id"],
        "candidate_manifest_sha256": observation["candidate_manifest_sha256"],
        "model_state_sha256": observation["model_state_sha256"],
        "primary_seed": observation["primary_seed"],
        "best_epoch": observation["best_epoch"],
        "four_class_order": list(observation["four_class_order"]),
        "subtype_order": list(observation["subtype_order"]),
        "binary_class_order": list(observation["binary_class_order"]),
        "binary_decision_threshold": observation["binary_decision_threshold"],
        "probability_calibrated": observation["probability_calibrated"],
        "episode_policy_status": observation["episode_policy_status"],
        "episode_merge_gap_seconds": observation["episode_merge_gap_seconds"],
        "timezone": observation["timezone"],
        "night_window": {
            "start": observation["night_start"],
            "end": observation["night_end"],
        },
        "source_daily_config_sha256": daily_manifest["mental_health_config_sha256"],
        "duration_semantics": observation["duration_semantics"],
    }
    expected = {name: profile_identity[name] for name in _OBSERVATION_IDENTITY_FIELDS}
    if observed != expected:
        raise WanderingBaselineDeviationPreviewError(
            "baseline_version_reset_required",
            "candidate/model/class/threshold/calibration/episode/night/config/duration "
            "identity drift",
        )


def _build_preview_row(
    *,
    baseline_manifest_sha256: str,
    observation_manifest_sha256: str,
    profile: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = float(observation["any_window_covered_seconds"])
    readiness = str(profile["profile_readiness_status"])
    preview_status = (
        "observation_unavailable"
        if coverage <= 0.0
        else "profile_warming_up"
        if readiness == "warming_up"
        else "ready_preview"
    )
    observation_values = (
        {name: None for name in METRIC_ORDER}
        if preview_status == "observation_unavailable"
        else _observation_metric_values(observation)
    )
    metrics = {
        name: _build_metric_preview(
            name=name,
            preview_status=preview_status,
            observation_value=observation_values[name],
            stats=profile["metric_stats"][name],
        )
        for name in METRIC_ORDER
    }
    return {
        "schema_version": BASELINE_DEVIATION_PREVIEW_SCHEMA_VERSION,
        "product_name": PRODUCT_NAME,
        "product_stage": PRODUCT_STAGE,
        "status": STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "validation_scope": VALIDATION_SCOPE,
        "synthetic_person_id": observation["synthetic_person_id"],
        "timezone": observation["timezone"],
        "observation_local_date": observation["local_date"],
        "person_binding_verified": False,
        "profile_contract_id": profile["profile_contract_id"],
        "profile_identity": profile["profile_identity"],
        "profile_readiness_status": readiness,
        "profile_window_last_local_date": profile["profile_window_last_local_date"],
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "observation_daily_manifest_sha256": observation_manifest_sha256,
        "observation_any_window_covered_seconds": coverage,
        "deviation_preview_status": preview_status,
        "metric_order": list(METRIC_ORDER),
        "baseline_deviation_preview": metrics,
        "real_baseline_ready": False,
        "baseline_deviation": None,
        "eligible_for_risk": False,
        "risk_level": None,
        "risk_score": None,
        "recommended_action": None,
        "alert_decision": None,
        "medical_diagnosis": None,
        "algorithm_event": None,
        "algorithm_event_emitted": False,
    }


def _observation_metric_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    presence_hours = float(row["presence_seconds"]) / 3600.0
    values = {
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
    return values


def _build_metric_preview(
    *,
    name: str,
    preview_status: str,
    observation_value: float | None,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    reference_count = int(stats["count"])
    median = stats["median"]
    p90 = stats["p90"]
    if preview_status == "observation_unavailable":
        status = "observation_unavailable"
    elif preview_status == "profile_warming_up":
        status = "profile_warming_up"
    elif observation_value is None:
        status = "observation_value_unavailable"
    elif reference_count == 0:
        status = "reference_unavailable"
    else:
        status = "ready"
    ready = status == "ready"
    return {
        "status": status,
        "observation_value": observation_value,
        "reference_count": reference_count,
        "reference_median": median,
        "reference_p90": p90,
        "delta_from_median": (
            float(observation_value) - float(median) if ready else None
        ),
        "delta_from_p90": (
            float(observation_value) - float(p90) if ready else None
        ),
    }


def _canonical_json(value: Any, role: str) -> bytes:
    try:
        return canonical_json_bytes(value)
    except CameraAdapterError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is not finite JSON"
        ) from exc


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]], role: str) -> bytes:
    try:
        return canonical_jsonl_bytes(rows)
    except CameraAdapterError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is not finite JSONL"
        ) from exc


def _descriptor(payload: bytes) -> dict[str, Any]:
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


__all__ = [
    "BASELINE_DEVIATION_PREVIEW_MANIFEST_SCHEMA_VERSION",
    "BASELINE_DEVIATION_PREVIEW_SCHEMA_VERSION",
    "WanderingBaselineDeviationPreviewBuildResult",
    "WanderingBaselineDeviationPreviewError",
    "build_wandering_baseline_deviation_preview",
]

# === M0-CAM-MVP-3D READ-ONLY LOADER EXTENSION ===
MVP3D_PRODUCER_SOURCE_SHA256 = (
    "60d2482aa18fdebf352f29ef585dbf6b0f02b8476d858beffbfd642e9c691db9"
)
_MVP3D_LOADER_MARKER = b"# === M0-CAM-MVP-3D READ-ONLY LOADER EXTENSION ===\n"
_MVP3D_LOADER_BOUNDARY = b"]\n\n"


@dataclass(frozen=True)
class ValidatedWanderingBaselineDeviationPreview:
    bundle_dir: Path
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    baseline_deviation_preview_bytes: bytes
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def preview_count(self) -> int:
        return len(self.rows)


def load_validated_wandering_baseline_deviation_preview(
    path: str | Path,
) -> ValidatedWanderingBaselineDeviationPreview:
    """Fully validate and load one canonical exact-schema MVP-3D final."""

    bundle = Path(path)
    try:
        if not bundle.is_dir():
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle", "bundle is not a directory"
            )
        entries = {entry.name: entry for entry in bundle.iterdir()}
    except OSError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "cannot inspect bundle"
        ) from exc
    if set(entries) != _FINAL_TOP_LEVEL or any(
        not entry.is_file() for entry in entries.values()
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "top-level collection has drifted"
        )
    rows_path = bundle / "baseline_deviation_preview.jsonl"
    manifest_path = bundle / "manifest.json"
    rows = _load_canonical_jsonl(rows_path, "baseline deviation previews")
    manifest = _load_canonical_json(manifest_path, "baseline deviation preview manifest")
    row_bytes = _read_bytes(rows_path, "baseline deviation previews")
    manifest_bytes = _read_bytes(manifest_path, "baseline deviation preview manifest")
    _validate_bundle(bundle, rows, manifest)
    return ValidatedWanderingBaselineDeviationPreview(
        bundle_dir=bundle.resolve(),
        rows=rows,
        manifest=manifest,
        baseline_deviation_preview_bytes=row_bytes,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _validate_bundle(
    bundle: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if not rows:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "at least one preview row is required"
        )
    for row in rows:
        _validate_row(row)
    expected_order = sorted(
        rows,
        key=lambda row: (
            str(row["observation_local_date"]),
            str(row["synthetic_person_id"]),
            str(row["timezone"]),
        ),
    )
    if list(rows) != expected_order:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "preview rows are not stably sorted"
        )
    keys = [
        (
            str(row["synthetic_person_id"]),
            str(row["timezone"]),
            str(row["observation_local_date"]),
        )
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "preview person/timezone/day is duplicated"
        )

    _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if (
        manifest.get("schema_version")
        != BASELINE_DEVIATION_PREVIEW_MANIFEST_SCHEMA_VERSION
        or manifest.get("product_name") != PRODUCT_NAME
        or manifest.get("product_stage") != PRODUCT_STAGE
        or manifest.get("status") != STATUS
        or manifest.get("evidence_scope") != EVIDENCE_SCOPE
        or manifest.get("validation_scope") != VALIDATION_SCOPE
        or manifest.get("metric_order") != list(METRIC_ORDER)
        or manifest.get("delta_semantics") != DELTA_SEMANTICS
        or manifest.get("person_binding_verified") is not False
        or manifest.get("real_baseline_emitted") is not False
        or manifest.get("baseline_deviation_emitted") is not False
        or manifest.get("risk_or_alert_decision_emitted") is not False
        or manifest.get("algorithm_event_emitted") is not False
        or manifest.get("real_human_media_consumed") is not False
        or manifest.get("m0cam_d_started") is not False
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "manifest fixed contract has drifted"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "baseline_deviation_preview.jsonl"
    }:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "artifact set has drifted"
        )
    _verify_descriptor(
        bundle / "baseline_deviation_preview.jsonl",
        artifacts["baseline_deviation_preview.jsonl"],
    )
    baseline_descriptor = manifest.get("baseline_manifest")
    _validate_descriptor(baseline_descriptor, "baseline manifest")
    observations = manifest.get("observation_daily_manifests")
    if not isinstance(observations, list) or not observations:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "observation descriptors are invalid"
        )
    for descriptor in observations:
        _validate_descriptor(descriptor, "observation daily manifest")
    if observations != sorted(
        observations, key=lambda descriptor: str(descriptor["sha256"])
    ) or len({descriptor["sha256"] for descriptor in observations}) != len(observations):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "observation descriptors have drifted"
        )
    for name in (
        "observation_bundle_count",
        "observation_row_count",
        "synthetic_person_count",
        "observation_local_date_count",
        "deviation_preview_count",
    ):
        _nonnegative_integer(manifest.get(name), f"manifest {name}")
    if (
        manifest["observation_bundle_count"] != len(observations)
        or manifest["observation_row_count"] != len(rows)
        or manifest["deviation_preview_count"] != len(rows)
        or manifest["synthetic_person_count"]
        != len({str(row["synthetic_person_id"]) for row in rows})
        or manifest["observation_local_date_count"]
        != len({str(row["observation_local_date"]) for row in rows})
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "manifest count has drifted"
        )
    contract_ids = manifest.get("baseline_profile_contract_ids")
    if (
        not isinstance(contract_ids, list)
        or contract_ids != sorted(contract_ids)
        or len(set(contract_ids)) != len(contract_ids)
        or set(contract_ids) != {str(row["profile_contract_id"]) for row in rows}
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile contract IDs have drifted"
        )
    for contract_id in contract_ids:
        _validate_digest(contract_id, "profile contract ID")
    _validate_digest(manifest.get("builder_source_sha256"), "builder source")
    active_sha256 = _producer_source_sha256(Path(__file__))
    if manifest.get("builder_source_sha256") != active_sha256:
        raise WanderingBaselineDeviationPreviewError(
            "active_builder_source_mismatch",
            "manifest is not bound to the active MVP-3D producer prefix",
        )
    baseline_sha = str(baseline_descriptor["sha256"])
    observation_shas = {str(descriptor["sha256"]) for descriptor in observations}
    if any(
        row["baseline_manifest_sha256"] != baseline_sha
        or row["observation_daily_manifest_sha256"] not in observation_shas
        for row in rows
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "row input manifest binding has drifted"
        )
    _reject_nonempty_decisions(rows, "preview rows")
    _reject_nonempty_decisions(manifest, "preview manifest")


def _validate_row(row: Mapping[str, Any]) -> None:
    _require_exact_fields(row, _ROW_FIELDS, "preview row")
    if (
        row.get("schema_version") != BASELINE_DEVIATION_PREVIEW_SCHEMA_VERSION
        or row.get("product_name") != PRODUCT_NAME
        or row.get("product_stage") != PRODUCT_STAGE
        or row.get("status") != STATUS
        or row.get("evidence_scope") != EVIDENCE_SCOPE
        or row.get("validation_scope") != VALIDATION_SCOPE
        or row.get("person_binding_verified") is not False
        or row.get("metric_order") != list(METRIC_ORDER)
        or row.get("real_baseline_ready") is not False
        or row.get("baseline_deviation") is not None
        or row.get("eligible_for_risk") is not False
        or row.get("risk_level") is not None
        or row.get("risk_score") is not None
        or row.get("recommended_action") is not None
        or row.get("alert_decision") is not None
        or row.get("medical_diagnosis") is not None
        or row.get("algorithm_event") is not None
        or row.get("algorithm_event_emitted") is not False
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "row fixed contract has drifted"
        )
    person_id = row.get("synthetic_person_id")
    if not isinstance(person_id, str) or not _SYNTHETIC_PERSON_TOKEN.fullmatch(person_id):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "synthetic person ID is invalid"
        )
    timezone_name = row.get("timezone")
    if not isinstance(timezone_name, str):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "timezone is invalid"
        )
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "timezone is invalid"
        ) from exc
    observation_date = _date_string(row.get("observation_local_date"), "observation date")
    reference_last = _date_string(
        row.get("profile_window_last_local_date"), "reference last date"
    )
    if reference_last >= observation_date:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "reference window is not strictly prior"
        )
    readiness = row.get("profile_readiness_status")
    if readiness not in _READINESS_STATUSES:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile readiness is invalid"
        )
    coverage = row.get("observation_any_window_covered_seconds")
    if not _finite_number(coverage) or float(coverage) < 0.0:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "observation coverage is invalid"
        )
    expected_status = (
        "observation_unavailable"
        if float(coverage) <= 0.0
        else "profile_warming_up"
        if readiness == "warming_up"
        else "ready_preview"
    )
    if row.get("deviation_preview_status") not in _PREVIEW_STATUSES or row.get(
        "deviation_preview_status"
    ) != expected_status:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "preview status has drifted"
        )
    identity = row.get("profile_identity")
    _validate_profile_identity(identity)
    if identity["timezone"] != timezone_name:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile identity timezone has drifted"
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
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile contract ID has drifted"
        )
    _validate_digest(row.get("baseline_manifest_sha256"), "baseline manifest")
    _validate_digest(
        row.get("observation_daily_manifest_sha256"), "observation daily manifest"
    )
    metrics = row.get("baseline_deviation_preview")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_ORDER):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "preview metric set has drifted"
        )
    for name in METRIC_ORDER:
        _validate_metric_preview(name, metrics[name], expected_status)


def _validate_metric_preview(name: str, value: Any, preview_status: str) -> None:
    _require_exact_fields(value, _METRIC_FIELDS, f"metric preview {name}")
    reference_count = _nonnegative_integer(
        value.get("reference_count"), f"metric preview {name} reference_count"
    )
    observation = _metric_number(name, value.get("observation_value"), nullable=True)
    median = _metric_number(name, value.get("reference_median"), nullable=True)
    p90 = _metric_number(name, value.get("reference_p90"), nullable=True)
    if reference_count == 0:
        if median is not None or p90 is not None:
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle", "empty reference must have null stats"
            )
    elif median is None or p90 is None or median > p90:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "reference stats are invalid"
        )
    if preview_status == "observation_unavailable":
        expected_status = "observation_unavailable"
        if observation is not None:
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle", "unavailable observation must be null"
            )
    elif preview_status == "profile_warming_up":
        expected_status = "profile_warming_up"
    elif observation is None:
        expected_status = "observation_value_unavailable"
    elif reference_count == 0:
        expected_status = "reference_unavailable"
    else:
        expected_status = "ready"
    if value.get("status") not in _METRIC_STATUSES or value.get("status") != expected_status:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"metric preview {name} status has drifted"
        )
    delta_median = value.get("delta_from_median")
    delta_p90 = value.get("delta_from_p90")
    if expected_status != "ready":
        if delta_median is not None or delta_p90 is not None:
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle", "non-ready metric delta must be null"
            )
        return
    if not _finite_number(delta_median) or not _finite_number(delta_p90):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "ready metric delta is invalid"
        )
    if float(delta_median) != observation - median or float(
        delta_p90
    ) != observation - p90:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"metric preview {name} delta has drifted"
        )


def _metric_number(name: str, value: Any, *, nullable: bool) -> float | None:
    if value is None and nullable:
        return None
    if not _finite_number(value):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"metric {name} value is invalid"
        )
    number = float(value)
    if name in {"trajectory_coverage", "night_wandering_like_ratio"}:
        valid = 0.0 <= number <= 1.0
    elif name == "presence_hours":
        valid = number > 0.0
    else:
        valid = number >= 0.0
    if not valid:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"metric {name} value is out of range"
        )
    return number


def _validate_profile_identity(value: Any) -> None:
    _require_exact_fields(value, _PROFILE_IDENTITY_FIELDS, "profile identity")
    for name in (
        "candidate_manifest_sha256",
        "model_state_sha256",
        "source_daily_config_sha256",
        "baseline_config_sha256",
    ):
        _validate_digest(value.get(name), name)
    if not isinstance(value.get("candidate_id"), str) or not value["candidate_id"]:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "candidate ID is invalid"
        )
    for name in ("primary_seed", "best_epoch"):
        _nonnegative_integer(value.get(name), name)
    if (
        value.get("four_class_order") != ["direct", "pacing", "lapping", "random"]
        or value.get("subtype_order") != ["pacing", "lapping", "random"]
        or value.get("binary_class_order")
        != ["direct_or_non_wandering", "wandering_like"]
        or value.get("probability_calibrated") is not False
        or value.get("duration_semantics")
        != "candidate_interval_sum_not_presence_time"
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile class/calibration/duration drift"
        )
    for name in ("binary_decision_threshold", "episode_merge_gap_seconds"):
        if not _finite_number(value.get(name)) or float(value[name]) < 0.0:
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle", f"profile {name} is invalid"
            )
    if float(value["binary_decision_threshold"]) > 1.0:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "binary threshold is invalid"
        )
    if not isinstance(value.get("episode_policy_status"), str) or not value[
        "episode_policy_status"
    ]:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "episode policy is invalid"
        )
    timezone_name = value.get("timezone")
    if not isinstance(timezone_name, str):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile timezone is invalid"
        )
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "profile timezone is invalid"
        ) from exc
    night = value.get("night_window")
    _require_exact_fields(night, _NIGHT_WINDOW_FIELDS, "night window")
    if not _clock_string(night.get("start")) or not _clock_string(night.get("end")) or night[
        "start"
    ] == night["end"]:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "night window is invalid"
        )
    effective = value.get("effective_baseline_config")
    _require_exact_fields(effective, _EFFECTIVE_CONFIG_FIELDS, "effective baseline config")
    for name in ("initial_days", "stable_days", "max_window_days"):
        if _nonnegative_integer(effective.get(name), name) <= 0:
            raise WanderingBaselineDeviationPreviewError(
                "invalid_deviation_preview_bundle", "baseline day config is invalid"
            )
    if not effective["initial_days"] <= effective["stable_days"] <= effective[
        "max_window_days"
    ]:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "baseline day config ordering is invalid"
        )
    quantile = effective.get("upper_quantile")
    if not _finite_number(quantile) or not 0.0 < float(quantile) < 1.0:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "baseline quantile is invalid"
        )
    if value["baseline_config_sha256"] != hashlib.sha256(
        _canonical_json(effective, "effective baseline config")
    ).hexdigest():
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", "baseline config digest has drifted"
        )


def _require_exact_fields(value: Any, fields: frozenset[str], role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} field set has drifted"
        )


def _validate_descriptor(value: Any, role: str) -> None:
    _require_exact_fields(value, _DESCRIPTOR_FIELDS, f"{role} descriptor")
    _nonnegative_integer(value.get("byte_count"), f"{role} byte_count")
    _validate_digest(value.get("sha256"), role)


def _verify_descriptor(path: Path, value: Any) -> None:
    _validate_descriptor(value, path.name)
    if value != _descriptor(_read_bytes(path, path.name)):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{path.name} descriptor does not match bytes"
        )


def _validate_digest(value: Any, role: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is not a SHA-256 digest"
        )


def _nonnegative_integer(value: Any, role: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} must be a non-negative integer"
        )
    return value


def _date_string(value: Any, role: str) -> str:
    if not isinstance(value, str):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is invalid"
        )
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is invalid"
        ) from exc
    return value


def _clock_string(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        return False
    try:
        hour = int(value[:2])
        minute = int(value[3:])
    except ValueError:
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59 and f"{hour:02d}:{minute:02d}" == value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_bytes(path: Path, role: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"cannot read {role}"
        ) from exc


def _load_canonical_json(path: Path, role: str) -> dict[str, Any]:
    raw = _read_bytes(path, role)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"cannot parse {role}"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_json(value, role):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is not canonical JSON"
        )
    return value


def _load_canonical_jsonl(path: Path, role: str) -> tuple[dict[str, Any], ...]:
    raw = _read_bytes(path, role)
    try:
        rows = tuple(json.loads(line.decode("utf-8")) for line in raw.splitlines())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"cannot parse {role}"
        ) from exc
    if any(not isinstance(row, dict) for row in rows) or raw != _canonical_jsonl(rows, role):
        raise WanderingBaselineDeviationPreviewError(
            "invalid_deviation_preview_bundle", f"{role} is not canonical JSONL"
        )
    return rows


def _reject_nonempty_decisions(value: Any, role: str) -> None:
    if isinstance(value, Mapping):
        for name, nested in value.items():
            if name in _FORBIDDEN_DECISION_FIELDS and nested is not None:
                raise WanderingBaselineDeviationPreviewError(
                    "invalid_deviation_preview_bundle", f"{role} contains non-empty {name}"
                )
            _reject_nonempty_decisions(nested, role)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_nonempty_decisions(nested, role)


def _producer_source_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise WanderingBaselineDeviationPreviewError(
            "active_builder_source_mismatch", f"cannot hash producer source: {path}"
        ) from exc
    if payload.count(_MVP3D_LOADER_MARKER) != 1:
        raise WanderingBaselineDeviationPreviewError(
            "active_builder_source_mismatch", "producer marker count has drifted"
        )
    marker_index = payload.index(_MVP3D_LOADER_MARKER)
    if payload[marker_index - len(_MVP3D_LOADER_BOUNDARY):marker_index] != (
        _MVP3D_LOADER_BOUNDARY
    ):
        raise WanderingBaselineDeviationPreviewError(
            "active_builder_source_mismatch", "producer prefix boundary has drifted"
        )
    producer_prefix = payload[: marker_index - 1]
    source_sha256 = hashlib.sha256(producer_prefix).hexdigest()
    if source_sha256 != MVP3D_PRODUCER_SOURCE_SHA256:
        raise WanderingBaselineDeviationPreviewError(
            "active_builder_source_mismatch", "producer prefix bytes have drifted"
        )
    return source_sha256


__all__.extend(
    [
        "MVP3D_PRODUCER_SOURCE_SHA256",
        "ValidatedWanderingBaselineDeviationPreview",
        "load_validated_wandering_baseline_deviation_preview",
    ]
)
