from __future__ import annotations

import copy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable

import pytest

import test_wandering_camera_daily_summary as daily_fixture
from elderly_monitoring.modules.mental_health.wandering import (
    camera_baseline_preview as preview_module,
)
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_baseline_preview import (
    BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION,
    BASELINE_PREVIEW_SCHEMA_VERSION,
    METRIC_ORDER,
    WanderingBaselinePreviewError,
    build_wandering_baseline_preview,
    load_validated_wandering_baseline_preview,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_summary import (
    build_wandering_daily_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/wandering/build_wandering_baseline_preview.py"
PROFILE_KEYS = {
    "schema_version", "product_name", "product_stage", "status", "evidence_scope",
    "validation_scope", "synthetic_person_id", "timezone", "person_binding_verified",
    "eligible_for_baseline", "profile_contract_id", "profile_identity", "source_day_count",
    "usable_day_count", "excluded_day_count", "excluded_day_reason_counts",
    "profile_window_first_local_date", "profile_window_last_local_date",
    "profile_window_day_count", "profile_readiness_status", "metric_order", "metric_stats",
    "reference_days",
    "real_baseline_ready", "eligible_for_risk", "baseline_deviation", "risk_level",
    "risk_score", "recommended_action", "alert_decision", "medical_diagnosis",
    "algorithm_event", "algorithm_event_emitted",
}
PROFILE_IDENTITY_KEYS = {
    "candidate_id", "candidate_manifest_sha256", "model_state_sha256", "primary_seed",
    "best_epoch", "four_class_order", "subtype_order", "binary_class_order",
    "binary_decision_threshold", "probability_calibrated", "episode_policy_status",
    "episode_merge_gap_seconds", "timezone", "night_window", "source_daily_config_sha256",
    "duration_semantics", "baseline_config_sha256", "effective_baseline_config",
}
EFFECTIVE_CONFIG_KEYS = {"initial_days", "stable_days", "max_window_days", "upper_quantile"}
STATS_KEYS = {"count", "median", "p90", "min", "max"}
REFERENCE_DAY_KEYS = {"local_date", "metric_values"}
MANIFEST_KEYS = {
    "schema_version", "product_name", "product_stage", "status", "evidence_scope",
    "validation_scope", "artifacts", "input_daily_manifests", "input_bundle_count",
    "input_daily_row_count", "synthetic_person_count", "baseline_profile_count",
    "metric_order", "quantile_method", "effective_baseline_config",
    "baseline_config_sha256", "mental_health_config_sha256", "builder_source_sha256",
    "profile_contracts", "person_binding_verified", "eligible_for_baseline",
    "real_baseline_emitted", "baseline_deviation_emitted",
    "risk_or_alert_decision_emitted", "algorithm_event_emitted",
    "real_human_media_consumed", "m0cam_d_started",
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(canonical_jsonl_bytes(rows))


def _day_specs(count: int, *, start: str = "2026-07-01") -> list[dict[str, object]]:
    first = date.fromisoformat(start)
    return [{"local_date": (first + timedelta(days=index)).isoformat()} for index in range(count)]


def _daily_bundle(
    parent: Path,
    name: str,
    specs: list[dict[str, object]],
    *,
    identity_overrides: dict[str, object] | None = None,
    config_sha256: str | None = None,
) -> Path:
    product = daily_fixture._write_product(
        parent / f"{name}-product",
        ["ready"],
        source_suffix=name,
    )
    binding = daily_fixture._write_binding(
        parent / f"{name}-binding.json",
        [
            daily_fixture._session_binding(
                product,
                binding_id=f"SYN-BIND-{name.upper()}",
                session_id=f"SYN-SESSION-{name.upper()}",
            )
        ],
    )
    output = parent / name
    build_wandering_daily_summary([product], binding, output)
    [template] = _load_jsonl(output / "daily_summary.jsonl")
    manifest = _load_json(output / "manifest.json")
    rows: list[dict[str, object]] = []
    for raw_spec in specs:
        spec = dict(raw_spec)
        row = copy.deepcopy(template)
        row["local_date"] = spec.pop("local_date")
        row["synthetic_person_id"] = spec.pop(
            "synthetic_person_id", "SYN-PERSON-001"
        )
        row["timezone"] = spec.pop("timezone", row["timezone"])
        presence = float(spec.pop("presence_seconds", 3600.0))
        covered = float(spec.pop("any_window_covered_seconds", presence))
        row["presence_seconds"] = presence
        row["any_window_covered_seconds"] = covered
        row["ready_covered_seconds"] = covered
        row["unavailable_covered_seconds"] = 0.0
        row["inference_error_covered_seconds"] = 0.0
        row["trajectory_coverage"] = round(covered / presence, 6) if presence else 0.0
        row["ready_coverage"] = round(covered / presence, 6) if presence else 0.0
        row["unavailable_coverage"] = 0.0
        row["inference_error_coverage"] = 0.0
        row["status_overlap_seconds"] = 0.0
        row["window_count"] = 1 if covered > 0 else 0
        row["quality_flags"] = [] if covered > 0 else ["no_window_coverage"]

        counts = dict(spec.pop("episode_counts", {"pacing": 1}))
        durations = dict(spec.pop("episode_durations", {"pacing": 40.0}))
        for pattern in ("direct", "pacing", "lapping", "random"):
            row[f"{pattern}_episode_count"] = int(counts.get(pattern, 0))
            row[f"{pattern}_duration_sum_seconds"] = float(durations.get(pattern, 0.0))
        wandering_count = sum(int(counts.get(name, 0)) for name in ("pacing", "lapping", "random"))
        wandering_duration = sum(
            float(durations.get(name, 0.0)) for name in ("pacing", "lapping", "random")
        )
        night_duration = float(spec.pop("night_duration", 0.0))
        row["wandering_like_episode_count"] = wandering_count
        row["wandering_like_duration_sum_seconds"] = wandering_duration
        row["night_wandering_like_duration_sum_seconds"] = night_duration
        row["night_wandering_like_ratio"] = (
            round(night_duration / wandering_duration, 6) if wandering_duration > 0 else None
        )
        for key, value in spec.items():
            row[key] = value
        if identity_overrides:
            row.update(identity_overrides)
        rows.append(row)

    rows.sort(key=lambda row: (row["local_date"], row["synthetic_person_id"], row["timezone"]))
    _write_jsonl(output / "daily_summary.jsonl", rows)
    payload = (output / "daily_summary.jsonl").read_bytes()
    manifest["artifacts"]["daily_summary.jsonl"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest["synthetic_person_count"] = len({row["synthetic_person_id"] for row in rows})
    manifest["local_date_count"] = len({row["local_date"] for row in rows})
    manifest["daily_summary_count"] = len(rows)
    if identity_overrides:
        manifest.update(
            {
                key: value
                for key, value in identity_overrides.items()
                if key not in {"night_start", "night_end"}
            }
        )
    if rows:
        manifest["night_window"] = {
            "start": rows[0]["night_start"],
            "end": rows[0]["night_end"],
        }
    if config_sha256 is not None:
        manifest["mental_health_config_sha256"] = config_sha256
    _write_json(output / "manifest.json", manifest)
    return output


def _profiles(output: Path) -> list[dict[str, object]]:
    return _load_jsonl(output / "baseline_profiles.jsonl")


def _rewrite_profile_and_descriptor(
    output: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    rows = _profiles(output)
    mutation(rows[0])
    _write_jsonl(output / "baseline_profiles.jsonl", rows)
    manifest = _load_json(output / "manifest.json")
    payload = (output / "baseline_profiles.jsonl").read_bytes()
    manifest["artifacts"]["baseline_profiles.jsonl"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)


def _independent_stats(values: list[float], probability: float) -> dict[str, object]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)

    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 6)

    return {
        "count": len(ordered),
        "median": quantile(0.5),
        "p90": quantile(probability),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
    }


def test_rehashed_negative_reference_stats_are_rejected_by_public_loader(
    tmp_path: Path,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    output = tmp_path / "preview"
    build_wandering_baseline_preview([daily], output)
    rows = _profiles(output)
    rows[0]["metric_stats"]["trajectory_coverage"] = {
        "count": 3,
        "median": -1.0,
        "p90": -1.0,
        "min": -1.0,
        "max": -1.0,
    }
    _write_jsonl(output / "baseline_profiles.jsonl", rows)
    manifest = _load_json(output / "manifest.json")
    payload = (output / "baseline_profiles.jsonl").read_bytes()
    manifest["artifacts"]["baseline_profiles.jsonl"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)

    with pytest.raises(WanderingBaselinePreviewError, match="reference_stats_mismatch"):
        load_validated_wandering_baseline_preview(output)


def test_git_external_fifteen_day_v2_e2e_recomputes_all_stats_and_rejects_attacks(
    tmp_path: Path,
) -> None:
    specs: list[dict[str, object]] = []
    for index, day in enumerate(_day_specs(15)):
        presence = 1800.0 + index * 120.0
        wandering_count = index % 4
        wandering_duration = float(index * 75.0) if wandering_count else 0.0
        specs.append(
            {
                **day,
                "presence_seconds": presence,
                "any_window_covered_seconds": presence * (index + 1) / 16.0,
                "episode_counts": {
                    "direct": index % 3,
                    "pacing": wandering_count,
                    "lapping": 0,
                    "random": 0,
                },
                "episode_durations": {
                    "direct": float(index * 5.0),
                    "pacing": wandering_duration,
                },
                "night_duration": wandering_duration / 4.0,
            }
        )
    daily = _daily_bundle(tmp_path, "daily-e2e", specs)
    valid = tmp_path / "preview-v2-valid"
    build_wandering_baseline_preview([daily], valid)

    loaded = load_validated_wandering_baseline_preview(valid)
    [profile] = loaded.rows
    assert profile["schema_version"] == "wandering-baseline-preview-v2"
    assert profile["profile_window_day_count"] == 14
    assert profile["profile_window_first_local_date"] == "2026-07-02"
    assert profile["profile_window_last_local_date"] == "2026-07-15"
    assert profile["profile_readiness_status"] == "stable_ready_preview"
    probability = profile["profile_identity"]["effective_baseline_config"][
        "upper_quantile"
    ]
    independently_recomputed = {}
    for name in METRIC_ORDER:
        values = [
            float(day["metric_values"][name])
            for day in profile["reference_days"]
            if day["metric_values"][name] is not None
        ]
        independently_recomputed[name] = _independent_stats(values, float(probability))
    assert profile["metric_stats"] == independently_recomputed

    valid_profile_bytes = loaded.baseline_profiles_bytes
    valid_manifest_bytes = loaded.manifest_bytes
    for name, replacement in (("negative", -1.0), ("in_range_wrong", 0.5)):
        attacked = tmp_path / f"preview-v2-{name}"
        shutil.copytree(valid, attacked)

        def mutate(candidate: dict[str, object]) -> None:
            candidate["metric_stats"]["trajectory_coverage"] = {
                "count": 14,
                "median": replacement,
                "p90": replacement,
                "min": replacement,
                "max": replacement,
            }

        _rewrite_profile_and_descriptor(attacked, mutate)
        with pytest.raises(WanderingBaselinePreviewError) as captured:
            load_validated_wandering_baseline_preview(attacked)
        assert captured.value.code == "reference_stats_mismatch"

    reloaded = load_validated_wandering_baseline_preview(valid)
    assert reloaded.baseline_profiles_bytes == valid_profile_bytes
    assert reloaded.manifest_bytes == valid_manifest_bytes


def test_rehashed_in_range_but_wrong_reference_stats_are_rejected(
    tmp_path: Path,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    output = tmp_path / "preview"
    build_wandering_baseline_preview([daily], output)

    def mutate(profile: dict[str, object]) -> None:
        profile["metric_stats"]["trajectory_coverage"] = {
            "count": 3,
            "median": 0.5,
            "p90": 0.5,
            "min": 0.5,
            "max": 0.5,
        }

    _rewrite_profile_and_descriptor(output, mutate)
    with pytest.raises(WanderingBaselinePreviewError) as captured:
        load_validated_wandering_baseline_preview(output)
    assert captured.value.code == "reference_stats_mismatch"


@pytest.mark.parametrize(
    "attack",
    (
        "day_missing_key",
        "day_extra_key",
        "metric_missing_key",
        "metric_extra_key",
        "date_out_of_order",
        "date_duplicate",
        "coverage_low",
        "coverage_high",
        "night_ratio_low",
        "night_ratio_high",
        "presence_zero",
        "presence_negative",
        "negative_rate",
    ),
)
def test_reference_day_exact_schema_order_and_numeric_ranges_fail_closed(
    attack: str,
    tmp_path: Path,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    output = tmp_path / "preview"
    build_wandering_baseline_preview([daily], output)

    def mutate(profile: dict[str, object]) -> None:
        days = profile["reference_days"]
        if attack == "day_missing_key":
            days[0].pop("local_date")
        elif attack == "day_extra_key":
            days[0]["rogue"] = True
        elif attack == "metric_missing_key":
            days[0]["metric_values"].pop("trajectory_coverage")
        elif attack == "metric_extra_key":
            days[0]["metric_values"]["rogue"] = 0.0
        elif attack == "date_out_of_order":
            days[0], days[1] = days[1], days[0]
        elif attack == "date_duplicate":
            days[1]["local_date"] = days[0]["local_date"]
        elif attack == "coverage_low":
            days[0]["metric_values"]["trajectory_coverage"] = -0.1
        elif attack == "coverage_high":
            days[0]["metric_values"]["trajectory_coverage"] = 1.1
        elif attack == "night_ratio_low":
            days[0]["metric_values"]["night_wandering_like_ratio"] = -0.1
        elif attack == "night_ratio_high":
            days[0]["metric_values"]["night_wandering_like_ratio"] = 1.1
        elif attack == "presence_zero":
            days[0]["metric_values"]["presence_hours"] = 0.0
        elif attack == "presence_negative":
            days[0]["metric_values"]["presence_hours"] = -1.0
        else:
            days[0]["metric_values"]["pacing_episode_count_per_presence_hour"] = -1.0

    _rewrite_profile_and_descriptor(output, mutate)
    with pytest.raises(WanderingBaselinePreviewError) as captured:
        load_validated_wandering_baseline_preview(output)
    assert captured.value.code == "invalid_reference_day_evidence"


@pytest.mark.parametrize(
    "attack",
    ("wrong_first", "wrong_last", "wrong_count", "wrong_readiness", "day_value_drift"),
)
def test_reference_days_recompute_window_readiness_and_stats(
    attack: str,
    tmp_path: Path,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(7))
    output = tmp_path / "preview"
    build_wandering_baseline_preview([daily], output)

    def mutate(profile: dict[str, object]) -> None:
        if attack == "wrong_first":
            profile["profile_window_first_local_date"] = "2026-06-30"
        elif attack == "wrong_last":
            profile["profile_window_last_local_date"] = "2026-07-08"
        elif attack == "wrong_count":
            profile["profile_window_day_count"] = 6
        elif attack == "wrong_readiness":
            profile["profile_readiness_status"] = "initial_ready_preview"
        else:
            profile["reference_days"][0]["metric_values"]["trajectory_coverage"] = 0.5

    _rewrite_profile_and_descriptor(output, mutate)
    with pytest.raises(WanderingBaselinePreviewError) as captured:
        load_validated_wandering_baseline_preview(output)
    assert captured.value.code == "reference_stats_mismatch"


def test_active_builder_source_is_recomputed_on_public_readback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    output = tmp_path / "preview"
    build_wandering_baseline_preview([daily], output)
    manifest = _load_json(output / "manifest.json")
    assert manifest["builder_source_sha256"] == hashlib.sha256(
        Path(preview_module.__file__).read_bytes()
    ).hexdigest()
    monkeypatch.setattr(preview_module, "_sha256_file", lambda path: "0" * 64)

    with pytest.raises(WanderingBaselinePreviewError) as captured:
        load_validated_wandering_baseline_preview(output)
    assert captured.value.code == "active_builder_source_mismatch"


def test_v1_preview_bundle_is_blocked_from_v2_downstream(tmp_path: Path) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    output = tmp_path / "preview"
    build_wandering_baseline_preview([daily], output)
    rows = _profiles(output)
    rows[0]["schema_version"] = "wandering-baseline-preview-v1"
    _write_jsonl(output / "baseline_profiles.jsonl", rows)
    manifest = _load_json(output / "manifest.json")
    manifest["schema_version"] = "wandering-baseline-preview-manifest-v1"
    payload = (output / "baseline_profiles.jsonl").read_bytes()
    manifest["artifacts"]["baseline_profiles.jsonl"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)

    with pytest.raises(WanderingBaselinePreviewError) as captured:
        load_validated_wandering_baseline_preview(output)
    assert captured.value.code == "mvp3s_v1_reaudit_required"


@pytest.mark.parametrize(
    ("day_count", "expected_status", "expected_window_count", "first_date"),
    (
        (1, "warming_up", 1, "2026-07-01"),
        (2, "warming_up", 2, "2026-07-01"),
        (3, "initial_ready_preview", 3, "2026-07-01"),
        (7, "stable_ready_preview", 7, "2026-07-01"),
        (14, "stable_ready_preview", 14, "2026-07-01"),
        (15, "stable_ready_preview", 14, "2026-07-02"),
    ),
)
def test_readiness_and_recent_fourteen_usable_day_window(
    day_count: int,
    expected_status: str,
    expected_window_count: int,
    first_date: str,
    tmp_path: Path,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(day_count))
    output = tmp_path / "preview"

    result = build_wandering_baseline_preview([daily], output)

    assert result.profile_count == 1
    assert {path.name for path in output.iterdir()} == {
        "baseline_profiles.jsonl",
        "manifest.json",
    }
    [profile] = _profiles(output)
    manifest = _load_json(output / "manifest.json")
    assert set(profile) == PROFILE_KEYS
    assert set(profile["profile_identity"]) == PROFILE_IDENTITY_KEYS
    assert set(profile["profile_identity"]["effective_baseline_config"]) == EFFECTIVE_CONFIG_KEYS
    assert set(profile["profile_identity"]["night_window"]) == {"start", "end"}
    assert set(profile["metric_stats"]) == set(METRIC_ORDER)
    assert all(set(value) == STATS_KEYS for value in profile["metric_stats"].values())
    assert len(profile["reference_days"]) == expected_window_count
    assert all(set(value) == REFERENCE_DAY_KEYS for value in profile["reference_days"])
    assert all(
        set(value["metric_values"]) == set(METRIC_ORDER)
        for value in profile["reference_days"]
    )
    assert set(manifest) == MANIFEST_KEYS
    assert set(manifest["artifacts"]) == {"baseline_profiles.jsonl"}
    assert set(manifest["artifacts"]["baseline_profiles.jsonl"]) == {"byte_count", "sha256"}
    assert all(
        set(value) == {"byte_count", "sha256"}
        for value in manifest["input_daily_manifests"]
    )
    assert all(
        set(value) == {"profile_contract_id", "profile_identity"}
        for value in manifest["profile_contracts"]
    )
    assert profile["schema_version"] == BASELINE_PREVIEW_SCHEMA_VERSION
    assert BASELINE_PREVIEW_SCHEMA_VERSION == "wandering-baseline-preview-v2"
    assert (
        BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION
        == "wandering-baseline-preview-manifest-v2"
    )
    assert profile["profile_readiness_status"] == expected_status
    assert profile["profile_window_day_count"] == expected_window_count
    assert profile["profile_window_first_local_date"] == first_date
    assert profile["profile_window_last_local_date"] == (
        date(2026, 7, 1) + timedelta(days=day_count - 1)
    ).isoformat()
    assert profile["metric_order"] == list(METRIC_ORDER)
    contract_payload = {
        "schema_version": BASELINE_PREVIEW_SCHEMA_VERSION,
        "metric_order": list(METRIC_ORDER),
        "profile_identity": profile["profile_identity"],
        "reference_day_contract_version": "wandering-baseline-reference-days-v1",
    }
    assert profile["profile_contract_id"] == hashlib.sha256(
        canonical_json_bytes(contract_payload)
    ).hexdigest()
    assert profile["person_binding_verified"] is False
    assert profile["real_baseline_ready"] is False
    assert profile["eligible_for_risk"] is False
    for field in (
        "baseline_deviation",
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
    ):
        assert profile[field] is None


def test_input_order_is_byte_deterministic_and_people_and_date_gaps_stay_isolated(
    tmp_path: Path,
) -> None:
    first = _daily_bundle(
        tmp_path,
        "daily-a",
        [
            {"local_date": "2026-07-01", "synthetic_person_id": "SYN-PERSON-001"},
            {"local_date": "2026-07-02", "synthetic_person_id": "SYN-PERSON-002"},
        ],
    )
    second = _daily_bundle(
        tmp_path,
        "daily-b",
        [{"local_date": "2026-07-03", "synthetic_person_id": "SYN-PERSON-001"}],
    )

    build_wandering_baseline_preview([first, second], tmp_path / "forward")
    build_wandering_baseline_preview([second, first], tmp_path / "reverse")

    assert (tmp_path / "forward/baseline_profiles.jsonl").read_bytes() == (
        tmp_path / "reverse/baseline_profiles.jsonl"
    ).read_bytes()
    assert (tmp_path / "forward/manifest.json").read_bytes() == (
        tmp_path / "reverse/manifest.json"
    ).read_bytes()
    profiles = _profiles(tmp_path / "forward")
    assert [profile["synthetic_person_id"] for profile in profiles] == [
        "SYN-PERSON-001",
        "SYN-PERSON-002",
    ]
    assert profiles[0]["profile_window_day_count"] == 2
    assert profiles[0]["profile_window_first_local_date"] == "2026-07-01"
    assert profiles[0]["profile_window_last_local_date"] == "2026-07-03"
    assert profiles[1]["profile_window_day_count"] == 1


def test_metric_rates_linear_p90_zero_wandering_null_night_and_duration_over_3600(
    tmp_path: Path,
) -> None:
    daily = _daily_bundle(
        tmp_path,
        "daily",
        [
            {
                "local_date": "2026-07-01",
                "presence_seconds": 100.0,
                "episode_counts": {"direct": 1, "pacing": 2},
                "episode_durations": {"direct": 10.0, "pacing": 200.0},
                "night_duration": 50.0,
            },
            {
                "local_date": "2026-07-02",
                "presence_seconds": 200.0,
                "episode_counts": {},
                "episode_durations": {},
                "night_duration": 0.0,
            },
        ],
    )

    build_wandering_baseline_preview([daily], tmp_path / "preview")
    [profile] = _profiles(tmp_path / "preview")
    stats = profile["metric_stats"]
    assert stats["wandering_like_candidate_duration_seconds_per_presence_hour"] == {
        "count": 2,
        "median": 3600.0,
        "p90": 6480.0,
        "min": 0.0,
        "max": 7200.0,
    }
    assert (
        profile["reference_days"][0]["metric_values"]
        ["wandering_like_candidate_duration_seconds_per_presence_hour"]
        == 7200.0
    )
    assert stats["pacing_episode_count_per_presence_hour"]["max"] == 72.0
    assert stats["night_wandering_like_ratio"] == {
        "count": 1,
        "median": 0.25,
        "p90": 0.25,
        "min": 0.25,
        "max": 0.25,
    }


def test_no_window_day_is_excluded_without_hidden_coverage_threshold(tmp_path: Path) -> None:
    daily = _daily_bundle(
        tmp_path,
        "daily",
        [
            {"local_date": "2026-07-01", "any_window_covered_seconds": 0.0,
             "episode_counts": {}, "episode_durations": {}},
            {"local_date": "2026-07-02", "any_window_covered_seconds": 0.000001},
        ],
    )

    build_wandering_baseline_preview([daily], tmp_path / "preview")
    [profile] = _profiles(tmp_path / "preview")
    assert profile["source_day_count"] == 2
    assert profile["usable_day_count"] == 1
    assert profile["excluded_day_count"] == 1
    assert profile["excluded_day_reason_counts"] == {"no_window_coverage": 1}
    assert profile["profile_window_first_local_date"] == "2026-07-02"


def test_duplicate_manifest_duplicate_person_day_and_timezone_drift_fail_closed(
    tmp_path: Path,
) -> None:
    first = _daily_bundle(tmp_path, "daily-a", [{"local_date": "2026-07-01"}])
    with pytest.raises(WanderingBaselinePreviewError, match="duplicate_daily_manifest"):
        build_wandering_baseline_preview([first, first], tmp_path / "duplicate-manifest")

    duplicate_day = _daily_bundle(tmp_path, "daily-b", [{"local_date": "2026-07-01"}])
    with pytest.raises(WanderingBaselinePreviewError, match="duplicate_person_day"):
        build_wandering_baseline_preview([first, duplicate_day], tmp_path / "duplicate-day")

    drift = _daily_bundle(
        tmp_path,
        "daily-c",
        [{"local_date": "2026-07-02", "timezone": "Asia/Tokyo"}],
    )
    with pytest.raises(WanderingBaselinePreviewError) as captured:
        build_wandering_baseline_preview([first, drift], tmp_path / "timezone-drift")
    assert captured.value.code == "baseline_version_reset_required"
    assert not (tmp_path / "timezone-drift").exists()


@pytest.mark.parametrize(
    ("identity_overrides", "config_sha256"),
    (
        ({"candidate_id": "changed-candidate"}, None),
        ({"candidate_manifest_sha256": "1" * 64}, None),
        ({"model_state_sha256": "2" * 64}, None),
        ({"primary_seed": 123}, None),
        ({"best_epoch": 99}, None),
        ({"binary_decision_threshold": 0.6}, None),
        ({"episode_policy_status": "changed-policy"}, None),
        ({"episode_merge_gap_seconds": 2.0}, None),
        ({"night_start": "21:00"}, None),
        ({}, "3" * 64),
    ),
)
def test_same_person_model_policy_night_or_config_drift_requires_version_reset(
    identity_overrides: dict[str, object],
    config_sha256: str | None,
    tmp_path: Path,
) -> None:
    first = _daily_bundle(tmp_path, "daily-a", [{"local_date": "2026-07-01"}])
    second = _daily_bundle(
        tmp_path,
        "daily-b",
        [{"local_date": "2026-07-02"}],
        identity_overrides=identity_overrides,
        config_sha256=config_sha256,
    )

    with pytest.raises(WanderingBaselinePreviewError) as captured:
        build_wandering_baseline_preview([first, second], tmp_path / "preview")
    assert captured.value.code == "baseline_version_reset_required"
    assert not (tmp_path / "preview").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("four_class_order", ["pacing", "direct", "lapping", "random"]),
        ("probability_calibrated", True),
        ("duration_semantics", "presence_time"),
    ),
)
def test_same_person_class_calibration_or_duration_drift_requires_version_reset(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    first = _daily_bundle(tmp_path, "daily-a", [{"local_date": "2026-07-01"}])
    drift = _daily_bundle(
        tmp_path,
        "daily-b",
        [{"local_date": "2026-07-02"}],
        identity_overrides={field: value},
    )

    with pytest.raises(WanderingBaselinePreviewError) as captured:
        build_wandering_baseline_preview([first, drift], tmp_path / "preview")
    assert captured.value.code == "baseline_version_reset_required"
    assert not (tmp_path / "preview").exists()


@pytest.mark.parametrize(
    "attack",
    (
        "profile_missing",
        "profile_extra",
        "metric_order",
        "metric_stats_missing",
        "metric_stats_extra",
        "nonempty_deviation",
        "nonempty_risk",
        "nonempty_event",
        "manifest_missing",
        "manifest_extra",
        "rogue_top_level",
    ),
)
def test_staging_profile_manifest_and_decision_attacks_fail_without_final(
    attack: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    original = preview_module._verify_staging

    def attack_staging(staging: Path, *args: object) -> None:
        if attack == "rogue_top_level":
            _write_json(staging / "rogue.json", {"rogue": True})
        elif attack.startswith("manifest_"):
            manifest = _load_json(staging / "manifest.json")
            if attack == "manifest_missing":
                manifest.pop("product_name")
            else:
                manifest["rogue"] = True
            _write_json(staging / "manifest.json", manifest)
        else:
            rows = _load_jsonl(staging / "baseline_profiles.jsonl")
            if attack == "profile_missing":
                rows[0].pop("product_name")
            elif attack == "profile_extra":
                rows[0]["rogue"] = True
            elif attack == "metric_order":
                rows[0]["metric_order"] = list(reversed(rows[0]["metric_order"]))
            elif attack == "metric_stats_missing":
                rows[0]["metric_stats"].pop(METRIC_ORDER[0])
            elif attack == "metric_stats_extra":
                rows[0]["metric_stats"]["rogue"] = {}
            elif attack == "nonempty_deviation":
                rows[0]["baseline_deviation"] = {"score": 1.0}
            elif attack == "nonempty_risk":
                rows[0]["risk_level"] = "high"
            else:
                rows[0]["algorithm_event"] = {"module": "mental_health"}
            _write_jsonl(staging / "baseline_profiles.jsonl", rows)
            manifest = _load_json(staging / "manifest.json")
            payload = (staging / "baseline_profiles.jsonl").read_bytes()
            manifest["artifacts"]["baseline_profiles.jsonl"] = {
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            _write_json(staging / "manifest.json", manifest)
        original(staging, *args)

    monkeypatch.setattr(preview_module, "_verify_staging", attack_staging)
    with pytest.raises(WanderingBaselinePreviewError):
        build_wandering_baseline_preview([daily], tmp_path / "preview")
    assert not (tmp_path / "preview").exists()
    assert not list(tmp_path.glob(".preview.tmp-*"))


def test_fresh_only_owned_staging_cleanup_and_public_readback(tmp_path: Path) -> None:
    daily = _daily_bundle(tmp_path, "daily", _day_specs(3))
    output = tmp_path / "preview"
    unrelated = tmp_path / ".preview.tmp-unrelated"
    unrelated.mkdir()

    build_wandering_baseline_preview([daily], output)
    validated = load_validated_wandering_baseline_preview(output)
    assert validated.bundle_dir == output.resolve()
    assert validated.profile_count == 1
    assert validated.rows == tuple(_profiles(output))
    with pytest.raises(FileExistsError):
        build_wandering_baseline_preview([daily], output)
    assert unrelated.is_dir()


def test_cli_help_parameter_error_and_fifteen_day_synthetic_e2e(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--daily-bundle" in help_result.stdout
    assert "--output" in help_result.stdout
    for forbidden in ("person", "timezone", "metric", "quantile", "threshold", "risk"):
        assert forbidden not in help_result.stdout.lower()
    error = subprocess.run(
        [sys.executable, str(CLI)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert error.returncode == 2

    specs = _day_specs(15)
    for index in range(5, len(specs)):
        specs[index]["local_date"] = (
            date.fromisoformat(str(specs[index]["local_date"])) + timedelta(days=1)
        ).isoformat()
    specs[5].update({"episode_counts": {}, "episode_durations": {}})
    daily = _daily_bundle(tmp_path, "daily", specs)
    output = tmp_path / "preview-e2e"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--daily-bundle",
            str(daily),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "profiles=1" in completed.stdout
    validated = load_validated_wandering_baseline_preview(output)
    assert validated.manifest["schema_version"] == BASELINE_PREVIEW_MANIFEST_SCHEMA_VERSION
    [profile] = validated.rows
    assert profile["profile_window_day_count"] == 14
    assert profile["profile_window_first_local_date"] == "2026-07-02"
    assert profile["algorithm_event_emitted"] is False


def test_mvp3s_documentation_is_utf8_and_relevant_markdown_links_resolve() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "docs/README.md",
        ROOT / "docs/tasks/README.md",
        ROOT / "docs/modules/mental_health/README.md",
        ROOT / "docs/modules/mental_health/plans/徘徊识别技术文档2.md",
        ROOT / "docs/modules/mental_health/plans/M0-CAM-MVP-3S执行任务书.md",
        ROOT / "docs/modules/mental_health/plans/M0-CAM-MVP-3S-F1执行任务书.md",
        ROOT / "reports/mental_health/wandering_camera_mvp3s_v1/README.md",
        ROOT / "reports/mental_health/wandering_camera_mvp3s_v1/VERIFICATION.md",
        ROOT / "reports/mental_health/wandering_camera_mvp3s_v1/MVP3S_REAUDIT_20260814.md",
    )
    for path in paths:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        assert text.encode("utf-8") == raw
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if not any(
                marker in target
                for marker in (
                    "M0-CAM-MVP-3S",
                    "wandering_camera_mvp3s_v1",
                    "VERIFICATION.md",
                )
            ):
                continue
            relative = target.split("#", 1)[0]
            assert (path.parent / relative).resolve().exists(), f"broken link in {path}: {target}"
