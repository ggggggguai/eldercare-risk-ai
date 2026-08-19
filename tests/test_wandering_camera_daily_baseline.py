from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_baseline import (
    CAMERA_DAILY_REPORT_SCHEMA_VERSION,
    DailyVideoBinding,
    WanderingCameraDailyBaselineError,
    aggregate_wandering_daily_reports,
    build_camera_daily_baseline_bundle,
    build_rolling_baseline_rows,
    load_camera_daily_baseline_config,
    load_validated_camera_daily_baseline_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_daily_baseline_v1.yaml"
CLI = ROOT / "scripts/wandering/run_camera_daily_baseline.py"
IDENTITY = {
    "model_id": "shape-model",
    "model_sha256": "1" * 64,
    "config_id": "daily-config",
    "config_sha256": "2" * 64,
    "policy_id": "daily-policy",
    "policy_sha256": "3" * 64,
}
SOURCE_REF = {
    "ref_type": "fixture",
    "ref_id": "fixture-input",
    "artifact_path": "/fixture/input.jsonl",
    "sha256": "4" * 64,
}


def _episode(
    *,
    episode_id: str,
    source_video_id: str,
    session_id: str,
    person_id: str,
    start_sec: float,
    end_sec: float,
    status: str = "ready",
    binary_label: str | None = "wandering_like",
    shape: str | None = "pacing",
) -> dict[str, object]:
    ready = binary_label is not None and shape is not None
    return {
        "schema_version": "wandering-handoff-episode-result-v1",
        "module": "mental_health",
        "record_id": f"record-{episode_id}",
        "episode_id": episode_id,
        "person_id": person_id,
        "session_id": session_id,
        "source_video_id": source_video_id,
        "start_time": None,
        "end_time": None,
        "start_sec": start_sec,
        "end_sec_exclusive": end_sec,
        "duration_seconds": end_sec - start_sec,
        "status": status,
        "proposal_status": "proposed" if status == "ready" else status,
        "qc_status": "ready" if ready else status,
        "run_status": "auto_accepted" if status == "ready" else "uncertain" if status == "uncertain" else "rejected",
        "technical_segment_index": 0,
        "binary": (
            None
            if binary_label is None
            else {
                "class_order": ["direct_or_non_wandering", "wandering_like"],
                "probabilities": [0.1, 0.9] if binary_label == "wandering_like" else [0.9, 0.1],
                "predicted_label": binary_label,
                "decision_threshold": 0.5,
            }
        ),
        "four_class": (
            None
            if shape is None
            else {
                "class_order": ["direct", "pacing", "lapping", "random"],
                "probabilities": [0.1, 0.7, 0.1, 0.1],
                "predicted_label": shape,
                "decision_threshold": None,
            }
        ),
        "reason_codes": ["fixture"],
        "quality_flags": ["fixture"],
        "identity": dict(IDENTITY),
        "source_refs": [dict(SOURCE_REF)],
    }


def _binding(
    *,
    video: str,
    session: str,
    person: str,
    started_at: str,
    presence: tuple[tuple[float, float], ...] = ((0.0, 60.0),),
    tracking: tuple[tuple[float, float], ...] = ((0.0, 60.0),),
) -> DailyVideoBinding:
    return DailyVideoBinding(
        source_video_id=video,
        session_id=session,
        person_id=person,
        timezone="Asia/Shanghai",
        started_at=datetime.fromisoformat(started_at),
        presence_intervals_sec=presence,
        tracking_intervals_sec=tracking,
        time_basis="fixture_explicit",
        presence_basis="fixture_explicit",
        source_refs=(dict(SOURCE_REF),),
    )


def _daily_row(day: int, *, person: str = "P-001", usable: bool = True) -> dict[str, object]:
    local_date = (datetime(2026, 1, 1) + timedelta(days=day - 1)).date().isoformat()
    presence = 3600.0 if usable else None
    coverage = 0.8 if usable else None
    row = {
        "schema_version": CAMERA_DAILY_REPORT_SCHEMA_VERSION,
        "module": "mental_health",
        "record_id": f"daily-{person}-{local_date}",
        "person_id": person,
        "session_id": None,
        "source_video_id": None,
        "local_date": local_date,
        "timezone": "Asia/Shanghai",
        "status": "ready" if usable else "unavailable",
        "presence_seconds": presence,
        "tracking_coverage_seconds": 2880.0 if usable else None,
        "tracking_coverage": coverage,
        "qc_coverage": 1.0 if usable else None,
        "episode_counts": {"direct": 1, "pacing": day, "lapping": 0, "random": 0, "wandering_like": day},
        "episode_duration_seconds": {"direct": 1.0, "pacing": float(day), "lapping": 0.0, "random": 0.0, "wandering_like": float(day)},
        "confidence_tiers": {
            "high_confidence": {"count": day, "duration_seconds": float(day)},
            "uncertain": {"count": 0, "duration_seconds": 0.0},
            "unavailable": {"count": 0, "duration_seconds": 0.0},
            "error": {"count": 0, "duration_seconds": 0.0},
        },
        "rates_per_presence_hour": {
            "wandering_like_count": float(day) if usable else None,
            "wandering_like_duration_seconds": float(day) if usable else None,
        },
        "night_wandering_like_ratio": 0.0 if usable else None,
        "context_counts": {},
        "unavailable_count": 0,
        "error_count": 0,
        "baseline_readiness": "warming_up",
        "quality_flags": [] if usable else ["presence_unavailable"],
        "identity": dict(IDENTITY),
        "source_refs": [dict(SOURCE_REF)],
    }
    return row


def test_real_binding_merges_sessions_without_crossing_people_and_keeps_tiers() -> None:
    bindings = (
        _binding(video="v1", session="s1", person="P-001", started_at="2026-01-02T09:00:00+08:00"),
        _binding(video="v2", session="s2", person="P-001", started_at="2026-01-02T10:00:00+08:00"),
        _binding(video="v3", session="s3", person="P-002", started_at="2026-01-02T09:00:00+08:00"),
    )
    episodes = [
        _episode(episode_id="e1", source_video_id="v1", session_id="s1", person_id="P-001", start_sec=5, end_sec=15),
        _episode(episode_id="e2", source_video_id="v2", session_id="s2", person_id="P-001", start_sec=5, end_sec=15, status="uncertain", shape="lapping"),
        _episode(episode_id="e3", source_video_id="v2", session_id="s2", person_id="P-001", start_sec=20, end_sec=25, status="unavailable", binary_label=None, shape=None),
        _episode(episode_id="e4", source_video_id="v3", session_id="s3", person_id="P-002", start_sec=5, end_sec=10, binary_label="direct_or_non_wandering", shape="direct"),
    ]
    contexts = {
        "e1": {"status": "ready", "context_label": "searching", "provider": "openai-compatible"},
        "e2": {"status": "ready", "context_label": "cleaning", "provider": "deterministic-fake"},
    }
    before = copy.deepcopy(episodes)

    rows = aggregate_wandering_daily_reports(
        episodes,
        contexts,
        bindings,
        identity=IDENTITY,
        source_refs=(SOURCE_REF,),
        night_start="22:00",
        night_end="06:00",
        minimum_usable_tracking_coverage=0.1,
    )

    assert episodes == before
    assert len(rows) == 2
    p1 = next(row for row in rows if row["person_id"] == "P-001")
    p2 = next(row for row in rows if row["person_id"] == "P-002")
    assert p1["session_id"] is None
    assert p1["source_video_id"] is None
    assert p1["episode_counts"]["wandering_like"] == 2
    assert p1["confidence_tiers"]["high_confidence"]["count"] == 1
    assert p1["confidence_tiers"]["uncertain"]["count"] == 1
    assert p1["unavailable_count"] == 1
    assert p1["context_counts"] == {
        "not_triggered": 1,
        "searching": 1,
        "unknown": 1,
    }
    assert "context_fake_excluded_from_semantic_counts" in p1["quality_flags"]
    assert p2["episode_counts"]["direct"] == 1
    assert p2["episode_counts"]["wandering_like"] == 0
    assert p2["confidence_tiers"]["high_confidence"]["count"] == 0


def test_cross_midnight_splits_presence_tracking_and_duration_by_iana_day() -> None:
    binding = _binding(
        video="v1",
        session="s1",
        person="P-001",
        started_at="2026-01-02T23:59:50+08:00",
        presence=((0.0, 30.0),),
        tracking=((0.0, 30.0),),
    )
    episode = _episode(
        episode_id="e1",
        source_video_id="v1",
        session_id="s1",
        person_id="P-001",
        start_sec=5,
        end_sec=20,
    )

    rows = aggregate_wandering_daily_reports(
        [episode],
        {},
        [binding],
        identity=IDENTITY,
        source_refs=(SOURCE_REF,),
        night_start="22:00",
        night_end="06:00",
        minimum_usable_tracking_coverage=0.1,
    )

    assert [row["local_date"] for row in rows] == ["2026-01-02", "2026-01-03"]
    assert [row["presence_seconds"] for row in rows] == [10.0, 20.0]
    assert [row["tracking_coverage_seconds"] for row in rows] == [10.0, 20.0]
    assert rows[0]["episode_counts"]["wandering_like"] == 1
    assert rows[1]["episode_counts"]["wandering_like"] == 0
    assert rows[0]["episode_duration_seconds"]["wandering_like"] == 5.0
    assert rows[1]["episode_duration_seconds"]["wandering_like"] == 10.0
    assert rows[0]["night_wandering_like_ratio"] == 1.0
    assert rows[1]["night_wandering_like_ratio"] == 1.0


def test_baseline_uses_strict_prior_days_and_transitions_at_3_7_14() -> None:
    daily = [_daily_row(day) for day in range(1, 17)]

    profiles, deviations, updated = build_rolling_baseline_rows(
        daily,
        identity=IDENTITY,
        source_refs=(SOURCE_REF,),
        initial_days=3,
        stable_days=7,
        max_window_days=14,
        upper_quantile=0.9,
        minimum_usable_tracking_coverage=0.1,
    )

    assert len(profiles) == len(deviations) == len(updated) == 16
    by_day = {row["local_date"]: row for row in profiles}
    assert by_day["2026-01-01"]["readiness_status"] == "warming_up"
    assert by_day["2026-01-04"]["readiness_status"] == "initial_ready"
    assert by_day["2026-01-08"]["readiness_status"] == "stable_ready"
    assert by_day["2026-01-15"]["usable_day_count"] == 14
    assert by_day["2026-01-16"]["usable_day_count"] == 14
    assert by_day["2026-01-16"]["reference_first_local_date"] == "2026-01-02"
    assert by_day["2026-01-16"]["reference_last_local_date"] == "2026-01-15"
    assert all(profile["reference_last_local_date"] is None or profile["reference_last_local_date"] < profile["local_date"] for profile in profiles)
    day8 = next(row for row in deviations if row["local_date"] == "2026-01-08")
    assert day8["status"] == "ready"
    assert day8["metrics"]["wandering_like_count_per_presence_hour"]["status"] == "ready"
    assert day8["metrics"]["wandering_like_count_per_presence_hour"]["delta_from_median"] is not None


def test_unusable_day_is_not_filled_with_zero_or_used_as_reference() -> None:
    daily = [_daily_row(1), _daily_row(2, usable=False), _daily_row(3), _daily_row(4)]

    profiles, deviations, updated = build_rolling_baseline_rows(
        daily,
        identity=IDENTITY,
        source_refs=(SOURCE_REF,),
        initial_days=3,
        stable_days=7,
        max_window_days=14,
        upper_quantile=0.9,
        minimum_usable_tracking_coverage=0.1,
    )

    unavailable = updated[1]
    assert unavailable["presence_seconds"] is None
    assert unavailable["rates_per_presence_hour"]["wandering_like_count"] is None
    assert deviations[1]["status"] == "unavailable"
    assert profiles[-1]["usable_day_count"] == 2
    assert profiles[-1]["readiness_status"] == "warming_up"


def _write_fixture_inputs(root: Path) -> Path:
    tracking = root / "tracking.jsonl"
    sidecar = root / "sidecar.json"
    index = root / "development_index.jsonl"
    episodes = root / "episode_results.jsonl"
    contexts = root / "context_reviews.jsonl"
    schemas = ROOT / "configs/schemas/wandering_5d_v1"
    tracking_rows = [
        {"frame_id": value, "track_id": 1, "timestamp_sec": float(value), "bbox": [0, 0, 10, 10], "track_confidence": 0.9}
        for value in range(11)
    ]
    tracking.write_text(
        "".join(json.dumps(row) + "\n" for row in tracking_rows),
        encoding="utf-8",
        newline="",
    )
    sidecar_payload = {"schema_version": "wandering-media-v1", "duration_sec": 10.0, "nominal_fps": 1.0, "capture_started_at": None, "timezone": "Asia/Shanghai"}
    sidecar.write_bytes(canonical_json_bytes(sidecar_payload))
    index_row = {
        "schema_version": "wandering-camera-development-index-v1",
        "batch_id": "B01",
        "dataset_role": "development",
        "development_reuse_allowed": True,
        "historical_role": "fixture",
        "participant_id": "P-001",
        "session_id": "S-001",
        "source_group_id": "fixture",
        "source_video_id": "v1",
        "device_id": "camera",
        "camera_setup_id": "setup",
        "sidecar_setup_id": "setup",
        "setup_binding_status": "consistent",
        "clock_domain_id": "clock",
        "stream_epoch": "epoch",
        "timezone": "Asia/Shanghai",
        "artifacts": {
            "tracking": {"path": tracking.as_posix(), "path_kind": "workspace_generated", "byte_count": tracking.stat().st_size, "sha256": hashlib.sha256(tracking.read_bytes()).hexdigest()},
            "media_sidecar": {"path": sidecar.as_posix(), "path_kind": "workspace_generated", "byte_count": sidecar.stat().st_size, "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()},
            "video": {"path": None, "path_kind": "fixture", "byte_count": 0, "sha256": "5" * 64},
            "truth": {"path": None, "path_kind": "fixture", "byte_count": 0, "sha256": "6" * 64},
            "cvat_xml": {"path": None, "path_kind": "fixture", "byte_count": 0, "sha256": "7" * 64},
        },
    }
    index.write_bytes(canonical_jsonl_bytes([index_row]))
    episodes.write_bytes(canonical_jsonl_bytes([_episode(episode_id="e1", source_video_id="v1", session_id="S-001", person_id="P-001", start_sec=1, end_sec=8)]))
    contexts.write_bytes(b"")
    config = {
        "schema_version": "wandering-camera-daily-baseline-config-v1",
        "daily_baseline_id": "fixture-daily-baseline-v1",
        "validation_scope": "b01_b02_development",
        "home_input_status": "awaiting_input",
        "home_smoke_status": "not_run_input_unavailable",
        "inputs": {
            "development_index": index.as_posix(),
            "episode_results": episodes.as_posix(),
            "context_reviews": contexts.as_posix(),
            "mental_health_config": (ROOT / "configs/modules/mental_health.yaml").as_posix(),
        },
        "schemas": {
            "binding_manifest": (schemas / "daily_binding.schema.json").as_posix(),
            "daily_reports": (schemas / "daily_report_v2.schema.json").as_posix(),
            "baseline_profiles": (schemas / "baseline_profile.schema.json").as_posix(),
            "baseline_deviations": (schemas / "baseline_deviation.schema.json").as_posix(),
            "run_summary": (schemas / "run_summary.schema.json").as_posix(),
        },
        "aggregation": {"minimum_usable_tracking_coverage": 0.1, "tracking_max_gap_seconds": 5.0},
        "session_bindings": [
            {
                "batch_id": "B01",
                "session_id": "S-001",
                "person_id": "P-001",
                "timezone": "Asia/Shanghai",
                "session_started_at": "2026-01-02T09:00:00+08:00",
                "clip_gap_seconds": 1.0,
                "clip_order": ["v1"],
                "time_basis": "fixture_explicit",
                "presence_basis": "fixture_explicit_full_clip",
            }
        ],
        "replay": {"person_id": "P-REPLAY-001", "start_local_date": "2026-02-01", "default_days": 15},
        "identity": {"config_id": "fixture-daily", "policy_id": "fixture-policy"},
    }
    config_path = root / "config.json"
    config_path.write_bytes(canonical_json_bytes(config))
    return config_path


def test_builder_public_loader_recomputes_and_nonoverwrite(tmp_path: Path) -> None:
    config = _write_fixture_inputs(tmp_path)
    output = tmp_path / "bundle"

    result = build_camera_daily_baseline_bundle(
        project_root=ROOT,
        config_path=config,
        output_dir=output,
        run_id="fixture-run",
    )
    loaded = load_validated_camera_daily_baseline_bundle(output)

    assert result.daily_report_count == 1
    assert loaded.manifest["public_loader_recomputed"] is True
    assert loaded.daily_reports[0]["baseline_readiness"] == "warming_up"
    assert loaded.baseline_profiles[0]["readiness_status"] == "warming_up"

    semantic_tamper = tmp_path / "binding-semantic-tamper"
    shutil.copytree(output, semantic_tamper)
    binding_path = semantic_tamper / "binding_manifest.json"
    manifest_path = semantic_tamper / "handoff_manifest.partial.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["person_identity_source"] = "track_id_inference"
    binding_payload = canonical_json_bytes(binding)
    binding_path.write_bytes(binding_payload)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["binding_manifest.json"]["byte_count"] = len(binding_payload)
    manifest["artifacts"]["binding_manifest.json"]["sha256"] = hashlib.sha256(
        binding_payload
    ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(WanderingCameraDailyBaselineError, match="person identity source"):
        load_validated_camera_daily_baseline_bundle(semantic_tamper)

    with pytest.raises(FileExistsError, match="already exists"):
        build_camera_daily_baseline_bundle(
            project_root=ROOT,
            config_path=config,
            output_dir=output,
        )

    daily_path = output / "daily_reports.jsonl"
    daily_path.write_bytes(daily_path.read_bytes() + b"\n")
    with pytest.raises(WanderingCameraDailyBaselineError, match="descriptor"):
        load_validated_camera_daily_baseline_bundle(output)


def test_builder_accepts_path_objects_for_fresh_stage_overrides(tmp_path: Path) -> None:
    config = _write_fixture_inputs(tmp_path)
    config_payload = json.loads(config.read_text(encoding="utf-8"))

    result = build_camera_daily_baseline_bundle(
        project_root=ROOT,
        config_path=config,
        output_dir=tmp_path / "override-bundle",
        run_id="fixture-path-overrides",
        episode_results_path=Path(config_payload["inputs"]["episode_results"]),
        context_reviews_path=Path(config_payload["inputs"]["context_reviews"]),
    )

    assert result.daily_report_count == 1


def test_replay_bundle_proves_readiness_without_claiming_real_longitudinal_data(tmp_path: Path) -> None:
    config = _write_fixture_inputs(tmp_path)
    output = tmp_path / "replay"

    build_camera_daily_baseline_bundle(
        project_root=ROOT,
        config_path=config,
        output_dir=output,
        run_id="fixture-replay",
        replay_days=15,
    )
    loaded = load_validated_camera_daily_baseline_bundle(output)

    assert len(loaded.daily_reports) == 15
    assert loaded.manifest["validation_scope"] == "deterministic_replay"
    assert loaded.baseline_profiles[3]["readiness_status"] == "initial_ready"
    assert loaded.baseline_profiles[7]["readiness_status"] == "stable_ready"
    assert loaded.baseline_profiles[14]["usable_day_count"] == 14
    assert all("deterministic_replay_not_real_longitudinal_observation" in row["quality_flags"] for row in loaded.daily_reports)


def test_production_config_and_cli_help() -> None:
    config = load_camera_daily_baseline_config(CONFIG)
    assert config["validation_scope"] == "b01_b02_development"
    assert {row["session_id"] for row in config["session_bindings"]} == {"S-20260814", "S-20260816"}
    assert all("declared" in row["time_basis"] for row in config["session_bindings"])

    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output-dir" in completed.stdout
    assert "--replay-days" in completed.stdout
