from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

import test_wandering_camera_product as mvp1_fixture
from elderly_monitoring.modules.mental_health.wandering import camera_daily_summary as daily_module
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_episode import (
    aggregate_episode_candidates,
)
from elderly_monitoring.modules.mental_health.wandering.camera_product import (
    PRODUCT_MANIFEST_SCHEMA_VERSION,
    SESSION_EVIDENCE_SCHEMA_VERSION,
    load_validated_wandering_camera_product,
    summarize_primary_camera_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_summary import (
    DAILY_BINDING_SCHEMA_VERSION,
    DAILY_SUMMARY_MANIFEST_SCHEMA_VERSION,
    DAILY_SUMMARY_SCHEMA_VERSION,
    MVP2S_PRODUCER_SOURCE_SHA256,
    WanderingDailySummaryError,
    build_wandering_daily_summary,
    load_validated_wandering_daily_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/wandering/build_wandering_daily_summary.py"
SOURCE_FIELDS = (
    "source_group_id",
    "source_video_id",
    "device_id",
    "setup_id",
    "stream_epoch",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(canonical_jsonl_bytes(rows))


def _named_probabilities(pattern: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    subtype_by_pattern = {
        "direct": [0.8, 0.1, 0.1],
        "pacing": [0.8, 0.1, 0.1],
        "lapping": [0.1, 0.8, 0.1],
        "random": [0.1, 0.1, 0.8],
    }
    wandering_probability = 0.2 if pattern == "direct" else 0.8
    subtype = subtype_by_pattern[pattern]
    four = [1.0 - wandering_probability, *(wandering_probability * item for item in subtype)]
    return (
        {
            "class_order": ["direct_or_non_wandering", "wandering_like"],
            "probabilities": [1.0 - wandering_probability, wandering_probability],
            "predicted_label": (
                "direct_or_non_wandering" if pattern == "direct" else "wandering_like"
            ),
        },
        {
            "class_order": ["pacing", "lapping", "random"],
            "probabilities": subtype,
            "predicted_label": max(
                zip(subtype, ["pacing", "lapping", "random"]),
                key=lambda item: item[0],
            )[1],
        },
        {
            "class_order": ["direct", "pacing", "lapping", "random"],
            "probabilities": four,
            "predicted_label": pattern,
        },
    )


def _rewrite_primary(
    primary: Path,
    *,
    source_suffix: str,
    intervals: list[tuple[float, float]] | None = None,
    patterns: list[str] | None = None,
    track_ids: tuple[int, ...] = (42,),
) -> None:
    media = _load_json(primary / "media_sidecar.json")
    tracking = _load_jsonl(primary / "tracking_input.jsonl")
    tracklets = _load_jsonl(primary / "bbox_tracklets.jsonl")
    windows = _load_jsonl(primary / "window_records.jsonl")
    predictions = _load_jsonl(primary / "predictions.jsonl")

    if len(track_ids) > 1:
        assert len(track_ids) == 2 and len(windows) == 1 and len(predictions) == 1
        original_track = int(track_ids[0])
        extra_track = int(track_ids[1])
        for row in tracking:
            row["track_id"] = original_track
        if tracklets:
            tracklets[0]["track_id"] = original_track
            tracklets[0]["tracklet_id"] = f"tracklet-{original_track}"
            duplicate = copy.deepcopy(tracklets[0])
            duplicate["track_id"] = extra_track
            duplicate["tracklet_id"] = f"tracklet-{extra_track}"
            tracklets.append(duplicate)
        windows[0]["track_id"] = original_track
        windows[0]["parent_tracklet_id"] = f"tracklet-{original_track}"
        predictions[0]["track_id"] = original_track
        predictions[0]["parent_tracklet_id"] = f"tracklet-{original_track}"
        window_copy = copy.deepcopy(windows[0])
        prediction_copy = copy.deepcopy(predictions[0])
        window_copy["track_id"] = extra_track
        window_copy["parent_tracklet_id"] = f"tracklet-{extra_track}"
        window_copy["window_id"] = f"{window_copy['window_id']}-t{extra_track}"
        prediction_copy["track_id"] = extra_track
        prediction_copy["parent_tracklet_id"] = f"tracklet-{extra_track}"
        prediction_copy["window_id"] = window_copy["window_id"]
        windows.append(window_copy)
        predictions.append(prediction_copy)
        tracking.extend([{**copy.deepcopy(row), "track_id": extra_track} for row in tracking])

    source_scope = {
        "source_group_id": f"session-{source_suffix}",
        "source_video_id": f"video-{source_suffix}",
        "device_id": f"camera-{source_suffix}",
        "setup_id": f"setup-{source_suffix}",
        "stream_epoch": f"epoch-{source_suffix}",
    }
    media.update(source_scope)
    for rows in (tracklets, windows, predictions):
        for row in rows:
            row.update(source_scope)

    if intervals is not None:
        assert len(intervals) == len(windows) == len(predictions)
        for window, prediction, (start, end) in zip(
            windows, predictions, intervals, strict=True
        ):
            window["window_start_sec"] = start
            window["window_end_sec"] = end
            prediction["window_start_sec"] = start
            prediction["window_end_sec"] = end
        media["duration_sec"] = max(end for _, end in intervals)

    if patterns is not None:
        assert len(patterns) == len(predictions)
        for prediction, pattern in zip(predictions, patterns, strict=True):
            if prediction["window_status"] != "ready":
                continue
            binary, subtype, four = _named_probabilities(pattern)
            prediction["binary"] = binary
            prediction["subtype"] = subtype
            prediction["four_class"] = four

    episodes = aggregate_episode_candidates(predictions, merge_gap_seconds=0.0)
    qc = _load_json(primary / "qc_summary.json")
    status_counts = {
        name: sum(row["window_status"] == name for row in predictions)
        for name in ("ready", "unavailable", "inference_error")
    }
    reason_counts: dict[str, int] = {}
    for prediction in predictions:
        for reason in prediction["reason_codes"]:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    qc.update(
        observation_count=len(tracking),
        normalized_tracking_row_count=len(tracking),
        input_track_scope_count=len({int(row["track_id"]) for row in tracking}),
        tracklet_count=len(tracklets),
        window_count=len(predictions),
        window_status_counts=status_counts,
        reason_counts=dict(sorted(reason_counts.items())),
        episode_candidate_count=len(episodes),
    )
    execution = _load_json(primary / "execution.json")
    invocation_count = sum(
        row["window_status"] in {"ready", "inference_error"} for row in predictions
    )
    execution["model_forward_invocation_count"] = invocation_count
    execution["runtime"]["observed_batch_sizes"] = [1] * invocation_count
    execution["latency_ms"]["sample_count"] = invocation_count
    execution["latency_ms"]["p50"] = 1.0 if invocation_count else None
    execution["latency_ms"]["p95"] = 1.0 if invocation_count else None

    payloads = {
        "media_sidecar.json": canonical_json_bytes(media),
        "tracking_input.jsonl": canonical_jsonl_bytes(tracking),
        "bbox_tracklets.jsonl": canonical_jsonl_bytes(tracklets),
        "window_records.jsonl": canonical_jsonl_bytes(windows),
        "predictions.jsonl": canonical_jsonl_bytes(predictions),
        "episode_candidates.jsonl": canonical_jsonl_bytes(episodes),
        "qc_summary.json": canonical_json_bytes(qc),
        "execution.json": canonical_json_bytes(execution),
    }
    for name, payload in payloads.items():
        (primary / name).write_bytes(payload)
    primary_manifest = _load_json(primary / "manifest.json")
    artifacts = primary_manifest["artifacts"]
    assert isinstance(artifacts, dict)
    for name, payload in payloads.items():
        artifacts[name] = {
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    primary_manifest["normalized_tracking_sha256"] = hashlib.sha256(
        payloads["tracking_input.jsonl"]
    ).hexdigest()
    _write_json(primary / "manifest.json", primary_manifest)


def _write_product(
    directory: Path,
    statuses: list[str],
    *,
    source_suffix: str = "0001",
    intervals: list[tuple[float, float]] | None = None,
    patterns: list[str] | None = None,
    track_ids: tuple[int, ...] = (42,),
) -> Path:
    product = directory
    primary = product / "primary"
    product.mkdir(parents=True, exist_ok=False)
    mvp1_fixture._write_primary_bundle(primary, statuses)
    _rewrite_primary(
        primary,
        source_suffix=source_suffix,
        intervals=intervals,
        patterns=patterns,
        track_ids=track_ids,
    )
    evidence = summarize_primary_camera_bundle(primary)
    evidence_bytes = canonical_json_bytes(evidence)
    primary_manifest_bytes = (primary / "manifest.json").read_bytes()
    manifest = {
        "schema_version": PRODUCT_MANIFEST_SCHEMA_VERSION,
        "status": "wandering_m0cam_session_evidence_prototype",
        "product_stage": "session_evidence_prototype",
        "evidence_scope": "synthetic_contract_only",
        "validation_scope": "synthetic_camera_contract",
        "session_status": evidence["session_status"],
        "degraded": evidence["degraded"],
        "artifacts": {
            "primary/manifest.json": {
                "byte_count": len(primary_manifest_bytes),
                "sha256": hashlib.sha256(primary_manifest_bytes).hexdigest(),
            },
            "wandering_evidence.json": {
                "byte_count": len(evidence_bytes),
                "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            },
        },
        "primary_manifest_schema_version": "wandering-camera-primary-run-manifest-v1",
        "wandering_evidence_schema_version": SESSION_EVIDENCE_SCHEMA_VERSION,
        "primary_builder_invocation_count": 1,
        "algorithm_event_emitted": False,
        "risk_or_alert_decision_emitted": False,
        "real_human_media_consumed": False,
        "m0cam_d_started": False,
    }
    (product / "wandering_evidence.json").write_bytes(evidence_bytes)
    _write_json(product / "manifest.json", manifest)
    return product


def _product_digest(product: Path) -> str:
    return hashlib.sha256((product / "manifest.json").read_bytes()).hexdigest()


def _refresh_product_primary_descriptor(product: Path) -> None:
    manifest = _load_json(product / "manifest.json")
    payload = (product / "primary/manifest.json").read_bytes()
    manifest["artifacts"]["primary/manifest.json"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _write_json(product / "manifest.json", manifest)


def _source_scope(product: Path) -> dict[str, object]:
    evidence = _load_json(product / "wandering_evidence.json")
    value = evidence["source_scope"]
    assert isinstance(value, dict)
    return value


def _session_binding(
    product: Path,
    *,
    session_id: str = "SYN-SESSION-001",
    binding_id: str = "SYN-BIND-001",
    started_at: str = "2026-08-14T09:00:00+08:00",
    timezone: str = "Asia/Shanghai",
    person_tracks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "binding_id": binding_id,
        "product_manifest_sha256": _product_digest(product),
        "session_id": session_id,
        "session_started_at": started_at,
        "timezone": timezone,
        "source_scope": _source_scope(product),
        "person_tracks": person_tracks
        or [
            {
                "synthetic_person_id": "SYN-PERSON-001",
                "track_ids": [42],
                "presence_intervals_sec": [
                    {"start_sec": 0.0, "end_sec_exclusive": 40.0}
                ],
            }
        ],
    }


def _write_binding(path: Path, sessions: list[dict[str, object]]) -> Path:
    _write_json(
        path,
        {
            "schema_version": DAILY_BINDING_SCHEMA_VERSION,
            "evidence_scope": "synthetic_binding_fixture",
            "sessions": sessions,
        },
    )
    return path


def _rows(output: Path) -> list[dict[str, object]]:
    return _load_jsonl(output / "daily_summary.jsonl")


def _refresh_daily_artifact_descriptor(output: Path) -> None:
    manifest = _load_json(output / "manifest.json")
    payload = (output / "daily_summary.jsonl").read_bytes()
    manifest["artifacts"]["daily_summary.jsonl"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _write_json(output / "manifest.json", manifest)


def test_mvp2s_producer_prefix_same_length_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(daily_module.__file__).resolve()
    original_read_bytes = Path.read_bytes
    payload = original_read_bytes(source)
    before = b'PRODUCT_NAME = "WanderingDailySummary"'
    after = b'PRODUCT_NAME = "WanderingDailySummarz"'
    assert len(before) == len(after)
    assert payload.count(before) == 1
    mutated = payload.replace(before, after, 1)
    assert len(mutated) == len(payload)

    def read_bytes(path: Path) -> bytes:
        if path.resolve() == source:
            return mutated
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(WanderingDailySummaryError, match="producer_source_identity_mismatch"):
        daily_module._sha256_file(source)


def test_mvp2s_producer_prefix_exact_bytes_match_frozen_identity() -> None:
    source = Path(daily_module.__file__).resolve()
    payload = source.read_bytes()
    marker = daily_module._MVP3S_LOADER_MARKER
    assert payload.count(marker) == 1
    marker_index = payload.index(marker)
    assert payload[marker_index - 2 : marker_index] == b"\n\n"
    producer_prefix = payload[: marker_index - 1]
    assert hashlib.sha256(producer_prefix).hexdigest() == MVP2S_PRODUCER_SOURCE_SHA256


@pytest.mark.parametrize(
    "attack",
    (
        "prefix_byte",
        "marker_missing",
        "marker_duplicate",
        "boundary_extra_lf",
        "boundary_missing_lf",
    ),
)
def test_mvp2s_producer_marker_and_prefix_boundary_attacks_fail_closed(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(daily_module.__file__).resolve()
    original_read_bytes = Path.read_bytes
    payload = original_read_bytes(source)
    marker = daily_module._MVP3S_LOADER_MARKER
    marker_index = payload.index(marker)
    if attack == "prefix_byte":
        payload = payload.replace(
            b'PRODUCT_NAME = "WanderingDailySummary"',
            b'PRODUCT_NAME = "WanderingDailySummarz"',
            1,
        )
    elif attack == "marker_missing":
        payload = payload[:marker_index] + b"!" + payload[marker_index + 1 :]
    elif attack == "marker_duplicate":
        payload = payload + b"\n" + marker
    elif attack == "boundary_extra_lf":
        payload = payload[:marker_index] + b"\n" + payload[marker_index:]
    else:
        payload = payload[: marker_index - 1] + payload[marker_index:]

    def read_bytes(path: Path) -> bytes:
        if path.resolve() == source:
            return payload
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(WanderingDailySummaryError, match="producer_source_identity_mismatch"):
        daily_module._sha256_file(source)


def test_daily_builder_and_public_loader_recheck_active_producer_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    valid = tmp_path / "valid"
    build_wandering_daily_summary([product], binding, valid)

    source = Path(daily_module.__file__).resolve()
    original_read_bytes = Path.read_bytes
    payload = original_read_bytes(source).replace(
        b'PRODUCT_NAME = "WanderingDailySummary"',
        b'PRODUCT_NAME = "WanderingDailySummarz"',
        1,
    )

    def read_bytes(path: Path) -> bytes:
        if path.resolve() == source:
            return payload
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(WanderingDailySummaryError, match="producer_source_identity_mismatch"):
        load_validated_wandering_daily_summary(valid)
    with pytest.raises(WanderingDailySummaryError, match="producer_source_identity_mismatch"):
        build_wandering_daily_summary([product], binding, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_public_validated_daily_loader_round_trips_complete_final(tmp_path: Path) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    output = tmp_path / "daily"
    build_wandering_daily_summary([product], binding, output)

    validated = load_validated_wandering_daily_summary(output)

    assert validated.bundle_dir == output.resolve()
    assert validated.manifest_bytes == (output / "manifest.json").read_bytes()
    assert validated.daily_summary_bytes == (output / "daily_summary.jsonl").read_bytes()
    assert validated.manifest_sha256 == hashlib.sha256(validated.manifest_bytes).hexdigest()
    assert validated.rows == tuple(_rows(output))
    assert validated.manifest["builder_source_sha256"] == MVP2S_PRODUCER_SOURCE_SHA256


@pytest.mark.parametrize(
    "attack",
    (
        "row_missing",
        "row_extra",
        "row_nonempty_decision",
        "manifest_missing",
        "manifest_extra",
        "descriptor_size",
        "descriptor_sha",
        "count",
        "rogue_top_level",
    ),
)
def test_public_validated_daily_loader_rejects_exact_schema_descriptor_and_decision_attacks(
    attack: str,
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    output = tmp_path / "daily"
    build_wandering_daily_summary([product], binding, output)

    if attack.startswith("row_"):
        rows = _rows(output)
        if attack == "row_missing":
            rows[0].pop("product_name")
        elif attack == "row_extra":
            rows[0]["rogue"] = True
        else:
            rows[0]["risk_level"] = "high"
        _write_jsonl(output / "daily_summary.jsonl", rows)
        _refresh_daily_artifact_descriptor(output)
    elif attack == "rogue_top_level":
        _write_json(output / "rogue.json", {"rogue": True})
    else:
        manifest = _load_json(output / "manifest.json")
        if attack == "manifest_missing":
            manifest.pop("product_name")
        elif attack == "manifest_extra":
            manifest["rogue"] = True
        elif attack == "descriptor_size":
            manifest["artifacts"]["daily_summary.jsonl"]["byte_count"] += 1
        elif attack == "descriptor_sha":
            manifest["artifacts"]["daily_summary.jsonl"]["sha256"] = "0" * 64
        else:
            manifest["daily_summary_count"] += 1
        _write_json(output / "manifest.json", manifest)

    with pytest.raises(WanderingDailySummaryError):
        load_validated_wandering_daily_summary(output)


def test_single_session_single_person_daily_summary_is_exact_and_boundary_empty(
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    output = tmp_path / "daily"

    result = build_wandering_daily_summary([product], binding, output)

    assert result.output_dir == output
    assert {path.name for path in output.iterdir()} == {"daily_summary.jsonl", "manifest.json"}
    [row] = _rows(output)
    assert row["schema_version"] == DAILY_SUMMARY_SCHEMA_VERSION
    assert row["synthetic_person_id"] == "SYN-PERSON-001"
    assert row["local_date"] == "2026-08-14"
    assert row["presence_seconds"] == 40.0
    assert row["ready_covered_seconds"] == 40.0
    assert row["pacing_episode_count"] == 1
    assert row["pacing_duration_sum_seconds"] == 40.0
    assert row["direct_episode_count"] == 0
    assert row["wandering_like_episode_count"] == 1
    assert row["uncertain_episode_count"] is None
    assert row["uncertain_ratio"] is None
    assert row["person_binding_verified"] is False
    assert row["eligible_for_baseline"] is False
    for name in (
        "baseline_deviation",
        "risk_level",
        "risk_score",
        "recommended_action",
        "alert_decision",
        "medical_diagnosis",
        "algorithm_event",
    ):
        assert row[name] is None
    manifest = _load_json(output / "manifest.json")
    assert manifest["schema_version"] == DAILY_SUMMARY_MANIFEST_SCHEMA_VERSION
    assert manifest["algorithm_event_emitted"] is False
    assert manifest["baseline_emitted"] is False


def test_cross_midnight_splits_duration_but_counts_episode_on_start_day(tmp_path: Path) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    session = _session_binding(
        product,
        started_at="2026-08-14T23:59:30+08:00",
    )
    binding = _write_binding(tmp_path / "binding.json", [session])

    build_wandering_daily_summary([product], binding, tmp_path / "daily")
    rows = _rows(tmp_path / "daily")

    assert [row["local_date"] for row in rows] == ["2026-08-14", "2026-08-15"]
    assert [row["pacing_duration_sum_seconds"] for row in rows] == [30.0, 10.0]
    assert [row["pacing_episode_count"] for row in rows] == [1, 0]
    assert [row["night_wandering_like_duration_sum_seconds"] for row in rows] == [
        30.0,
        10.0,
    ]
    assert [row["night_wandering_like_ratio"] for row in rows] == [1.0, 1.0]


@pytest.mark.parametrize(
    "started_at",
    (
        "2026-03-08T01:59:30-05:00",
        "2026-11-01T01:59:30-04:00",
    ),
)
def test_dst_uses_epoch_elapsed_seconds(started_at: str, tmp_path: Path) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(
        tmp_path / "binding.json",
        [
            _session_binding(
                product,
                started_at=started_at,
                timezone="America/New_York",
            )
        ],
    )

    build_wandering_daily_summary([product], binding, tmp_path / "daily")

    [row] = _rows(tmp_path / "daily")
    assert row["presence_seconds"] == 40.0
    assert row["ready_covered_seconds"] == 40.0
    assert row["pacing_duration_sum_seconds"] == 40.0


def test_presence_union_status_union_and_status_overlap_are_explicit(tmp_path: Path) -> None:
    product = _write_product(
        tmp_path / "product",
        ["ready", "unavailable", "inference_error"],
        intervals=[(0.0, 40.0), (20.0, 60.0), (50.0, 80.0)],
    )
    session = _session_binding(
        product,
        person_tracks=[
            {
                "synthetic_person_id": "SYN-PERSON-001",
                "track_ids": [42],
                "presence_intervals_sec": [
                    {"start_sec": 0.0, "end_sec_exclusive": 55.0},
                    {"start_sec": 30.0, "end_sec_exclusive": 80.0},
                ],
            }
        ],
    )
    binding = _write_binding(tmp_path / "binding.json", [session])

    build_wandering_daily_summary([product], binding, tmp_path / "daily")

    [row] = _rows(tmp_path / "daily")
    assert row["presence_seconds"] == 80.0
    assert row["any_window_covered_seconds"] == 80.0
    assert row["ready_covered_seconds"] == 40.0
    assert row["unavailable_covered_seconds"] == 40.0
    assert row["inference_error_covered_seconds"] == 30.0
    assert row["status_overlap_seconds"] == 30.0
    assert "status_coverage_overlap" in row["quality_flags"]
    assert row["pacing_episode_count"] == 1
    assert row["direct_episode_count"] == 0


def test_direct_is_separate_and_failure_windows_never_become_episode_evidence(
    tmp_path: Path,
) -> None:
    product = _write_product(
        tmp_path / "product",
        ["ready", "ready", "unavailable", "inference_error"],
        patterns=["direct", "random", "pacing", "pacing"],
    )
    session = _session_binding(
        product,
        person_tracks=[
            {
                "synthetic_person_id": "SYN-PERSON-001",
                "track_ids": [42],
                "presence_intervals_sec": [
                    {"start_sec": 0.0, "end_sec_exclusive": 160.0}
                ],
            }
        ],
    )
    binding = _write_binding(tmp_path / "binding.json", [session])

    build_wandering_daily_summary([product], binding, tmp_path / "daily")

    [row] = _rows(tmp_path / "daily")
    assert row["direct_episode_count"] == 1
    assert row["random_episode_count"] == 1
    assert row["wandering_like_episode_count"] == 1
    assert row["unavailable_coverage"] == pytest.approx(0.25)
    assert row["inference_error_coverage"] == pytest.approx(0.25)


def test_multiple_sessions_merge_same_person_day_and_two_people_stay_isolated(
    tmp_path: Path,
) -> None:
    first = _write_product(
        tmp_path / "product-1",
        ["ready"],
        source_suffix="0001",
        track_ids=(42, 43),
    )
    second = _write_product(
        tmp_path / "product-2",
        ["ready"],
        source_suffix="0002",
    )
    sessions = [
        _session_binding(
            first,
            session_id="SYN-SESSION-001",
            binding_id="SYN-BIND-001",
            person_tracks=[
                {
                    "synthetic_person_id": "SYN-PERSON-001",
                    "track_ids": [42],
                    "presence_intervals_sec": [
                        {"start_sec": 0.0, "end_sec_exclusive": 40.0}
                    ],
                },
                {
                    "synthetic_person_id": "SYN-PERSON-002",
                    "track_ids": [43],
                    "presence_intervals_sec": [
                        {"start_sec": 0.0, "end_sec_exclusive": 40.0}
                    ],
                },
            ],
        ),
        _session_binding(
            second,
            session_id="SYN-SESSION-002",
            binding_id="SYN-BIND-002",
            started_at="2026-08-14T09:01:00+08:00",
        ),
    ]
    binding = _write_binding(tmp_path / "binding.json", sessions)

    build_wandering_daily_summary([first, second], binding, tmp_path / "daily")

    rows = _rows(tmp_path / "daily")
    assert [row["synthetic_person_id"] for row in rows] == [
        "SYN-PERSON-001",
        "SYN-PERSON-002",
    ]
    assert rows[0]["session_count"] == 2
    assert rows[0]["presence_seconds"] == 80.0
    assert rows[1]["session_count"] == 1
    assert rows[1]["presence_seconds"] == 40.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda value: value.pop("evidence_scope"), "binding.*field set"),
        (lambda value: value.__setitem__("rogue", True), "binding.*field set"),
        (
            lambda value: value["sessions"][0]["person_tracks"][0].pop("track_ids"),
            "person track.*field set",
        ),
        (
            lambda value: value["sessions"][0]["person_tracks"][0][
                "presence_intervals_sec"
            ][0].__setitem__("rogue", True),
            "presence interval.*field set",
        ),
        (
            lambda value: value["sessions"][0]["person_tracks"][0]["track_ids"].append(42),
            "track_ids",
        ),
        (
            lambda value: value["sessions"][0]["person_tracks"][0].__setitem__(
                "track_ids", [99]
            ),
            "track",
        ),
        (
            lambda value: value["sessions"][0].__setitem__(
                "session_started_at", "2026-08-14T09:00:00"
            ),
            "offset",
        ),
        (
            lambda value: value["sessions"][0].__setitem__("timezone", "Mars/Olympus"),
            "timezone",
        ),
        (
            lambda value: value["sessions"][0].__setitem__("timezone", "Asia/Tokyo"),
            "offset and timezone",
        ),
        (
            lambda value: value["sessions"][0]["source_scope"].__setitem__(
                "setup_id", "wrong-setup"
            ),
            "source_scope",
        ),
        (
            lambda value: value["sessions"][0].__setitem__(
                "product_manifest_sha256", "0" * 64
            ),
            "product hash",
        ),
        (
            lambda value: value["sessions"][0]["person_tracks"][0].__setitem__(
                "synthetic_person_id", "Alice Smith"
            ),
            "synthetic_person_id",
        ),
        (
            lambda value: value["sessions"][0]["person_tracks"][0].__setitem__(
                "presence_intervals_sec", []
            ),
            "presence",
        ),
        (
            lambda value: value["sessions"][0]["person_tracks"][0].__setitem__(
                "presence_intervals_sec",
                [{"start_sec": 0.0, "end_sec_exclusive": 20.0}],
            ),
            "presence",
        ),
    ),
)
def test_binding_drift_and_unsafe_identity_or_time_fail_closed(
    mutation: object,
    match: str,
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    value = {
        "schema_version": DAILY_BINDING_SCHEMA_VERSION,
        "evidence_scope": "synthetic_binding_fixture",
        "sessions": [_session_binding(product)],
    }
    mutation(value)  # type: ignore[operator]
    binding = tmp_path / "binding.json"
    _write_json(binding, value)

    with pytest.raises(WanderingDailySummaryError, match=match):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")

    assert not (tmp_path / "daily").exists()
    assert not list(tmp_path.glob(".daily.tmp-*"))


def test_product_final_and_primary_are_fully_revalidated_with_rehashed_attack(
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    mvp1_fixture._mutate_primary_artifact(
        product / "primary",
        "predictions.jsonl",
        lambda row: row.pop("model_purpose"),
    )
    product_manifest = _load_json(product / "manifest.json")
    primary_manifest_bytes = (product / "primary/manifest.json").read_bytes()
    product_manifest["artifacts"]["primary/manifest.json"] = {
        "byte_count": len(primary_manifest_bytes),
        "sha256": hashlib.sha256(primary_manifest_bytes).hexdigest(),
    }
    _write_json(product / "manifest.json", product_manifest)
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])

    with pytest.raises(WanderingDailySummaryError, match="product"):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")

    assert not (tmp_path / "daily").exists()


def test_duplicate_product_binding_and_cross_person_track_overlap_fail_closed(
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    first = _session_binding(product)
    duplicate = copy.deepcopy(first)
    duplicate["binding_id"] = "SYN-BIND-002"
    duplicate["session_id"] = "SYN-SESSION-002"
    binding = _write_binding(tmp_path / "duplicate.json", [first, duplicate])
    with pytest.raises(WanderingDailySummaryError, match="duplicate product"):
        build_wandering_daily_summary([product], binding, tmp_path / "duplicate-output")

    overlap = _session_binding(
        product,
        person_tracks=[
            {
                "synthetic_person_id": "SYN-PERSON-001",
                "track_ids": [42],
                "presence_intervals_sec": [
                    {"start_sec": 0.0, "end_sec_exclusive": 40.0}
                ],
            },
            {
                "synthetic_person_id": "SYN-PERSON-002",
                "track_ids": [42],
                "presence_intervals_sec": [
                    {"start_sec": 0.0, "end_sec_exclusive": 40.0}
                ],
            },
        ],
    )
    binding = _write_binding(tmp_path / "overlap.json", [overlap])
    with pytest.raises(WanderingDailySummaryError, match="across synthetic people"):
        build_wandering_daily_summary([product], binding, tmp_path / "overlap-output")


def test_same_person_day_timezone_drift_fails_closed(tmp_path: Path) -> None:
    first = _write_product(tmp_path / "product-1", ["ready"], source_suffix="0001")
    second = _write_product(tmp_path / "product-2", ["ready"], source_suffix="0002")
    binding = _write_binding(
        tmp_path / "binding.json",
        [
            _session_binding(first),
            _session_binding(
                second,
                binding_id="SYN-BIND-002",
                session_id="SYN-SESSION-002",
                started_at="2026-08-14T10:00:00+09:00",
                timezone="Asia/Tokyo",
            ),
        ],
    )

    with pytest.raises(WanderingDailySummaryError, match="timezones"):
        build_wandering_daily_summary([first, second], binding, tmp_path / "daily")


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    (
        (
            "model_bindings.json",
            lambda value: value.__setitem__("binary_decision_threshold", 0.6),
        ),
        (
            "model_bindings.json",
            lambda value: value.__setitem__(
                "four_class_order", ["pacing", "direct", "lapping", "random"]
            ),
        ),
        (
            "model_bindings.json",
            lambda value: value.__setitem__("probability_calibrated", True),
        ),
        (
            "execution.json",
            lambda value: value.__setitem__("episode_merge_gap_seconds", 1.0),
        ),
    ),
)
def test_model_class_threshold_calibration_and_episode_policy_drift_fail_upstream(
    artifact: str,
    mutation: object,
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    mvp1_fixture._mutate_primary_artifact(product / "primary", artifact, mutation)
    _refresh_product_primary_descriptor(product)
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])

    with pytest.raises(WanderingDailySummaryError, match="product"):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")


def test_same_length_product_evidence_tamper_with_rehashed_descriptors_fails_closed(
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    evidence_path = product / "wandering_evidence.json"
    before = evidence_path.read_bytes()
    evidence = _load_json(evidence_path)
    evidence["session_status"] = "error"
    _write_json(evidence_path, evidence)
    after = evidence_path.read_bytes()
    assert len(after) == len(before)
    manifest = _load_json(product / "manifest.json")
    manifest["session_status"] = "error"
    manifest["artifacts"]["wandering_evidence.json"] = {
        "byte_count": len(after),
        "sha256": hashlib.sha256(after).hexdigest(),
    }
    _write_json(product / "manifest.json", manifest)
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])

    with pytest.raises(WanderingDailySummaryError, match="product"):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")


@pytest.mark.parametrize("drift", ("summary_extra", "summary_risk", "manifest_extra"))
def test_staged_summary_and_manifest_exact_schema_or_nonempty_risk_fail_without_final(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    if drift.startswith("summary"):
        original = daily_module._canonical_jsonl

        def drifting_rows(rows: object, role: str) -> bytes:
            values = [dict(row) for row in rows]  # type: ignore[arg-type]
            if role == "daily summary" and values:
                if drift == "summary_extra":
                    values[0]["rogue_summary_field"] = True
                else:
                    values[0]["risk_level"] = "high"
            return original(values, role)

        monkeypatch.setattr(daily_module, "_canonical_jsonl", drifting_rows)
    else:
        original_json = daily_module._canonical_json

        def drifting_manifest(value: object, role: str) -> bytes:
            if role == "daily summary manifest":
                value = {**value, "rogue_manifest_field": True}  # type: ignore[misc]
            return original_json(value, role)

        monkeypatch.setattr(daily_module, "_canonical_json", drifting_manifest)

    with pytest.raises(WanderingDailySummaryError):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")

    assert not (tmp_path / "daily").exists()
    assert not list(tmp_path.glob(".daily.tmp-*"))


def test_rogue_staging_and_competing_final_fail_closed_without_deleting_unrelated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    unrelated = tmp_path / ".daily.tmp-unrelated"
    unrelated.mkdir()
    original = daily_module._verify_staging

    def rogue(staging: Path, *args: object) -> None:
        (staging / "rogue.json").write_bytes(canonical_json_bytes({"rogue": True}))
        original(staging, *args)

    monkeypatch.setattr(daily_module, "_verify_staging", rogue)
    with pytest.raises(WanderingDailySummaryError, match="top-level"):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")
    assert unrelated.is_dir()
    assert not (tmp_path / "daily").exists()
    assert list(tmp_path.glob(".daily.tmp-*")) == [unrelated]

    monkeypatch.setattr(daily_module, "_verify_staging", original)

    def competing(staging: Path, *args: object) -> None:
        original(staging, *args)
        (tmp_path / "daily").mkdir()

    monkeypatch.setattr(daily_module, "_verify_staging", competing)
    with pytest.raises(FileExistsError):
        build_wandering_daily_summary([product], binding, tmp_path / "daily")
    assert (tmp_path / "daily").is_dir()
    assert list(tmp_path.glob(".daily.tmp-*")) == [unrelated]


def test_loader_rejects_rogue_product_top_level_and_builder_refuses_overwrite(
    tmp_path: Path,
) -> None:
    product = _write_product(tmp_path / "product", ["ready"])
    (product / "rogue.json").write_bytes(canonical_json_bytes({"rogue": True}))
    with pytest.raises(Exception, match="top-level"):
        load_validated_wandering_camera_product(product)
    (product / "rogue.json").unlink()
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    output = tmp_path / "daily"
    build_wandering_daily_summary([product], binding, output)
    with pytest.raises(FileExistsError):
        build_wandering_daily_summary([product], binding, output)


def test_cli_help_parameter_error_and_canonical_determinism(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for required in ("--product-bundle", "--binding-manifest", "--output"):
        assert required in help_result.stdout
    for forbidden in ("threshold", "class-order", "episode-merge", "risk", "baseline"):
        assert forbidden not in help_result.stdout.lower()
    error = subprocess.run(
        [sys.executable, str(CLI)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert error.returncode == 2

    product = _write_product(tmp_path / "product", ["ready"])
    binding = _write_binding(tmp_path / "binding.json", [_session_binding(product)])
    build_wandering_daily_summary([product], binding, tmp_path / "first")
    build_wandering_daily_summary([product], binding, tmp_path / "second")
    assert (tmp_path / "first/daily_summary.jsonl").read_bytes() == (
        tmp_path / "second/daily_summary.jsonl"
    ).read_bytes()
    assert (tmp_path / "first/manifest.json").read_bytes() == (
        tmp_path / "second/manifest.json"
    ).read_bytes()


def test_git_external_multi_session_cross_midnight_cli_e2e(tmp_path: Path) -> None:
    first = _write_product(tmp_path / "product-1", ["ready"], source_suffix="e2e-1")
    second = _write_product(tmp_path / "product-2", ["ready"], source_suffix="e2e-2")
    binding = _write_binding(
        tmp_path / "binding.json",
        [
            _session_binding(
                first,
                binding_id="SYN-BIND-E2E-001",
                session_id="SYN-SESSION-E2E-001",
                started_at="2026-08-14T23:59:30+08:00",
            ),
            _session_binding(
                second,
                binding_id="SYN-BIND-E2E-002",
                session_id="SYN-SESSION-E2E-002",
                started_at="2026-08-15T00:01:00+08:00",
            ),
        ],
    )
    output = tmp_path / "daily-e2e"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--product-bundle",
            str(first),
            "--product-bundle",
            str(second),
            "--binding-manifest",
            str(binding),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "products=2" in completed.stdout
    assert "sessions=2" in completed.stdout
    assert {path.name for path in output.iterdir()} == {"daily_summary.jsonl", "manifest.json"}
    rows = _rows(output)
    assert [row["local_date"] for row in rows] == ["2026-08-14", "2026-08-15"]
    assert [row["pacing_duration_sum_seconds"] for row in rows] == [30.0, 50.0]
    assert [row["pacing_episode_count"] for row in rows] == [1, 1]
    assert all(row["risk_level"] is None for row in rows)
    assert all(row["algorithm_event"] is None for row in rows)
    manifest = _load_json(output / "manifest.json")
    summary_payload = (output / "daily_summary.jsonl").read_bytes()
    assert manifest["artifacts"]["daily_summary.jsonl"] == {
        "byte_count": len(summary_payload),
        "sha256": hashlib.sha256(summary_payload).hexdigest(),
    }
    assert manifest["input_product_count"] == 2
    assert manifest["input_session_count"] == 2
    assert manifest["daily_summary_count"] == 2
    assert manifest["algorithm_event_emitted"] is False
    assert manifest["risk_or_alert_decision_emitted"] is False


def test_mvp2s_documentation_is_utf8_and_relevant_markdown_links_resolve() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "docs/README.md",
        ROOT / "docs/tasks/README.md",
        ROOT / "docs/modules/mental_health/README.md",
        ROOT / "docs/modules/mental_health/plans/徘徊识别技术文档2.md",
        ROOT / "docs/modules/mental_health/plans/M0-CAM-MVP-2S执行任务书.md",
        ROOT / "reports/mental_health/wandering_camera_mvp2s_v1/README.md",
        ROOT / "reports/mental_health/wandering_camera_mvp2s_v1/VERIFICATION.md",
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
                    "M0-CAM-MVP-2S",
                    "wandering_camera_mvp2s_v1",
                    "VERIFICATION.md",
                )
            ):
                continue
            relative = target.split("#", 1)[0]
            assert (path.parent / relative).resolve().exists(), f"broken link in {path}: {target}"
