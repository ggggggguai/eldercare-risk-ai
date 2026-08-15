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

import test_wandering_camera_baseline_preview as baseline_fixture
from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_baseline_preview import (
    METRIC_ORDER,
    build_wandering_baseline_preview,
)
from elderly_monitoring.modules.mental_health.wandering.camera_baseline_deviation_preview import (
    BASELINE_DEVIATION_PREVIEW_MANIFEST_SCHEMA_VERSION,
    BASELINE_DEVIATION_PREVIEW_SCHEMA_VERSION,
    MVP3D_PRODUCER_SOURCE_SHA256,
    WanderingBaselineDeviationPreviewError,
    build_wandering_baseline_deviation_preview,
    load_validated_wandering_baseline_deviation_preview,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/wandering/build_wandering_baseline_deviation_preview.py"
ROW_KEYS = {
    "schema_version", "product_name", "product_stage", "status", "evidence_scope",
    "validation_scope", "synthetic_person_id", "timezone", "observation_local_date",
    "person_binding_verified", "profile_contract_id", "profile_identity",
    "profile_readiness_status", "profile_window_last_local_date",
    "baseline_manifest_sha256", "observation_daily_manifest_sha256",
    "observation_any_window_covered_seconds", "deviation_preview_status", "metric_order",
    "baseline_deviation_preview", "real_baseline_ready", "baseline_deviation",
    "eligible_for_risk", "risk_level", "risk_score", "recommended_action",
    "alert_decision", "medical_diagnosis", "algorithm_event", "algorithm_event_emitted",
}
METRIC_KEYS = {
    "status", "observation_value", "reference_count", "reference_median",
    "reference_p90", "delta_from_median", "delta_from_p90",
}
MANIFEST_KEYS = {
    "schema_version", "product_name", "product_stage", "status", "evidence_scope",
    "validation_scope", "artifacts", "baseline_manifest",
    "observation_daily_manifests", "observation_bundle_count", "observation_row_count",
    "synthetic_person_count", "observation_local_date_count", "deviation_preview_count",
    "baseline_profile_contract_ids", "metric_order", "delta_semantics",
    "builder_source_sha256", "person_binding_verified", "real_baseline_emitted",
    "baseline_deviation_emitted", "risk_or_alert_decision_emitted",
    "algorithm_event_emitted", "real_human_media_consumed", "m0cam_d_started",
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(canonical_jsonl_bytes(rows))


def _history(
    parent: Path,
    *,
    day_count: int = 3,
    start: str = "2026-07-01",
    specs: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    daily = baseline_fixture._daily_bundle(
        parent,
        "history-daily",
        specs if specs is not None else baseline_fixture._day_specs(day_count, start=start),
    )
    baseline = parent / "baseline"
    build_wandering_baseline_preview([daily], baseline)
    return daily, baseline


def _observation(
    parent: Path,
    name: str = "observation",
    *,
    local_date: str = "2026-07-04",
    specs: list[dict[str, object]] | None = None,
    identity_overrides: dict[str, object] | None = None,
    config_sha256: str | None = None,
) -> Path:
    return baseline_fixture._daily_bundle(
        parent,
        name,
        specs if specs is not None else [{"local_date": local_date}],
        identity_overrides=identity_overrides,
        config_sha256=config_sha256,
    )


def _rehash_rows(bundle: Path) -> None:
    manifest = _load_json(bundle / "manifest.json")
    payload = (bundle / "baseline_deviation_preview.jsonl").read_bytes()
    manifest["artifacts"]["baseline_deviation_preview.jsonl"] = {
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    _write_json(bundle / "manifest.json", manifest)


def _rewrite_row(
    bundle: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    rows = _load_jsonl(bundle / "baseline_deviation_preview.jsonl")
    mutation(rows[0])
    _write_jsonl(bundle / "baseline_deviation_preview.jsonl", rows)
    _rehash_rows(bundle)


def _metric_values(row: dict[str, object]) -> dict[str, float | None]:
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


def test_builder_loader_and_cli_contract_exist_and_round_trip(tmp_path: Path) -> None:
    _, baseline = _history(tmp_path)
    observation = _observation(tmp_path)
    output = tmp_path / "deviation"

    result = build_wandering_baseline_deviation_preview(baseline, [observation], output)
    validated = load_validated_wandering_baseline_deviation_preview(output)

    assert result.output_dir == output
    assert validated.bundle_dir == output.resolve()
    assert {path.name for path in output.iterdir()} == {
        "baseline_deviation_preview.jsonl",
        "manifest.json",
    }
    [row] = validated.rows
    manifest = validated.manifest
    assert set(row) == ROW_KEYS
    assert set(row["baseline_deviation_preview"]) == set(METRIC_ORDER)
    assert all(set(metric) == METRIC_KEYS for metric in row["baseline_deviation_preview"].values())
    assert set(manifest) == MANIFEST_KEYS
    assert row["schema_version"] == BASELINE_DEVIATION_PREVIEW_SCHEMA_VERSION
    assert manifest["schema_version"] == BASELINE_DEVIATION_PREVIEW_MANIFEST_SCHEMA_VERSION
    assert row["product_name"] == "WanderingBaselineDeviationPreview"
    assert manifest["builder_source_sha256"] == MVP3D_PRODUCER_SOURCE_SHA256


@pytest.mark.parametrize("reference_days", (3, 7))
def test_initial_and_stable_profiles_compute_all_signed_deltas_independently(
    reference_days: int,
    tmp_path: Path,
) -> None:
    _, baseline = _history(tmp_path, day_count=reference_days)
    observation = _observation(
        tmp_path,
        local_date=(date(2026, 7, 1) + timedelta(days=reference_days)).isoformat(),
        specs=[{
            "local_date": (date(2026, 7, 1) + timedelta(days=reference_days)).isoformat(),
            "presence_seconds": 100.0,
            "any_window_covered_seconds": 50.0,
            "episode_counts": {"direct": 1, "pacing": 2, "lapping": 3, "random": 4},
            "episode_durations": {"direct": 10.0, "pacing": 200.0, "lapping": 50.0},
            "night_duration": 25.0,
        }],
    )
    build_wandering_baseline_deviation_preview(baseline, [observation], tmp_path / "out")

    [daily_row] = _load_jsonl(observation / "daily_summary.jsonl")
    [profile] = _load_jsonl(baseline / "baseline_profiles.jsonl")
    [row] = _load_jsonl(tmp_path / "out/baseline_deviation_preview.jsonl")
    expected_values = _metric_values(daily_row)
    assert row["deviation_preview_status"] == "ready_preview"
    for name in METRIC_ORDER:
        metric = row["baseline_deviation_preview"][name]
        stats = profile["metric_stats"][name]
        assert metric["status"] == "ready"
        assert metric["observation_value"] == pytest.approx(expected_values[name])
        assert metric["delta_from_median"] == pytest.approx(
            expected_values[name] - stats["median"]
        )
        assert metric["delta_from_p90"] == pytest.approx(
            expected_values[name] - stats["p90"]
        )
    assert row["baseline_deviation_preview"][
        "wandering_like_candidate_duration_seconds_per_presence_hour"
    ]["observation_value"] > 3600.0


@pytest.mark.parametrize(
    ("case", "code"),
    (
        ("same_day", "reference_window_not_prior"),
        ("future_reference", "reference_window_not_prior"),
        ("empty_reference", "reference_window_not_prior"),
        ("manifest_reuse", "observation_manifest_reused_as_reference"),
        ("missing_profile", "baseline_profile_not_found"),
        ("duplicate_day", "duplicate_observation_day"),
    ),
)
def test_reference_is_strictly_prior_unique_and_not_reused(
    case: str,
    code: str,
    tmp_path: Path,
) -> None:
    history_daily, baseline = _history(tmp_path)
    observations: list[Path]
    if case == "same_day":
        observations = [_observation(tmp_path, local_date="2026-07-03")]
    elif case == "future_reference":
        observations = [_observation(tmp_path, local_date="2026-07-02")]
    elif case == "empty_reference":
        _, baseline = _history(
            tmp_path / "empty",
            specs=[{
                "local_date": "2026-07-01",
                "any_window_covered_seconds": 0.0,
                "episode_counts": {},
                "episode_durations": {},
            }],
        )
        observations = [_observation(tmp_path / "empty", local_date="2026-07-02")]
    elif case == "manifest_reuse":
        observations = [history_daily]
    elif case == "missing_profile":
        observations = [_observation(
            tmp_path,
            specs=[{"local_date": "2026-07-04", "synthetic_person_id": "SYN-PERSON-999"}],
        )]
    else:
        observations = [
            _observation(tmp_path, "observation-a"),
            _observation(tmp_path, "observation-b"),
        ]
    with pytest.raises(WanderingBaselineDeviationPreviewError) as captured:
        build_wandering_baseline_deviation_preview(baseline, observations, tmp_path / "rejected")
    assert captured.value.code == code
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    ("identity_overrides", "config_sha256"),
    (
        ({"candidate_id": "changed"}, None),
        ({"candidate_manifest_sha256": "1" * 64}, None),
        ({"model_state_sha256": "2" * 64}, None),
        ({"primary_seed": 99}, None),
        ({"best_epoch": 99}, None),
        ({"binary_decision_threshold": 0.6}, None),
        ({"episode_policy_status": "changed"}, None),
        ({"episode_merge_gap_seconds": 2.0}, None),
        ({"night_start": "21:00"}, None),
        ({}, "3" * 64),
        ({"duration_semantics": "presence_time"}, None),
        ({"four_class_order": ["pacing", "direct", "lapping", "random"]}, None),
        ({"probability_calibrated": True}, None),
    ),
)
def test_complete_observation_identity_drift_requires_version_reset(
    identity_overrides: dict[str, object],
    config_sha256: str | None,
    tmp_path: Path,
) -> None:
    _, baseline = _history(tmp_path)
    observation = _observation(
        tmp_path,
        identity_overrides=identity_overrides,
        config_sha256=config_sha256,
    )
    with pytest.raises(WanderingBaselineDeviationPreviewError) as captured:
        build_wandering_baseline_deviation_preview(
            baseline, [observation], tmp_path / "rejected"
        )
    assert captured.value.code == "baseline_version_reset_required"


def test_observation_timezone_drift_requires_version_reset(tmp_path: Path) -> None:
    _, baseline = _history(tmp_path)
    observation = _observation(
        tmp_path,
        specs=[{"local_date": "2026-07-04", "timezone": "Asia/Tokyo"}],
    )
    with pytest.raises(WanderingBaselineDeviationPreviewError) as captured:
        build_wandering_baseline_deviation_preview(
            baseline, [observation], tmp_path / "rejected"
        )
    assert captured.value.code == "baseline_version_reset_required"


def test_warming_observation_unavailable_and_nullable_metric_statuses(tmp_path: Path) -> None:
    _, warming = _history(tmp_path / "warming", day_count=2)
    warm_obs = _observation(tmp_path / "warming", local_date="2026-07-03")
    build_wandering_baseline_deviation_preview(warming, [warm_obs], tmp_path / "warm")
    [warm_row] = _load_jsonl(tmp_path / "warm/baseline_deviation_preview.jsonl")
    assert warm_row["deviation_preview_status"] == "profile_warming_up"
    assert all(
        metric["status"] == "profile_warming_up"
        and metric["delta_from_median"] is None
        and metric["delta_from_p90"] is None
        for metric in warm_row["baseline_deviation_preview"].values()
    )

    _, baseline = _history(tmp_path / "unavailable")
    unavailable = _observation(
        tmp_path / "unavailable",
        specs=[{
            "local_date": "2026-07-04", "any_window_covered_seconds": 0.0,
            "episode_counts": {"pacing": 9}, "episode_durations": {"pacing": 900.0},
        }],
    )
    build_wandering_baseline_deviation_preview(baseline, [unavailable], tmp_path / "no-cover")
    [unavailable_row] = _load_jsonl(tmp_path / "no-cover/baseline_deviation_preview.jsonl")
    assert unavailable_row["deviation_preview_status"] == "observation_unavailable"
    assert all(
        metric["status"] == "observation_unavailable"
        and metric["observation_value"] is None
        and metric["delta_from_median"] is None
        and metric["delta_from_p90"] is None
        for metric in unavailable_row["baseline_deviation_preview"].values()
    )


def test_night_null_and_reference_count_zero_never_become_zero_delta(tmp_path: Path) -> None:
    zero_specs = [
        {**day, "episode_counts": {}, "episode_durations": {}}
        for day in baseline_fixture._day_specs(3)
    ]
    _, baseline = _history(tmp_path, specs=zero_specs)
    null_observation = _observation(
        tmp_path,
        "null-observation",
        specs=[{"local_date": "2026-07-04", "episode_counts": {}, "episode_durations": {}}],
    )
    build_wandering_baseline_deviation_preview(
        baseline, [null_observation], tmp_path / "null-out"
    )
    [null_row] = _load_jsonl(tmp_path / "null-out/baseline_deviation_preview.jsonl")
    null_metric = null_row["baseline_deviation_preview"]["night_wandering_like_ratio"]
    assert null_metric["status"] == "observation_value_unavailable"
    assert null_metric["reference_count"] == 0
    assert null_metric["delta_from_median"] is None
    assert null_metric["delta_from_p90"] is None

    observed = _observation(
        tmp_path,
        "observed",
        specs=[{
            "local_date": "2026-07-05", "episode_counts": {"pacing": 1},
            "episode_durations": {"pacing": 100.0}, "night_duration": 25.0,
        }],
    )
    build_wandering_baseline_deviation_preview(baseline, [observed], tmp_path / "ref-zero")
    [ref_row] = _load_jsonl(tmp_path / "ref-zero/baseline_deviation_preview.jsonl")
    ref_metric = ref_row["baseline_deviation_preview"]["night_wandering_like_ratio"]
    assert ref_metric["status"] == "reference_unavailable"
    assert ref_metric["observation_value"] == 0.25
    assert ref_metric["delta_from_median"] is None
    assert ref_metric["delta_from_p90"] is None


@pytest.mark.parametrize(
    "attack",
    (
        "row_missing", "row_extra", "nested_missing", "nested_extra", "wrong_delta",
        "nonempty_risk", "nonempty_event", "manifest_missing", "manifest_extra",
        "rogue_top_level",
    ),
)
def test_descriptor_aware_exact_schema_decision_and_nested_delta_attacks_fail_closed(
    attack: str,
    tmp_path: Path,
) -> None:
    _, baseline = _history(tmp_path)
    observation = _observation(tmp_path)
    output = tmp_path / "out"
    build_wandering_baseline_deviation_preview(baseline, [observation], output)
    if attack == "rogue_top_level":
        _write_json(output / "rogue.json", {"rogue": True})
    elif attack.startswith("manifest_"):
        manifest = _load_json(output / "manifest.json")
        if attack == "manifest_missing":
            manifest.pop("product_name")
        else:
            manifest["rogue"] = True
        _write_json(output / "manifest.json", manifest)
    else:
        def mutate(row: dict[str, object]) -> None:
            if attack == "row_missing":
                row.pop("product_name")
            elif attack == "row_extra":
                row["rogue"] = True
            elif attack == "nested_missing":
                row["baseline_deviation_preview"][METRIC_ORDER[0]].pop("reference_p90")
            elif attack == "nested_extra":
                row["baseline_deviation_preview"][METRIC_ORDER[0]]["abnormal"] = True
            elif attack == "wrong_delta":
                row["baseline_deviation_preview"][METRIC_ORDER[0]]["delta_from_median"] += 0.5
            elif attack == "nonempty_risk":
                row["risk_level"] = "high"
            else:
                row["algorithm_event"] = {"module": "mental_health"}
        _rewrite_row(output, mutate)
    with pytest.raises(WanderingBaselineDeviationPreviewError):
        load_validated_wandering_baseline_deviation_preview(output)


def test_input_order_is_canonical_and_risk_boundary_remains_empty(tmp_path: Path) -> None:
    _, baseline = _history(tmp_path)
    first = _observation(tmp_path, "obs-a", local_date="2026-07-04")
    second = _observation(tmp_path, "obs-b", local_date="2026-07-05")
    build_wandering_baseline_deviation_preview(baseline, [first, second], tmp_path / "forward")
    build_wandering_baseline_deviation_preview(baseline, [second, first], tmp_path / "reverse")
    for filename in ("baseline_deviation_preview.jsonl", "manifest.json"):
        assert (tmp_path / "forward" / filename).read_bytes() == (
            tmp_path / "reverse" / filename
        ).read_bytes()
    for row in _load_jsonl(tmp_path / "forward/baseline_deviation_preview.jsonl"):
        assert row["person_binding_verified"] is False
        assert row["real_baseline_ready"] is False
        assert row["baseline_deviation"] is None
        assert row["eligible_for_risk"] is False
        assert row["algorithm_event_emitted"] is False
        for field in (
            "risk_level", "risk_score", "recommended_action", "alert_decision",
            "medical_diagnosis", "algorithm_event",
        ):
            assert row[field] is None


def test_producer_prefix_marker_identity_and_active_source_drift_fail_without_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elderly_monitoring.modules.mental_health.wandering import (
        camera_baseline_deviation_preview as module,
    )

    source = Path(module.__file__).resolve()
    payload = source.read_bytes()
    marker = module._MVP3D_LOADER_MARKER
    assert payload.count(marker) == 1
    marker_index = payload.index(marker)
    assert payload[marker_index - 2:marker_index] == b"\n\n"
    producer_prefix = payload[: marker_index - 1]
    assert hashlib.sha256(producer_prefix).hexdigest() == MVP3D_PRODUCER_SOURCE_SHA256

    _, baseline = _history(tmp_path)
    observation = _observation(tmp_path)
    original_read_bytes = Path.read_bytes
    mutated = payload.replace(
        b'PRODUCT_NAME = "WanderingBaselineDeviationPreview"',
        b'PRODUCT_NAME = "WanderingBaselineDeviationPreviews"',
        1,
    )

    def read_bytes(path: Path) -> bytes:
        if path.resolve() == source:
            return mutated
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(WanderingBaselineDeviationPreviewError) as captured:
        build_wandering_baseline_deviation_preview(
            baseline, [observation], tmp_path / "rejected"
        )
    assert captured.value.code == "active_builder_source_mismatch"
    assert not (tmp_path / "rejected").exists()


def test_fresh_only_owned_staging_cleanup_cli_help_and_e2e(tmp_path: Path) -> None:
    _, baseline = _history(tmp_path, day_count=7)
    observation = _observation(tmp_path, local_date="2026-07-08")
    unrelated = tmp_path / ".out.tmp-unrelated"
    unrelated.mkdir()
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable, str(CLI), "--baseline-bundle", str(baseline),
            "--observation-daily-bundle", str(observation), "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "previews=1" in completed.stdout
    load_validated_wandering_baseline_deviation_preview(output)
    with pytest.raises(FileExistsError):
        build_wandering_baseline_deviation_preview(baseline, [observation], output)
    assert unrelated.is_dir()

    help_result = subprocess.run(
        [sys.executable, str(CLI), "--help"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    for required in ("--baseline-bundle", "--observation-daily-bundle", "--output"):
        assert required in help_result.stdout
    for forbidden in ("threshold", "risk-policy", "reference-date", "stats-json"):
        assert forbidden not in help_result.stdout.lower()


def test_git_external_seven_day_cli_e2e_recomputes_deltas_and_isolates_attacks(
    tmp_path: Path,
) -> None:
    history_daily, baseline = _history(tmp_path, day_count=7)
    observation = _observation(
        tmp_path,
        local_date="2026-07-08",
        specs=[{
            "local_date": "2026-07-08",
            "presence_seconds": 100.0,
            "any_window_covered_seconds": 25.0,
            "episode_counts": {"direct": 2, "pacing": 1, "random": 3},
            "episode_durations": {"direct": 10.0, "pacing": 150.0, "random": 50.0},
            "night_duration": 40.0,
        }],
    )
    valid = tmp_path / "valid-final"
    subprocess.run(
        [
            sys.executable, str(CLI), "--baseline-bundle", str(baseline),
            "--observation-daily-bundle", str(observation), "--output", str(valid),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    [daily_row] = _load_jsonl(observation / "daily_summary.jsonl")
    [profile] = _load_jsonl(baseline / "baseline_profiles.jsonl")
    [preview] = _load_jsonl(valid / "baseline_deviation_preview.jsonl")
    independent_values = _metric_values(daily_row)
    for name in METRIC_ORDER:
        metric = preview["baseline_deviation_preview"][name]
        stats = profile["metric_stats"][name]
        assert metric["status"] == "ready"
        assert metric["observation_value"] == pytest.approx(independent_values[name])
        assert metric["delta_from_median"] == pytest.approx(
            independent_values[name] - stats["median"]
        )
        assert metric["delta_from_p90"] == pytest.approx(
            independent_values[name] - stats["p90"]
        )

    same_day = _observation(tmp_path, "same-day", local_date="2026-07-07")
    with pytest.raises(WanderingBaselineDeviationPreviewError) as captured:
        build_wandering_baseline_deviation_preview(
            baseline, [same_day], tmp_path / "same-day-rejected"
        )
    assert captured.value.code == "reference_window_not_prior"
    with pytest.raises(WanderingBaselineDeviationPreviewError) as captured:
        build_wandering_baseline_deviation_preview(
            baseline, [history_daily], tmp_path / "reuse-rejected"
        )
    assert captured.value.code == "observation_manifest_reused_as_reference"

    attacked = tmp_path / "attacked-copy"
    shutil.copytree(valid, attacked)

    def tamper(row: dict[str, object]) -> None:
        row["baseline_deviation_preview"][METRIC_ORDER[0]]["delta_from_p90"] += 0.25

    _rewrite_row(attacked, tamper)
    with pytest.raises(WanderingBaselineDeviationPreviewError):
        load_validated_wandering_baseline_deviation_preview(attacked)
    assert load_validated_wandering_baseline_deviation_preview(valid).preview_count == 1


def test_mvp3d_documentation_is_utf8_and_markdown_links_resolve() -> None:
    paths = (
        ROOT / "README.md",
        ROOT / "docs/README.md",
        ROOT / "docs/tasks/README.md",
        ROOT / "docs/modules/mental_health/README.md",
        ROOT / "docs/modules/mental_health/plans/徘徊识别技术文档2.md",
        ROOT / "docs/modules/mental_health/plans/M0-CAM-MVP-3D执行任务书.md",
        ROOT / "reports/mental_health/wandering_camera_mvp3d_v1/README.md",
        ROOT / "reports/mental_health/wandering_camera_mvp3d_v1/VERIFICATION.md",
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
                    "M0-CAM-MVP-3D",
                    "wandering_camera_mvp3d_v1",
                    "VERIFICATION.md",
                )
            ):
                continue
            relative = target.split("#", 1)[0]
            assert (path.parent / relative).resolve().exists(), (
                f"broken link in {path}: {target}"
            )
