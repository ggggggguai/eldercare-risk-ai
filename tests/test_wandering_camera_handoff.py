from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
)
from elderly_monitoring.modules.mental_health.wandering.camera_daily_baseline import (
    build_camera_daily_baseline_bundle,
)
from elderly_monitoring.modules.mental_health.wandering.camera_handoff import (
    HANDOFF_FILES,
    CameraHandoffError,
    build_camera_handoff_bundle,
    load_validated_camera_handoff_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_handoff_v1.yaml"
EPISODE_BUNDLE = (
    ROOT
    / "reports/mental_health/wandering_camera_episode_pipeline_w5d02_v4"
)
CONTEXT_BUNDLE = (
    ROOT
    / "reports/mental_health/wandering_camera_context_core_w5d03a_v2"
)
DISABLED_CONTEXT_BUNDLE = (
    ROOT
    / "reports/mental_health/wandering_camera_context_core_w5d03a_disabled_v2"
)
DAILY_BUNDLE = (
    ROOT
    / "reports/mental_health/wandering_camera_daily_baseline_w5d04_v1"
)
CLI = ROOT / "scripts/wandering/run_camera_handoff.py"
E2E_CLI = ROOT / "scripts/wandering/run_camera_handoff_e2e.py"


def _build(output: Path, *, run_id: str = "w5d05-test-v1"):
    return build_camera_handoff_bundle(
        project_root=ROOT,
        config_path=CONFIG,
        output_dir=output,
        run_id=run_id,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _descriptor(payload: bytes, schema_version: str | None, count: int | None) -> dict:
    return {
        "schema_version": schema_version,
        "record_count": count,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_full_b01_b02_handoff_has_exact_contract_and_public_readback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "handoff"
    result = _build(output)

    assert {path.name for path in output.iterdir()} == set(HANDOFF_FILES)
    assert result.episode_result_count == 129
    assert result.context_review_count == 82
    assert result.daily_report_count == 2
    assert result.baseline_profile_count == 2
    assert result.baseline_deviation_count == 2
    assert result.delivery_status == "algorithm_ready_with_home_smoke_pending"

    loaded = load_validated_camera_handoff_bundle(output)
    assert len(loaded.episode_results) == 129
    assert len(loaded.context_reviews) == 82
    assert len(loaded.daily_reports) == 2
    assert len(loaded.baseline_profiles) == 2
    assert len(loaded.baseline_deviations) == 2
    assert loaded.manifest["module"] == "mental_health"
    assert loaded.manifest["algorithm_event_emitted"] is False
    assert loaded.manifest["medical_diagnosis_emitted"] is False
    assert loaded.manifest["backend_or_frontend_implemented"] is False
    assert loaded.manifest["home_input_status"] == "awaiting_input"
    assert loaded.manifest["home_smoke_status"] == "not_run_input_unavailable"
    assert loaded.manifest["delivery_status"] != "home_validated"
    assert loaded.run_summary["delivery_status"] == loaded.manifest["delivery_status"]
    assert loaded.run_summary["output_counts"] == {
        "baseline_deviations": 2,
        "baseline_profiles": 2,
        "context_reviews": 82,
        "daily_reports": 2,
        "episode_results": 129,
    }
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert readme.count("conda run -n eldercare-ai") == 1
    assert "<fresh_run_id>" not in readme


def test_handoff_passthrough_is_lossless_and_ids_are_stable(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build(first, run_id="stable-w5d05-id")
    _build(second, run_id="stable-w5d05-id")

    source_files = {
        "episode_results.jsonl": EPISODE_BUNDLE / "episode_results.jsonl",
        "context_reviews.jsonl": CONTEXT_BUNDLE / "context_reviews.jsonl",
        "daily_reports.jsonl": DAILY_BUNDLE / "daily_reports.jsonl",
        "baseline_profiles.jsonl": DAILY_BUNDLE / "baseline_profiles.jsonl",
        "baseline_deviations.jsonl": DAILY_BUNDLE / "baseline_deviations.jsonl",
    }
    for name, source in source_files.items():
        assert (first / name).read_bytes() == source.read_bytes()
        assert (second / name).read_bytes() == source.read_bytes()

    first_manifest = _load_json(first / "handoff_manifest.json")
    second_manifest = _load_json(second / "handoff_manifest.json")
    assert first_manifest["handoff_id"] == second_manifest["handoff_id"]
    assert _load_json(first / "run_summary.json")["run_id"] == "stable-w5d05-id"
    assert _load_json(second / "run_summary.json")["run_id"] == "stable-w5d05-id"
    for filename in source_files:
        first_ids = [row["record_id"] for row in _load_jsonl(first / filename)]
        second_ids = [row["record_id"] for row in _load_jsonl(second / filename)]
        assert first_ids == second_ids


def test_existing_output_is_rejected_without_mutation(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _build(output)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert {path.name for path in output.iterdir()} == {"sentinel.txt"}


def test_public_loader_rejects_descriptor_tamper(tmp_path: Path) -> None:
    output = tmp_path / "handoff"
    _build(output)
    with (output / "episode_results.jsonl").open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(CameraHandoffError, match="descriptor"):
        load_validated_camera_handoff_bundle(output)


def test_public_loader_rejects_artifact_schema_descriptor_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "handoff"
    _build(output)
    manifest = _load_json(output / "handoff_manifest.json")
    manifest["artifacts"]["episode_results.jsonl"]["schema_version"] = (
        "wandering-handoff-episode-result-v999"
    )
    (output / "handoff_manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CameraHandoffError, match="artifact schema identity"):
        load_validated_camera_handoff_bundle(output)


def test_public_loader_rejects_cross_stage_identity_tamper_even_with_new_hash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "handoff"
    _build(output)
    rows = _load_jsonl(output / "context_reviews.jsonl")
    rows[0] = {**rows[0], "episode_record_id": "tampered-record-id"}
    payload = canonical_jsonl_bytes(rows)
    (output / "context_reviews.jsonl").write_bytes(payload)
    manifest = _load_json(output / "handoff_manifest.json")
    manifest["artifacts"]["context_reviews.jsonl"] = _descriptor(
        payload,
        "wandering-handoff-context-review-v2",
        len(rows),
    )
    (output / "handoff_manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CameraHandoffError, match="context.*episode"):
        load_validated_camera_handoff_bundle(output)


def test_configured_stage_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "episode-bundle"
    shutil.copytree(EPISODE_BUNDLE, copied)
    manifest_path = copied / "handoff_manifest.partial.json"
    manifest = _load_json(manifest_path)
    manifest["handoff_id"] = "tampered-stage"
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(CameraHandoffError, match="manifest sha256"):
        build_camera_handoff_bundle(
            project_root=ROOT,
            config_path=CONFIG,
            output_dir=tmp_path / "rejected",
            episode_bundle_dir=copied,
        )
    assert not (tmp_path / "rejected").exists()


def test_disabled_context_is_nonblocking_for_daily_baseline_and_handoff(
    tmp_path: Path,
) -> None:
    daily_config = yaml.safe_load(
        (ROOT / "configs/modules/wandering_camera_daily_baseline_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    daily_config["inputs"]["context_reviews"] = (
        DISABLED_CONTEXT_BUNDLE / "context_reviews.jsonl"
    ).as_posix()
    daily_config["daily_baseline_id"] = "w5d04-disabled-context-test"
    daily_config["identity"]["config_id"] = "w5d04-disabled-context-test"
    daily_config_path = tmp_path / "daily-disabled.yaml"
    daily_config_path.write_text(
        yaml.safe_dump(daily_config, sort_keys=False), encoding="utf-8"
    )
    disabled_daily = tmp_path / "daily-disabled"
    build_camera_daily_baseline_bundle(
        project_root=ROOT,
        config_path=daily_config_path,
        output_dir=disabled_daily,
        run_id="daily-disabled-test",
    )

    output = tmp_path / "handoff-disabled"
    result = build_camera_handoff_bundle(
        project_root=ROOT,
        config_path=CONFIG,
        output_dir=output,
        run_id="handoff-disabled-test",
        context_bundle_dir=DISABLED_CONTEXT_BUNDLE,
        daily_bundle_dir=disabled_daily,
        allow_unpinned_stage_overrides=True,
    )
    loaded = load_validated_camera_handoff_bundle(output)

    assert result.episode_result_count == 129
    assert result.context_review_count == 3
    assert result.daily_report_count == 2
    assert result.baseline_profile_count == 2
    assert result.baseline_deviation_count == 2
    assert all(row["status"] == "unavailable" for row in loaded.context_reviews)
    assert "context_provider_disabled" in loaded.run_summary["degraded_components"]
    assert loaded.run_summary["status"] == "uncertain"
    assert loaded.run_summary["delivery_status"] == (
        "algorithm_ready_with_home_smoke_pending"
    )


def test_no_medical_or_algorithm_event_fields_exist_in_backend_rows(
    tmp_path: Path,
) -> None:
    output = tmp_path / "handoff"
    _build(output)
    forbidden = {
        "algorithm_event",
        "diagnosis",
        "medical_diagnosis",
        "risk_decision",
        "alert_decision",
    }
    for name in (
        "episode_results.jsonl",
        "context_reviews.jsonl",
        "daily_reports.jsonl",
        "baseline_profiles.jsonl",
        "baseline_deviations.jsonl",
    ):
        for row in _load_jsonl(output / name):
            assert forbidden.isdisjoint(row)


def test_handoff_and_e2e_cli_help_expose_fresh_output_contract() -> None:
    for cli, expected in (
        (CLI, "--output-dir"),
        (E2E_CLI, "--output-root"),
    ):
        completed = subprocess.run(
            [sys.executable, str(cli), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert expected in completed.stdout
    assert "--episode-bundle" in subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
