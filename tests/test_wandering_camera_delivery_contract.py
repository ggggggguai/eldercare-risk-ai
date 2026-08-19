from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from elderly_monitoring.modules.mental_health.wandering.camera_delivery_contract import (
    DELIVERY_CONTRACT_CONFIG_SCHEMA_VERSION,
    DEVELOPMENT_INDEX_SCHEMA_VERSION,
    DeliveryContractError,
    build_wandering_camera_delivery_contract,
    load_delivery_contract_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/modules/wandering_camera_5d_contract_v1.yaml"


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(root / relative)}


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    raw = tmp_path / "raw"
    project.mkdir()
    (raw / "B01/per_video_xml").mkdir(parents=True)
    (project / "artifacts/tracking/vid-b01-0002").mkdir(parents=True)
    (project / "artifacts/truth/vid-b01-0002").mkdir(parents=True)
    (project / "schemas").mkdir()

    video = raw / "B01/VID-B01-0002_tracking-smoke.mp4"
    video.write_bytes(b"synthetic-video")
    cvat = raw / "B01/per_video_xml/VID-B01-0002_tracking-smoke.xml"
    cvat.write_text("<annotations/>\n", encoding="utf-8")

    tracking = project / "artifacts/tracking/vid-b01-0002/tracking.jsonl"
    tracking.write_bytes(
        _canonical_json(
            {
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "frame_id": 0,
                "timestamp_sec": 0.0,
                "track_confidence": 0.9,
                "track_id": 1,
            }
        )
    )
    sidecar = project / "artifacts/tracking/vid-b01-0002/media_sidecar.json"
    sidecar.write_bytes(
        _canonical_json(
            {
                "schema_version": "wandering-media-v1",
                "source_video_id": "vid-b01-0002",
                "source_group_id": "m0cam-b01-development",
                "device_id": "ezviz-c6c",
                "setup_id": "C6C-OFFICE-01",
                "stream_epoch": "vid-b01-0002-epoch-01",
                "media_ref": "B01/VID-B01-0002_tracking-smoke.mp4",
                "source_sha256": _sha256(video),
                "tracking_jsonl_sha256": _sha256(tracking),
                "timezone": "Asia/Shanghai",
            }
        )
    )
    truth = project / "artifacts/truth/vid-b01-0002/episode_truth.jsonl"
    truth.write_bytes(
        _canonical_json(
            {
                "schema_version": "wandering-camera-episode-truth-v1",
                "episode_id": "vid-b01-0002-cvat-1",
                "source_video_id": "vid-b01-0002",
            }
        )
    )

    primary = project / "primary.yaml"
    primary.write_text("schema_version: primary-fixture-v1\n", encoding="utf-8")
    camera = project / "camera.yaml"
    camera.write_text("schema_version: camera-fixture-v1\n", encoding="utf-8")
    episode = project / "episode.yaml"
    episode.write_text("schema_version: episode-fixture-v1\n", encoding="utf-8")
    preprocessing = project / "preprocessing.yaml"
    preprocessing.write_text("schema_version: preprocessing-fixture-v1\n", encoding="utf-8")
    candidate = project / "candidate_manifest.json"
    candidate.write_bytes(
        _canonical_json(
            {
                "schema_version": "candidate-fixture-v1",
                "candidate_id": "candidate-fixture",
                "artifacts": {"model_state": {"sha256": "1" * 64}},
                "inference_contract": {
                    "binary_decision": "sigmoid >= 0.5",
                    "input_shapes": {
                        "model_features": [80, 14],
                        "point_mask": [80],
                        "shape_normalized_points": [80, 2],
                    },
                },
            }
        )
    )
    contract_doc = project / "output_contract.md"
    contract_doc.write_text("# Fixture output contract\n", encoding="utf-8")

    schema_names = {
        "development_index": "wandering-camera-development-index-v1",
        "contract_manifest": "wandering-camera-5d-contract-manifest-v1",
        "episode_results": "wandering-handoff-episode-result-v1",
        "context_reviews": "wandering-handoff-context-review-v1",
        "daily_reports": "wandering-handoff-daily-report-v1",
        "baseline_profiles": "wandering-handoff-baseline-profile-v1",
        "baseline_deviations": "wandering-handoff-baseline-deviation-v1",
        "handoff_manifest": "wandering-handoff-manifest-v1",
        "run_summary": "wandering-handoff-run-summary-v1",
    }
    schema_paths: dict[str, str] = {}
    for name, schema_id in schema_names.items():
        path = project / f"schemas/{name}.schema.json"
        path.write_bytes(
            _canonical_json(
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$id": schema_id,
                    "type": "object",
                }
            )
        )
        schema_paths[name] = path.relative_to(project).as_posix()

    config = {
        "schema_version": DELIVERY_CONTRACT_CONFIG_SCHEMA_VERSION,
        "contract_id": "fixture-contract-v1",
        "development_index": {
            "schema_version": DEVELOPMENT_INDEX_SCHEMA_VERSION,
            "dataset_role": "development",
            "default_source_root": raw.as_posix(),
            "batches": [
                {
                    "batch_id": "B01",
                    "participant_id": "P-OFFICE-01",
                    "session_id": "S-20260814",
                    "camera_setup_id": "C6C-OFFICE-01",
                    "clock_domain_id": "CLOCK-OFFICE-01",
                    "expected_development_video_count": 1,
                    "accepted_legacy_setup_ids": [],
                }
            ],
            "artifact_sets": [
                {
                    "batch_id": "B01",
                    "tracking_root": "artifacts/tracking",
                    "truth_root": "artifacts/truth",
                    "historical_role": "development",
                }
            ],
            "excluded_inputs": [],
        },
        "home_acceptance_input": {
            "status": "awaiting_input",
            "role": "real_scene_acceptance_then_development_if_tuned",
        },
        "fixed_primary": {
            "candidate_id": "candidate-fixture",
            "candidate_manifest": _descriptor(project, "candidate_manifest.json"),
            "model_state_sha256": "1" * 64,
            "primary_config": _descriptor(project, "primary.yaml"),
            "camera_preprocessing": {
                "input_point_count": 80,
                "model_feature_count": 14,
                "shape_coordinate_count": 2,
                "binary_decision_threshold": 0.5,
                "files": [
                    _descriptor(project, "camera.yaml"),
                    _descriptor(project, "episode.yaml"),
                    _descriptor(project, "preprocessing.yaml"),
                ],
            },
        },
        "runtime": {
            "device": "cpu",
            "intra_op_threads": 8,
            "inter_op_threads": 1,
            "maximum_batch_size": 64,
        },
        "output_contract": {
            "document_path": "output_contract.md",
            "fresh_output_root": "tmp/wandering_camera_5d_runs",
            "freshness_policy": "final_output_directory_must_not_exist",
            "statuses": ["ready", "uncertain", "unavailable", "error"],
            "required_files": [
                "episode_results.jsonl",
                "context_reviews.jsonl",
                "daily_reports.jsonl",
                "baseline_profiles.jsonl",
                "baseline_deviations.jsonl",
                "handoff_manifest.json",
                "run_summary.json",
                "README.md",
            ],
            "schemas": schema_paths,
        },
    }
    config_path = project / "contract.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return project, config_path


def _fake_primary_verifier(**_: Any) -> dict[str, Any]:
    return {
        "candidate_cpu_load": "passed",
        "runtime_device": "cpu",
        "model_mode": "eval",
    }


def test_builder_verifies_bindings_and_commits_a_fresh_contract_bundle(
    tmp_path: Path,
) -> None:
    project, config_path = _write_fixture(tmp_path)
    output = tmp_path / "contract-output"

    result = build_wandering_camera_delivery_contract(
        project_root=project,
        config_path=config_path,
        output_dir=output,
        primary_runtime_verifier=_fake_primary_verifier,
    )

    assert result.development_record_count == 1
    assert result.batch_counts == {"B01": 1}
    row = json.loads((output / "development_index.jsonl").read_text(encoding="utf-8"))
    assert row["schema_version"] == DEVELOPMENT_INDEX_SCHEMA_VERSION
    assert row["dataset_role"] == "development"
    assert row["development_reuse_allowed"] is True
    assert row["source_video_id"] == "vid-b01-0002"
    assert row["sidecar_setup_id"] == "C6C-OFFICE-01"
    assert row["setup_binding_status"] == "consistent"
    for field in (
        "video",
        "tracking",
        "media_sidecar",
        "truth",
        "cvat_xml",
    ):
        locator = row["artifacts"][field]
        path = Path(locator["path"])
        assert path.is_file()
        assert locator["sha256"] == _sha256(path)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["development_index"]["record_count"] == 1
    assert manifest["fixed_primary"]["candidate_id"] == "candidate-fixture"
    assert manifest["fixed_primary"]["input_point_count"] == 80
    assert manifest["fixed_primary"]["binary_decision_threshold"] == 0.5
    assert manifest["primary_runtime_verification"]["runtime_device"] == "cpu"
    assert manifest["fresh_output_policy"]["overwrite_allowed"] is False
    assert set(manifest["output_contract"]["schemas"]) == set(
        load_delivery_contract_config(config_path)["output_contract"]["schemas"]
    )
    assert (output / "baseline_verification.txt").read_text(encoding="utf-8").endswith(
        "non_overwrite_policy=passed\n"
    )

    with pytest.raises(FileExistsError):
        build_wandering_camera_delivery_contract(
            project_root=project,
            config_path=config_path,
            output_dir=output,
            primary_runtime_verifier=_fake_primary_verifier,
        )


def test_builder_fails_closed_before_commit_when_tracking_hash_drifts(
    tmp_path: Path,
) -> None:
    project, config_path = _write_fixture(tmp_path)
    tracking = project / "artifacts/tracking/vid-b01-0002/tracking.jsonl"
    tracking.write_bytes(tracking.read_bytes() + b"{}\n")
    output = tmp_path / "bad-output"

    with pytest.raises(DeliveryContractError, match="tracking"):
        build_wandering_camera_delivery_contract(
            project_root=project,
            config_path=config_path,
            output_dir=output,
            primary_runtime_verifier=_fake_primary_verifier,
        )
    assert not output.exists()


def test_builder_rejects_an_undeclared_sidecar_setup_alias(tmp_path: Path) -> None:
    project, config_path = _write_fixture(tmp_path)
    sidecar_path = project / "artifacts/tracking/vid-b01-0002/media_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["setup_id"] = "unexpected-setup"
    sidecar_path.write_bytes(_canonical_json(sidecar))

    with pytest.raises(DeliveryContractError, match="camera setup binding drifted"):
        build_wandering_camera_delivery_contract(
            project_root=project,
            config_path=config_path,
            output_dir=tmp_path / "bad-setup-output",
            primary_runtime_verifier=_fake_primary_verifier,
        )


def test_builder_rejects_output_schema_identity_drift_before_commit(
    tmp_path: Path,
) -> None:
    project, config_path = _write_fixture(tmp_path)
    schema_path = project / "schemas/episode_results.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["$id"] = "drifted-episode-result-schema"
    schema_path.write_bytes(_canonical_json(schema))
    output = tmp_path / "bad-schema-output"

    with pytest.raises(DeliveryContractError, match="episode_results schema identity drifted"):
        build_wandering_camera_delivery_contract(
            project_root=project,
            config_path=config_path,
            output_dir=output,
            primary_runtime_verifier=_fake_primary_verifier,
        )
    assert not output.exists()


def test_production_contract_freezes_primary_runtime_and_handoff_schema_set() -> None:
    config = load_delivery_contract_config(CONFIG)

    assert config["fixed_primary"]["candidate_id"] == (
        "topowander-m0s-seed20260731-epoch0005"
    )
    assert config["fixed_primary"]["camera_preprocessing"] == {
        "input_point_count": 80,
        "model_feature_count": 14,
        "shape_coordinate_count": 2,
        "binary_decision_threshold": 0.5,
        "files": config["fixed_primary"]["camera_preprocessing"]["files"],
    }
    assert config["runtime"]["device"] == "cpu"
    assert config["output_contract"]["statuses"] == [
        "ready",
        "uncertain",
        "unavailable",
        "error",
    ]
    assert config["output_contract"]["required_files"] == [
        "episode_results.jsonl",
        "context_reviews.jsonl",
        "daily_reports.jsonl",
        "baseline_profiles.jsonl",
        "baseline_deviations.jsonl",
        "handoff_manifest.json",
        "run_summary.json",
        "README.md",
    ]

    expected_schema_ids = {
        "development_index": "wandering-camera-development-index-v1",
        "contract_manifest": "wandering-camera-5d-contract-manifest-v1",
        "episode_results": "wandering-handoff-episode-result-v1",
        "context_reviews": "wandering-handoff-context-review-v1",
        "daily_reports": "wandering-handoff-daily-report-v1",
        "baseline_profiles": "wandering-handoff-baseline-profile-v1",
        "baseline_deviations": "wandering-handoff-baseline-deviation-v1",
        "handoff_manifest": "wandering-handoff-manifest-v1",
        "run_summary": "wandering-handoff-run-summary-v1",
    }
    assert set(config["output_contract"]["schemas"]) == set(expected_schema_ids)
    for name, schema_id in expected_schema_ids.items():
        schema_path = ROOT / config["output_contract"]["schemas"][name]
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == schema_id

    episode_schema_path = ROOT / config["output_contract"]["schemas"]["episode_results"]
    episode_schema = json.loads(episode_schema_path.read_text(encoding="utf-8"))
    assert {
        "technical_segment_index",
        "duration_seconds",
        "reason_codes",
    }.issubset(episode_schema["required"])
    assert episode_schema["properties"]["run_status"] == {
        "enum": ["auto_accepted", "uncertain", "rejected"]
    }
