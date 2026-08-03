"""Focused integrity tests for the ART-001 model package."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from elderly_monitoring.modules.mental_health.mood_social.model_package import (
    RUNTIME_EXPERT_NAMES,
    OFFLINE_NAMES,
    ONLINE_NAMES,
    ModelPackageConfig,
    ModelPackageError,
    _verify_checksums,
    build_model_package,
    load_model_package_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/artifacts/mood_social_model_package_v3_3_3.yaml"


def test_config_freezes_online_offline_and_safety_boundaries() -> None:
    config = load_model_package_config(CONFIG, repository_root=ROOT)
    assert tuple(config.payload["online_artifacts"]) == ONLINE_NAMES
    assert tuple(config.payload["offline_artifacts"]) == OFFLINE_NAMES
    assert config.payload["selection"]["pass_choice"] == (
        "opt_fusion_002_deployment_candidate"
    )
    assert tuple(config.payload["runtime_confidence"]["expert_reliability"]) == (
        RUNTIME_EXPERT_NAMES
    )
    assert (
        config.payload["runtime_confidence"][
            "public_dataset_id_required_at_inference"
        ]
        is False
    )
    assert config.payload["boundaries"] == {
        "backend_code_changed": False,
        "model006_online": False,
        "overwrite_upstream": False,
        "select_production_threshold": False,
        "change_attention_levels": False,
    }


def test_build_copies_all_entries_and_preserves_sources(tmp_path: Path) -> None:
    frozen = load_model_package_config(CONFIG, repository_root=ROOT)
    payload = deepcopy(frozen.payload)
    payload["output"] = {
        "package_directory": str(tmp_path / "package"),
        "report_directory": str(tmp_path / "report"),
        "overwrite": False,
    }
    config = ModelPackageConfig(ROOT, payload, CONFIG)
    result = build_model_package(config)
    assert result["status"] == "pass"
    assert result["online_entry_count"] == 7
    assert result["offline_entry_count"] == 1
    assert (config.package_directory / "offline/offline_auxiliary_models.joblib").is_file()
    assert (config.package_directory / "runtime_confidence.json").is_file()


def test_checksum_verification_rejects_unlisted_file(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text("{}\n", encoding="utf-8")
    (package / "extra.txt").write_text("unexpected\n", encoding="utf-8")
    checksum = package / "SHA256SUMS"
    manifest_hash = hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
    checksum.write_text(
        f"{manifest_hash}  manifest.json\n",
        encoding="ascii",
    )
    with pytest.raises(ModelPackageError, match="coverage"):
        _verify_checksums(package, checksum)
