from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from elderly_monitoring.modules.mental_health.wandering import camera_portability
from elderly_monitoring.modules.mental_health.wandering.camera_portability import (
    BLOCKER_ASSET_SOURCE_UNAVAILABLE,
    CANONICAL_CONTRACT_RELATIVE_PATH,
    PortableAssetBlockedError,
    PortableContractError,
    build_runtime_asset_bundle,
    load_portable_contract,
    preflight_camera_runtime,
    restore_camera_runtime_assets,
)


ROOT = Path(__file__).resolve().parents[1]
_REAL_ENFORCE_FROZEN_IDENTITY = camera_portability._enforce_frozen_scientific_identity
_REAL_ENFORCE_SOURCE_IDENTITY = camera_portability._enforce_source_fixed_identity


@pytest.fixture(autouse=True)
def _synthetic_scientific_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep mechanism tests self-contained; production gates stay hard-coded."""

    monkeypatch.setattr(
        camera_portability, "_enforce_source_fixed_identity", lambda assets: None
    )
    monkeypatch.setattr(
        camera_portability, "_enforce_frozen_scientific_identity", lambda contract: None
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"size_bytes": len(payload), "sha256": _sha(payload)}


def _asset_rows(source_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role, member, destination in camera_portability.FIXED_RUNTIME_ASSET_LAYOUT:
        source = source_root / member
        descriptor = _write(source, (f"fixture:{role}\n").encode("utf-8"))
        rows.append(
            {
                "role": role,
                "member": member,
                "source_path": str(source.resolve()),
                "destination": destination,
                **descriptor,
            }
        )
    return rows


def _build_fixture_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    source_descriptor = {
        "schema_version": camera_portability.SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "purpose": "owner_approved_offline_runtime_asset_source",
        "archive_logical_id": "fixture-runtime-assets-v1",
        "detector_provenance_token": "owner-approved-fixture-token",
        "assets": _asset_rows(tmp_path / "sources"),
    }
    source_path = tmp_path / "source-descriptor.yaml"
    source_path.write_text(yaml.safe_dump(source_descriptor, sort_keys=False), encoding="utf-8")
    archive = tmp_path / "offline" / "runtime-assets.zip"
    observed = tmp_path / "offline" / "observed.json"
    first = build_runtime_asset_bundle(
        project_root=project,
        source_descriptor_path=source_path,
        output_archive_path=archive,
        output_observed_descriptor_path=observed,
    )
    return archive, first


def _contract_from_observed(
    project: Path,
    observed: dict[str, object],
    *,
    source_kind: str = "git_tree",
) -> dict[str, object]:
    trust_roots: list[dict[str, object]] = []
    for relative in camera_portability.REQUIRED_TRUST_ROOT_PATHS:
        descriptor = _write(project / relative, (f"trusted:{relative}\n").encode("utf-8"))
        trust_roots.append({"path": relative, **descriptor})

    assets = observed["assets"]
    assert isinstance(assets, list)
    by_role = {str(row["role"]): row for row in assets}
    candidate = by_role["candidate_manifest"]
    preprocessing_config = next(
        row for row in trust_roots if row["path"] == "configs/data/wandering_preprocessing_v1.yaml"
    )
    source_identity: dict[str, object] = {
        "kind": source_kind,
        "identifier": "b" * 40,
        "base_git_commit": "b" * 40 if source_kind == "git_tree" else "c" * 40,
        "descriptors": trust_roots,
    }
    if source_kind == "approved_patch_archive":
        patch_descriptor = _write(project / "source/portable-approved.patch", b"fixture patch\n")
        source_identity["identifier"] = patch_descriptor["sha256"]
        source_identity["patch_archive"] = {
            "path": "source/portable-approved.patch",
            **patch_descriptor,
        }
    contract = {
        "schema_version": camera_portability.CONTRACT_SCHEMA_VERSION,
        "purpose": "deployability_only",
        "archive": observed["archive"],
        "assets": assets,
        "candidate": {
            "candidate_id": "topowander-m0s-seed20260731-epoch0005",
            "primary_seed": 20260731,
            "best_epoch": 5,
            "manifest_schema_version": "wandering-m0rh-scoring-candidate-manifest-v3",
            "manifest_path": candidate["destination"],
            "manifest_size_bytes": candidate["size_bytes"],
            "manifest_sha256": candidate["sha256"],
        },
        "preprocessing": {
            "config": preprocessing_config,
            "manifest": by_role["preprocessing_manifest"],
            "feature_stats": by_role["preprocessing_feature_stats"],
        },
        "detector": {
            "provenance_token": "owner-approved-fixture-token",
            "model_id": "yolov8n.pt",
            "asset": by_role["detector_weight"],
        },
        "byte_track": {
            "distribution": "ultralytics",
            "distribution_version": "8.4.78",
            "resource": "cfg/trackers/bytetrack.yaml",
            "size_bytes": 856,
            "sha256": "395701d947a749179dee3e327b1181b730c4ca98e7cac5a2ab05280aae573b8b",
        },
        "dependencies": [
            {"distribution": name, "specifier": specifier}
            for name, specifier in camera_portability.REQUIRED_DEPENDENCY_SPECIFIERS
        ],
        "source_identity": source_identity,
        "safeguards": {
            "reads_video": False,
            "runs_tracker": False,
            "runs_qc": False,
            "runs_forward": False,
            "changes_c0_c3": False,
        },
    }
    path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return contract


def _fake_active_checkout(monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    active = project / camera_portability.PORTABILITY_SOURCE_RELATIVE_PATH
    active.parent.mkdir(parents=True, exist_ok=True)
    if not active.exists():
        active.write_text("# fixture active module\n", encoding="utf-8")
    monkeypatch.setattr(camera_portability, "_ACTIVE_MODULE_FILE", active.resolve())
    monkeypatch.setattr(camera_portability, "_editable_project_root", lambda: project.resolve())
    monkeypatch.setattr(
        camera_portability,
        "_read_git_head_without_subprocess",
        lambda root: "b" * 40,
    )
    monkeypatch.setattr(
        camera_portability,
        "_git_blob_bytes",
        lambda root, commit, relative: (project / relative).read_bytes(),
        raising=False,
    )
    monkeypatch.setattr(
        camera_portability,
        "_verify_candidate_source_cross_bindings",
        lambda root, manifest_path, contract: None,
        raising=False,
    )


def _mock_ready_preflight_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        camera_portability,
        "_installed_dependency_versions",
        lambda: {
            name: specifier.removeprefix("==")
            for name, specifier in camera_portability.REQUIRED_DEPENDENCY_SPECIFIERS
        },
    )
    monkeypatch.setattr(
        camera_portability,
        "_verify_preprocessing_cross_binding",
        lambda root, contract: None,
    )
    monkeypatch.setattr(
        camera_portability,
        "_safe_load_candidate",
        lambda *args, **kwargs: SimpleNamespace(
            model=SimpleNamespace(training=False)
        ),
    )
    monkeypatch.setattr(
        camera_portability,
        "_load_detector_without_inference",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        camera_portability,
        "_package_bytetrack_descriptor",
        lambda: {
            "distribution": "ultralytics",
            "distribution_version": "8.4.78",
            "resource": "cfg/trackers/bytetrack.yaml",
            "size_bytes": 856,
            "sha256": "395701d947a749179dee3e327b1181b730c4ca98e7cac5a2ab05280aae573b8b",
        },
    )


def _ready_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_kind: str = "git_tree",
) -> tuple[Path, Path, dict[str, object]]:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed, source_kind=source_kind)
    _fake_active_checkout(monkeypatch, project)
    restore_camera_runtime_assets(project_root=project, archive_path=archive)
    _mock_ready_preflight_dependencies(monkeypatch)
    return archive, project, observed


@pytest.mark.parametrize(
    "attack",
    (
        "/absolute/member",
        "../escape",
        "C:drive-relative",
        "C:\\absolute\\path",
        "\\\\server\\share",
        "back\\slash",
        "https://example.invalid/asset",
        "token=secret",
        "control\x01byte",
        "a" * 257,
    ),
)
def test_contract_rejects_unsafe_member_and_destination_paths(
    tmp_path: Path, attack: str
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path)
    project = tmp_path / "contract-project"
    project.mkdir()
    path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    for field in ("member", "destination"):
        contract = _contract_from_observed(project, observed)
        contract["assets"][0][field] = attack
        path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
        with pytest.raises(PortableContractError):
            load_portable_contract(path)
    assert archive.is_file()


def test_contract_rejects_unknown_duplicate_and_case_colliding_assets(tmp_path: Path) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path)
    project = tmp_path / "contract-project"
    project.mkdir()
    contract = _contract_from_observed(project, observed)
    path = project / CANONICAL_CONTRACT_RELATIVE_PATH

    for mutation in ("extra", "duplicate", "case"):
        altered = json.loads(json.dumps(contract))
        row = dict(altered["assets"][0])
        if mutation == "extra":
            row.update(role="extra", member="extra/file", destination="extra/file")
        elif mutation == "case":
            row["destination"] = str(row["destination"]).upper()
        altered["assets"].append(row)
        path.write_text(yaml.safe_dump(altered, sort_keys=False), encoding="utf-8")
        with pytest.raises(PortableContractError):
            load_portable_contract(path)


def test_contract_rejects_duplicate_yaml_keys_and_real_gate_rejects_candidate_drift(
    tmp_path: Path,
) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path)
    project = tmp_path / "contract-project"
    project.mkdir()
    contract = _contract_from_observed(project, observed)
    path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    path.write_text(
        path.read_text(encoding="utf-8") + "purpose: deployability_only\n",
        encoding="utf-8",
    )
    with pytest.raises(PortableContractError, match="parse"):
        load_portable_contract(path)

    contract["candidate"]["candidate_id"] = "different-candidate"
    with pytest.raises(PortableAssetBlockedError, match="scientific identity"):
        _REAL_ENFORCE_FROZEN_IDENTITY(contract)
    with pytest.raises(PortableContractError, match="scientific asset"):
        _REAL_ENFORCE_SOURCE_IDENTITY(observed["assets"])


def test_builder_is_deterministic_fresh_only_and_never_searches_or_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_archive, first = _build_fixture_bundle(tmp_path / "first")
    second_archive, second = _build_fixture_bundle(tmp_path / "second")
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first["archive"]["sha256"] == second["archive"]["sha256"]
    assert first["archive"]["encoding"] == {
        "format": "zip",
        "version": "portable-zip-v1",
        "compression": "stored",
        "member_mode": "0600",
        "member_timestamp": "1980-01-01T00:00:00Z",
        "host_metadata": "stripped",
        "member_order": "lexicographic",
    }
    with pytest.raises(FileExistsError):
        _build_fixture_bundle(tmp_path / "first")


def test_restore_is_fresh_only_and_writes_content_bound_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "restore-project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)

    result = restore_camera_runtime_assets(project_root=project, archive_path=archive)
    assert result["status"] == "ready"
    marker = json.loads((project / camera_portability.READY_MARKER_RELATIVE_PATH).read_text())
    assert marker["archive_sha256"] == observed["archive"]["sha256"]
    assert marker["asset_set_sha256"] == result["asset_set_sha256"]
    assert marker["committed_groups"] == ["candidate", "preprocessing", "detector"]
    for row in observed["assets"]:
        destination = project / str(row["destination"])
        assert destination.stat().st_size == row["size_bytes"]
        assert _sha(destination.read_bytes()) == row["sha256"]
    with pytest.raises(PortableAssetBlockedError, match="already exists"):
        restore_camera_runtime_assets(project_root=project, archive_path=archive)


def test_restore_rejects_extra_duplicate_symlink_and_same_length_tamper_before_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    for attack in ("extra", "missing", "duplicate", "symlink", "same_length", "metadata"):
        project = tmp_path / f"project-{attack}"
        project.mkdir()
        _contract_from_observed(project, observed)
        _fake_active_checkout(monkeypatch, project)
        attacked = tmp_path / f"{attack}.zip"
        with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
            attacked, "w", compression=zipfile.ZIP_STORED
        ) as target:
            for index, info in enumerate(source.infolist()):
                if attack == "missing" and index == 0:
                    continue
                payload = source.read(info)
                if attack == "same_length" and info.filename == source.infolist()[0].filename:
                    payload = bytes([payload[0] ^ 1]) + payload[1:]
                if attack == "metadata" and index == 0:
                    info.date_time = (1981, 1, 1, 0, 0, 0)
                target.writestr(info, payload)
            if attack == "extra":
                target.writestr("unexpected/file", b"extra")
            elif attack == "duplicate":
                info = source.infolist()[0]
                target.writestr(info, source.read(info))
            elif attack == "symlink":
                info = zipfile.ZipInfo("symlink")
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                target.writestr(info, b"target")
        altered = json.loads(json.dumps(observed))
        altered["archive"]["size_bytes"] = attacked.stat().st_size
        altered["archive"]["sha256"] = _sha(attacked.read_bytes())
        contract = yaml.safe_load((project / CANONICAL_CONTRACT_RELATIVE_PATH).read_text())
        contract["archive"] = altered["archive"]
        (project / CANONICAL_CONTRACT_RELATIVE_PATH).write_text(
            yaml.safe_dump(contract, sort_keys=False), encoding="utf-8"
        )
        with pytest.raises(PortableAssetBlockedError):
            restore_camera_runtime_assets(project_root=project, archive_path=attacked)
        assert not any((project / destination).exists() for destination in camera_portability.GROUP_DESTINATIONS)


def test_restore_rolls_back_only_this_invocation_after_commit_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "restore-project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    unrelated = project / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    original = camera_portability._atomic_rename
    calls = 0

    def fail_second_group(source: Path, destination: Path) -> None:
        nonlocal calls
        if destination in {project / path for path in camera_portability.GROUP_DESTINATIONS}:
            calls += 1
            if calls == 2:
                raise OSError("injected commit failure")
        original(source, destination)

    monkeypatch.setattr(camera_portability, "_atomic_rename", fail_second_group)
    with pytest.raises(OSError, match="injected"):
        restore_camera_runtime_assets(project_root=project, archive_path=archive)
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert not (project / camera_portability.READY_MARKER_RELATIVE_PATH).exists()
    assert not any((project / destination).exists() for destination in camera_portability.GROUP_DESTINATIONS)
    assert not list(project.rglob("*.portable-staging-*"))


@pytest.mark.parametrize("residue", ("partial_destination", "staging"))
def test_restore_refuses_partial_destination_and_prior_staging_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, residue: str
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "restore-project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    candidate = project / camera_portability.GROUP_DESTINATIONS[0]
    if residue == "partial_destination":
        candidate.mkdir(parents=True)
        (candidate / "partial.txt").write_text("preserve", encoding="utf-8")
    else:
        candidate.parent.mkdir(parents=True)
        (candidate.parent / f".{candidate.name}.portable-staging-residue").mkdir()
    with pytest.raises(PortableAssetBlockedError):
        restore_camera_runtime_assets(project_root=project, archive_path=archive)
    assert not (project / camera_portability.READY_MARKER_RELATIVE_PATH).exists()
    if residue == "partial_destination":
        assert (candidate / "partial.txt").read_text(encoding="utf-8") == "preserve"


def test_contract_rejects_member_size_over_extraction_limit(tmp_path: Path) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path)
    project = tmp_path / "contract-project"
    project.mkdir()
    contract = _contract_from_observed(project, observed)
    contract["assets"][0]["size_bytes"] = 64 * 1024 * 1024 + 1
    path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    with pytest.raises(PortableContractError, match="size"):
        load_portable_contract(path)


def test_missing_canonical_contract_is_blocked_before_data_network_or_runtime_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _fake_active_checkout(monkeypatch, project)
    forbidden = mock.Mock(side_effect=AssertionError("protected work is unreachable"))
    monkeypatch.setattr(camera_portability, "_safe_load_candidate", forbidden)
    monkeypatch.setattr(camera_portability, "_load_detector_without_inference", forbidden)
    monkeypatch.setattr(camera_portability, "_package_bytetrack_descriptor", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == BLOCKER_ASSET_SOURCE_UNAVAILABLE
    assert report["network_attempt_count"] == 0
    assert report["safeguards"] == {
        "reads_video": False,
        "runs_tracker": False,
        "runs_qc": False,
        "runs_forward": False,
        "changes_c0_c3": False,
    }
    forbidden.assert_not_called()
    assert "path" not in json.dumps(report).lower()


def test_preflight_ready_safe_loads_without_forward_and_checks_detector_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    restore_camera_runtime_assets(project_root=project, archive_path=archive)
    forward = mock.Mock(side_effect=AssertionError("candidate forward is forbidden"))
    candidate = mock.Mock(
        return_value=SimpleNamespace(model=SimpleNamespace(training=False, forward=forward))
    )
    detector_inference = mock.Mock(
        side_effect=AssertionError("detector inference is forbidden")
    )
    detector = mock.Mock(return_value=SimpleNamespace(predict=detector_inference))
    monkeypatch.setattr(camera_portability, "_safe_load_candidate", candidate)
    monkeypatch.setattr(camera_portability, "_load_detector_without_inference", detector)
    monkeypatch.setattr(
        camera_portability, "_verify_preprocessing_cross_binding", lambda root, contract: None
    )
    monkeypatch.setattr(
        camera_portability,
        "_installed_dependency_versions",
        lambda: {name: specifier.removeprefix("==") for name, specifier in camera_portability.REQUIRED_DEPENDENCY_SPECIFIERS},
    )
    monkeypatch.setattr(
        camera_portability,
        "_package_bytetrack_descriptor",
        lambda: {
            "distribution": "ultralytics",
            "distribution_version": "8.4.78",
            "resource": "cfg/trackers/bytetrack.yaml",
            "size_bytes": 856,
            "sha256": "395701d947a749179dee3e327b1181b730c4ca98e7cac5a2ab05280aae573b8b",
        },
    )
    attempts: list[object] = []
    monkeypatch.setattr(camera_portability, "_NETWORK_ATTEMPTS", attempts)

    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "ready"
    assert report["blocker_code"] is None
    assert report["network_denial_guard_enforced"] is True
    assert report["network_attempt_count"] == 0
    candidate.assert_called_once()
    detector.assert_called_once()
    forward.assert_not_called()
    detector_inference.assert_not_called()
    assert detector.call_args.args[0].name == "yolov8n.pt"


def test_preflight_missing_detector_blocks_before_detector_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    restore_camera_runtime_assets(project_root=project, archive_path=archive)
    (project / camera_portability.DETECTOR_DESTINATION).unlink()
    loader = mock.Mock(side_effect=AssertionError("detector import must remain unreachable"))
    monkeypatch.setattr(camera_portability, "_load_detector_without_inference", loader)
    monkeypatch.setattr(
        camera_portability,
        "_installed_dependency_versions",
        lambda: {name: specifier.removeprefix("==") for name, specifier in camera_portability.REQUIRED_DEPENDENCY_SPECIFIERS},
    )
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == "detector_weight_missing"
    loader.assert_not_called()


def test_preflight_dependency_and_active_source_drift_fail_before_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    restore_camera_runtime_assets(project_root=project, archive_path=archive)
    candidate_loader = mock.Mock(side_effect=AssertionError("loader is unreachable"))
    detector_loader = mock.Mock(side_effect=AssertionError("loader is unreachable"))
    monkeypatch.setattr(camera_portability, "_safe_load_candidate", candidate_loader)
    monkeypatch.setattr(camera_portability, "_load_detector_without_inference", detector_loader)
    versions = {
        name: specifier.removeprefix("==")
        for name, specifier in camera_portability.REQUIRED_DEPENDENCY_SPECIFIERS
    }
    versions["ultralytics"] = "0.0.0"
    monkeypatch.setattr(camera_portability, "_installed_dependency_versions", lambda: versions)
    report = preflight_camera_runtime(project_root=project)
    assert report["blocker_code"] == "dependency_version_mismatch"
    candidate_loader.assert_not_called()
    detector_loader.assert_not_called()

    trust_root = project / camera_portability.REQUIRED_TRUST_ROOT_PATHS[7]
    trust_root.write_bytes(trust_root.read_bytes() + b"drift")
    report = preflight_camera_runtime(project_root=project)
    assert report["blocker_code"] == "active_source_or_config_drift"
    candidate_loader.assert_not_called()
    detector_loader.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "expected_blocker"),
    (
        ("preprocessing", "asset_sha_mismatch"),
        ("candidate", "candidate_safe_load_failed"),
        ("bytetrack", "asset_sha_mismatch"),
    ),
)
def test_preflight_candidate_preprocessing_and_bytetrack_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_blocker: str,
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    restore_camera_runtime_assets(project_root=project, archive_path=archive)
    monkeypatch.setattr(
        camera_portability,
        "_installed_dependency_versions",
        lambda: {
            name: specifier.removeprefix("==")
            for name, specifier in camera_portability.REQUIRED_DEPENDENCY_SPECIFIERS
        },
    )
    if failure != "preprocessing":
        monkeypatch.setattr(
            camera_portability,
            "_verify_preprocessing_cross_binding",
            lambda root, contract: None,
        )
    if failure == "candidate":
        monkeypatch.setattr(
            camera_portability,
            "_safe_load_candidate",
            mock.Mock(side_effect=ValueError("synthetic safe-load failure")),
        )
    else:
        monkeypatch.setattr(
            camera_portability,
            "_safe_load_candidate",
            mock.Mock(return_value=SimpleNamespace(model=SimpleNamespace(training=False))),
        )
    monkeypatch.setattr(
        camera_portability,
        "_load_detector_without_inference",
        mock.Mock(return_value=SimpleNamespace()),
    )
    tracker_descriptor = {
        "distribution": "ultralytics",
        "distribution_version": "8.4.78",
        "resource": "cfg/trackers/bytetrack.yaml",
        "size_bytes": 856,
        "sha256": "0" * 64 if failure == "bytetrack" else "395701d947a749179dee3e327b1181b730c4ca98e7cac5a2ab05280aae573b8b",
    }
    monkeypatch.setattr(
        camera_portability, "_package_bytetrack_descriptor", lambda: tracker_descriptor
    )
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == expected_blocker


def test_preflight_fake_editable_checkout_blocks_before_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _fake_active_checkout(monkeypatch, project)
    other = tmp_path / "different-editable-root"
    other.mkdir()
    monkeypatch.setattr(camera_portability, "_editable_project_root", lambda: other)
    report = preflight_camera_runtime(project_root=project)
    assert report["blocker_code"] == "editable_checkout_mismatch"


def test_git_tree_contract_and_runtime_source_cannot_self_sign_same_dirty_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    contract_path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    immutable_blobs = {
        relative: (project / relative).read_bytes()
        for relative in (
            CANONICAL_CONTRACT_RELATIVE_PATH,
            *camera_portability.REQUIRED_TRUST_ROOT_PATHS,
        )
    }
    monkeypatch.setattr(
        camera_portability,
        "_git_blob_bytes",
        lambda root, commit, relative: immutable_blobs[relative],
        raising=False,
    )

    relative = "src/elderly_monitoring/modules/mental_health/wandering/camera_collection.py"
    source = project / relative
    source.write_bytes(source.read_bytes() + b"# dirty self-signed drift\n")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    descriptor = next(
        row for row in contract["source_identity"]["descriptors"] if row["path"] == relative
    )
    descriptor.update(size_bytes=source.stat().st_size, sha256=_sha(source.read_bytes()))
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    parsed = load_portable_contract(contract_path)
    marker_path = project / camera_portability.READY_MARKER_RELATIVE_PATH
    marker_path.write_bytes(
        camera_portability._canonical_json_bytes(
            camera_portability._ready_marker(contract_path.read_bytes(), parsed)
        )
    )

    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == "active_source_or_config_drift"


def test_git_tree_canonical_contract_bytes_must_match_commit_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    contract_path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    committed_contract = contract_path.read_bytes()
    immutable_blobs = {
        relative: (project / relative).read_bytes()
        for relative in camera_portability.REQUIRED_TRUST_ROOT_PATHS
    }
    immutable_blobs[CANONICAL_CONTRACT_RELATIVE_PATH] = committed_contract
    monkeypatch.setattr(
        camera_portability,
        "_git_blob_bytes",
        lambda root, commit, relative: immutable_blobs[relative],
        raising=False,
    )
    contract_path.write_bytes(committed_contract + b"\n")
    parsed = load_portable_contract(contract_path)
    marker_path = project / camera_portability.READY_MARKER_RELATIVE_PATH
    marker_path.write_bytes(
        camera_portability._canonical_json_bytes(
            camera_portability._ready_marker(contract_path.read_bytes(), parsed)
        )
    )
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == "active_source_or_config_drift"


def test_approved_patch_mode_without_external_approval_anchor_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    contract = _contract_from_observed(
        project, observed, source_kind="approved_patch_archive"
    )
    _fake_active_checkout(monkeypatch, project)
    with pytest.raises(PortableAssetBlockedError) as caught:
        camera_portability._verify_trust_roots(project, contract)
    assert caught.value.blocker_code == "source_integration_pending"


def test_runtime_trust_root_closure_covers_actual_camera_import_graph() -> None:
    required = set(camera_portability.REQUIRED_TRUST_ROOT_PATHS)
    assert {
        "src/elderly_monitoring/modules/mental_health/wandering/camera_adapter.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_inference.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_qc.py",
        "src/elderly_monitoring/modules/mental_health/wandering/camera_episode.py",
        "src/elderly_monitoring/modules/mental_health/wandering/preprocessing.py",
        "src/elderly_monitoring/modules/mental_health/wandering/preprocessing_bundle.py",
        "scripts/wandering/prepare_camera_session.py",
        "scripts/wandering/run_camera_development.py",
        "scripts/wandering/run_topowander_camera_inference.py",
    }.issubset(required)


def test_candidate_model_and_release_source_identities_are_cross_bound(
    tmp_path: Path,
) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    contract = _contract_from_observed(project, observed)
    trust = {
        row["path"]: {
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in contract["source_identity"]["descriptors"]
    }
    model_path = "src/elderly_monitoring/modules/mental_health/wandering/model.py"
    release_path = "src/elderly_monitoring/modules/mental_health/wandering/release.py"
    manifest = {
        "release_implementation_identity": {
            "identity": {
                "files": {
                    model_path: trust[model_path],
                    release_path: trust[release_path],
                }
            }
        },
        "training_candidate_identity": {
            "identity": {
                "training_source_bundle": {
                    "files": {model_path: dict(trust[model_path])}
                }
            }
        },
    }
    path = tmp_path / "candidate-manifest.json"
    path.write_bytes(camera_portability._canonical_json_bytes(manifest))
    camera_portability._verify_candidate_source_cross_bindings(
        project, path, contract
    )
    manifest["training_candidate_identity"]["identity"]["training_source_bundle"][
        "files"
    ][model_path]["sha256"] = "0" * 64
    path.write_bytes(camera_portability._canonical_json_bytes(manifest))
    with pytest.raises(PortableAssetBlockedError, match="source identity drift"):
        camera_portability._verify_candidate_source_cross_bindings(
            project, path, contract
        )


def test_network_guard_rejects_a_swallowed_attempt_and_restores_socket() -> None:
    original = socket.create_connection
    with pytest.raises(camera_portability.NetworkAccessDeniedError):
        with camera_portability.deny_network_access():
            try:
                socket.create_connection(("example.invalid", 443))
            except camera_portability.NetworkAccessDeniedError:
                pass
    assert socket.create_connection is original


def test_preflight_network_attempt_count_is_scoped_to_this_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    attempts = ["historical-attempt"]
    monkeypatch.setattr(camera_portability, "_NETWORK_ATTEMPTS", attempts)
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "ready"
    assert report["network_attempt_count"] == 0
    assert attempts == ["historical-attempt"]


@pytest.mark.parametrize("loader_role", ("candidate", "detector"))
def test_preflight_blocks_when_loader_swallows_network_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader_role: str,
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)

    def swallowed_attempt(*args, **kwargs):
        try:
            socket.create_connection(("example.invalid", 443))
        except camera_portability.NetworkAccessDeniedError:
            pass
        if loader_role == "candidate":
            return SimpleNamespace(model=SimpleNamespace(training=False))
        return SimpleNamespace()

    monkeypatch.setattr(
        camera_portability,
        "_safe_load_candidate" if loader_role == "candidate" else "_load_detector_without_inference",
        swallowed_attempt,
    )
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == "network_guard_unavailable"
    assert report["network_attempt_count"] == 1
    expected_check = {
        "candidate": "candidate_safe_load",
        "detector": "detector_local_loadability",
    }[loader_role]
    assert not any(
        row["check_id"] == expected_check
        and row["status"] == "ready"
        for row in report["checks"]
    )


@pytest.mark.parametrize("group_index", (0, 1, 2))
def test_restore_group_commit_is_atomic_no_replace_and_preserves_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    group_index: int,
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    target = project / camera_portability.GROUP_DESTINATIONS[group_index]
    original = camera_portability._atomic_rename

    def race(source: Path, destination: Path) -> None:
        if destination == target and not destination.exists():
            destination.mkdir(parents=True)
        original(source, destination)

    monkeypatch.setattr(camera_portability, "_atomic_rename", race)
    with pytest.raises((FileExistsError, OSError, PortableAssetBlockedError)):
        restore_camera_runtime_assets(project_root=project, archive_path=archive)
    assert target.is_dir()
    assert list(target.iterdir()) == []
    for index, relative in enumerate(camera_portability.GROUP_DESTINATIONS):
        if index != group_index:
            assert not (project / relative).exists()
    assert not (project / camera_portability.READY_MARKER_RELATIVE_PATH).exists()


@pytest.mark.parametrize("output_role", ("archive", "observed"))
def test_builder_commit_is_atomic_no_replace_and_preserves_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_role: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    descriptor = {
        "schema_version": camera_portability.SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "purpose": "owner_approved_offline_runtime_asset_source",
        "archive_logical_id": "fixture-runtime-assets-v1",
        "detector_provenance_token": "owner-approved-fixture-token",
        "assets": _asset_rows(tmp_path / "sources"),
    }
    descriptor_path = tmp_path / "source-descriptor.yaml"
    descriptor_path.write_text(
        yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8"
    )
    archive = tmp_path / "outputs" / "runtime-assets.zip"
    observed = tmp_path / "outputs" / "observed.json"
    target = archive if output_role == "archive" else observed
    original = camera_portability._atomic_rename
    competitor = b"competitor-owned-by-another-writer\n"

    def race(source: Path, destination: Path) -> None:
        if destination == target and not destination.exists():
            destination.write_bytes(competitor)
        original(source, destination)

    monkeypatch.setattr(camera_portability, "_atomic_rename", race)
    with pytest.raises((FileExistsError, OSError)):
        build_runtime_asset_bundle(
            project_root=project,
            source_descriptor_path=descriptor_path,
            output_archive_path=archive,
            output_observed_descriptor_path=observed,
        )
    assert target.read_bytes() == competitor
    other = observed if output_role == "archive" else archive
    assert not other.exists()


def test_ready_marker_commit_is_atomic_no_replace_and_preserves_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    marker = project / camera_portability.READY_MARKER_RELATIVE_PATH
    original = camera_portability._atomic_rename
    competitor = b"competitor marker\n"

    def race(source: Path, destination: Path) -> None:
        if destination == marker and not destination.exists():
            destination.write_bytes(competitor)
        original(source, destination)

    monkeypatch.setattr(camera_portability, "_atomic_rename", race)
    with pytest.raises((FileExistsError, OSError)):
        restore_camera_runtime_assets(project_root=project, archive_path=archive)
    assert marker.read_bytes() == competitor
    assert not any(
        (project / relative).exists()
        for relative in camera_portability.GROUP_DESTINATIONS
    )


def test_preflight_report_commit_is_atomic_no_replace_and_preserves_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    output = tmp_path / "preflight.json"
    original = camera_portability._atomic_rename
    competitor = b"competitor report\n"

    def race(source: Path, destination: Path) -> None:
        if destination == output and not destination.exists():
            destination.write_bytes(competitor)
        original(source, destination)

    monkeypatch.setattr(camera_portability, "_atomic_rename", race)
    with pytest.raises((FileExistsError, OSError)):
        preflight_camera_runtime(project_root=project, output_path=output)
    assert output.read_bytes() == competitor


def test_restore_keyboard_interrupt_cleans_all_process_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    original = camera_portability._atomic_rename
    calls = 0

    def interrupt_second(source: Path, destination: Path) -> None:
        nonlocal calls
        if destination in {
            project / relative for relative in camera_portability.GROUP_DESTINATIONS
        }:
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
        original(source, destination)

    monkeypatch.setattr(camera_portability, "_atomic_rename", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        restore_camera_runtime_assets(project_root=project, archive_path=archive)
    assert not any(
        (project / relative).exists()
        for relative in camera_portability.GROUP_DESTINATIONS
    )
    assert not list(project.rglob("*.portable-staging-*"))


@pytest.mark.parametrize("encoding", ("pretty", "duplicate"))
def test_ready_marker_requires_unique_keys_and_exact_canonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    marker_path = project / camera_portability.READY_MARKER_RELATIVE_PATH
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if encoding == "pretty":
        marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    else:
        canonical = marker_path.read_text(encoding="utf-8")
        duplicate = (
            '{"schema_version":"wandering-camera-runtime-assets-ready-v1",'
            + canonical[1:]
        )
        marker_path.write_text(duplicate, encoding="utf-8")
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == "asset_sha_mismatch"


@pytest.mark.parametrize(
    "source_path",
    (
        r"\\server\share\yolov8n.pt",
        r"\\?\C:\runtime\yolov8n.pt",
        r"\\.\PhysicalDrive0",
        "https://example.invalid/yolov8n.pt",
        "/tmp/token=secret/yolov8n.pt",
    ),
)
def test_source_descriptor_rejects_unc_device_url_and_credentials(
    tmp_path: Path, source_path: str
) -> None:
    descriptor = {
        "schema_version": camera_portability.SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "purpose": "owner_approved_offline_runtime_asset_source",
        "archive_logical_id": "fixture-runtime-assets-v1",
        "detector_provenance_token": "owner-approved-fixture-token",
        "assets": _asset_rows(tmp_path / "sources"),
    }
    descriptor["assets"][0]["source_path"] = source_path
    path = tmp_path / "source-descriptor.yaml"
    path.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    with pytest.raises(PortableContractError):
        camera_portability._load_source_descriptor(path)


def test_canonical_contract_parent_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    modules = project / "configs/modules"
    external = tmp_path / "external-modules"
    modules.rename(external)
    try:
        modules.symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("test filesystem does not permit directory symlinks")
    with pytest.raises(PortableAssetBlockedError, match="link or reparse") as caught:
        camera_portability._fixed_contract_path(project)
    assert caught.value.blocker_code == "active_source_or_config_drift"


def test_missing_restore_archive_is_expected_structured_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, observed = _build_fixture_bundle(tmp_path / "bundle")
    project = tmp_path / "project"
    project.mkdir()
    _contract_from_observed(project, observed)
    _fake_active_checkout(monkeypatch, project)
    with pytest.raises(PortableAssetBlockedError) as caught:
        restore_camera_runtime_assets(
            project_root=project,
            archive_path=tmp_path / "missing-runtime-assets.zip",
        )
    assert caught.value.blocker_code == "asset_source_unavailable"


def test_missing_bytetrack_resource_is_expected_structured_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        camera_portability,
        "_package_bytetrack_descriptor",
        mock.Mock(side_effect=FileNotFoundError("missing package resource")),
    )
    report = preflight_camera_runtime(project_root=project)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == "asset_source_unavailable"


def test_detector_second_validation_is_bound_to_first_immutable_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    token = camera_portability.resolve_verified_detector_asset(project)
    contract_path = project / CANONICAL_CONTRACT_RELATIVE_PATH
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    detector_path = project / camera_portability.DETECTOR_DESTINATION
    replacement = b"replacement detector fixture bytes"
    detector_path.write_bytes(replacement)
    replacement_descriptor = {
        "size_bytes": len(replacement),
        "sha256": _sha(replacement),
    }
    for row in contract["assets"]:
        if row["role"] == "detector_weight":
            row.update(replacement_descriptor)
    contract["detector"]["asset"].update(replacement_descriptor)
    for row in contract["archive"]["members"]:
        if row["member"] == "detector/yolov8n.pt":
            row.update(replacement_descriptor)
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    parsed = load_portable_contract(contract_path)
    marker_path = project / camera_portability.READY_MARKER_RELATIVE_PATH
    marker_path.write_bytes(
        camera_portability._canonical_json_bytes(
            camera_portability._ready_marker(contract_path.read_bytes(), parsed)
        )
    )
    with pytest.raises(PortableAssetBlockedError):
        camera_portability.verify_detector_asset_again(project, token)


@pytest.mark.parametrize("drift", ("contract", "marker", "weight"))
def test_detector_token_rejects_each_bound_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    _archive, project, _observed = _ready_fixture(tmp_path, monkeypatch)
    token = camera_portability.resolve_verified_detector_asset(project)
    if drift == "contract":
        path = project / CANONICAL_CONTRACT_RELATIVE_PATH
        path.write_bytes(path.read_bytes() + b"\n")
    elif drift == "marker":
        path = project / camera_portability.READY_MARKER_RELATIVE_PATH
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = project / camera_portability.DETECTOR_DESTINATION
        payload = path.read_bytes()
        path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(PortableAssetBlockedError):
        camera_portability.verify_detector_asset_again(project, token)


def test_production_preflight_cli_missing_contract_is_structured_blocked_exit_2() -> None:
    cli = ROOT / "scripts/wandering/preflight_camera_runtime.py"
    completed = subprocess.run(
        [sys.executable, str(cli), "--project-root", str(ROOT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["status"] == "blocked"
    assert report["blocker_code"] == BLOCKER_ASSET_SOURCE_UNAVAILABLE
    assert completed.stderr == ""


def test_preflight_import_and_missing_contract_with_preimport_network_http_subprocess_bombs() -> None:
    code = """
import json
import socket
import subprocess
import urllib.request
calls = []
def bomb(*args, **kwargs):
    calls.append('forbidden')
    raise AssertionError('network/http/subprocess is forbidden')
socket.socket.connect = bomb
socket.socket.connect_ex = bomb
socket.create_connection = bomb
urllib.request.urlopen = bomb
subprocess.Popen = bomb
from elderly_monitoring.modules.mental_health.wandering.camera_portability import preflight_camera_runtime
report = preflight_camera_runtime(project_root=r'__ROOT__')
assert report['status'] == 'blocked'
assert report['blocker_code'] == 'asset_source_unavailable'
assert report['network_attempt_count'] == 0
assert calls == []
print(json.dumps(report, sort_keys=True))
""".replace("__ROOT__", str(ROOT))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "blocked"


def test_portable_cli_help_and_parameter_errors_use_the_fixed_exit_contract() -> None:
    scripts = {
        "build_camera_runtime_asset_bundle.py": (
            "--source-descriptor",
            "--output-archive",
            "--output-observed-descriptor",
        ),
        "restore_camera_runtime_assets.py": ("--archive",),
        "preflight_camera_runtime.py": ("--output",),
    }
    for name, required in scripts.items():
        cli = ROOT / "scripts/wandering" / name
        help_result = subprocess.run(
            [sys.executable, str(cli), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 0, help_result.stderr
        assert "--project-root" in help_result.stdout
        for option in required:
            assert option in help_result.stdout
        for forbidden in ("--download", "--url", "--video", "--tracking", "--forward"):
            assert forbidden not in help_result.stdout.lower()

    invalid = subprocess.run(
        [sys.executable, str(ROOT / "scripts/wandering/restore_camera_runtime_assets.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 1
    assert invalid.stdout == ""
