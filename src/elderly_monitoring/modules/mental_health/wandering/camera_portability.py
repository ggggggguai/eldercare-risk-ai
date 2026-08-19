"""Offline, SHA-bound runtime asset restore and zero-video preflight.

This module is deliberately independent from camera datasets.  Production
restore and preflight never read media, authorization receipts, collection
manifests, tracking, sidecars, annotations, or model inputs, and never run a
tracker, Camera QC, a candidate forward pass, or an evaluator.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import importlib.resources as importlib_resources
import json
import ctypes
import errno
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import unquote, urlparse

import yaml
from yaml.constructor import ConstructorError


CONTRACT_SCHEMA_VERSION = "wandering-camera-portable-contract-v1"
SOURCE_DESCRIPTOR_SCHEMA_VERSION = "wandering-camera-runtime-asset-source-descriptor-v1"
OBSERVED_DESCRIPTOR_SCHEMA_VERSION = "wandering-camera-runtime-asset-observed-v1"
READY_MARKER_SCHEMA_VERSION = "wandering-camera-runtime-assets-ready-v1"
PREFLIGHT_REPORT_SCHEMA_VERSION = "wandering-camera-runtime-preflight-v1"

CANONICAL_CONTRACT_RELATIVE_PATH = "configs/modules/wandering_camera_portable_v1.yaml"
PORTABILITY_SOURCE_RELATIVE_PATH = (
    "src/elderly_monitoring/modules/mental_health/wandering/camera_portability.py"
)
READY_MARKER_RELATIVE_PATH = (
    "artifacts/mental_health/wandering_camera_runtime/v1/READY.json"
)
DETECTOR_DESTINATION = (
    "artifacts/mental_health/wandering_camera_runtime/v1/detector/yolov8n.pt"
)

EXPECTED_CANDIDATE_ID = "topowander-m0s-seed20260731-epoch0005"
EXPECTED_CANDIDATE_SEED = 20260731
EXPECTED_CANDIDATE_EPOCH = 5
EXPECTED_CANDIDATE_MANIFEST_SCHEMA = "wandering-m0rh-scoring-candidate-manifest-v3"
EXPECTED_CANDIDATE_DIRECTORY = (
    "reports/mental_health/wandering_performance/m0r_score_entry_hardening_v1/"
    "artifacts/topowander_m0r_candidate_v3"
)
EXPECTED_PREPROCESSING_DIRECTORY = "data/processed/wandering/preprocessing/v1"

FIXED_RUNTIME_ASSET_LAYOUT: tuple[tuple[str, str, str], ...] = (
    (
        "candidate_manifest",
        "candidate/candidate_manifest.json",
        f"{EXPECTED_CANDIDATE_DIRECTORY}/candidate_manifest.json",
    ),
    (
        "candidate_model_state",
        "candidate/model_state.npz",
        f"{EXPECTED_CANDIDATE_DIRECTORY}/model_state.npz",
    ),
    (
        "candidate_forward_config",
        "candidate/forward_config.yaml",
        f"{EXPECTED_CANDIDATE_DIRECTORY}/forward_config.yaml",
    ),
    (
        "candidate_performance_config",
        "candidate/performance_config.yaml",
        f"{EXPECTED_CANDIDATE_DIRECTORY}/performance_config.yaml",
    ),
    (
        "candidate_frozen_wp_rf_config",
        "candidate/frozen_wp_rf_config.yaml",
        f"{EXPECTED_CANDIDATE_DIRECTORY}/frozen_wp_rf_config.yaml",
    ),
    (
        "preprocessing_manifest",
        "preprocessing/manifest.json",
        f"{EXPECTED_PREPROCESSING_DIRECTORY}/manifest.json",
    ),
    (
        "preprocessing_feature_stats",
        "preprocessing/feature_stats.json",
        f"{EXPECTED_PREPROCESSING_DIRECTORY}/feature_stats.json",
    ),
    (
        "detector_weight",
        "detector/yolov8n.pt",
        DETECTOR_DESTINATION,
    ),
)

_EXPECTED_ASSET_LAYOUT = {
    role: (member, destination) for role, member, destination in FIXED_RUNTIME_ASSET_LAYOUT
}
_EXPECTED_CANDIDATE_ASSETS: dict[str, tuple[int, str]] = {
    "candidate_manifest": (
        12642,
        "3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7",
    ),
    "candidate_model_state": (
        838934,
        "94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031",
    ),
    "candidate_forward_config": (
        4469,
        "debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7",
    ),
    "candidate_performance_config": (
        1840,
        "ecc4c5a00dc9c30b1a4a16b943d39d9de85cce27077c8ccba1ba223c529de5f2",
    ),
    "candidate_frozen_wp_rf_config": (
        3398,
        "d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35",
    ),
}
_EXPECTED_PREPROCESSING_ASSETS: dict[str, tuple[int, str]] = {
    "preprocessing_manifest": (
        1278,
        "242072bdfe4b969d320a091ecc499aff445c30a937dfb1e6739ad869aa938593",
    ),
    "preprocessing_feature_stats": (
        2844,
        "249ad8383043dba1d95452533ddc6004dfecc2c9e99be43f788dcfcbfb434c8d",
    ),
}
EXPECTED_PREPROCESSING_CONFIG = {
    "path": "configs/data/wandering_preprocessing_v1.yaml",
    "size_bytes": 1982,
    "sha256": "5b69243337cddeeec8beed4f081330b06c47eadeaee06dc3b3e4ba026f36ff45",
}

GROUP_DESTINATIONS: tuple[str, ...] = (
    EXPECTED_CANDIDATE_DIRECTORY,
    EXPECTED_PREPROCESSING_DIRECTORY,
    "artifacts/mental_health/wandering_camera_runtime/v1/detector",
)
_GROUP_BY_ROLE = {
    **{role: "candidate" for role in _EXPECTED_CANDIDATE_ASSETS},
    **{role: "preprocessing" for role in _EXPECTED_PREPROCESSING_ASSETS},
    "detector_weight": "detector",
}
_GROUP_DESTINATION_BY_NAME = dict(zip(("candidate", "preprocessing", "detector"), GROUP_DESTINATIONS))

REQUIRED_TRUST_ROOT_PATHS: tuple[str, ...] = (
    "environment.yml",
    "pyproject.toml",
    "configs/data/wandering_preprocessing_v1.yaml",
    "configs/data/wandering_camera_collection_v1.yaml",
    "configs/modules/wandering_camera_v1.yaml",
    "configs/modules/wandering_camera_primary_v1.yaml",
    "configs/modules/wandering_camera_development_v1.yaml",
    "src/elderly_monitoring/modules/fall_risk/tracking.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_adapter.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_collection.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_component.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_dataset.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_development.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_episode.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_inference.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_primary_inference.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_qc.py",
    "src/elderly_monitoring/modules/mental_health/wandering/camera_portability.py",
    "src/elderly_monitoring/modules/mental_health/wandering/model.py",
    "src/elderly_monitoring/modules/mental_health/wandering/preprocessing.py",
    "src/elderly_monitoring/modules/mental_health/wandering/preprocessing_bundle.py",
    "src/elderly_monitoring/modules/mental_health/wandering/release.py",
    "scripts/wandering/build_camera_tracking_pair.py",
    "scripts/wandering/build_camera_runtime_asset_bundle.py",
    "scripts/wandering/prepare_camera_session.py",
    "scripts/wandering/restore_camera_runtime_assets.py",
    "scripts/wandering/run_camera_development.py",
    "scripts/wandering/run_topowander_camera_inference.py",
    "scripts/wandering/preflight_camera_runtime.py",
)

REQUIRED_DEPENDENCY_SPECIFIERS: tuple[tuple[str, str], ...] = (
    ("python", "==3.11.15"),
    ("elderly-monitoring-algorithms", "==0.2.0"),
    ("PyYAML", "==6.0.3"),
    ("numpy", "==2.4.6"),
    ("torch", "==2.13.0"),
    ("opencv-python", "==4.13.0.92"),
    ("ultralytics", "==8.4.78"),
)

BLOCKER_ASSET_SOURCE_UNAVAILABLE = "asset_source_unavailable"
BLOCKER_DETECTOR_WEIGHT_MISSING = "detector_weight_missing"
BLOCKER_ASSET_SHA_MISMATCH = "asset_sha_mismatch"
BLOCKER_DEPENDENCY_VERSION_MISMATCH = "dependency_version_mismatch"
BLOCKER_EDITABLE_CHECKOUT_MISMATCH = "editable_checkout_mismatch"
BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT = "active_source_or_config_drift"
BLOCKER_CANDIDATE_SAFE_LOAD_FAILED = "candidate_safe_load_failed"
BLOCKER_NETWORK_GUARD_UNAVAILABLE = "network_guard_unavailable"
BLOCKER_SOURCE_INTEGRATION_PENDING = "source_integration_pending"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_MARKERS = (
    "://",
    "token=",
    "access_token",
    "authorization:",
    "bearer ",
    "password",
    "secret",
    "cookie:",
)
_ARCHIVE_ENCODING = {
    "format": "zip",
    "version": "portable-zip-v1",
    "compression": "stored",
    "member_mode": "0600",
    "member_timestamp": "1980-01-01T00:00:00Z",
    "host_metadata": "stripped",
    "member_order": "lexicographic",
}
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_ACTIVE_MODULE_FILE = Path(__file__).resolve()
_NETWORK_ATTEMPTS: list[str] = []


class PortableContractError(ValueError):
    """Portable descriptor or contract syntax is invalid."""


class PortableAssetBlockedError(RuntimeError):
    """Expected deployability state prevents restore or preflight."""

    def __init__(self, blocker_code: str, message: str) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code


class NetworkAccessDeniedError(RuntimeError):
    """A runtime attempted network access inside the denial guard."""


@dataclass(frozen=True)
class VerifiedDetectorAsset:
    """Immutable first-pass identity carried to the immediate pre-tracker check."""

    project_root: Path
    path: Path
    size_bytes: int
    sha256: str
    contract_sha256: str
    ready_marker_sha256: str
    stat_device: int
    stat_inode: int


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in output
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _canonical_json_bytes(value: object) -> bytes:
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


def _unique_json_object(payload: bytes, role: str) -> dict[str, Any]:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate key in {role}")
            output[key] = value
        return output

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH, f"{role} is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH, f"{role} must be an object"
        )
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_mapping(path: Path, role: str) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PortableContractError(f"cannot parse {role}") from exc
    if not isinstance(value, dict):
        raise PortableContractError(f"{role} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], role: str) -> None:
    if set(value) != keys:
        raise PortableContractError(f"{role} has an invalid exact field set")


def _safe_token(value: Any, role: str) -> str:
    if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
        raise PortableContractError(f"{role} is invalid")
    lowered = value.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        raise PortableContractError(f"{role} contains sensitive or network material")
    return value


def _safe_relative_path(value: Any, role: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise PortableContractError(f"{role} is not a bounded relative path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise PortableContractError(f"{role} contains a control character")
    if "\\" in value or any(marker in value.lower() for marker in _SENSITIVE_MARKERS):
        raise PortableContractError(f"{role} is not a safe POSIX relative path")
    windows = PureWindowsPath(value)
    path = PurePosixPath(value)
    if windows.drive or windows.root or path.is_absolute():
        raise PortableContractError(f"{role} must be relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PortableContractError(f"{role} contains traversal")
    return path.as_posix()


def _size(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _MAX_MEMBER_BYTES:
        raise PortableContractError(f"{role} is invalid")
    return value


def _sha(value: Any, role: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise PortableContractError(f"{role} is invalid")
    return value


def _asset_descriptor(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PortableContractError(f"{role} must be a mapping")
    _exact_keys(
        value,
        {"role", "member", "destination", "size_bytes", "sha256"},
        role,
    )
    asset_role = _safe_token(value["role"], f"{role}.role")
    return {
        "role": asset_role,
        "member": _safe_relative_path(value["member"], f"{role}.member"),
        "destination": _safe_relative_path(
            value["destination"], f"{role}.destination"
        ),
        "size_bytes": _size(value["size_bytes"], f"{role}.size_bytes"),
        "sha256": _sha(value["sha256"], f"{role}.sha256"),
    }


def _validate_asset_set(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) != len(_EXPECTED_ASSET_LAYOUT):
        raise PortableContractError("assets must contain the exact fixed runtime set")
    assets = [_asset_descriptor(value, f"assets[{index}]") for index, value in enumerate(values)]
    roles = [asset["role"] for asset in assets]
    members = [asset["member"] for asset in assets]
    destinations = [asset["destination"] for asset in assets]
    for values_to_check, label in (
        (roles, "role"),
        (members, "member"),
        (destinations, "destination"),
    ):
        if len(values_to_check) != len(set(values_to_check)):
            raise PortableContractError(f"duplicate asset {label}")
        if len(values_to_check) != len({value.casefold() for value in values_to_check}):
            raise PortableContractError(f"case-colliding asset {label}")
    if set(roles) != set(_EXPECTED_ASSET_LAYOUT):
        raise PortableContractError("asset roles do not match the fixed runtime set")
    by_role = {asset["role"]: asset for asset in assets}
    for asset_role, (member, destination) in _EXPECTED_ASSET_LAYOUT.items():
        asset = by_role[asset_role]
        if asset["member"] != member or asset["destination"] != destination:
            raise PortableContractError(f"fixed layout drift for {asset_role}")
    return [by_role[role] for role in _EXPECTED_ASSET_LAYOUT]


def _archive_descriptor(value: Any, assets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PortableContractError("archive must be a mapping")
    _exact_keys(
        value,
        {"logical_id", "filename", "size_bytes", "sha256", "encoding", "members"},
        "archive",
    )
    filename = _safe_relative_path(value["filename"], "archive.filename")
    if PurePosixPath(filename).name != filename:
        raise PortableContractError("archive filename must be a basename")
    if value["encoding"] != _ARCHIVE_ENCODING:
        raise PortableContractError("archive encoding contract drift")
    members = value["members"]
    expected_members = sorted(
        [
        {
            "member": asset["member"],
            "size_bytes": asset["size_bytes"],
            "sha256": asset["sha256"],
        }
        for asset in assets
        ],
        key=lambda descriptor: descriptor["member"],
    )
    if members != expected_members:
        raise PortableContractError("archive member descriptor set drift")
    return {
        "logical_id": _safe_token(value["logical_id"], "archive.logical_id"),
        "filename": filename,
        "size_bytes": _size(value["size_bytes"], "archive.size_bytes"),
        "sha256": _sha(value["sha256"], "archive.sha256"),
        "encoding": dict(_ARCHIVE_ENCODING),
        "members": expected_members,
    }


def _trust_descriptor(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PortableContractError(f"{role} must be a mapping")
    _exact_keys(value, {"path", "size_bytes", "sha256"}, role)
    return {
        "path": _safe_relative_path(value["path"], f"{role}.path"),
        "size_bytes": _size(value["size_bytes"], f"{role}.size_bytes"),
        "sha256": _sha(value["sha256"], f"{role}.sha256"),
    }


def load_portable_contract(path: str | Path) -> dict[str, Any]:
    """Load an exact-schema canonical restore/preflight contract."""

    contract = _read_mapping(Path(path), "portable contract")
    _exact_keys(
        contract,
        {
            "schema_version",
            "purpose",
            "archive",
            "assets",
            "candidate",
            "preprocessing",
            "detector",
            "byte_track",
            "dependencies",
            "source_identity",
            "safeguards",
        },
        "portable contract",
    )
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise PortableContractError("portable contract schema drift")
    if contract["purpose"] != "deployability_only":
        raise PortableContractError("portable contract purpose drift")
    assets = _validate_asset_set(contract["assets"])
    by_role = {asset["role"]: asset for asset in assets}
    archive = _archive_descriptor(contract["archive"], assets)

    candidate = contract["candidate"]
    if not isinstance(candidate, Mapping):
        raise PortableContractError("candidate must be a mapping")
    _exact_keys(
        candidate,
        {
            "candidate_id",
            "primary_seed",
            "best_epoch",
            "manifest_schema_version",
            "manifest_path",
            "manifest_size_bytes",
            "manifest_sha256",
        },
        "candidate",
    )
    manifest_asset = by_role["candidate_manifest"]
    candidate_value = {
        "candidate_id": _safe_token(candidate["candidate_id"], "candidate.candidate_id"),
        "primary_seed": candidate["primary_seed"],
        "best_epoch": candidate["best_epoch"],
        "manifest_schema_version": _safe_token(
            candidate["manifest_schema_version"], "candidate.manifest_schema_version"
        ),
        "manifest_path": _safe_relative_path(candidate["manifest_path"], "candidate.manifest_path"),
        "manifest_size_bytes": _size(candidate["manifest_size_bytes"], "candidate.manifest_size_bytes"),
        "manifest_sha256": _sha(candidate["manifest_sha256"], "candidate.manifest_sha256"),
    }
    if isinstance(candidate_value["primary_seed"], bool) or not isinstance(
        candidate_value["primary_seed"], int
    ):
        raise PortableContractError("candidate seed is invalid")
    if isinstance(candidate_value["best_epoch"], bool) or not isinstance(
        candidate_value["best_epoch"], int
    ):
        raise PortableContractError("candidate epoch is invalid")
    if (
        candidate_value["manifest_path"] != manifest_asset["destination"]
        or candidate_value["manifest_size_bytes"] != manifest_asset["size_bytes"]
        or candidate_value["manifest_sha256"] != manifest_asset["sha256"]
    ):
        raise PortableContractError("candidate manifest cross-binding drift")

    preprocessing = contract["preprocessing"]
    if not isinstance(preprocessing, Mapping):
        raise PortableContractError("preprocessing must be a mapping")
    _exact_keys(preprocessing, {"config", "manifest", "feature_stats"}, "preprocessing")
    preprocessing_value = {
        "config": _trust_descriptor(preprocessing["config"], "preprocessing.config"),
        "manifest": _asset_descriptor(preprocessing["manifest"], "preprocessing.manifest"),
        "feature_stats": _asset_descriptor(
            preprocessing["feature_stats"], "preprocessing.feature_stats"
        ),
    }
    if preprocessing_value["manifest"] != by_role["preprocessing_manifest"]:
        raise PortableContractError("preprocessing manifest cross-binding drift")
    if preprocessing_value["feature_stats"] != by_role["preprocessing_feature_stats"]:
        raise PortableContractError("preprocessing feature-stats cross-binding drift")

    detector = contract["detector"]
    if not isinstance(detector, Mapping):
        raise PortableContractError("detector must be a mapping")
    _exact_keys(detector, {"provenance_token", "model_id", "asset"}, "detector")
    detector_value = {
        "provenance_token": _safe_token(
            detector["provenance_token"], "detector.provenance_token"
        ),
        "model_id": _safe_token(detector["model_id"], "detector.model_id"),
        "asset": _asset_descriptor(detector["asset"], "detector.asset"),
    }
    if detector_value["model_id"] != "yolov8n.pt":
        raise PortableContractError("detector model ID drift")
    if detector_value["asset"] != by_role["detector_weight"]:
        raise PortableContractError("detector asset cross-binding drift")

    byte_track = contract["byte_track"]
    if not isinstance(byte_track, Mapping):
        raise PortableContractError("byte_track must be a mapping")
    _exact_keys(
        byte_track,
        {"distribution", "distribution_version", "resource", "size_bytes", "sha256"},
        "byte_track",
    )
    byte_track_value = {
        "distribution": _safe_token(byte_track["distribution"], "byte_track.distribution"),
        "distribution_version": _safe_token(
            byte_track["distribution_version"], "byte_track.distribution_version"
        ),
        "resource": _safe_relative_path(byte_track["resource"], "byte_track.resource"),
        "size_bytes": _size(byte_track["size_bytes"], "byte_track.size_bytes"),
        "sha256": _sha(byte_track["sha256"], "byte_track.sha256"),
    }

    dependencies = contract["dependencies"]
    if not isinstance(dependencies, list):
        raise PortableContractError("dependencies must be a list")
    dependency_value: list[dict[str, str]] = []
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, Mapping):
            raise PortableContractError("dependency entry must be a mapping")
        _exact_keys(dependency, {"distribution", "specifier"}, f"dependencies[{index}]")
        dependency_value.append(
            {
                "distribution": _safe_token(
                    dependency["distribution"], f"dependencies[{index}].distribution"
                ),
                "specifier": str(dependency["specifier"]),
            }
        )
    if dependency_value != [
        {"distribution": name, "specifier": specifier}
        for name, specifier in REQUIRED_DEPENDENCY_SPECIFIERS
    ]:
        raise PortableContractError("dependency contract drift")

    source_identity = contract["source_identity"]
    if not isinstance(source_identity, Mapping):
        raise PortableContractError("source_identity must be a mapping")
    kind = source_identity.get("kind")
    expected_source_keys = (
        {"kind", "identifier", "base_git_commit", "descriptors", "patch_archive"}
        if kind == "approved_patch_archive"
        else {"kind", "identifier", "base_git_commit", "descriptors"}
    )
    _exact_keys(source_identity, expected_source_keys, "source_identity")
    if kind not in {"git_tree", "approved_patch_archive"}:
        raise PortableContractError("source identity kind is invalid")
    identifier = source_identity["identifier"]
    identifier_pattern = _HEX40 if kind == "git_tree" else _HEX64
    if not isinstance(identifier, str) or not identifier_pattern.fullmatch(identifier):
        raise PortableContractError("source identity identifier is invalid")
    base_git_commit = source_identity["base_git_commit"]
    if not isinstance(base_git_commit, str) or not _HEX40.fullmatch(base_git_commit):
        raise PortableContractError("source identity base commit is invalid")
    if kind == "git_tree" and base_git_commit != identifier:
        raise PortableContractError("Git source identity/base commit cross-binding drift")
    descriptors = source_identity["descriptors"]
    if not isinstance(descriptors, list):
        raise PortableContractError("source descriptors must be a list")
    trust_roots = [
        _trust_descriptor(value, f"source_identity.descriptors[{index}]")
        for index, value in enumerate(descriptors)
    ]
    paths = [descriptor["path"] for descriptor in trust_roots]
    if paths != list(REQUIRED_TRUST_ROOT_PATHS):
        raise PortableContractError("source trust-root path set or order drift")
    if len(paths) != len(set(path.casefold() for path in paths)):
        raise PortableContractError("source trust-root case collision")
    source_value: dict[str, Any] = {
        "kind": kind,
        "identifier": identifier,
        "base_git_commit": base_git_commit,
        "descriptors": trust_roots,
    }
    if kind == "approved_patch_archive":
        patch_descriptor = _trust_descriptor(
            source_identity["patch_archive"], "source_identity.patch_archive"
        )
        if patch_descriptor["sha256"] != identifier:
            raise PortableContractError("approved patch identity cross-binding drift")
        source_value["patch_archive"] = patch_descriptor

    safeguards = contract["safeguards"]
    expected_safeguards = {
        "reads_video": False,
        "runs_tracker": False,
        "runs_qc": False,
        "runs_forward": False,
        "changes_c0_c3": False,
    }
    if safeguards != expected_safeguards:
        raise PortableContractError("portable safeguards drift")

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "purpose": "deployability_only",
        "archive": archive,
        "assets": assets,
        "candidate": candidate_value,
        "preprocessing": preprocessing_value,
        "detector": detector_value,
        "byte_track": byte_track_value,
        "dependencies": dependency_value,
        "source_identity": source_value,
        "safeguards": expected_safeguards,
    }


def _load_source_descriptor(path: Path) -> dict[str, Any]:
    descriptor = _read_mapping(path, "owner-approved source descriptor")
    _exact_keys(
        descriptor,
        {
            "schema_version",
            "purpose",
            "archive_logical_id",
            "detector_provenance_token",
            "assets",
        },
        "owner-approved source descriptor",
    )
    if descriptor["schema_version"] != SOURCE_DESCRIPTOR_SCHEMA_VERSION:
        raise PortableContractError("source descriptor schema drift")
    if descriptor["purpose"] != "owner_approved_offline_runtime_asset_source":
        raise PortableContractError("source descriptor purpose drift")
    values = descriptor["assets"]
    if not isinstance(values, list) or len(values) != len(_EXPECTED_ASSET_LAYOUT):
        raise PortableContractError("source descriptor has the wrong asset count")
    assets: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise PortableContractError("source asset must be a mapping")
        _exact_keys(
            value,
            {"role", "member", "source_path", "destination", "size_bytes", "sha256"},
            f"source assets[{index}]",
        )
        source_path = value["source_path"]
        if not isinstance(source_path, str) or not source_path or len(source_path) > 1024:
            raise PortableContractError("source_path is invalid")
        lowered = source_path.lower()
        windows_spelling = source_path.replace("/", "\\")
        if any(ord(character) < 32 for character in source_path) or any(
            marker in lowered for marker in _SENSITIVE_MARKERS
        ) or windows_spelling.startswith(("\\\\", "\\?\\", "\\.\\", "\\??\\")):
            raise PortableContractError("source_path contains unsafe material")
        if not Path(source_path).is_absolute():
            raise PortableContractError("source_path must be an explicit absolute path")
        asset = _asset_descriptor(
            {key: value[key] for key in ("role", "member", "destination", "size_bytes", "sha256")},
            f"source assets[{index}]",
        )
        asset["source_path"] = source_path
        assets.append(asset)
    normalized = _validate_asset_set(
        [{key: asset[key] for key in ("role", "member", "destination", "size_bytes", "sha256")} for asset in assets]
    )
    sources_by_role = {asset["role"]: asset["source_path"] for asset in assets}
    for asset in normalized:
        asset["source_path"] = sources_by_role[asset["role"]]
    _enforce_source_fixed_identity(normalized)
    return {
        "schema_version": SOURCE_DESCRIPTOR_SCHEMA_VERSION,
        "purpose": descriptor["purpose"],
        "archive_logical_id": _safe_token(
            descriptor["archive_logical_id"], "archive_logical_id"
        ),
        "detector_provenance_token": _safe_token(
            descriptor["detector_provenance_token"], "detector_provenance_token"
        ),
        "assets": normalized,
    }


def _enforce_source_fixed_identity(assets: Sequence[Mapping[str, Any]]) -> None:
    by_role = {str(asset["role"]): asset for asset in assets}
    for role, expected in {**_EXPECTED_CANDIDATE_ASSETS, **_EXPECTED_PREPROCESSING_ASSETS}.items():
        observed = by_role[role]
        if (observed["size_bytes"], observed["sha256"]) != expected:
            raise PortableContractError(f"fixed scientific asset identity drift: {role}")


def _enforce_frozen_scientific_identity(contract: Mapping[str, Any]) -> None:
    candidate = contract["candidate"]
    if (
        candidate["candidate_id"] != EXPECTED_CANDIDATE_ID
        or candidate["primary_seed"] != EXPECTED_CANDIDATE_SEED
        or candidate["best_epoch"] != EXPECTED_CANDIDATE_EPOCH
        or candidate["manifest_schema_version"] != EXPECTED_CANDIDATE_MANIFEST_SCHEMA
    ):
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "fixed candidate scientific identity drift"
        )
    by_role = {asset["role"]: asset for asset in contract["assets"]}
    for role, expected in {**_EXPECTED_CANDIDATE_ASSETS, **_EXPECTED_PREPROCESSING_ASSETS}.items():
        observed = by_role[role]
        if (observed["size_bytes"], observed["sha256"]) != expected:
            raise PortableAssetBlockedError(
                BLOCKER_ASSET_SHA_MISMATCH, f"fixed runtime asset identity drift: {role}"
            )
    preprocessing = contract["preprocessing"]["config"]
    if preprocessing != EXPECTED_PREPROCESSING_CONFIG:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "fixed preprocessing config identity drift",
        )


def _require_external_fresh_output(path: Path, project_root: Path, role: str) -> Path:
    spelling = path.absolute()
    _require_exact_case(spelling)
    if _path_has_symlink_component(spelling.parent):
        raise PortableContractError(f"{role} parent contains a link or reparse point")
    destination = spelling.resolve(strict=False)
    root = project_root.resolve(strict=True)
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise PortableContractError(f"{role} must be outside the project checkout")
    if os.path.lexists(destination):
        raise FileExistsError(f"{role} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(destination.parent):
        raise PortableContractError(f"{role} parent changed to a link or reparse point")
    return destination


def _zip_info(member: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.flag_bits = 0x800
    return info


def build_runtime_asset_bundle(
    *,
    project_root: str | Path,
    source_descriptor_path: str | Path,
    output_archive_path: str | Path,
    output_observed_descriptor_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic offline archive from explicit approved sources."""

    supplied_root = Path(project_root).absolute()
    _require_exact_case(supplied_root)
    if _path_has_symlink_component(supplied_root):
        raise PortableContractError("project root contains a link or reparse point")
    root = supplied_root.resolve(strict=True)
    archive = _require_external_fresh_output(Path(output_archive_path), root, "archive output")
    observed_path = _require_external_fresh_output(
        Path(output_observed_descriptor_path), root, "observed descriptor output"
    )
    if archive == observed_path:
        raise PortableContractError("archive and observed descriptor outputs must be distinct")
    descriptor_spelling = Path(source_descriptor_path).absolute()
    _require_exact_case(descriptor_spelling)
    if _path_has_symlink_component(descriptor_spelling):
        raise PortableContractError("source descriptor contains a link or reparse point")
    descriptor = _load_source_descriptor(descriptor_spelling.resolve(strict=True))
    source_payloads: dict[str, bytes] = {}
    for asset in descriptor["assets"]:
        source = Path(asset["source_path"])
        try:
            _require_exact_case(source.absolute())
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise PortableContractError(f"explicit source is unavailable: {asset['role']}") from exc
        if _path_has_symlink_component(source) or not resolved.is_file():
            raise PortableContractError(f"explicit source is not a regular file: {asset['role']}")
        payload = resolved.read_bytes()
        if len(payload) != asset["size_bytes"] or _sha256_bytes(payload) != asset["sha256"]:
            raise PortableContractError(f"explicit source identity drift: {asset['role']}")
        source_payloads[asset["member"]] = payload

    archive_staging = archive.with_name(f".{archive.name}.portable-staging-{uuid.uuid4().hex}")
    observed_staging = observed_path.with_name(
        f".{observed_path.name}.portable-staging-{uuid.uuid4().hex}"
    )
    archive_created = False
    observed_created = False
    try:
        with zipfile.ZipFile(archive_staging, "x", compression=zipfile.ZIP_STORED) as output:
            for member in sorted(source_payloads):
                output.writestr(_zip_info(member), source_payloads[member])
        archive_payload = archive_staging.read_bytes()
        archive_descriptor = {
            "logical_id": descriptor["archive_logical_id"],
            "filename": archive.name,
            "size_bytes": len(archive_payload),
            "sha256": _sha256_bytes(archive_payload),
            "encoding": dict(_ARCHIVE_ENCODING),
            "members": sorted(
                [
                {
                    "member": asset["member"],
                    "size_bytes": asset["size_bytes"],
                    "sha256": asset["sha256"],
                }
                for asset in descriptor["assets"]
                ],
                key=lambda member_descriptor: member_descriptor["member"],
            ),
        }
        assets = [
            {key: asset[key] for key in ("role", "member", "destination", "size_bytes", "sha256")}
            for asset in descriptor["assets"]
        ]
        observed = {
            "schema_version": OBSERVED_DESCRIPTOR_SCHEMA_VERSION,
            "purpose": "independently_recompute_before_canonical_freeze",
            "archive": archive_descriptor,
            "assets": assets,
            "detector_provenance_token": descriptor["detector_provenance_token"],
        }
        observed_payload = _canonical_json_bytes(observed)
        observed_staging.write_bytes(observed_payload)
        if _canonical_json_bytes(json.loads(observed_staging.read_text(encoding="utf-8"))) != observed_payload:
            raise OSError("observed descriptor staging readback failed")
        try:
            _atomic_rename(archive_staging, archive)
        except BaseException:
            archive_created = not os.path.lexists(archive_staging) and os.path.lexists(
                archive
            )
            raise
        archive_created = True
        try:
            _atomic_rename(observed_staging, observed_path)
        except BaseException:
            observed_created = not os.path.lexists(
                observed_staging
            ) and os.path.lexists(observed_path)
            raise
        observed_created = True
        return observed
    except BaseException as original:
        cleanup_errors: list[str] = []
        for temporary in (archive_staging, observed_staging):
            if os.path.lexists(temporary):
                try:
                    temporary.unlink()
                except BaseException:
                    cleanup_errors.append("staging")
        if archive_created and archive.exists():
            try:
                archive.unlink()
            except BaseException:
                cleanup_errors.append("archive_output")
        if observed_created and observed_path.exists():
            try:
                observed_path.unlink()
            except BaseException:
                cleanup_errors.append("observed_descriptor_output")
        residue = []
        if os.path.lexists(archive_staging) or os.path.lexists(observed_staging):
            residue.append("staging")
        if archive_created and os.path.lexists(archive):
            residue.append("archive_output")
        if observed_created and os.path.lexists(observed_path):
            residue.append("observed_descriptor_output")
        if cleanup_errors or residue:
            roles = ",".join(sorted(set(cleanup_errors + residue)))
            raise OSError(f"bundle rollback incomplete for logical roles: {roles}") from original
        raise


def _atomic_rename(source: Path, destination: Path) -> None:
    """Atomically rename one sibling staging object without replacing a rival."""

    if os.name == "nt":
        os.rename(source, destination)
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise OSError(
            errno.ENOSYS,
            "atomic no-replace rename is unavailable on this filesystem runtime",
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _require_exact_case(path: Path) -> None:
    """Reject aliases whose spelling differs from an existing directory entry."""

    absolute = path.absolute()
    if absolute.anchor:
        current = Path(absolute.anchor)
        parts = absolute.parts[1:]
    else:
        current = Path.cwd()
        parts = absolute.parts
    for part in parts:
        if not current.is_dir():
            return
        try:
            names = {entry.name for entry in os.scandir(current)}
        except OSError as exc:
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
                "path case identity cannot be verified",
            ) from exc
        candidate = current / part
        if candidate.exists() or os.path.lexists(candidate):
            if part not in names:
                raise PortableAssetBlockedError(
                    BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
                    "path uses a case alias",
                )
        current = candidate


def _path_has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current = current / part
        if _is_link_or_reparse(current):
            return True
    return False


def _require_active_checkout(project_root: str | Path) -> Path:
    try:
        supplied_root = Path(project_root).absolute()
        _require_exact_case(supplied_root)
        if _path_has_symlink_component(supplied_root):
            raise OSError("caller checkout contains a link or reparse component")
        root = supplied_root.resolve(strict=True)
        if not root.is_dir():
            raise OSError("caller checkout is not a directory")
        expected_spelling = root / PORTABILITY_SOURCE_RELATIVE_PATH
        _require_exact_case(expected_spelling)
        if _path_has_symlink_component(expected_spelling):
            raise OSError("active portability source contains a link or reparse component")
        expected = expected_spelling.resolve(strict=True)
        if not _ACTIVE_MODULE_FILE.samefile(expected):
            raise OSError("active portability source is outside caller checkout")
        editable = _editable_project_root()
        if not editable.samefile(root):
            raise OSError("editable distribution points outside caller checkout")
    except OSError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_EDITABLE_CHECKOUT_MISMATCH, "active/editable checkout identity mismatch"
        ) from exc
    return root


def _editable_project_root() -> Path:
    distribution = importlib_metadata.distribution("elderly-monitoring-algorithms")
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise OSError("editable direct_url metadata is unavailable")
    try:
        direct_url = json.loads(direct_url_text)
        if direct_url.get("dir_info", {}).get("editable") is not True:
            raise OSError("project distribution is not editable")
        parsed = urlparse(direct_url["url"])
        if parsed.scheme != "file":
            raise OSError("editable distribution is not file-backed")
        path_text = unquote(parsed.path)
        if os.name == "nt" and path_text.startswith("/") and PureWindowsPath(path_text[1:]).drive:
            path_text = path_text[1:]
        return Path(path_text).resolve(strict=True)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OSError("editable direct_url metadata is invalid") from exc


def _fixed_contract_path(root: Path) -> Path:
    candidate = root / CANONICAL_CONTRACT_RELATIVE_PATH
    _require_exact_case(candidate)
    if _path_has_symlink_component(candidate):
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "canonical portable contract path contains a link or reparse point",
        )
    if not candidate.is_file():
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SOURCE_UNAVAILABLE, "canonical portable contract is unavailable"
        )
    resolved = candidate.resolve(strict=True)
    if not resolved.samefile(candidate):
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "canonical portable contract identity drift",
        )
    return resolved


def _safe_destination(root: Path, relative: str) -> Path:
    validated = _safe_relative_path(relative, "destination")
    destination = root / validated
    current = root
    for part in PurePosixPath(validated).parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
                "destination path contains a link or reparse point",
            )
    _require_exact_case(destination)
    try:
        destination.resolve(strict=False).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise PortableContractError("destination escapes the project checkout") from exc
    return destination


def _verify_archive(
    archive_path: Path, contract: Mapping[str, Any]
) -> dict[str, bytes]:
    try:
        archive = archive_path.resolve(strict=True)
    except OSError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SOURCE_UNAVAILABLE, "offline archive is unavailable"
        ) from exc
    if _path_has_symlink_component(archive_path.absolute()) or not archive.is_file():
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SOURCE_UNAVAILABLE, "offline archive is not a regular file"
        )
    descriptor = contract["archive"]
    if archive.name != descriptor["filename"]:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH, "offline archive filename drift"
        )
    if archive.stat().st_size != descriptor["size_bytes"] or _sha256_file(archive) != descriptor["sha256"]:
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "offline archive identity drift")
    expected = {asset["member"]: asset for asset in contract["assets"]}
    payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(archive, "r") as source:
            infos = source.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
                raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "duplicate archive member")
            if names != sorted(expected):
                raise PortableAssetBlockedError(
                    BLOCKER_ASSET_SHA_MISMATCH, "archive member set or order drift"
                )
            total = 0
            for info in infos:
                _safe_relative_path(info.filename, "archive member")
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type not in {0, stat.S_IFREG}:
                    raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "archive symlink refused")
                asset = expected[info.filename]
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.file_size != asset["size_bytes"]
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.create_system != 3
                    or (info.external_attr >> 16) != (stat.S_IFREG | 0o600)
                    or info.extra != b""
                    or info.comment != b""
                ):
                    raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "archive member size drift")
                total += info.file_size
                if total > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "archive size limit exceeded")
                with source.open(info, "r") as handle:
                    payload = handle.read(asset["size_bytes"] + 1)
                if len(payload) != asset["size_bytes"] or _sha256_bytes(payload) != asset["sha256"]:
                    raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "archive member identity drift")
                payloads[info.filename] = payload
    except (zipfile.BadZipFile, OSError, KeyError) as exc:
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "cannot verify offline archive") from exc
    return payloads


def _asset_set_sha256(assets: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json_bytes(list(assets)))


def _ready_marker(contract_bytes: bytes, contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": READY_MARKER_SCHEMA_VERSION,
        "purpose": "deployability_only",
        "contract_sha256": _sha256_bytes(contract_bytes),
        "archive_sha256": contract["archive"]["sha256"],
        "asset_set_sha256": _asset_set_sha256(contract["assets"]),
        "committed_groups": ["candidate", "preprocessing", "detector"],
        "reads_video": False,
        "runs_tracker": False,
        "runs_qc": False,
        "runs_forward": False,
    }


def _remove_created(path: Path) -> None:
    if _is_link_or_reparse(path):
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


def _ensure_parent_directories(path: Path, root: Path, created: list[Path]) -> None:
    missing: list[Path] = []
    current = path
    while current != root and not current.exists():
        missing.append(current)
        current = current.parent
    if _is_link_or_reparse(current):
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "restore parent contains a link or reparse point",
        )
    for directory in reversed(missing):
        directory.mkdir()
        created.append(directory)


def restore_camera_runtime_assets(
    *, project_root: str | Path, archive_path: str | Path
) -> dict[str, Any]:
    """Fresh-only, grouped, failure-atomic restore of ignored runtime assets."""

    root = _require_active_checkout(project_root)
    contract_path = _fixed_contract_path(root)
    contract_bytes = contract_path.read_bytes()
    contract = load_portable_contract(contract_path)
    _enforce_frozen_scientific_identity(contract)
    payloads = _verify_archive(Path(archive_path), contract)

    group_finals = {
        group: _safe_destination(root, relative)
        for group, relative in _GROUP_DESTINATION_BY_NAME.items()
    }
    marker_final = _safe_destination(root, READY_MARKER_RELATIVE_PATH)
    for destination in (*group_finals.values(), marker_final):
        if os.path.lexists(destination):
            raise PortableAssetBlockedError(
                BLOCKER_ASSET_SOURCE_UNAVAILABLE, "restore-managed destination already exists"
            )
        parent = destination.parent
        if parent.exists() and list(parent.glob(f".{destination.name}.portable-staging-*")):
            raise PortableAssetBlockedError(
                BLOCKER_ASSET_SOURCE_UNAVAILABLE, "restore staging residue already exists"
            )

    staged: dict[str, Path] = {}
    created: list[Path] = []
    created_parents: list[Path] = []
    marker_staging: Path | None = None
    try:
        for group in ("candidate", "preprocessing", "detector"):
            final = group_finals[group]
            _ensure_parent_directories(final.parent, root, created_parents)
            staging = final.with_name(f".{final.name}.portable-staging-{uuid.uuid4().hex}")
            staging.mkdir()
            staged[group] = staging
            for asset in contract["assets"]:
                if _GROUP_BY_ROLE[asset["role"]] != group:
                    continue
                relative_inside_group = Path(asset["destination"]).relative_to(
                    _GROUP_DESTINATION_BY_NAME[group]
                )
                destination = staging / relative_inside_group
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(payloads[asset["member"]])
                    handle.flush()
                    os.fsync(handle.fileno())
                if destination.stat().st_size != asset["size_bytes"] or _sha256_file(destination) != asset["sha256"]:
                    raise OSError("restore staging identity readback failed")

        for group in ("candidate", "preprocessing", "detector"):
            final = group_finals[group]
            if os.path.lexists(final):
                raise PortableAssetBlockedError(
                    BLOCKER_ASSET_SOURCE_UNAVAILABLE, "restore destination appeared during staging"
                )
            try:
                _atomic_rename(staged[group], final)
            except BaseException:
                if not os.path.lexists(staged[group]) and os.path.lexists(final):
                    created.append(final)
                raise
            created.append(final)

        marker = _ready_marker(contract_bytes, contract)
        marker_payload = _canonical_json_bytes(marker)
        _ensure_parent_directories(marker_final.parent, root, created_parents)
        marker_staging = marker_final.with_name(
            f".{marker_final.name}.portable-staging-{uuid.uuid4().hex}"
        )
        with marker_staging.open("xb") as handle:
            handle.write(marker_payload)
            handle.flush()
            os.fsync(handle.fileno())
        if marker_staging.read_bytes() != marker_payload:
            raise OSError("ready marker staging readback failed")
        try:
            _atomic_rename(marker_staging, marker_final)
        except BaseException:
            if not os.path.lexists(marker_staging) and os.path.lexists(marker_final):
                created.append(marker_final)
            raise
        created.append(marker_final)
        if json.loads(marker_final.read_text(encoding="utf-8")) != marker:
            raise OSError("ready marker final readback failed")
        return {
            "schema_version": READY_MARKER_SCHEMA_VERSION,
            "status": "ready",
            "archive_logical_id": contract["archive"]["logical_id"],
            "asset_set_sha256": marker["asset_set_sha256"],
            "committed_groups": marker["committed_groups"],
        }
    except BaseException as original:
        cleanup_errors: list[str] = []
        residue_roles: list[str] = []
        for group, staging in staged.items():
            if os.path.lexists(staging):
                try:
                    _remove_created(staging)
                except BaseException:
                    cleanup_errors.append(f"{group}_staging")
        if marker_staging is not None and os.path.lexists(marker_staging):
            try:
                _remove_created(marker_staging)
            except BaseException:
                cleanup_errors.append("ready_marker_staging")
        role_by_destination = {
            **{path: group for group, path in group_finals.items()},
            marker_final: "ready_marker",
        }
        for destination in reversed(created):
            try:
                _remove_created(destination)
            except BaseException:
                cleanup_errors.append(role_by_destination[destination])
        for directory in reversed(created_parents):
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            except BaseException:
                cleanup_errors.append("created_parent")
        for group, destination in group_finals.items():
            if destination in created and os.path.lexists(destination):
                residue_roles.append(group)
        if marker_final in created and os.path.lexists(marker_final):
            residue_roles.append("ready_marker")
        for group, staging in staged.items():
            if os.path.lexists(staging):
                residue_roles.append(f"{group}_staging")
        if marker_staging is not None and os.path.lexists(marker_staging):
            residue_roles.append("ready_marker_staging")
        if cleanup_errors or residue_roles:
            roles = ",".join(sorted(set(cleanup_errors + residue_roles)))
            raise OSError(
                f"restore rollback incomplete for logical roles: {roles}"
            ) from original
        raise


def _regular_file_identity(path: Path) -> tuple[int, str, os.stat_result]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise OSError("path is not a regular file")
    observed_sha = _sha256_file(path)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise OSError("file identity changed while hashing")
    return after.st_size, observed_sha, after


def _verify_regular_asset(root: Path, asset: Mapping[str, Any]) -> Path:
    destination = _safe_destination(root, str(asset["destination"]))
    if not destination.exists():
        blocker = (
            BLOCKER_DETECTOR_WEIGHT_MISSING
            if asset["role"] == "detector_weight"
            else BLOCKER_ASSET_SOURCE_UNAVAILABLE
        )
        raise PortableAssetBlockedError(blocker, f"runtime asset is missing: {asset['role']}")
    if _is_link_or_reparse(destination) or not destination.is_file():
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "runtime asset is not regular")
    try:
        observed_size, observed_sha, _metadata = _regular_file_identity(destination)
    except OSError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH, "runtime asset identity cannot be verified"
        ) from exc
    if observed_size != asset["size_bytes"] or observed_sha != asset["sha256"]:
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "runtime asset identity drift")
    return destination.resolve(strict=True)


def _verify_ready_marker(root: Path, contract_path: Path, contract: Mapping[str, Any]) -> None:
    marker_path = _safe_destination(root, READY_MARKER_RELATIVE_PATH)
    if not marker_path.is_file() or _is_link_or_reparse(marker_path):
        raise PortableAssetBlockedError(BLOCKER_ASSET_SOURCE_UNAVAILABLE, "ready marker is unavailable")
    try:
        marker_payload = marker_path.read_bytes()
    except OSError as exc:
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "ready marker is invalid") from exc
    marker = _unique_json_object(marker_payload, "ready marker")
    expected = _ready_marker(contract_path.read_bytes(), contract)
    if marker != expected or marker_payload != _canonical_json_bytes(expected):
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "ready marker binding drift")


def _git_blob_bytes(root: Path, commit: str, relative: str) -> bytes:
    """Read one specified commit blob without consulting working-tree bytes."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{relative}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError) as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "Git object verification is unavailable",
        ) from exc
    if completed.returncode != 0:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "required Git blob is unavailable",
        )
    return completed.stdout


def _verify_trust_roots(root: Path, contract: Mapping[str, Any]) -> None:
    source_identity = contract["source_identity"]
    if source_identity["kind"] == "approved_patch_archive":
        raise PortableAssetBlockedError(
            BLOCKER_SOURCE_INTEGRATION_PENDING,
            "approved patch mode lacks a contract-external approval anchor",
        )

    commit = source_identity["identifier"]
    observed_head = _read_git_head_without_subprocess(root)
    if observed_head != commit:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "active Git source identity drift"
        )
    contract_path = _fixed_contract_path(root)
    contract_payload = contract_path.read_bytes()
    if _git_blob_bytes(root, commit, CANONICAL_CONTRACT_RELATIVE_PATH) != contract_payload:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "canonical contract differs from the specified Git blob",
        )

    for descriptor in source_identity["descriptors"]:
        path = _safe_destination(root, descriptor["path"])
        if _is_link_or_reparse(path) or not path.is_file():
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "active source or config is unavailable"
            )
        committed_payload = _git_blob_bytes(root, commit, descriptor["path"])
        if (
            len(committed_payload) != descriptor["size_bytes"]
            or _sha256_bytes(committed_payload) != descriptor["sha256"]
        ):
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
                "source descriptor differs from the specified Git blob",
            )
        try:
            observed_size, observed_sha, _metadata = _regular_file_identity(path)
        except OSError as exc:
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
                "active source or config identity cannot be verified",
            ) from exc
        if observed_size != len(committed_payload) or observed_sha != _sha256_bytes(
            committed_payload
        ):
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "active source or config identity drift"
            )


def _read_git_head_without_subprocess(root: Path) -> str:
    dot_git = root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise PortableAssetBlockedError(
                BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "Git metadata indirection is invalid"
            )
        git_dir = (root / text.removeprefix("gitdir: ")).resolve(strict=True)
    else:
        git_dir = dot_git.resolve(strict=True)
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = _safe_relative_path(head.removeprefix("ref: "), "Git HEAD reference")
        reference_path = git_dir / reference
        if reference_path.is_file():
            head = reference_path.read_text(encoding="utf-8").strip()
        else:
            packed = git_dir / "packed-refs"
            matches = [
                line.split(" ", 1)[0]
                for line in packed.read_text(encoding="utf-8").splitlines()
                if line.endswith(f" {reference}")
            ]
            if len(matches) != 1:
                raise PortableAssetBlockedError(
                    BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "Git HEAD cannot be resolved"
                )
            head = matches[0]
    if not _HEX40.fullmatch(head):
        raise PortableAssetBlockedError(BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT, "Git HEAD is invalid")
    return head


def _installed_dependency_versions() -> dict[str, str]:
    versions = {"python": ".".join(str(value) for value in sys.version_info[:3])}
    for name, _specifier in REQUIRED_DEPENDENCY_SPECIFIERS:
        if name != "python":
            versions[name] = importlib_metadata.version(name)
    return versions


def _verify_dependencies(contract: Mapping[str, Any]) -> dict[str, str]:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_DEPENDENCY_VERSION_MISMATCH, "PEP 440 verifier is unavailable"
        ) from exc
    try:
        installed = _installed_dependency_versions()
    except importlib_metadata.PackageNotFoundError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_DEPENDENCY_VERSION_MISMATCH, "required distribution is unavailable"
        ) from exc
    for dependency in contract["dependencies"]:
        name = dependency["distribution"]
        try:
            if Version(installed[name]) not in SpecifierSet(dependency["specifier"]):
                raise PortableAssetBlockedError(
                    BLOCKER_DEPENDENCY_VERSION_MISMATCH, "dependency version mismatch"
                )
        except (KeyError, ValueError) as exc:
            raise PortableAssetBlockedError(
                BLOCKER_DEPENDENCY_VERSION_MISMATCH, "dependency version is invalid"
            ) from exc
    return installed


def _verify_preprocessing_cross_binding(root: Path, contract: Mapping[str, Any]) -> None:
    manifest_asset = next(
        asset for asset in contract["assets"] if asset["role"] == "preprocessing_manifest"
    )
    stats_asset = next(
        asset for asset in contract["assets"] if asset["role"] == "preprocessing_feature_stats"
    )
    manifest_path = _safe_destination(root, manifest_asset["destination"])
    stats_path = _safe_destination(root, stats_asset["destination"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "preprocessing binding is invalid") from exc
    config_sha = contract["preprocessing"]["config"]["sha256"]
    if (
        manifest.get("schema_version") != "wandering-preprocessing-manifest-v1"
        or manifest.get("preprocessing_config_sha256") != config_sha
        or manifest.get("artifacts", {}).get("feature_stats.json")
        != {"byte_count": stats_asset["size_bytes"], "sha256": stats_asset["sha256"]}
        or stats.get("schema_version") != "wandering-feature-stats-v1"
        or stats.get("binding_hashes", {}).get("preprocessing_config") != config_sha
    ):
        raise PortableAssetBlockedError(BLOCKER_ASSET_SHA_MISMATCH, "preprocessing cross-binding drift")


def _verify_candidate_source_cross_bindings(
    root: Path, manifest_path: Path, contract: Mapping[str, Any]
) -> None:
    """Bind the frozen candidate's implementation identities to active trust roots."""

    try:
        manifest = _unique_json_object(manifest_path.read_bytes(), "candidate manifest")
        release_files = manifest["release_implementation_identity"]["identity"]["files"]
        training_files = manifest["training_candidate_identity"]["identity"][
            "training_source_bundle"
        ]["files"]
    except (OSError, KeyError, TypeError) as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH,
            "candidate source identity is unavailable",
        ) from exc
    model_path = "src/elderly_monitoring/modules/mental_health/wandering/model.py"
    release_path = "src/elderly_monitoring/modules/mental_health/wandering/release.py"
    try:
        release_model = {
            key: release_files[model_path][key] for key in ("size_bytes", "sha256")
        }
        training_model = {
            key: training_files[model_path][key] for key in ("size_bytes", "sha256")
        }
        release_loader = {
            key: release_files[release_path][key] for key in ("size_bytes", "sha256")
        }
    except (KeyError, TypeError) as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "candidate implementation source identity is incomplete",
        ) from exc
    trust_by_path = {
        descriptor["path"]: {
            "size_bytes": descriptor["size_bytes"],
            "sha256": descriptor["sha256"],
        }
        for descriptor in contract["source_identity"]["descriptors"]
    }
    if (
        release_model != training_model
        or release_model != trust_by_path.get(model_path)
        or release_loader != trust_by_path.get(release_path)
    ):
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "candidate implementation source identity drift",
        )


def _safe_load_candidate(manifest_path: Path, expected_sha256: str) -> Any:
    from elderly_monitoring.modules.mental_health.wandering.release import (
        load_candidate_from_manifest,
    )

    return load_candidate_from_manifest(
        manifest_path, expected_manifest_sha256=expected_sha256
    )


def _load_detector_without_inference(weight_path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weight_path), task="detect", verbose=False)


def _package_bytetrack_descriptor() -> dict[str, Any]:
    resource = importlib_resources.files("ultralytics").joinpath("cfg/trackers/bytetrack.yaml")
    with importlib_resources.as_file(resource) as path:
        resolved = Path(path).resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file():
            raise OSError("package ByteTrack resource is not a regular file")
        payload = resolved.read_bytes()
    try:
        parsed = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise OSError("package ByteTrack resource is not parseable") from exc
    if not isinstance(parsed, dict) or parsed.get("tracker_type") != "bytetrack":
        raise OSError("package ByteTrack resource type drift")
    return {
        "distribution": "ultralytics",
        "distribution_version": importlib_metadata.version("ultralytics"),
        "resource": "cfg/trackers/bytetrack.yaml",
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


@contextmanager
def deny_network_access() -> Iterator[None]:
    """Deny socket connection attempts for local model loading/tracking."""

    attempt_start = len(_NETWORK_ATTEMPTS)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def denied(*args: Any, **kwargs: Any) -> Any:
        _NETWORK_ATTEMPTS.append("socket")
        raise NetworkAccessDeniedError("network access denied by camera portability guard")

    try:
        socket.socket.connect = denied  # type: ignore[method-assign]
        socket.socket.connect_ex = denied  # type: ignore[method-assign]
        socket.create_connection = denied
    except Exception as exc:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        raise PortableAssetBlockedError(
            BLOCKER_NETWORK_GUARD_UNAVAILABLE, "network denial guard cannot be enforced"
        ) from exc
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        if len(_NETWORK_ATTEMPTS) > attempt_start:
            raise NetworkAccessDeniedError(
                "network access was attempted inside camera portability guard"
            )


def _check(check_id: str, status: str, **details: Any) -> dict[str, Any]:
    return {"check_id": check_id, "status": status, **details}


def _preflight_base() -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "purpose": "deployability_only",
        "status": "blocked",
        "blocker_code": None,
        "checks": [],
        "network_denial_guard_enforced": False,
        "network_attempt_count": 0,
        "safeguards": {
            "reads_video": False,
            "runs_tracker": False,
            "runs_qc": False,
            "runs_forward": False,
            "changes_c0_c3": False,
        },
    }


def _commit_report(output_path: Path, report: Mapping[str, Any]) -> None:
    if os.path.lexists(output_path):
        raise FileExistsError("preflight report output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(
        f".{output_path.name}.portable-staging-{uuid.uuid4().hex}"
    )
    payload = _canonical_json_bytes(report)
    output_created = False
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if staging.read_bytes() != payload:
            raise OSError("preflight report readback failed")
        try:
            _atomic_rename(staging, output_path)
        except BaseException:
            output_created = not os.path.lexists(staging) and os.path.lexists(
                output_path
            )
            raise
        output_created = True
    except BaseException as original:
        cleanup_error = False
        if os.path.lexists(staging):
            try:
                staging.unlink()
            except BaseException:
                cleanup_error = True
        if output_created and os.path.lexists(output_path):
            try:
                output_path.unlink()
            except BaseException:
                cleanup_error = True
        if cleanup_error:
            raise OSError(
                "preflight report rollback incomplete for logical output"
            ) from original
        raise


def preflight_camera_runtime(
    *, project_root: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    """Return only a structured zero-video deployability ready/blocked report."""

    report = _preflight_base()
    attempt_start = len(_NETWORK_ATTEMPTS)
    current_check = "active_editable_checkout"
    try:
        with deny_network_access():
            report["network_denial_guard_enforced"] = True
            root = _require_active_checkout(project_root)
            report["checks"].append(_check(current_check, "ready"))

            current_check = "canonical_contract_and_source_trust"
            contract_path = _fixed_contract_path(root)
            contract = load_portable_contract(contract_path)
            _enforce_frozen_scientific_identity(contract)
            _verify_trust_roots(root, contract)
            report["checks"].append(
                _check(
                    current_check,
                    "ready",
                    source_identity=contract["source_identity"]["identifier"],
                )
            )

            current_check = "pep440_dependencies"
            installed = _verify_dependencies(contract)
            report["checks"].append(
                _check(current_check, "ready", verified_distribution_count=len(installed))
            )

            current_check = "runtime_asset_identity"
            _verify_ready_marker(root, contract_path, contract)
            verified_assets = {
                asset["role"]: _verify_regular_asset(root, asset)
                for asset in contract["assets"]
            }
            report["checks"].append(
                _check(
                    current_check,
                    "ready",
                    asset_set_sha256=_asset_set_sha256(contract["assets"]),
                    verified_asset_count=len(verified_assets),
                )
            )

            current_check = "preprocessing_cross_binding"
            _verify_preprocessing_cross_binding(root, contract)
            report["checks"].append(_check(current_check, "ready"))

            current_check = "active_runtime_source_identity"
            _verify_trust_roots(root, contract)
            report["checks"].append(_check(current_check, "ready"))

            current_check = "candidate_safe_load"
            _verify_candidate_source_cross_bindings(
                root, verified_assets["candidate_manifest"], contract
            )
            try:
                with deny_network_access():
                    runtime = _safe_load_candidate(
                        verified_assets["candidate_manifest"],
                        contract["candidate"]["manifest_sha256"],
                    )
                    if getattr(getattr(runtime, "model", None), "training", None) is not False:
                        raise RuntimeError("candidate did not safe-load in eval mode")
            except Exception as exc:
                if isinstance(exc, NetworkAccessDeniedError):
                    raise PortableAssetBlockedError(
                        BLOCKER_NETWORK_GUARD_UNAVAILABLE, "candidate safe-load attempted network access"
                    ) from exc
                raise PortableAssetBlockedError(
                    BLOCKER_CANDIDATE_SAFE_LOAD_FAILED, "candidate safe-load failed"
                ) from exc
            report["checks"].append(_check(current_check, "ready"))

            current_check = "detector_local_loadability"
            detector_asset = next(
                asset for asset in contract["assets"] if asset["role"] == "detector_weight"
            )
            detector_path = _verify_regular_asset(root, detector_asset)
            try:
                with deny_network_access():
                    _load_detector_without_inference(detector_path)
            except Exception as exc:
                if isinstance(exc, NetworkAccessDeniedError):
                    raise PortableAssetBlockedError(
                        BLOCKER_NETWORK_GUARD_UNAVAILABLE, "detector load attempted network access"
                    ) from exc
                raise PortableAssetBlockedError(
                    BLOCKER_ASSET_SHA_MISMATCH, "local detector is not loadable"
                ) from exc
            report["checks"].append(
                _check(
                    current_check,
                    "ready",
                    detector_model_id=contract["detector"]["model_id"],
                    expected_size=detector_asset["size_bytes"],
                    observed_size=detector_asset["size_bytes"],
                    expected_sha256=detector_asset["sha256"],
                    observed_sha256=detector_asset["sha256"],
                )
            )

            current_check = "package_bytetrack_identity"
            try:
                observed_tracker = _package_bytetrack_descriptor()
            except (OSError, importlib_metadata.PackageNotFoundError) as exc:
                raise PortableAssetBlockedError(
                    BLOCKER_ASSET_SOURCE_UNAVAILABLE,
                    "package ByteTrack resource is unavailable",
                ) from exc
            if observed_tracker != contract["byte_track"]:
                raise PortableAssetBlockedError(
                    BLOCKER_ASSET_SHA_MISMATCH, "package ByteTrack config identity drift"
                )
            report["checks"].append(
                _check(
                    current_check,
                    "ready",
                    expected_size=contract["byte_track"]["size_bytes"],
                    observed_size=observed_tracker["size_bytes"],
                    expected_sha256=contract["byte_track"]["sha256"],
                    observed_sha256=observed_tracker["sha256"],
                )
            )
        report["status"] = "ready"
    except NetworkAccessDeniedError:
        report["blocker_code"] = BLOCKER_NETWORK_GUARD_UNAVAILABLE
        report["checks"].append(
            _check(
                current_check,
                "blocked",
                blocker_code=BLOCKER_NETWORK_GUARD_UNAVAILABLE,
            )
        )
    except PortableAssetBlockedError as exc:
        report["blocker_code"] = exc.blocker_code
        report["checks"].append(_check(current_check, "blocked", blocker_code=exc.blocker_code))
    finally:
        report["network_attempt_count"] = len(_NETWORK_ATTEMPTS) - attempt_start
    if output_path is not None:
        _commit_report(Path(output_path), report)
    return report


def resolve_verified_detector_asset(
    project_root: str | Path,
) -> VerifiedDetectorAsset:
    """Resolve and freeze the local detector identity after authorization gates."""

    root = _require_active_checkout(project_root)
    contract_path = _fixed_contract_path(root)
    contract_bytes = contract_path.read_bytes()
    contract = load_portable_contract(contract_path)
    if contract_path.read_bytes() != contract_bytes:
        raise PortableAssetBlockedError(
            BLOCKER_ACTIVE_SOURCE_OR_CONFIG_DRIFT,
            "canonical contract changed while resolving detector",
        )
    _enforce_frozen_scientific_identity(contract)
    _verify_trust_roots(root, contract)
    marker_path = _safe_destination(root, READY_MARKER_RELATIVE_PATH)
    _verify_ready_marker(root, contract_path, contract)
    try:
        marker_bytes = marker_path.read_bytes()
        _verify_ready_marker(root, contract_path, contract)
        marker_bytes_after = marker_path.read_bytes()
    except OSError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH,
            "ready marker identity cannot be frozen",
        ) from exc
    if marker_bytes_after != marker_bytes:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH,
            "ready marker changed while resolving detector",
        )
    detector_asset = next(
        asset for asset in contract["assets"] if asset["role"] == "detector_weight"
    )
    detector_path = _verify_regular_asset(root, detector_asset)
    try:
        observed_size, observed_sha, metadata = _regular_file_identity(detector_path)
    except OSError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH,
            "detector identity cannot be frozen",
        ) from exc
    if (
        observed_size != detector_asset["size_bytes"]
        or observed_sha != detector_asset["sha256"]
    ):
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH, "detector identity drift"
        )
    return VerifiedDetectorAsset(
        project_root=root,
        path=detector_path,
        size_bytes=observed_size,
        sha256=observed_sha,
        contract_sha256=_sha256_bytes(contract_bytes),
        ready_marker_sha256=_sha256_bytes(marker_bytes),
        stat_device=metadata.st_dev,
        stat_inode=metadata.st_ino,
    )


def verify_detector_asset_again(
    project_root: str | Path, detector: VerifiedDetectorAsset
) -> Path:
    """Re-verify only the immutable first-pass token immediately before YOLO."""

    if not isinstance(detector, VerifiedDetectorAsset):
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH, "detector verification token is invalid"
        )
    root = _require_active_checkout(project_root)
    try:
        if not root.samefile(detector.project_root):
            raise OSError("detector token checkout drift")
        contract_path = _fixed_contract_path(root)
        if _sha256_bytes(contract_path.read_bytes()) != detector.contract_sha256:
            raise OSError("detector token contract drift")
        marker_path = _safe_destination(root, READY_MARKER_RELATIVE_PATH)
        if (
            not marker_path.is_file()
            or _is_link_or_reparse(marker_path)
            or _sha256_bytes(marker_path.read_bytes())
            != detector.ready_marker_sha256
        ):
            raise OSError("detector token ready-marker drift")
        expected = _safe_destination(root, DETECTOR_DESTINATION)
        if not detector.path.samefile(expected):
            raise OSError("detector token path drift")
        observed_size, observed_sha, metadata = _regular_file_identity(expected)
    except OSError as exc:
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH,
            "detector identity changed before tracking",
        ) from exc
    if (
        observed_size != detector.size_bytes
        or observed_sha != detector.sha256
        or metadata.st_dev != detector.stat_device
        or metadata.st_ino != detector.stat_inode
    ):
        raise PortableAssetBlockedError(
            BLOCKER_ASSET_SHA_MISMATCH,
            "detector identity changed before tracking",
        )
    return expected.resolve(strict=True)


__all__ = [
    "BLOCKER_ASSET_SOURCE_UNAVAILABLE",
    "CANONICAL_CONTRACT_RELATIVE_PATH",
    "PortableAssetBlockedError",
    "PortableContractError",
    "VerifiedDetectorAsset",
    "build_runtime_asset_bundle",
    "deny_network_access",
    "load_portable_contract",
    "preflight_camera_runtime",
    "resolve_verified_detector_asset",
    "restore_camera_runtime_assets",
    "verify_detector_asset_again",
]
