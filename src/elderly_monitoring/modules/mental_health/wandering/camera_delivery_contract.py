from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import yaml


DELIVERY_CONTRACT_CONFIG_SCHEMA_VERSION = (
    "wandering-camera-5d-delivery-contract-config-v1"
)
DEVELOPMENT_INDEX_SCHEMA_VERSION = "wandering-camera-development-index-v1"
DELIVERY_CONTRACT_MANIFEST_SCHEMA_VERSION = (
    "wandering-camera-5d-contract-manifest-v1"
)
UNIFIED_STATUSES = ("ready", "uncertain", "unavailable", "error")
REQUIRED_HANDOFF_FILES = (
    "episode_results.jsonl",
    "context_reviews.jsonl",
    "daily_reports.jsonl",
    "baseline_profiles.jsonl",
    "baseline_deviations.jsonl",
    "handoff_manifest.json",
    "run_summary.json",
    "README.md",
)
REQUIRED_SCHEMA_IDS = {
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
REQUIRED_SCHEMA_NAMES = tuple(REQUIRED_SCHEMA_IDS)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "development_index",
        "home_acceptance_input",
        "fixed_primary",
        "runtime",
        "output_contract",
    }
)
_BATCH_FIELDS = frozenset(
    {
        "batch_id",
        "participant_id",
        "session_id",
        "camera_setup_id",
        "clock_domain_id",
        "expected_development_video_count",
        "accepted_legacy_setup_ids",
    }
)
_ARTIFACT_SET_FIELDS = frozenset(
    {"batch_id", "tracking_root", "truth_root", "historical_role"}
)
_EXCLUDED_FIELDS = frozenset({"batch_id", "filename", "reason"})
_DESCRIPTOR_FIELDS = frozenset({"path", "sha256"})
_PRIMARY_VERIFIER = Callable[..., Mapping[str, Any]]


class DeliveryContractError(ValueError):
    """The W5D contract, input binding, or immutable artifact drifted."""


@dataclass(frozen=True)
class DeliveryContractBuildResult:
    output_dir: Path
    development_record_count: int
    batch_counts: dict[str, int]
    manifest_sha256: str


def load_delivery_contract_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the versioned W5D-00 contract configuration."""

    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DeliveryContractError("cannot read delivery contract config") from exc
    if not isinstance(value, dict):
        raise DeliveryContractError("delivery contract config must be a mapping")
    _require_exact_fields(value, _CONFIG_FIELDS, "delivery contract config")
    if value.get("schema_version") != DELIVERY_CONTRACT_CONFIG_SCHEMA_VERSION:
        raise DeliveryContractError("delivery contract config schema drifted")
    _token(value.get("contract_id"), "contract_id")
    _validate_development_config(value.get("development_index"))
    _validate_home_input(value.get("home_acceptance_input"))
    _validate_fixed_primary(value.get("fixed_primary"))
    _validate_runtime(value.get("runtime"))
    _validate_output_contract(value.get("output_contract"))
    return value


def build_wandering_camera_delivery_contract(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    source_root: str | Path | None = None,
    primary_runtime_verifier: _PRIMARY_VERIFIER | None = None,
) -> DeliveryContractBuildResult:
    """Build the verified B01+B02 index and immutable W5D output contract."""

    output = Path(output_dir).resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"delivery contract output already exists: {output}")
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    _require_under(root, config_file, "config_path")
    config = load_delivery_contract_config(config_file)

    primary = config["fixed_primary"]
    primary_config_path = _resolve_descriptor(root, primary["primary_config"], "primary_config")
    candidate_manifest_path = _resolve_descriptor(
        root, primary["candidate_manifest"], "candidate_manifest"
    )
    preprocessing_paths = [
        _resolve_descriptor(root, descriptor, "camera_preprocessing file")
        for descriptor in primary["camera_preprocessing"]["files"]
    ]
    candidate_manifest = _load_json(candidate_manifest_path, "candidate_manifest")
    _validate_candidate_manifest(candidate_manifest, primary)

    verifier = primary_runtime_verifier or _verify_primary_runtime
    verification = dict(
        verifier(
            project_root=root,
            config_path=primary_config_path,
            manifest_path=candidate_manifest_path,
            expected_manifest_sha256=primary["candidate_manifest"]["sha256"],
        )
    )
    if verification.get("candidate_cpu_load") != "passed":
        raise DeliveryContractError("fixed primary candidate did not load safely on CPU")
    if verification.get("runtime_device") != "cpu":
        raise DeliveryContractError("fixed primary runtime is not CPU")

    configured_source = (
        Path(source_root)
        if source_root is not None
        else Path(config["development_index"]["default_source_root"])
    )
    if not configured_source.is_absolute():
        configured_source = root / configured_source
    resolved_source = configured_source.resolve(strict=True)
    rows = _build_development_index(
        root=root,
        source_root=resolved_source,
        config=config["development_index"],
    )
    batch_counts = dict(sorted(Counter(row["batch_id"] for row in rows).items()))

    output_contract = config["output_contract"]
    contract_document = _resolve_repo_file(
        root, output_contract["document_path"], "output contract document"
    )
    schema_payloads: dict[str, bytes] = {}
    schema_ids: dict[str, str] = {}
    for name in REQUIRED_SCHEMA_NAMES:
        path = _resolve_repo_file(
            root, output_contract["schemas"][name], f"{name} schema"
        )
        schema = _load_json(path, f"{name} schema")
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise DeliveryContractError(f"{name} is not a JSON Schema 2020-12 document")
        schema_id = _nonempty_string(schema.get("$id"), f"{name} schema $id")
        if schema_id != REQUIRED_SCHEMA_IDS[name]:
            raise DeliveryContractError(f"{name} schema identity drifted")
        schema_ids[name] = schema_id
        schema_payloads[f"schemas/{path.name}"] = _canonical_json(schema, f"{name} schema")

    index_payload = _canonical_jsonl(rows, "development index")
    contract_payload = contract_document.read_bytes()
    baseline_payload = _baseline_verification_payload(
        config=config,
        verification=verification,
        development_record_count=len(rows),
        batch_counts=batch_counts,
        schema_ids=schema_ids,
    )
    artifact_payloads = {
        "development_index.jsonl": index_payload,
        "output_contract.md": contract_payload,
        "baseline_verification.txt": baseline_payload,
        **schema_payloads,
    }
    manifest = {
        "schema_version": DELIVERY_CONTRACT_MANIFEST_SCHEMA_VERSION,
        "contract_id": config["contract_id"],
        "status": "ready",
        "dataset_role": "development",
        "contract_config": _file_descriptor(config_file),
        "development_index": {
            "schema_version": DEVELOPMENT_INDEX_SCHEMA_VERSION,
            "record_count": len(rows),
            "batch_counts": batch_counts,
            "all_paths_resolved": True,
            "all_hashes_verified": True,
            "development_reuse_allowed": True,
        },
        "fixed_primary": {
            "candidate_id": primary["candidate_id"],
            "candidate_manifest_sha256": primary["candidate_manifest"]["sha256"],
            "model_state_sha256": primary["model_state_sha256"],
            "primary_config_sha256": primary["primary_config"]["sha256"],
            "input_point_count": primary["camera_preprocessing"]["input_point_count"],
            "model_feature_count": primary["camera_preprocessing"]["model_feature_count"],
            "shape_coordinate_count": primary["camera_preprocessing"][
                "shape_coordinate_count"
            ],
            "binary_decision_threshold": primary["camera_preprocessing"][
                "binary_decision_threshold"
            ],
            "camera_preprocessing_files": [
                _file_descriptor(path) for path in preprocessing_paths
            ],
        },
        "runtime": dict(config["runtime"]),
        "primary_runtime_verification": verification,
        "fresh_output_policy": {
            "fresh_output_root": output_contract["fresh_output_root"],
            "policy": output_contract["freshness_policy"],
            "overwrite_allowed": False,
            "atomic_commit": True,
        },
        "home_acceptance_input": dict(config["home_acceptance_input"]),
        "output_contract": {
            "statuses": list(UNIFIED_STATUSES),
            "required_files": list(REQUIRED_HANDOFF_FILES),
            "schemas": schema_ids,
            "algorithm_event_emitted": False,
            "backend_or_frontend_implemented": False,
        },
        "artifacts": {
            name: _payload_descriptor(payload)
            for name, payload in sorted(artifact_payloads.items())
        },
    }
    manifest_payload = _canonical_json(manifest, "delivery contract run manifest")
    final_payloads = {**artifact_payloads, "run_manifest.json": manifest_payload}
    _commit_new_directory(output, final_payloads)
    return DeliveryContractBuildResult(
        output_dir=output,
        development_record_count=len(rows),
        batch_counts=batch_counts,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )


def _build_development_index(
    *, root: Path, source_root: Path, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    batch_bindings = {row["batch_id"]: row for row in config["batches"]}
    records: dict[str, dict[str, Any]] = {}
    for artifact_set in config["artifact_sets"]:
        batch_id = artifact_set["batch_id"]
        binding = batch_bindings[batch_id]
        tracking_root = _resolve_repo_directory(
            root, artifact_set["tracking_root"], "tracking_root"
        )
        truth_root = _resolve_repo_directory(root, artifact_set["truth_root"], "truth_root")
        sidecars = sorted(tracking_root.glob("*/media_sidecar.json"))
        if not sidecars:
            raise DeliveryContractError(f"no media sidecars found under {tracking_root}")
        for sidecar_path in sidecars:
            sidecar = _load_json(sidecar_path, "media_sidecar")
            source_video_id = _token(sidecar.get("source_video_id"), "source_video_id")
            if source_video_id in records:
                raise DeliveryContractError(
                    f"duplicate development source_video_id: {source_video_id}"
                )
            if sidecar.get("schema_version") != "wandering-media-v1":
                raise DeliveryContractError(f"media sidecar schema drifted for {source_video_id}")
            media_ref = _safe_relative_path(sidecar.get("media_ref"), "media_ref")
            if not media_ref.parts or media_ref.parts[0].lower() != batch_id.lower():
                raise DeliveryContractError(
                    f"media_ref batch does not match {batch_id}: {media_ref.as_posix()}"
                )
            video_path = _resolve_external_file(source_root, media_ref, "source video")
            tracking_path = sidecar_path.parent / "tracking.jsonl"
            if not tracking_path.is_file():
                raise DeliveryContractError(f"tracking file is missing for {source_video_id}")
            tracking_sha = _sha256_file(tracking_path)
            if tracking_sha != sidecar.get("tracking_jsonl_sha256"):
                raise DeliveryContractError(f"tracking hash drifted for {source_video_id}")
            video_sha = _sha256_file(video_path)
            if video_sha != sidecar.get("source_sha256"):
                raise DeliveryContractError(f"source video hash drifted for {source_video_id}")
            truth_path = truth_root / source_video_id / "episode_truth.jsonl"
            truth_rows = _load_jsonl(truth_path, "episode truth")
            if not truth_rows or any(
                row.get("source_video_id") != source_video_id for row in truth_rows
            ):
                raise DeliveryContractError(
                    f"episode truth source binding drifted for {source_video_id}"
                )
            cvat_path = _resolve_external_file(
                source_root,
                PurePosixPath(batch_id, "per_video_xml", f"{video_path.stem}.xml"),
                "per-video CVAT XML",
            )
            sidecar_setup_id = _nonempty_string(sidecar.get("setup_id"), "setup_id")
            if sidecar_setup_id == binding["camera_setup_id"]:
                setup_binding_status = "consistent"
            elif sidecar_setup_id in binding["accepted_legacy_setup_ids"]:
                setup_binding_status = "owner_confirmed_legacy_alias"
            else:
                raise DeliveryContractError(
                    f"camera setup binding drifted for {source_video_id}: "
                    f"{sidecar_setup_id} != {binding['camera_setup_id']}"
                )
            records[source_video_id] = {
                "schema_version": DEVELOPMENT_INDEX_SCHEMA_VERSION,
                "dataset_role": config["dataset_role"],
                "development_reuse_allowed": True,
                "batch_id": batch_id,
                "historical_role": artifact_set["historical_role"],
                "participant_id": binding["participant_id"],
                "session_id": binding["session_id"],
                "camera_setup_id": binding["camera_setup_id"],
                "sidecar_setup_id": sidecar_setup_id,
                "setup_binding_status": setup_binding_status,
                "clock_domain_id": binding["clock_domain_id"],
                "source_video_id": source_video_id,
                "source_group_id": _nonempty_string(
                    sidecar.get("source_group_id"), "source_group_id"
                ),
                "device_id": _nonempty_string(sidecar.get("device_id"), "device_id"),
                "stream_epoch": _nonempty_string(
                    sidecar.get("stream_epoch"), "stream_epoch"
                ),
                "timezone": sidecar.get("timezone"),
                "artifacts": {
                    "video": _file_locator(video_path, kind="external_read_only"),
                    "tracking": _file_locator(tracking_path, kind="workspace_generated"),
                    "media_sidecar": _file_locator(
                        sidecar_path, kind="workspace_generated"
                    ),
                    "truth": _file_locator(truth_path, kind="workspace_generated"),
                    "cvat_xml": _file_locator(cvat_path, kind="external_read_only"),
                },
            }

    counts = Counter(record["batch_id"] for record in records.values())
    for binding in batch_bindings.values():
        expected = binding["expected_development_video_count"]
        if counts[binding["batch_id"]] != expected:
            raise DeliveryContractError(
                f"{binding['batch_id']} development count drifted: "
                f"expected {expected}, got {counts[binding['batch_id']]}"
            )
    for excluded in config["excluded_inputs"]:
        excluded_path = _resolve_external_file(
            source_root,
            PurePosixPath(excluded["batch_id"], excluded["filename"]),
            "excluded input",
        )
        if any(
            record["artifacts"]["video"]["path"] == excluded_path.as_posix()
            for record in records.values()
        ):
            raise DeliveryContractError(f"excluded input entered development index: {excluded_path}")
    return [records[key] for key in sorted(records)]


def _verify_primary_runtime(
    *,
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    import elderly_monitoring

    from elderly_monitoring.modules.mental_health.wandering.camera_primary_inference import (
        load_primary_camera_config,
        load_primary_camera_runtime,
    )

    config = load_primary_camera_config(config_path)
    runtime = load_primary_camera_runtime(
        config=config,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    devices = sorted({parameter.device.type for parameter in runtime.model.parameters()})
    module_path = Path(elderly_monitoring.__file__).resolve(strict=True)
    module_under_project = module_path.is_relative_to(project_root)
    if devices != ["cpu"] or runtime.model.training or not module_under_project:
        raise DeliveryContractError("fixed primary runtime or editable source binding drifted")
    return {
        "candidate_cpu_load": "passed",
        "runtime_device": "cpu",
        "parameter_devices": devices,
        "model_mode": "eval",
        "editable_module_path": module_path.as_posix(),
        "module_under_project_root": True,
    }


def _validate_development_config(value: Any) -> None:
    if not isinstance(value, dict):
        raise DeliveryContractError("development_index must be a mapping")
    _require_exact_fields(
        value,
        frozenset(
            {
                "schema_version",
                "dataset_role",
                "default_source_root",
                "batches",
                "artifact_sets",
                "excluded_inputs",
            }
        ),
        "development_index",
    )
    if value.get("schema_version") != DEVELOPMENT_INDEX_SCHEMA_VERSION:
        raise DeliveryContractError("development index schema drifted")
    if value.get("dataset_role") != "development":
        raise DeliveryContractError("B01+B02 must be assigned to development")
    _nonempty_string(value.get("default_source_root"), "default_source_root")
    batches = value.get("batches")
    if not isinstance(batches, list) or not batches:
        raise DeliveryContractError("development batches must be a non-empty list")
    batch_ids: set[str] = set()
    for row in batches:
        _require_exact_fields(row, _BATCH_FIELDS, "batch binding")
        batch_id = _token(row.get("batch_id"), "batch_id")
        if batch_id in batch_ids:
            raise DeliveryContractError(f"duplicate batch binding: {batch_id}")
        batch_ids.add(batch_id)
        for name in (
            "participant_id",
            "session_id",
            "camera_setup_id",
            "clock_domain_id",
        ):
            _token(row.get(name), name)
        expected = row.get("expected_development_video_count")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
            raise DeliveryContractError("expected_development_video_count must be positive")
        legacy_ids = row.get("accepted_legacy_setup_ids")
        if not isinstance(legacy_ids, list):
            raise DeliveryContractError("accepted_legacy_setup_ids must be a list")
        for legacy_id in legacy_ids:
            _token(legacy_id, "accepted legacy setup id")
        if row["camera_setup_id"] in legacy_ids or len(legacy_ids) != len(set(legacy_ids)):
            raise DeliveryContractError("accepted legacy setup ids are invalid")
    if batch_ids != {"B01", "B02"} and len(batch_ids) != 1:
        raise DeliveryContractError("production development batches must bind B01 and B02")
    artifact_sets = value.get("artifact_sets")
    if not isinstance(artifact_sets, list) or not artifact_sets:
        raise DeliveryContractError("artifact_sets must be a non-empty list")
    for row in artifact_sets:
        _require_exact_fields(row, _ARTIFACT_SET_FIELDS, "artifact set")
        if row.get("batch_id") not in batch_ids:
            raise DeliveryContractError("artifact set references an unknown batch")
        for name in ("tracking_root", "truth_root", "historical_role"):
            _nonempty_string(row.get(name), name)
    excluded = value.get("excluded_inputs")
    if not isinstance(excluded, list):
        raise DeliveryContractError("excluded_inputs must be a list")
    for row in excluded:
        _require_exact_fields(row, _EXCLUDED_FIELDS, "excluded input")
        if row.get("batch_id") not in batch_ids:
            raise DeliveryContractError("excluded input references an unknown batch")
        _nonempty_string(row.get("filename"), "excluded filename")
        _nonempty_string(row.get("reason"), "excluded reason")


def _validate_home_input(value: Any) -> None:
    _require_exact_fields(value, frozenset({"status", "role"}), "home_acceptance_input")
    if value.get("status") not in {"awaiting_input", "registered"}:
        raise DeliveryContractError("home input status is invalid")
    if value.get("role") != "real_scene_acceptance_then_development_if_tuned":
        raise DeliveryContractError("home input role drifted")


def _validate_fixed_primary(value: Any) -> None:
    _require_exact_fields(
        value,
        frozenset(
            {
                "candidate_id",
                "candidate_manifest",
                "model_state_sha256",
                "primary_config",
                "camera_preprocessing",
            }
        ),
        "fixed_primary",
    )
    _token(value.get("candidate_id"), "candidate_id")
    _validate_descriptor(value.get("candidate_manifest"), "candidate_manifest")
    _validate_digest(value.get("model_state_sha256"), "model_state_sha256")
    _validate_descriptor(value.get("primary_config"), "primary_config")
    preprocessing = value.get("camera_preprocessing")
    _require_exact_fields(
        preprocessing,
        frozenset(
            {
                "input_point_count",
                "model_feature_count",
                "shape_coordinate_count",
                "binary_decision_threshold",
                "files",
            }
        ),
        "camera_preprocessing",
    )
    if preprocessing.get("input_point_count") != 80:
        raise DeliveryContractError("camera input point count must remain 80")
    if preprocessing.get("model_feature_count") != 14:
        raise DeliveryContractError("camera model feature count must remain 14")
    if preprocessing.get("shape_coordinate_count") != 2:
        raise DeliveryContractError("camera shape coordinate count must remain 2")
    if preprocessing.get("binary_decision_threshold") != 0.5:
        raise DeliveryContractError("camera binary threshold must remain 0.5")
    files = preprocessing.get("files")
    if not isinstance(files, list) or len(files) < 3:
        raise DeliveryContractError("camera preprocessing identity is incomplete")
    for descriptor in files:
        _validate_descriptor(descriptor, "camera preprocessing file")


def _validate_runtime(value: Any) -> None:
    expected = {
        "device": "cpu",
        "intra_op_threads": 8,
        "inter_op_threads": 1,
        "maximum_batch_size": 64,
    }
    if value != expected:
        raise DeliveryContractError("fixed primary runtime identity drifted")


def _validate_output_contract(value: Any) -> None:
    _require_exact_fields(
        value,
        frozenset(
            {
                "document_path",
                "fresh_output_root",
                "freshness_policy",
                "statuses",
                "required_files",
                "schemas",
            }
        ),
        "output_contract",
    )
    _nonempty_string(value.get("document_path"), "document_path")
    _nonempty_string(value.get("fresh_output_root"), "fresh_output_root")
    if value.get("freshness_policy") != "final_output_directory_must_not_exist":
        raise DeliveryContractError("fresh output policy drifted")
    if value.get("statuses") != list(UNIFIED_STATUSES):
        raise DeliveryContractError("unified output statuses drifted")
    if value.get("required_files") != list(REQUIRED_HANDOFF_FILES):
        raise DeliveryContractError("fixed handoff filenames drifted")
    schemas = value.get("schemas")
    if not isinstance(schemas, dict) or tuple(schemas) != REQUIRED_SCHEMA_NAMES:
        raise DeliveryContractError("output schema set or order drifted")
    for name, path in schemas.items():
        _nonempty_string(path, f"{name} schema path")


def _validate_candidate_manifest(
    manifest: Mapping[str, Any], primary: Mapping[str, Any]
) -> None:
    if manifest.get("candidate_id") != primary["candidate_id"]:
        raise DeliveryContractError("candidate_id drifted")
    model_state = manifest.get("artifacts", {}).get("model_state", {})
    if model_state.get("sha256") != primary["model_state_sha256"]:
        raise DeliveryContractError("candidate model_state hash drifted")
    inference = manifest.get("inference_contract", {})
    expected_shapes = {
        "model_features": [80, 14],
        "point_mask": [80],
        "shape_normalized_points": [80, 2],
    }
    if inference.get("input_shapes") != expected_shapes:
        raise DeliveryContractError("candidate input shapes drifted")
    if inference.get("binary_decision") != "sigmoid >= 0.5":
        raise DeliveryContractError("candidate binary decision drifted")


def _baseline_verification_payload(
    *,
    config: Mapping[str, Any],
    verification: Mapping[str, Any],
    development_record_count: int,
    batch_counts: Mapping[str, int],
    schema_ids: Mapping[str, str],
) -> bytes:
    primary = config["fixed_primary"]
    lines = [
        f"contract_id={config['contract_id']}",
        f"candidate_id={primary['candidate_id']}",
        f"candidate_cpu_load={verification['candidate_cpu_load']}",
        f"runtime_device={verification['runtime_device']}",
        f"model_mode={verification.get('model_mode', 'unknown')}",
        f"input_point_count={primary['camera_preprocessing']['input_point_count']}",
        f"binary_decision_threshold={primary['camera_preprocessing']['binary_decision_threshold']}",
        f"development_record_count={development_record_count}",
        "batch_counts=" + ",".join(f"{key}:{batch_counts[key]}" for key in sorted(batch_counts)),
        f"schema_count={len(schema_ids)}",
        "all_input_paths_resolved=passed",
        "all_input_hashes_verified=passed",
        "fresh_output_policy=passed",
        "non_overwrite_policy=passed",
    ]
    if "editable_module_path" in verification:
        lines.insert(1, f"editable_module_path={verification['editable_module_path']}")
        lines.insert(
            2,
            "module_under_project_root="
            + str(bool(verification.get("module_under_project_root"))).lower(),
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _commit_new_directory(output: Path, files: Mapping[str, bytes]) -> None:
    if output.exists():
        raise FileExistsError(f"delivery contract output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in sorted(files.items()):
            target = staging / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for relative, payload in files.items():
            target = staging / PurePosixPath(relative)
            if target.read_bytes() != payload:
                raise DeliveryContractError(f"staging verification failed: {relative}")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _resolve_descriptor(root: Path, value: Mapping[str, Any], role: str) -> Path:
    _validate_descriptor(value, role)
    path = _resolve_repo_file(root, value["path"], role)
    if _sha256_file(path) != value["sha256"]:
        raise DeliveryContractError(f"{role} hash drifted")
    return path


def _resolve_repo_file(root: Path, value: Any, role: str) -> Path:
    relative = _safe_relative_path(value, role)
    path = (root / relative).resolve(strict=True)
    _require_under(root, path, role)
    if not path.is_file():
        raise DeliveryContractError(f"{role} must be a file")
    return path


def _resolve_repo_directory(root: Path, value: Any, role: str) -> Path:
    relative = _safe_relative_path(value, role)
    path = (root / relative).resolve(strict=True)
    _require_under(root, path, role)
    if not path.is_dir():
        raise DeliveryContractError(f"{role} must be a directory")
    return path


def _resolve_external_file(root: Path, relative: PurePosixPath, role: str) -> Path:
    path = (root / relative).resolve(strict=True)
    _require_under(root, path, role)
    if not path.is_file():
        raise DeliveryContractError(f"{role} must be a file")
    return path


def _safe_relative_path(value: Any, role: str) -> PurePosixPath:
    text = _nonempty_string(value, role).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DeliveryContractError(f"{role} must be a safe relative path")
    return path


def _require_under(root: Path, path: Path, role: str) -> None:
    if not path.is_relative_to(root):
        raise DeliveryContractError(f"{role} escapes its configured root")


def _file_locator(path: Path, *, kind: str) -> dict[str, Any]:
    descriptor = _file_descriptor(path)
    return {"path_kind": kind, **descriptor}


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "byte_count": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _payload_descriptor(payload: bytes) -> dict[str, Any]:
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _validate_descriptor(value: Any, role: str) -> None:
    _require_exact_fields(value, _DESCRIPTOR_FIELDS, role)
    _nonempty_string(value.get("path"), f"{role} path")
    _validate_digest(value.get("sha256"), f"{role} sha256")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeliveryContractError(f"cannot read {role}") from exc
    if not isinstance(value, dict):
        raise DeliveryContractError(f"{role} must contain a JSON object")
    return value


def _load_jsonl(path: Path, role: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DeliveryContractError(f"{role} file is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict):
                raise DeliveryContractError(f"{role} rows must be objects")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeliveryContractError(f"cannot read {role}") from exc
    return rows


def _canonical_json(value: Any, role: str) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise DeliveryContractError(f"{role} is not finite JSON") from exc


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]], role: str) -> bytes:
    return b"".join(_canonical_json(dict(row), role) for row in rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_fields(value: Any, fields: frozenset[str], role: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise DeliveryContractError(f"{role} fields drifted")


def _token(value: Any, role: str) -> str:
    text = _nonempty_string(value, role)
    if _TOKEN.fullmatch(text) is None:
        raise DeliveryContractError(f"{role} must be a stable token")
    return text


def _nonempty_string(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeliveryContractError(f"{role} must be a non-empty string")
    return value


def _validate_digest(value: Any, role: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DeliveryContractError(f"{role} must be lowercase SHA-256")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "DELIVERY_CONTRACT_CONFIG_SCHEMA_VERSION",
    "DELIVERY_CONTRACT_MANIFEST_SCHEMA_VERSION",
    "DEVELOPMENT_INDEX_SCHEMA_VERSION",
    "REQUIRED_SCHEMA_IDS",
    "DeliveryContractBuildResult",
    "DeliveryContractError",
    "build_wandering_camera_delivery_contract",
    "load_delivery_contract_config",
]
