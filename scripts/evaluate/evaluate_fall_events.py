from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any, Sequence

import elderly_monitoring
import elderly_monitoring.modules.fall_risk.evaluation as fall_risk_evaluation_module
import yaml

from elderly_monitoring.modules.fall_risk.evaluation import (
    PROVISIONAL_PROTOCOL_STATUSES,
    evaluate_event_predictions,
    write_evaluation_bundle,
)


@dataclass(frozen=True)
class _InputSnapshot:
    value: Any
    sha256: str


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError("YAML mapping keys must be hashable") from exc
        if duplicate:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        evaluation_implementation_sha256 = _evaluation_implementation_sha256()
        config_snapshot = _load_yaml_mapping_snapshot(args.config)
        config = config_snapshot.value
        protocol_status = str(config.get("protocol_status", ""))
        if protocol_status in PROVISIONAL_PROTOCOL_STATUSES and not args.allow_provisional:
            raise ValueError(
                "development/provisional protocol requires --allow-provisional"
            )
        if protocol_status not in PROVISIONAL_PROTOCOL_STATUSES and protocol_status != "frozen":
            raise ValueError(f"unsupported protocol_status: {protocol_status!r}")
        git_state = _git_state()
        if protocol_status == "frozen" and git_state["dirty"]:
            raise ValueError("frozen evaluation requires a clean Git worktree")

        split_snapshot = _load_json_object_snapshot(args.split)
        prediction_snapshot = _load_jsonl_snapshot(args.predictions)
        assignments_snapshot = _load_jsonl_snapshot(args.assignments)
        ground_truth_snapshot = _load_jsonl_snapshot(args.ground_truth)
        manifest_snapshot = _load_jsonl_snapshot(args.manifest)
        split_metadata = split_snapshot.value
        predictions = prediction_snapshot.value
        assignments = assignments_snapshot.value
        all_ground_truth = ground_truth_snapshot.value
        all_manifest_rows = manifest_snapshot.value
        split_id = _validate_split_metadata(split_metadata, config)
        validation_report_snapshot = (
            _load_json_object_snapshot(args.validation_report)
            if args.validation_report is not None
            else None
        )
        validation_report_hash = _validate_validation_report_binding(
            protocol_status=protocol_status,
            split_metadata=split_metadata,
            validation_report_snapshot=validation_report_snapshot,
        )
        test_release_metadata = _validate_test_partition_access(
            partition=args.partition,
            protocol_status=protocol_status,
            protocol_version=str(config.get("protocol_version", "")),
            split_metadata=split_metadata,
            acknowledgement_path=args.test_release_ack,
            evaluation_config_sha256=config_snapshot.sha256,
            prediction_sha256=prediction_snapshot.sha256,
            labels_sha256=str(split_metadata.get("labels_sha256", "")),
            code_commit=str(git_state["commit"]),
        )
        expected_assignments_hash = str(split_metadata.get("assignments_sha256", ""))
        if not expected_assignments_hash:
            raise ValueError("split metadata is missing assignments_sha256")
        if assignments_snapshot.sha256 != expected_assignments_hash:
            raise ValueError("assignments SHA-256 does not match split metadata")
        _validate_split_root(split_metadata, assignments)
        selected_video_ids = _partition_video_ids(
            assignments,
            str(split_id),
            args.partition,
            str(config.get("task_type", "")),
        )
        if not selected_video_ids:
            raise ValueError(f"partition {args.partition!r} has no assigned videos")

        ground_truth_types = set(config.get("ground_truth_event_types", ()))
        task_ground_truth = [
            row
            for row in all_ground_truth
            if str(row.get("event_type")) in ground_truth_types
        ]
        expected_labels_hash = str(split_metadata.get("labels_sha256", ""))
        if not expected_labels_hash:
            raise ValueError("split metadata is missing labels_sha256")
        if _canonical_rows_sha256(task_ground_truth) != expected_labels_hash:
            raise ValueError("ground-truth labels SHA-256 does not match split metadata")
        ground_truth = [
            row
            for row in task_ground_truth
            if str(row.get("video_id")) in selected_video_ids
        ]
        outside_predictions = sorted(
            {
                str(row.get("video_id"))
                for row in predictions
                if str(row.get("video_id")) not in selected_video_ids
            }
        )
        if outside_predictions:
            raise ValueError(
                "predictions contain videos outside the selected split partition"
            )
        for row in ground_truth:
            if row.get("split_id") not in (None, "", split_id):
                raise ValueError("ground-truth split_id does not match split metadata")
            row["split_id"] = split_id
            row.setdefault("task_type", config["task_type"])
        for row in predictions:
            if str(row.get("split_id")) != str(split_id):
                raise ValueError("prediction split_id does not match split metadata")

        expected_manifest_hash = str(split_metadata.get("manifest_sha256", ""))
        if not expected_manifest_hash:
            raise ValueError("split metadata is missing manifest_sha256")
        if manifest_snapshot.sha256 != expected_manifest_hash:
            raise ValueError("manifest SHA-256 does not match split metadata")
        expected_canonical_manifest_hash = str(
            split_metadata.get("manifest_canonical_sha256", "")
        )
        if (
            not expected_canonical_manifest_hash
            or _canonical_rows_sha256(all_manifest_rows)
            != expected_canonical_manifest_hash
        ):
            raise ValueError("canonical manifest SHA-256 does not match split metadata")
        manifest = [
            row
            for row in all_manifest_rows
            if str(row.get("video_id", row.get("asset_id"))) in selected_video_ids
        ]
        result = evaluate_event_predictions(
            ground_truth,
            predictions,
            manifest=manifest,
            config=config,
        )
        reproduction_command = shlex.join(
            [
                "conda",
                "run",
                "-n",
                "eldercare-ai",
                "python",
                "scripts/evaluate/evaluate_fall_events.py",
                *(argv or sys.argv[1:]),
            ]
        )
        metadata = {
            "code_version": git_state["commit"],
            "git_dirty": git_state["dirty"],
            "tracked_diff_sha256": git_state["tracked_diff_sha256"],
            "evaluation_implementation_sha256": evaluation_implementation_sha256,
            "python_version": platform.python_version(),
            **_environment_fingerprint(),
            "manifest_hash": manifest_snapshot.sha256,
            "split_id": split_id,
            "split_sha256": split_metadata.get("split_sha256"),
            "label_version": args.label_version,
            "ground_truth_hash": ground_truth_snapshot.sha256,
            "prediction_hash": prediction_snapshot.sha256,
            "evaluation_config_file_sha256": config_snapshot.sha256,
            "evaluation_config_normalized_sha256": result.config_hash,
            "validation_report_hash": validation_report_hash,
            "partition": args.partition,
            "reproduction_command": reproduction_command,
            **test_release_metadata,
        }
        write_evaluation_bundle(
            result,
            args.output_dir,
            metadata=metadata,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fall-risk event predictions with a versioned protocol."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help=(
            "Validation report bound by split metadata; required for frozen "
            "evaluation."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-version", required=True)
    parser.add_argument(
        "--test-release-ack",
        type=Path,
        default=None,
        help="Required governance acknowledgement for a frozen test partition run.",
    )
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Permit a development/provisional protocol for non-formal smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluation bundle; disabled by default.",
    )
    return parser


def _read_snapshot_bytes(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _decode_snapshot(raw: bytes, path: Path) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"input is not valid UTF-8: {path}") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _parse_json(text: str, *, path: Path, line_no: int | None = None) -> Any:
    location = str(path) if line_no is None else f"{path}:{line_no}"
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {location}: {exc.msg}") from exc


def _load_json_object_snapshot(path: Path) -> _InputSnapshot:
    raw, sha256 = _read_snapshot_bytes(path)
    value = _parse_json(_decode_snapshot(raw, path), path=path)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return _InputSnapshot(value=value, sha256=sha256)


def _load_jsonl_snapshot(path: Path) -> _InputSnapshot:
    raw, sha256 = _read_snapshot_bytes(path)
    text = _decode_snapshot(raw, path)
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        value = _parse_json(stripped, path=path, line_no=line_no)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_no}")
        rows.append(value)
    return _InputSnapshot(value=rows, sha256=sha256)


def _load_yaml_mapping_snapshot(path: Path) -> _InputSnapshot:
    raw, sha256 = _read_snapshot_bytes(path)
    text = _decode_snapshot(raw, path)
    try:
        value = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML at {path}: {exc}") from exc
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return _InputSnapshot(value=value, sha256=sha256)


def _evaluation_implementation_sha256() -> str:
    repo_root = Path.cwd().resolve()
    core_module_file = getattr(fall_risk_evaluation_module, "__file__", None)
    if core_module_file is None:
        raise ValueError("fall-risk evaluation module has no source file")
    implementation_files = {
        "scripts/evaluate/evaluate_fall_events.py": Path(__file__).resolve(),
        "src/elderly_monitoring/modules/fall_risk/evaluation.py": Path(
            core_module_file
        ).resolve(),
    }
    contents: dict[str, bytes] = {}
    for relative_path, source_path in implementation_files.items():
        expected_path = (repo_root / relative_path).resolve()
        if source_path != expected_path:
            raise ValueError(
                "evaluation implementation is not bound to the current repository: "
                f"{relative_path}"
            )
        contents[relative_path] = source_path.read_bytes()
    return _implementation_files_sha256(contents)


def _implementation_files_sha256(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(files):
        path_bytes = relative_path.encode("utf-8")
        content = files[relative_path]
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _partition_video_ids(
    assignments: Sequence[dict[str, Any]],
    split_id: str,
    partition: str,
    task_type: str,
) -> set[str]:
    video_ids: set[str] = set()
    for row in assignments:
        if str(row.get("split_id")) != split_id:
            raise ValueError("assignment split_id does not match split metadata")
        if str(row.get("task_type")) != task_type:
            raise ValueError("assignment task_type does not match evaluation config")
        if row.get("partition") != partition:
            continue
        video_id = row.get("video_id", row.get("asset_id"))
        if not video_id:
            raise ValueError("split assignment is missing video_id/asset_id")
        video_ids.add(str(video_id))
    return video_ids


def _canonical_rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    rendered = sorted(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        for row in rows
    )
    digest = hashlib.sha256()
    for row in rendered:
        digest.update(row)
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_split_metadata(
    split_metadata: dict[str, Any], config: dict[str, Any]
) -> str:
    status = str(split_metadata.get("status", ""))
    if status not in {"ready", "frozen"}:
        raise ValueError("evaluation requires a ready or frozen split")
    split_name = str(split_metadata.get("split_name", ""))
    split_sha256 = str(split_metadata.get("split_sha256", ""))
    split_id = str(split_metadata.get("split_id", ""))
    if not split_name or not _is_sha256(split_sha256):
        raise ValueError("split metadata has an invalid split_name or split_sha256")
    expected_split_id = f"{split_name}:sha256:{split_sha256}"
    if split_id != expected_split_id:
        raise ValueError("split_id does not match split_name and split_sha256")
    if str(split_metadata.get("task_type")) != str(config.get("task_type")):
        raise ValueError("split task_type does not match evaluation config")
    if str(config.get("protocol_status")) == "frozen":
        if status != "frozen":
            raise ValueError("frozen evaluation requires a frozen split")
        if str(split_metadata.get("protocol_status", "")) != "frozen":
            raise ValueError(
                "frozen evaluation requires split protocol_status=frozen"
            )
        if not _is_sha256(split_metadata.get("validation_report_sha256")):
            raise ValueError(
                "frozen evaluation requires a valid validation_report_sha256"
            )
    return split_id


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _validate_validation_report_binding(
    *,
    protocol_status: str,
    split_metadata: dict[str, Any],
    validation_report_snapshot: _InputSnapshot | None,
) -> str | None:
    if validation_report_snapshot is None:
        if protocol_status == "frozen":
            raise ValueError("frozen evaluation requires --validation-report")
        return None
    expected_sha256 = split_metadata.get("validation_report_sha256")
    if not _is_sha256(expected_sha256):
        raise ValueError(
            "split metadata does not contain a valid validation_report_sha256"
        )
    if validation_report_snapshot.sha256 != expected_sha256:
        raise ValueError(
            "validation report SHA-256 does not match split metadata"
        )
    return validation_report_snapshot.sha256


def _validate_split_root(
    split_metadata: dict[str, Any], assignments: Sequence[dict[str, Any]]
) -> None:
    required = {
        "schema_version",
        "split_name",
        "task_type",
        "protocol_status",
        "status",
        "manifest_sha256",
        "manifest_canonical_sha256",
        "validation_report_sha256",
        "labels_sha256",
        "config_sha256",
        "split_sha256",
    }
    missing = sorted(required - set(split_metadata))
    if missing:
        raise ValueError(f"split metadata is missing root fields: {missing}")
    base_assignments = []
    for row in assignments:
        base = dict(row)
        base.pop("split_id", None)
        base_assignments.append(base)
    base_assignments.sort(
        key=lambda row: (
            ("train", "validation", "test").index(str(row["partition"])),
            str(row.get("sample_id", "")),
        )
    )
    payload = {
        "schema_version": split_metadata["schema_version"],
        "split_name": split_metadata["split_name"],
        "task_type": split_metadata["task_type"],
        "protocol_status": split_metadata["protocol_status"],
        "status": split_metadata["status"],
        "manifest_sha256": split_metadata["manifest_sha256"],
        "manifest_canonical_sha256": split_metadata["manifest_canonical_sha256"],
        "validation_report_sha256": split_metadata["validation_report_sha256"],
        "labels_sha256": split_metadata["labels_sha256"],
        "config_sha256": split_metadata["config_sha256"],
        "assignments": base_assignments,
    }
    actual = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if actual != split_metadata["split_sha256"]:
        raise ValueError("split root SHA-256 does not match metadata and assignments")


def _validate_test_partition_access(
    *,
    partition: str,
    protocol_status: str,
    protocol_version: str,
    split_metadata: dict[str, Any],
    acknowledgement_path: Path | None,
    evaluation_config_sha256: str,
    prediction_sha256: str,
    labels_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    if partition != "test":
        return {}
    if (
        protocol_status != "frozen"
        or split_metadata.get("status") != "frozen"
        or split_metadata.get("protocol_status") != "frozen"
    ):
        raise ValueError("test partition requires both frozen protocol and frozen split")
    if acknowledgement_path is None:
        raise ValueError("test partition requires --test-release-ack")
    acknowledgement_snapshot = _load_json_object_snapshot(acknowledgement_path)
    acknowledgement = acknowledgement_snapshot.value
    expected = {
        "partition": "test",
        "split_id": split_metadata["split_id"],
        "protocol_version": protocol_version,
        "blind_test_governance_confirmed": True,
        "evaluation_config_sha256": evaluation_config_sha256,
        "prediction_sha256": prediction_sha256,
        "labels_sha256": labels_sha256,
        "code_commit": code_commit,
    }
    for field, value in expected.items():
        if acknowledgement.get(field) != value:
            raise ValueError(f"test release acknowledgement mismatch: {field}")
    if not str(acknowledgement.get("authorization_id", "")).strip():
        raise ValueError("test release acknowledgement requires authorization_id")
    if not str(acknowledgement.get("authorized_by_role", "")).strip():
        raise ValueError("test release acknowledgement requires authorized_by_role")
    if not str(acknowledgement.get("evaluation_run_id", "")).strip():
        raise ValueError("test release acknowledgement requires evaluation_run_id")
    return {
        "test_release_ack_sha256": acknowledgement_snapshot.sha256,
        "test_authorization_id": acknowledgement["authorization_id"],
        "test_authorized_by_role": acknowledgement["authorized_by_role"],
        "test_evaluation_run_id": acknowledgement["evaluation_run_id"],
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True, "tracked_diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if status else None,
    }


def _environment_fingerprint() -> dict[str, Any]:
    environment_name = os.environ.get("CONDA_DEFAULT_ENV")
    if environment_name != "eldercare-ai":
        raise ValueError(
            "evaluation must run inside the eldercare-ai conda environment"
        )
    repo_root = Path.cwd().resolve()
    package_path = Path(elderly_monitoring.__file__).resolve()
    expected_package_root = (repo_root / "src/elderly_monitoring").resolve()
    if package_path != expected_package_root / "__init__.py":
        raise ValueError(
            "elderly_monitoring import is not bound to the current repository editable source"
        )
    packages = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib_metadata.distributions()
    )
    payload = "\n".join(packages).encode("utf-8")
    return {
        "environment_name": environment_name,
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "package_source": package_path.relative_to(repo_root).as_posix(),
        "environment_package_count": len(packages),
        "environment_packages_sha256": hashlib.sha256(payload).hexdigest(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
