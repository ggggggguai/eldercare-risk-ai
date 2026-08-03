"""Independently verify DATA-001 artefacts against the current source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from elderly_monitoring.modules.mental_health.mood_social import (
    feature_schema_manifest,
)


EXPECTED_DATASETS = {
    "psyche_d",
    "resilient",
    "nhanes",
    "shenzhen_elderly",
    "hefei_elderly",
}
DATASET_DIRECTORIES = {
    "psyche_d": "PSYCHE-D",
    "resilient": "RESILIENT",
    "nhanes": "NHANES",
    "shenzhen_elderly": "DRYAD深证社区老年心理健康",
    "hefei_elderly": "合肥社区老年衰弱数据",
}
DEFAULT_OUTPUT_RELATIVE = Path(
    "data/processed/mental_health/mood_social/v3.3.3/manifests"
)
RESILIENT_SENSOR_FILES = {
    "ScanWatch_HR.csv",
    "ScanWatch_Steps.csv",
    "Sleep_physio.csv",
    "Sleep_state.csv",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _collection_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["relative_path"].encode("utf-8")):
        digest.update(
            (f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "数据集" / "心理").is_dir() and (
            candidate / "algorithm" / "eldercare-risk-ai-main"
        ).is_dir():
            return candidate
    raise RuntimeError("workspace root was not found")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_file(path: Path, source_root: Path) -> tuple[str, str]:
    relative_parts = path.relative_to(source_root).parts
    if path.name == ".DS_Store":
        return "os_metadata", "ds_store"
    if "__MACOSX" in relative_parts:
        return "os_metadata", "appledouble"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        role = (
            "source_metadata"
            if "Resilient_metadata" in relative_parts
            else "source_data"
        )
        return role, "csv"
    known = {
        ".md": ("source_documentation", "markdown"),
        ".tsv": ("source_metadata", "tsv"),
        ".xls": ("source_data", "xls"),
        ".xpt": ("source_data", "sas_xport"),
        ".yaml": ("source_metadata", "yaml"),
        ".yml": ("source_metadata", "yaml"),
    }
    if suffix in known:
        return known[suffix]
    with path.open("rb") as handle:
        prefix = handle.read(4)
        if path.stat().st_size >= 4:
            handle.seek(-4, os.SEEK_END)
            suffix_magic = handle.read(4)
        else:
            suffix_magic = b""
    if prefix == b"PAR1" and suffix_magic == b"PAR1":
        return "source_data", "parquet"
    raise RuntimeError("validator found an unsupported source file format")


def _resilient_aliases(source_root: Path) -> dict[str, str]:
    names = sorted(
        {
            path.parent.name
            for path in source_root.rglob("*.csv")
            if path.name in RESILIENT_SENSOR_FILES
            and "__MACOSX" not in path.relative_to(source_root).parts
        },
        key=lambda value: value.encode("utf-8"),
    )
    return {name: f"participant_{index:04d}" for index, name in enumerate(names, 1)}


def _public_relative_path(
    workspace_root: Path,
    dataset_id: str,
    source_root: Path,
    source_path: Path,
    aliases: dict[str, str],
) -> str:
    parts = list(source_path.relative_to(source_root).parts)
    if dataset_id == "resilient":
        for index, part in enumerate(parts):
            prefix = "._" if part.startswith("._") else ""
            candidate = part[2:] if prefix else part
            if candidate in aliases:
                parts[index] = f"{prefix}{aliases[candidate]}"
    logical_root = workspace_root / "数据集" / "心理" / DATASET_DIRECTORIES[dataset_id]
    return (logical_root / Path(*parts)).relative_to(workspace_root).as_posix()


def validate(workspace_root: Path, output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "dataset_audit_summary.json"
    validation_path = output_root / "validation_checks.json"
    manifest_path = output_root / "dataset_file_manifest.jsonl"
    schema_path = output_root / "feature_schema_manifest.json"
    summary = _load_json(summary_path)
    reported_validation = _load_json(validation_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    schema = _load_json(schema_path)
    if not rows:
        raise RuntimeError("file manifest is empty")
    if {row["dataset_id"] for row in rows} != EXPECTED_DATASETS:
        raise RuntimeError("file manifest dataset set does not match DATA-001")
    expected_keys = {
        "bytes",
        "dataset_id",
        "file_role",
        "format",
        "relative_path",
        "sha256",
    }
    for row in rows:
        if set(row) != expected_keys:
            raise RuntimeError("file manifest contains unexpected keys")
        relative = str(row["relative_path"])
        if "\\" in relative or PurePosixPath(relative).is_absolute():
            raise RuntimeError("file manifest path is not relative POSIX")
        relative_path = PurePosixPath(relative)
        if ".." in relative_path.parts or relative_path.parts[:2] != ("数据集", "心理"):
            raise RuntimeError("manifest source path is outside 数据集/心理")
    for dataset_id in EXPECTED_DATASETS:
        dataset_rows = [row for row in rows if row["dataset_id"] == dataset_id]
        source_root = (
            workspace_root / "数据集" / "心理" / DATASET_DIRECTORIES[dataset_id]
        )
        source_files = sorted(
            (path for path in source_root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(workspace_root)
            .as_posix()
            .encode("utf-8"),
        )
        aliases = _resilient_aliases(source_root) if dataset_id == "resilient" else {}
        source_records: list[dict[str, Any]] = []
        source_collection_rows: list[dict[str, Any]] = []
        for path in source_files:
            before = path.stat()
            file_sha256 = _sha256_file(path)
            after = path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise RuntimeError("source file changed during validation")
            file_role, file_format = _classify_file(path, source_root)
            source_records.append(
                {
                    "bytes": int(after.st_size),
                    "dataset_id": dataset_id,
                    "file_role": file_role,
                    "format": file_format,
                    "relative_path": _public_relative_path(
                        workspace_root,
                        dataset_id,
                        source_root,
                        path,
                        aliases,
                    ),
                    "sha256": file_sha256,
                }
            )
            source_collection_rows.append(
                {
                    "bytes": int(after.st_size),
                    "relative_path": path.relative_to(workspace_root).as_posix(),
                    "sha256": file_sha256,
                }
            )
        expected_file_rows = [
            {
                "bytes": int(row["bytes"]),
                "dataset_id": str(row["dataset_id"]),
                "file_role": str(row["file_role"]),
                "format": str(row["format"]),
                "relative_path": str(row["relative_path"]),
                "sha256": str(row["sha256"]),
            }
            for row in dataset_rows
        ]

        def row_sort_key(row: dict[str, Any]) -> bytes:
            return str(row["relative_path"]).encode("utf-8")

        if sorted(source_records, key=row_sort_key) != sorted(
            expected_file_rows, key=row_sort_key
        ):
            raise RuntimeError("source file path/hash bindings do not match manifest")
        collection = summary["datasets"][dataset_id]["collection"]
        if collection["file_count"] != len(dataset_rows):
            raise RuntimeError("collection file count does not match manifest")
        if collection["total_bytes"] != sum(row["bytes"] for row in dataset_rows):
            raise RuntimeError("collection byte count does not match manifest")
        if collection["collection_sha256"] != _collection_sha256(
            source_collection_rows
        ):
            raise RuntimeError("collection SHA-256 does not match manifest")
        if dataset_id == "resilient":
            participant_names = {
                path.parent.name
                for path in source_root.rglob("*.csv")
                if path.name in RESILIENT_SENSOR_FILES
                and "__MACOSX" not in path.relative_to(source_root).parts
            }
            public_parts = {
                part
                for row in dataset_rows
                for part in PurePosixPath(str(row["relative_path"])).parts
            }
            if participant_names & public_parts:
                raise RuntimeError("RESILIENT manifest leaks participant path segments")
    current_schema = feature_schema_manifest()
    if schema != current_schema:
        raise RuntimeError("feature schema snapshot is not the current MH-003 manifest")
    artifact_hashes = reported_validation["artifact_sha256"]
    for name in (
        "dataset_file_manifest.jsonl",
        "dataset_audit_summary.json",
        "feature_schema_manifest.json",
    ):
        if _sha256_file(output_root / name) != artifact_hashes[name]:
            raise RuntimeError("reported artefact SHA-256 does not match")
    if (
        summary["feature_schema_target"]["sha256"]
        != hashlib.sha256(_canonical_json_bytes(current_schema)).hexdigest()
    ):
        raise RuntimeError("feature schema SHA-256 does not match current MH-003")
    if summary["scope"]["total_files"] != len(rows):
        raise RuntimeError("summary total file count does not match manifest")
    if not reported_validation["audit_complete"]:
        raise RuntimeError("reported audit is incomplete")
    return {
        "dataset_count": len(EXPECTED_DATASETS),
        "file_count": len(rows),
        "open_issue_count": len(summary["issues"]),
        "status": "pass",
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DATA-001 audit artefacts")
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = _find_workspace_root(args.workspace_root or Path.cwd())
    algorithm = (workspace / "algorithm" / "eldercare-risk-ai-main").resolve()
    output = (
        args.output_root.resolve()
        if args.output_root is not None
        else (algorithm / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    if not output.is_relative_to(algorithm):
        print(json.dumps({"status": "failed", "error": "output outside algorithm"}))
        return 2
    try:
        result = validate(workspace, output)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
