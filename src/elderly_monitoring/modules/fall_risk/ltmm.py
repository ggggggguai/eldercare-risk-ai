from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CLINICAL_TABLES = {
    "ClinicalDemogData_COFL.xlsx": "clinical_demographic_table",
    "ReportHome75h.xlsx": "home_monitoring_report",
}


def build_ltmm_manifest(raw_dir: str | Path) -> list[dict[str, Any]]:
    """Build a manifest for locally downloaded LTMM files."""
    root = Path(raw_dir)
    records = _read_records(root / "RECORDS")
    checksum_map = _read_sha256sums(root / "SHA256SUMS.txt")
    rows: list[dict[str, Any]] = []

    for filename, table_type in CLINICAL_TABLES.items():
        path = root / filename
        rows.append(
            {
                "dataset": "ltmm",
                "record_id": path.stem,
                "path": _posix(path),
                "record_type": table_type,
                "subject_id": "unknown",
                "group": "mixed",
                "label_source": "clinical_table",
                "modality": "clinical_table",
                "available": path.exists(),
                "sha256": checksum_map.get(filename),
            }
        )

    record_names = records or _discover_record_names(root)
    for record_name in record_names:
        group = _ltmm_group(record_name)
        record_type = "lab_walk_accelerometer" if record_name.startswith("LabWalks/") else (
            "home_accelerometer"
        )
        hea_path = root / f"{record_name}.hea"
        dat_path = root / f"{record_name}.dat"
        rows.append(
            {
                "dataset": "ltmm",
                "record_id": record_name,
                "path": _posix(dat_path),
                "header_path": _posix(hea_path),
                "record_type": record_type,
                "subject_id": _subject_id(record_name),
                "group": group,
                "label_source": "record_name_and_clinical_table",
                "modality": "waist_accelerometer",
                "available": dat_path.exists() and hea_path.exists(),
                "header_available": hea_path.exists(),
                "data_available": dat_path.exists(),
                "sha256": checksum_map.get(f"{record_name}.dat"),
                "header_sha256": checksum_map.get(f"{record_name}.hea"),
            }
        )

    return sorted(rows, key=lambda row: (str(row["record_type"]), str(row["record_id"])))


def write_ltmm_manifest(raw_dir: str | Path, output_path: str | Path) -> int:
    rows = build_ltmm_manifest(raw_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _read_records(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _read_sha256sums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        checksum, filename = parts
        checksums[filename.lstrip("*")] = checksum
    return checksums


def _discover_record_names(root: Path) -> list[str]:
    names = set()
    for header in root.rglob("*.hea"):
        if header.name.startswith("."):
            continue
        names.add(header.relative_to(root).with_suffix("").as_posix())
    return sorted(names)


def _ltmm_group(record_name: str) -> str:
    name = Path(record_name).name.lower()
    if name.startswith("co"):
        return "control"
    if name.startswith("fl"):
        return "faller"
    return "unknown"


def _subject_id(record_name: str) -> str:
    name = Path(record_name).name
    match = re.match(r"(?P<prefix>[A-Za-z]+)(?P<number>\d+)", name)
    if not match:
        return "unknown"
    return f"ltmm_{match.group('prefix').lower()}_{match.group('number')}"


def _posix(path: Path) -> str:
    return path.as_posix()
