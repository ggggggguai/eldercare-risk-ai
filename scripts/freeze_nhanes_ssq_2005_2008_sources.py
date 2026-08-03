"""Freeze the official DATA-006 NHANES 2005-2008 DPQ+SSQ sources.

The generated source manifest contains only file metadata and codebook fields.
It never contains participant records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


DATASET_ID = "nhanes_ssq_2005_2008"
MANIFEST_VERSION = "nhanes-ssq-source-manifest-v1"
COLLECTION_HASH_VERSION = "mood-social-collection-sha256-v1"
SOURCE_RELATIVE = Path("数据集/心理/NHANES-SSQ-2005-2008")
SOURCE_MANIFEST_NAME = "source_manifest.json"
FROZEN_DOWNLOAD_DATE = date(2026, 7, 31)

CYCLES = {"D": "2005-2006", "E": "2007-2008"}
COMPONENTS = ("DEMO", "DPQ", "SSQ")
EXPECTED_FIELDS: dict[str, tuple[str, ...]] = {
    "DEMO_D": (
        "SEQN",
        "SDDSRVYR",
        "RIDSTATR",
        "RIDEXMON",
        "RIAGENDR",
        "RIDAGEYR",
        "RIDAGEMN",
        "RIDAGEEX",
        "RIDRETH1",
        "DMQMILIT",
        "DMDBORN",
        "DMDCITZN",
        "DMDYRSUS",
        "DMDEDUC3",
        "DMDEDUC2",
        "DMDSCHOL",
        "DMDMARTL",
        "DMDHHSIZ",
        "DMDFMSIZ",
        "INDHHINC",
        "INDFMINC",
        "INDFMPIR",
        "RIDEXPRG",
        "DMDHRGND",
        "DMDHRAGE",
        "DMDHRBRN",
        "DMDHREDU",
        "DMDHRMAR",
        "DMDHSEDU",
        "SIALANG",
        "SIAPROXY",
        "SIAINTRP",
        "FIALANG",
        "FIAPROXY",
        "FIAINTRP",
        "MIALANG",
        "MIAPROXY",
        "MIAINTRP",
        "AIALANG",
        "WTINT2YR",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ),
    "DEMO_E": (
        "SEQN",
        "SDDSRVYR",
        "RIDSTATR",
        "RIDEXMON",
        "RIAGENDR",
        "RIDAGEYR",
        "RIDAGEMN",
        "RIDAGEEX",
        "RIDRETH1",
        "DMQMILIT",
        "DMDBORN2",
        "DMDCITZN",
        "DMDYRSUS",
        "DMDEDUC3",
        "DMDEDUC2",
        "DMDSCHOL",
        "DMDMARTL",
        "DMDHHSIZ",
        "DMDFMSIZ",
        "INDHHIN2",
        "INDFMIN2",
        "INDFMPIR",
        "RIDEXPRG",
        "DMDHRGND",
        "DMDHRAGE",
        "DMDHRBR2",
        "DMDHREDU",
        "DMDHRMAR",
        "DMDHSEDU",
        "SIALANG",
        "SIAPROXY",
        "SIAINTRP",
        "FIALANG",
        "FIAPROXY",
        "FIAINTRP",
        "MIALANG",
        "MIAPROXY",
        "MIAINTRP",
        "AIALANG",
        "WTINT2YR",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ),
    "DPQ_D": (
        "SEQN",
        "DPQ010",
        "DPQ020",
        "DPQ030",
        "DPQ040",
        "DPQ050",
        "DPQ060",
        "DPQ070",
        "DPQ080",
        "DPQ090",
        "DPQ100",
    ),
    "DPQ_E": (
        "SEQN",
        "DPQ010",
        "DPQ020",
        "DPQ030",
        "DPQ040",
        "DPQ050",
        "DPQ060",
        "DPQ070",
        "DPQ080",
        "DPQ090",
        "DPQ100",
    ),
    "SSQ_D": (
        "SEQN",
        "SSQ011",
        "SSQ021A",
        "SSQ021B",
        "SSQ021C",
        "SSQ021D",
        "SSQ021E",
        "SSQ021F",
        "SSQ021G",
        "SSQ021H",
        "SSQ021I",
        "SSQ021J",
        "SSQ021K",
        "SSQ021L",
        "SSQ021M",
        "SSQ021N",
        "SSQ031",
        "SSQ041",
        "SSD044",
        "SSQ051",
        "SSQ061",
    ),
    "SSQ_E": (
        "SEQN",
        "SSQ011",
        "SSQ021A",
        "SSQ021B",
        "SSQ021C",
        "SSQ021D",
        "SSQ021E",
        "SSQ021F",
        "SSQ021G",
        "SSQ021H",
        "SSQ021I",
        "SSQ021J",
        "SSQ021K",
        "SSQ021L",
        "SSQ021M",
        "SSQ021N",
        "SSQ031",
        "SSQ041",
        "SSD044",
        "SSQ051",
        "SSQ061",
    ),
}
EXPECTED_CODEBOOK_FIELDS = {
    **EXPECTED_FIELDS,
    "DPQ_D": (
        "SEQN",
        "DPQ001",
        "DPQ010",
        "DPQ020",
        "DPQ030",
        "DPQ040",
        "DPQ050",
        "DPQ060",
        "DPQ070",
        "DPQ080",
        "DPQ090",
        "DPQ095",
        "DPQ100",
    ),
    "DPQ_E": (
        "SEQN",
        "DPQ001",
        "DPQ010",
        "DPQ020",
        "DPQ030",
        "DPQ040",
        "DPQ050",
        "DPQ060",
        "DPQ070",
        "DPQ080",
        "DPQ090",
        "DPQ095",
        "DPQ100",
    ),
}


class SourceFreezeError(RuntimeError):
    """Raised when an official source cannot be frozen deterministically."""


@dataclass(frozen=True)
class OfficialFile:
    cycle: str
    component: str
    kind: str
    file_name: str
    url: str


def _official_files() -> tuple[OfficialFile, ...]:
    rows: list[OfficialFile] = []
    for cycle, public_year in (("D", "2005"), ("E", "2007")):
        for component in COMPONENTS:
            stem = f"{component}_{cycle}"
            for kind, suffix in (("xpt", "xpt"), ("codebook", "htm")):
                rows.append(
                    OfficialFile(
                        cycle=cycle,
                        component=component,
                        kind=kind,
                        file_name=f"{stem}.{suffix}",
                        url=(
                            "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
                            f"{public_year}/DataFiles/{stem}.{suffix}"
                        ),
                    )
                )
    return tuple(rows)


OFFICIAL_FILES = _official_files()


class _CodebookFieldParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "h3":
            return
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        field = attributes.get("id")
        if "vartitle" in classes and field:
            self.fields.append(field)


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


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collection_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(
        rows, key=lambda value: str(value["relative_path"]).encode("utf-8")
    ):
        digest.update(
            (f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n").encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _find_workspace_root(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "algorithm" / "eldercare-risk-ai-main").is_dir() and (
            candidate / "数据集" / "心理"
        ).is_dir():
            return candidate
    raise SourceFreezeError("workspace root was not found")


def _download(official: OfficialFile, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not destination.is_file():
        raise SourceFreezeError(f"source target is not a file: {official.file_name}")
    if destination.is_file() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        official.url,
        headers={"User-Agent": "eldercare-risk-ai/DATA-006 source freeze"},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{official.file_name}.", suffix=".download", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            try:
                with urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        raise SourceFreezeError(
                            f"official download failed: {official.file_name}"
                        )
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            except (HTTPError, URLError, TimeoutError) as exc:
                raise SourceFreezeError(
                    f"official download failed: {official.file_name}"
                ) from exc
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _xpt_fields(path: Path) -> tuple[str, ...]:
    try:
        frame = pd.read_sas(path, format="xport", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise SourceFreezeError(f"XPT is not readable: {path.name}") from exc
    return tuple(str(column) for column in frame.columns)


def _codebook_fields(path: Path) -> tuple[str, ...]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SourceFreezeError(f"codebook is not readable: {path.name}") from exc
    parser = _CodebookFieldParser()
    parser.feed(content)
    parser.close()
    if not parser.fields or len(parser.fields) != len(set(parser.fields)):
        raise SourceFreezeError(f"codebook fields are invalid: {path.name}")
    return tuple(parser.fields)


def build_source_manifest(
    workspace_root: Path,
    *,
    download: bool = False,
    overwrite_downloads: bool = False,
) -> dict[str, Any]:
    source_root = workspace_root / SOURCE_RELATIVE
    if download:
        for official in OFFICIAL_FILES:
            _download(
                official,
                source_root / official.file_name,
                overwrite=overwrite_downloads,
            )
    missing = [
        official.file_name
        for official in OFFICIAL_FILES
        if not (source_root / official.file_name).is_file()
    ]
    if missing:
        raise SourceFreezeError("official source set is incomplete")

    xpt_field_sets: dict[str, tuple[str, ...]] = {}
    codebook_field_sets: dict[str, tuple[str, ...]] = {}
    for cycle in CYCLES:
        for component in COMPONENTS:
            stem = f"{component}_{cycle}"
            expected_xpt = EXPECTED_FIELDS[stem]
            expected_codebook = EXPECTED_CODEBOOK_FIELDS[stem]
            xpt_fields = _xpt_fields(source_root / f"{stem}.xpt")
            codebook_fields = _codebook_fields(source_root / f"{stem}.htm")
            if xpt_fields != expected_xpt or codebook_fields != expected_codebook:
                raise SourceFreezeError(f"official fields changed: {stem}")
            xpt_field_sets[stem] = expected_xpt
            codebook_field_sets[stem] = expected_codebook

    files: list[dict[str, Any]] = []
    collection_rows: list[dict[str, Any]] = []
    for official in OFFICIAL_FILES:
        path = source_root / official.file_name
        stat = path.stat()
        relative = path.relative_to(workspace_root).as_posix()
        digest = _sha256_file(path)
        stem = f"{official.component}_{official.cycle}"
        fields = (
            xpt_field_sets[stem]
            if official.kind == "xpt"
            else codebook_field_sets[stem]
        )
        row = {
            "bytes": stat.st_size,
            "component": official.component,
            "cycle": official.cycle,
            "downloaded_on": FROZEN_DOWNLOAD_DATE.isoformat(),
            "field_count": len(fields),
            "fields": list(fields),
            "file_name": official.file_name,
            "kind": official.kind,
            "relative_path": relative,
            "sha256": digest,
            "url": official.url,
        }
        files.append(row)
        collection_rows.append(
            {
                "relative_path": relative,
                "bytes": stat.st_size,
                "sha256": digest,
            }
        )

    files.sort(key=lambda row: str(row["relative_path"]).encode("utf-8"))
    return {
        "collection_hash_version": COLLECTION_HASH_VERSION,
        "dataset_id": DATASET_ID,
        "downloaded_on": FROZEN_DOWNLOAD_DATE.isoformat(),
        "file_count": len(files),
        "files": files,
        "manifest_version": MANIFEST_VERSION,
        "participant_values_written": False,
        "source_collection_sha256": _collection_sha256(collection_rows),
        "source_relative": SOURCE_RELATIVE.as_posix(),
        "survey_cycles": CYCLES,
        "total_bytes": sum(int(row["bytes"]) for row in files),
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    content = _canonical_json_bytes(payload)
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".stage", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite-downloads", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace_root = _find_workspace_root(args.workspace_root or Path.cwd())
    manifest = build_source_manifest(
        workspace_root,
        download=args.download,
        overwrite_downloads=args.overwrite_downloads,
    )
    manifest_path = workspace_root / SOURCE_RELATIVE / SOURCE_MANIFEST_NAME
    _write_manifest(manifest_path, manifest)
    summary = {
        "dataset_id": DATASET_ID,
        "file_count": manifest["file_count"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_collection_sha256": manifest["source_collection_sha256"],
        "status": "pass",
        "total_bytes": manifest["total_bytes"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
