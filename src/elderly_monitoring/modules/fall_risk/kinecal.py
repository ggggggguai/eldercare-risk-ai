from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, getproxies, urlopen


KINECAL_RISK_GROUPS = frozenset({"NF", "FHs", "FHm"})
KINECAL_BASE_URL = "https://physionet.org/files/kinecal/1.0.3/"
KINECAL_MOVEMENTS = (
    "3m-walk-Front-View",
    "Get-Up-And-Go-Front-View",
    "STS-5",
)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int


@dataclass(frozen=True)
class DownloadResult:
    url: str
    path: str
    status: str
    size: int
    sha256: str


class HttpClient(Protocol):
    def read(self, url: str, *, timeout: float) -> bytes: ...


class _HttpxClient:
    def __init__(self, client: Any, http_error: type[Exception]) -> None:
        self._client = client
        self._http_error = http_error

    def read(self, url: str, *, timeout: float) -> bytes:
        try:
            response = self._client.get(url, timeout=timeout)
            response.raise_for_status()
        except self._http_error as exc:
            raise OSError(str(exc)) from exc
        return bytes(response.content)


def read_register_csv(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in csv.DictReader(io.StringIO(text)):
        rows.append(
            {
                str(key): str(value or "").strip()
                for key, value in row.items()
                if key is not None and str(key).strip()
            }
        )
    return rows


def select_risk_group_participants(
    rows: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    return [dict(row) for row in rows if row.get("group") in KINECAL_RISK_GROUPS]


def age_group_mismatches(
    rows: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    mismatches: list[dict[str, str]] = []
    for row in rows:
        age = str(row.get("age", "")).strip()
        if age.isdigit() and int(age) < 65:
            mismatches.append(
                {
                    "part_id": str(row.get("part_id", "")),
                    "group": str(row.get("group", "")),
                    "age": age,
                }
            )
    return mismatches


def participant_number(part_id: str) -> str:
    match = re.search(r"([0-9]+)$", part_id)
    if match is None:
        raise ValueError(f"Invalid KINECAL participant ID: {part_id!r}")
    return match.group(1)


def parse_directory_files(html: str, *, suffix: str) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for line in html.splitlines():
        match = re.search(r'href="([^"]+)"', line)
        if match is None:
            continue
        name = unquote(unescape(match.group(1)))
        if not name.endswith(suffix) or "/" in name or name in {".", ".."}:
            continue
        size_match = re.search(r"([0-9]+)\s*$", line)
        if size_match is None:
            continue
        files.append(RemoteFile(name=name, size=int(size_match.group(1))))
    return files


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    output_path: str | Path,
    *,
    expected_size: int | None = None,
    retries: int = 3,
    timeout: float = 30.0,
    client: HttpClient | None = None,
) -> DownloadResult:
    path = Path(output_path)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return DownloadResult(
            url=url,
            path=path.as_posix(),
            status="skipped",
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    attempts = retries + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if client is None:
                request = Request(url, headers={"User-Agent": "eldercare-ai-kinecal/0.1"})
                with urlopen(request, timeout=timeout) as response, partial.open("wb") as file:
                    while chunk := response.read(1024 * 1024):
                        file.write(chunk)
            else:
                partial.write_bytes(client.read(url, timeout=timeout))
            size = partial.stat().st_size
            if expected_size is not None and size != expected_size:
                raise ValueError(
                    f"Downloaded size mismatch for {url}: expected {expected_size}, got {size}"
                )
            partial.replace(path)
            return DownloadResult(
                url=url,
                path=path.as_posix(),
                status="downloaded",
                size=size,
                sha256=sha256_file(path),
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    assert last_error is not None
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def build_download_summary(
    participants: Iterable[Mapping[str, str]],
    recordings: Iterable[Mapping[str, object]],
    manifest_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    participant_rows = [dict(row) for row in participants]
    recording_rows = [dict(row) for row in recordings]
    download_rows = [dict(row) for row in manifest_rows]
    group_counts = {
        group: sum(row.get("group") == group for row in participant_rows)
        for group in ("NF", "FHs", "FHm")
    }
    recording_counts: dict[str, int] = {}
    for row in recording_rows:
        status = str(row.get("status", "unknown"))
        recording_counts[status] = recording_counts.get(status, 0) + 1
    usable_rows = [
        row for row in download_rows if row.get("status") in {"downloaded", "skipped"}
    ]
    status_values = sorted({str(row.get("status", "unknown")) for row in download_rows})
    return {
        "participant_count": len(participant_rows),
        "group_counts": group_counts,
        "age_group_mismatch": age_group_mismatches(participant_rows),
        "recording_counts": recording_counts,
        "file_count": len(usable_rows),
        "total_bytes": sum(int(row.get("size", 0) or 0) for row in usable_rows),
        "download_status_counts": {
            status: sum(str(row.get("status", "unknown")) == status for row in download_rows)
            for status in status_values
        },
    }


def download_kinecal(
    output_dir: str | Path,
    *,
    base_url: str = KINECAL_BASE_URL,
    workers: int = 8,
    retries: int = 3,
    timeout: float = 30.0,
    strict: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_root = base_url.rstrip("/") + "/"
    download_file(
        urljoin(source_root, "register.csv"),
        root / "register.csv",
        retries=retries,
        timeout=timeout,
    )
    download_file(
        urljoin(source_root, "LICENSE.txt"),
        root / "LICENSE.txt",
        retries=retries,
        timeout=timeout,
    )
    participants = select_risk_group_participants(
        read_register_csv((root / "register.csv").read_text(encoding="utf-8"))
    )

    recording_inputs = [
        (participant, movement)
        for participant in participants
        for movement in KINECAL_MOVEMENTS
    ]

    def inspect_recording(item: tuple[dict[str, str], str]) -> dict[str, Any]:
        participant, movement = item
        part_number = participant_number(participant["part_id"])
        relative_dir = f"kinecal/{part_number}/{part_number}_{movement}/skel/"
        directory_url = urljoin(source_root, relative_dir)
        try:
            html = _fetch_text(
                directory_url,
                retries=retries,
                timeout=timeout,
            )
            files = parse_directory_files(html, suffix=".txt")
        except RuntimeError as exc:
            return {
                "part_id": participant["part_id"],
                "group": participant["group"],
                "age": participant.get("age"),
                "movement": movement,
                "status": "missing",
                "file_count": 0,
                "bytes": 0,
                "source_url": directory_url,
                "error": str(exc),
                "files": [],
                "relative_dir": relative_dir,
            }
        status = "available" if files else "empty"
        return {
            "part_id": participant["part_id"],
            "group": participant["group"],
            "age": participant.get("age"),
            "movement": movement,
            "status": status,
            "file_count": len(files),
            "bytes": sum(file.size for file in files),
            "source_url": directory_url,
            "files": files,
            "relative_dir": relative_dir,
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        recordings = list(pool.map(inspect_recording, recording_inputs))

    download_tasks: list[tuple[dict[str, Any], RemoteFile]] = []
    for recording in recordings:
        for remote_file in recording.pop("files"):
            download_tasks.append((recording, remote_file))
    download_tasks.sort(
        key=lambda item: (
            str(item[0]["part_id"]),
            str(item[0]["movement"]),
            item[1].name,
        )
    )

    def download_task(item: tuple[dict[str, Any], RemoteFile]) -> dict[str, Any]:
        recording, remote_file = item
        relative_path = Path(str(recording["relative_dir"])) / remote_file.name
        source_url = urljoin(str(recording["source_url"]), remote_file.name)
        try:
            result = download_file(
                source_url,
                root / relative_path,
                expected_size=remote_file.size,
                retries=retries,
                timeout=timeout,
                client=client,
            )
            return {
                "part_id": recording["part_id"],
                "group": recording["group"],
                "age": recording.get("age"),
                "movement": recording["movement"],
                "source_url": source_url,
                "path": relative_path.as_posix(),
                "status": result.status,
                "size": result.size,
                "sha256": result.sha256,
            }
        except RuntimeError as exc:
            return {
                "part_id": recording["part_id"],
                "group": recording["group"],
                "age": recording.get("age"),
                "movement": recording["movement"],
                "source_url": source_url,
                "path": relative_path.as_posix(),
                "status": "failed",
                "size": remote_file.size,
                "sha256": None,
                "error": str(exc),
            }

    client: HttpClient | None = None
    with _pooled_http_client(source_root, workers=workers, timeout=timeout) as client:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            manifest_rows = list(pool.map(download_task, download_tasks))
    manifest_rows.sort(key=lambda row: str(row["path"]))

    summary = build_download_summary(participants, recordings, manifest_rows)
    summary.update(
        {
            "dataset": "kinecal",
            "source_version": "1.0.3",
            "source_url": source_root,
            "movements": list(KINECAL_MOVEMENTS),
            "recordings": recordings,
        }
    )
    _write_jsonl(root / "download_manifest.jsonl", manifest_rows)
    _write_json(root / "download_summary.json", summary)

    missing_count = sum(
        count
        for status, count in summary["recording_counts"].items()
        if status != "available"
    )
    failed_count = int(summary["download_status_counts"].get("failed", 0))
    if strict and (missing_count or failed_count):
        raise RuntimeError(
            f"KINECAL download incomplete: {missing_count} unavailable recordings, "
            f"{failed_count} failed files. See {root / 'download_summary.json'}."
        )
    return summary


def verify_kinecal_download(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    errors: list[str] = []
    for required in ("register.csv", "LICENSE.txt", "download_manifest.jsonl"):
        if not (root / required).is_file():
            errors.append(f"Missing required file: {required}")

    manifest_path = root / "download_manifest.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    if manifest_path.is_file():
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                manifest_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid manifest JSON at line {line_number}: {exc}")

    verified_files = 0
    for row in manifest_rows:
        if row.get("status") not in {"downloaded", "skipped"}:
            errors.append(f"Manifest contains failed file: {row.get('path')}")
            continue
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"Unsafe manifest path: {relative}")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing downloaded file: {relative.as_posix()}")
            continue
        expected_size = int(row.get("size", -1))
        if path.stat().st_size != expected_size:
            errors.append(
                f"Size mismatch for {relative.as_posix()}: "
                f"expected {expected_size}, got {path.stat().st_size}"
            )
            continue
        expected_hash = str(row.get("sha256", ""))
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch for {relative.as_posix()}")
            continue
        verified_files += 1

    binary_files = list(root.rglob("*.bin"))
    if binary_files:
        errors.append(f"Unexpected depth binary files: {len(binary_files)}")
    return {
        "verified_files": verified_files,
        "manifest_rows": len(manifest_rows),
        "errors": errors,
    }


def _fetch_text(url: str, *, retries: int, timeout: float) -> str:
    attempts = retries + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "eldercare-ai-kinecal/0.1"})
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
    assert last_error is not None
    raise RuntimeError(f"Failed to read {url}: {last_error}") from last_error


@contextmanager
def _pooled_http_client(
    base_url: str,
    *,
    workers: int,
    timeout: float,
) -> Iterator[HttpClient | None]:
    try:
        import httpx
    except ImportError:
        yield None
        return

    parsed = urlparse(base_url)
    proxy = getproxies().get(parsed.scheme)
    limits = httpx.Limits(
        max_connections=max(1, workers),
        max_keepalive_connections=max(1, workers),
    )
    try:
        client = httpx.Client(
            proxy=proxy,
            trust_env=False,
            limits=limits,
            timeout=timeout,
            headers={"User-Agent": "eldercare-ai-kinecal/0.1"},
        )
    except (ImportError, ValueError):
        yield None
        return

    try:
        yield _HttpxClient(client, httpx.HTTPError)
    finally:
        client.close()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    partial.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    partial.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)
