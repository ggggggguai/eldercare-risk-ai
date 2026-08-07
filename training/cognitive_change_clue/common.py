"""Shared deterministic I/O for V3.3 cognitive offline data jobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ALGORITHM_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ALGORITHM_ROOT.parents[1]
DEFAULT_PROCESSED_ROOT = (
    ALGORITHM_ROOT / "data" / "processed" / "cognitive_change_clue" / "v3.3.0"
)
DEFAULT_MANIFEST_PATH = DEFAULT_PROCESSED_ROOT / "cogpic_manifest.parquet"
DEFAULT_SPLIT_PATH = DEFAULT_PROCESSED_ROOT / "cogpic_subject_split_v33.json"
DEFAULT_EXCLUSION_REPORT_PATH = DEFAULT_PROCESSED_ROOT / "cogpic_exclusion_report.json"
ASR_MODEL_VERSION = "asr-paraformer-zh-v1.0"
FEATURE_VERSION = "cognitive_features_v3.3.0"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        _replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, payload)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    atomic_write_bytes(path, payload)


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".parquet", dir=target.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=False)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".npz", dir=target.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def workspace_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    workspace = WORKSPACE_ROOT.resolve()
    try:
        return resolved.relative_to(workspace).as_posix()
    except ValueError:
        data_link = WORKSPACE_ROOT / "数据集"
        if data_link.exists():
            data_root = data_link.resolve()
            try:
                return (Path("数据集") / resolved.relative_to(data_root)).as_posix()
            except ValueError:
                pass
        raise ValueError(f"path is outside the workspace and dataset link: {resolved}")


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def resolve_cogpic_root(value: str | Path | None = None) -> Path:
    candidate = (
        resolve_workspace_path(value)
        if value is not None
        else WORKSPACE_ROOT / "数据集" / "心理" / "CogPic"
    )
    if (candidate / "Train").is_dir() and (candidate / "Test").is_dir():
        return candidate.resolve()
    nested = candidate / "CogPic"
    if (nested / "Train").is_dir() and (nested / "Test").is_dir():
        return nested.resolve()
    raise FileNotFoundError(f"CogPic Train/Test directories are missing under {candidate}")


def _replace_with_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 20,
    initial_delay_seconds: float = 0.05,
) -> None:
    """Replace atomically while tolerating short-lived Windows reader locks."""

    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(1.0, initial_delay_seconds * (attempt + 1)))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_parquet(Path(path), engine="pyarrow")


__all__ = [
    "ALGORITHM_ROOT",
    "ASR_MODEL_VERSION",
    "DEFAULT_EXCLUSION_REPORT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_PROCESSED_ROOT",
    "DEFAULT_SPLIT_PATH",
    "FEATURE_VERSION",
    "WORKSPACE_ROOT",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_npz",
    "atomic_write_parquet",
    "canonical_json_bytes",
    "load_json",
    "read_manifest",
    "resolve_cogpic_root",
    "resolve_workspace_path",
    "sha256_bytes",
    "sha256_file",
    "workspace_relative",
]
