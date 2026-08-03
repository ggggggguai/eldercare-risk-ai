"""Verify the native FFmpeg tools used by the ASR media decoder."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from elderly_monitoring.modules.asr.settings import ASRSettings


class ASRNativeRuntimeError(RuntimeError):
    """Raised when the configured native media runtime is not frozen."""


def render_native_assets_manifest(settings: ASRSettings) -> str:
    payload: dict[str, Any] = {"schema_version": "asr_native_assets_v1", "tools": {}}
    for name, command in _commands(settings).items():
        path = _resolve_command(command)
        payload["tools"][name] = {
            "command": command,
            "filename": path.name,
            "version": _version_line(path),
            "sha256": _sha256_file(path),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def verify_native_assets(settings: ASRSettings) -> None:
    manifest_path = settings.native_assets_manifest.resolve()
    if not manifest_path.is_file():
        raise ASRNativeRuntimeError(f"ASR native runtime manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        tools = payload["tools"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ASRNativeRuntimeError("ASR native runtime manifest is invalid") from exc
    if payload.get("schema_version") != "asr_native_assets_v1":
        raise ASRNativeRuntimeError("ASR native runtime manifest version is unsupported")
    for name, command in _commands(settings).items():
        expected = tools.get(name)
        if not isinstance(expected, dict):
            raise ASRNativeRuntimeError(f"ASR native runtime entry is missing: {name}")
        path = _resolve_command(command)
        actual = _sha256_file(path)
        if actual != expected.get("sha256"):
            raise ASRNativeRuntimeError(f"ASR native runtime checksum mismatch: {name}")


def _commands(settings: ASRSettings) -> dict[str, str]:
    return {"ffmpeg": settings.ffmpeg_path, "ffprobe": settings.ffprobe_path}


def _resolve_command(command: str) -> Path:
    path = Path(command)
    resolved = path if path.is_file() else shutil.which(command)
    if not resolved:
        raise ASRNativeRuntimeError(f"ASR native runtime executable is unavailable: {command}")
    return Path(resolved).resolve()


def _version_line(path: Path) -> str:
    try:
        completed = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ASRNativeRuntimeError(f"could not query native runtime: {path}") from exc
    line = completed.stdout.decode("utf-8", errors="replace").splitlines()
    if completed.returncode != 0 or not line:
        raise ASRNativeRuntimeError(f"could not query native runtime: {path}")
    return line[0]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ASRNativeRuntimeError", "render_native_assets_manifest", "verify_native_assets"]
