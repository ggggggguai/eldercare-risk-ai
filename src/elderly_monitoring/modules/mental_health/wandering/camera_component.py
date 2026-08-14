"""Portable detector/tracker identity validation for camera preparation paths."""

from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from elderly_monitoring.modules.mental_health.wandering.camera_adapter import (
    CameraAdapterError,
)


_FIELDS = {
    "detector": frozenset({"backend", "model", "version"}),
    "tracker": frozenset({"backend", "config", "version"}),
}
_BACKEND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}$")
_SENSITIVE_MARKERS = (
    "://",
    "url=",
    "credential",
    "header",
    "token",
    "password",
    "secret",
    "cookie",
    "authorization",
    "bearer",
)


def validate_camera_component(value: Any, *, name: str) -> dict[str, str]:
    """Validate one exact component mapping as portable deidentified metadata."""

    fields = _FIELDS.get(name)
    if fields is None:
        raise CameraAdapterError("camera component name must be detector or tracker")
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise CameraAdapterError(f"{name} fields must match the exact v1 set")
    output = {
        field: _safe_text(value[field], f"{name}.{field}", limit=256)
        for field in sorted(fields)
    }
    output["backend"] = _portable_scalar(
        output["backend"], f"{name}.backend", pattern=_BACKEND
    )
    output["version"] = _portable_scalar(
        output["version"], f"{name}.version", pattern=_VERSION
    )
    filename_field = "model" if name == "detector" else "config"
    _portable_filename(output[filename_field], f"{name}.{filename_field}")
    return output


def validate_camera_components(sidecar: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Validate both component mappings from a normalized camera sidecar."""

    return {
        "detector": validate_camera_component(sidecar.get("detector"), name="detector"),
        "tracker": validate_camera_component(sidecar.get("tracker"), name="tracker"),
    }


def _portable_scalar(value: str, field: str, *, pattern: re.Pattern[str]) -> str:
    lowered = value.lower()
    windows = PureWindowsPath(value)
    if (
        pattern.fullmatch(value) is None
        or value in {".", ".."}
        or any(marker in lowered for marker in _SENSITIVE_MARKERS)
        or any(marker in value for marker in ("@", "?", "#", "/", "\\"))
        or bool(windows.drive)
        or windows.is_absolute()
        or PurePosixPath(value).is_absolute()
    ):
        raise CameraAdapterError(f"{field} must be portable deidentified metadata")
    return value


def _portable_filename(value: str, field: str) -> str:
    lowered = value.lower()
    windows = PureWindowsPath(value)
    if (
        value in {".", ".."}
        or any(marker in lowered for marker in _SENSITIVE_MARKERS)
        or any(marker in value for marker in ("@", "?", "#", "/", "\\"))
        or bool(windows.drive)
        or windows.is_absolute()
        or PurePosixPath(value).is_absolute()
    ):
        raise CameraAdapterError(f"{field} must be a deidentified file name, not a path")
    return value


def _safe_text(value: Any, field: str, *, limit: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > limit
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise CameraAdapterError(f"{field} must be a safe non-empty string")
    return value


__all__ = ["validate_camera_component", "validate_camera_components"]
