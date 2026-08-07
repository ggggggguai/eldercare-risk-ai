"""Stable cognitive API errors and FastAPI validation mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from elderly_monitoring.modules.mental_health.submodules.cognitive_change_clue.schemas import (
    ERROR_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    SUPPORTED_MODEL_VERSIONS,
    CognitiveErrorCode,
    CognitiveErrorItem,
    CognitiveErrorResponse,
)


ERROR_HTTP_STATUS: dict[str, int] = {
    "AUTHENTICATION_FAILED": 401,
    "UNSUPPORTED_SCHEMA_VERSION": 422,
    "UNKNOWN_FIELD": 422,
    "FIELD_VALIDATION_ERROR": 422,
    "MEDIA_INVALID": 422,
    "MEDIA_TOO_LARGE": 422,
    "MEDIA_TOO_LONG": 422,
    "MEDIA_MISALIGNED": 422,
    "MEDIA_DECODE_FAILED": 422,
    "MODEL_VERSION_UNSUPPORTED": 422,
    "MEDIA_UNREACHABLE": 408,
    "MODEL_ARTIFACT_UNAVAILABLE": 503,
    "ALGORITHM_TIMEOUT": 504,
    "INTERNAL_ERROR": 500,
}

ERROR_MESSAGES: dict[str, str] = {
    "AUTHENTICATION_FAILED": "cognitive service authentication failed",
    "UNSUPPORTED_SCHEMA_VERSION": "cognitive request schema version is unsupported",
    "UNKNOWN_FIELD": "cognitive request contains an unknown field",
    "FIELD_VALIDATION_ERROR": "cognitive request validation failed",
    "MEDIA_INVALID": "cognitive media declaration is invalid",
    "MEDIA_TOO_LARGE": "cognitive media exceeds the size limit",
    "MEDIA_TOO_LONG": "cognitive media duration is outside the supported range",
    "MEDIA_MISALIGNED": "cognitive audio and video are not aligned",
    "MEDIA_DECODE_FAILED": "cognitive media could not be decoded",
    "MODEL_VERSION_UNSUPPORTED": "cognitive model version is unsupported",
    "MEDIA_UNREACHABLE": "cognitive media URL could not be reached",
    "MODEL_ARTIFACT_UNAVAILABLE": "cognitive model package is unavailable",
    "ALGORITHM_TIMEOUT": "cognitive inference exceeded the time limit",
    "INTERNAL_ERROR": "cognitive algorithm service internal error",
}


class RequestValidationErrorLike(Protocol):
    body: object

    def errors(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class CognitiveAPIError(Exception):
    code: CognitiveErrorCode
    request_id: str | None = None
    errors: tuple[CognitiveErrorItem, ...] = ()

    @property
    def status_code(self) -> int:
        return ERROR_HTTP_STATUS[self.code]

    def as_response(self) -> CognitiveErrorResponse:
        return build_error_response(
            code=self.code,
            request_id=self.request_id,
            errors=self.errors,
        )


def build_error_response(
    *,
    code: CognitiveErrorCode,
    request_id: str | None,
    errors: Sequence[CognitiveErrorItem | Mapping[str, str]] = (),
) -> CognitiveErrorResponse:
    return CognitiveErrorResponse.model_validate(
        {
            "detail": {
                "schema_version": ERROR_SCHEMA_VERSION,
                "request_id": request_id,
                "code": code,
                "message": ERROR_MESSAGES[code],
                "errors": list(errors),
            }
        }
    )


def validation_error_response(
    exception: RequestValidationErrorLike,
) -> CognitiveErrorResponse:
    body = exception.body
    raw_errors = exception.errors()
    request_id = _request_id(body)
    code = _classify_validation_error(body, raw_errors)
    errors = tuple(
        CognitiveErrorItem(
            field=_field_path(error.get("loc", ())),
            reason=str(error.get("msg") or "invalid value"),
        )
        for error in raw_errors
    )
    if code == "UNSUPPORTED_SCHEMA_VERSION":
        errors = (
            CognitiveErrorItem(
                field="schema_version",
                reason=f"must equal {REQUEST_SCHEMA_VERSION}",
            ),
        )
    elif code == "MODEL_VERSION_UNSUPPORTED":
        errors = (
            CognitiveErrorItem(
                field="model_version",
                reason="must be null, " + ", or ".join(SUPPORTED_MODEL_VERSIONS),
            ),
        )
    return build_error_response(code=code, request_id=request_id, errors=errors)


def _classify_validation_error(
    body: object,
    errors: Sequence[Mapping[str, Any]],
) -> CognitiveErrorCode:
    if isinstance(body, Mapping):
        schema_version = body.get("schema_version")
        if schema_version is not None and schema_version != REQUEST_SCHEMA_VERSION:
            return "UNSUPPORTED_SCHEMA_VERSION"
    if any(str(error.get("type", "")) == "extra_forbidden" for error in errors):
        return "UNKNOWN_FIELD"
    if isinstance(body, Mapping):
        version = body.get("model_version")
        if version is not None and version not in SUPPORTED_MODEL_VERSIONS:
            return "MODEL_VERSION_UNSUPPORTED"
    if any(
        _field_path(error.get("loc", ())).endswith(".source")
        and "HTTP or HTTPS" in str(error.get("msg", ""))
        for error in errors
    ):
        return "MEDIA_INVALID"
    return "FIELD_VALIDATION_ERROR"


def _request_id(body: object) -> str | None:
    if not isinstance(body, Mapping):
        return None
    value = body.get("request_id")
    return value if isinstance(value, str) else None


def _field_path(location: Sequence[object]) -> str:
    values = list(location)
    if values[:1] == ["body"]:
        values = values[1:]
    result = ""
    for item in values:
        if isinstance(item, int):
            result += f"[{item}]"
        else:
            result += ("." if result else "") + str(item)
    return result or "$"


__all__ = [
    "ERROR_HTTP_STATUS",
    "ERROR_MESSAGES",
    "CognitiveAPIError",
    "build_error_response",
    "validation_error_response",
]
