from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

from elderly_monitoring.modules.mental_health.mood_social.config import (
    MoodSocialConfig,
    load_mood_social_config,
)
from elderly_monitoring.modules.mental_health.mood_social.schemas import (
    ERROR_SCHEMA_VERSION,
    MOOD_SOCIAL_TIMEZONE,
    REQUEST_SCHEMA_VERSION,
    MoodSocialErrorCode,
    MoodSocialErrorItem,
    MoodSocialErrorResponse,
    MoodSocialInferRequest,
    MoodSocialInferResponse,
)
from elderly_monitoring.modules.mental_health.mood_social.pipeline import (
    MoodSocialPipeline,
    MoodSocialPipelineUnavailableError,
)


MOOD_SOCIAL_INFER_PATH = "/v1/mental-health/mood-social/infer"

ERROR_HTTP_STATUS: dict[str, int] = {
    "AUTHENTICATION_FAILED": 401,
    "UNSUPPORTED_SCHEMA_VERSION": 422,
    "FIELD_VALIDATION_ERROR": 422,
    "DATE_VALIDATION_ERROR": 422,
    "SOURCE_DECLARATION_MISMATCH": 422,
    "MODEL_ARTIFACT_UNAVAILABLE": 503,
    "INTERNAL_ERROR": 500,
}

ERROR_MESSAGES: dict[str, str] = {
    "AUTHENTICATION_FAILED": "算法服务鉴权失败",
    "UNSUPPORTED_SCHEMA_VERSION": "请求 schema 版本不受支持",
    "FIELD_VALIDATION_ERROR": "请求字段校验失败",
    "DATE_VALIDATION_ERROR": "请求日期校验失败",
    "SOURCE_DECLARATION_MISMATCH": "数据来源声明与领域字段不一致",
    "MODEL_ARTIFACT_UNAVAILABLE": "情绪与社交关注模型包不可用",
    "INTERNAL_ERROR": "情绪与社交关注算法服务内部错误",
}


class RequestValidationErrorLike(Protocol):
    body: object

    def errors(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class MoodSocialAPIError(Exception):
    code: MoodSocialErrorCode
    request_id: str | None = None
    errors: tuple[MoodSocialErrorItem, ...] = ()

    @property
    def status_code(self) -> int:
        return ERROR_HTTP_STATUS[self.code]

    def as_response(self) -> MoodSocialErrorResponse:
        return build_mood_social_error_response(
            code=self.code,
            request_id=self.request_id,
            errors=self.errors,
        )


def build_mood_social_error_response(
    *,
    code: MoodSocialErrorCode,
    request_id: str | None,
    errors: Sequence[MoodSocialErrorItem | Mapping[str, str]] = (),
) -> MoodSocialErrorResponse:
    return MoodSocialErrorResponse.model_validate(
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
) -> MoodSocialErrorResponse:
    body = exception.body
    request_id = _request_id_from_body(body)
    raw_errors = exception.errors()
    code = _classify_validation_error(body, raw_errors)
    errors = [
        MoodSocialErrorItem(
            field=_error_field(error),
            reason=str(error.get("msg") or "invalid value"),
        )
        for error in raw_errors
    ]
    if code == "UNSUPPORTED_SCHEMA_VERSION":
        errors = [
            MoodSocialErrorItem(
                field="schema_version",
                reason=f"must equal {REQUEST_SCHEMA_VERSION}",
            )
        ]
    return build_mood_social_error_response(
        code=code,
        request_id=request_id,
        errors=errors,
    )


def infer_mood_social(
    request: MoodSocialInferRequest,
    *,
    config: MoodSocialConfig | None = None,
    pipeline: MoodSocialPipeline | None = None,
) -> MoodSocialInferResponse:
    """Run the V3 pipeline while preserving the frozen artifact 503 boundary."""
    current = config or load_mood_social_config()
    package_directory = (
        current.models.directory
        / "packages"
        / "MH-20260802-013"
    )
    try:
        runtime = pipeline or (
            _load_default_pipeline(str(package_directory.resolve()))
            if config is None
            else MoodSocialPipeline.from_package(
                package_directory,
                config=current,
            )
        )
        return runtime.predict(request)
    except MoodSocialPipelineUnavailableError as exc:
        raise MoodSocialAPIError(
            code="MODEL_ARTIFACT_UNAVAILABLE",
            request_id=request.request_id,
        ) from exc


@lru_cache(maxsize=4)
def _load_default_pipeline(package_directory: str) -> MoodSocialPipeline:
    return MoodSocialPipeline.from_package(Path(package_directory))


def _request_id_from_body(body: object) -> str | None:
    if not isinstance(body, Mapping):
        return None
    value = body.get("request_id")
    return value if isinstance(value, str) else None


def _classify_validation_error(
    body: object,
    errors: Sequence[Mapping[str, Any]],
) -> MoodSocialErrorCode:
    if isinstance(body, Mapping):
        schema_version = body.get("schema_version")
        if (
            schema_version is not None
            and schema_version != REQUEST_SCHEMA_VERSION
        ):
            return "UNSUPPORTED_SCHEMA_VERSION"

    error_types = {str(error.get("type", "")) for error in errors}
    date_collection_fields = {
        "target_date",
        "current_daily_features",
        "history_daily_features",
        "history_attention_indices",
    }
    custom_error_types = {
        "date_validation_error",
        "field_validation_error",
        "source_declaration_mismatch",
    }
    standard_errors = [
        error
        for error in errors
        if str(error.get("type", "")) not in custom_error_types
    ]
    non_date_standard_errors = [
        error
        for error in standard_errors
        if not _is_date_collection_length_error(
            error,
            date_collection_fields=date_collection_fields,
        )
    ]
    if non_date_standard_errors:
        return "FIELD_VALIDATION_ERROR"
    if standard_errors:
        return "DATE_VALIDATION_ERROR"

    if _has_early_field_error(errors):
        return "FIELD_VALIDATION_ERROR"
    if (
        "date_validation_error" in error_types
        or _raw_date_relationship_invalid(body)
    ):
        return "DATE_VALIDATION_ERROR"
    if (
        "source_declaration_mismatch" in error_types
        or _raw_source_declaration_mismatch(body)
    ):
        return "SOURCE_DECLARATION_MISMATCH"
    return "FIELD_VALIDATION_ERROR"


def _is_date_collection_length_error(
    error: Mapping[str, Any],
    *,
    date_collection_fields: set[str],
) -> bool:
    if str(error.get("type", "")) != "too_long":
        return False
    location = tuple(error.get("loc", ()))
    body_location = location[1:] if location[:1] == ("body",) else location
    return bool(
        body_location
        and body_location[0] in date_collection_fields
    )


def _has_early_field_error(
    errors: Sequence[Mapping[str, Any]],
) -> bool:
    for error in errors:
        context = error.get("ctx")
        if (
            isinstance(context, Mapping)
            and context.get("field") == "available_sources"
        ):
            return True
    return False


def _raw_date_relationship_invalid(body: object) -> bool:
    if not isinstance(body, Mapping):
        return False
    target_date = _raw_date(body.get("target_date"))
    if target_date is None:
        return False
    if target_date >= datetime.now(ZoneInfo(MOOD_SOCIAL_TIMEZONE)).date():
        return True

    current = body.get("current_daily_features")
    if isinstance(current, Mapping):
        current_date = _raw_date(current.get("date"))
        if current_date is not None and current_date != target_date:
            return True

    history = body.get("history_daily_features")
    if isinstance(history, list):
        history_dates = [
            _raw_date(record.get("date"))
            for record in history
            if isinstance(record, Mapping)
        ]
        if len(history_dates) == len(history) and all(
            value is not None for value in history_dates
        ):
            concrete_history_dates = [
                value for value in history_dates if value is not None
            ]
            if (
                concrete_history_dates != sorted(concrete_history_dates)
                or len(concrete_history_dates)
                != len(set(concrete_history_dates))
            ):
                return True
            earliest = target_date - timedelta(days=28)
            latest = target_date - timedelta(days=1)
            if any(
                not earliest <= value <= latest
                for value in concrete_history_dates
            ):
                return True

    attention_history = body.get("history_attention_indices")
    if isinstance(attention_history, list):
        attention_dates = [
            _raw_date(record.get("date"))
            for record in attention_history
            if isinstance(record, Mapping)
        ]
        if len(attention_dates) == len(attention_history) and all(
            value is not None for value in attention_dates
        ):
            concrete_attention_dates = [
                value for value in attention_dates if value is not None
            ]
            if (
                concrete_attention_dates != sorted(concrete_attention_dates)
                or len(concrete_attention_dates)
                != len(set(concrete_attention_dates))
                or any(value >= target_date for value in concrete_attention_dates)
            ):
                return True
    return False


def _raw_source_declaration_mismatch(body: object) -> bool:
    if not isinstance(body, Mapping):
        return False
    raw_sources = body.get("available_sources")
    if not isinstance(raw_sources, list) or not all(
        isinstance(source, str) for source in raw_sources
    ):
        return False
    declared = set(raw_sources)

    raw_records: list[object] = [body.get("current_daily_features")]
    history = body.get("history_daily_features")
    if isinstance(history, list):
        raw_records.extend(history)
    for record in raw_records:
        if not isinstance(record, Mapping):
            continue
        if record.get("activity") is not None and "camera" not in declared:
            return True
        if (
            record.get("sleep") is not None
            or record.get("physiology") is not None
        ) and "sleep_device" not in declared:
            return True
        if record.get("social") is not None and "s10" not in declared:
            return True
    return False


def _raw_date(value: object) -> date | None:
    if (
        not isinstance(value, str)
        or len(value) != 10
        or value[4] != "-"
        or value[7] != "-"
    ):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _error_field(error: Mapping[str, Any]) -> str:
    location = list(error.get("loc", ()))
    if location[:1] == ["body"]:
        location = location[1:]
    location_field = _format_error_location(location)

    context = error.get("ctx")
    if isinstance(context, Mapping):
        explicit_field = context.get("field")
        if isinstance(explicit_field, str) and explicit_field:
            if not location_field:
                return explicit_field
            if (
                explicit_field == location_field
                or explicit_field.startswith(f"{location_field}.")
                or explicit_field.startswith(f"{location_field}[")
                or location_field.endswith(f".{explicit_field}")
                or f".{explicit_field}[" in location_field
                or location_field.startswith(f"{explicit_field}[")
            ):
                return (
                    explicit_field
                    if explicit_field.startswith(location_field)
                    else location_field
                )
            return f"{location_field}.{explicit_field}"

    return location_field or "$"


def _format_error_location(location: Sequence[object]) -> str:
    result = ""
    for item in location:
        if isinstance(item, int):
            result += f"[{item}]"
        else:
            result += ("." if result else "") + str(item)
    return result


__all__ = [
    "ERROR_HTTP_STATUS",
    "ERROR_MESSAGES",
    "MOOD_SOCIAL_INFER_PATH",
    "MoodSocialAPIError",
    "build_mood_social_error_response",
    "infer_mood_social",
    "validation_error_response",
]
