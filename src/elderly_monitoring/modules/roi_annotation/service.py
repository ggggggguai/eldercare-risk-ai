from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from typing import Any, Protocol

from elderly_monitoring.modules.roi_annotation.ezviz_client import EzvizVisionModelError, EzvizVisionModelResult
from elderly_monitoring.modules.roi_annotation.validation import RoiValidationError, parse_model_json, validate_roi_payload


class RoiClient(Protocol):
    def annotate_rois(
        self,
        *,
        image_base64: str,
        mime_type: str,
        scene_hint: str,
        expected_types: list[str],
    ) -> EzvizVisionModelResult:
        ...


class RoiAnnotationError(RuntimeError):
    def __init__(self, category: str, message: str, status_code: int = 422):
        self.category = category
        self.status_code = status_code
        super().__init__(message)


def annotate_roi_image(
    *,
    image_base64: str,
    mime_type: str,
    image_width: int,
    image_height: int,
    scene_hint: str,
    expected_types: list[str],
    client: RoiClient,
) -> dict[str, Any]:
    raw, image_hash = _decode_image(image_base64)
    model_image_base64, model_mime_type = _compress_for_model(raw, mime_type)

    try:
        model_result = client.annotate_rois(
            image_base64=model_image_base64,
            mime_type=model_mime_type,
            scene_hint=scene_hint,
            expected_types=expected_types,
        )
        model_payload = parse_model_json(model_result.content)
        normalized = validate_roi_payload(
            model_payload,
            image_width=image_width,
            image_height=image_height,
            expected_types=expected_types,
        )
    except EzvizVisionModelError as exc:
        raise RoiAnnotationError(exc.category, str(exc), exc.status_code or 503) from exc
    except RoiValidationError as exc:
        raise RoiAnnotationError("validation", f"ROI 解析失败：{exc}", 422) from exc

    normalized["image"]["sha256"] = image_hash
    normalized["model"] = {
        "provider": "ezviz",
        "name": model_result.model,
        "request_id": model_result.request_id,
        "usage": model_result.usage or {},
    }
    return normalized


def _decode_image(image_base64: str) -> tuple[bytes, str]:
    text = image_base64.strip()
    if "," in text and text.lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        raw = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise RoiAnnotationError("bad_image", "image_base64 不是合法 Base64", 400) from exc
    if not raw:
        raise RoiAnnotationError("bad_image", "图片内容为空", 400)
    return raw, hashlib.sha256(raw).hexdigest()


def _compress_for_model(raw: bytes, mime_type: str) -> tuple[str, str]:
    try:
        from PIL import Image
    except ImportError:
        return base64.b64encode(raw).decode(), mime_type

    try:
        with Image.open(BytesIO(raw)) as image:
            image = image.convert("RGB")
            image.thumbnail((1280, 1280))
            output = BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(output.getvalue()).decode(), "image/jpeg"
    except Exception as exc:
        raise RoiAnnotationError("bad_image", "图片无法解析或压缩", 400) from exc
