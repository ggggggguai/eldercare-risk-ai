from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_VERSION = "roi_annotation_v1"

ROI_TYPES = {
    "bed",
    "sofa",
    "dining_table",
    "doorway",
    "bathroom_entrance",
    "high_risk_passage",
    "activity_area",
    "ignore_area",
    "unknown",
}

ROI_LABELS = {
    "bed": "床",
    "sofa": "沙发",
    "dining_table": "餐桌",
    "doorway": "门口",
    "bathroom_entrance": "卫生间门口",
    "high_risk_passage": "高风险通道",
    "activity_area": "活动统计区",
    "ignore_area": "忽略区",
    "unknown": "未知区域",
}


class RoiValidationError(ValueError):
    pass


def parse_model_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise RoiValidationError("模型返回为空")
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RoiValidationError("模型返回不是合法 JSON") from exc
    if not isinstance(payload, dict):
        raise RoiValidationError("模型返回 JSON 根节点必须是对象")
    return payload


def validate_roi_payload(
    payload: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    expected_types: list[str] | None = None,
) -> dict[str, Any]:
    rois = payload.get("rois")
    if not isinstance(rois, list):
        raise RoiValidationError("缺少 rois 数组")
    if len(rois) > 40:
        raise RoiValidationError("ROI 数量超过 40 个")

    normalized_rois = [_normalize_roi(roi, index) for index, roi in enumerate(rois, start=1)]
    seen_ids: set[str] = set()
    for roi in normalized_rois:
        if roi["roi_id"] in seen_ids:
            raise RoiValidationError(f"ROI ID 重复: {roi['roi_id']}")
        seen_ids.add(roi["roi_id"])

    expected = [item for item in (expected_types or []) if item in ROI_TYPES]
    present = {roi["type"] for roi in normalized_rois}
    missing_expected = list(dict.fromkeys(payload.get("missing_expected") or []))
    for roi_type in expected:
        if roi_type not in present and roi_type not in missing_expected:
            missing_expected.append(roi_type)

    warnings = list(dict.fromkeys(payload.get("warnings") or []))
    if payload.get("schema_version") not in (None, SCHEMA_VERSION):
        warnings.append("schema_version 不匹配，已按 roi_annotation_v1 解析")

    return {
        "schema_version": SCHEMA_VERSION,
        "image": {
            "width": image_width,
            "height": image_height,
            "coordinate_system": "normalized_0_1",
        },
        "rois": normalized_rois,
        "missing_expected": missing_expected,
        "warnings": list(dict.fromkeys(warnings)),
        "needs_human_review": True,
    }


def _normalize_roi(roi: Any, index: int) -> dict[str, Any]:
    if not isinstance(roi, dict):
        raise RoiValidationError("ROI 必须是对象")
    roi_type = str(roi.get("type") or "unknown")
    if roi_type not in ROI_TYPES:
        raise RoiValidationError(f"未知 ROI type: {roi_type}")

    shape = roi.get("shape")
    if not isinstance(shape, dict):
        raise RoiValidationError("ROI 缺少 shape")
    if shape.get("type") != "polygon":
        raise RoiValidationError("ROI shape.type 必须为 polygon")

    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise RoiValidationError("polygon 至少需要 3 个点")
    normalized_points = [_normalize_point(point) for point in points]

    bbox = shape.get("bbox")
    if not isinstance(bbox, dict):
        raise RoiValidationError("ROI 缺少 bbox")
    normalized_bbox = {
        "x1": _normalize_unit(bbox.get("x1"), "bbox.x1"),
        "y1": _normalize_unit(bbox.get("y1"), "bbox.y1"),
        "x2": _normalize_unit(bbox.get("x2"), "bbox.x2"),
        "y2": _normalize_unit(bbox.get("y2"), "bbox.y2"),
    }
    if normalized_bbox["x1"] >= normalized_bbox["x2"] or normalized_bbox["y1"] >= normalized_bbox["y2"]:
        raise RoiValidationError("bbox 范围无效")

    confidence = _normalize_confidence(roi.get("confidence", 0.0))
    quality_flags = list(dict.fromkeys(roi.get("quality_flags") or []))
    if confidence < 0.5 and "low_confidence" not in quality_flags:
        quality_flags.append("low_confidence")

    activity_stats = _default_activity_stats(roi_type)
    activity_stats.update(_safe_dict(roi.get("activity_stats")))
    risk = _default_risk(roi_type)
    risk.update(_safe_dict(roi.get("risk")))
    risk["risk_tags"] = list(dict.fromkeys(risk.get("risk_tags") or []))

    return {
        "roi_id": str(roi.get("roi_id") or f"roi_{roi_type}_{index:02d}")[:64],
        "type": roi_type,
        "label_zh": str(roi.get("label_zh") or ROI_LABELS[roi_type])[:64],
        "room_id": _optional_str(roi.get("room_id"), 64),
        "zone_id": _optional_str(roi.get("zone_id"), 64),
        "shape": {"type": "polygon", "points": normalized_points, "bbox": normalized_bbox},
        "confidence": confidence,
        "activity_stats": activity_stats,
        "risk": risk,
        "quality_flags": quality_flags,
        "evidence": str(roi.get("evidence") or "")[:500],
    }


def _normalize_point(point: Any) -> dict[str, float]:
    if not isinstance(point, dict):
        raise RoiValidationError("polygon point 必须是对象")
    return {"x": _normalize_unit(point.get("x"), "point.x"), "y": _normalize_unit(point.get("y"), "point.y")}


def _normalize_unit(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RoiValidationError(f"{field_name} 不是数字") from exc
    if number < 0 or number > 1:
        raise RoiValidationError(f"{field_name} 坐标越界")
    return round(number, 6)


def _normalize_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return round(min(1.0, max(0.0, number)), 4)


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_str(value: Any, max_length: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_length] if text else None


def _default_activity_stats(roi_type: str) -> dict[str, Any]:
    if roi_type in {"bed", "sofa"}:
        return {"allowed": False, "use": "exclude_resting"}
    if roi_type == "dining_table":
        return {"allowed": False, "use": "exclude_seated_meal"}
    if roi_type == "activity_area":
        return {"allowed": True, "use": "movement_and_stay"}
    if roi_type in {"doorway", "bathroom_entrance", "high_risk_passage"}:
        return {"allowed": True, "use": "movement_transition"}
    return {"allowed": False, "use": "ignore"}


def _default_risk(roi_type: str) -> dict[str, Any]:
    if roi_type in {"bathroom_entrance", "high_risk_passage"}:
        return {"is_high_risk": True, "risk_tags": [roi_type]}
    return {"is_high_risk": False, "risk_tags": []}
