from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class EzvizVisionModelError(RuntimeError):
    def __init__(self, category: str, message: str, status_code: int | None = None):
        self.category = category
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class EzvizVisionModelResult:
    content: str
    model: str
    request_id: str | None
    usage: dict[str, Any] | None
    raw_response: dict[str, Any]


@dataclass(frozen=True)
class EzvizVisionModelClient:
    api_key: str
    base_url: str = "https://openai.ezviz.com/v1"
    model: str = "qwen3.6-plus"
    timeout_seconds: float = 30.0

    def annotate_rois(
        self,
        *,
        image_base64: str,
        mime_type: str,
        scene_hint: str,
        expected_types: list[str],
    ) -> EzvizVisionModelResult:
        if not self.api_key:
            raise EzvizVisionModelError("auth", "EZVIZ_LLM_API_KEY 未配置", 503)

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是居家安全摄像头画面 ROI 标注助手。只输出 JSON，不输出 Markdown。"
                        "不要诊断老人状态，不要猜测被遮挡物体。所有坐标必须为 0..1 归一化坐标。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt(scene_hint, expected_types)},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                    ],
                },
            ],
        }

        try:
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise EzvizVisionModelError("timeout", "萤石大模型请求超时", 504) from exc
        except httpx.HTTPError as exc:
            raise EzvizVisionModelError("network", "萤石大模型网络请求失败", 503) from exc

        if response.status_code >= 400:
            raise self._http_error(response)

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise EzvizVisionModelError("parse", "萤石大模型未返回 choices", 502)
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise EzvizVisionModelError("parse", "萤石大模型返回内容格式异常", 502)
        return EzvizVisionModelResult(
            content=content,
            model=body.get("model") or self.model,
            request_id=body.get("id"),
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            raw_response=body,
        )

    def _http_error(self, response: httpx.Response) -> EzvizVisionModelError:
        message = "萤石大模型调用失败"
        try:
            error_body: Any = response.json()
            if isinstance(error_body, dict):
                error = error_body.get("error") or error_body
                if isinstance(error, dict):
                    meta = error.get("meta")
                    if isinstance(meta, dict):
                        message = str(meta.get("message") or message)
                    else:
                        message = str(error.get("message") or message)
        except ValueError:
            pass

        if response.status_code in (401, 403):
            return EzvizVisionModelError("auth", "萤石大模型鉴权失败", response.status_code)
        if response.status_code == 404:
            return EzvizVisionModelError("model_not_found", "萤石大模型不存在或账号未开通", response.status_code)
        if response.status_code == 429:
            category = "quota" if "余额不足" in message or "quota" in message.lower() else "rate_limit"
            return EzvizVisionModelError(category, message or "萤石大模型限流", response.status_code)
        return EzvizVisionModelError("provider", message, response.status_code)

    def _prompt(self, scene_hint: str, expected_types: list[str]) -> str:
        expected = ", ".join(expected_types) if expected_types else "按画面可见区域判断"
        return (
            "请基于这张摄像头抽帧图片生成一次性 ROI 标注草稿。"
            f"场景提示：{scene_hint or '未提供'}。"
            f"期望关注类型：{expected}。"
            "固定 type 枚举：bed, sofa, dining_table, doorway, bathroom_entrance, "
            "high_risk_passage, activity_area, ignore_area, unknown。"
            "床、沙发、餐桌用于场景语义；门口、卫生间门口、高风险通道用于风险解释；"
            "activity_area 表示适合做人移动/停留统计的地面区域；床/沙发主体默认不计入行走活动统计。"
            "不要标注看不清或被遮挡后无法确认的物体，可写入 missing_expected 或 warnings。"
            "严格输出 JSON，结构为："
            "{\"schema_version\":\"roi_annotation_v1\",\"image\":{\"coordinate_system\":\"normalized_0_1\"},"
            "\"rois\":[{\"roi_id\":\"roi_bed_01\",\"type\":\"bed\",\"label_zh\":\"床\","
            "\"room_id\":\"bedroom\",\"zone_id\":\"bed_01\",\"shape\":{\"type\":\"polygon\","
            "\"points\":[{\"x\":0.1,\"y\":0.2},{\"x\":0.4,\"y\":0.2},{\"x\":0.4,\"y\":0.6}],"
            "\"bbox\":{\"x1\":0.1,\"y1\":0.2,\"x2\":0.4,\"y2\":0.6}},"
            "\"confidence\":0.86,\"activity_stats\":{\"allowed\":false,\"use\":\"exclude_resting\"},"
            "\"risk\":{\"is_high_risk\":false,\"risk_tags\":[]},\"quality_flags\":[],"
            "\"evidence\":\"床沿和枕头清晰可见\"}],\"missing_expected\":[],\"warnings\":[],"
            "\"needs_human_review\":true}"
        )
