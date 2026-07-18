# ROI 标注模块说明

## 模块定位

ROI 标注模块属于算法服务能力，不属于业务后端。业务后端需要 ROI 草稿时，应调用算法服务 HTTP 接口，由算法服务完成图片处理、多模态大模型调用和结构化 ROI 校验。

当前接口是无状态的：算法服务不保存老人、设备、ROI 版本，也不管理 `draft/confirmed/rejected/superseded`。这些业务状态应由后端保存。算法服务只返回一次图片对应的 ROI 草稿。

## 主要能力

- 接收摄像头抽帧图片，支持 `image/jpeg`、`image/png`、`image/webp`。
- 原图不落盘，不保存 base64，只计算图片 `sha256` 并返回。
- 在内存中将图片长边压到 1280，并转为 JPEG quality 85 后送模型。
- 调用萤石开放平台 OpenAI-compatible 大模型接口。
- 要求模型只返回 JSON，并解析为统一的 `roi_annotation_v1`。
- 输出坐标统一为 `0..1` 归一化坐标。
- 校验 ROI 类型、polygon 点数、bbox、坐标范围和数量上限。

## 代码入口

- FastAPI 接口：`elderly_monitoring.service.app`
- 请求/响应 schema：`elderly_monitoring.service.schemas`
- 模型客户端：`elderly_monitoring.modules.roi_annotation.ezviz_client`
- 图片处理与编排：`elderly_monitoring.modules.roi_annotation.service`
- JSON 解析与校验：`elderly_monitoring.modules.roi_annotation.validation`

## 服务配置

算法服务通过环境变量读取萤石大模型配置：

```env
EZVIZ_LLM_API_KEY=你的萤石大模型APIKey
EZVIZ_LLM_BASE_URL=https://openai.ezviz.com/v1
EZVIZ_LLM_MODEL=qwen3.6-plus
EZVIZ_LLM_TIMEOUT_SECONDS=30
```

不要把真实 API Key 写入代码、README、配置模板或提交记录。

## HTTP 接口

### 生成 ROI 草稿

`POST /v1/roi/annotate`

鉴权方式沿用算法服务 Bearer token：

```http
Authorization: Bearer ${ALGORITHM_API_TOKEN}
```

请求体：

```json
{
  "image_base64": "...",
  "mime_type": "image/webp",
  "image_width": 658,
  "image_height": 419,
  "scene_hint": "客厅，门口在画面左侧，沙发在中间",
  "expected_types": ["sofa", "doorway", "activity_area"],
  "device_id": "camera_living_room_01"
}
```

返回示例：

```json
{
  "schema_version": "roi_annotation_v1",
  "image": {
    "width": 658,
    "height": 419,
    "coordinate_system": "normalized_0_1",
    "sha256": "..."
  },
  "rois": [
    {
      "roi_id": "roi_activity_area_01",
      "type": "activity_area",
      "label_zh": "活动统计区",
      "room_id": "living_room",
      "zone_id": "activity_01",
      "shape": {
        "type": "polygon",
        "points": [
          {"x": 0.18, "y": 0.42},
          {"x": 0.82, "y": 0.42},
          {"x": 0.9, "y": 0.92},
          {"x": 0.1, "y": 0.92}
        ],
        "bbox": {"x1": 0.1, "y1": 0.42, "x2": 0.9, "y2": 0.92}
      },
      "confidence": 0.72,
      "activity_stats": {"allowed": true, "use": "movement_and_stay"},
      "risk": {"is_high_risk": false, "risk_tags": []},
      "quality_flags": [],
      "evidence": "地面区域无遮挡，适合做人移动和停留统计"
    }
  ],
  "missing_expected": [],
  "warnings": [],
  "needs_human_review": true,
  "model": {
    "provider": "ezviz",
    "name": "qwen3.6-plus",
    "request_id": "cmpl_xxx",
    "usage": {}
  }
}
```

## ROI 类型枚举

```text
bed
sofa
dining_table
doorway
bathroom_entrance
high_risk_passage
activity_area
ignore_area
unknown
```

## 校验规则

- 返回内容必须是 JSON object。
- `rois` 必须是数组，最多 40 个。
- `type` 必须属于固定枚举。
- `shape.type` 必须是 `polygon`。
- `points` 至少 3 个点。
- 所有坐标必须在 `0..1`。
- `bbox.x1 < bbox.x2`，`bbox.y1 < bbox.y2`。
- `roi_id` 不能重复。
- 低置信度 ROI 可以返回，但会自动补充 `quality_flags=["low_confidence"]`。

## 后端如何使用

后端不应直接调用萤石大模型，也不应在 `backend/` 中实现 ROI 识别算法。推荐流程：

1. 后端从设备抽帧或接收小程序上传的抽帧图。
2. 后端调用算法服务 `POST /v1/roi/annotate`。
3. 后端保存算法服务返回的 ROI 草稿和图片 `sha256`。
4. 小程序展示草稿并允许人工修正。
5. 用户确认后，后端将该 ROI 版本标记为生效。
6. 算法服务后续进行活动统计或风险解释时，再由后端下发已确认 ROI 配置。

## 测试

在算法项目目录运行：

```bash
cd algorithm/eldercare-risk-ai-main
python -m pytest tests/test_roi_annotation.py
```

