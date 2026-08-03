# ASR 与认知模块开发协作文档 V1.0

更新时间：2026-08-03

本文说明比赛首阶段 ASR、认知功能变化线索模块和业务后端之间的工程关系。重点是让现有 Paraformer 能以稳定接口被认知模块调用，不延伸到模型训练、业务存储或主动任务流程。

## 1. 当前实现结论

当前工程保留两种调用方式，但用途已经固定：

```text
模块测试/内部工具：modules.asr.api.transcribe -> 进程级 Paraformer 单例
比赛部署：认知算法服务 -> 独立 ASR FastAPI -> 进程级 Paraformer 单例
```

比赛首阶段的 ASR 部署入口确定为 `elderly_monitoring.modules.asr.api:app`，由专用 `eldercare-asr` 环境运行。该环境不启动完整算法服务 `elderly_monitoring.service.app:app`。认知算法服务只接触版本化 ASR HTTP 接口，不接触 FunASR、模型权重或推理实现；业务后端仍只调用认知接口。同进程 Python 入口保留给测试和算法内部工具。

已落地的代码包括：

```text
src/elderly_monitoring/modules/asr/                         ASR 独立模块
src/elderly_monitoring/modules/mental_health/
  feature_extraction/cognitive_tasks/asr_adapter.py          认知侧薄适配器
src/elderly_monitoring/modules/asr/api.py                    V3.3 Python 入口与独立 HTTP 服务
src/elderly_monitoring/service/app.py                        完整算法服务兼容路由
tests/test_asr_*.py                                          契约与工程测试
tests/test_mental_health_cognitive_asr_adapter.py             认知接入测试
```

## 2. 模块分工

| 组件 | 当前关注内容 | 当前不涉及的内容 |
|---|---|---|
| 业务后端 | 提供可读取的临时音频 URL 或 base64 音频；调用认知算法 HTTP 接口；关联业务请求 | ASR 服务编排、ASR 模型加载、音频特征、认知特征、认知评分 |
| ASR 模块 | 音频解码、单声道与 16 kHz 统一、VAD、中文转写、标点、分句、时间戳和质量描述 | 用户、设备、数据库、文件留存、主动任务字段、认知判断 |
| 认知模块 | 调用 ASR；读取文本、分句、时间戳和质量掩码；与音频及面部模态衔接 | FunASR 内部对象、模型路径、后端存储、任务调度 |
| 算法服务层 | Bearer 鉴权、请求模型校验、把 HTTP 请求转给 ASR facade | 重复实现 ASR 或认知算法 |

ASR 返回的是转写事实和质量信息，不生成认知障碍标签。认知模块输出的是认知功能变化线索，不表示医学诊断。

## 3. 首阶段调用链

```mermaid
flowchart LR
    S10["S10 主动小测音频"] --> B["后端媒体适配"]
    B -->|"cognitive_infer_request_v1"| C0["认知算法服务"]
    C0 -->|"asr_request_v1：URL 或 base64"| H["独立 ASR HTTP 服务"]
    H --> F["modules.asr.api.transcribe"]
    F --> P["Paraformer + FSMN-VAD + CT-Punc"]
    P --> T["ASRTranscript v1"]
    T -->|"asr_transcript_v1"| A["认知 ASR 适配器"]
    A --> C["认知文本分支与缺失模态掩码"]
```

视频通话音轨、说话人分离、ASR 微调和任务字段建模属于第二阶段方向，不影响当前接口。

## 4. Python 内部调用

单元测试或算法内部工具已经构造 `ASRRequest` 时，可以按 V3.3 固定路径调用：

```python
from elderly_monitoring.modules.asr.api import transcribe
from elderly_monitoring.modules.asr.schemas import ASRRequest
from elderly_monitoring.modules.mental_health.feature_extraction.cognitive_tasks import (
    build_cognitive_text_input,
)

transcript = transcribe(ASRRequest.model_validate(asr_payload))
text_input = build_cognitive_text_input(transcript)
```

本地 WAV/MP3/M4A 调试可以继续使用 `transcribe_audio(path, request_id=...)`。认知侧读取规范化后的 `text`、`char_count`、`mean_confidence`、`quality_weight`、`text_available`、`asr_status` 和 `warnings`。`quality_weight` 对应 V3.3 的 `q_text`，只用于质量门控、置信度和响应说明，不作为独立认知特征。

## 5. HTTP 调用

接口：

```text
POST /v1/asr/transcribe
Authorization: Bearer <ALGORITHM_API_TOKEN>
Content-Type: application/json
```

URL 输入示例：

```json
{
  "schema_version": "asr_request_v1",
  "request_id": "asr-req-001",
  "audio_input": {
    "source_type": "url",
    "source": "<temporary-audio-url>",
    "format": "wav",
    "sample_rate": null,
    "channels": null
  },
  "language": "zh",
  "enable_vad": true,
  "enable_punctuation": true,
  "return_timestamps": true,
  "model_version": "asr-paraformer-zh-v1.0"
}
```

`format` 固定为 `wav/mp3/m4a`。`sample_rate` 和 `channels` 字段始终出现，未知时为 `null`；非空时分别限制在 8000–48000 Hz 和 1/2 声道，并与解码媒体一致。`base64` 输入使用 RFC 4648 编码，不带 `data:` 前缀。HTTP 契约不接收本地文件路径，算法内部调试入口可以接收 `Path`、路径字符串或音频字节。

输入音频时长范围为 3–60 秒，解码前大小上限为 50 MiB。URL 连接超时为 5 秒、总读取时间为 15 秒，最多跟随 2 次重定向。FFmpeg 作为 SoundFile/PCM-WAV 之外的容器解码路径，M4A/AAC 索引需要随机访问时使用单次解码生命周期内的临时文件，调用结束后自动删除。

## 6. 返回字段

```json
{
  "schema_version": "asr_transcript_v1",
  "request_id": "asr-req-001",
  "status": "completed",
  "text": "老人回答文本。",
  "segments": [
    {
      "segment_id": "seg-001",
      "start_ms": 0,
      "end_ms": 3200,
      "text": "老人回答文本。",
      "confidence": null
    }
  ],
  "language": "zh",
  "quality": {
    "audio_duration_ms": 4200,
    "speech_duration_ms": 3500,
    "speech_ratio": 0.8333,
    "mean_confidence": null,
    "decode_status": "ok",
    "vad_status": "ok"
  },
  "model": {
    "name": "funasr-paraformer-zh",
    "version": "asr-paraformer-zh-v1.0"
  },
  "warnings": [
    "confidence_unavailable",
    "speech_duration_estimated_from_timestamps"
  ]
}
```

Paraformer 当前结果不提供可直接作为概率解释的置信度，因此 `segments[].confidence` 和 `quality.mean_confidence` 保持 `null`。原始时间戳会钳制在 `[0, audio_duration_ms]`，出现越界修正时增加 `timestamp_clamped`。

当前 FunASR 组合结果没有单独暴露 VAD 区间，`speech_duration_ms` 由有效词时间戳区间合并估算，相应结果带有 `speech_duration_estimated_from_timestamps`。这一字段适合缺失模态和覆盖质量判断，不解释为模型置信概率。

状态含义：

| `status` | 调用方可采用的处理 |
|---|---|
| `completed` | 文本分支可用，继续构建认知文本输入 |
| `completed_empty_speech` | 文本分支记为缺失，音频和面部模态仍可继续 |
| `unsupported_audio` | 后端可检查音频编码；认知侧不生成空文本特征 |
| `model_unavailable` | 算法运行环境或模型包暂不可用 |
| `failed` | 本次推理未完成，其他模态仍可独立运行 |

## 7. 字段对齐

| 上游/ASR 字段 | ASR 用途 | 认知模块用途 | 进入首阶段认知模型 |
|---|---|---|---|
| `request_id` | 请求关联 | 日志和结果关联 | 否 |
| `audio_input.source` | 获取本次音频 | 不读取 | 音频本身进入音频分支 |
| `text` | 转写结果 | 文本编码器输入 | 是 |
| `segments[].text` | 分句结果 | 分句池化 | 是，可选 |
| `segments[].start_ms/end_ms` | 时间轴 | 音频文本对齐 | 是，可选 |
| `segments[].confidence` | 可获得的分句置信度 | `q_text` 的置信度项 | 只作质量门控；当前 Paraformer 为 null |
| 规范化文本 `char_count` | 不属于 ASR 响应 | `q_text` 的长度项 | 只作质量门控 |
| `status`、`warnings` | 描述处理状态 | 缺失模态降级 | 否 |
| `model.version` | 结果版本 | 复现记录 | 否 |

`task_id`、任务名称、题目、提示词、期望答案、设备 ID、用户资料、数据库 ID 和留存策略没有进入 `asr_request_v1`。这些字段也不作为首阶段认知模型特征。第二阶段若评估任务条件建模，可以在认知模块另建版本化输入，不改变 ASR 契约。

认知适配器按 V3.3 完成 NFKC、空白折叠、数字替换为字面 `<NUM>`、允许字符过滤和 `char_count`。文本质量计算为：

```text
length_score    = clip(char_count / 80, 0, 1)
mean_confidence = 有限且非空分句置信度均值；没有可用值时为 0
q_text          = 0.70*length_score + 0.30*clip(mean_confidence, 0, 1)
```

`char_count<5` 或 `q_text<0.30` 时，`text_available=false` 且文本分支不进入融合。Paraformer 当前没有提供可解释的分句置信度，因此该项按 0 处理，不用 `speech_ratio` 替代。

## 8. 运行配置

ASR 专用环境由 `environment-asr.yml` 维护，模型资产位于：

```text
models/asr/asr-paraformer-zh-v1.0/
  paraformer/
  fsmn-vad/
  ct-punc/
  sha256sums.txt
```

`environment-asr.yml` 是人工维护的直接依赖声明，`environment-asr.lock.yml` 是当前可运行环境导出的完整解析快照。锁文件保留 CUDA 12.8 PyTorch 下载源，去除本机 `prefix` 和本仓库 editable 包；环境创建完成后，项目代码通过 `python -m pip install -e . --no-deps` 接入。

模型清单使用“SHA-256、两个空格、相对路径”的固定格式，并覆盖模型目录内除 `sha256sums.txt` 自身以外的全部文件。当前清单自身 SHA-256 为 `5f3d9fc1bd67db333e60d993cec281744a5f0bfc3798c2e3dc70d925f8705b37`。FFmpeg/FFprobe 8.1.2 的版本和二进制 SHA-256 保存在 `configs/runtime/asr-native-assets.json`。独立服务 lifespan 先校验原生工具，再在模型预热时校验模型包，最后加载模型；任一校验失败时服务不会进入 ready。

可用环境变量：

| 变量 | 默认值/含义 |
|---|---|
| `ASR_MODEL_ROOT` | 上述本地模型包路径 |
| `ASR_DEVICE` | `auto`，可选 `cpu` 或 `cuda:0` |
| `ASR_TARGET_SAMPLE_RATE` | `16000` |
| `ASR_MIN_AUDIO_SECONDS` | `3` |
| `ASR_MAX_AUDIO_SECONDS` | `60` |
| `ASR_MAX_SOURCE_BYTES` | `52428800` |
| `ASR_URL_CONNECT_TIMEOUT_SECONDS` | `5` |
| `ASR_URL_TOTAL_TIMEOUT_SECONDS` | `15` |
| `ASR_URL_MAX_REDIRECTS` | `2` |
| `ASR_MEDIA_DECODE_TIMEOUT_SECONDS` | `15` |
| `ASR_FFMPEG_PATH` / `ASR_FFPROBE_PATH` | `ffmpeg` / `ffprobe` |
| `ASR_NATIVE_ASSETS_MANIFEST` | `configs/runtime/asr-native-assets.json` |
| `ASR_BATCH_SIZE_SECONDS` | `60` |
| `ASR_MERGE_LENGTH_SECONDS` | `15` |

相同配置在进程内只创建一个 `ParaformerEngine`。同进程直接调用采用首次按需加载；独立 ASR HTTP 服务在 FastAPI lifespan 启动阶段预热，ready 仅在模型加载完成后返回 200。同一引擎实例对推理调用加锁，适合当前单卡比赛演示链路。

2026-08-03 在当前机器重新构建 `eldercare-asr` 环境后的记录为：短 M4A 独立 HTTP 预热 55.346 秒，预热后推理 4.856 秒；严格 60.000 秒 M4A 的 2 个并发请求均为 `200/completed`，分别耗时 7.206 秒和 5.979 秒，总墙钟 7.207 秒，每个结果包含 11 个分段。冷加载发生在服务 ready 之前，不进入认知 V3.3 的单次 ASR 20 秒预算。

## 9. 联调方式

轻量契约测试在主算法环境运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
conda run -n eldercare-ai python -m pytest `
  tests/test_asr_schemas.py `
  tests/test_asr_audio_decode.py `
  tests/test_asr_quality.py `
  tests/test_asr_facade.py `
  tests/test_asr_integrity.py `
  tests/test_asr_service_api.py `
  tests/test_asr_standalone_api.py `
  tests/test_mental_health_cognitive_asr_adapter.py -q
```

真实 Paraformer 联调在 `eldercare-asr` 环境运行：

```powershell
conda run -n eldercare-asr python -m pip install -e . --no-deps
conda run -n eldercare-asr python scripts/smoke_test_asr.py "D:\sample\answer.wav"
conda run -n eldercare-asr python scripts/transcribe_asr.py "D:\sample\answer.wav"
conda run -n eldercare-asr python scripts/smoke_test_asr_http.py "D:\sample\answer.wav"
conda run -n eldercare-asr python scripts/smoke_test_asr_http.py "D:\sample\answer.m4a" --requests 2 --concurrency 2
```

独立服务可从 ASR 环境启动：

```powershell
$env:ALGORITHM_API_TOKEN='replace-with-local-token'
conda run -n eldercare-asr python -m uvicorn `
  elderly_monitoring.modules.asr.api:app --host 0.0.0.0 --port 8899 --workers 1
```

`scripts/start_asr_service.ps1` 封装了同一独立服务入口。`eldercare-asr` 不承担完整心理算法服务 `elderly_monitoring.service.app:app` 的依赖，因此两者不混用启动入口。单卡部署固定一个 worker；并发请求由同一进程排队并通过推理锁串行执行。

`scripts/smoke_test_asr.py` 检查本地模型资产和 FunASR 原始结果；`scripts/transcribe_asr.py` 展示版本化结果；`scripts/smoke_test_asr_http.py` 使用真实模型检查启动预热、Bearer 鉴权、HTTP 请求和并发排队。

## 10. 当前阶段边界

当前实现覆盖 3–60 秒 WAV、MP3 和 M4A，支持 SoundFile、PCM-WAV 和 FFmpeg 解码、媒体声明一致性检查、VAD、标点、分句、时间戳修正、V3.3 `q_text`、进程级单例、服务预热、Python 内部调用和独立 HTTP 服务。专项契约测试为 26 项全部通过；转换后的 AAC/M4A、严格 60 秒音频和 2 请求并发均已完成端到端转写。

S10 主动小测真实音频与设备接入并行推进，媒体到位后按相同脚本补充联调，不影响当前模型包、字段契约和服务入口。完整认知三模态模型联调仍待认知服务实现；视频通话音轨、说话人分离、流式识别、方言专项适配、ASR 微调和任务条件建模仍属于第二阶段方向。
