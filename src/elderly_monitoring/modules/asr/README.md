# ASR 模块

本目录提供独立于认知模型的中文 ASR 能力。模型权重位于 `models/asr/asr-paraformer-zh-v1.0/`，源码目录不保存模型文件。

当前文件职责：

```text
api.py          # V3.3 Python 入口与独立 FastAPI 服务
facade.py       # 解码、推理和状态映射
schemas.py      # asr_request_v1 / asr_transcript_v1
engine.py       # FunASR 进程级单例、预热与推理
audio_decode.py # WAV/MP3/M4A 解码、媒体声明校验和 16 kHz 单声道转换
quality.py      # 时间戳、语音占比和质量字段
settings.py     # 本地模型与推理配置
```

认知 V3.3 的模块级 Python 调用（仅用于单元测试和算法内部工具）：

```python
from elderly_monitoring.modules.asr.api import transcribe
from elderly_monitoring.modules.asr.schemas import ASRRequest

result = transcribe(ASRRequest.model_validate(payload))
```

比赛部署链路使用独立 HTTP 服务，入口为 `elderly_monitoring.modules.asr.api:app`，提供 `POST /v1/asr/transcribe`、Bearer 鉴权和启动预热。本地文件调试仍可使用包根导出的 `transcribe_audio()`。完整字段和协作关系见[ASR 与认知模块开发协作文档](../../../../docs/modules/mental_health/guides/ASR与认知模块开发协作文档V1.0.md)。

## 部署入口

`eldercare-asr` 是独立 ASR 运行环境，唯一服务入口为：

```text
elderly_monitoring.modules.asr.api:app
```

可使用 `scripts/start_asr_service.ps1` 启动。该脚本固定 `--workers 1`，使单张 GPU 只加载一份约 1.94 GiB 的模型，并由引擎锁串行执行推理。`elderly_monitoring.service.app:app` 属于完整算法服务，不是 `eldercare-asr` 环境的启动入口；其中保留的 `/v1/asr/transcribe` 仅用于完整算法环境的兼容适配和测试注入。

模型目录必须包含 `sha256sums.txt`。服务预热会校验文件集合和每个文件的 SHA-256，缺失、增加或内容变化都会使启动失败。清单由 `scripts/freeze_asr_package.py` 生成。FFmpeg/FFprobe 的版本与二进制哈希保存在 `configs/runtime/asr-native-assets.json`，服务启动时一并校验。

完整解析的环境快照保存在 `environment-asr.lock.yml`，日常直接依赖声明仍由 `environment-asr.yml` 维护。锁文件保留 CUDA 12.8 PyTorch 下载源，不包含本仓库 editable 包或本机 `prefix`；从锁文件创建环境后，项目代码通过 `python -m pip install -e . --no-deps` 单独接入。

当前已完成转换后 M4A 的真实转写，以及严格 60 秒 M4A 的 2 请求并发检查。S10 接入与本模块并行开发；真实 S10 音频到位后，继续使用 `scripts/smoke_test_asr_http.py` 检查设备实际编码、音质和请求排队。
