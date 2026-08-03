# ASR 模型下载目录

本目录用于保存 ASR 推理模型，不放 ASR Python 代码。代码目录为：

```text
src/elderly_monitoring/modules/asr/
```

## 首阶段模型

| 模块 | 推荐开源模型 | 下载地址 | 放置目录 |
|---|---|---|---|
| 中文语音识别 | FunASR Paraformer | <https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch> | `models/asr/asr-paraformer-zh-v1.0/paraformer/` |
| 语音活动检测 | FSMN-VAD | <https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch> | `models/asr/asr-paraformer-zh-v1.0/fsmn-vad/` |
| 中文标点恢复 | CT-Transformer Punc | <https://www.modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large> | `models/asr/asr-paraformer-zh-v1.0/ct-punc/` |

三个模型组合为：

```text
音频解码 -> FSMN-VAD -> Paraformer -> CT-Punc -> ASRTranscript v1
```

## 开源项目与依赖

- FunASR 源码：<https://github.com/modelscope/FunASR>
- FunASR Python 包：<https://pypi.org/project/funasr/>
- ModelScope Python SDK：<https://github.com/modelscope/modelscope>
- PyTorch：<https://pytorch.org/get-started/locally/>
- torchaudio：<https://pytorch.org/audio/stable/index.html>
- SoundFile：<https://pypi.org/project/soundfile/>
- FFmpeg：<https://ffmpeg.org/download.html>

FFmpeg 用于从 S10 云端录制的 MP4 中提取音频轨；FunASR、PyTorch、torchaudio、SoundFile 用于算法侧转写和音频解码。

## 目录结构

```text
models/asr/
├── asr-paraformer-zh-v1.0/
│   ├── paraformer/    # Paraformer 中文 ASR 模型文件
│   ├── fsmn-vad/      # FSMN-VAD 模型文件
│   └── ct-punc/       # CT-Punc 标点模型文件
├── MODEL_SHA256.txt
└── README.md
```

下载后的模型目录中应直接出现模型文件，不要再套一层同名目录。例如：

```text
models/asr/asr-paraformer-zh-v1.0/paraformer/config.yaml
models/asr/asr-paraformer-zh-v1.0/paraformer/model.pt
```

模型名称、文件名和实际权重格式以 ModelScope 模型页为准。首阶段不需要下载说话人识别、声纹识别或方言专用模型。

## 本机部署状态

Windows 主路径使用独立 Conda 环境 `eldercare-asr`，环境定义位于项目根目录的 `environment-asr.yml`。这样不会改变主项目的 `eldercare-ai` 环境。

```powershell
conda env create -f environment-asr.yml
conda run -n eldercare-asr python scripts/smoke_test_asr.py "D:\揭榜挂帅专项赛数据集\数据集\心理\CMDC\part1\HC01\Q1.wav"
```

烟测脚本只从上述本地目录加载模型，并输出识别文本、时间戳以及模型加载和推理耗时。模型权重的校验值记录在 `MODEL_SHA256.txt`。

FFmpeg 用于后续从 S10 的 MP4 文件提取音轨。通过 Winget 安装后，需要打开新的 PowerShell 窗口使更新后的 `PATH` 生效：

```powershell
ffmpeg -version
```

## 可选的 Fun-ASR-Nano / vLLM 路径

本机另有一套面向吞吐量的 WSL2 部署，用于第二阶段对比，不替代当前 Windows 主路径：

```text
WSL 发行版：Ubuntu-24.04
Python 环境：/opt/eldercare-asr-vllm
FunASR 示例：/opt/FunASR/examples/industrial_data_pretraining/fun_asr_nano/serve_vllm.py
模型缓存：/root/.cache/modelscope/models/FunAudioLLM--Fun-ASR-Nano-2512
vLLM：0.19.1
```

PowerShell 启停命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_fun_asr_nano_wsl.ps1
powershell -ExecutionPolicy Bypass -File scripts/stop_fun_asr_nano_wsl.ps1
```

服务由 WSL 内的 `fun-asr-nano.service` systemd 单元管理，单元源文件位于 `deploy/systemd/fun-asr-nano.service`。启动脚本默认占用 GPU 0、端口 8899，并使用 `gpu-memory-utilization=0.5`。当前项目不使用说话人识别，启动时关闭了额外的说话人模型，以适配 8 GB 显存。

当前官方 `serve_vllm.py` 提供 `/asr`、`/v1/audio/transcriptions` 和 `/ws`，没有 `/health`。因此可用性检查使用：

```powershell
curl.exe -fsS http://localhost:8899/openapi.json
```

接口说明页面为 <http://localhost:8899/docs>。首次启动会进行 Triton 编译，等待时间明显长于后续启动。
