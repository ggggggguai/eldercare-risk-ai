# 徘徊步骤 7：bbox-only adapter、Camera QC 与最小离线推理链

更新时间：2026-08-04

## 结论

步骤 7 已按《徘徊样行为识别技术方案》完成。当前唯一正式入口为 `scripts/wandering/run_camera_inference.py`，支持互斥的本地 MP4 前置跟踪模式，以及受信 `TrackObservation JSONL + wandering-media-v1 sidecar` 模式。核心验收使用后一种模式和固定合成 bbox fixture，实际串通：

```text
tracking JSONL + media sidecar
  -> 严格字段/哈希/尺寸校验
  -> bbox-bottom + bbox height
  -> 完整 scope 隔离与 2 Hz bucket
  -> Camera QC/切段/40 秒窗口
  -> bbox-height compensation + 静止保护
  -> 步骤 4 共享纯函数 + 冻结 feature stats
  -> RF/TCN primary seed 四组独立预测
  -> 原子、不可覆盖、可回放 bundle
```

本步骤没有训练或选择模型，没有读取 public benchmark 或 SmartCare official，没有融合 RF/TCN，没有接增强、校准、OOD/`uncertain`、episode、日级心理风险、`AlgorithmEvent` 或萤石鉴权/直播会话。

## 实际实现

- `configs/modules/wandering_camera_v1.yaml`：精确字段校验，固定 2 Hz、40 秒/80 bucket、20 秒 stride、Camera QC 初值、步骤 4/5/6 信任根与 `primary_seed=20260731`。配置 SHA-256 为 `08bbec6ef263dd45fef3262c407ce675107f87ea584584f7ab61142cd734ae8e`。
- `camera_adapter.py`：严格验证 `wandering-media-v1`、来源 tracking SHA、像素 bbox、时间/帧递增和完整隔离键；只消费 `frame_id/track_id/bbox/track_confidence/timestamp_sec`，不使用上游 `person_id/center/speed_px_per_sec`。`authorization_status` 只允许 `synthetic_fixture/authorized_camera_engineering_smoke/authorized_camera_labeled_evaluation`，未知值失败关闭。规范副本只保留五字段。
- `camera_qc.py`：全局 `0.0 s` 网格的置信度加权 0.5 秒 bucket、长缺口/跳跃/高度位置突变切段、40 秒窗口、短缺口插值、5 点高度滚动中位数、`[0.5,2.0]` gain、高度补偿和静止保护。候选窗范围由过滤前的合法原始观测时间覆盖决定，低置信度边缘或全低置信度轨迹仍保留候选窗并显式输出 `unavailable`。
- `camera_inference.py`：验证步骤 4 manifest/feature stats，调用 `arc_length_resample()`、`robust_isotropic_normalize()`、`build_raw_features()`、`compute_topology()`、`apply_feature_stats()` 和 `extract_features_from_points()`；RF/TCN 只经各自安全 loader 加载主 seed。仅明确的受信模型 forward 或输出校验异常降级为单窗 `inference_error`，其余数据、配置和代码契约错误终止整次构建。
- `run_camera_inference.py`：唯一离线入口。MP4 模式直接复用现有 `run_yolov8_bytetrack()`，不复制 YOLO/ByteTrack；tracking+sidecar 模式作为确定性事实输入。

`TrajectorySample v1`、公开 `PreprocessingInput`、步骤 4 bundle、RF/TCN 源码/配置/制品和 `mental_health` 顶层导出均未修改。

## schema 与失败关闭

本步骤新增并以 exact-fields 测试固定：

- `wandering-media-v1`
- `wandering-bbox-tracklet-v1`
- `wandering-camera-window-v1`
- `wandering-camera-prediction-v1`
- `wandering-camera-run-manifest-v1`

sidecar 禁止绝对路径、直播 URL 和凭据。非法 JSON/NaN、bbox 越界、hash 漂移、重复 frame、同轨时间或 frame 非递增、未知/拒绝/拼写错误的授权状态、路径逃逸、已有输出目录、official/public-test 路径和未绑定模型均在产生输出前失败。三个合法授权状态分别一一映射到 `synthetic_camera_contract`、`authorized_camera_engineering_smoke`、`authorized_camera_labeled_evaluation`，不再把任意非合成字符串描述为已授权。`camera_motion_state=moved` 使窗口不可用；`not_checked` 只写 `camera_motion_not_verified`，不冒充已自动检测相机移动。单个受信模型 forward 或输出异常只产生该窗 `inference_error`，四组概率整体为 `null`，不保留部分结果；预处理、特征或其他契约异常使整次运行失败且不提交输出目录。

## 合成 camera contract 验收

固定夹具为单一完整 scope 的 80 条合法 bbox 记录，画面 `640x480`、25 FPS、0.5 秒一条、共 40 秒，`camera_motion_state=not_checked`。它没有摄像头行为真值，只用于工程契约验收。

两次全新目录实际构建均得到：

| 计数 | 结果 |
|---|---:|
| tracking observations | 80 |
| input track scopes | 1 |
| bbox tracklets | 1 |
| 40 秒 windows | 1 |
| ready | 1 |
| unavailable | 0 |
| inference_error | 0 |

两份目录的 8 个文件集合、字节长度和 SHA-256 全部一致，manifest SHA-256 均为：

```text
fe56b2b9d394ee5e3e34e96514485a45daeb5f9ad9207ed2dbe214c86c20d98f
```

该 manifest 绑定的 7 个非自身产物为：

| 产物 | SHA-256 |
|---|---|
| `media_sidecar.json` | `c9dea423e1f2624b00432a2b0598a798173ca76d5b9fd1dfa72bcdb96c47acb3` |
| `tracking_input.jsonl` | `8a40093f9497c569db4c229389e12f54f820b23f18b7f0603e4827fa15e4d7f4` |
| `bbox_tracklets.jsonl` | `ab7d4ccca593920b8d22db2b87ca3622fcd7992c47844ecd1344aa8ba3727d9a` |
| `window_records.jsonl` | `89c93b1b84b6abf399b040d89738b6d1e191639fc1b099e5b3dcf4546b958a87` |
| `predictions.jsonl` | `e86e02df849bb0ffaf7c5da06c49123f158d4a1c26d83b1c4930a37c1af2e94f` |
| `qc_summary.json` | `9c3add82faa96138ee9b25f9eecf66cb88f1e6d8a2314ad618ed71131cb02d5e` |
| `model_bindings.json` | `984b105a5db857646b352e964e35f48e3eef755e8d600fe970a6e24a34a895a7` |

ready 记录恰有 `RF four_class / RF binary / TCN four_class / TCN binary` 四组有限、和为 1 的概率，固定 `probability_calibrated=false`、`fusion_performed=false` 和 `model_purpose=comparison_only`。由于夹具没有标签，本报告不解释类别或置信度，不计算准确率、召回率、F1 或误报率。

## 故障与数值覆盖

自动测试覆盖：

- bbox-bottom、高度和置信度加权 bucket 手算；
- video/device/setup/epoch/track 完整隔离；
- 3 个空桶等于 1.5 秒并允许插值，4 个空桶切段；
- 首尾不外推、observed/interpolated mask 和 1.0/0.5 quality；
- 第 79 桶低置信度和全低置信度轨迹仍生成 `[0,40)` 候选窗，并以 `unavailable/unobserved_window_edge` 留下审计记录；
- 5 点滚动中位数、gain 截断、平滑高度与位置并发突变；
- hard jump/`suspected_id_switch`、长缺口、显式 moved 状态；
- 静止抖动在归一化与模型调用前得到 `unavailable/insufficient_motion`；
- `[80,2]`、`[80,14]`、冻结 stats、时间通道全 0、mask/质量语义；
- 步骤 4 ready 夹具的两条 26 维特征入口逐项完全一致；
- 四个安全 loader、主 seed、外部 manifest SHA、类别顺序与前向前失败关闭；
- 授权状态 exact enum、非法/拒绝/拼写错误失败关闭，以及授权标注评估不被降级描述为工程烟测；
- unavailable 零 forward、单模型 forward/输出异常不保留部分概率，非模型契约异常终止整次构建且不落盘；
- tracking+sidecar CLI 实际构建、两次全新目录逐字节一致、原子提交和拒绝覆盖。

## 测试记录

环境门禁：

```text
Editable project location: /mnt/c/Users/lenovo/Desktop/心理算法
```

实现前 red test：三个测试文件均因对应 camera 模块不存在而收集失败，符合先写失败测试要求。2026-08-04 工程评审后又先补三类回归测试；修复前得到 `8 failed, 32 passed`，失败分别覆盖四个非法授权值、低置信度候选窗消失、模型输出原因不准确、非模型契约异常被吞，以及授权标注评估被误写为工程烟测。

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_inference.py -q
```

结果：`40 passed`。

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py -q
```

结果：`148 passed, 104 subtests passed`。唯一 warning 是当前 PyTorch 构建与本机 NVIDIA 驱动不匹配导致 CUDA 初始化 warning；步骤 6/7 协议均使用 CPU，不影响本次结果。

```bash
conda run -n eldercare-ai python -m pytest -q
```

结果：`526 passed, 1 failed, 171 subtests passed`。唯一失败为既有跌倒验收配置引用的文件缺失：

```text
data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi
```

该失败与徘徊步骤 7 无关；没有修改跌倒代码或配置规避它。

## 证据边界与下一步

本次实际验证范围仅为 `synthetic_camera_contract`。尚未验证授权目标摄像头、真实老人、真实行为目的、真实 ID switch 检出率、相机运动自动检测、摄像头分类准确率、概率校准、跨摄像头身份或临床有效性。MP4 包装入口已实现并复用既有 tracker，但当前没有授权视频，因此未运行真实 MP4 烟测，也不声称萤石直播已经联调。

步骤 7 完成后，唯一下一算法任务是步骤 8：语义保持增强、通用污染和冻结兼容性报告。通用合成污染必须继续明确不是目标摄像头实测分布。
