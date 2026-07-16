# 萤石直播 HTTP 算法服务执行计划

> 归档状态：该实施计划已完成主要工程落地，不再作为待办维护。当前接口见[跌倒风险服务后端对接说明](../../interfaces/跌倒风险算法服务后端对接说明.md)。

## 1. 给 Codex 的执行要求

本计划用于把当前跌倒风险算法原型改造成可由业务后端通过 HTTP 启动和停止的最小算法服务包。

执行时必须遵守以下规则：

1. 严格按本计划顺序实施，不扩展本计划明确排除的功能。
2. 每一步先补失败测试，再写满足测试的最小实现。
3. 修改 Python 代码后先运行对应窄范围测试，再运行完整测试集。
4. 所有 Python 和测试命令必须使用 `eldercare-ai` conda 环境，禁止使用裸 `python` 或裸 `pytest`。
5. 保留现有离线 JSONL 脚本和公开 Python API，不破坏已有 69 项测试。
6. 不切换姿态模型，不训练新模型；实时主线固定使用现有 `YOLOv8-pose + ByteTrack`。
7. 不在实时主路径用临时 JSONL 文件传递帧和窗口，模块之间使用内存对象。
8. 每完成一个任务，立即运行该任务列出的验收命令；失败时先定位原因，不跳过验证。
9. 未完成真实萤石直播验证前，不得在文档中宣称已经完成萤石平台闭环。
10. 不创建提交、不推送远端，除非用户另外明确要求。

## 2. 最终目标

交付一个 Docker 化的单路直播算法服务，业务后端能够：

1. 通过 HTTP 提交萤石开放平台返回的可播放直播地址。
2. 查询算法会话状态。
3. 更新过期的直播地址。
4. 通过 HTTP 停止算法会话并释放资源。
5. 在算法检测到风险时，通过 HTTP 回调接收标准化风险事件 JSON。

最小闭环：

```text
后端提交直播地址
  -> 算法服务拉取直播流
  -> 人体检测与跟踪
  -> 姿态关键点提取
  -> 关键点质量控制与时序平滑
  -> 步态 / 坐站 / 近跌倒分析
  -> 个体基线层
  -> 轻量规则融合
  -> 事件节流
  -> HTTP 回调后端
  -> 后端停止会话
  -> 算法释放直播流和线程
```

## 3. 固定范围

### 3.1 本次必须实现

- 单个算法服务进程最多运行一个直播会话。
- 支持 `rtsp://`、`rtmp://`、`http://` 和 `https://` 可播放地址。
- FastAPI HTTP 启动、状态查询、直播地址更新和停止接口。
- 后台线程拉流、断流有限重连和停止信号处理。
- YOLOv8-pose 单次推理同时取得人体框、ByteTrack ID 和姿态关键点。
- 内存滚动姿态窗口。
- 复用现有姿态质量、步态、坐站、近跌倒、个体基线和风险融合模块。
- 最小场景风险规则、活动量统计、疑似跌倒和倒地后静止规则。
- 风险事件去重、升级立即发送和简单冷却时间。
- HTTP 事件回调，包含超时和有限重试。
- API Bearer Token 和回调 Bearer Token。
- wheel 构建、Docker 镜像和运行说明。
- 单元测试、服务生命周期集成测试和真实视频烟测。

### 3.2 明确不实现

- 多路直播并发和单容器多会话。
- Redis、Celery、Kafka、RabbitMQ 或其他消息队列。
- 数据库、任务持久化和服务重启后自动恢复会话。
- Kubernetes、服务发现、自动扩缩容和 GPU 调度。
- 用户、角色、权限、设备管理或管理后台。
- WebSocket、前端页面、录像回放或可视化叠加。
- RTMPose 切换、模型训练、模型微调或阈值研究。
- HMAC、多租户密钥系统和证书管理；第一版只使用 Bearer Token。
- 完整临床评估、性能压测和多摄像头压力测试。
- 把 `ezopen://` 地址转换成直播地址。业务后端必须传入算法容器可直接解码的地址。

## 4. 接口契约

### 4.1 启动会话

```http
POST /v1/monitoring/sessions
Authorization: Bearer ${ALGORITHM_API_TOKEN}
Content-Type: application/json
```

请求：

```json
{
  "request_id": "start-cam-001-20260712",
  "stream_url": "https://example.invalid/live.m3u8",
  "device_id": "cam_001",
  "person_id": "elder_001",
  "scene_region": "living_room",
  "callback_url": "https://backend.example.invalid/api/algorithm-events"
}
```

成功返回 `202 Accepted`：

```json
{
  "session_id": "b7dd2db3-3877-44fd-a510-e19821f8597d",
  "status": "starting"
}
```

约束：

- `request_id`、`device_id`、`person_id`、`scene_region` 均为非空字符串。
- `stream_url` 仅允许 `rtsp`、`rtmp`、`http` 和 `https`。
- `callback_url` 仅允许 `http` 和 `https`。
- 相同 `request_id` 重复提交时返回原会话，不创建第二个线程。
- 已有非终止会话时，新的不同 `request_id` 返回 `409 Conflict`。
- 请求只负责创建后台任务，不等待模型完成首帧推理。

### 4.2 查询会话

```http
GET /v1/monitoring/sessions/{session_id}
Authorization: Bearer ${ALGORITHM_API_TOKEN}
```

返回字段固定为：

```json
{
  "session_id": "b7dd2db3-3877-44fd-a510-e19821f8597d",
  "status": "running",
  "device_id": "cam_001",
  "person_id": "elder_001",
  "started_at": "2026-07-12T14:20:00+08:00",
  "last_frame_at": "2026-07-12T14:20:03+08:00",
  "last_error": null
}
```

状态只允许：

```text
starting -> running -> reconnecting -> running
starting/running/reconnecting -> stopping -> stopped
starting/running/reconnecting -> failed
```

### 4.3 更新直播地址

```http
PUT /v1/monitoring/sessions/{session_id}/stream-url
Authorization: Bearer ${ALGORITHM_API_TOKEN}
Content-Type: application/json
```

```json
{
  "stream_url": "https://example.invalid/refreshed-live.m3u8"
}
```

行为：记录新地址，关闭当前 `VideoCapture`，由同一会话线程使用新地址重连；不更换 `session_id`，不创建第二个推理线程。

### 4.4 停止会话

```http
POST /v1/monitoring/sessions/{session_id}/stop
Authorization: Bearer ${ALGORITHM_API_TOKEN}
```

成功返回 `202 Accepted`。重复停止必须幂等，已经 `stopped` 时返回当前状态，不报错。

### 4.5 健康检查

```text
GET /health/live   进程存活即返回 200
GET /health/ready  配置可读取且模型文件存在时返回 200，否则返回 503
```

健康检查不要求 Bearer Token。

### 4.6 风险事件回调

回调使用：

```http
POST {callback_url}
Authorization: Bearer ${CALLBACK_TOKEN}
Content-Type: application/json
```

在现有 `AlgorithmEvent` 字段外增加以下服务字段：

```json
{
  "event_id": "f5f4883c-3f42-46e9-9db5-dbe2bb0040d0",
  "session_id": "b7dd2db3-3877-44fd-a510-e19821f8597d",
  "schema_version": "1.0"
}
```

同一待发送事件的重试必须复用同一个 `event_id`。HTTP `2xx` 视为成功；其他状态或网络异常最多重试 3 次，退避间隔为 0.5、1.0、2.0 秒。三次失败后记录错误并继续推理，不终止直播会话。

## 5. 运行时设计约束

### 5.1 单次视觉推理，保留分层契约

不要分别运行 `yolov8n.pt` 跟踪和 `yolov8n-pose.pt` 姿态两套实时推理。每个直播会话只创建一个 `YOLO(yolov8n-pose.pt)` 实例，并对每帧调用带 `persist=True` 的 ByteTrack 推理。

同一个 Ultralytics 结果必须按顺序适配为：

1. `TrackObservation`：人体框、`track_id`、中心点和跟踪置信度。
2. `PoseObservation`：复用同一个 `track_id`，生成姿态关键点。

这样保留“检测与跟踪 -> 姿态关键点”的逻辑层，同时避免重复模型推理和不同跟踪器产生 ID 不一致。

### 5.2 主目标选择

第一版请求中只有一个 `person_id`。每帧按以下规则选择唯一主轨迹：

1. 如果上一帧主 `track_id` 仍存在，继续使用它。
2. 否则选择画面中 bbox 面积最大的行人。
3. 把请求的长期 `person_id` 写入该主轨迹的姿态记录，Ultralytics `track_id` 只作为当前会话内跟踪 ID。
4. 主轨迹连续丢失超过配置秒数后清空窗口，重新选择，避免把两个人的姿态窗口拼接。

不实现人脸识别、ReID 或多人老人身份判断。

### 5.3 推理和窗口频率

- 从直播流持续读取帧，但默认最多以 8 FPS 送入模型，多余帧丢弃。
- 每条主轨迹保留最近 10 秒原始姿态记录。
- 每 0.5 秒运行一次姿态质量处理和局部分支分析。
- 步态和近跌倒使用现有 1-2 秒窗口配置。
- 坐站模块使用窗口内候选事件结果。
- 每 2 秒运行一次最终风险融合。
- 风险等级升高，或触发新的近跌倒、疑似跌倒事件时，立即评估并发送。

所有时间判断使用帧到达时的单调时钟；对外 `timestamp` 使用带时区的 ISO 8601 系统时间。不要依赖直播源一定提供可靠 FPS 或 PTS。

### 5.4 个体基线最小实现

- 新增可选环境变量 `BASELINE_HISTORY_PATH`，指向现有基线模块可读取的历史 JSONL。
- 文件存在时，启动会话时加载并构建对应 `person_id` 的基线。
- 文件缺失、人员无历史或历史不足时，仍经过个体基线层，输出 `baseline_deviation_score = 0.0`，并保留 `insufficient_baseline_history` 质量标记。
- 不在本次实现数据库、日级定时任务或长期历史写回。

### 5.5 缺失融合特征的最小生产逻辑

必须新增以下最小规则，不能由 API 请求方手工填写：

- `scene_risk_score`：仅由 `scene_region` 固定映射产生；映射写入配置文件，未知区域为 0。
- `activity_rhythm_score`：如果有个人历史基线，则用当前窗口活动量与基线活动量比较；没有历史时为 0，并降低 `feature_coverage`。
- `fall_event_score`：根据髋部快速下沉、躯干趋于水平、人体框中心快速下降组合产生；低质量窗口不得触发。
- `long_static_score`：只有先触发疑似跌倒，随后连续低运动达到配置时长才产生；单独静止不触发紧急事件。

这些规则是工程 baseline，文档中必须明确不是临床结论。

### 5.6 事件发送策略

- 不回调 0 级正常状态。
- `risk_level >= 1` 才允许回调。
- 风险等级升高立即回调。
- 相同 `risk_level + trigger_event + risk_factors` 在 30 秒冷却期内不重复回调。
- 风险等级降低不发送恢复事件。
- 低质量或没有主目标时不调用最终融合，也不生成“正常”事件。

## 6. 代码结构

新增文件：

```text
src/elderly_monitoring/runtime/
  __init__.py
  streaming_pose.py
  stream_reader.py
  fall_state.py
  feature_assembly.py
  realtime_fall_risk.py
  event_policy.py

src/elderly_monitoring/service/
  __init__.py
  schemas.py
  settings.py
  callback.py
  session.py
  app.py

tests/
  test_fall_risk_streaming_pose.py
  test_fall_risk_fall_state.py
  test_fall_risk_feature_assembly.py
  test_fall_risk_realtime.py
  test_fall_risk_event_policy.py
  test_fall_risk_service_callback.py
  test_fall_risk_service_session.py
  test_fall_risk_service_api.py

configs/modules/fall_risk_service.yaml
Dockerfile
.dockerignore
```

允许最小修改：

```text
src/elderly_monitoring/modules/fall_risk/pose.py
src/elderly_monitoring/modules/fall_risk/tracking.py
src/elderly_monitoring/modules/fall_risk/pipeline.py
src/elderly_monitoring/common/schemas.py
pyproject.toml
environment.yml
README.md
docs/interfaces/算法事件输出接口.md
docs/architecture/实时视频监测前后端算法链路.md
docs/modules/fall_risk/README.md
```

只在需要提取可复用的内存函数、严格校验融合输入或更新真实运行说明时修改现有文件。不要顺手重构无关模块。

## 7. 分步执行任务

### 任务 0：记录基线状态

执行：

```bash
git status --short
conda run -n eldercare-ai python -m pytest -q
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py --help
conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features --help
```

验收：完整测试通过；记录已有未提交改动，不覆盖、不删除用户现有修改。

### 任务 1：增加服务依赖、配置和严格 HTTP Schema

先写 `tests/test_fall_risk_service_api.py` 中的 Schema 测试，覆盖：

- 合法启动请求。
- 空 ID 拒绝。
- 非法直播协议拒绝。
- 非 HTTP(S) 回调拒绝。
- 缺少 Authorization 返回 `401`。

实现：

- 在 `pyproject.toml` 增加 `service` extra：`fastapi`、`uvicorn`、`httpx`。
- 同步更新 `environment.yml`，开发环境安装 `.[vision,service]`。
- 使用 Pydantic 模型定义请求和响应，不手写字典校验。
- `settings.py` 从环境变量和 `fall_risk_service.yaml` 读取配置，环境变量覆盖 YAML。

窄范围验证：

```bash
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_service_api.py -q
```

### 任务 2：提取逐帧 YOLOv8-pose + ByteTrack 后端

先写 `tests/test_fall_risk_streaming_pose.py`，使用假的 Ultralytics result，不加载真实模型，覆盖：

- 同一结果生成一致 `track_id` 的 `TrackObservation` 和 `PoseObservation`。
- 主轨迹保持策略。
- 主轨迹丢失后清空。
- 无检测、无关键点和无 track ID 时不崩溃。

实现 `StreamingPoseTracker`：

- 构造时创建一次 YOLO 模型。
- `process_frame(frame, frame_id, timestamp_sec)` 返回内存记录。
- 每个会话拥有独立实例，禁止跨会话共享 `persist=True` 跟踪状态。
- 把现有 `pose.py` 的 Ultralytics 结果适配逻辑提取成纯函数，离线入口复用该纯函数。

验证：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_pose.py \
  tests/test_fall_risk_tracking.py \
  tests/test_fall_risk_streaming_pose.py -q
```

然后运行现有真实视频姿态烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_service_pose_smoke.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5
```

### 任务 3：实现疑似跌倒和倒地静止状态机

先写 `tests/test_fall_risk_fall_state.py`，用合成姿态序列覆盖：

- 正常行走不触发。
- 只有躯干倾斜不触发。
- 快速下沉、趋于水平且质量足够时提高 `fall_event_score`。
- 疑似跌倒后持续静止才提高 `long_static_score`。
- 低质量窗口、短暂蹲下和坐下不触发紧急分数。
- 状态重置后不会沿用旧事件。

实现 `FallStateDetector`，阈值全部来自配置，不把数值散落在代码中。

验证：

```bash
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_fall_state.py -q
```

### 任务 4：实现内存特征装配和实时算法引擎

先写：

```text
tests/test_fall_risk_feature_assembly.py
tests/test_fall_risk_realtime.py
```

覆盖：

- 姿态记录按主轨迹进入 10 秒 deque。
- 到达分析间隔后调用现有 `process_pose_records()`。
- 从现有 gait、sit-stand、near-fall 结果选择当前窗口的局部分数和风险因子。
- 经过个体基线层，即使历史不足也输出质量标记。
- 场景风险由配置产生。
- 有历史和无历史两种活动量处理。
- 低质量或无主目标时不融合、不生成正常事件。
- 合法特征最终调用 `FallRiskPipeline.predict_from_features()` 生成 `AlgorithmEvent`。

实现时优先直接调用现有内存函数：

```text
process_pose_records
extract_gait_windows
extract_sit_stand_events
extract_near_fall_events
build_personal_baselines
score_baseline_deviation
FallRiskPipeline.predict_from_features
```

如果现有函数只接受文件路径，提取一个内存核心函数，让 JSONL 包装器调用它。不得在实时引擎里落盘再读取。

验证：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_pose_quality.py \
  tests/test_fall_risk_gait.py \
  tests/test_fall_risk_sit_stand.py \
  tests/test_fall_risk_near_fall.py \
  tests/test_fall_risk_baseline.py \
  tests/test_fall_risk_pipeline.py \
  tests/test_fall_risk_feature_assembly.py \
  tests/test_fall_risk_realtime.py -q
```

### 任务 5：实现事件去重和回调

先写：

```text
tests/test_fall_risk_event_policy.py
tests/test_fall_risk_service_callback.py
```

覆盖：

- 0 级不发送。
- 首个 1-4 级事件发送。
- 风险升级立即发送。
- 相同事件在冷却期内抑制。
- 冷却期后允许再次发送。
- 回调包含固定 `event_id`、`session_id` 和 `schema_version`。
- 网络错误、`5xx` 按规定重试。
- `2xx` 不重试。
- 三次失败不抛出到推理线程。
- Authorization 头正确且日志不包含 Token、直播地址。

回调使用一个会话内独立的 `httpx.Client`。停止会话时关闭 Client。

验证：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_event_policy.py \
  tests/test_fall_risk_service_callback.py -q
```

### 任务 6：实现直播读取和单会话生命周期

先写 `tests/test_fall_risk_service_session.py`，通过依赖注入使用假的 `StreamReader` 和假的实时引擎，覆盖：

- `starting -> running`。
- 启动失败进入 `failed` 并记录清晰错误。
- 连续读帧失败进入 `reconnecting`。
- 在最大重试次数内恢复后回到 `running`。
- 超过最大重试次数进入 `failed`。
- 更新 URL 后关闭旧 capture 并使用新地址。
- 停止信号使线程结束并释放 capture、模型引用和回调 Client。
- 重复停止幂等。
- 同时只能有一个非终止会话。

`StreamReader` 使用 OpenCV `VideoCapture`，设置可用的打开和读取超时；每次重连必须先 `release()`。线程循环中不得使用超过 1 秒的不可中断 `sleep`，改用 `stop_event.wait(timeout)`。

验证：

```bash
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_service_session.py -q
```

### 任务 7：实现 FastAPI 接口

补全 `tests/test_fall_risk_service_api.py`，使用假的 `SessionManager` 覆盖：

- 启动返回 `202`。
- 相同 `request_id` 幂等。
- 第二个不同会话返回 `409`。
- 查询已知和未知会话。
- 更新直播地址。
- 停止和重复停止。
- `/health/live`。
- 模型存在和不存在时 `/health/ready` 的 `200/503`。
- 未捕获异常转换成不泄露内部路径和 Token 的 `500` JSON。

应用工厂必须支持依赖注入：

```python
create_app(settings=..., session_manager=...)
```

模块级 `app` 仅用于 Uvicorn 启动。不要在导入模块时加载 YOLO 模型；模型在会话开始时加载。

验证：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_service_api.py \
  tests/test_fall_risk_service_session.py \
  tests/test_fall_risk_service_callback.py -q
```

### 任务 8：端到端离线视频服务烟测

新增一个仅用于验证的脚本：

```text
scripts/collect/run_fall_service_smoke.py
```

脚本必须复用真实 `SessionManager`、实时引擎和 API 应用，但允许把本地视频路径通过内部测试配置交给 `StreamReader`。生产 HTTP Schema 仍禁止本地文件路径。

烟测步骤：

1. 启动本地回调接收器。
2. 通过 FastAPI 测试客户端或本地 HTTP 请求创建会话。
3. 读取指定真实视频的有限帧数。
4. 等待会话进入 `running`。
5. 验证姿态、质量控制和至少一个局部分支实际执行。
6. 调用停止接口。
7. 验证会话进入 `stopped` 且资源释放。
8. 若视频没有产生风险事件，允许回调数为 0；烟测不能伪造风险结论。

运行：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_service_smoke.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --model yolov8n-pose.pt \
  --max-frames 30
```

### 任务 9：Docker 和包构建

实现：

- `Dockerfile` 使用 Python 3.11 slim 基础镜像。
- 安装 OpenCV 运行所需系统库。
- 安装项目 `.[vision,service]`。
- 创建非 root 用户。
- 模型从固定 `/models/yolov8n-pose.pt` 读取，不在容器启动时下载。
- 暴露 `8080`。
- 入口固定为单 worker：

```text
uvicorn elderly_monitoring.service.app:app --host 0.0.0.0 --port 8080 --workers 1
```

- `.dockerignore` 排除数据集、报告、Git 元数据、缓存和本地模型；构建时通过明确方式复制所需模型，或在运行时只读挂载 `/models`。二选一并在 README 中写清楚，不同时实现两套自动策略。

验证 wheel：

```bash
rm -rf /tmp/elderly_monitoring_service_dist
mkdir -p /tmp/elderly_monitoring_service_dist
conda run -n eldercare-ai python -m pip wheel . --no-deps \
  -w /tmp/elderly_monitoring_service_dist
```

验证镜像：

```bash
docker build -t elderly-monitoring-algorithm:0.2.0 .
docker run --rm \
  -p 8080:8080 \
  -e ALGORITHM_API_TOKEN=test-api-token \
  -e CALLBACK_TOKEN=test-callback-token \
  -v "$PWD/yolov8n-pose.pt:/models/yolov8n-pose.pt:ro" \
  elderly-monitoring-algorithm:0.2.0
```

另一个终端验证：

```bash
curl -fsS http://127.0.0.1:8080/health/live
curl -fsS http://127.0.0.1:8080/health/ready
```

如果本机没有 Docker，必须说明未执行镜像验证及具体原因，不能把 wheel 构建成功替代为 Docker 验证成功。

### 任务 10：文档和最终回归

更新现有文档，只写已经实现且验证过的行为：

- `README.md`：安装、环境变量和启动命令。
- `docs/interfaces/算法事件输出接口.md`：HTTP 请求、响应和回调字段。
- `docs/architecture/实时视频监测前后端算法链路.md`：实际状态机和时序。
- `docs/modules/fall_risk/README.md`：实时入口、工程 baseline 和限制。
- `configs/modules/fall_risk.yaml`：同步 `available_stages`，不得继续漏写已实现阶段。

最终验证：

```bash
conda run -n eldercare-ai python -m pytest -q

conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_tracks_service_final.jsonl \
  --model yolov8n.pt \
  --scene-region home \
  --max-frames 5

conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_poses_service_final.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5

conda run -n eldercare-ai python scripts/collect/run_fall_service_smoke.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --model yolov8n-pose.pt \
  --max-frames 30
```

## 8. 真实萤石联调步骤

真实萤石地址和 Token 不写入仓库、测试快照或日志。由业务后端在联调环境中提供短期有效的可播放 URL。

联调按以下顺序执行：

1. 启动算法容器并确认 `/health/ready` 返回 200。
2. 启动后端回调接收接口。
3. 后端调用 `POST /v1/monitoring/sessions`。
4. 轮询状态直到 `running`，确认 `last_frame_at` 持续更新。
5. 保持直播至少 2 分钟，确认不会因短时空帧直接退出。
6. 使用 `PUT /stream-url` 替换为新的有效地址，确认会话 ID 不变且恢复出帧。
7. 调用停止接口，确认 5 秒内进入 `stopped`。
8. 检查日志中没有直播 URL、API Token、回调 Token。
9. 若测试动作触发风险，确认后端收到事件并按 `event_id` 去重；若没有触发，不制造假事件。

真实联调无法自动完成时，Codex 必须停在这里，列出需要用户提供的外部条件，不得伪造通过结果。

## 9. 完成定义

只有同时满足以下条件才可宣称本计划完成：

- 新增和既有测试全部通过。
- 真实视频服务烟测通过启动、运行和停止全生命周期。
- wheel 构建成功并能隔离导入 FastAPI 应用。
- Docker 镜像构建成功，或明确记录由于本机缺少 Docker 而未验证。
- 后端能够通过 HTTP 启动、查询、更新 URL 和停止单路会话。
- 算法能从直播帧实际运行到最终 `AlgorithmEvent`，中间不依赖临时 JSONL 文件。
- 风险事件能够通过 HTTP 回调，失败不会终止推理线程。
- 停止后直播连接、线程和 HTTP Client 均释放。
- 文档没有把规则 baseline 描述成临床模型，也没有把未执行的萤石联调写成已完成。
