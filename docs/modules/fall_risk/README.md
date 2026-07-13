# 跌倒风险算法模块

本模块承接 `docs/modules/fall_risk/plans/跌倒风险算法研发计划.md`。

各任务的检测、跟踪、姿态、平滑、步态、坐站、近跌倒、个体基线和风险融合模型候选，见 `docs/modules/fall_risk/plans/跌倒风险各任务模型调研与选型矩阵.md`。该文档区分当前主线、短期对照和数据充足后的增强实验，不能把候选清单理解为已实现能力。

新同事加入或开始较大改动前，建议先阅读 `docs/modules/fall_risk/guides/跌倒风险算法协作开发指南.md`。该文档汇总了当前实现状态、固定算法路线、开发注意事项、验证命令和后续优化方向。

## 算法主线

```text
视频或离线特征
  -> 人体检测与跟踪
  -> 姿态关键点
  -> 关键点质量控制与时序平滑
  -> 步态/坐站/转身/近跌倒特征
  -> 个体化行为基线
  -> 规则评分卡或轻量模型
  -> 风险事件 JSON
```

## 第一版代码目标

- 先跑通从结构化特征到风险 JSON 的闭环。
- 姿态模型、检测模型、跟踪模型后续逐步接入。
- 实时演示路径优先稳定，复杂模型只做增强实验。
- 输出是风险提示/预警事件，不是医疗诊断结果。

## 目录对应

```text
src/elderly_monitoring/modules/fall_risk/features.py
  跌倒风险特征整理

src/elderly_monitoring/modules/fall_risk/pipeline.py
  跌倒风险推理主流程

src/elderly_monitoring/modules/fall_risk/tracking.py
  人体检测与跟踪，输出人体框和轨迹

src/elderly_monitoring/modules/fall_risk/pose.py
  姿态关键点提取，输出人体骨架序列

src/elderly_monitoring/modules/fall_risk/pose_quality.py
  关键点质量控制与时序平滑，输出带质量标记的稳定姿态序列

src/elderly_monitoring/modules/fall_risk/gait.py
  步态稳定性特征提取和规则 baseline，输出 gait_risk_score

src/elderly_monitoring/modules/fall_risk/sit_stand.py
  坐站转换能力特征提取和规则 baseline，输出 sit_stand_risk_score

src/elderly_monitoring/modules/fall_risk/near_fall.py
  近跌倒事件检测规则 baseline，输出 near_fall_event_score

src/elderly_monitoring/modules/fall_risk/baseline.py
  个体化行为基线建模，输出 baseline_deviation_score

data/annotations/fall_risk/
  动作级、事件级、风险级标注

data/processed/fall_risk/
  关键点、轨迹、步态、坐站、近跌倒和个体基线特征
```

## 人体检测与姿态关键点

人体检测与跟踪：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output data/processed/fall_risk/tracks/home_01_video_1_tracks.jsonl \
  --model yolov8n.pt \
  --scene-region home
```

姿态关键点提取默认仍使用已验证的 YOLOv8-pose 后端：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output data/processed/fall_risk/poses/home_01_video_1_poses.jsonl \
  --backend yolov8-pose \
  --model yolov8n-pose.pt \
  --scene-region home
```

研发计划中的目标姿态模型是 RTMPose。当前工程已新增 RTMPose/MMPose 可选后端，输出仍复用同一套 pose JSONL 契约：`frame_id`、`person_id`、`track_id`、`bbox`、`scene_region`、`keypoints`、`pose_confidence`、`keypoint_quality` 和 `timestamp_sec`。默认坐标为 0-1 归一化坐标，增加 `--absolute-coordinates` 后输出像素坐标。

RTMPose 后端当前只做预训练权重推理，不包含训练或微调流程。若已在 `eldercare-ai` 环境安装 MMPose、MMCV 和 MMEngine，可直接运行：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output data/processed/fall_risk/poses/home_01_video_1_rtmpose_poses.jsonl \
  --backend rtmpose \
  --pose-config human \
  --device cpu \
  --scene-region home
```

也可以通过 `--pose-config` 指向具体 RTMPose config 或 MMPose 模型别名，通过 `--pose-checkpoint` 指向对应预训练权重。当前 RTMPose CLI 使用 MMPose 推理结果直接适配为统一 JSONL；若 MMPose 输出中没有稳定 `track_id`，会按帧内人体序号生成 `track_id/person_id`，后续接入 tracking JSONL 时可复用同一适配层替换为稳定轨迹 ID。若这些可选依赖未安装，脚本会在选择 `--backend rtmpose` 时抛出清晰的 `RuntimeError`，提示安装 MMPose 相关依赖；YOLOv8-pose 后端和普通姿态单元测试不受影响。

关键点质量控制与时序平滑：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose_quality.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses.jsonl \
  --output data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl
```

当前质量控制层是规则/统计 baseline，不输出 `risk_level`、`risk_score` 或 `recommended_action`。它只消费姿态 JSONL，并按 `person_id + track_id` 分组生成后续步态、坐站和近跌倒模块可复用的稳定关键点序列。

新增输出字段：

| 字段 | 含义 |
|---|---|
| `core_keypoint_quality` | 面向下游动作分析的核心关键点质量分 |
| `valid_core_count` | 有效核心关键点数量 |
| `missing_core_names` | 当前帧缺失或低置信度的核心关键点名称 |
| `quality_state` | `usable`、`missing_core`、`low_quality` 或 `low_quality_run` |
| `low_quality_run_length` | 当前连续低质量帧长度 |
| `keypoints[].valid` | 该关键点是否可作为有效观测或短缺失插值 |
| `keypoints[].source` | `observed`、`low_confidence`、`missing` 或 `interpolated` |
| `keypoints[].x_smooth` / `keypoints[].y_smooth` | 指数平滑后的坐标，不覆盖原始 `x/y` |
| `keypoints[].is_jump_outlier` | 是否触发归一化坐标异常跳变标记 |
| `window_quality` | 当前时间窗质量摘要和下游模块可用性 |

默认规则：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `min_keypoint_score` | `0.30` | 低于该置信度的关键点标记为低置信度 |
| `low_quality_threshold` | `0.45` | 低于该核心质量分的帧标记为低质量 |
| `low_quality_run_frames` | `3` | 连续低质量帧达到该长度后标记为 `low_quality_run` |
| `max_interp_gap_frames` | `2` | 最多对 1-2 帧短缺失片段做线性插值 |
| `alpha` | `0.40` | 指数平滑系数 |
| `jump_threshold_norm` | `0.18` | 核心关键点相邻帧异常跳变阈值 |
| `window_sec` | `1.0` | 窗口质量摘要默认秒级窗口 |

下游模块应优先使用 `x_smooth/y_smooth`、`valid` 和 `window_quality` 判断关键点是否适合计算步态、坐站转换或近跌倒特征；异常跳变点不会被删除，但会被标记并在平滑中降权。

## 步态稳定性分析

步态稳定性分析对应固定算法路线中的第 5 步：

```text
关键点质量控制与时序平滑
  -> 步态稳定性分析
  -> 坐站转换能力分析
```

当前实现是可解释规则/统计 baseline，不训练深度模型，也不直接输出最终 `risk_level` 或 `recommended_action`。它从 cleaned pose JSONL 中读取髋、膝、踝关键点，优先使用：

- `x_smooth` / `y_smooth`
- `valid`
- `window_quality.usable_for_gait`

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_gait.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl \
  --output data/processed/fall_risk/features/home_01_video_1_gait.jsonl
```

输出每条记录表示一个步态分析窗口，核心字段包括：

| 字段 | 含义 |
|---|---|
| `gait_risk_score` | 0-1 规则步态风险分，越高表示步态稳定性风险越高 |
| `gait_stability_features` | 中心速度稳定性、左右下肢对称性、髋部摆动、停顿/拖步迹象等中间特征 |
| `quality_coverage` | 可用帧比例、髋膝踝关键点覆盖率、插值比例、跳变数和质量不足标记 |
| `risk_factors` | 规则命中的可解释风险因子 |

当前特征包括：

- 中心点速度均值、标准差和变异系数。
- 左右踝相对髋中心的运动幅度和不对称性。
- 髋部中心相对行走路径的横向偏移。
- 停顿/犹豫比例。
- 疑似拖步或小碎步的踝部运动不足 proxy。
- 步态窗口质量覆盖率。

局限：

- 该分数尚未经过真实老人步态风险标签校准。
- 归一化坐标速度和幅度不能直接代表真实世界步速或步幅。
- 相机视角、遮挡、下肢出画和跟踪 ID 切换会显著影响特征。
- 当前输出可用于工程链路、规则 baseline 和误差分析，不能解释为临床结论。

## 坐站转换能力分析

坐站转换能力分析对应固定算法路线中的第 6 步：

```text
步态稳定性分析
  -> 坐站转换能力分析
  -> 近跌倒事件检测
```

当前实现是可解释规则/统计 baseline，不训练深度动作识别模型，也不直接输出最终 `risk_level` 或 `recommended_action`。它从 cleaned pose JSONL 中读取肩、髋、膝、踝关键点，优先使用：

- `x_smooth` / `y_smooth`
- `valid`
- `is_jump_outlier`
- `core_keypoint_quality`
- `window_quality.usable_for_sit_stand`

腕部关键点只用于“疑似支撑使用”的弱 proxy；腕部缺失不会阻断基础坐站指标计算。

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_sit_stand.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl \
  --output data/processed/fall_risk/features/home_01_video_1_sit_stand.jsonl
```

输出每条记录表示一次候选坐站转换事件，核心字段包括：

| 字段 | 含义 |
|---|---|
| `transition_type` | `sit_to_stand`、`stand_to_sit` 或 `unknown_transition` |
| `sit_stand_risk_score` | 0-1 坐站局部风险分，越高表示坐站能力下降线索越强 |
| `duration` | 起身或坐下耗时，单位秒 |
| `failed_attempts` | 疑似起身失败次数 proxy |
| `trunk_forward_angle` | 转换过程中躯干前倾角 proxy |
| `post_stand_sway` | 起身后横向摇晃程度 proxy |
| `support_usage` | 疑似借助支撑 proxy 及证据字段 |
| `stabilization_time` | 起身后站稳所需时间，单位秒 |
| `sit_stand_features` | 髋部垂直位移、腿部伸展、评分组件等中间特征 |
| `quality_coverage` | 可用帧比例、关键点覆盖率、插值比例、跳变数和质量不足标记 |
| `risk_factors` | 规则命中的可解释风险因子 |

低质量窗口不会被当成高风险：当 `window_quality.usable_for_sit_stand` 不足、核心点覆盖率不足或有效帧过少时，模块输出 `sit_stand_risk_score = 0.0`，并在 `risk_factors` 中标记 `insufficient_sit_stand_quality`。

局部风险解释建议：

| `sit_stand_risk_score` | 局部含义 |
|---:|---|
| `< 0.25` | 未见明显坐站困难 |
| `0.25 - 0.50` | 轻微坐站异常 |
| `0.50 - 0.70` | 疑似坐站困难 |
| `>= 0.70` | 明显坐站困难，需融合层重点关注 |

局限：

- 该分数尚未经过真实老人坐站风险标签校准。
- 归一化 2D 坐标不能直接代表真实世界距离、速度或角度。
- 相机视角、遮挡、下肢出画和跟踪 ID 切换会显著影响髋部高度、躯干角和摇晃指标。
- 没有家具、墙面、扶手检测或真实接触标签时，`support_usage` 只能表示“疑似支撑使用”，不能可靠判断扶了什么。
- `sit_stand_risk_score` 是坐站局部风险分，应交给后续融合层结合步态、近跌倒、个体基线和场景信号生成最终跌倒风险事件。

## 近跌倒事件检测

近跌倒事件检测对应固定算法路线中的第 7 步：

```text
坐站转换能力分析
  -> 近跌倒事件检测
  -> 个体化行为基线建模
```

当前实现是可解释规则/统计 baseline，不训练时序深度模型，不输出真正跌倒事件分数，也不直接输出最终 `risk_level` 或 `recommended_action`。它从 cleaned pose JSONL 中读取肩、髋、膝、踝关键点，优先使用：

- `x_smooth` / `y_smooth`
- `valid`
- `is_jump_outlier`
- `core_keypoint_quality`
- `window_quality.usable_for_near_fall`

腕部关键点只用于“疑似支撑接触”的弱 proxy；腕部缺失不会阻断横向失衡、快速下沉恢复、急停恢复等基础近跌倒线索识别。

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_near_fall.py \
  --input data/processed/fall_risk/poses/home_01_video_1_poses_cleaned.jsonl \
  --output data/processed/fall_risk/features/home_01_video_1_near_fall.jsonl
```

输出每条记录表示一个近跌倒候选事件或一个质量不足窗口，核心字段包括：

| 字段 | 含义 |
|---|---|
| `near_fall_event_score` | 0-1 近跌倒局部分数，越高表示短时近跌倒 proxy 线索越强 |
| `event_type` | `stumble_or_lateral_loss`、`rapid_descent_recovery`、`sudden_stop_recovery`、`support_contact_proxy`、`abnormal_crouch_recovery` 或 `unknown_near_fall` |
| `near_fall_features` | 横向速度/加速度、髋部下沉恢复、路径偏移、躯干角变化、急停恢复、支撑 proxy 和评分组件 |
| `quality_coverage` | 可用帧比例、核心点覆盖率、腕部覆盖率、插值比例、跳变数和质量不足标记 |
| `risk_factors` | 规则命中的机器可读可解释风险因子 |
| `evidence` | 结构化触发证据片段 |

低质量窗口不会被当成高风险：当 `window_quality.usable_for_near_fall` 不足、核心点覆盖率不足或有效帧过少时，模块输出 `near_fall_event_score = 0.0`、`event_type = unknown_near_fall`，并在 `risk_factors` 中标记 `insufficient_near_fall_quality`。

局部分数解释建议：

| `near_fall_event_score` | 局部含义 |
|---:|---|
| `< 0.25` | 未见明显近跌倒事件 |
| `0.25 - 0.50` | 轻微信号或弱证据，需要观察 |
| `0.50 - 0.70` | 疑似近跌倒事件，建议融合层关注 |
| `>= 0.70` | 近跌倒强证据，融合层可触发高风险逻辑 |

局限：

- 该分数尚未经过真实老人近跌倒标签校准。
- 横向速度、加速度、位移和躯干角都是归一化 2D 图像坐标 proxy，不能解释为真实世界物理量。
- 相机视角、遮挡、跟踪 ID 切换、正常转身/绕障/下蹲/挥手等动作都可能影响近跌倒 proxy。
- `support_contact_proxy` 没有家具、墙面或真实接触标签，只能作为弱证据，不能单独强判高风险。
- `near_fall_event_score` 是局部事件分数，可被 `FallRiskPipeline` 消费；近跌倒模块本身不直接调用融合流程。

## 个体化行为基线建模

个体化行为基线建模对应固定算法路线中的第 8 步：

```text
近跌倒事件检测
  -> 个体化行为基线建模
  -> 轻量风险融合模型 + 规则校准
```

当前实现是滚动均值、标准差和分位数偏离的规则/统计 baseline，不训练深度模型，也不直接输出最终 `risk_level`、`risk_score` 或 `recommended_action`。它从步态、坐站、近跌倒、活动节律和场景聚合 JSONL 中读取结构化结果，按 `person_id` 建立个人历史统计；`track_id` 只作为辅助维度记录，不会把同一老人不同轨迹误建成不同老人。

运行命令：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_baseline.py \
  --baseline-input data/processed/fall_risk/features/history_features.jsonl \
  --current-input data/processed/fall_risk/features/current_features.jsonl \
  --output data/processed/fall_risk/features/current_baseline.jsonl \
  --min-history-days 3 \
  --stable-history-days 7 \
  --min-history-records 10
```

输入来源包括：

- 步态窗口：`gait_risk_score`、`gait_stability_features.mean_center_speed_norm_per_sec`、`center_speed_cv`、`hip_lateral_sway`。
- 坐站事件：`sit_stand_risk_score`、`duration`、`failed_attempts`、`stabilization_time`。
- 近跌倒事件：`near_fall_event_score`、`event_type`。
- 活动节律和场景聚合：`nighttime_activity_count`、`activity_volume`、`scene_region`。
- 通用元数据：`person_id`、`track_id`、`timestamp` / `timestamp_sec` / `start_time` / `end_time`、`quality_coverage`。

输出每条记录表示一个按天或按小时聚合的当前观测窗口，核心字段包括：

| 字段 | 含义 |
|---|---|
| `baseline_deviation_score` | 0-1 个体基线偏离分，越高表示相对个人历史偏离越明显 |
| `baseline_features` | 当前窗口聚合后的平均步速、坐站耗时、近跌倒频率、夜间活动、活动量和场景分布 |
| `baseline_reference` | 个人历史均值、标准差、p10/p25/p50/p75/p90、样本数、场景分布等摘要 |
| `deviation_factors` | 机器可读偏离因子，如 `gait_speed_drop_from_baseline` |
| `baseline_quality` | 历史天数、历史记录数、当前质量、基线置信和低样本/低质量标记 |

当前可解释偏离因子包括：

- `gait_speed_drop_from_baseline`
- `sit_stand_duration_increase_from_baseline`
- `near_fall_frequency_increase`
- `nighttime_activity_increase`
- `activity_volume_drop`
- `scene_region_pattern_shift`
- `insufficient_baseline_history`
- `reduced_baseline_quality`

低样本量和低质量数据不会被当成高风险：历史样本不足时输出 `insufficient_baseline_history` 并限制偏离分上限；历史或当前质量不足时输出 `reduced_baseline_quality` 并降低 `baseline_quality.baseline_confidence`。这些记录可供后续融合层降权或人工复核，但不应单独解释为跌倒风险等级。

局限：

- 当前基线特征是工程 proxy，尚未经过真实老人长期数据标定。
- 平均步速、转身稳定性和活动量受相机角度、遮挡、采样策略和上游聚合方式影响。
- 场景模式变化只表示相对个人历史区域分布异常，不等同于危险场景判断。
- 本模块只输出 `baseline_deviation_score` 和解释性偏离因子，不能解释为医疗诊断结论。

## 最小可用版本输入字段

当前最小可用版本从结构化特征推理，不直接消费原始视频。视频、姿态关键点、IMU 等模态应先转成下列 0-1 风险特征：

| 字段 | 含义 |
|---|---|
| `gait_risk_score` | 步态不稳、步速/步幅异常等风险 |
| `sit_stand_risk_score` | 坐站转换困难、起身失败或耗时增加 |
| `near_fall_event_score` | 近跌倒、踉跄恢复、快速扶物等前置事件 |
| `baseline_deviation_score` | 相对个体行为基线的异常偏离 |
| `scene_risk_score` | 夜间、床边、浴室等场景风险 |
| `activity_rhythm_score` | 活动节律下降或昼夜活动模式改变 |
| `fall_event_score` | 疑似跌倒事件强触发分数 |
| `long_static_score` | 疑似跌倒后长时间静止强触发分数 |
| `keypoint_quality` | 姿态/检测质量，用于估计置信度 |
| `feature_coverage` | 上游特征覆盖率；缺省时按核心特征自动估计 |

样例输入见 `examples/features/fall_risk_sample.json`。该样例是工程验证用特征样例，不代表真实老人健康数据。

## 最小可用版本输出字段

入口函数：

```python
from elderly_monitoring.modules.fall_risk import FallRiskPipeline

event = FallRiskPipeline().predict_from_features(sample)
payload = event.to_dict()
```

输出遵循全局 `AlgorithmEvent` schema，不在本模块另建接口：

| 字段 | 含义 |
|---|---|
| `risk_score` | 0-1 风险分数，越高表示风险提示越强 |
| `risk_level` | 0-4 整数等级：0 正常，1 低风险，2 中风险，3 高风险，4 紧急风险 |
| `risk_factors` | 触发风险提示的解释性因子 |
| `confidence` | 0-1 置信度，综合关键点质量、特征覆盖率和风险强度 |
| `trigger_event` | 主触发事件，如 `near_fall`、`fall_or_long_static` |
| `recommended_action` | 建议动作编码，如 `notify_guardian`、`emergency_alert` |
| `model_version` | 当前规则评分卡版本 |

如果前端或报告需要 0-100 分或 `low / medium / high` 文案，应在展示层从现有 schema 映射，不改变算法事件接口。

## 当前算法

第一版使用可解释规则评分卡：

- 加权融合步态、坐站、近跌倒、个体基线、场景和活动节律风险。
- `near_fall_event_score >= 0.70` 触发 3 级高风险提示。
- `fall_event_score >= 0.80` 或 `long_static_score >= 0.80` 触发 4 级紧急风险提示。
- 置信度由 `keypoint_quality`、`feature_coverage` 和风险强度估计。

## 判断频率与输出策略

`FallRiskPipeline.predict_from_features()` 仍保留单次特征推理 API；实时入口已经实现内存滑窗调度，使用以下策略：

| 场景 | 策略 |
|---|---|
| 轨迹和姿态关键点 | 同一次 YOLOv8-pose + ByteTrack 推理生成内存对象，默认最多 8 FPS |
| 动作识别 | 使用 1-2 秒滑窗滚动判断 |
| 风险融合 | 默认每 2 秒更新一次内部风险状态 |
| 普通状态 | 0 级不回调 |
| 风险等级升高或触发事件变化 | 立即输出事件 JSON |
| 近跌倒、疑似跌倒、长时间静止 | 不等待节流周期，立即输出 |

HTTP 入口为 `elderly_monitoring.service.app:app`，支持创建、查询、更新地址和幂等停止单路直播会话。真实视频全生命周期烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_service_smoke.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --model yolov8n-pose.pt \
  --max-frames 30
```

当前实现未完成真实萤石平台直播联调；需要业务侧提供算法容器可直接解码的短期直播地址和可接收回调的 HTTP 端点。服务不会把 `ezopen` 地址转换为直播地址。

## 运行方式

```bash
conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module fall_risk \
  --input examples/features/fall_risk_sample.json
```

测试：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_pipeline.py \
  tests/test_fall_risk_tracking.py \
  tests/test_fall_risk_pose.py \
  tests/test_fall_risk_pose_quality.py \
  tests/test_fall_risk_gait.py \
  tests/test_fall_risk_sit_stand.py \
  tests/test_fall_risk_near_fall.py \
  tests/test_fall_risk_baseline.py \
  -q
```

## 局限与升级路线

当前版本没有使用真实风险标签训练，不应解释为诊断模型或临床结论。LE2I/IMViA 样例可用于跌倒检测和跟踪链路验证，但它主要提供跌倒/非跌倒事件，不等同于长期跌倒风险标签。

后续接入真实数据时，建议固定 `predict_from_features()` 的输入输出契约：

1. 从 `data/annotations/fall_risk/risk_labels.jsonl` 和 `event_labels.jsonl` 建立训练、验证、测试划分。
2. 用当前规则评分卡作为可复现 baseline，报告召回率、误报率、F1、AUC、提前预警时间和分级一致性。
3. 在不改变 `AlgorithmEvent` schema 的前提下，将内部 scorer 替换为 Logistic Regression、LightGBM 或姿态时序模型。
4. 保留 `risk_factors`，通过规则命中、特征贡献或 SHAP/重要性分数提供解释。
5. 对真实个人健康数据只保存脱敏 ID、聚合特征和授权记录，避免在样例文件和日志中写入个人身份信息。
