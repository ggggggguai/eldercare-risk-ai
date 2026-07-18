# 心理健康风险算法模块

本模块输出行为与睡眠变化的工程特征和独立心理健康风险事件，用于风险预警和人工复核，不输出医学诊断。当前已完成数据适配、日级聚合、个人基线、持续异常、风险评分和离线日级 CLI。

跌倒风险与心理健康风险共享上游感知、人员身份和姿态质量数据，但分别评分并分别输出各自的 `AlgorithmEvent`。心理健康管线只产生 `module=mental_health` 的独立事件。

## 数据来源与身份前提

行为输入复用现有 YOLOv8 Pose、ByteTrack、`PoseObservation` 和姿态质量控制结果，不创建第二套人体检测、姿态模型或实时循环。每条记录必须包含上游已经绑定的非空业务 `person_id`。`track_id` 只表示单路视频内关联，不能代替 `person_id`，也不能用于跨设备合并。

单设备记录可以省略 `device_id`；进入日级时间线的同一人员记录只要出现具名设备，其他记录也必须提供 `device_id`。同一时刻的多设备记录先按有效性、姿态质量和稳定设备键保留一个来源；异步区间仍重叠时，再按同样原则选择唯一来源。重叠会产生 `overlapping_device_observations` 标记，且不会直接制造场景转移。

## 绝对时间契约

行为记录必须通过以下一种方式得到事件时间：

- 带时区的 ISO-8601 `observed_at`；
- 带时区的 ISO-8601 `session_start_time` 加有限、非负的 `timestamp_sec`。

两者同时提供时以 `observed_at` 为准，并按配置容差核对推导时间。冲突记录保留在合理观测时长中，但不会进入有效观测时长。无时区、非法时间或只有相对秒的记录不会被猜测归入某个自然日；适配器会保留不可用标记，日级聚合跳过这些记录且不把标记归到其他日期。某人员完全没有有效绝对时间时快速失败。所有分桶先转换到 `aggregation.timezone`，默认 `Asia/Shanghai`；跨午夜区间在本地午夜拆分。

事件时间取当前人员和评估日最后一条合法输入的 `observed_at`。当没有合法事件时间时，CLI 要求显式传入带时区的 `evaluation_time`；它只可作为 `insufficient_data` 事件的时间回退，不能替代可评分风险结论的事件时间，也不会读取系统当前时间。

## 行为字段与计算

输入沿用命名关键点列表，每个关键点包含 `name`、归一化 `x/y` 和 `score`。若已经经过姿态质量控制，聚合器优先使用有效的 `x_smooth/y_smooth`，并保留上游质量标记。像素坐标、非有限坐标、低质量端点和被质量控制拒绝的记录不能进入有效时长。

聚合按 `person_id + date` 输出，计算规则如下：

- `observation_seconds`：只累加同一人员、同一设备、相邻且 `0 < delta <= max_gap_seconds` 的区间；不使用首尾时间差。
- `valid_observation_seconds`：区间两端质量合格、坐标有效、未被质量控制拒绝，且共同可见核心点数达标时才累加。
- `normalized_motion_proxy`：共同可见核心点的归一化欧氏位移中位数除以区间秒数。它是图像运动代理量，不是真实米/秒。
- `activity_volume`：有效区间的 `normalized_motion_proxy * delta_seconds` 之和。
- `active_ratio`：有效活动秒数除以有效观测秒数；分母为零时为 `None`。
- `nighttime_activity_ratio`：夜间有效活动秒数除以夜间有效观测秒数；默认夜间为本地时间 `[22:00, 06:00)`。
- `observation_coverage`：有效观测秒数除以合理观测秒数，不等同于后续评分阶段的模态覆盖率。
- `scene_region_distribution`：采用左端点场景的有效秒数分布；未知场景不产生转移。
- `scene_transition_count`：只统计质量有效、时间连续、无设备重叠的相邻记录场景变化。

当分母为零时，相应比例返回 `None` 并添加机器可读质量标记；缺失数据不填成正常值 `0`。

## 日间活动特征工程

`feature_extraction.activity` 已提供 V1 日间活动工程入口，面向“情绪低落/社交退缩关注”模块使用。它不训练新的心理模型，而是把摄像头结构化结果先聚合为 10 秒活动窗口，再聚合为日间行为特征。

Python 调用入口：

```python
from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_activity_windows,
    aggregate_daytime_activity_from_windows,
    extract_daytime_activity_features,
)

windows = aggregate_activity_windows(frame_or_second_records)
daily_activity = aggregate_daytime_activity_from_windows(windows, sleep_records=sleep_records)
```

帧/秒级输入可包含 `person_id`、带时区 `timestamp` 或 `observed_at`、`camera_id`、`bbox`、`bbox_confidence`、`keypoints`、`keypoint_confidence`、`zone`、`room`、`posture` 和 `tracking_confidence`。`bbox` 默认按 `[x, y, width, height]` 解释；如输入为 `[x1, y1, x2, y2]`，需显式传入 `bbox_format: "xyxy"`。

10 秒窗口输出 `active_score`、`motion_state`、`room`、`zone`、`posture`、`valid_detection_ratio` 和 `data_quality`。默认活动分公式为：

```text
active_score =
  0.55 * center_motion_score
+ 0.30 * pose_motion_score
+ 0.10 * zone_transition_score
+ 0.05 * posture_change_score
```

日级输出包括 `daytime_active_minutes`、`weighted_daytime_activity`、`sedentary_*`、`daytime_bed_*`、`room_transition_count`、`bedroom_stay_ratio`、`outdoor_*`、`wake_activation_delay_minutes`、`routine_stability_score` 和进餐时段相关活动字段。低质量、离线、遮挡和身份不确定窗口只进入质量标记，不会被当作低运动。

## 睡眠适配

睡眠适配只是标准字段校验，不代表已经接入萤石或其他真实设备协议。

| 字段 | 规则 |
|---|---|
| `person_id` | 非空业务人员 ID |
| `date` | 严格 `YYYY-MM-DD`；也可用带时区 `observed_at` / `timestamp` 转换 |
| `sleep_onset_latency` | 分钟，有限数值，`0-720`，可缺失 |
| `night_awakenings` | 整数，`0-100`，可缺失 |
| `sleep_efficiency` | 比例，`0.0-1.0`，可缺失；不接受或转换百分数 |
| `device_source` | 可选非空字符串 |
| `quality_score` | 可选有限数值，`0.0-1.0` |
| `quality_flags` | 可选非空字符串列表 |

缺失的三个睡眠指标保持 `None` 并写入质量标记。非法单位、范围或类型会抛出包含记录号和字段名的 `MentalHealthDataError`，不会静默裁剪。

## 可选自评输入

自评记录使用 JSON 或 JSONL，每条记录包含稳定 `person_id`、严格 `YYYY-MM-DD` 日期，或带时区的 `observed_at` / `timestamp`。可选分数字段为 `social_withdrawal_score`、`negative_affect_score` 和 `self_report_risk_score`，均必须是 `0.0-1.0` 的有限数值；`manual_emergency_flag` 必须是显式布尔值。缺失字段保持不可用，不默认成 0。

## 徘徊行为线索

认知功能变化线索模块已提供 MVP 徘徊检测规则。输入是上游跟踪输出的中心点轨迹，输出是可解释的行为线索，不包含医学诊断：

```python
from elderly_monitoring.modules.mental_health import (
    aggregate_daily_wandering,
    detect_wandering_events,
)

events = detect_wandering_events(track_points)
daily = aggregate_daily_wandering(events, history_daily_features=history_days)
```

默认规则按 120 秒窗口和 30 秒步长计算路径效率、转向次数、重复网格比例、闭环得分、长条往返比例和跟踪质量。事件分为 `wandering_candidate` 与 `wandering_event`，形态分为 `pacing`、`lapping`、`random` 和 `mixed`。低质量轨迹只记录低置信度候选，不进入确认事件。日级输出包含夜间徘徊次数、总分钟数、形态计数、连续夜间次数和个人历史基线偏离量，可作为评分卡中的安全行为线索之一。

## 配置与调用

默认配置位于 `configs/modules/mental_health.yaml`：

```yaml
aggregation:
  timezone: Asia/Shanghai
  max_gap_seconds: 5.0
  timestamp_conflict_tolerance_seconds: 1.0
  min_keypoint_quality: 0.45
  min_common_core_keypoints: 4
  active_motion_threshold: 0.02
  night_start: "22:00"
  night_end: "06:00"

baseline:
  initial_days: 3
  stable_days: 7
  max_window_days: 14
  abnormal_score_threshold: 0.6

scoring:
  thresholds:
    level_1: 0.25
    level_2: 0.45
    level_3: 0.65
  min_persistent_days_for_level_3: 3
  passive_max_level: 3
```

这些值是算法原型的工程默认值，未经临床校准。Python 调用入口如下：

```python
from elderly_monitoring.modules.mental_health import (
    MentalHealthRiskPipeline,
    adapt_sleep_records,
    aggregate_daily_behavior,
    score_daily_mental_health,
)

daily_features = aggregate_daily_behavior(behavior_records)
sleep_records = adapt_sleep_records(raw_sleep_records)
baseline_features = score_daily_mental_health(history_days, current_days)
event = MentalHealthRiskPipeline().predict_from_features(baseline_features[0])
```

## 离线日级 CLI

历史行为和当前行为必须使用 JSONL，每个非空行是一个行为对象。睡眠和自评可使用单对象 JSON、对象数组 JSON 或 JSONL。睡眠/自评中与当前 `person_id + date` 匹配的记录进入当前日，其余记录作为历史模态；提供 `evaluation_time` 时，该自然日的可选模态也可建立当前人员日。

示例文件位于 `examples/features/mental_health_*`。以下命令只运行心理健康分支并输出 `module=mental_health` 事件：

```bash
PYTHONPATH=src conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module mental_health \
  --history-behavior examples/features/mental_health_history_behavior.jsonl \
  --current-behavior examples/features/mental_health_current_behavior.jsonl \
  --sleep examples/features/mental_health_sleep.json \
  --self-report examples/features/mental_health_self_report.json \
  --output /tmp/mental_health_daily.jsonl
```

省略 `--output` 时写到 stdout。当前行为没有合法事件时间时必须增加例如 `--evaluation-time 2026-07-04T18:30:00+08:00`。旧的单特征入口仍保留：

```bash
PYTHONPATH=src conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features \
  --module mental_health \
  --input examples/features/mental_health_sample.json
```

日级输出按 `date + person_id` 排序，每个 `person_id + date` 一行：

```json
{"person_id":"p01","date":"2026-07-04","daily_features":{},"baseline_features":{},"event":{}}
```

`daily_features` 是行为、睡眠和自评的当日标准化结果；`baseline_features` 包含个人历史偏离、持续天数、覆盖率、质量与两个独立窗口；`event` 是统一 `AlgorithmEvent`，且 `module` 固定为 `mental_health`。输出不包含当前处理时间。相同输入、配置和代码产生字节级稳定 JSONL。所有文件会先完整解析、验证和计算；任一 JSONL 记录错误会报告文件、行号和字段，退出码非 0，输出路径采用临时文件原子替换，不留下部分结果。

## 个人基线与偏离分

基线按业务 `person_id` 和自然日独立建立，当前评估日不会进入自身基线。同一自然日重复输入只计为一个历史日；初始基线需要 3 个合格自然日，7 日后视为稳定，统计窗口最多保留最近 14 个合格自然日。

每个标量特征同时计算风险方向上的标准化差异、相对变化和分位数越界，取三者最大值。活动量和活跃比例只对下降评分；入睡潜伏期和夜间觉醒只对上升评分；睡眠效率只对下降评分；夜间活动比例和场景转移次数使用双侧偏离。零方差历史使用相对与绝对缩放下限，避免真实突变被零标准差掩盖。

`persistent_abnormal_days` 只累计达到异常阈值的连续、合格自然日。日历日期缺失或当日质量不合格会中断证据链，不会被当作正常日，也不会增加持续天数。`baseline_window` 记录历史参考范围，`evidence_window` 单独记录当前结论的连续异常范围。

## 缺失模态与风险评分

默认覆盖率分母只包含活动下降、睡眠扰动和规律性偏离三个预期特征，并按 YAML 中对应权重计算。可选的社交退缩、负面情感和自评分数仅在合法提供时参与风险评分；缺失值保持不可用，不以 0 分参与。风险总分只在当前可用特征之间重新归一化权重。

普通风险等级先按 YAML 阈值得到 0-3 级候选值，再依次应用覆盖率、基线成熟度和持续天数上限。纯摄像头与睡眠数据最高为 3 级。4 级只允许由达到阈值的合法自评风险分或显式 `manual_emergency_flag: true` 触发，3-4 级建议动作均为 `manual_review`。强证据不会被被动数据上限降级，但覆盖不足、基线不足和持续证据不足仍会降低置信度并写入 metadata。

没有任何可评分特征时，为兼容公共 schema 输出 `risk_score=0.0`，同时固定为 `risk_level=0`、`confidence=0.0` 和 `trigger_event=insufficient_data`；该兼容值不表示已经判断为正常。置信度表示特征覆盖、基线质量和持续性证据的支持程度，不表示医学结论正确率。

## 当前边界

- 不从摄像头推断负面情绪、社交退缩或医学状态。
- 不实现跨镜 ReID、人脸识别或身份数据库。
- 不实现真实睡眠设备协议、外部 API、数据库、推送或处置系统。
- 尚未经过真实设备数据、临床标签或临床有效性验证。

## Motor-Cognitive Gait Clues

`feature_extraction.gait_transfer` provides the cognitive-change clue module's gait feature entry points. It reuses upstream pose records and the fall-risk gait/sit-stand feature extractors, but produces independent `mental_health`-side daily features rather than fall-risk events or diagnosis labels.

Python entry points:

```python
from elderly_monitoring.modules.mental_health import (
    CognitiveGaitConfig,
    detect_turn_events,
    extract_cognitive_gait_features,
)

daily_features = extract_cognitive_gait_features(pose_records)
```

Daily output fields include `gait_speed_norm_per_sec`, optional `gait_speed_mps` when a per-scene meter scale is provided, `sit_stand_duration_seconds`, `turn_duration_seconds`, `turn_stability_score`, `gait_cycle_stability_score`, `motor_cognitive_clue_score`, event/window counts, quality flags, `diagnosis: false`, and `model_version`.

The current implementation is an engineering baseline for behavioral trend clues. It does not infer dementia, cognitive impairment, depression, or any medical diagnosis.
