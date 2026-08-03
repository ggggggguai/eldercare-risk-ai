# 日间活动特征工程模块

本目录同时保留旧 V1 日间行为特征和 V3.3.3 情绪与社交关注摄像头特征。两条路径用途不同，不能把 V1 的扩展字段直接提交给严格 V3 接口。两者都只输出行为观测，不输出心理疾病诊断结论。

## V3.3.3 生产格式 Python 入口

```python
from elderly_monitoring.modules.mental_health.feature_extraction.activity import (
    aggregate_mood_social_camera_daily,
    aggregate_mood_social_camera_windows,
    extract_mood_social_camera_features,
)

camera_windows = aggregate_mood_social_camera_windows(raw_pose_frames)
daily = aggregate_mood_social_camera_daily(
    camera_windows,
    camera_gait_records=gait_events,
)

# 输入也可以全部是原始帧或全部是预聚合窗口
daily = extract_mood_social_camera_features(
    raw_pose_frames_or_windows,
    camera_gait_records=gait_events,
)
```

这里的“生产格式”表示输出符合 V3.3.3 日级 Schema；它是算法侧上游 Python 入口，不表示 HTTP 生产链已经接通。`/v1/mental-health/mood-social/infer` 在 API-001/API-002 完成前仍固定返回 503。

日级返回项为：

```text
{
  "person_id": "...",
  "date": "YYYY-MM-DD",
  "activity": { ...严格 MoodSocialActivity 字段... }
}
```

完全没有该日摄像头记录时，调用方不应制造零覆盖结果，而应在 V3 日级对象中使用 `activity=null`。存在记录但没有任何有效窗口时，聚合器返回完整零覆盖对象。

## V3 窗口与有效观测

- 时区固定从 V3.3.3 配置读取，当前为 `Asia/Shanghai`。
- 活动窗口按 Unix epoch 向下取整到 10 秒，使用半开区间 `[start, start+10s)`。
- 日间范围为本地时间 `[06:00:00, 18:00:00)`；`06:00:00` 和 `17:59:50` 窗口计入，`05:59:50` 和 `18:00:00` 窗口不计入。
- 原始帧按每个时间戳最多贡献 1 秒支持区间计算覆盖并集；重复同一时刻的帧不能伪造覆盖。
- 窗口有效覆盖阈值沿用 `0.60`。还必须至少有两个不同时刻的有效样本，并能同时计算中心运动和姿态运动；缺失子分数不按剩余权重重新归一化。
- 直接输入预聚合窗口时，必须严格为 10 秒、对齐 epoch 网格，并显式提供 `valid_detection_ratio`。无效或分数缺失窗口不计观测，也不能成为低活动。

活动分数固定为：

```text
active_score =
  0.55 * center_motion_score
+ 0.30 * pose_motion_score
+ 0.10 * zone_transition_score
+ 0.05 * posture_change_score
```

多摄像头先按 `person_id + camera_id + scene_version + 10秒槽` 生成候选，再按 `person_id + 10秒槽` 选择唯一有效来源。排序固定为：

```text
identity_confidence 降序
→ pose_validity 降序
→ tracking_confidence 降序
→ camera_id、scene_version 字典序升序
```

不得按最大 `active_score` 选取，也不得求和或平均多个机位。该去重只影响活动时间线，未被选中的摄像头步态仍按场景保留。

## V3 日级活动语义

严格 `activity` 对象只包含：

- `daytime_active_minutes`
- `weighted_daytime_activity`
- `valid_daytime_detection_minutes`
- `low_activity_minutes`
- `sedentary_bout_total_minutes`
- `longest_sedentary_bout_minutes`
- `activity_peak_minute_of_day`
- `hourly_activity_intensity`
- `hourly_valid_detection_minutes`
- `camera_gait_metrics`

其中：

```text
daytime_active_minutes = Σ(10秒窗口分钟), active_score >= 0.40
low_activity_minutes = Σ(10秒窗口分钟), active_score <= 0.20
weighted_daytime_activity = Σ(active_score * 10秒窗口分钟)
```

连续低活动不要求沙发 ROI 或坐姿。缺测槽、无效槽、非低活动槽和时间空档都会立即中断；片段达到 180 个连续槽、即 30 分钟时，整个片段才进入 `sedentary_bout_total_minutes` 和最长片段。

逐小时强度由去重后的同一组有效窗口计算：

```text
hourly_activity_intensity[h] =
    Σ(active_score * valid_window_minutes) / Σ(valid_window_minutes)
```

小时无覆盖时强度为 `null`；06:00–18:00 外覆盖固定为 0、强度固定为 `null`。活动峰值按本地分钟累计 `active_score * window_seconds / 60`，取质量最大的分钟，完全并列时取最早分钟；全天有效分数均为 0 时峰值为 `null`。

存在记录但日间有效覆盖为 0 时，`valid_daytime_detection_minutes=0`；`daytime_active_minutes`、`weighted_daytime_activity`、`low_activity_minutes`、两个连续片段字段和峰值均为 `null`，小时覆盖为 24 个 0，小时强度为 24 个 `null`，步态数组为空。存在有效观测时，真实静止和未达到 30 分钟片段分别保留数值 0。

MH-003 的 7 日映射固定只对 D-6 至 D 聚合：`active_ratio = Σ daytime_active_minutes / Σ valid_daytime_detection_minutes`，`sedentary_ratio = Σ low_activity_minutes / Σ valid_daytime_detection_minutes`。`sedentary_bout_total_minutes` 不是 `sedentary_ratio` 的分子。零覆盖或字段为 `null` 的日不进入分子和分母；有效观测下的真实 0 进入聚合。

## 场景化步态

`camera_gait_records` 必须携带 `person_id`、日期或带时区时间、`camera_id` 和 `scene_version`。日级按场景分别取中位数并稳定排序：

- `gait_speed_image_norm_per_sec`
- `sit_to_stand_duration_seconds`
- `turn_duration_seconds`
- `postural_stability`

适配器也接受旧内部原始别名 `gait_speed_norm_per_sec`，但只转换成 V3 的图像尺度字段，不会把它解释成米/秒。若输入上游已确认的步行帧，还可用 `walking_segment_id`、`track_id` 和左右髋关键点计算相邻图像归一化髋中心速度；缺髋、无效帧、时间间隔大于 1 秒、跨轨迹、跨步行片段或跨场景时不连线。

同场景日级步速、坐站耗时和转身耗时分别取有效值中位数。姿态稳定性优先使用显式日级值，其次使用步态周期稳定性，最后回退转身稳定性。CAM-001 不计算 Q10/Q90、`walking_speed_norm_camera` 或同日多场景个人归一化。MH-003 只把 D-28 至 D（历史加目标日）的原始场景日级记录按日期、`camera_id`、`scene_version` 稳定排序后放入内部 `trend_context.camera_gait_days`，不合并同日多场景；Q10/Q90、最近 14 个有效历史日选择、当日多场景归一化值的中位数均由 TREND-001 计算。

V3 路径拒绝 `daily_step_count`、`sedentary_total_minutes`、`hourly_activity_vector`、`gait_speed_mps`、`walking_speed_norm` 和 `walking_speed_norm_camera`。

## V1 兼容路径

以下入口和 `/v1/mental-health/daytime-activity` 继续保留，供旧规则评分卡、ROI 调试和历史调用使用：

- `aggregate_activity_windows(records)`
- `aggregate_daytime_activity_from_windows(windows)`
- `extract_daytime_activity_features(records)`
- `POST /v1/mental-health/daytime-activity`

V1 仍输出床区、房间转换、外出、起床激活、进餐和作息稳定等扩展字段。它不是 `mood_social_infer_request_v3.current_daily_features.activity` 的合法生产者；V3 调用必须使用本页的独立 V3 入口。
