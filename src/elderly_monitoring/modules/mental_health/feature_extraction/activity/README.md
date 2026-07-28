# 日间活动特征工程模块

本模块属于“情绪低落/社交退缩关注模块”的日间活动特征层，只输出行为趋势指标，不输出心理疾病诊断结论。

## 功能范围

- 将摄像头结构化帧数据聚合为 10 秒活动窗口。
- 统计日间活动时长、久坐/久卧、房间转换、卧室停留占比、外出事件、起床后首次有效活动延迟、用餐时段活动和作息稳定性。
- 支持直接输入已聚合窗口，便于算法流水线复用中间结果。
- 支持算法服务接口接收 ROI 标注；当帧数据缺少 `zone/room` 时，服务层会用 ROI 给帧补齐区域信息。

## 主要入口

- `aggregate_activity_windows(records)`：帧级/秒级数据转活动窗口。
- `aggregate_daytime_activity_from_windows(windows)`：窗口数据转每日特征。
- `extract_daytime_activity_features(records)`：从帧数据到每日特征的一体化入口。
- `POST /v1/mental-health/daytime-activity`：算法服务 HTTP 接口，供后端调用。

## 输入约定

帧数据建议包含：

- `person_id`
- `timestamp` 或 `observed_at`
- `bbox`
- `keypoints`
- `zone` / `room`，缺失时可由 ROI 补齐
- `posture`
- `bbox_confidence`、`keypoint_confidence`、`tracking_confidence`

窗口数据建议包含：

- `window_start`
- `window_end`
- `person_id`
- `room`
- `zone`
- `active_score`
- `motion_state`
- `valid_detection_ratio`
- `data_quality`

## 输出字段

每日特征包含：

- `daytime_active_minutes`
- `weighted_daytime_activity`
- `valid_daytime_detection_minutes`
- `sedentary_total_minutes`
- `sedentary_bouts_count`
- `daytime_bed_stay_minutes`
- `daytime_bed_bouts_count`
- `room_transition_count`
- `bedroom_stay_ratio`
- `outdoor_event_count`
- `outdoor_minutes`
- `wake_activation_delay_minutes`
- `routine_stability_score`
- `meal_window_activity_count`
- `quality_flags`

这些字段会继续交给心理安全评分卡使用，作为“近期行为模式变化值得关注”的证据之一。
