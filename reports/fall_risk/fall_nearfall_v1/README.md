# `fall_nearfall_v1` 居家模拟动作工程评测

## 结论

本报告评测冻结比赛交付版本 `fall-risk-competition-v1-20260819` 在隔离 `fall_nearfall_v1` 居家模拟动作候选集上的表现。结果是开发性工程证据，不是 frozen test、老人域泛化或临床有效性证据。

| 任务 | 真值事件 | 预测事件 | TP | FP | FN | Precision | Recall | F1 | PR-AUC | 平均检测延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 跌倒事件 | 24 | 40 | 10 | 30 | 14 | 0.2500 | 0.4167 | 0.3125 | 0.1647 | 1.3500 s |
| 近跌倒事件 | 27 | 28 | 1 | 27 | 26 | 0.0357 | 0.0370 | 0.0364 | 0.0015 | 2.0000 s* |

`*` 近跌倒只有 1 个匹配事件，延迟不具稳定统计意义。事件匹配采用项目现有一对一协议：IoU `0.5` 或 onset 误差阈值；跌倒 onset 容差 `0.25 s`，近跌倒 onset 容差 `0.5 s`。分数阈值均为 `0.5`。

跌倒分支另有一个不替代事件 F1 的视频级审计口径：32/115 个视频产生告警，其中 23/24 个跌倒真值视频产生至少一条告警，22/24 个视频的任意告警和首条告警均落在标注事件区间内；9 个无跌倒真值视频产生告警。8 个视频仍有重复告警（共 8 条重复预测）。该口径说明当前主要问题包含事件边界、持续阳性和负视频误报，不能作为事件定位效果的替代指标。

标准事件协议只匹配 10/24 个跌倒真值。其余真值中，12 个视频虽然有告警落在标注区间但未通过 onset/IoU 一对一匹配，1 个视频的告警在区间外，1 个视频没有跌倒告警。这是当前“视频级看起来能报、事件 F1 仍低”的主要原因。

按场景拆分的事件计数如下（顺序为 `TP/FP/FN`）：

| 场景 | 跌倒 | 近跌倒 |
|---|---:|---:|
| base | `0/3/0` | `0/1/6` |
| bed | `2/2/2` | `0/1/2` |
| block | `4/8/4` | `0/18/1` |
| dark | `2/2/2` | `1/1/5` |
| dining | `0/9/4` | `0/3/6` |
| hall | `2/6/2` | `0/3/6` |

误报补充：91 个无跌倒真值视频中，跌倒分支有 9 个视频产生至少一条预测，近跌倒分支有 13 个视频产生至少一条预测。由于本批次不是连续 camera-hour 采集，报告不计算 FP/hour；事件级 FP 已计入上表。

报告中的 95% bootstrap 区间仅作审计输出：本批次只有 1 个主体/来源组，cluster bootstrap 的区间退化为单点，不能解释为跨主体不确定性区间。

## 评测材料与运行口径

- 数据：115 条视频，24 条跌倒真值、27 条近跌倒真值、64 条无目标事件视频；单人、单住宅、单次采集批次。
- 输入链路：YOLOv8-Pose + ByteTrack，随后使用现有 `PoseQualityConfig` 清洗；输出缓存保存在 `pose/`。候选集声明为单人视频，评测按源帧选择 `pose_confidence` 最高的观测，质量和框面积作为确定性 tie-breaker，并将轨迹碎片重映射为 `track_id=primary`；选择前后的数量记录在逐视频 `evaluation_track` 中。
- 跌倒分支：冻结三 seed 连续因果 TCN，4 秒窗口、8 FPS、0.25 秒最大 gap、0.5 秒 stride，三 seed 平均概率，阈值 `0.5`；连续阳性窗口按 2 秒 reset gap 合并。
- 近跌倒分支：冻结 `near-fall-rule-v0.1` 规则主链，阈值 `0.5`；未接入历史近跌倒 TCN 候选。
- 事件解码：连续阳性窗口先按同一主轨迹合并为事件，再送入事件评估器；窗口阳性数不直接当作事件数。当前解码仍会把跌倒后持续阳性扩展为长事件，因此同时保留视频内告警审计口径。
- 评测状态：`development_provisional`，`test_pose_read=false`，`test_evaluated=false`。

冻结模型、配置、数据和运行参数的哈希见 [`contract.json`](contract.json)。完整机器结果见 [`metrics.json`](metrics.json)、[`metrics_fall_event.json`](metrics_fall_event.json) 和 [`metrics_near_fall_event.json`](metrics_near_fall_event.json)；逐视频失败/预测证据见 [`failures.jsonl`](failures.jsonl)、`predictions_*.jsonl` 和 `videos/`。

## 复现命令

```bash
conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_nearfall_v1.py \
  --manifest data/evaluations/fall_risk/fall_nearfall_v1/manifest.jsonl \
  --ground-truth data/evaluations/fall_risk/fall_nearfall_v1/ground_truth_events.jsonl \
  --fall-config configs/evaluation/fall_event_v1.provisional.yaml \
  --near-fall-config configs/evaluation/near_fall_event_v1.provisional.yaml \
  --release configs/modules/fall_risk_release_v1.yaml \
  --pose-model models/yolov8n-pose.pt \
  --fall-checkpoint reports/fall_risk/fall_event_continuous_tcn_v2/seed42/best_model.pt \
  --fall-checkpoint reports/fall_risk/fall_event_continuous_tcn_v2/seed43/best_model.pt \
  --fall-checkpoint reports/fall_risk/fall_event_continuous_tcn_v2/seed44/best_model.pt \
  --output-dir reports/fall_risk/fall_nearfall_v1 \
  --device cpu
```

## 限制与后续

本结果只说明当前冻结运行版本在这批居家模拟动作上的工程表现。单一成年主体、单一住宅、单次 CVAT 导出、未完成独立双审，不能支持跨主体、老人域、跨摄像头或临床结论。当前跌倒误报率高、近跌倒召回极低，应优先检查姿态/轨迹质量、事件时间边界、动作域差异和困难负例覆盖；不得在这批候选集上调阈值后继续称为独立评测。
