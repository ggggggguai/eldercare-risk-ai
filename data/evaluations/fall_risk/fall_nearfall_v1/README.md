# fall_nearfall_v1 居家评估候选集

这是从外部 `fall_nearfall.zip` CVAT 项目导入的隔离评估候选集，不修改项目根 manifest、v2/v3 标签或现有训练 split。

## 内容

- 115 条居家视频，媒体位于用户原始目录；项目内只保存路径、媒体哈希和标注派生物，不复制约 2.5 GB 原视频。
- 272 条有效 CVAT 动作轨迹，原始 273 条中的 `U01_unable_to_judge` 被明确排除。
- 24 条跌倒事件真值：`D01/D02`（当前批次没有 `D03/D05`）。
- 27 条近跌倒事件真值：`C03/C04/C05`。
- 64 条没有上述事件真值的视频，用于正常活动、功能动作和困难负例的误报检查；未标注区间不会被自动当作负事件。
- `D04_long_static_after_fall` 保留在动作和映射事件文件中，但按跌倒后静止延续处理，不重复计为第二个跌倒事件。

## 文件

`manifest.jsonl` 绑定每条视频的绝对路径、SHA-256、FPS、帧数、分辨率、场景和单一采集主体；`action_labels.jsonl` 和 `event_labels.jsonl` 是通过项目 CVAT 转换器生成的可追溯候选标签；`ground_truth_events.jsonl` 是事件评估使用的简化真值；`assignments.jsonl` 和 `split.json` 将 115 条视频全部放在独立 `test` 候选分区；`source_annotations_redacted.zip` 是删除 CVAT 账号/邮箱节点并规范化任务名后的来源包；`import_report.json` 记录输入哈希、转换计数和门禁。

原始标注包 SHA-256：`2bdd736b0dc4bda13a1ccbbca4cf2e4737372e3bf5a7948f169514e81d37f9e5`。

## 复现导入

```bash
conda run -n eldercare-ai python scripts/annotation/import_fall_nearfall_eval.py \
  --annotations "/Users/guai/Documents/qq files/fall_nearfall.zip" \
  --media-dir "/Users/guai/Documents/qq files/居家环境自采_MVP_V1.zip/居家环境自采_MVP_V1/videos/fall_nearfall" \
  --output-dir data/evaluations/fall_risk/fall_nearfall_v1
```

输出目录已存在时导入器会拒绝覆盖，以防止无意中改变评估材料。

## 使用边界

当前 `split.json` 是 `development_provisional` 的 `test` 候选，不是已授权释放的正式 frozen test。不能用该集调阈值或选择模型后继续报告独立测试结果；一旦调参，应将其标为开发集并另建封存测试集。数据来自单名成年采集者、单一住宅和单次采集批次，不能支持跨人员、老人域或临床有效性结论。

## 当前评测结果

冻结比赛交付版本已完成一次不调参的开发性工程回放：跌倒事件 `Precision=0.2500`、`Recall=0.4167`、`F1=0.3125`、`PR-AUC=0.1647`；近跌倒规则 `Precision=0.0357`、`Recall=0.0370`、`F1=0.0364`、`PR-AUC=0.0015`。跌倒分支在视频级 23/24 个真值视频触发告警，其中 22/24 个视频至少有一条告警落在标注区间；事件级指标仍会惩罚重复告警和长时间持续阳性。结果、阈值、模型哈希、逐视频预测和限制见[`reports/fall_risk/fall_nearfall_v1/README.md`](../../../../reports/fall_risk/fall_nearfall_v1/README.md)。这些数字仍属于 `development_provisional`，不能写成正式 test 或泛化指标。
