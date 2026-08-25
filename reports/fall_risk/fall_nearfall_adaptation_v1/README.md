# `fall_nearfall_v1` 域内适配实验

本实验把自采居家视频的一部分加入跌倒事件 TCN 的开发训练数据，随后只在未参与训练、验证、归一化和阈值选择的留出视频上复测。它是 `development_provisional` 适配实验，不修改或替换冻结比赛发布版。

## 数据拆分

- 训练视频：57 条（`fall_f01_*` 至 `fall_f03_*`、部分 `fall_f04_*`、`fall_n01_*` 至 `fall_n04_*`）。
- 适配验证视频：14 条（部分 `fall_f04_*`、`fall_n05_*`、`fall_n06_*`）。
- 最终留出视频：44 条（`fall_f05_*`、`fall_n07_*`、`fall_n08_*`、`fall_r01_*`、`fall_r02_*`）。
- 自采物化样本：154 条，其中跌倒正例 11 条、负例 143 条；`D01/D02` 作为正例，近跌倒/正常/功能动作作为负例，`D04` 跌倒后静止延续不重复作为跌倒正例。
- 合并训练数据：7,658 个窗口，训练/验证为 5,844/1,814；留出视频未进入 `dataset.npz`。

拆分、来源哈希和泄漏约束见 [`split.json`](split.json)，物化数据见 [`dataset.npz`](dataset.npz) 和 [`samples.jsonl`](samples.jsonl)。

## 训练

使用现有连续因果 TCN 结构，从头训练三份 seed 42/43/44，再取三 seed 平均概率。训练过程指标仅用于适配过程，不作为独立效果证据：

| seed | 验证 Presence F1 | PR-AUC |
|---|---:|---:|
| 42 | 0.6506 | 0.6659 |
| 43 | 0.6776 | 0.6534 |
| 44 | 0.6659 | 0.6732 |

模型文件位于 [`seed42`](seed42)、[`seed43`](seed43)、[`seed44`](seed44)。

## 未见视频结果

比较对象使用完全相同的 44 条 holdout 视频、姿态缓存、4 秒窗口、8 FPS、0.5 秒步长和 0.5 分数阈值：

| 版本 | TP | FP | FN | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 冻结 v1 | 7 | 10 | 5 | 0.4118 | 0.5833 | 0.4828 | 0.2938 |
| 加入自采训练后 | 8 | 7 | 4 | 0.5333 | 0.6667 | 0.5926 | 0.3283 |

视频级告警方面，两版对 12/12 个跌倒视频都至少触发一次告警；冻结 v1 在 32 个无跌倒留出视频中有 1 个视频误报，适配版为 0 个。事件级指标仍有 3 个重复/边界问题，不能只用视频级 100% 代替事件 F1。

机器结果和逐视频证据：

- 适配版：[`holdout_evaluation`](holdout_evaluation)
- 冻结版同一留出集：[`frozen_holdout_evaluation`](frozen_holdout_evaluation)
- 留出真值：[`holdout_ground_truth_events.jsonl`](holdout_ground_truth_events.jsonl)
- 留出清单：[`holdout_manifest.jsonl`](holdout_manifest.jsonl)

## 结论和边界

在这批单一成年主体、单一住宅、单次采集批次的视频上，加入 11 条跌倒正例后，未见视频事件 F1 从 0.4828 提升到 0.5926，FP 从 10 降至 7，说明存在域内适配收益。但这是小样本、同域留出结果，不能证明跨主体、老人域、跨摄像头或临床泛化；近跌倒任务没有在该 holdout 中设置真值，不能据此评价近跌倒能力。该实验 release ID 为 `fall-nearfall-adaptation-v1-20260822`，不得覆盖 `fall-risk-competition-v1-20260819`。

复现数据准备：

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_fall_nearfall_adaptation.py \
  --output-dir reports/fall_risk/fall_nearfall_adaptation_v1
```

复现训练和留出评测的完整参数见各 seed 的 `config.json`、评测目录的 `contract.json`。
