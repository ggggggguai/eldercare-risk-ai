# 跌倒 candidate-clip TCN v3 split provisional pilot

日期：2026-08-04

当前状态更新（2026-08-08）：本报告固定记录 `splitv3_f89832f6cad5b9c5f630d00b` 上的历史开发实验。当前 manifest 绑定的 v3 split 已更新为 `splitv3_e71a045eb58489f43dc5fd11`；标签与 assignment 内容虽未变化，但严格 hash/split 契约已经变化，因此下列结果不能直接作为当前 split 的 E1 复现证据。P0 门禁未通过，本次执行未重建 dataset、未重训、未读取 test。

## 结论

本轮使用当时的 v3 action labels、`splitv3_f89832f6cad5b9c5f630d00b` 和现有姿态缓存，重新构建并训练 `fall_action_presence_proxy_v1` candidate-clip TCN。三次固定 seed `42/43/44` 的 train/validation 结果已由独立 evaluator 复算一致。

这仍是“预先截取的姿态窗口是否包含明确跌倒动作”的 proxy，不是连续视频跌倒事件定位器；`event_labels_v3.jsonl` 的正式事件边界、连续背景分母和 test 指标没有被这次训练替代。方向 subtype 头按 pilot 配置保持 `not_trained`。模型不接入实时告警，`fall_state` 规则继续作为主路径和安全覆盖。

## 数据与门禁

- v3 validation：`valid=true`，`training_ready.fall_event=true`。
- source split：`splitv3_f89832f6cad5b9c5f630d00b`，保护组跨分区无泄漏。
- supervision：`D01/D02/D03/D05` 为明确跌倒动作，`A03/A05/A06/A09` 为明确非跌倒动作 proxy；未标注背景不转为负例。
- 开发 dataset：`2,533` 个窗口；train `1,834`（正/负 `1,102/732`），validation `699`（正/负 `365/334`）。
- validation 保护组：`9` 个；来源以 NTU RGB+D 为主，另有 Fall Detection 2017 和 LE2I。
- test 锁定：`684` 条 test 标签只计数；`test_pose_read=false`、`test_evaluated=false`。
- 拒绝窗口：观测帧不足 `352`、质量不足 `32`、无匹配轨迹 `1`。
- dataset SHA-256：`91ab4f332f0b63267b95ba56b911dedaa9fc5a8f6b7886ed855a6ff633077b39`。

## 三 seed 结果

指标由固定 checkpoint 在 validation 分区按固定开发阈值 `0.5` 复算；独立 evaluator 与训练器结果一致。该阈值不是 frozen 正式协议。

| seed | best epoch | epochs | Precision | Recall | F1 | Balanced accuracy | PR-AUC | ROC-AUC | confusion `[[TN,FP],[FN,TP]]` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 42 | 28 | 36 | 0.975 | 0.942 | 0.958 | 0.958 | 0.991 | 0.991 | `[[325,9],[21,344]]` |
| 43 | 34 | 40 | 0.975 | 0.948 | 0.961 | 0.960 | 0.995 | 0.993 | `[[325,9],[19,346]]` |
| 44 | 23 | 31 | 0.969 | 0.956 | 0.963 | 0.962 | 0.988 | 0.989 | `[[323,11],[16,349]]` |

三次运行均 `test_evaluated=false`，subtype 均为 `not_trained`。Fall Detection 2017 validation balanced accuracy 为 seed 42/43/44 的 `0.826/0.840/0.826`；这类分域差异和 validation 保护组数量不足，限制了总体分数的解释范围。

## 运行产物

- 数据：`data/processed/fall_risk/fall_event_proxy_v2_v3split/`
- seed 42：`reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42/`
- seed 43：`reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed43/`
- seed 44：`reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed44/`

独立 evaluator checkpoint SHA-256：

- seed 42：`fcb7d1390bc6ceff4cba5448d6eedab261ac6fad1249961c3780b8d64ccf6d2c`
- seed 43：`955539b6c025cb3f7b331eab0b67155a1a140e9897e54d9b6b8375003dba9e92`
- seed 44：`06b1cdc51d24f945f8b924bc3d05f9a214ae8a7705b01459145890f94b0dcd7b`

## 复现命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_fall_event_dataset.py \
  --labels data/annotations/fall_risk/action_labels_v3.jsonl \
  --output-dir data/processed/fall_risk/fall_event_proxy_v2_v3split \
  --allow-provisional

conda run -n eldercare-ai python scripts/train/train_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v2_v3split/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v2_v3split/metadata.json \
  --profile pilot \
  --output-dir reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42 \
  --allow-provisional

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_event_tcn.py \
  --data data/processed/fall_risk/fall_event_proxy_v2_v3split/dataset.npz \
  --metadata data/processed/fall_risk/fall_event_proxy_v2_v3split/metadata.json \
  --checkpoint reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42/best_model.pt \
  --output reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42/independent-validation.json \
  --device cpu
```

seed 43/44 使用相同数据、配置和 pilot 超参数，仅将训练 seed 改为 `43/44`；三次运行均未读取 test。

## 下一门禁

下一步不是继续追加 epoch，而是补齐连续背景、正式 onset/offset 和跨来源/老人域验证，再冻结连续事件评估协议。完成这些门禁前，candidate TCN 只能 shadow 运行，不能驱动 `fall_risk` 告警或替换规则 `fall_state`。
