# 坐站事件首轮 provisional 训练

日期：2026-08-03

## 结论

训练状态为 **provisional**。本轮完成的是 `sit_stand_event_presence_proxy_v1` 的 E0/E2 train/validation 闭环，并实际完成了 `sit_stand_candidate_clip_tcn_v1` pilot；它仍不是正式 `sit_stand_event_localization_v1`，不提供 test、边界 IoU、onset/offset 误差或 FP/hour 结论，也不接入实时主路径。

正式事件定位仍因显式连续背景为 0、`linked_event_id` 为 0、经确认 onset/offset 为 0 而 blocked。动作三分类、相位分割和 functional proxy 的既有门禁未改变。

## 数据与运行

最终数据版本：`data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/`

- dataset SHA-256：`5293cca410459694b37268160fcad38966558c5fb14babae1c88d66d7c0317bf`
- derived split SHA-256：`73f2e745b7a6eaeea7d289fec8dc3cd8f2f4d14261897db920422ac26655c2ed`
- train：1,646 窗口、1,574 事件、121 个保护组，正/负事件 868/706。
- validation：1,439 窗口、1,409 事件、11 个保护组，正/负事件 973/436。
- 锁定 test：1,276 条标签候选仅计数，姿态和特征未读取。
- 输入：4 FPS x 4 秒 = 16 帧，14 点，7 通道；每个事件的全部窗口权重和固定为 1。
- 拒绝：观测帧不足 562、质量不足 29、无匹配轨迹 1、标签时间内无姿态 1。

首版和 v2 数据目录是本轮构建过程中的可追溯早期产物；最终训练只使用内容完整的 v3 metadata。三个版本的 dataset 内容 hash 相同，未覆盖或删除任何目录。

## 训练命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_sit_stand_event_dataset.py \
  --output-dir data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3 \
  --allow-provisional

conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand_rule.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --output-dir reports/fall_risk/sit_stand_event_v1/e0-rule-provisional-20260803-seed42-v2 \
  --seed 42

conda run -n eldercare-ai python scripts/train/train_sit_stand_logistic.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --profile smoke \
  --output-dir reports/fall_risk/sit_stand_event_v1/smoke-logreg-provisional-20260803-seed42-v2 \
  --allow-provisional

conda run -n eldercare-ai python scripts/train/train_sit_stand_logistic.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --profile pilot \
  --output-dir reports/fall_risk/sit_stand_event_v1/pilot-logreg-provisional-20260803-seed42-v2 \
  --allow-provisional

conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --checkpoint reports/fall_risk/sit_stand_event_v1/pilot-logreg-provisional-20260803-seed42-v2/checkpoint.joblib \
  --partition validation \
  --output reports/fall_risk/sit_stand_event_v1/pilot-logreg-provisional-20260803-seed42-v2/validation_evaluation.json

conda run -n eldercare-ai python scripts/train/train_sit_stand_candidate_tcn.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --profile smoke \
  --output-dir reports/fall_risk/sit_stand_event_v1/smoke-candidate-tcn-provisional-20260803-seed42-v3 \
  --allow-provisional

conda run -n eldercare-ai python scripts/evaluate/evaluate_sit_stand_tcn.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --checkpoint reports/fall_risk/sit_stand_event_v1/smoke-candidate-tcn-provisional-20260803-seed42-v3/best_model.pt \
  --partition validation --device cpu \
  --output reports/fall_risk/sit_stand_event_v1/smoke-candidate-tcn-provisional-20260803-seed42-v3/validation_evaluation.json

smoke 通过后启动 pilot：

conda run -n eldercare-ai python scripts/train/train_sit_stand_candidate_tcn.py \
  --data data/processed/fall_risk/sit_stand_event_v1/provisional-20260803-seed42-v3/dataset.npz \
  --profile pilot \
  --output-dir reports/fall_risk/sit_stand_event_v1/pilot-candidate-tcn-provisional-20260803-seed42-v1 \
  --allow-provisional
```

## 实测结果

| 运行 | 状态 | Balanced accuracy | F1 | PR-AUC | Recall | Brier | 用时 |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 `e0-rule-provisional-20260803-seed42-v2` | provisional validation | 0.270 | 0.246 | 0.636 | 0.181 | 0.764 | 0.012 秒 |
| smoke `smoke-logreg-provisional-20260803-seed42-v2` | completed | 0.924 | 0.933 | 0.993 | 0.893 | 0.082 | 0.052 秒 |
| pilot `pilot-logreg-provisional-20260803-seed42-v2` | completed | 0.924 | 0.933 | 0.993 | 0.893 | 0.082 | 0.048 秒 |

pilot checkpoint SHA-256：`cee3d9e47dbf5a05ccd78b29016231ab32a6566db59adfa2f4e6a01ec0f57c50`。run config SHA-256：`66da6e0f3e72ded83d793929c118998c21865b601f9b5853604a355e5646f9b1`。代码 fingerprint：`7ae109d82eb062053af0de349c60e356c94bfb2f86f2512de0205a2e254f5862`。独立 evaluator 复算结果一致。pilot 峰值 RSS 为 207 MiB，run 目录约 560 KiB。

### 候选 clip TCN smoke

运行 ID：`smoke-candidate-tcn-provisional-20260803-seed42-v3`。模型为共享轻量 TCN 加 presence/direction 两个头；presence 监督为 A03/A04 对显式 hard negative，direction 只在 A03/A04 上计算。选模指标冻结为 validation presence/direction balanced accuracy 的均值。2 个 epoch 后 best epoch 为 1，用时 4.055 秒，CPU 峰值 RSS 437 MiB，参数量 2,340；test 未读取。

| 头 | validation 实测 |
|---|---:|
| presence balanced accuracy | 0.836 |
| presence F1 / PR-AUC | 0.809 / 0.978 |
| direction balanced accuracy | 0.915 |
| direction macro-F1 | 0.914 |

独立 evaluator 复算一致。checkpoint SHA-256：`d6b51d0220af546fa16f3c0af1f5fb95f45751e337598d01b4b15d20d40da8f5`；run config SHA-256：`c241f0f22a5f3c50ad4f3da82867f5bbda97b12e009daefb0fb200d2caa5d9f3`；代码 fingerprint：`e4208ee0a14a4ea65ffee02d9bf347c397b3a84bdbc9b302d58872025bbf718b`。`best_model.pt`、`last_checkpoint.pt`、history、validation predictions、失败案例、环境和 provenance 均已保存。

该 smoke 只证明 TCN 的数据、损失、方向 mask、反向传播、checkpoint 恢复和 validation 链路可运行；它不是连续事件定位结果，也不替换规则主路径。

### 候选 clip TCN pilot

运行 ID：`pilot-candidate-tcn-provisional-20260803-seed42-v1`。配置为 seed 42、CPU、40 个 epoch 上限、patience 8；在 epoch 30 early stopping，best epoch 为 22。训练用时 297.741 秒，参数量 8,068，峰值 RSS 447.766 MiB，run 目录约 980 KiB。训练和独立 evaluator 都只读取 train/validation；`test_access=false`、`test_evaluated=false`。

| 头 | validation 实测 |
|---|---:|
| presence balanced accuracy | 0.981 |
| presence F1 / PR-AUC | 0.986 / 0.999 |
| direction balanced accuracy | 0.994 |
| direction macro-F1 | 0.994 |

独立 evaluator 复算完全一致，报告保存在 `pilot-candidate-tcn-provisional-20260803-seed42-v1/independent_validation.json`。最佳 checkpoint SHA-256：`ad1d271d97b86889c88355bcdf90561dae0d779a0bea8827a5e4c24d7db6ceb1`；run config SHA-256：`5530615774f43b34403049c0031c8f8d728506655418a506c16bee1a4e6c7c8f`；代码 fingerprint：`e4208ee0a14a4ea65ffee02d9bf347c397b3a84bdbc9b302d58872025bbf718b`。共记录 29 个 validation 失败事件，其中 presence 错误 27 个、direction 错误 21 个（两者可重叠），明细位于该 run 的 `failure_cases.jsonl`。

该 pilot 的高分只说明预分段动作 clip 上的存在性/方向 proxy 可拟合；它不证明连续视频中的起止定位能力，也不能据此解锁正式 test 或替换规则主路径。

## 预估偏差

| 指标 | E2 Logistic 训练前预估 | Logistic pilot 实测 | 相对预估上界 |
|---|---:|---:|---:|
| Balanced accuracy | 0.55-0.75 | 0.924 | +0.174 |
| F1 | 0.60-0.82 | 0.933 | +0.113 |
| PR-AUC | 0.70-0.90 | 0.993 | +0.093 |
| 训练时长 | 1-30 秒 | 0.048 秒 | 更低 |
| 峰值内存 | 0.25-0.60 GiB | 208 MiB | 略低 |
| run 磁盘 | < 5 MiB | 约 560 KiB | 更低 |

准确率超出区间不构成升级证据。validation 的 1,308/1,409 个事件来自 NTU，且 `A04` 491 条零错误；Fall Detection 2017 只有 32 个事件，F1 为 0.545、Recall 为 0.500。高总体分主要反映动作 clip 中垂直运动模式容易分离，以及 validation 来源集中，不能外推连续视频定位。TCN pilot 没有被事后填入训练前 forecast；其上表数值是实测 provisional 结果，不是预估值。

## 失败案例

Logistic pilot 共保存 124 个 validation 失败事件：

- `A03 controlled_sit_down` 漏检 104 个；`A04` 漏检 0 个。
- hard-negative 误报 20 个：`A05` 6、`D01` 4、`D02` 6、`D03` 3、`D05` 1。
- 失败案例明细位于 `pilot-logreg-provisional-20260803-seed42-v2/failure_cases.jsonl`。

TCN pilot 另保存 29 个 validation 失败事件；由于多任务头的错误可以重叠，分别统计为 presence 27 个、direction 21 个，明细位于 `pilot-candidate-tcn-provisional-20260803-seed42-v1/failure_cases.jsonl`。

下一项最小行动不是继续调 E2 或读取 test，而是人工复核一小批真实连续视频，补充显式背景和真实事件 onset/offset，并冻结坐站专用事件协议；门禁通过后才有资格启动 E4。
