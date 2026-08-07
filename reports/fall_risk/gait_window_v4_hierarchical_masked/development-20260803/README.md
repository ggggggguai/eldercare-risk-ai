# 步态层级门控与质量捷径隔离实验

日期：2026-08-03

状态：开发实验完成；没有读取 test，候选未通过替换门槛。

数据集：`data/processed/fall_risk/gait_window_v3_observable_instability/dataset.npz`，SHA-256 为 `b0fbf265528c9da1af8f371251fa9bfe2d6ccdfa156d293c10c1aafe47d2b2b4`；split 为 `splitv3_32db8736e890c9fad95a8292`。固定开发协议见 `configs/training/gait_hierarchical_v1.yaml`。

## 受控改动

本轮在不增加数据集的条件下只引入以下改动：

- `quality` 连续值不再进入 TCN 编码器，只用于可见性判断和时间池化；保留旧 checkpoint 的五通道权重形状。
- 使用现有 `action_id` 构建 walking gate：A01/B01-B04 为 walking，A02-A12 为 non-walking。
- 异常头只在 walking 窗口计算损失；最终分数为 `P(walking) * P(abnormal | walking)`。
- train-only 增强固定为左右镜像、最多 1 帧时间平移、5% 关键点丢失和标准差 0.005 的坐标扰动；增强后重新计算速度通道。
- 迁移现有五类动作预训练编码器，冻结编码器 8 epoch；其他训练参数和 frozen split 不变。

冻结协议仍只有 15 个异常训练窗口和 2 个异常验证窗口。三个 seed 均未启用 `--evaluate-test`，产物中 `test_evaluated=false` 且没有 test prediction 文件。

## 动作段级结果

| Seed | Best epoch | Threshold | BA | Precision | Recall | Specificity | F1 | FP | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 16 | 0.261 | 0.952 | 0.032 | 1.000 | 0.905 | 0.063 | 60 | 0.024 | 0.059 |
| 43 | 25 | 0.297 | 0.964 | 0.043 | 1.000 | 0.929 | 0.082 | 45 | 0.022 | 0.046 |
| 44 | 33 | 0.243 | 0.975 | 0.061 | 1.000 | 0.951 | 0.114 | 31 | 0.017 | 0.032 |

三个 seed 的 walking gate 窗口 F1 分别为 0.647、0.663、0.671，说明现有 A 类标签可以监督活动适用性。但主任务的 recall=1.0 只对应同两条 B04 验证动作段，不能解释为稳定异常步态召回。

旧 scratch seed 42 在同一极小 frozen validation 上为 FP=35、F1=0.103、Brier=0.042。新方法只有 seed 44 的 FP/F1 优于该单 seed 对照，seed 42/43 的误报更多；校准数值整体改善，但分类收益不具备跨 seed 一致性。不得挑选 seed 44 对外报告为已提升模型。

## 决策

- 保留层级门控、质量 mask-only 和物理一致增强的工程能力，供后续标签充分后重新训练。
- 三个 checkpoint 只用于开发复现，不写入 `configs/modules/fall_risk.yaml`。
- 不根据当前两条验证正例继续搜索 gate loss、增强幅度、阈值或模型容量。
- 运行时继续使用规则 fallback；下一项有效工作仍是复核现有训练标签/轨迹并从现有视频恢复被姿态质量门禁拒绝的窗口。

checkpoint SHA-256：

| Seed | SHA-256 |
|---:|---|
| 42 | `9ba8842388ca93adf0a025ce7e3ce1acdecc1ef4b50f1876f7d4cf72243f4af0` |
| 43 | `86a3df68386f2a120c7e2c22fbf3012b7e3830b2c93de21592e9a69c93ab3c84` |
| 44 | `eea1c8a162d9ac6e905d405a00bef09fdc33df5c5d178144fa2880c315422318` |

## 复现

工作树基于 commit `6049c80` 且包含未提交开发改动，因此不能只用 commit 声称完全复现。报告完成时的关键文件 SHA-256 为：

| 文件 | SHA-256 |
|---|---|
| `gait_tcn.py` | `cd237bd8318328514b8cd14184803074f517f955e7911fc21a4370f7bd5ad5d1` |
| `gait_training.py` | `eb1120452217b2ade7e7f59dfe5b7d1b761e25f46b3ff2cd72ba905204723ea4` |
| `gait_action_pretraining.py` | `fad23b5905a60fe13e0370c8e72f1c5eaf8251191d28a0d4b13a235bc691e37d` |
| `train_gait_tcn.py` | `cdb392563d66d2a1b143c7091f270785f5f2924f3843a68be4322057899fa4de` |
| `gait_hierarchical_v1.yaml` | `9cc7009e0e27272241c18a1c161a41b6c2ffde602cd73863398d3c34de0e1f49` |

每个 seed 的复现命令只替换 `{seed}`：

```bash
conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_window_v3_observable_instability/dataset.npz \
  --output-dir reports/fall_risk/gait_window_v4_hierarchical_masked/development-20260803/seed-{seed} \
  --partition-scheme frozen \
  --epochs 80 --patience 15 --batch-size 16 \
  --hidden-channels 48 --kernel-size 5 --dilations 1,2,4,8 \
  --dropout 0.20 --learning-rate 0.001 --weight-decay 0.0001 \
  --seed {seed} \
  --pretrained-checkpoint reports/fall_risk/action_pretraining_v2_semantic5/development-20260803/encoder.pt \
  --freeze-encoder-epochs 8 \
  --device cpu
```

命令故意不包含 `--evaluate-test`。
