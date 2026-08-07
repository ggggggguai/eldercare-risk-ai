# 步态 TCN v5：六 seed 集成开发实验

日期：2026-08-03

状态：开发实验完成；六 seed 集成优于单模型，但数据仍是 functional proxy，未读取 test，未接入实时主路径。

## 目标

在已完成质量捷径隔离、walking gate 和物理一致增强的实现上继续训练，优先降低单 seed 波动。训练配置保持一致，只增加随机种子 45、46、47，并与已完成的 42、43、44 做概率集成。

## 数据与协议

- 数据：`data/processed/fall_risk/gait_window_v5_b01_b04/dataset.npz`
- 数据 SHA-256：`381d992dfc8cbc701c98cde35c111acb6492713e7b3ce1c8264be02b2182b465`
- 任务：B01-B04 observable gait instability vs A01-A12 normal activity
- 训练/验证/测试窗口：676 / 710 / 675；验证动作段为 631 个负段、12 个正段
- `quality` 只用于 mask/pooling；walking gate 使用 A01、B01-B04；最终分数为 `P(walking) * P(abnormal | walking)`
- 预训练编码器：`reports/fall_risk/action_pretraining_v3_visibility_mask/development-20260803/encoder.pt`
- 六次训练均未启用 `--evaluate-test`，没有读取 test 姿态、标签或指标

训练命令（仅替换 `{seed}`）：

```bash
conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_window_v5_b01_b04/dataset.npz \
  --output-dir reports/fall_risk/gait_window_v5_effect_first/development-20260803/b01-gate2-seed-{seed} \
  --epochs 80 --patience 15 --batch-size 16 \
  --hidden-channels 48 --kernel-size 5 --dilations 1,2,4,8 \
  --dropout 0.20 --learning-rate 0.001 --weight-decay 0.0001 \
  --walking-gate-loss-weight 2.0 --temporal-shift-frames 1 \
  --keypoint-dropout-probability 0.05 --coordinate-jitter-std 0.005 \
  --pretrained-checkpoint reports/fall_risk/action_pretraining_v3_visibility_mask/development-20260803/encoder.pt \
  --freeze-encoder-epochs 8 --seed {seed} --device cpu
```

## 单模型结果

指标均为验证动作段级别，阈值由各自 validation 选择。

| Seed | Best epoch | 阈值 | Balanced accuracy | Precision | Recall | F1 | FP | FP/小时 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 7 | 0.10496 | 0.917 | 0.103 | 1.000 | 0.186 | 105 | 203.9 |
| 43 | 5 | 0.14997 | 0.933 | 0.125 | 1.000 | 0.222 | 84 | 163.1 |
| 44 | 11 | 0.00566 | 0.888 | 0.078 | 1.000 | 0.145 | 141 | 273.8 |
| 45 | 5 | 0.14305 | 0.944 | 0.145 | 1.000 | 0.253 | 71 | 137.9 |
| 46 | 16 | 0.00669 | 0.907 | 0.093 | 1.000 | 0.170 | 117 | 227.2 |
| 47 | 19 | 0.00907 | 0.926 | 0.114 | 1.000 | 0.205 | 93 | 180.6 |

seed 45 是本轮最好的单模型，但仍有 71 个动作段误报，不能单独作为稳定改进结论。

## 六 seed 概率集成

对六个 checkpoint 的动作段概率取算术平均，再只在 validation 上选择阈值 `0.5878497064`。结果为：

| 样本数 | TN | FP | FN | TP | Balanced accuracy | Precision | Recall | F1 | 正常误报/小时 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 643 | 610 | 21 | 6 | 6 | 0.733 | 0.222 | 0.500 | 0.308 | 40.8 |

集成相对最好单模型的 F1 从 `0.253` 提升到 `0.308`，FP 从 71 降到 21；代价是 recall 从 1.0 降为 0.5。该结果只说明在当前开发 proxy 上集成降低了随机波动，不是正式泛化证据。

集成成员和 SHA-256 见 [`ensemble_manifest.json`](ensemble_manifest.json)。该 manifest 仅供离线复现，不是当前运行时 checkpoint。

## 决策与限制

- 保留六个 checkpoint 和集成 manifest，作为后续补充标签后的 warm-start/对照。
- 不写入 `configs/modules/fall_risk.yaml`，运行时继续规则 fallback，默认 checkpoint 保持 `null`。
- B01-B04 是可观察动作 proxy，不是临床诊断；验证集只有 12 个正动作段，阈值和 F1 对少数样本非常敏感。
- test 集保持锁定，当前没有 test 指标、外部泛化、延迟或长视频误报证据。
- 下一步收益最高的是补充跨来源、连续背景和老人域正负样本，再在相同协议上重训并重新冻结阈值。
