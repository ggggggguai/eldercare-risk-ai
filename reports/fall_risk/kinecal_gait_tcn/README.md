# KINECAL 轻量步态 TCN baseline

日期：2026-07-17

状态：训练链路已跑通，模型效果未达到可用门槛，不接入实时主链。

## 任务与数据

- 目标：`NF=0`、`FHs/FHm=1` 的回顾性跌倒史 proxy 二分类。
- 输入：3 米步行的统一 14 点二维骨架，通道为 `x/y/dx/dy/quality`。
- 窗口：30 FPS、128 帧、步长 64 帧。
- 独立参与者：50 人，其中 `NF=28`、`FHs/FHm=22`。
- 窗口：219 个。
- 按参与者固定划分：train 36 人、validation 7 人、test 7 人。
- 7 名风险组参与者因官方版本没有 3 米步行目录而排除；没有构造空序列。

处理后的数据 SHA-256：

```text
fe8e50797813c6beb5b773217549768bd94521e922e5868a7328c9874d354b7f
```

## 模型与训练

- 深度可分离时间卷积，dilation 为 `1/2/4/8`。
- 隐藏通道：48。
- 参数量：14,114。
- AdamW，学习率 `0.001`，weight decay `0.0001`。
- batch size 16，最多 80 轮，patience 15。
- 固定 seed 42，CPU 训练，左右镜像增强。
- 第 52 轮早停，最佳权重来自第 37 轮。

最佳模型 SHA-256：

```text
8cd9e1bf633eb300799325180d2ade033de0f7fb32dec48fc270bb23caed620d
```

## 结果

以下为按参与者聚合窗口概率后的指标，阈值固定为 0.5：

| 分区 | 人数 | Accuracy | Balanced accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 7 | 0.571 | 0.542 | 0.400 | 0.833 |
| Test | 7 | 0.286 | 0.333 | 0.444 | 0.583 |

测试集混淆矩阵为 `TN=0, FP=4, FN=1, TP=2`。模型没有识别出任何 `NF` 测试参与者，当前固定划分下不可用。验证和测试 ROC-AUC 差异也说明小样本方差很大，不能用验证集结果宣称模型有效。

## 结论

这次实验只证明以下工程能力已打通：

- Kinect 25 点到 COCO 兼容 14 点的确定性转换。
- 参与者级无泄漏划分。
- 轻量 TCN 训练、早停、权重保存和参与者级评估。

它没有证明 KINECAL 可以单独训练出可部署的步态风险模型。进入主链前至少需要：

1. 重复参与者级交叉验证和置信区间。
2. 与 Logistic/LightGBM 和规则步态特征做同划分对照。
3. 使用真实 RGB 老人步态视频提取统一 14 点骨架进行微调。
4. 在独立 RGB 测试集验证跨相机、遮挡和姿态误差。
5. 单独做概率校准；不得把跌倒史 proxy 写成未来跌倒预测。

## 复现

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_kinecal_gait_tcn.py \
  --input-dir data/external/kinecal/raw \
  --output-dir data/processed/fall_risk/kinecal_gait_tcn \
  --target-fps 30 \
  --max-gap-sec 0.10 \
  --window-frames 128 \
  --stride-frames 64 \
  --seed 42 \
  --overwrite

conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/kinecal_gait_tcn/dataset.npz \
  --output-dir reports/fall_risk/kinecal_gait_tcn \
  --epochs 80 \
  --batch-size 16 \
  --hidden-channels 48 \
  --kernel-size 5 \
  --dilations 1,2,4,8 \
  --dropout 0.20 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --patience 15 \
  --seed 42 \
  --device cpu \
  --overwrite
```

本地机器可读产物为 `metrics.json`、`history.jsonl`、`test_participant_predictions.jsonl` 和 `best_model.pt`。这些生成文件由 Git 忽略，本文保留实测结论和复现参数。
