# 坐站片段识别候选报告

日期：2026-08-27

状态：`development_provisional`

## 结论

在不读取锁定 test 姿态和 test 指标的前提下，新增的 `sit-stand-random-forest-v1` 在内部确认集上通过片段级优秀门槛。该模型识别动作片段中是否存在正常坐下/起立动作，不做连续视频中的起止时刻定位，也未接入冻结的实时坐站分支。

| 指标 | 选择集 | 内部确认集 | 确认门槛 |
|---|---:|---:|---:|
| Precision | 0.9855 | 0.9292 | >=0.85 |
| Recall | 0.9855 | 0.9807 | >=0.90 |
| F1 | 0.9855 | 0.9543 | >=0.90 |
| PR-AUC | 0.9987 | 0.9922 | >=0.95 |

确认集包含 630 个事件，其中正例 415、显式困难负例 215；混淆矩阵为 TP=407、FP=31、FN=8、TN=184。阈值 `0.21946296296296294` 仅由选择集确定。

## 数据与防泄漏

- 训练只使用冻结 v3 assignment 的 `train`，共 2,018 个窗口、1,932 个事件、135 个保护组和 7 个数据集。
- 原 `validation` 按 `split_group_id` 的固定哈希拆成选择集和确认集；两个集合的保护组不重叠。
- 选择集有 271 个事件、3 个保护组，来自 NTU RGB+D 与 LE2I。
- 确认集有 630 个事件、6 个保护组，来自 NTU RGB+D 与 Fall Detection 2017。
- 1,032 个 test 标签仅计数；未读取 test 姿态、未生成 test 特征、未计算 test 指标。

事实源：

- 协议：`configs/evaluation/sit_stand_clip_presence_v1.provisional.yaml`
- 数据：`data/processed/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b/`
- 指标：`reports/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b/random-forest-seed42/metrics.json`
- checkpoint：`reports/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b/random-forest-seed42/best_model.joblib`
- checkpoint SHA-256：`cf4bfd7b3e08ac255f3376e1a3254ad8faea60efbaf8931023de9544bc5a3bf1`

复现命令：

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_sit_stand_event_dataset.py \
  --output-dir data/processed/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b \
  --allow-provisional

conda run -n eldercare-ai python scripts/train/train_sit_stand_tabular.py \
  --data data/processed/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b/dataset.npz \
  --metadata data/processed/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b/metadata.json \
  --output-dir reports/fall_risk/sit_stand_clip_presence_v1/splitv3-3342705b/random-forest-seed42
```

## 不能合并的指标

连续坐站事件定位仍使用 v1 冻结 TCN。其统一自采 115 视频工程回放为 Precision=0.2018、Recall=0.4259、F1=0.2738、PR-AUC=0.2876；现有正式开发流 F1=0.449。片段级 F1=0.9543 不能替代这些连续定位指标，也不能据此升级实时 release。

下一步若要提升实时坐站能力，必须补连续视频的坐姿/站姿状态真值、起止边界和床上转移/下蹲/弯腰困难负例，并在未见人员或来源的冻结确认集上重新训练状态转移模型。
