# 步态真实上下文弱监督 v2：受限消融

更新时间：2026-08-10

## 结论

本实验把短动作段扩展为 4 秒窗口，但只使用同一姿态 `track_id` 的真实前后帧。标注外上下文保持未标注，不作为负样本；`label_span_mask` 只允许标注区间进入监督池化。该协议把 train 正类可监督动作段从旧协议的 61 段提高到 92 段，但没有解除正式训练门禁。

当前仍是 `development_provisional`：validation 主口径只有 9 个正动作段、1 个 source group；弱上下文敏感性口径只有 1 个正动作段。仓库已完成 seed 42 的单次开发训练，但该结果不能用于正式模型选择、概率校准、test 评估或替换规则主路径。

## 数据契约

- 窗口：4 FPS、16 帧、4 秒名义时长。
- 上下文：先在标注区间内按框重合、覆盖率和质量锁定目标轨迹，再仅沿同一轨迹补真实帧。
- 缺帧：保留为 `valid_mask=0`，不插值生成动作证据。
- 监督：上下文不进入标签语义；TCN 用 `label_span_mask` 做监督池化。
- 主证据：标注区间内至少 10 个有效观测，loss weight 1.0。
- 弱证据：5-9 个有效观测，只用于 train 弱监督，loss weight 0.35；validation 单列敏感性结果。
- 2-4 个观测：保留在数据集，binary loss weight 0；少于 2 个只作审计。
- 负样本：按主证据/弱证据时长层级重加权；训练器继续执行来源平衡，连续质量值不作为分类特征。

协议配置见 `configs/training/gait_context_v2.provisional.yaml`。
独立审计产物为 `training_audit.md`、`training_blockers.md` 和 `reports/reproducibility/gait_context_v2_training_manifest.json`，状态为 `development_provisional`。

## 构建结果

双构建 `splitv3-e71a045-build-a` 与 `splitv3-e71a045-build-b` 的 `dataset.npz` 逐字节一致：

| 项目 | 结果 |
|---|---:|
| dataset SHA-256 | `6af44dfac9820e6e7eaf393e36039e50fbff560cc4e889c2f4e0641364304872` |
| 总窗口 | 2,112 |
| train / validation 窗口 | 1,589 / 523 |
| train 正类主证据段 | 60 |
| train 正类弱证据段 | 32 |
| train 正类仅保留段 | 9 |
| validation 正类主证据段 | 9 |
| validation 正类弱证据段 | 1 |
| test 姿态 / tensor | 未读取 / 未生成 |

train 共保留 101 个正动作段，其中 92 个进入二分类监督。仍未保留的 8 个 train 正动作段以及只保留的 9 段，主要受同轨上下文不足或姿态质量限制；协议没有跨轨拼接或伪造帧。

## Seed 42 开发训练

从头训练最多 80 epoch，因 validation loss 连续 15 epoch 未改善而在第 30 epoch 提前停止，最佳 epoch 为 15。当前机器没有 CUDA/MPS，且预注册的 action-pretraining encoder 文件不存在，因此本次使用 CPU、未迁移旧 encoder。

| 口径 | balanced accuracy | F1 | 说明 |
|---|---:|---:|---|
| 主 validation 动作段 | 0.989 | 0.720 | 340 段中只有 9 个正类 |
| LE2I 子集 | 0.882 | 0.818 | 9 个正类和 17 个负类，仍很小 |
| 弱上下文敏感性 | 0.956 | 0.154 | 126 段中只有 1 个正类，不用于选择 |

checkpoint SHA-256 为 `4e90913f1ba43008827f973284b643af1e656695ab011ac4d4d4315467dc9cde`，绑定 dataset SHA-256 `6af44d...4872`，并记录 `test_evaluated=false`、`test_pose_read=false`。总体指标受到 314 个单一负类 NTU 段影响，不能视为跨来源泛化证据；LE2I 子集和弱证据结果也不足以支持部署。

## Seed 42 烟测

3 epoch TCN 的 checkpoint SHA-256 为 `164ea182326e3520eb2f0948844dca26bba5f848fc76b4a48cb369f15609480b`。主 validation 动作段 balanced accuracy 为 0.971、F1 为 0.486；弱上下文敏感性口径 balanced accuracy 为 0.936、F1 为 0.111。

这些点估计高度不稳定：主口径只有 9 个正段，敏感性口径只有 1 个正段；validation 负类又主要来自 NTU，而正类仅来自 LE2I，存在明显来源捷径风险。因此结果不得与旧 v1 smoke 直接解释为真实性能提升。

## 复现命令

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_gait_window_dataset.py \
  --output-dir data/processed/fall_risk/gait_observable_context_v2/splitv3-e71a045-build-a \
  --target-profile observable_instability_b02_b04 \
  --expand-real-context

conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_observable_context_v2/splitv3-e71a045-build-a/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/tcn-context-smoke-seed-42 \
  --epochs 3 --patience 2 --batch-size 16 --seed 42 --device cpu

conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_observable_context_v2/splitv3-e71a045-build-a/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/development-seed-42 \
  --epochs 80 --patience 15 --batch-size 16 --seed 42 --device cpu
```

## 未解除阻塞

- validation 正类仍少于 30 段，B02/B03/B04 分别仍只有 2/4/3 个主证据动作段，且只有 1 个 source group。
- 部分源视频或姿态产物没有足够的同轨前后帧；需要补采或重新生成更长的姿态轨迹，不能靠协议变更替代。
- split 和评估协议未冻结，连续正常背景、老人域、延迟与失败案例证据仍不足。
- test 未读取；默认 checkpoint 和规则主路径均未修改。
