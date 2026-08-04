# 全动作共享编码器预训练与步态迁移实验

日期：2026-08-03

状态：开发实验；未读取 test 指标，不满足部署或正式泛化门禁。

## 数据审计

v3 根动作标签共 9,314 条。按冻结 assignments、具体动作 tier 和输入质量约束：

- 6,405 条非 test、非 `ignore`、非 U01 且 `action_type_training_tier != ignore` 的动作标签可进入开发预训练。
- 6,405/6,405 均有 cleaned pose 缓存。
- 2,296 个动作段生成 2,661 个 `[16,14,5]` 窗口；4,097 个动作段因不足 10 个有效重采样帧等质量约束没有生成窗口。
- 训练/验证/排除窗口为 1,249/1,346/66；锁定 test 使用数为 0。
- Pre_VFallp 的 246 条 v3 assignment 全部属于 test，因此不得进入预训练、微调、阈值选择或开发验证。

输入产物：

```text
data/processed/fall_risk/action_pretraining_v2_semantic5/
```

## 预训练任务

所有可用 A/B/C/D 动作被映射为五个语义类：

```text
normal_locomotion
normal_transition_or_activity
functional_impairment
balance_loss
fall_or_post_fall
```

模型复用 14,114 参数轻量 TCN 的 `input_projection` 和 `temporal_blocks`，辅助分类头不迁移。训练使用动作段总权重归一化、tier 权重、来源平衡、类别平衡、左右镜像和冻结 validation 早停。

预训练结果：

| 指标 | 结果 |
|---|---:|
| validation accuracy | 0.876 |
| validation balanced accuracy | 0.702 |
| validation macro-F1 | 0.658 |
| functional impairment recall | 0.667 |
| best epoch | 26 |

checkpoint：

```text
reports/fall_risk/action_pretraining_v2_semantic5/development-20260803/encoder.pt
```

## 步态迁移对照

目标为 `observable_instability_b02_b04`。只使用 frozen train/validation，不读取 test。该合法划分仅有 15 个异常训练窗口和 2 个异常验证动作段窗口，因此所有结果方差极高。

| seed 42 模型 | BA | Recall | Specificity | Precision | F1 | Brier | Log loss | FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scratch | 0.972 | 1.000 | 0.945 | 0.054 | 0.103 | 0.042 | 0.130 | 35 |
| 5 类预训练，直接微调 | 0.960 | 1.000 | 0.921 | 0.038 | 0.074 | 0.023 | 0.080 | 50 |
| 5 类预训练，冻结编码器 8 epoch | 0.969 | 1.000 | 0.938 | 0.049 | 0.093 | 0.036 | 0.109 | 39 |

共享预训练改善了 Brier、log loss 和 ECE，但没有超过 scratch 的分类 F1 或误报数。由于验证正例只有 2 个，不继续搜索冻结轮数、学习率或阈值，也不运行 test。

## Fold A/B 复核

旧 `fold_a_partitions`/`fold_b_partitions` 把 frozen test 中的 Pre_VFallp 重新标为 validation/train。旧 LightGBM Fold A/B balanced accuracy 0.692/0.690 及相关 TCN 结果因此不能作为无泄漏选模证据。

当前代码执行两层阻断：

- 正式 v3 数据构建只发布 `frozen` partition scheme。
- TCN 和表格训练器拒绝任何把 frozen train/validation/test 重新分配到其他有效 partition 的旧数据集。

## 结论

全动作资源已经用于共享表示学习，但不能解决步态头的直接监督不足。当前最可信结论不是“模型达到 97%”，而是：在仅 2 个验证正例上，任何单点分类指标都不稳定；共享预训练改善校准但未减少误报。默认步态 checkpoint 继续为 `null`，运行时保留规则 fallback。
