# 徘徊方案步骤 6 纯 TCN 深度对照报告

日期：2026-08-04

本报告只覆盖 `comparison_only` 的纯 TCN 深度对照。它不作为运行 fallback，不接摄像头、数据增强、概率校准、拒识、episode、日级心理风险、TopoWander 组件或 SmartCare official 20 条，也不构成真实老人、人员级泛化或临床有效性证据。

## 结论

- 为 `four_class` 和 `binary` 分别训练了五个从随机初始化开始、互不共享任何参数/优化器/checkpoint 的 CPU TCN，共 10 个安全 NPZ。四分类使用分层 gate + subtype 头，二分类使用独立 sigmoid 头；参数量分别固定为 40,836 和 34,433。
- 唯一模型输入是步骤 4 ready 记录的 `float32 [80,14] model_features` 与显式 `[80] point_mask`。零基索引 10/11 的时间通道保持 0，索引 12 与外部 mask 完全一致；公开 ready mask 全为 1，部分 mask 只通过接口/单元测试，不冒充真实缺失轨迹鲁棒性。
- development 继续复用步骤 5 fail-closed loader。四分类为 WP `1120 train / 240 validation`，二分类为 WP+SmartCare `1257/278`；15 条 unavailable 留在 1,790 条分母中但未进入 tensor/loss。训练入口没有打开 RF public manifest、WP test 或 official。
- 配置、源码和 development manifest 冻结后，evaluator 才通过两个外部 SHA 对固定 240 条 WP test 推理。evaluator 没有 fit 路径，未重新训练、改 early stopping、阈值、类别顺序或 seed。
- TCN 并非全面优于 RF：四分类 validation/test 都低于 RF；二分类 validation/test 都高于 RF。该差值只作同 seed 描述，不宣称统计显著性，也没有据此回改 v1。

## 固定信任根与样本口径

| 项目 | 固定值 |
|---|---|
| TCN config SHA-256 | `8a8ab3a051dd6a00356df507b4fa3f06d4a7ed7ec7ecca8f04a3ff28dfd22cca` |
| RF config SHA-256 | `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35` |
| RF development manifest | `fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2` |
| RF public benchmark manifest | `8b89b0b4f14e2c9e4054b8121c9ee8428eab8b6311c32a411b699f46cb02167c` |
| TCN development manifest | `0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e` |
| TCN public benchmark manifest | `1b0c67f86b291dfba4d48314d470d5bc09f84089aac7e23c718e3cd77ad8c2c4` |

| 任务/来源 | train fit | validation | frozen WP test |
|---|---:|---:|---:|
| WP 四分类 | 1,120；每类 280 | 240；每类 60 | 240；每类 60 |
| WP 二分类 | 1,120；0/1=`280/840` | 240；0/1=`60/180` | 240；0/1=`60/180` |
| SmartCare 二分类 | 137；0/1=`63/74` | 38；0/1=`17/21` | 无；official 20 继续封存 |

二分类训练逐样本复用步骤 5 的“来源 × 类别”四组等总权重；四分类统一权重 1。所有指标按真实样本无权重计算。

## 五个固定 seed 的真实结果

`VF` 为 WP 四分类 validation macro-F1；`VB` 为二分类 WP/SmartCare 两来源 macro-F1 的等权均值；`TF/TB` 为固定 WP public shape benchmark 的四/二分类 macro-F1。

| seed | VF | VB | VB-WP | VB-SmartCare | TF | TB |
|---:|---:|---:|---:|---:|---:|---:|
| 20260731（主 artifact） | 0.923717 | 0.997238 | 0.994475 | 1.000000 | 0.920147 | 0.988889 |
| 20260801 | 0.953808 | 0.994505 | 0.989010 | 1.000000 | 0.949845 | 0.971745 |
| 20260802 | 0.949607 | 0.997238 | 0.994475 | 1.000000 | 0.945756 | 0.994475 |
| 20260803 | 0.936884 | 0.997238 | 0.994475 | 1.000000 | 0.932599 | 0.983425 |
| 20260804 | 0.941461 | 0.997238 | 0.994475 | 1.000000 | 0.941513 | 0.983240 |
| 五 seed 均值 ± 总体标准差 | `0.941096 ± 0.010526` | `0.996691 ± 0.001093` | `0.993382`（均值） | `1.000000`（均值） | `0.937972 ± 0.010590` | `0.984355 ± 0.007538` |

同 seed `TCN-RF` 主指标差：

| 阶段/任务 | 差值均值 ± 总体标准差 | 方向 |
|---|---:|---|
| validation 四分类 | `-0.031485 ± 0.011394` | TCN 低于 RF |
| validation 二分类 | `+0.045956 ± 0.007233` | TCN 高于 RF |
| WP test 四分类 | `-0.032753 ± 0.010590` | TCN 低于 RF |
| WP test 二分类 | `+0.024212 ± 0.009102` | TCN 高于 RF |

主 seed 混淆矩阵按固定类别顺序为：

- WP 四分类 validation：`[[60,0,0,0],[0,59,0,1],[0,2,54,4],[2,3,6,49]]`。
- 二分类 WP validation：`[[60,0],[1,179]]`；SmartCare validation：`[[17,0],[0,21]]`。
- WP 四分类 test：`[[60,0,0,0],[0,58,0,2],[0,1,53,6],[1,4,5,50]]`。
- WP 二分类 test：`[[59,1],[1,179]]`。

完整 accuracy、balanced accuracy、逐类 precision/recall/F1，以及二分类 AUROC、AUPRC、sensitivity、specificity、Brier 和截断 NLL 均保存在 `validation_metrics.json` 与 `test_metrics.json`。概率未经校准，不能按临床风险概率解释。

## 训练历史、确定性与安全加载

| 任务 | seed | best epoch | stop epoch | 停止原因 |
|---|---:|---:|---:|---|
| four_class | 20260731 | 51 | 71 | early stopping |
| four_class | 20260801 | 145 | 150 | max epochs |
| four_class | 20260802 | 132 | 150 | max epochs |
| four_class | 20260803 | 89 | 109 | early stopping |
| four_class | 20260804 | 81 | 101 | early stopping |
| binary | 20260731 | 54 | 74 | early stopping |
| binary | 20260801 | 54 | 74 | early stopping |
| binary | 20260802 | 62 | 82 | early stopping |
| binary | 20260803 | 85 | 105 | early stopping |
| binary | 20260804 | 63 | 83 | early stopping |

- early stopping 只使用 validation 主指标，严格 `metric > best + 1e-4` 才算改善；并列保留更早 epoch，最少 30 epoch，patience 20。manifest 明确记录 `validation_used_for_checkpoint_selection=true`。
- 正式 development 目录包含 45 个 manifest 绑定 artifact，加 `manifest.json` 共 46 个文件；独立完整训练到 `tmp/wandering_step6_determinism/development_rebuild` 后，两目录完整路径、长度和 SHA-256 差异数为 0。
- 正式 WP test 目录包含 13 个绑定 artifact，加 manifest 共 14 个文件；用同一双 SHA 重放到 `tmp/wandering_step6_determinism/public_shape_benchmark_rebuild`，差异数同样为 0。
- checkpoint 是固定 ZIP 元数据、按 state key 排序、仅含有限 `float32` 数组的 NPZ。加载固定使用 `numpy.load(..., allow_pickle=False)`，不调用 `torch.load`，也不保存 `.pt/.pth/.ckpt`。
- 安全顺序为：外部 manifest SHA → manifest 绑定的全部文件长度/SHA → config/源码/环境/metadata → NPZ key/dtype/shape/finite → semantic fingerprint → 才允许 forward。回归测试证明 checkpoint 哈希漂移时 NPZ loader 不会被调用。

## CPU batch=1 基准

环境为 Python `3.11.15`、NumPy `2.4.6`、PyTorch `2.13.0+cu130`、scikit-learn `1.9.0`；CUDA 不可用，intra-op/inter-op 均固定 1。20 次预热、200 次测量，时间不进入确定性 manifest。

| 任务 | 参数量 | 主 NPZ | tensor median/P95 | forward median/P95 | 合计 median/P95 |
|---|---:|---:|---:|---:|---:|
| four_class | 40,836 | 170,880 B | 0.107/0.156 ms | 0.547/0.757 ms | 0.664/0.937 ms |
| binary | 34,433 | 144,276 B | 0.106/0.136 ms | 0.529/0.758 ms | 0.637/0.849 ms |

10 个 NPZ checkpoint 合计 1,575,780 B。该基准只反映当前机器单线程 CPU 的工程延迟，不是跨硬件保证。

## 测试与复现命令

- 测试先行：新增测试首次收集得到预期 `ModuleNotFoundError`，随后才实现模块。
- 步骤 5+6 联合窄测试：`35 passed, 30 subtests passed`。
- 全部 `tests/test_wandering_*.py`：`108 passed, 104 subtests passed`。
- 完整测试集：`486 passed, 1 failed, 171 subtests passed`。唯一失败为既有跌倒测试找不到 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`；另有既知 CUDA 驱动警告，二者均不属于步骤 6 改动。

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_tcn_baseline.py -q

conda run -n eldercare-ai python scripts/wandering/train_tcn_baseline.py \
  --config configs/modules/wandering_tcn_v1.yaml \
  --project-root . \
  --output reports/mental_health/wandering_step6/development/v1

conda run -n eldercare-ai python scripts/wandering/evaluate_tcn_baseline.py \
  --config configs/modules/wandering_tcn_v1.yaml \
  --project-root . \
  --development-dir reports/mental_health/wandering_step6/development/v1 \
  --expected-development-manifest-sha256 0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e \
  --expected-rf-public-manifest-sha256 8b89b0b4f14e2c9e4054b8121c9ee8428eab8b6311c32a411b699f46cb02167c \
  --output reports/mental_health/wandering_step6/public_shape_benchmark/v1

conda run -n eldercare-ai python scripts/wandering/benchmark_tcn_baseline.py \
  --config configs/modules/wandering_tcn_v1.yaml \
  --project-root . \
  --development-dir reports/mental_health/wandering_step6/development/v1 \
  --expected-development-manifest-sha256 0f4c48d948f0f4355dd577c89330b050ecb1bb513ca83ef1b25234897e27a10e \
  --output reports/mental_health/wandering_step6/runtime_benchmark.json
```

三个正式输出都必须事先不存在。不要从任意 manifest 内部反读“期望哈希”作为外部信任根。

## 证据边界

WP test 已在步骤 5 公开运行过，因此本次只能称“TCN 配置、源码和 development manifest 冻结后的固定评估”，不是整个项目第一次看到的完全盲测。WP `<0.05` 仍有 4,054 对跨分区形状近邻且没有人员/session 分组；SmartCare 自然日也只是替代组。这里的结果只能证明固定公开 shape benchmark 上的纯 TCN 深度对照，不能称为人员级泛化、摄像头域、真实老人、徘徊 episode、心理状态判断或医学诊断。步骤 6 未接入 `mental_health` 聚合、评分、运行时或 `AlgorithmEvent`；下一步只能进入步骤 7 的 bbox-only adapter、Camera QC 和最小离线推理链。
