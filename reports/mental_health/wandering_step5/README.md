# 徘徊方案步骤 5 手工特征 + Random Forest 对照基线报告

日期：2026-08-03

本报告只覆盖 `comparison_only` 的传统机器学习对照基线。它不作为运行 fallback，不接摄像头、数据增强、概率校准、拒识、episode、日级心理风险或 SmartCare official 20 条，也不构成真实老人、人员级泛化或临床有效性证据。

## 结论

- 实现了步骤 4 bundle 的 fail-closed loader，固定校验 RF 配置、步骤 4 四个机器文件、canonical split、assignments、近邻审计与 `human_review_passed` 原始字节哈希。loader 为完整性解析 1,790 行，但 development API 只暴露 train/validation，frozen API 只暴露 240 条 WP test；配置中不存在 official 文件路径。
- 固定 26 维 `wandering-handcrafted-features-v1` 只读取 ready 记录的 `shape_normalized_points`、`point_mask`、未标准化 `raw_features` 质量通道与 `topology`。平移、旋转、镜像、时间反向及“来源点→稳健中心/各向同性尺度→特征”的正统一缩放不变性均通过测试。
- development 特征表恰为 1,535 条 ready train+validation；15 条 unavailable 仍在 1,790 条分母中留痕，但没有进入特征表或 fit。四分类只含 WP `1120/240`，二分类含 WP+SmartCare `1257/278`；所有模型只在 train 拟合。
- 两个任务均使用预注册的 500 棵树参数和五个固定 seed。二分类 train 的 WP/0、WP/1、SmartCare/0、SmartCare/1 分别为 `280/840/63/74`，单样本权重约为 `1.1223214286/0.3741071429/4.9880952381/4.2466216216`，四组总权重相等；四分类统一权重 1。
- development manifest 的最终外部信任根为 `fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2`。frozen evaluator 先校验这个外部值、当前 RF 配置/feature schema、全部 artifact、依赖版本和 joblib 哈希，之后才反序列化并检查类别顺序、固定 RF 参数和 semantic fingerprint。
- public shape benchmark manifest 为 `8b89b0b4f14e2c9e4054b8121c9ee8428eab8b6311c32a411b699f46cb02167c`，显式绑定上述 development manifest，记录 `models_retrained=false`、`official_source_opened=false` 和 `temporal_features_enabled=false`。

## 固定样本与阶段边界

| 任务/来源 | train fit | validation | frozen WP test |
|---|---:|---:|---:|
| WP 四分类 | 1,120；每类 280 | 240；每类 60 | 240；每类 60 |
| WP 二分类 | 1,120；0/1=`280/840` | 240；0/1=`60/180` | 240；0/1=`60/180` |
| SmartCare 二分类 | 137；0/1=`63/74` | 38；0/1=`17/21` | 无；official 20 继续封存 |

development 产物中不存在 test feature、test prediction 或 test metric。只有项目负责人把训练命令打印的 manifest SHA-256 显式传给 evaluator 后，才首次为 240 条 WP test 提取 26 维特征并计算指标。

## 五个固定 seed 的真实结果

`VF` 为 WP 四分类 validation macro-F1；`VB` 为二分类 WP/SmartCare 两来源 macro-F1 的等权均值；`TF/TB` 为 frozen WP public shape benchmark 的四/二分类 macro-F1。

| seed | VF | VB | VB-WP | VB-SmartCare | TF | TB |
|---:|---:|---:|---:|---:|---:|---:|
| 20260731（主 artifact） | 0.975132 | 0.954274 | 0.962526 | 0.946023 | 0.970725 | 0.962141 |
| 20260801 | 0.970894 | 0.954082 | 0.962141 | 0.946023 | 0.970725 | 0.962141 |
| 20260802 | 0.974995 | 0.945619 | 0.972958 | 0.918280 | 0.970725 | 0.956955 |
| 20260803 | 0.966750 | 0.940210 | 0.962141 | 0.918280 | 0.970725 | 0.956955 |
| 20260804 | 0.975132 | 0.959490 | 0.972958 | 0.946023 | 0.970725 | 0.962526 |
| 五 seed 均值 ± 总体标准差 | `0.972580 ± 0.003337` | `0.950735 ± 0.006890` | `0.966545`（均值） | `0.934925`（均值） | `0.970725 ± 0.000000` | `0.960143 ± 0.002607` |

主 seed 混淆矩阵按固定类别顺序为：

- WP 四分类 validation：`[[59,1,0,0],[0,59,0,1],[0,2,58,0],[0,1,1,58]]`。
- 二分类 WP validation：`[[60,0],[7,173]]`；SmartCare validation：`[[15,2],[0,21]]`。
- WP 四分类 test：`[[59,0,0,1],[0,58,1,1],[0,0,60,0],[1,3,0,56]]`。
- WP 二分类 test：`[[59,1],[6,174]]`。

完整 accuracy、macro-F1、balanced accuracy、逐类 precision/recall/F1，以及二分类 AUROC、AUPRC、sensitivity、specificity、Brier、截断 NLL，均保存在 `validation_metrics.json` 与 `test_metrics.json`。这些概率未经校准，不能按临床风险概率解释。

## 解释、失败案例与性能

- 五个 seed 均保存 RF impurity importance 和 validation 上 `n_repeats=20`、macro-F1 的 permutation importance。四分类 permutation 均值最高的是 `turn_direction_coherence`；二分类最高的是 `reversal_angle_mean_deg`。impurity importance 只描述树分裂使用情况，不是因果贡献。
- 主 seed 对全部 518 条 validation 任务记录保存逐样本 `local_sensitivity`：每次只把一个特征替换为该任务 train median，记录类别概率变化。它不是 SHAP、不可相加，也不是因果解释。
- train 中 `valid_point_ratio`、`mean_quality`、`min_quality` 均为零方差，但按照固定 26 维 schema 保留。没有为改善结果删特征或替换为来源、ID、输入长度、时间或画布信息。
- 高置信错误按 cohort 保存：主 seed validation 有四分类 WP 6 条、二分类 WP 7 条、SmartCare 2 条；WP test 两个任务各列 7 条。每条包含 sample ID、真值、预测、概率和 26 维特征。
- 固定 CPU、`n_jobs=1`、20 次预热和 200 次测量：四分类单样本特征/RF/合计 median 为约 `3.45/14.23/17.72 ms`，P95 为约 `4.33/16.49/19.98 ms`；二分类约 `3.32/13.59/16.98 ms`，P95 约 `3.79/14.58/18.15 ms`。主模型 joblib 大小约为四分类 `4.47 MB`、二分类 `3.08 MB`。时钟值不进入确定性 manifest。

## 确定性与安全加载

- 在两个不存在的新 development 目录完整运行全部五 seed 和 importance 后，文件列表均为 29 个，逐文件 SHA-256 差异为 0；两个 manifest SHA-256 相同。比较范围包含 data index、feature table、全部 validation predictions/metrics、importance、local sensitivity、model semantic fingerprints 和 10 个 joblib。
- 对已经存在的正式 output 再次运行训练命令，以退出码 1 明确拒绝覆盖；正式 manifest SHA-256 前后不变。
- 单测证明 manifest trust root 或 joblib SHA 不匹配时，joblib loader 不会被调用；匹配后还要通过依赖版本、类别、固定参数和 semantic fingerprint。保存后安全加载的逐样本概率与原内存模型完全一致。
- `.gitignore` 已增加 `*.joblib`。joblib 是 Pickle 信任边界，不能加载本流程之外的未知文件。

## 环境与测试

训练环境：Python `3.11.15`、NumPy `2.4.6`、scikit-learn `1.9.0`、joblib `1.5.3`、CPU 标识 `x86_64`。editable 安装指向当前仓库。

- 测试先行：三份新测试首次收集得到 3 个预期 `ModuleNotFoundError`，随后才实现模块。
- 步骤 5 固定窄测试：`22 passed, 21 subtests passed`。
- 全部 `tests/test_wandering_*.py`：`95 passed, 95 subtests passed`。
- 完整测试集：`473 passed, 1 failed, 162 subtests passed`。唯一失败为既有跌倒测试 `tests/test_fall_runtime_fingerprint.py` 找不到固定视频 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`；另有一条既有 CUDA 驱动版本警告。二者都不属于步骤 5 改动。

## 复现命令

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_preprocessing_bundle.py \
  tests/test_wandering_handcrafted_features.py \
  tests/test_wandering_rf_baseline.py -q

conda run -n eldercare-ai python scripts/wandering/train_rf_baseline.py \
  --config configs/modules/wandering_rf_v1.yaml \
  --project-root . \
  --output reports/mental_health/wandering_step5/development/v1 \
  --report-root reports/mental_health/wandering_step5

conda run -n eldercare-ai python scripts/wandering/evaluate_rf_baseline.py \
  --config configs/modules/wandering_rf_v1.yaml \
  --project-root . \
  --development-dir reports/mental_health/wandering_step5/development/v1 \
  --expected-development-manifest-sha256 fff6340e868de32bee2021ec1000f166b8caabe5caeb1132abb8ab822bfaaaf2 \
  --output reports/mental_health/wandering_step5/public_shape_benchmark/v1 \
  --report-root reports/mental_health/wandering_step5
```

两个 output 目录都必须事先不存在。不要从任意 manifest 内部反读“期望哈希”作为信任根。

## 2026-08-04 独立结项复核

- 当前 RF config SHA-256 为 `d29db531f42ae5d7cf4ed540a85681ddab78e483695633ccc8b382d98e745e35`。development manifest 绑定的 28 个 artifact、public benchmark manifest 绑定的 13 个 artifact 均逐文件复核长度和实际 SHA-256，无缺失、多余或漂移。
- 1,535 条 development feature 与 240 条 test feature 的 ID、顺序、来源、split 和标签计数复算正确，二者 ID 交集为 0；所有记录均为有限 26 维。train 中只有第 24–26 维为零方差，值均为 1，符合当前全 1 mask/质量契约。
- 五个 seed 的 validation/test macro-F1 从逐条预测重新计算后与本报告完全一致；概率和、argmax、`correct` 标记、主 seed 失败案例、518 条 local sensitivity 和 `2 tasks × 5 seeds × 26 features × 20 repeats` importance 均通过一致性核验。
- 10 个 joblib 的 manifest fingerprint、独立 fingerprint、类别、seed、固定参数和 500 棵树结构一致。项目 conda 环境窄测试、全部徘徊回归和完整测试结果仍分别为 `22 passed, 21 subtests passed`、`95 passed, 95 subtests passed`、`473 passed, 1 failed, 162 subtests passed`；完整测试唯一失败仍是跌倒模块固定视频缺失。
- 在不存在的新临时目录完成独立全量训练，并以正式 development manifest 外部信任根重放 frozen evaluator，再次得到完全相同的 development/public benchmark manifest SHA-256。结论：步骤 5 的当前工作区技术完成门槛通过，可以进入步骤 6。
- SmartCare official 没有进入步骤 5 配置、loader、训练、预测或正式结果。为核验原文件未变，本次审计只读取其原始字节计算一次 SHA-256；没有解析样本或用于模型，但 Windows `LastAccessTime` 因该读取更新，内容和 `LastWriteTime` 未改变。
- 两个非阻塞维护项：训练函数在 formal development 目录提交后才计算 runtime/补充报告，若该后处理失败会影响同路径重跑；现有夹具没有为 26 个特征中的每一维都保存独立手算 golden 值。当前正式结果齐全且独立复算通过，不回改 RF v1；步骤 6 改用确定性 bundle 原子提交和独立 benchmark 命令。
- 当前机器产物被 Git 忽略，步骤 3–5 配置/源码/报告仍有未跟踪文件。因此本结论是“当前工作区可复现”，不是“仅凭现有 Git 提交可换机复现”；交接前仍需提交文本事实源，并另行归档受 manifest 绑定的模型或在目标机器确定性复建。

## 证据边界

WP test `<0.05` 仍有 4,054 对跨分区形状近邻，且 WP 没有人员/session 分组；SmartCare 自然日也只是替代组。这里的高分只能说明固定公开 shape benchmark 上的可解释下限较强，不能称为人员级泛化、摄像头域、真实老人、徘徊 episode、心理状态判断或医学诊断。步骤 5 没有接入 `mental_health` 聚合、评分、运行时或 `AlgorithmEvent`，下一步只能进入同协议的纯 TCN 对照（步骤 6），不能直接把 RF 写成部署模型。
