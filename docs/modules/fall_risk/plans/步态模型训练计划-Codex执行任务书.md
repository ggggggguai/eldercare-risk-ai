# 步态模型训练计划：Codex 执行任务书

更新时间：2026-08-10

> 本文件交给具有仓库与终端访问权限的 Codex 执行。它规定步态模型的工作顺序、数据门禁、实验矩阵、产物和停止条件，不是训练完成证明，也不授权读取 test 或替换规则主路径。

## 1. 任务目标

你是本仓库的计算机视觉与机器学习工程师。请在：

```text
/Users/guai/Documents/project/挑战杯/algorithm
```

建立一条可追溯、无泄漏、可复现的步态模型训练链。主任务固定为：

```text
task = gait_instability_vs_normal_activity
negative = A01-A12 normal activity / hard negative
positive = B02 dragging_walk + B03 shuffling_walk + B04 swaying_walk
functional_proxy = B01 slow_walk（单列，不作为主任务正类）
```

模型输入是经过人体检测、跟踪、姿态质量控制和时序平滑的关键点窗口；模型只输出跌倒模块内部的 `gait_risk_score`、置信度和解释证据。它不输出医学诊断、未来跌倒概率、最终 `risk_level` 或跨模块综合结论。

完成标准不是得到一个较高的 validation 分数，而是：

1. 训练输入绑定当前标签、manifest、split、姿态和配置 hash。
2. train/validation/test 在人员、来源、样本和相邻窗口层面无泄漏。
3. test 在候选、阈值、校准器和代码冻结前不可读取。
4. 与规则、Logistic、LightGBM/EBM 在同一数据和指标协议上比较。
5. 报告动作段、数据源、连续背景误报、校准、拒识、延迟和失败案例。
6. 低质量或模型失败时继续使用规则 fallback；候选默认只进入 shadow。

## 2. 当前事实快照

开始执行时必须重新读取机器产物，不得盲信本节数字。本节旧快照已由 2026-08-17 的 SCF 训练治理报告 supersede：

- editable 安装已指向当前仓库。
- 当前 v3 split 为 `splitv3_3342705b7b1ac51570148337`，`valid=true`、无机器报告泄漏，但仍是 provisional，未冻结。
- `training_ready.action_type=false`；fall/near-fall 事件门禁通过不能替代步态动作门禁。
- 当前 9,314 条 v3 动作标签中，B02/B03/B04 分别为 36/36/65 段；按父任务 `training_tier` 的 primary 数分别为 32/27/63。
- B03 的 `action_type_training_tier` 没有 primary：34 条 auxiliary、2 条 ignore，因此 subtype 头不具备正式监督资格。
- 当前分区中的 B02/B03/B04 标签总数分别为 train/validation/test `25/8/3`、`24/6/6`、`60/3/2`。在姿态质量过滤前，validation 正类上限也只有 17 个动作段，不能支撑完整调参或稳定阈值选择。
- 当前 cleaned pose 缓存有 6,512 个 JSONL；仓库中没有绑定当前 split 的 gait window 数据集。
- 2026-08-03 的 gait 数据集、预训练 encoder、checkpoint 和指标绑定旧 split，不能作为当前 split 的模型效果或 warm-start 输入。
- 当前 `prepare_gait_window_dataset.py` 会读取并写出 test 特征；两个训练 CLI 仍可通过 `--evaluate-test` 直接读取 test；v3 数据 metadata 还会写成 `split_is_provisional=false` 和 `frozen_training_labels_v3`。这些问题修复前不得启动正式训练。
- 默认运行 checkpoint 仍为 `null`，规则步态分支是主路径和 fallback。

事实、推断和决策必须分开记录：上面是当前机器事实；本任务书后文的样本量与性能门槛是预注册的工程决策，不是现有能力。

## 3. 开始前必须阅读

完整阅读以下文件，并在审计报告中引用实际路径：

```text
AGENTS.md
README.md
docs/README.md
docs/architecture/算法工程骨架.md
docs/interfaces/算法事件输出接口.md
docs/tasks/README.md
docs/modules/fall_risk/README.md
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/plans/步态模型训练方案.md
docs/modules/fall_risk/data/跌倒风险标签字典.md
docs/modules/fall_risk/data/数据集标注规范.md
data/annotations/fall_risk/README.md
configs/data/fall_risk_action_label_schema_v3.json
configs/data/fall_risk_training_decision_20260804.json
configs/training/gait_hierarchical_v1.yaml
reports/fall_risk/training-labels-v3-validation.json
reports/reproducibility/dataset_and_split_versions.md
reports/fall_risk/gait-action-pretraining-20260803.md
reports/fall_risk/gait_window_v5_effect_first/development-20260803/README.md
```

使用 `rg --files` 和 `rg` 查找步态数据准备、结构化 baseline、TCN、预训练、运行时加载、fallback 和测试代码。文档引用不存在的产物时记录缺口，不得创建空文件伪装完成。

## 4. 执行边界

### 4.1 允许

- 检查 git 状态、环境绑定、文件 hash、数据元数据和现有报告。
- 为步态专项新增 fail-closed 审计、配置、测试、数据准备、训练、评估、校准、shadow 和报告代码。
- 在 train/validation 范围内构建确定性数据集并运行 bounded smoke。
- 在数据门禁通过后运行本任务书预注册的训练矩阵。
- 新产物使用新目录和版本号，保留 provenance、配置与失败记录。

### 4.2 禁止

- 不得回退、覆盖、删除或移动用户已有修改、原始数据、正式标签、split 或历史报告。
- 不得复用旧 split 的 gait dataset、encoder、checkpoint、阈值或指标。
- 不得在候选冻结前读取 test 姿态、构建 test tensor、运行 test 指标或用 test 做困难样本挖掘。
- 不得把 B01 慢走写成步态不稳、临床异常或未来跌倒真值。
- 不得把未标注背景自动写成 negative；不确定、严重遮挡、目标冲突和边界不可靠样本必须 ignore 或人工复核。
- 不得通过复制窗口、拆分同源保护组、降低质量门禁或把 auxiliary 放入 validation/test 来补样本数。
- 不得自动修改默认 checkpoint、替换规则主路径、修改 `AlgorithmEvent` schema 或新增综合风险模块。
- 不得下载大模型/大型数据、安装依赖、改环境、提交或推送，除非用户另行批准。

## 5. 总体阶段与停止规则

| 阶段 | 内容 | 当前是否可执行 | 退出条件 |
|---|---|---|---|
| P0 | 当前事实与训练授权审计 | 可立即执行 | 输出机器 manifest、审计和 blocker 清单 |
| P1 | test 隔离、provisional 语义和 fail-closed 修复 | 可立即执行 | 回归测试证明默认路径不读取 test |
| P2 | 当前 split 的 train/validation 数据集重建 | P1 通过后执行 | 两次干净构建一致，覆盖报告完整 |
| P3 | E0/E1 baseline 与单 seed TCN smoke | P2 通过后执行 | 只证明链路；不做超参搜索 |
| P4 | 补数、复核与 split 冻结 | 依赖人工数据工作 | 步态专项监督、连续背景和老人域门禁通过 |
| P5 | 三 seed 正式开发训练与校准 | P4 通过后执行 | 候选、阈值、校准器和配置冻结 |
| P6 | 一次性 test 与外部压力评估 | 依赖保管人授权 | 发布不可变 test bundle |
| P7 | shadow、延迟和主链晋级评审 | P6 通过后执行 | 只提出晋级建议，不自动切主路径 |

命中某一阻塞时只停止依赖该阻塞的阶段。继续完成审计器、测试、报告、合成数据和短 smoke；不得用“训练还可以跑”绕过数据门禁。

## 6. 详细执行步骤

### P0：建立步态专项审计

先补失败测试，再实现：

```text
src/elderly_monitoring/modules/fall_risk/gait_training_audit.py
scripts/audit/audit_gait_training.py
tests/test_gait_training_audit.py
configs/evaluation/gait_v1.provisional.yaml
```

审计必须绑定并输出：

- action labels、schema、manifest、assignments、split report、v3 validation report、训练配置和评估配置 SHA-256。
- `split_id`、provisional/frozen 状态和 leakage issues。
- A01-A12、B01-B04 按父 `training_tier`、`action_type_training_tier`、partition、dataset、source/sample/split group 的计数。
- B02-B04 独立动作段和独立保护组覆盖；B01 单列，不合并进主任务正类。
- 6,512 份姿态缓存的实际命中、缺失、空文件、hash/解析和目标轨迹可用性。
- 每个分区在姿态过滤后的窗口、动作段、类别、subtype、来源、时长、拒识原因和质量分布。
- 历史 gait 报告/数据集的 `source_split_id` 与当前 split 是否一致。
- test 姿态是否被读取、test tensor 是否被生成、test 指标是否被计算。

审计输出固定为：

```text
reports/reproducibility/gait_training_manifest.json
reports/fall_risk/gait_training_audit.md
reports/fall_risk/gait_training_blockers.md
```

审计状态只能是：

- `infrastructure_only`：hash、泄漏、输入、test 隔离或 train/validation 二分类基本条件失败。
- `development_provisional`：可在当前 provisional train/validation 做有界开发，但不可读 test、不可称正式结果。
- `candidate_frozen`：标签、split、评估协议和候选配置均冻结，且补数门禁通过。
- `test_released`：存在保管人签发且 hash 绑定的单次 test 发布记录。

当前预期状态不得高于 `development_provisional`；若审计给出更高状态，视为审计器缺陷并先修复。

执行命令：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python scripts/audit/audit_gait_training.py --help
conda run -n eldercare-ai python scripts/audit/audit_gait_training.py
```

### P1：修复 test 隔离和 provisional 语义

采用测试驱动完成以下行为：

1. v3 当前 split 必须写成 `development_provisional`，不得再写 `split_is_provisional=false` 或 `frozen_training_labels_v3`。
2. 数据准备默认只读取 train/validation pose，只写 train/validation tensor；对 test 只记录标签/组计数和 `test_pose_read=false`。
3. train/validation 数据集不得包含可反推 test 标签或特征的预测文件。
4. 从训练 CLI 移除或 fail-close `--evaluate-test`。训练职责只到 validation；test 使用独立评估入口。
5. 独立 test 评估入口必须要求 release 文件，至少绑定 `release_id`、`approved_by`、`split_id`、标签/manifest/assignments/config/checkpoint hash 和发布时间。缺字段、hash 漂移或非 frozen 候选一律失败。
6. 本地代码不能伪造“保管人”或真正的一次性外部控制；若缺人工签发 release 文件，明确阻塞并停止 P6。
7. metadata、checkpoint 和 metrics 都必须包含 `protocol_status`、`test_pose_read`、`test_evaluated`、`source_split_id` 和输入 hash。

至少增加以下回归测试：

- 当前 provisional v3 split 不会被写成 frozen。
- 默认准备阶段不打开 test pose 文件；可用一个会在读取时抛错的 test fixture 证明。
- 默认数据集没有 test tensor/partition。
- 训练 CLI 无法通过单个布尔开关解锁 test。
- release 缺失、split/hash 不匹配、候选未冻结时 test evaluator fail closed。
- 旧 split dataset/checkpoint 不能进入当前训练或评估。

### P2：重建当前 split 的主任务数据集

主数据集固定使用：

```text
target_profile = observable_instability_b02_b04
target_fps = 4.0
window_sec = 4.0
window_frames = 16
stride_frames = 8
max_gap_sec = 0.5
min_observed_frames = 10
max_windows_per_segment = 4
channels = [x, y, dx, dy, quality]
quality_usage = mask_and_pooling_only
```

先在两个全新目录完成确定性双构建，不使用 `--overwrite`：

```bash
conda run -n eldercare-ai python scripts/prepare/prepare_gait_window_dataset.py \
  --output-dir data/processed/fall_risk/gait_observable_v1/splitv3-e71a045-build-a \
  --target-profile observable_instability_b02_b04

conda run -n eldercare-ai python scripts/prepare/prepare_gait_window_dataset.py \
  --output-dir data/processed/fall_risk/gait_observable_v1/splitv3-e71a045-build-b \
  --target-profile observable_instability_b02_b04
```

`splitv3-e71a045` 只是当前路径示例。执行时必须由审计器读取当前 `split_id` 生成目录；split 变化后不得沿用该目录名。

双构建验收：

- `dataset.npz`、`metadata.json`、`samples.jsonl`、派生 assignments 和 split sidecar 逐字节一致，或由报告解释唯一允许的非语义差异。
- 没有 test pose 读取和 test tensor。
- 同一动作段窗口总权重为 1；auxiliary 只进入 train 且总权重再乘 0.35。
- split/source/sample group 不跨 partition；相邻窗口不跨 partition。
- 输出每个 action/subtype/source/quality/rejection 的动作段与窗口计数。
- `quality` 连续值不作为分类内容，只用于 mask 和池化。

构建后执行硬停止判断：

```text
若 validation 独立 B02+B03+B04 动作段 < 30，
或任一 subtype 的 validation 独立动作段 < 10，
或 validation 正类来自少于 3 个独立 source group，
则禁止完整超参搜索、概率校准和候选冻结。
```

按当前标签计数，validation 正类在姿态过滤前最多 17 段，因此本轮预期会命中停止条件。此时继续 P3 的 baseline 和单 seed smoke，但直接转入 P4 补数，不得跑六 seed 或读取 test。

### P3：同协议 baseline 与单 seed smoke

在同一个 P2 数据集上运行：

| 编号 | 模型 | 当前目的 | 当前允许范围 |
|---|---|---|---|
| E0 | 规则 score | 可解释下限与失败分析 | 完整 train/validation 评估 |
| E1a | Logistic | 小样本线性对照 | seed 42 |
| E1b | LightGBM | 非线性结构化对照 | seed 42 |
| E1c | EBM | 可解释非线性对照 | seed 42 |
| E2-smoke | 层级 TCN | 验证数据/训练/checkpoint 链路 | seed 42，最多 3 epoch |

结构化 baseline 禁止用 quality 和尺度捷径作为唯一收益来源；至少比较 `exclude_quality` 和 `exclude_quality_and_scale`。示例命令：

```bash
conda run -n eldercare-ai python scripts/train/train_gait_tabular_baselines.py \
  --data data/processed/fall_risk/gait_observable_v1/<current-split>/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_v1/<current-split>/baseline-seed-42 \
  --models rule,logistic,lightgbm,ebm \
  --feature-profile exclude_quality \
  --seed 42

conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_observable_v1/<current-split>/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_v1/<current-split>/tcn-smoke-seed-42 \
  --epochs 3 --patience 2 --batch-size 16 \
  --hidden-channels 48 --kernel-size 5 --dilations 1,2,4,8 \
  --dropout 0.20 --learning-rate 0.001 --weight-decay 0.0001 \
  --seed 42 --device cpu
```

smoke 结果只能证明工程链路。报告必须包含 `test_evaluated=false`、`test_pose_read=false`、当前 split/config/dataset hash 和“禁止模型选择”的限制。

### P4：补齐训练与评估数据

当前最高收益不是追加 epoch，而是补齐以下真实监督：

1. B02/B03/B04 每类至少 50 个独立 primary sample group，来自至少 3 个独立 source group；冻结分区后每类 train/validation/test 至少 `30/10/10` 个独立动作段。
2. B03 必须先达到 schema 的 primary 条件；不得把现有 auxiliary 人为升级或复制成 primary。
3. validation 正类总计至少 30 个独立动作段，且每类不少于 10；阈值和分 subtype 指标低于此门槛不冻结。
4. 正常行走、转身、坐下、起立、弯腰、下蹲、躺下、跪地、调步、扶物和跳跃均进入 hard-negative 覆盖；每类至少覆盖 2 个来源或明确标记不可评估。
5. 建立至少 2 camera-hour 的连续正常 validation 背景，才能用 `/hour` 作为开发放行指标；片段拼接时长不能冒充连续背景。
6. 在候选 test 之外建立至少 20 camera-hour、20 名老人的目标摄像头域 shadow/外部压力数据，覆盖低光、遮挡、侧视、下肢出画、助行器和多人场景。达不到时只能报告现有域结果。
7. 补充 `subject_id` 或继续使用保守 source group；不为改善分区比例拆散未知人员保护组。

上述数字是本任务预注册的工程最低门槛，不代表临床验证标准。新标签必须遵循 v2 根标签与 v3 训练契约，经过人工复核、来源 hash、split 重建和校验；Codex 不自行制造人工裁决。

P4 完成后必须重跑迁移、split 和校验，并取得数据负责人冻结记录：

```bash
conda run -n eldercare-ai python scripts/annotation/migrate_fall_labels_v2_to_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/build_fall_training_split_v3.py --overwrite
conda run -n eldercare-ai python scripts/annotation/validate_fall_labels_v3.py --overwrite
conda run -n eldercare-ai python scripts/audit/audit_gait_training.py
```

只有专项审计达到 `candidate_frozen` 才进入 P5。

### P5：正式开发训练矩阵

固定数据、split、输入契约和指标实现后，按顺序执行，前一阶段无稳定收益则不进入后一阶段：

| 编号 | 模型 | 目的 | 启用条件 |
|---|---|---|---|
| E0 | 规则 baseline | 可解释对照与 fallback | 必跑 |
| E1 | Logistic、LightGBM、EBM | 小数据结构化强对照 | 必跑 |
| E2 | 层级 TCN scratch | 主时序候选 | 必跑 |
| E3 | 全动作预训练 encoder + 层级 TCN | 检验迁移收益 | 预训练数据同样隔离 test |
| E4 | subtype 辅助头 | 利用 B02/B03/B04 监督 | 三类均为 primary 且过覆盖门禁 |
| E5 | GroupDRO/最差域损失 | 降低来源捷径 | 每类有多个训练来源，E4 已稳定 |
| E6 | TCN + EBM/LightGBM | 降低方差 | 只有合法 OOF 预测时 |
| E7 | ST-GCN++ | 骨架拓扑强对照 | E0-E6 完成且数据量支持 |

E2/E3 固定初始参数：

```text
epochs = 80
batch_size = 16
hidden_channels = 48
kernel_size = 5
dilations = [1, 2, 4, 8]
dropout = 0.20
learning_rate = 0.001
weight_decay = 0.0001
patience = 15
walking_gate_loss_weight = 0.25
freeze_encoder_epochs = 8（仅 E3）
seeds = [42, 43, 44]
```

筛选策略：

1. 所有候选先跑 seed 42；只保留明显优于 E0/E1 且无泄漏/捷径异常的至多 2 个候选。
2. 入围候选再跑 42/43/44，报告均值、标准差和最差 seed；不挑最好 seed 对外报告。
3. 超参顺序固定为输入窗口、模型容量、正则、预训练、subtype、GroupDRO、集成；每阶段只改变一个因素。
4. 早停、阈值、校准和融合权重只使用 train/validation 或 train 内 OOF。
5. B01 只做独立敏感性/功能 proxy 报告，不参与主任务阈值选择。
6. quality-as-feature 和关闭 walking gate 只作为负控制/消融，不能静默成为默认配置。

主训练命令模板：

```bash
conda run -n eldercare-ai python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_observable_v1/<frozen-split>/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_v1/<frozen-split>/e2-seed-42 \
  --epochs 80 --patience 15 --batch-size 16 \
  --hidden-channels 48 --kernel-size 5 --dilations 1,2,4,8 \
  --dropout 0.20 --learning-rate 0.001 --weight-decay 0.0001 \
  --walking-gate-loss-weight 0.25 \
  --temporal-shift-frames 1 \
  --keypoint-dropout-probability 0.05 \
  --coordinate-jitter-std 0.005 \
  --seed 42 --device auto
```

不得包含 test 参数。每次运行写独立目录，禁止覆盖已引用产物。

### P6：冻结评估与一次性 test

候选冻结前必须生成：

```text
candidate_manifest.json
model_card.md
validation_predictions.jsonl
calibration.json
reliability_diagram.png
failure_cases.jsonl
latency.json
```

候选选择顺序固定为：

1. 最差数据源 balanced accuracy。
2. 异常 recall 与正常 specificity。
3. 连续正常背景两窗口确认后的 false alarms/hour。
4. Brier、log loss、ECE 和 reliability diagram。
5. 三 seed 稳定性、拒识率、延迟和内存。

开发候选最低门槛沿用现行方案：

```text
跨源 validation 平均 balanced accuracy >= 0.70
最差 validation 源 balanced accuracy >= 0.60
每个可评估 validation 源 recall 和 specificity >= 0.55
相对规则 baseline 平均提升 >= 0.05，或误报显著下降
三个 seed 的结论方向一致
连续正常 validation raw false positives <= 12/hour
两窗口连续确认 false alarms <= 2/hour
TCN-only CPU P95 <= 25 ms/window
TCN-only MPS P95 <= 10 ms/window
模型常驻内存增量 <= 200 MiB
```

指标以动作段为主要分类单位；窗口指标只作诊断。95% CI 使用 `source_group_id` 或 `sample_group_id` cluster bootstrap。没有足够来源、时长或元数据时写“不可评估”，不得输出虚假点估计。

满足门槛后，由非训练执行者复核 candidate manifest 并签发 test release。Codex 只能在 release 文件存在且 hash 全匹配时运行独立 test evaluator；执行后不得因 test 结果修改模型、阈值、校准器或样本。若 test 失败，候选退回开发并等待新的数据/协议版本，原 test bundle 保留不可变。

### P7：真实视频、shadow 与晋级

test 通过后仍不自动替换主路径。按以下顺序验收：

1. 使用真实 RGB 视频完成姿态与步态端到端 smoke。
2. 在固定硬件 batch size 1、预热后至少 1,000 次窗口推理，报告 P50/P95/P99、内存和吞吐。
3. shadow 同时保留 `model_score`、`fallback_score`、`score_source`、拒识原因和输入质量，不用模型分触发正式风险动作。
4. 在至少 20 camera-hour 目标老人域 shadow 数据上复核误报、拒识、低光、遮挡、助行器、多人和 ID 切换。
5. 证明模型异常、超时、checkpoint 不兼容和低质量输入均回退规则分支。
6. 输出晋级评审；只有用户/项目负责人另行批准，才修改默认 checkpoint 或运行配置。

真实视频 smoke：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_poses_smoke.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5

conda run -n eldercare-ai python scripts/collect/run_fall_gait.py \
  --input data/processed/fall_risk/pose_quality_y8n_v1/cleaned/le2i_home_01_video_1.jsonl \
  --output /tmp/fall_gait_shadow_smoke.jsonl \
  --tcn-checkpoint <frozen-candidate-checkpoint> \
  --tcn-window-frames 16
```

## 7. 必报指标和报告结构

每个正式实验报告必须包含：

- 输入标签、manifest、split、姿态、数据集、配置、代码和 checkpoint hash。
- 当前协议状态及 `test_pose_read`/`test_evaluated`。
- 窗口、动作段、source/sample/split group、时长和拒识统计。
- Accuracy、balanced accuracy、precision、recall、specificity、F1、ROC-AUC；正样本足够时报告 PR-AUC。
- Brier、log loss、ECE 和 reliability diagram。
- 数据源、B02/B03/B04、视角、低光、遮挡、下肢缺失和助行器分层；缺元数据时标不可评估。
- raw 与两窗口确认后的正常背景误报次数和合法 `/hour` 分母。
- cluster bootstrap 95% CI、三个 seed 均值/标准差/最差值。
- P50/P95/P99 延迟、内存、模型参数量和 checkpoint 大小。
- false positive、false negative、拒识和跟踪/姿态失败案例。
- 与规则、结构化模型和历史候选的可比性边界。
- 明确写出模型不能证明的结论。

## 8. 测试与验证命令

修改 Python 后先跑窄范围测试：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_gait_training_audit.py \
  tests/test_gait_model_training.py \
  tests/test_gait_action_pretraining.py \
  tests/test_fall_risk_gait.py -q
```

涉及运行时 checkpoint/fallback 时补跑：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_service_api.py \
  tests/test_fall_runtime_fingerprint.py \
  tests/test_fall_risk_pipeline.py -q
```

阶段完成后运行完整测试：

```bash
conda run -n eldercare-ai python -m pytest -q
```

所有新 CLI 必须验证 `--help`。命令失败时记录完整命令、退出码、关键输出和根因；不得删测试、放宽门禁或使用临时 `PYTHONPATH` 绕过。

## 9. 最终交付清单

Codex 最终必须交付：

```text
1. 当前事实与 blocker 摘要
2. 变更文件清单及设计理由
3. gait_training_manifest.json
4. gait_training_audit.md
5. gait_training_blockers.md
6. train/validation-only 确定性数据集及 hash
7. E0/E1 baseline 和 E2 smoke 报告
8. 数据门禁通过后的三 seed 比较、校准和失败案例
9. 获授权后的一次性 test bundle
10. shadow、延迟、内存和 fallback 证据
11. 实际运行的测试/烟测命令与结果
12. 尚未解除的人工或外部阻塞
```

若只完成 P0-P3，必须明确写“训练基础设施与 provisional smoke 完成，正式训练仍被数据门禁阻塞”。不得用旧 checkpoint、高 validation 点估计或 test 未读取来暗示候选已经可部署。
