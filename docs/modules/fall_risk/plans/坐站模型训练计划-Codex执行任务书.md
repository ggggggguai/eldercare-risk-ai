# 坐站模型训练计划：Codex 执行任务书

> 日期：2026-08-10
> 适用对象：具有本仓库和终端访问权限的 Codex 代理
> 任务性质：训练路线和执行门禁，不是已实现能力、冻结数据发布或正式模型结果

## 1. 任务目标

请在仓库 `/Users/guai/Documents/project/挑战杯/algorithm` 中，将坐站分支从“预裁剪动作 clip 的存在性/方向分类”推进为“连续姿态流中的坐站事件定位与能力特征提取”。

首要交付是 `sit_stand_event_localization_v1`：

```text
连续 cleaned pose
  -> 定位 sit_to_stand / stand_to_sit 的 onset 和 offset
  -> 输出方向、置信度、持续时间和输入质量
  -> 在有独立监督时输出失败尝试、代偿和站稳时间 proxy
  -> 规则 baseline 继续作为主路径、安全覆盖和低质量 fallback
```

完成标准不是再次得到一个高 clip F1，而是形成可追溯、无泄漏、只用 train/validation 选模、可在连续背景上计算事件 F1 和 FP/hour、能够安全 shadow 运行的候选模型。

本轮主模型固定为轻量多头因果 TCN。MS-TCN++ 只在逐帧相位标签门禁通过后作为增强对照；ST-GCN++ 只在因果 TCN 已形成可信连续基线、且同输入对照能回答明确问题时启动。不要同时铺开三个深度模型。

## 2. 当前事实快照

2026-08-10 使用当前工作区运行：

```bash
conda run -n eldercare-ai python scripts/audit/audit_sit_stand_training.py
```

得到以下事实：

| 项目 | 当前值 | 结论 |
|---|---:|---|
| `source_split_id` | `splitv3_e71a045eb58489f43dc5fd11` | 启动时必须重读，不得硬编码 |
| 审计状态 | `provisional` | 不能称为 frozen/formal |
| 显式连续背景标签 | `0` | 不能计算合法 FP/hour |
| `linked_event_id` | `0` | clip 与连续事件尚未建立绑定 |
| 经确认的真实 onset/offset | `0` | 不能训练或评价边界定位 |
| `event_presence_proxy_ready` | `true` | 只允许 provisional clip proxy |
| `event_localization_ready` | `false` | 正式连续定位训练被阻塞 |
| `action_type_formal_ready` | `false` | 不训练正式 normal/slow/failed 三分类 |
| `phase_segmentation_ready` | `false` | 不训练 MS-TCN++ 相位模型 |
| `functional_proxy_ready` | `false` | 不输出临床功能风险结论 |
| test 姿态读取 | `false` | 必须继续锁定 |

当前 A03/A04 正常坐下和起立样本较多，但 B06 全部不是 primary，C01 primary 只在 train；这不能支持正式困难起身分类。现有历史报告记录的候选 clip TCN validation presence F1 为 `0.986`，但 validation 的 `1,308/1,409` 个事件来自 NTU，Fall Detection 2017 F1 只有 `0.545`。该结果只说明预裁剪动作 clip 可拟合，不能外推到连续事件定位、老人居家域或真实误报率。

历史报告引用的 processed dataset 和 checkpoint 当前不在工作区。Codex 不得把报告中的路径存在性当作已验证；若需要复现，只能按当前 split/hash 在新目录重建，并将结果继续标记为 `development_provisional`。

## 3. 必须继承的边界

- 所有 Python、脚本和测试命令必须使用 `conda run -n eldercare-ai python ...`。
- v2 根标签与 v3 训练标签是两套独立契约；坐站专项事件标签不得反向覆盖两者。
- 坐站分支只输出跌倒风险模块内部证据，最终事件仍为 `module=fall_risk`；不新增综合模块或跨模块动作协调。
- 当前运行主路径是 `sit-stand-risk-rule-v0.1`。候选模型默认 checkpoint 必须保持 `null`，直到冻结评审明确批准。
- `sit_stand_risk_score` 是局部能力线索，不是医学诊断、临床量表或未来跌倒概率。
- 未标注背景不能自动变成 negative；不可见、多人身份不明、严重遮挡、边界不可靠或跟踪冲突必须进入 `ignore/uncertain`。
- test 不参与数据检查、困难样本挖掘、阈值、校准、早停、返工或模型选择。
- 不下载大型数据/权重、不安装依赖、不修改环境或 `AlgorithmEvent` 契约，除非用户另行批准。

## 4. 开始前必须阅读

完整读取并以当前文件和机器产物为准：

```text
AGENTS.md
docs/README.md
docs/architecture/算法工程骨架.md
docs/interfaces/算法事件输出接口.md
docs/tasks/README.md
docs/modules/fall_risk/README.md
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/plans/坐站模型训练方案.md
docs/modules/fall_risk/plans/算法链路剩余模型训练计划-比赛效果优先.md
reports/fall_risk/sit_stand_event_v1/README.md
reports/fall_risk/training-labels-v3-validation.json
reports/reproducibility/dataset_and_split_versions.md
configs/training/sit_stand_event_v1.yaml
data/annotations/fall_risk/action_labels_v3.jsonl
data/splits/fall_risk/training_labels_v3/split.json
```

使用 `rg --files` 和 `rg` 检查坐站审计、数据准备、训练、评估、规则运行时和测试入口。旧方案中“尚无训练 builder/训练脚本”的描述已经过时；当前事实以代码和 2026-08-03 之后的报告为准，并在本任务 P0 中同步修正文档。

## 5. 总体阶段与依赖

| 阶段 | 内容 | 当前能否执行 | 退出条件 |
|---|---|---|---|
| P0 | 当前 hash、split、产物和门禁审计 | 立即执行 | 机器清单、阻塞清单和 test 锁定证据完整 |
| P1 | 连续坐站事件标注契约、校验器和复核队列 | 立即执行基础设施；真值依赖人工 | schema/validator/queue 测试通过，首批人工复核完成 |
| P2 | 专项 split、连续数据集 builder 和确定性构建 | P1 有合格标签后 | 双构建一致，无泄漏，不读取 test 姿态 |
| P3 | 连续事件评估器和同协议规则 baseline | P2 后执行 | event F1、边界误差、方向和 FP/hour 可复算 |
| P4 | 因果多头 TCN 训练、消融和校准 | 开发门禁通过后 | 三 seed 候选优于规则且不以误报换 Recall |
| P5 | 相位/困难动作增强 | 独立标签门禁通过后 | MS-TCN++ 或动作头有稳定净增益，否则停止 |
| P6 | 冻结、一次性 test 和跨域压力评估 | 依赖负责人/保管人 | 不可变 test bundle 和 Go/No-Go 记录 |
| P7 | shadow、延迟、fallback 和晋级评审 | P6 通过后 | 只提交晋级建议，不自动切换主路径 |

命中人工数据阻塞时，只暂停依赖真实标签的训练。Codex 继续完成 schema、校验器、合成 fixture、数据隔离、评估器、规则 baseline 工具、测试和文档，不得用 clip pilot 绕过连续标签门禁。

## 6. P0：建立坐站专项事实清单

先运行只读检查：

```bash
git status --short
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python scripts/audit/audit_sit_stand_training.py
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_sit_stand.py \
  tests/test_sit_stand_model_training.py -q
```

采用测试驱动补强现有审计器，产物固定为：

```text
reports/reproducibility/sit_stand_training_manifest.json
reports/fall_risk/sit_stand_training_audit.md
reports/fall_risk/sit_stand_training_blockers.md
```

审计器必须记录：

1. action labels、manifest、assignments、split、v3 validation、训练配置和后续评估配置的 SHA-256。
2. 当前 `split_id`、是否存在明确 frozen 批准、保护组泄漏、标签/姿态缺失和 test 访问状态。
3. A03/A04/B06/C01 按 tier、partition、dataset、subject/source/sample/split group 的计数。
4. 显式背景时长、连续视频数、已确认事件数、方向、结果、边界精度、双人复核和困难负例覆盖。
5. 当前工作区内 historical dataset/checkpoint 是否真实存在，以及其 metadata 的 split/hash 是否匹配。
6. `event_localization`、`phase_segmentation`、`action_proxy` 和 `functional_proxy` 四个独立门禁，不能用 action type 门禁捆绑或解锁事件定位。
7. `test_pose_read=false`、`test_features_generated=false`、`test_evaluated=false`。

状态只能取：

```text
infrastructure_only
development_provisional
candidate_frozen
test_released
```

缺少显式冻结批准时必须 fail closed 为 provisional。当前预期不得高于 `development_provisional`；若输出更高状态，先视为审计器缺陷。

P0 不重新跑 40 epoch clip pilot。若需要验证历史代码可运行，只允许按当前 split 在新目录重建数据并运行 seed 42、最多 2 epoch 的 smoke；该结果不能进入比赛主表。

## 7. P1：建立连续事件标签契约

### 7.1 新增契约

不要继续往动作 clip 标签中塞连续语义。新增独立坐站事件契约：

```text
configs/data/fall_risk_sit_stand_event_schema_v1.json
data/annotations/fall_risk/sit_stand_event_labels_v1.jsonl
data/annotations/fall_risk/sit_stand_event_review_log_v1.jsonl
src/elderly_monitoring/modules/fall_risk/sit_stand_event_labels.py
scripts/annotation/validate_sit_stand_event_labels.py
scripts/annotation/build_sit_stand_review_queue.py
tests/test_sit_stand_event_labels.py
```

schema、校验器和合成 fixture 可以先完成；真实 JSONL 只有在至少一条人工复核记录存在后才创建。不得创建空标签文件或用规则输出填充，以伪装数据工作已经启动。

每条记录至少包含：

```text
label_id, event_id, asset_id, video_id,
subject_id/source_group_id/sample_group_id/split_group_id,
interval_type = event | explicit_background | ignore,
transition_type = sit_to_stand | stand_to_sit | null,
onset_time, offset_time, boundary_precision,
outcome = completed | interrupted | failed | uncertain,
attempt_count, attempt_intervals,
scene_region, visibility, quality_flags,
review_status, reviewed_by, eligibility,
source_action_label_ids
```

`explicit_background` 必须是人工确认的时间区间，不得由“没有事件标签”推导。`seat_off_proxy`、`seat_contact_proxy`、扶物和站稳时刻允许为 `null`；不可见时不能强制补值。

### 7.2 首批复核顺序

Codex 生成确定性复核队列和帧/时间索引，不替人填写真值。人工复核按以下顺序：

1. Fall Detection 2017、CaucaFall、LE2I 和 UR Fall 的真实连续片段，优先解决当前最差来源。
2. 每个事件保留前后背景，标注弯腰、蹲下、跪地、躺下、跌倒、床边转移和快速可控坐下。
3. NTU 只用于正常动作预训练和压力测试；抽样核实 clip 边界，不能让其继续主导 validation。
4. validation/test 的事件边界双人复核；train 至少抽查 20%，有分歧时记录两位标注和裁决。
5. 多视角同一动作、同一人员相邻片段和重复内容绑定同一保护组。

### 7.3 两级数据门禁

开发门禁用于开始单 seed 模型开发，不等于冻结：

- train 每个方向至少 40 个经确认独立事件，validation 每个方向至少 20 个。
- train/validation 各覆盖至少 3 个来源组，validation 不能由 NTU 单一主导。
- validation 至少包含弯腰、蹲下、跌倒、床边转移四类困难负例，每类不少于 10 个独立保护组。
- 至少有可计算 FP/hour 的显式连续背景；不足 5 camera-hour 时只报告事件指标，不报告稳定误报率。

候选冻结门禁沿用比赛计划的更高要求：

- validation 和 test 各至少 50 个独立坐站正事件；两个方向各不少于 20 个。
- validation/test 各覆盖至少 3 个数据来源、15 个保护组，并单列老人域。
- 弯腰、蹲下、跪地/地面活动、跌倒/快速下降、床边转移/躺下、快速可控坐下六类困难负例，在 validation/test 各至少 20 个独立保护组；任一类未覆盖则在报告中明确阻塞并禁止冻结。
- validation/test 分别至少 20 camera-hour 经复核连续背景，才能把 FP/hour 用作正式选择或结论。
- test 标签、姿态和指标由保管人管理；Codex 在候选冻结前不得读取。

样本数门槛是内部研发口径，不是赛事或临床标准。达不到时如实标记 `development_provisional`，不通过复制窗口或降低保护组粒度补数量。

## 8. P2：专项 split 与连续数据集

### 8.1 split

新增坐站专项 split builder，复用 v3 的人员/来源保护信息，但不直接把统一动作 split 当成连续事件 frozen split：

```text
src/elderly_monitoring/modules/fall_risk/sit_stand_event_split.py
scripts/annotation/build_sit_stand_event_split.py
tests/test_sit_stand_event_split.py
data/splits/fall_risk/sit_stand_event_v1/
  assignments.jsonl
  split.json
```

至少验证 subject、source、sample、content、linked action event 和相邻时间段不跨 partition。test 只输出标签/保护组计数，不读取 pose、不生成 tensor。

### 8.2 连续输入契约

现有 `[16,14,7]` clip 张量保留为历史兼容对照。连续主线采用滚动因果输入：

```text
target_fps = 8.0
context_sec = 8.0
window_frames = 64
joints = 14
base_channels = [x, y, dx, dy, quality, image_y, valid]
timing = [delta_t, frame_mask]
```

使用 8 秒上下文是为了覆盖缓慢起身、失败重试和站稳过程；4 秒窗口容易截断高风险事件。保留 `image_y` 或等价的相机坐标垂直位移，因为纯髋中心归一化会消掉坐站最关键的整体升降信号。速度必须使用真实时间差；缺帧、长 gap 和插值不能伪造运动。

在同一连续 validation 上增加 4 FPS/4 秒兼容消融。如果目标运行流长期只能提供 4 FPS，模型必须显式降级或选择 4 FPS 候选，不能静默把 8 FPS 训练模型用于低采样输入。

实现：

```text
src/elderly_monitoring/modules/fall_risk/sit_stand_continuous.py
scripts/prepare/prepare_sit_stand_continuous_dataset.py
tests/test_sit_stand_continuous.py
```

数据集输出：

```text
data/processed/fall_risk/sit_stand_continuous_v1/<split-id>/<build-id>/
  dataset.npz
  samples.jsonl
  metadata.json
  assignments.jsonl
  split.json
  preparation_report.json
```

必须在两个全新目录做确定性双构建。语义产物和 hash 必须一致；不能用 `--overwrite`。每个事件和每段背景的窗口总权重分别归一化，避免长视频或密集滑窗主导训练。

P2 回归测试至少覆盖：

- 时刻 `t` 的张量不包含未来观测。
- test pose 文件即使存在也不会被打开。
- 事件和显式背景之外的区间不会生成 negative。
- 同一事件/人员/重复内容不跨 partition。
- 缺帧、重复时间戳、长 gap、track 切换和低质量窗口的 fail-closed 行为。
- 整体垂直位移没有被中心化抹除。
- 双构建的样本顺序、数组和 metadata hash 一致。

## 9. P3：连续评价与规则 baseline

新增独立的坐站事件评估协议和入口：

```text
configs/evaluation/sit_stand_event_v1.provisional.yaml
src/elderly_monitoring/modules/fall_risk/sit_stand_event_evaluation.py
scripts/evaluate/evaluate_sit_stand_events.py
tests/test_sit_stand_event_evaluation.py
```

评估器必须在 score 阈值变化时重新进行同视频、同方向的一对一事件匹配，并输出：

- event Precision、Recall、F1、PR-AUC。
- direction macro-F1。
- onset/offset 中位绝对误差、P90 误差和 boundary IoU。
- 显式背景上的 FP/hour；没有合法时长分母时返回 unavailable。
- 困难负例误报率、质量拒识率和拒识后的有效覆盖。
- 按 dataset、scene、view、老人域、质量和 transition_type 分层结果。
- 以人员或保守保护组为单位的 bootstrap 95% CI。
- matches、false positives、false negatives、excluded、threshold curve 和失败案例索引。

先在合成 fixture 上测试完全匹配、方向错误、边界偏移、重复预测、跨视频、ignore 区间、无背景分母和低质量拒识。再在同一连续 validation 上运行当前规则 `extract_sit_stand_events`，形成 S0 baseline。规则和模型必须共享完全相同的真值、匹配器和连续背景分母。

冻结前预注册匹配阈值、事件合并间隔、确认帧数、边界容差、bootstrap 单位和主选择阈值。不要看到结果后改变 IoU 或容差。

## 10. P4：模型训练主线

### 10.1 模型与输出头

主候选为轻量因果 TCN，共享编码器后使用独立头：

```text
frame state head: background / sit_to_stand / stand_to_sit
boundary head: onset / offset
event confidence head: presence confidence
optional attempt head: 只有 attempt 标签达到门禁后启用
```

逐帧 state 只使用真实事件区间和显式背景监督；`ignore/uncertain` 帧 loss mask 为 0。方向和事件头不从质量字段学习类别捷径。质量、valid、插值和 gap 主要用于 mask、加权和拒识，并必须做去质量通道消融。

### 10.2 实验矩阵

| 编号 | 实验 | 启动条件 | 目的 |
|---|---|---|---|
| S0 | 当前规则状态机 | P3 完成 | 主路径 baseline |
| S1 | 结构化 Logistic/HMM | P2/P3 完成 | 小数据可解释连续 baseline |
| S2 | 历史 candidate clip TCN | 仅一次当前 split smoke | 验证旧链路，不参与连续主排名 |
| S3 | 8 FPS/8 秒因果 TCN，frame + direction | 开发数据门禁通过 | 最小连续学习模型 |
| S4 | S3 + onset/offset 辅助头 | 边界监督合格 | 主候选 |
| S5 | S4 + attempt/quality proxy 头 | 对应真值合格 | 验证功能特征净增益 |
| S6 | MS-TCN++ 相位模型 | 逐帧相位双人标注门禁通过 | 降低相位碎片的离线增强对照 |
| S7 | ST-GCN++ | S4 达线且骨架拓扑问题仍未回答 | 单 seed 强对照 |

S4 是默认比赛候选。S6 不是天然因果模型，若不能满足在线推理语义，只能作为离线研究对照，不能直接接实时主路径。S7 单 seed 相对 S4 的 validation PR-AUC、最差来源 F1 或边界误差没有至少 3 个百分点/等价明确增益时立即停止，不扩展到三 seed。

### 10.3 训练顺序

1. seed 42、2 epoch 合成或小样本 smoke，验证 loss、mask、反向传播、checkpoint 恢复和独立 evaluator。
2. 只在开发门禁通过后运行 seed 42 完整 pilot。
3. seed 42 同时优于 S0/S1 且没有明显来源捷径后，再运行 `42/43/44`。
4. 早停主指标为 validation event F1，但必须同时满足 Recall、FP/hour 和最差来源下限。
5. 阈值、事件合并、温度缩放/等距回归只用 train/validation 或 OOF 预测拟合。
6. 每轮只改变一个因素；保留 run config、环境、代码、输入和 checkpoint hash。

训练损失优先使用 class/source balanced cross-entropy 或 focal loss、boundary BCE 和方向 loss。先通过事件级权重和来源均衡解决相邻窗口/NTU 主导，不通过复制少数事件制造表面平衡。

### 10.4 必做消融

- 4 FPS/4 秒与 8 FPS/8 秒。
- clip presence 分类与连续事件定位。
- 单 frame-state 头与加入 boundary/direction 多任务头。
- 去掉 `image_y`/整体垂直位移。
- 去掉腕部与 support proxy。
- 去掉 quality 通道但保留 mask；验证质量捷径。
- NTU 预训练/不预训练，以及 NTU 内与跨来源结果。
- 规则、Logistic/HMM、因果 TCN 在同一连续协议上的比较。

## 11. 指标和 Go/No-Go

最低保留线沿用比赛效果计划：

| 指标 | 最低保留线 | 冲奖目标 |
|---|---:|---:|
| 坐站事件 Recall | `>= 0.85` | `>= 0.92` |
| 坐站事件 F1 | `>= 0.85` | `>= 0.92` |
| 方向 macro-F1 | `>= 0.85` | `>= 0.92` |
| onset/offset 中位绝对误差 | `<= 0.75 s` | `<= 0.50 s` |
| 最差来源 F1 | `>= 0.75` | `>= 0.85` |
| 跌倒/弯腰/蹲下等困难负例误报率 | `< 10%` | `< 5%` |

同时要求：

- 相对 S0 规则 event F1 提升至少 5 pp，或在 Recall 不下降时 FP/hour 明显下降。
- 三 seed event F1 标准差不高于 0.03。
- 95% CI、最差来源和老人域结果必须并列报告；总体高分不能掩盖跨来源失败。
- 低质量拒识后必须报告覆盖率，不能通过大量拒识制造高分。

Go：连续 validation 达到最低线、来源分层无不可接受退化、合法背景误报受控、输入和 fallback 契约完整。
No-Go：仍只在预裁剪 NTU clip 高分，或 Recall 提升依赖大量误报，或边界/背景/老人域证据不足。No-Go 时保留规则主路径，只展示 provisional 功能实验。

## 12. P5：相位和困难动作增强

这一阶段不是 P4 的前置条件。

- 只有逐帧相位规范、双人一致性、各核心相位覆盖和三分区保护组门禁通过后，才训练 MS-TCN++。
- 只有 `training_ready.action_type=true`，且 normal/slow/failed 在 train/validation/test 均有 primary 独立组后，才训练正式动作 proxy 头。
- 没有独立 STS-5/TUG/SPPB 或纵向功能真值时，`functional_proxy_score` 必须保持 unavailable。
- duration 可由经验证的边界计算；failed attempts、支撑和站稳时间只能在有对应监督或明确规则 proxy 时输出，并保留 `_proxy` 语义。

相位或动作增强相对 S4 没有稳定改善边界、最差来源或能力特征误差时停止，不因网络更复杂而保留。

## 13. P6：冻结和一次性 test

候选冻结前保存：

```text
checkpoint + checkpoint_sha256
label/schema/manifest/split/pose/config/code/environment hashes
joint/channel/FPS/context/normalization contract
thresholds + event merge policy + calibration artifact
validation predictions + metrics + CI + failure cases
model card + known limitations
```

正式 test 必须由独立保管人提供授权记录，至少绑定 `release_id`、`approved_by`、候选 hash、split/hash、评估配置、运行时间和唯一 run ID。缺少授权、hash 漂移、候选未冻结或评估协议仍 provisional 时，评估入口必须失败关闭。

test 只运行一次。结果低于门槛时记录 No-Go，不读取 test 失败案例返工同一版本。外部压力评估单列老人、沙发/床边/卫生间、视角、遮挡、辅具、低光和摄像机移动，不与内部 test 混成一个总体数字。

## 14. P7：shadow 与运行接入

新增模型 predictor 和运行适配时采用测试驱动，默认行为保持规则：

```text
checkpoint = null                 -> rule_baseline
model valid + contract matched    -> model shadow + rule comparison
model unavailable/error/timeout   -> rule_baseline
input quality insufficient        -> unavailable，不是零风险
```

shadow 输出至少保留：

```text
model_score, fallback_score, score_source, fallback_reason,
transition_type, onset/offset, duration,
quality_coverage, model_version, protocol_status,
model_rule_disagreement
```

不修改 `AlgorithmEvent` 公共字段契约。先离线真实 RGB 视频回放，再运行至少 24 小时开发 shadow；候选冻结后目标为 72 小时。报告 P50/P95 推理耗时、端到端延迟、CPU/GPU/内存、NaN/Inf、异常、重复事件、跨 epoch 状态污染、规则 fallback 和 FP/hour。服务端总延迟必须满足现有 `target_latency_ms=1000` 预算。

涉及姿态或实时视频链路时，除单元/集成测试外运行真实视频烟测；使用项目要求的 YOLOv8-Pose 输入，且不得因模型失败跳过规则 fallback。

## 15. 计划日程

| 日期 | Codex 工作 | 人工/负责人依赖 | 交付 |
|---|---|---|---|
| 08-10 至 08-12 | P0 审计、schema、validator、review queue、合成 fixture | 确认标注口径 | 审计/阻塞报告和可执行标注包 |
| 08-13 至 08-16 | split、连续 builder、评估器、规则 baseline 工具 | 连续事件与背景复核 | 可确定性构建和同协议 S0 |
| 08-17 至 08-18 | 数据 QC、开发门禁判断、单 seed smoke | 解决标注分歧 | Go/No-Go；不足则停止真实训练 |
| 08-19 至 08-20 | S1-S4 seed 42 pilot | 无 | 主候选与失败分析 |
| 08-21 至 08-22 | 通过门禁后跑三 seed、消融、校准 | 冻结候选确认 | validation 比较和模型卡 |
| 08-23 | 候选冻结评审 | test 保管人与负责人批准 | 冻结申请或明确 No-Go |

日程服从数据门禁，不为赶日期降低标注、split、背景分母或 test 隔离标准。

## 16. 测试与验证命令

开发中先运行最窄测试：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_sit_stand_event_labels.py \
  tests/test_sit_stand_event_split.py \
  tests/test_sit_stand_continuous.py \
  tests/test_sit_stand_event_evaluation.py -q
```

再运行现有坐站回归：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_fall_risk_sit_stand.py \
  tests/test_sit_stand_model_training.py -q
```

修改 Python 行为后，条件允许时运行完整测试：

```bash
conda run -n eldercare-ai python -m pytest -q
```

计划中的新 CLI 在实现前不得声称可运行。实现完成后，所有 `--help`、合成 smoke、双构建、训练和评估命令都要原样写入报告，并使用版本化输出目录；禁止覆盖被报告引用的旧产物。

## 17. Codex 执行与汇报规则

开始执行时先输出不超过 10 项的简短计划，然后立即处理所有不受人工阻塞的工作。功能和缺陷修复优先先写失败测试，再做最小实现；遇到失败先保留命令、输出、输入 hash 和根因，不通过改测试、删样本、放宽门禁或临时 `PYTHONPATH` 绕过。

每完成一个阶段按以下格式汇报：

```text
阶段与结论
事实 / 推断 / 建议
修改文件与新增产物
执行的确切命令
测试和指标结果
数据与 test 门禁状态
未解决风险和下一检查点
```

立即停止相关动作并请求用户决定：

- 需要删除、覆盖、移动用户文件、正式标签或冻结产物。
- 需要下载数据/权重、安装依赖、修改环境或公共接口契约。
- 需要读取 test 真值、重划已冻结 split 或放宽预注册协议。
- 人工标注、授权、老人域或连续背景证据不足以进入下一阶段。
- 出现两个会改变标签语义、模型因果性或正式评估口径的有效方案。
- 同一错误连续尝试两次仍未解决。

最终不能只写“训练完成”。每一项结论必须能回指标签、split、配置、预测、指标、hash、测试或失败案例；未通过门禁的模型统一标记为 `development_provisional` 或 `provisional_shadow`。
