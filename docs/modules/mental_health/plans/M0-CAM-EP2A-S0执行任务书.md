# M0-CAM-EP2A-S0 执行任务书

> 历史执行记录：本任务保存 S0 初始实现契约，不再限制当前分段器调优。现行 B01+B02 联合优化、边界 refinement 和完成门见 [M0-CAM-5D-T1](M0-CAM-5D-T1-B01B02分段器联合优化.md)。

日期：2026-08-16

状态：`implementation_complete_b01_b02_manual_evaluation_available`

本任务状态：`M0-CAM-EP2A-S0 completed (implementation-only)`

历史后继记录：`old_next=independent participant/setup validation`

现行任务：`W5D-01 B01+B02 segmenter joint development`

推荐执行模型：`GPT-5.6 Codex / gpt-5.6-sol`

推荐智商（推理强度）：**最高（max）**

后续状态说明：本文主体保留 S0 开发时点的 historical implementation-only 任务要求；B01/B02 manual boundary、policy freeze、旧 B02 holdout 和 candidate-inclusive S2A v2 均已在后续完成。实时状态只看 `docs/tasks/README.md` 与 W5D 总任务书。

## 1. 为什么现在做这条代码线

M0-CAM-EP1A 已完成“人工/CVAT/whole-clip 边界内的冻结 shape forward”，M0-CAM-EP1B 的批量 evaluator、主/条件指标、coverage、时长/分组/purposeful 诊断和失败清单也已完成。EP1B 尚未完成的不是代码，而是独立人工 truth：当前 15 条 P01/L01/R01/H01/H04/H05 标签仍是 AI 预标注，不能形成正式 camera pilot。

人工复核可以继续进行，但不应阻塞所有工程开发。下一条代码线只实现连续 tracking 到 **episode boundary proposal** 的最小 producer：

```text
tracking JSONL + media sidecar
  -> 复用技术 track split / accepted-observation 语义
  -> 运动开始与持续停留的简单状态机
  -> proposed / uncertain boundary intervals
  -> 人工复核与接受
  -> 既有 wandering-camera-episode-boundary-v1
  -> EP1A frozen shape forward
```

这里的关键是 **proposal，不是 truth**。S0 可以在没有人工 boundary 的情况下完成代码、合成测试和真实 truth-free smoke；但不能计算 boundary F1/tIoU，不能选择最优阈值，不能把 proposal 直接写成现有正式 boundary schema，也不能声称 automatic episode 已完成。

## 2. 本阶段目标

实现一个可复用、可检查、非覆盖的连续 episode boundary proposal 工具，优先解决以下工程问题：

1. 连续 tracking 中何时出现一个候选 locomotion episode；
2. track end、长 gap、疑似 ID switch 等技术硬断点如何统一传播；
3. 短暂停、转向、pacing 折返、lapping 闭环和 random 换向为何不能触发切段；
4. 持续静止如何形成软结束候选，运动恢复时如何撤销过早结束；
5. 证据不足时如何输出 `uncertain/manual_review_required`，而不是强造精确边界；
6. proposal 如何通过人工接受后再转换为 EP1A 已支持的正式 boundary 输入。

本阶段只完成 producer 工具。后续已拆成 EP2A-S1A evaluator implementation-only 与 EP2A-S1B human-boundary evaluation/freeze，均不是 S0 的完成门。

## 3. 研究语义与硬边界

### 3.1 episode 是完整 locomotion bout

人工协议仍是事实源：

- 大约连续迈出三步、行走已经成立时开始 episode，并把人工边界回退到实际第一步附近；
- 数秒停顿、犹豫、转身、折返、闭环仍属于同一 episode；
- 同一地点持续停留约 15 秒、坐下、出画、track 结束或明确开始另一活动时结束；
- 三步和约 15 秒是人工标注协议锚点，不是已经校准的自动阈值。

摄像头 tracking 没有可靠步态相位和活动语义，因此 S0 只能产生候选。自动 onset 可以使用持续位移证据并回溯到该运动 run 的开始，但必须保留 proposal reason 和不确定性。

### 3.2 shape 与 boundary 分离

以下信号禁止作为切段条件：

```text
方向改变
pacing 的端点反转
lapping 的闭环或回到起点
random 的多次换向
起点终点距离接近
冻结 shape 模型的预测类别或概率
```

否则 producer 会把真正的 pacing/lapping/random 拆成多个 direct，进一步放大当前 camera subtype 坍缩。boundary producer 不读取 `observable_pattern`、`purpose_context`、EP1B truth 或冻结模型输出。

### 3.3 purpose 不进入 producer

打电话来回走、搬运、锻炼、找东西和无明显任务的往返，在 shape 上都可能形成同样的 locomotion episode。S0 不尝试判断 purpose，也不输出 alert、risk、diagnosis 或 `AlgorithmEvent`。

### 3.4 proposal 与正式 boundary 分离

未经人工接受的输出必须使用新的独立 schema，例如：

```text
wandering-camera-episode-boundary-proposal-v1
```

不得直接冒充 EP1A 已使用的：

```text
wandering-camera-episode-boundary-v1
```

正式 boundary 只能由显式人工复核/接受步骤产生。不得修改原 CVAT XML、EP1B truth 或既有 proposal；接受结果写入新文件/新目录。

## 4. 复用边界

### 4.1 必须复用

- `camera_adapter.CameraObservation`；
- `camera_adapter.group_observations()`；
- `camera_adapter.weighted_bucket_observations()`；
- `camera_qc` 中现有 accepted-observation 与技术 hard-break 语义；
- EP1A 已有的 identity 字段、boundary importer 和非覆盖输出原则。

`camera_qc._split_observations()` 当前已经统一处理：

```text
long_internal_gap
suspected_id_switch
height_position_discontinuity
```

新 producer 不得复制一份略有不同的 gap/ID-switch 算法。实现时可以把这段逻辑提升为向后兼容的公开 helper，让既有 Camera QC 与新 producer 共用；也可以在最小范围内直接复用现有 helper，但必须用测试锁定只有一份权威语义。

### 4.2 不得复用成错误语义

- 不调用旧 `camera_episode.py` 的 40 秒窗口合并来产生 pre-model boundary；
- 不把 `run_camera_qc()` 的固定 40 秒窗口当完整 episode；
- 不读取 EP1B evaluator 的 truth 或预测结果来决定切点；
- 不改 `prepare_camera_window()`、冻结模型权重、0.5 threshold、四类定义或训练统计。

## 5. 最小代码边界

建议新增：

```text
configs/modules/wandering_camera_episode_boundary_proposal_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_episode_boundary.py
scripts/wandering/propose_camera_episode_boundaries.py
tests/test_wandering_camera_episode_boundary.py
reports/mental_health/wandering_camera_episode_boundary_proposal_v1/
```

只在共用技术 split 确实需要公开时，对 `camera_qc.py` 与对应测试做小型向后兼容重构。不要修改其他模块，不要新增 receipt、PORTABLE、逐文件 hash、字节对齐或 exact-schema 攻击审计。

## 6. 输入契约

CLI 最少接收：

```text
--tracking-jsonl
--media-sidecar
--output-dir
--config（可选，默认使用项目 v1 config）
```

输入身份必须沿用 adapter 的完整 camera isolation key：

```text
source_group_id
source_video_id
device_id
setup_id
stream_epoch
track_id
```

tracking 行至少使用：

```text
frame_id
timestamp_sec
track_id
bbox
track_confidence
```

输入必须先由既有 adapter 校验。多个 scope/track 不得合并；每个 scope 独立运行状态机。输出目录必须不存在，失败时不得留下看似成功的完整 bundle。

## 7. proposal 输出契约

每条 proposal 至少包含：

```text
schema_version
proposal_id
source_group_id
source_video_id
device_id
setup_id
stream_epoch
track_id
start_sec
end_sec_exclusive
duration_sec
proposal_status
start_reason
end_reason
hard_break_reasons
manual_review_required
source_observation_count
accepted_observation_count
source_bucket_count
producer_config_id
```

推荐状态：

```text
proposal_status = proposed | uncertain | rejected_by_qc
```

其中：

- `proposed` 表示 producer 有足够运动/结束证据，可进入人工复核，不表示真值正确；
- `uncertain` 表示 onset、offset、活动切换或 tracking 证据不足，必须人工处理；
- `rejected_by_qc` 表示没有形成可用 locomotion proposal，但仍要在 summary/diagnostic 中保留原因。

推荐输出目录：

```text
proposals.jsonl
summary.json
diagnostics.jsonl
README.md
```

S0 不自动生成 `wandering-camera-episode-truth-v1`，不写 shape/purpose/evaluation_role，也不直接调用冻结 shape forward。后续可以新增一个薄的人工 acceptance/export 工具，但它必须要求显式 reviewed decision，并输出到新路径。

## 8. hard 与 soft boundary signal

### 8.1 hard signal

hard signal 直接关闭当前 open episode 或隔离技术片段：

- scope/track 的可观察开始与结束；
- `long_internal_gap`；
- `suspected_id_switch`；
- `height_position_discontinuity`；
- session/stream epoch 结束；
- adapter/QC 能明确证明的出画/重入或 track replacement。

若 hard signal 附近 observation 稀疏，边界仍可关闭，但 proposal 应标记 manual review，不能把最后一个检测点自动解释成真实行为结束。

### 8.2 soft signal

soft signal 只形成候选，不应立即切段：

- 一段连续位移达到 locomotion-start 候选；
- body-height-normalized 位移持续低于 stationary 候选；
- 静止持续到 protocol-derived dwell candidate；
- 静止后重新运动；
- 可观察活动改变，但 tracking-only 无法可靠识别时只保留人工复核提示。

默认配置可以记录 `about_15_seconds` 的 protocol-derived 候选值，但字段和报告必须说明 `validated=false`。在独立人工 boundary 到位前不得根据 H02/H03 的视觉印象或模型输出调到“最好看”的数值。

## 9. 状态机

最小状态机固定为：

```text
idle -> open -> closing -> closed
           \-> uncertain
closing -> open      （短暂停后恢复运动，不切段）
任何状态 --hard break--> closed/uncertain
```

### `idle`

- 尚无有效 locomotion episode；
- 低置信度 observation、原地抖动或短暂检测不直接打开 episode；
- 累积持续位移证据，一旦成立，onset 回溯到该 movement run 的第一个 accepted bucket。

### `open`

- locomotion episode 已开始；
- 转弯、折返、闭环、方向变化保持 `open`；
- 短暂停只进入 `closing` 候选，不立即输出 boundary。

### `closing`

- 记录 stationary candidate 的起点；
- dwell 未达到候选时恢复运动，返回 `open`，整个 bout 保持一条 episode；
- dwell 达到候选后，以 stationary run 开始附近作为候选 offset，并进入 `closed`；
- tracking 证据不足时进入 `uncertain`，而不是选择一个伪精确切点。

### `closed`

- 只输出 proposal；
- 下一次独立 locomotion start 回到 `open`；
- 同一输出中的 proposal 区间不得重叠或倒序。

### `uncertain`

- 保留可用的时间范围、原因和人工复核要求；
- 不静默删除，不伪装成 `proposed`，也不进入无 truth 的指标。

## 10. 配置原则

配置只保存实际改变 producer 行为的少量值：

```text
bucket_seconds
minimum_track_confidence
movement_start_min_buckets
movement_start_min_displacement_body_heights
stationary_max_displacement_body_heights
stationary_dwell_candidate_seconds
minimum_episode_duration_seconds
maximum_internal_gap_buckets（应与共用 QC 语义一致）
```

要求：

- 单位明确，优先使用 bbox-height/body-height normalized displacement，避免纯像素阈值随透视变化；
- 15 秒只标为 protocol-derived candidate；
- 不包含 shape class、binary threshold 或模型 checkpoint；
- 不在无 truth 数据上网格搜索、自动寻优或根据输出观感反复调参；
- 配置中任何默认值都必须有合成行为测试，且报告说明未校准。

## 11. 测试驱动矩阵

先写聚焦测试，再实现状态机。至少覆盖：

| 场景 | 预期 |
| --- | --- |
| 静止 -> 连续移动 -> 持续静止 | 产生 1 条 proposed episode |
| 移动中短暂停后恢复 | 仍为 1 条 episode，不多切 |
| pacing 多次端点反转 | 不因反转切段 |
| lapping 多圈闭环 | 不因回到起点切段 |
| random 多方向换向 | 不因方向变化切段 |
| A->B 后立即返回，无持续停留 | 保持 1 条 bout 或 uncertain，不强拆为两个 direct |
| 两段移动之间有足够持续停留 | 产生 2 条 proposal |
| long internal gap | 复用 QC hard break，不能跨 gap 合并 |
| suspected ID switch | hard split，并传播 reason |
| height-position discontinuity | hard split，并传播 reason |
| 只有低于最小置信度的边缘点 | 不能虚假打开或延长 episode |
| track 一开始就在运动 | onset 为首个可接受 run，标明 track-start reason |
| track 结束时仍在运动 | 关闭 proposal，并要求复核真实 offset |
| observation 不足 | 输出 uncertain/rejected_by_qc，不静默删除 |
| 两个 track/scope | 状态隔离，不交叉合并 |
| 重跑同一输出目录 | 明确拒绝覆盖 |
| producer 输出被当正式 boundary 读入 | 在未显式人工接受时拒绝或保持 schema 隔离 |

测试不能依赖 shape truth，也不能加载冻结模型。

## 12. 真实数据 smoke

当前已存在 H02/H03 的 tracking JSONL 与 media sidecar，可用于 truth-free smoke：

```text
tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h02/
tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h03/
```

N01 当前仓库/工作树中尚未发现可用 tracking/sidecar；若后续准备完成，可作为自然非行走/短行走的附加 smoke。不得为了满足任务书伪造 N01 输入或标签。

truth-free smoke 只检查：

- CLI 能完整运行；
- 输出 identity、排序、区间和 reason 合法；
- proposal 不跨技术 hard break；
- 短暂停/换向没有明显造成碎片化；
- uncertain、QC rejection 和 coverage 计数完整；
- 输出目录非覆盖、可复跑。

没有人工 boundary 时禁止报告：

```text
boundary precision/recall/F1
tIoU
onset/offset error
漏切率/多切率
automatic-boundary + shape F1
最优阈值
```

## 13. 后续 EP2A-S1A/S1B

S0 完成后不直接跳到真实评价。后续分为：

### EP2A-S1A：evaluator implementation-only

1. 读取 proposal bundle、独立 human boundary、media sidecar 和分组 binding；
2. 复用 `camera_development.match_episodes_one_to_one()`，不得复制最大基数 matching；
3. 实现 candidate/proposed-only P/R/F1、matched tIoU/onset/offset、coverage、split/merge 和 failures；
4. 只用 synthetic fixture 验证，没有人工 boundary 时不产生真实成绩；
5. 详细契约见 [M0-CAM-EP2A-S1A 执行任务书](M0-CAM-EP2A-S1A执行任务书.md)。

### EP2A-S1B：human boundary evaluation/freeze

1. proposal 与独立人工 boundary 一对一匹配；
2. 报 boundary precision、recall、F1；
3. 报匹配 tIoU、onset absolute error、offset absolute error；
4. 报漏切、过切和碎片化；
5. 报 `proposed/uncertain/rejected_by_qc` coverage；
6. automatic proposal + frozen-shape 必须使用 producer 原始端点；人工修正端点只能单列 semiautomatic，人工 truth 端点属于 oracle；
7. 与 oracle-boundary EP1B、semiautomatic corrected 结果分开呈现，不能互相替代；
8. 只在 development boundary 上选择 producer 参数，随后冻结并留出独立 session/setup。

## 14. 实施顺序

1. 只读核对当前 dirty/untracked 工作树、editable install 和现有 Camera QC/EP1A/EP1B 代码；
2. 先补状态机与 hard-break 的合成测试；
3. 实现独立 config/module/CLI，不加载模型；
4. 必要时把 `camera_qc` 技术 splitter 提升为单一公开 helper，并保持旧行为回归；
5. 运行最窄 boundary tests；
6. 运行 EP1A/EP1B 与 adapter/QC 相邻回归；
7. 用 H02/H03 在全新 Git 外目录运行 truth-free smoke；
8. 输出 proposal bundle、诊断与准确状态；
9. 更新模块 README、任务表和报告；
10. 人工 truth 未到位时停止在 S0，不自动进入 S1、EP2B、EP2C 或 EP3。

## 15. 验证命令要求

所有 Python/pytest 必须通过 WSL `eldercare-ai`，不得使用 Windows 裸 Python、裸 pytest 或 base 环境。开始前确认 editable 安装指向当前仓库：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

建议验证顺序：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_adapter.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_inference.py
```

若改动触及共享 Camera QC，再跑全部 `tests/test_wandering_camera_*.py`。真实 smoke 必须使用新输出目录；不得删除或覆盖 EP1A/EP1B 既有 evidence。

## 16. 状态与完成门

### S0 可以写 implementation complete 的条件

- proposal config/module/CLI/tests 已实现；
- hard split 复用单一 Camera QC 语义；
- 状态机不会因转向、折返、闭环或 random 换向切段；
- proposal schema 与正式 accepted boundary schema 隔离；
- uncertainty/manual-review 路径完整；
- H02/H03 truth-free smoke 可复跑；
- EP1A、EP1B 与相邻 camera 回归无退化；
- 文档和报告准确写明没有 boundary truth 与自动边界性能。

人工 boundary 尚未完成时，状态必须是：

```text
m0cam_ep2a_s0_implementation_status=completed
m0cam_ep2a_human_boundary_status=pending
m0cam_ep2a_evaluation_status=pending
m0cam_ep2a_data_status=partial_truth_free_smoke_only
m0cam_ep2a_status=not_completed
```

2026-08-16 已达到 S0 implementation-only 完成门。随后功能复审发现并修复两个缺口：同一 0.5 秒 bucket 内技术 hard break 造成相邻 proposal 重叠；持续慢速平移可能被当成 stationary dwell。新增回归后，相邻技术段用 accepted observation 时间中点限界，慢漂移关闭降为 `uncertain + stationary_candidate_drift`。修复后的 H02/H03 两轮 truth-free smoke 共 4 条 `uncertain` proposal，逐文件一致；旧版 H03 的两条 `proposed` 不再是当前证据。当前证据与机器结果见 [EP2A-S0 报告](../../../../reports/mental_health/wandering_camera_episode_boundary_proposal_v1/README.md)。该记录不改变上述 `human_boundary/evaluation pending` 与顶层 `not_completed` 状态。

只有独立人工连续 boundary、producer、boundary 指标、automatic-boundary + shape 评价和可复跑报告全部到位后，才可在后续 S1 写：

```text
m0cam_ep2a_status=completed
```

### 必须停止并报告

- 需要修改人工 truth、原 XML、冻结模型、0.5 threshold 或类别定义；
- 需要根据 shape 预测反向决定切点或标签；
- 需要访问 sealed/未授权数据；
- 需要实现告警、风险、`AlgorithmEvent`、EP2B/EP2C/EP3；
- 需要修改其他模块、共享环境依赖、commit 或 push；
- 没有人工 boundary 却准备报告 tIoU/F1、调阈值或声称 automatic boundary 完成。

## 17. 已执行提示词（历史）

以下提示词对应已经完成的 S0，不再是当前代码任务。当前执行提示词见 [M0-CAM-EP2A-S1A 执行任务书](M0-CAM-EP2A-S1A执行任务书.md#14-可直接交给开发-ai-的目标提示词)。

```text
你是本项目 M0-CAM-EP2A-S0 的主开发 AI。你的任务是在不等待 EP1B 人工 truth 完成的前提下，实现 continuous episode boundary proposal 的 implementation-only 工具线；同时保持 EP1B 作为并行人工证据任务，绝不能把 AI 预标注或 proposal 冒充 truth。

AI 设置：
- 模型：优先 GPT-5.6 Codex / gpt-5.6-sol
- 智商（推理强度）：最高（max）
- 工作方式：持续执行到实现、测试、真实 truth-free smoke、报告和文档同步完成；普通局部代码/路径/timeout/测试问题自主修复。

开始前完整阅读并严格执行：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/README.md
4. docs/modules/mental_health/plans/徘徊识别技术文档2.md
5. docs/modules/mental_health/plans/M0-CAM-EP1B执行任务书.md
6. docs/modules/mental_health/plans/M0-CAM-EP2A-S0执行任务书.md
7. reports/mental_health/wandering_camera_episode_eval_v1/README.md
8. reports/mental_health/wandering_camera_episode_eval_v1/VERIFICATION.md
9. camera_adapter.py、camera_qc.py、camera_episode_import.py、camera_episode_inference.py 及对应测试源码

当前事实：
- EP1A 已完成 oracle-boundary episode inference。
- EP1B evaluator 已完成并通过回归，但 15 条 P01/L01/R01/H01/H04/H05 标签仍是 AI 预标注，S00 只有 3 条既有 CVAT direct truth。
- 本段目标提示词记录 S0 开发时点的历史前置状态；后续 EP1B 已在 B01/B02 manual-CVAT oracle-boundary descriptive pilot 范围完成，实时状态只看任务表和证据交接。
- S0 implementation 与后续 S1B 真实评价仍分层；S1B 已冻结 B01 policy 并完成 B02 一次性视频级 holdout，EP2A 总状态仍 not_completed。
- H02/H03 已有 tracking JSONL 与 media sidecar，可用于 truth-free smoke；N01 当前没有可用 tracking/sidecar，不得伪造。

实现目标：
1. 新增轻量 config/module/CLI/test，输入 tracking JSONL + media sidecar，输出独立 wandering-camera-episode-boundary-proposal-v1 proposal bundle。
2. 复用 CameraObservation、group_observations()、weighted_bucket_observations() 和 camera_qc 的 accepted-observation/technical hard-break 语义。long gap、suspected ID switch、height-position discontinuity 只能有一份权威实现；必要时做向后兼容的公共 helper 小重构，不能复制算法。
3. 实现 idle/open/closing/closed/uncertain 状态机。持续位移打开候选，持续静止形成软关闭，短暂停后恢复必须回到 open。
4. 转向、pacing 折返、lapping 闭环、random 换向、起终点接近和 shape 模型预测都不得作为切段信号。
5. proposal 与 accepted boundary 分离。未经人工确认的输出不得写成 wandering-camera-episode-boundary-v1，不得修改原 XML/truth；输出必须包含 reason、uncertain 和 manual_review_required。
6. 不读取 shape/purpose truth，不加载冻结模型，不训练、不调 0.5、不改类别定义。15 秒只作为 protocol-derived、validated=false 的候选，不在无 truth 的 H02/H03 上优化阈值。
7. 先写合成状态机测试，覆盖短暂停、pacing、lapping、random、两段移动、hard gap、ID switch、低置信度边缘点、多 track 隔离、uncertain 和 non-overwrite。
8. 用 H02/H03 在全新 Git 外目录做 truth-free smoke，只报告 proposal 数、区间、reason、coverage、uncertain/QC 状态和可复跑性；不得报告 boundary F1/tIoU/onset-offset/漏切多切或 automatic shape 性能。
9. 修改 Python 后先跑最窄 boundary test，再跑 EP1A/EP1B 与相邻 camera 回归；若触及共享 Camera QC，最后跑全部 test_wandering_camera_*.py。
10. 更新 docs/tasks/README.md、徘徊识别技术文档2.md、mental-health README、docs 索引和新的 report。没有人工 boundary 时准确写 implementation_complete + human_boundary_pending + evaluation_pending + partial_truth_free_smoke_only，m0cam_ep2a_status 保持 not_completed。

硬边界：
- 保护当前 dirty/untracked 工作树，不 reset、checkout、clean、stash，不覆盖既有输出。
- 所有 Python/pytest 使用 WSL eldercare-ai，先确认 editable install 指向当前仓库。
- 不修改 EP1B truth/XML，不按预测改标签，不访问 sealed 数据，不修改其他模块。
- 不实现 alert、risk、AlgorithmEvent、EP2B、EP2C、EP3；不扩展 receipt、PORTABLE、逐文件 hash、字节对齐或 exact-schema 攻击审计。
- 不 commit/push。

最终必须明确报告：实际实现文件、状态机语义、测试命令和实际结果、H02/H03 truth-free smoke 覆盖、proposal/accepted boundary 的证据边界、仍缺的人工 boundary 与 S1 评价内容。不要用测试通过或 proposal 生成冒充 automatic boundary 已验证。
```
