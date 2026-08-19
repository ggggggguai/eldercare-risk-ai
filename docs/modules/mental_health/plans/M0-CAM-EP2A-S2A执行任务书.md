# M0-CAM-EP2A-S2A 执行任务书

> 历史执行记录：candidate-inclusive v2 的 schema/状态语义继续复用；“B01/B02 不再调整、等待独立证据”已失效。当前自动 episode+shape 闭环见 [M0-CAM-5D-T2](M0-CAM-5D-T2自动Episode与轨迹识别闭环.md)。

日期：2026-08-16

任务：`automatic-proposal + frozen-shape runner implementation-only`

当前状态：

```text
m0cam_ep2a_s2a_implementation_status=completed
m0cam_ep2a_s1b_policy_status=frozen_on_b01_development
m0cam_ep2a_s1b_video_holdout_status=completed_once_on_b02
m0cam_ep2a_automatic_shape_coverage_status=17_of_18_on_reused_b02_post_hoc_development
m0cam_ep2a_s2a_data_status=implementation_plus_b02_post_hoc_development_evaluation
m0cam_ep2a_status_at_historical_snapshot=not_completed
```

范围裁剪：2026-08-16 初始实现为 proposed-only 最小 bridge；2026-08-17 经负责人明确授权形成 candidate-inclusive v2。本文中描述 `uncertain -> boundary_uncertain + skip` 的旧任务要求只作历史追溯，已被第 15 节 v2 完成记录取代；不扩展为新的 evaluator 或审计包。

历史交接曾要求等待独立 participant/setup；该要求现已失效。B01+B02 可反复用于 development，当前按 [W5D-T2](M0-CAM-5D-T2自动Episode与轨迹识别闭环.md)继续自动 episode+shape 闭环。

## 1. 为什么曾需要这条代码线

EP2A-S0 已生成独立 `wandering-camera-episode-boundary-proposal-v1`，EP2A-S1A 已能把 proposal 与独立人工 continuous boundary 做一对一评价。但是当前还缺一条不会污染 oracle 语义的桥接线：

```text
S0 original proposal endpoint
  -> episode-internal QC/preprocessing
  -> frozen TopoWander forward
  -> one proposal, one shape-result row
```

现有 EP1A public builder 不能直接承担这个职责。它只接受 accepted `wandering-camera-episode-boundary-v1`，并固定输出：

```text
status=wandering_m0cam_ep1a_oracle_boundary_ready
evaluation_name=oracle-boundary shape classification
quality_flags includes oracle_boundary
automatic_boundary_inference=false
```

把 S0 proposal 临时改写成 accepted boundary 再调用 EP1A，会把 automatic proposal 冒充人工/oracle boundary，并丢失 `proposal_status/reason_codes/technical_segment_index/manual_review_required`。因此 S2A 必须新增独立 runner 和独立输出 schema，同时只抽取复用 EP1A 已验证的 episode 内 QC、80 点预处理和冻结 forward 核心。

S1B 仍被独立人工 continuous boundary 阻塞。S2A 只关闭代码桥接，不评价性能、不冻结参数，也不把 EP2A 写成 completed。

## 2. 本阶段目标

实现一条 truth-free、非覆盖、可批量复跑的 automatic-proposal shape runner：

```text
S0 proposal bundle
  + original tracking JSONL
  + original wandering-media-v1 sidecar
  + participant/session/camera-setup/clock binding
  + frozen candidate manifest
  -> exact scope and input validation
  -> preserve every proposal row and original endpoint
  -> proposed: episode QC + 80-point preprocessing + frozen forward
  -> uncertain: preserve uncertain status/reason; episode QC + frozen forward as candidate diagnostic
  -> rejected_by_qc: unavailable, skip model
  -> fresh proposal-shape bundle
```

本阶段必须让以后能够按 `proposal_id` 把三类证据严格连接起来：

```text
S0 proposal row
  + S1A proposal-to-human-boundary match
  + S2A proposal-to-frozen-shape result
```

这样 S1B 才能在不替换 endpoint 的条件下计算 automatic proposal + frozen-shape 结果。

## 3. 科学与标签语义

### 3.1 Proposal 不等于 truth 或 accepted boundary

- 不生成 `wandering-camera-episode-boundary-v1`；
- 不调用 acceptance converter；
- 不修改 proposal 的 start/end、status、reason 或 ID；
- 不读取人工 boundary、shape truth、purpose truth、AI 预标注或 XML；
- 输出只说明“冻结模型在原始 proposal 区间上的行为”，不说明 proposal 正确。

### 3.2 Shape 与 purpose 继续分离

S2A 只运行冻结 shape forward。它不推断目的、告警、临床综合征或风险。未来人工 truth 到位后：

- independent binary head 的 wandering-like 结果仍是 camera 主要 shape 结果；
- direct/pacing/lapping/random 四分类仍是 subtype diagnostic；
- purposeful pacing/lapping/random 仍保持原 shape，只能在独立上下文/告警层处置。

### 3.3 三种结果永远分开

```text
oracle-boundary shape
  = independent human endpoint -> EP1A frozen shape

automatic proposal shape
  = S0 original endpoint -> S2A frozen shape

semiautomatic/manual-corrected shape
  = human accepted/corrected endpoint -> future separate path
```

任何人工修正后的 endpoint 都不得进入 S2A automatic 结果。任何 human endpoint 都不得覆盖 proposal endpoint。

## 4. 输入契约

### 4.1 Proposal bundle

只接受 S0 fresh bundle：

```text
summary.json
proposals.jsonl
diagnostics.jsonl
```

proposal schema 必须是：

```text
wandering-camera-episode-boundary-proposal-v1
```

必须逐条保留：

```text
proposal_id
source_group_id/source_video_id/device_id/setup_id/stream_epoch/track_id
technical_segment_index
start_sec/end_sec_exclusive/duration_sec
proposal_status
start_reason/end_reason/reason_codes/hard_break_reasons
manual_review_required
producer_config_id
```

### 4.2 Tracking 与 sidecar

每个 proposal bundle 必须显式绑定其原始 tracking JSONL 和 `wandering-media-v1` sidecar。proposal summary、proposal rows、tracking 和 sidecar 的 source identity 必须一致；不同 source/track/technical segment 不能串线。

runner 不重新运行 detector/tracker，也不复制 Camera QC technical splitter。它只对 proposal 的原始区间复用 EP1A 已有 episode 内 QC/preprocessing。

### 4.3 Batch index

建议一行一个 proposal bundle：

```json
{"bundle_id":"h03-01","proposal_bundle_dir":"...","tracking_jsonl":"...","media_sidecar":"...","participant_id":"P01","session_id":"S01","camera_setup_id":"C6C_OFFICE_01","clock_domain_id":"clock-S01"}
```

最少字段：

```text
bundle_id
proposal_bundle_dir
tracking_jsonl
media_sidecar
participant_id
session_id
camera_setup_id
clock_domain_id
```

重复 `bundle_id/proposal_id`、物理 source 重复绑定、身份漂移、非法路径、重叠 bundle ownership 或非有限区间必须在首次模型 forward 前 fail closed。

### 4.4 Frozen candidate

继续使用 EP1A/primary camera 的固定 config、candidate manifest、模型状态、类别顺序和 0.5 binary threshold。复用现有 loader 和 provenance 检查即可，不新增 receipt、逐文件 hash 或 PORTABLE 闭包。

## 5. Proposal 状态到预测状态的固定映射

### 5.1 `proposed`

使用 proposal 原始 `[start_sec, end_sec_exclusive)`：

1. 选择同一完整 camera scope 和 target track 的 observations；
2. 运行 EP1A 相同的 accepted-observation、technical break、bucket、gap、coverage、motion extent 和预处理检查；
3. 只有 episode QC ready 才调用冻结模型；
4. 模型成功时写 `prediction_status=ready`；
5. episode QC 失败写 `prediction_status=unavailable`；
6. trusted model forward/output 失败写 `prediction_status=inference_error`。

### 5.2 `uncertain`

固定写：

```text
prediction_status=boundary_uncertain
model_invocation_skipped=true
```

保留全部 proposal reason，不因没有概率而删行。uncertain 在未来 automatic end-to-end 主分母中是 pipeline miss，不是可静默过滤的记录。

### 5.3 `rejected_by_qc`

固定写：

```text
prediction_status=unavailable
model_invocation_skipped=true
reason includes proposal_rejected_by_qc
```

仍然一 proposal 一结果行，并保留 S0 reason/hard-break/QC context。

## 6. 复用与最小重构边界

不能直接调用 `build_camera_episode_inference_bundle()`，也不能伪造 accepted boundary 文件。建议在 `camera_episode_inference.py` 内抽取一个明确的低层 helper，职责仅为：

```text
validated interval identity
  + adapter input
  + explicit input kind / quality flag
  -> episode-internal QC and prepared tensors
```

要求：

- EP1A public `prepare_camera_episode()` 继续先验证 accepted boundary，并以 `oracle_boundary` 调用该 helper；
- S2A 先验证 proposal，再以 `automatic_boundary_proposal` 调用该 helper；
- helper 不决定 proposal 是否 accepted，不生成 truth，也不写 summary；
- EP1A 的 prediction schema、status、quality flags、D01/S00 概率和 summary 语义保持不变；
- S2A 输出不得出现 `oracle_boundary` quality flag；
- 冻结 forward 复用现有 `predict_camera_episode()` 或更低层可信 forward，不复制模型概率计算、类别映射或 0.5 threshold。

若直接复用现有函数会要求在输出中伪造 `boundary_source`，应先完成上述小型重构，而不是在 S2A 写完后字符串替换字段。

## 7. 独立输出 bundle

建议新增：

```text
proposal_shape_predictions.jsonl
summary.json
README.md
```

需要额外行级问题表时可增加：

```text
failures.jsonl
```

建议 schema：

```text
wandering-camera-episode-proposal-shape-prediction-v1
wandering-camera-episode-proposal-shape-summary-v1
```

每条 prediction 至少包含：

```text
proposal_id and full camera/group scope
technical_segment_index
original proposal start/end/duration/status/reasons
prediction_status and prediction_reason_codes
episode QC counts/coverage/reasons
quality_flags including automatic_boundary_proposal
binary/subtype/four-class probabilities when ready
predicted_binary/predicted_pattern when ready
binary_decision_threshold=0.5
probability_calibrated=false
model_invocation_skipped
```

summary 至少声明：

```text
input_boundary_kind=automatic_proposal
proposal_schema_version=wandering-camera-episode-boundary-proposal-v1
accepted_boundary_schema_emitted=false
proposal_modified=false
proposal_endpoint_modified=false
truth_labels_consumed=false
human_boundary_consumed=false
purpose_context_consumed=false
models_retrained=false
threshold_search_performed=false
binary_decision_threshold=0.5
automatic_boundary_validated=false
shape_performance_metrics_available=false
algorithm_event_emitted=false
risk_or_alert_decision_emitted=false
```

输出目录已存在时拒绝；构建失败不得留下看似完整的半成品目录。输入 proposal/tracking/sidecar 在运行前后保持不变。

## 8. 指标和证据边界

S2A 不输出 accuracy、precision、recall、F1、tIoU、onset/offset error、漏切、多切或 alert 指标。只输出：

- proposal/prediction 状态计数；
- proposed 的 episode-QC/ready coverage；
- model invoked/skipped counts；
- ready prediction 的描述性类别计数；
- row-level unavailable/boundary_uncertain/inference_error 原因。

H02/H03 的 2026-08-16 四条 `uncertain` truth-free skip smoke 是旧策略历史记录，只验证当时的身份、端点、状态保留、模型跳过和非覆盖。当前 v2 已由聚焦测试和复用 B02 的 post-hoc development/exploratory 实跑覆盖 proposed+uncertain forward；仍不能制造/修改 proposal 状态，也不能称为 independent holdout 或 accepted-boundary 性能。

## 9. 最低测试矩阵

先写测试，再实现。至少覆盖：

1. 只接受 S0 proposal schema，accepted boundary loader 仍拒绝 proposal；
2. proposal bundle、summary、diagnostic、tracking、sidecar 和 batch binding 完整校验；
3. `proposed` 使用原始 endpoint 进入 episode QC/preprocessing；
4. proposed ready 才调用 frozen forward，输出独立 binary head、subtype 和 four-class 概率；
5. `uncertain` 保持原 status/reason/manual-review，QC ready 后 forward 并标为 candidate diagnostic；不丢行、不升级为 accepted boundary；
6. `rejected_by_qc -> unavailable + skipped`，不丢行；
7. proposed 内部 QC failure 保持 unavailable，不能因没有概率被删除；
8. trusted forward failure 保持 inference_error；
9. 每个输入 proposal 恰有一个输出结果，ID、full scope、technical segment 和 endpoint 完全一致；
10. 输出没有 `oracle_boundary`，EP1A 输出继续包含 `oracle_boundary`；
11. 不读取 boundary/truth/purpose/XML，CLI 不提供这些输入参数；
12. 首次 forward 前拒绝重复 ID、identity drift、非法 enum、非有限数和错误 candidate binding；
13. 原输入字节不变、fresh output、atomic build、non-overwrite 和 deterministic output；
14. EP1A builder、D01/S00 oracle regression 和 EP1B evaluator 不退化；
15. S0、S1A、primary camera 与全部 `tests/test_wandering_camera_*.py` 回归；
16. CLI `--help` 和 compileall。

## 10. 建议文件边界

```text
configs/modules/wandering_camera_episode_proposal_shape_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_episode_proposal_inference.py
scripts/wandering/run_camera_episode_proposal_inference.py
tests/test_wandering_camera_episode_proposal_inference.py
reports/mental_health/wandering_camera_episode_proposal_inference_v1/
```

允许对 `camera_episode_inference.py` 做一次受测试保护的小型核心提取。不要修改 `camera_qc.py`、模型权重、训练代码、S0 proposal 或 S1A matching 行为。

## 11. 验证顺序

所有 Python/pytest 使用 WSL `eldercare-ai`：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_proposal_inference.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_adapter.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_*.py
```

随后运行 synthetic proposed fixture 和 H02/H03 truth-free uncertain smoke。两类 smoke 必须分开报告。

## 12. 完成门

只有独立 config/module/CLI/tests、proposal 一对一输出、状态保留、原 endpoint frozen forward、EP1A oracle 语义不变、synthetic proposed smoke、H02/H03 uncertain smoke 和 camera 回归全部完成后，才可写：

```text
m0cam_ep2a_s2a_implementation_status=completed
m0cam_ep2a_human_boundary_status=pending
m0cam_ep2a_evaluation_status=pending
m0cam_ep2a_s2a_data_status=synthetic_contract_plus_partial_truth_free_smoke_only
m0cam_ep2a_status=not_completed
```

没有独立人工 continuous boundary、development 参数冻结和留出评价时，不得写：

```text
m0cam_ep2a_evaluation_status=completed
m0cam_ep2a_status=completed
```

## 13. 禁止事项

- 不重训、不调参、不改冻结模型、类别、0.5 threshold 或 preprocessing 数学；
- 不把 proposal 转换或导出为 accepted boundary；
- 不修改 proposal endpoint/status/reason，不根据 shape prediction 回改 proposal；
- 不读取或生成 shape truth、purpose truth、人工 boundary、AI 预标注或 XML；
- 不实现 semiautomatic acceptance/correction、alert、risk 或 `AlgorithmEvent`；
- 不启动 EP2B、EP2C、EP3、C4 sealed 或产品接入；
- 不扩展 receipt、PORTABLE、逐文件 hash、字节对齐或攻击型 exact-schema 审计；
- 不修改跌倒或其他模块，不升级共享依赖；
- 不 reset、checkout、clean、stash，不覆盖既有输出，不 commit/push。

## 14. S2A 历史执行提示词（已完成，不再作为当前任务）

```text
你是本项目 M0-CAM-EP2A-S2A 的主开发 AI。请持续完成“automatic-proposal + frozen-shape runner implementation-only”代码线，重点是功能实现、状态不丢失和真实可复跑验证，不要把精力转向新的哈希、PORTABLE 或发布审计。

AI 设置：
- 模型：优先 GPT-5.6 Codex / gpt-5.6-sol
- 智商（推理强度）：最高（max）
- 工作方式：持续执行到实现、验证、报告和文档同步全部闭环；普通局部代码、路径、timeout 和测试问题自主排查修复。

开始前完整阅读并严格执行：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/README.md
4. docs/modules/mental_health/plans/徘徊识别技术文档2.md
5. docs/modules/mental_health/plans/M0-CAM-EP1B执行任务书.md
6. docs/modules/mental_health/plans/M0-CAM-EP2A-S0执行任务书.md
7. docs/modules/mental_health/plans/M0-CAM-EP2A-S1A执行任务书.md
8. docs/modules/mental_health/plans/M0-CAM-EP2A-S2A执行任务书.md
9. reports/mental_health/wandering_camera_episode_v1/README.md 与 VERIFICATION.md
10. reports/mental_health/wandering_camera_episode_boundary_proposal_v1/README.md 与 VERIFICATION.md
11. reports/mental_health/wandering_camera_episode_boundary_eval_v1/README.md 与 VERIFICATION.md
12. camera_episode_inference.py、camera_episode_boundary.py、camera_primary_inference.py 及对应测试源码

当前事实：
- EP1A oracle-boundary episode inference 已完成，public builder 和 quality flags 固定绑定 oracle 语义。
- EP1B evaluator 与 B01/B02 manual-CVAT oracle descriptive pilot 已完成；旧 15 条 AI 预标注未进入正式成绩。
- EP2A-S0 producer 已完成；当前 H02/H03 truth-free smoke 共 4 条 proposal，全部 uncertain，不是 automatic-boundary 性能。
- EP2A-S1A evaluator 已完成并经功能复审；其证据仍是 synthetic_contract_only，人工 boundary/evaluation pending。
- 本历史提示词的执行任务是 S2A。S1B 继续等待独立人工 continuous boundary，但不阻塞该工具线实现。

必须实现：
1. 新增独立 proposal-shape config/module/CLI/tests。输入为 S0 proposal bundle、原 tracking、原 sidecar、分组 binding 和现有冻结 candidate；不接收 truth/XML/purpose。
2. 一条 proposal 恰好输出一条 shape-result。完整保留 proposal_id、full scope、technical_segment_index、原 start/end/status/reasons/manual_review_required，禁止修改 endpoint。
3. proposed 使用原始 endpoint 进入与 EP1A 相同的 episode 内 QC、bucket、80 点弧长预处理和 frozen forward；仅 QC ready 才调用模型。
4. proposed+uncertain 保留原 endpoint/status/reason 并进入 episode QC，ready 后运行 frozen model；uncertain 结果只作 candidate diagnostic，不能提升为 accepted boundary。rejected_by_qc 固定保留为 unavailable 并跳过模型；内部 QC unavailable 和 inference_error 也不得静默删除。
5. 输出使用独立 wandering-camera-episode-proposal-shape-* schema，quality flag 为 automatic_boundary_proposal，绝不能写 oracle_boundary，也不能生成 wandering-camera-episode-boundary-v1。
6. 复用现有 candidate loader、概率计算、独立 binary head、subtype/four-class order 和 0.5 threshold。不得复制模型 forward 或从 four-class argmax 反推 binary。
7. 为避免伪造 accepted boundary，可在 camera_episode_inference.py 做一次最小核心提取：EP1A 继续验证 accepted boundary 并保持原输出；S2A 以明确 automatic proposal kind 调用共享 episode-QC/preprocessing helper。D01/S00 oracle 结果必须不变。
8. 输出 fresh、atomic、non-overwrite、deterministic；输入 proposal/tracking/sidecar 不变。不要增加新的 receipt/hash/PORTABLE 工程。
9. S2A 不计算 P/R/F1、tIoU、起止误差、漏切多切或告警指标。只报告状态、QC/ready coverage、model invoked/skipped 和描述性预测计数。
10. synthetic 与 B02 post-hoc development fixture 必须覆盖 proposed+uncertain frozen-forward 合同；H02/H03 旧 skip smoke 只作历史追溯。任何输入都不得改 proposal 状态或伪造 truth。
11. 先跑最窄测试，再跑 EP1A/S0/S1A/EP1B 与 primary camera 相邻回归，最后跑全部 tests/test_wandering_camera_*.py。所有 Python/pytest 必须使用 WSL eldercare-ai，并确认 editable install 指向当前仓库。
12. 更新根 README、docs 索引、docs/tasks、mental-health README、技术路线、新报告和本任务书完成记录。只有 S2A 工具与 smoke 全部到位才写 S2A implementation completed；EP2A 总状态仍 not_completed。

证据边界：
- automatic proposal + shape 必须使用 producer 原始 endpoint。
- 人工修正 endpoint 只能属于未来 semiautomatic；human truth endpoint 只能属于 oracle。
- proposal/测试通过/synthetic 预测都不是 boundary 性能或真实 camera shape 准确率。
- 不训练、不调参、不改标签、不根据预测修改 proposal，不访问 sealed 数据，不实现 alert/risk/AlgorithmEvent，不启动 EP2B/EP2C/EP3，不修改其他模块，不 commit/push。
- 保护 dirty/untracked 工作树：不 reset、checkout、clean、stash，不覆盖既有输出；与现有改动冲突时先读懂并在其上工作。

最终必须明确报告：实际实现文件、共享核心如何保持 EP1A oracle 语义、每种 proposal 状态如何处理、测试命令和真实结果、synthetic proposed 与 H02/H03 truth-free 覆盖、模型实际调用/跳过计数、没有计算的指标、证据边界，以及 S1B 仍缺的人工 boundary、参数冻结和独立 participant/session/setup 评价。
```

## 15. 完成记录

2026-08-16 已按打薄范围完成 proposed-only 历史 v1：

- 新增薄 config/module/CLI/tests 和三文件输出 bundle；不新增 evaluator、failure/audit/receipt/PORTABLE 制品；
- `camera_episode_inference.py` 抽出 interval QC/preprocessing 与 ready-interval forward helper。EP1A 仍先验证 accepted boundary，D01 12 条和 S00 三条 prediction bytes 与既有回归逐文件一致；
- synthetic 3 条 proposal 一对一输出，`proposed` 实际调用固定 frozen model 1 次，`uncertain/rejected_by_qc` 在旧策略下跳过 2 次；
- H02/H03 使用 S0 v2 四条原 `uncertain` proposal，4/4 保留 endpoint/status/reason，旧策略 forward/skipped=`0/4`；
- 聚焦 S2A `2 passed`、EP1A `11 passed`、相邻 camera `113 passed`、全部 camera `604 passed, 1 warning`；CLI help 与 compileall 通过；
- 没有读取 truth/XML/purpose，没有生成 accepted boundary，没有训练、调参或计算任何性能指标。
- 本阶段未 commit/push，也未 reset、checkout、clean 或 stash 当前工作树。

2026-08-17 经负责人明确授权完成 candidate-inclusive v2：

- policy ID=`m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2`；
- proposed+uncertain 的原 endpoint 进入 episode QC，ready 后调用同一 frozen model；uncertain 原 status/reason/manual-review 保持不变，并标为 `boundary_uncertain_candidate_diagnostic`；
- rejected_by_qc 继续 unavailable/skip；没有生成 accepted boundary；
- S2A/S2B 聚焦测试 `8 passed`；B02 21 条 proposal 实际 forward/skip=`17/4`；
- B02 post-hoc development 18 条 all-shape-eligible 中 ready=`17/18`，binary accuracy/macro-F1=`0.944444/0.970588`；17 条 candidate-ready binary=`17/17`；
- proposal/prediction endpoint 与 binding mismatch=`0/0`；模型、0.5 threshold、类别、producer/matching、人工 truth 均未改；
- B02 v2 因看过数据后建立并复跑，固定为 development/exploratory，不是 held-out。

v2 关闭了旧 `2/18` 送模 coverage 缺口，但仍缺新 participant/setup 上的预声明独立验证，所以 `m0cam_ep2a_status=not_completed`。详见 [S2A 报告](../../../../reports/mental_health/wandering_camera_episode_proposal_inference_v1/README.md)、[v2 B02 评价](../../../../reports/mental_health/wandering_camera_b02_candidate_inclusive_development_v1/README.md)和[当前证据交接](M0-CAM-EP1B-EP2A-S1B人工证据交接.md)。
