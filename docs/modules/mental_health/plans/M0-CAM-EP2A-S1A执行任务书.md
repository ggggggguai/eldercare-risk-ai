# M0-CAM-EP2A-S1A 执行任务书

> 历史执行记录：本任务保存 boundary evaluator 契约和旧证据，不再以独立 participant/setup 阻塞开发。当前 evaluator 用于 B01+B02 可重复 development 调参，见 [M0-CAM-5D-T1](M0-CAM-5D-T1-B01B02分段器联合优化.md)。

日期：2026-08-16

状态：`implementation_completed_b01_b02_manual_evaluation_available`

历史后继记录：`candidate-inclusive S2A v2 completed; old_next=independent evidence`

现行任务：`W5D-01 B01+B02 evaluator-driven segmenter development`

推荐执行模型：`GPT-5.6 Codex / gpt-5.6-sol`

推荐智商（推理强度）：**最高（max）**

后续状态说明：本文主体保留 S1A 开发时点的 historical synthetic-contract 任务要求；B01/B02 manual boundary、B01 policy freeze、旧 B02 holdout 和 candidate-inclusive S2A v2 均已在后续完成。实时状态只看 `docs/tasks/README.md` 与 W5D 总任务书。

## 1. 任务定位

EP2A-S0 已完成 continuous tracking 到 boundary proposal 的 implementation-only producer，并在功能复审后补齐两个重要边界条件：

- 同一 0.5 秒 bucket 内的技术 hard break 不再形成相互重叠的 proposal；
- 持续慢速平移不再被误当成高置信 stationary dwell，而是保留为 `uncertain + stationary_candidate_drift`。

当前仍没有独立人工 continuous boundary，因此不能计算真实 boundary P/R/F1、tIoU、起止误差、漏切或多切。但是 evaluator 的读取、身份绑定、一对一 matching、指标分母、coverage、split/merge 诊断、失败清单和非覆盖输出都可以先实现并用 synthetic fixture 完成测试。

因此 EP2A-S1 拆成两部分：

```text
EP2A-S1A：boundary evaluator implementation-only
  -> config/module/CLI/tests
  -> synthetic fixture 验证
  -> 无真实 boundary 指标

EP2A-S1B：human-boundary evaluation and development freeze
  -> 独立人工 continuous boundary
  -> development 参数选择与冻结
  -> 留出 session/setup 评价
  -> automatic-boundary + frozen-shape
```

S1A 的目标是让代码不再等待人工标注，但绝不把 proposal、AI 预标注、synthetic fixture 或人工修正后的端点冒充 automatic boundary 证据。

## 2. 本阶段必须实现

新增一条独立、只读、非覆盖的 boundary evaluation 工具线：

```text
S0 proposal bundle
  + independent human boundary JSONL
  + media sidecar
  + participant/session/setup/clock binding
  -> exact identity validation
  -> existing maximum-cardinality one-to-one matching
  -> all-ready-human-boundary denominators
  -> candidate/proposed-only metrics
  -> tIoU + onset/offset errors
  -> status coverage + split/merge diagnostics
  -> unmatched/failure lists
  -> fresh reproducible report bundle
```

S1A 必须完成以下工程能力：

1. 读取一个或多个 S0 proposal bundle；
2. 读取独立人工 boundary view，但不读取 shape/purpose truth；
3. 将 proposal、人工 boundary、media 和分组身份显式绑定；
4. 复用 `camera_development.match_episodes_one_to_one()`，不得复制最大基数 matching；
5. 输出 boundary P/R/F1、matched tIoU、onset/offset error 的计算实现；
6. 显式保留 `proposed`、`uncertain`、`rejected_by_qc` 和人工 `boundary_uncertain`；
7. 输出 unmatched truth、unmatched proposal、split/merge、QC rejection 和低质量匹配失败清单；
8. 在没有真实人工 boundary 时只形成 `synthetic_contract_only` 实现证据，不写真实性能；
9. 保持 proposal 和人工 truth 原文件逐字节不变，不提供 acceptance/export 或 truth 回写功能；
10. 为 S1B 留出明确输入契约，但不自动启动 shape inference、调参或 sealed evaluation。

## 3. 输入契约

### 3.1 Proposal 输入

只接受 S0 产物：

```text
summary.json
proposals.jsonl
diagnostics.jsonl
```

其中 `proposals.jsonl` 的 schema 必须是：

```text
wandering-camera-episode-boundary-proposal-v1
```

每条 proposal 的原始字段和状态必须保留。S1A 不得把它改写为 `wandering-camera-episode-boundary-v1`，也不得删除 `uncertain` 或 `rejected_by_qc`。

### 3.2 独立人工 boundary

人工 boundary 使用 EP1A 已存在的 truth-free boundary view：

```text
wandering-camera-episode-boundary-v1
```

优先复用：

```text
camera_episode_import.load_episode_boundaries()
```

人工文件只能包含边界字段，不能包含 `observable_pattern`、`purpose_context`、模型预测或 proposal 决策。`boundary_source` 可以是现有 `cvat_xml|simplified_jsonl|whole_clip`；S1B 的连续视频正式评价不得用 whole-clip shortcut 冒充人工 onset/offset。

人工 `boundary_status=ready` 才能进入真实 boundary 性能分母。`boundary_uncertain` 不得静默删除：必须计入 truth coverage、原因和失败清单，但不能被当成确定 TP/FN，也不能用于参数选择。

### 3.3 Batch index 与分组

新增最小 batch index，至少绑定：

```text
bundle_id
proposal_bundle_dir
human_boundary_jsonl
media_sidecar
participant_id
session_id
camera_setup_id
clock_domain_id
```

proposal 自带：

```text
source_group_id
source_video_id
device_id
setup_id
stream_epoch
track_id
```

evaluator 将两部分组合成现有 `EPISODE_SCOPE_FIELDS`，再调用共享 matching。不同 participant/session/setup/clock/source/track 的记录永远不能互相匹配。重复 `proposal_id`、重复 `episode_id`、重叠人工 ready boundaries、bundle identity 漂移和跨 batch 物理身份重复必须 fail closed。

## 4. Matching 契约

### 4.1 必须复用既有实现

唯一一对一算法是：

```python
camera_development.match_episodes_one_to_one()
```

S1A 只做字段适配：

```text
proposal_id -> prediction_id
episode_id  -> annotation_id
start_sec/end_sec_exclusive 保持原值
```

不得重新实现 greedy matching、Hungarian matching 或第二套最大流。共享实现已固定以下优化顺序：

```text
maximum match count
maximum total temporal IoU
minimum total onset+offset delta
stable prediction/annotation id
```

### 4.2 显式 provisional matching policy

S1A config 必须显式包含：

```text
policy_id
minimum_temporal_iou
maximum_onset_delta_sec
validated=false
```

实现测试可以使用固定 synthetic 值，但不得把它写成经过真实 development 数据验证的阈值。S1B 只能在 development truth 上选择并冻结 policy；留出 session/setup 结果不得回调用于同版本调参。

### 4.3 两个评价视图

为避免 `uncertain` 被静默删除，同时避免把人工复核候选说成全自动结果，必须同时输出：

1. `all_locomotion_candidates`：`proposed + uncertain` 均作为候选参加一对一 matching。这是 proposal queue 的主要定位能力视图；
2. `proposed_only_conditional`：只使用 `proposal_status=proposed`。这是 producer 自身高置信输出的条件诊断，不得单独冒充最终 automatic-boundary 成绩。

`rejected_by_qc` 不作为 locomotion candidate 参加 matching，但必须保留状态、区间、reason 和与 ready truth 的 overlap 诊断。若 ready truth 只被 uncertain 或 rejected interval 覆盖，在 proposed-only 视图中仍是未召回，不能从 recall 分母删除。

## 5. 指标与分母

### 5.1 Boundary detection

对每个评价视图输出：

```text
ready_truth_support
candidate_support
matched_count
unmatched_truth_count
unmatched_candidate_count
precision
recall
f1
```

定义：

```text
TP = one-to-one matched pair
FP = unmatched candidate
FN = unmatched ready human boundary
```

不得按视频先求平均再隐藏 support。总体、participant、session、camera setup、duration band 均报告原始 support；支持不足时返回 `not_computable`，不能用 0 或空列表伪装为已评价。

### 5.2 Localization

只对 matched pair 输出并汇总：

```text
temporal_iou
onset_signed_error_sec = proposal_start - truth_start
onset_absolute_error_sec
offset_signed_error_sec = proposal_end - truth_end
offset_absolute_error_sec
```

至少给出 count、mean、median、p90 和 max。没有 matched pair 时统一为 `not_computable`。tIoU 和 error 不能对 unmatched truth 或 rejected-by-QC 伪造数值。

### 5.3 Coverage

必须报告：

```text
human boundary ready/uncertain counts
proposal proposed/uncertain/rejected_by_qc counts
manual_review_required count
technical segment count
accepted-observation coverage
observed-bucket coverage
ready truth with proposed overlap
ready truth with uncertain-only overlap
ready truth with rejected-by-qc-only overlap
ready truth with no proposal overlap
```

coverage 与 P/R/F1 分开。高 tracking coverage 不等于高 boundary recall；proposal 数存在也不等于 boundary 正确。

### 5.4 Split/merge diagnostics

除一对一匹配外，用正时间 overlap 构造纯诊断关系：

- 一个 ready truth 与多个 locomotion proposals 重叠：`split_candidate`；
- 一个 locomotion proposal 与多个 ready truths 重叠：`merge_candidate`；
- proposal 与 truth 有 overlap 但未通过 matching policy：`subthreshold_overlap`；
- 相邻 proposal 极短或端点过近：只记录 `fragmentation_candidate`，不自动合并。

这些是失败分类，不得改写 proposal 或 truth。漏切/多切率的正式定义与阈值在 S1B development protocol 冻结后再写；S1A 先输出可复核的原始计数和行级证据。

## 6. 输出 bundle

建议使用独立目录和 schema，不覆盖 S0 或 EP1A/EP1B：

```text
summary.json
metrics.json
matches.jsonl
unmatched_truth.jsonl
unmatched_proposals.jsonl
split_merge_diagnostics.jsonl
failures.jsonl
README.md
```

输出至少声明：

```text
proposal_schema_version
human_boundary_schema_version
matching_policy_id
matching_policy_validated=false|true
truth_source=independent_human|synthetic_fixture
shape_predictions_consumed=false
shape_truth_consumed=false
proposal_modified=false
human_boundary_modified=false
automatic_shape_evaluated=false
models_trained_or_updated=false
```

输出目录已存在时必须拒绝。构建失败不得留下看似完整的半成品目录。

## 7. Failure list

每条 failure 至少包含完整 scope、proposal/truth ID（存在时）、区间、状态、reason 和 failure type。类型至少覆盖：

```text
unmatched_ready_truth
unmatched_proposed_candidate
unmatched_uncertain_candidate
uncertain_only_truth_coverage
rejected_by_qc_truth_coverage
subthreshold_overlap
split_candidate
merge_candidate
large_onset_error
large_offset_error
low_tiou_match
```

错误阈值必须来自 config 并标记是否 validated。S1A synthetic 测试只验证分类行为，不主张科学阈值有效。

## 8. Automatic-boundary + shape 的证据边界

S1A 不运行 shape inference。S1B 后续运行时必须区分三种输入：

```text
oracle-boundary shape:
  使用独立人工 truth endpoint

automatic proposal shape:
  使用 producer 原始 start/end，不能替换成人工端点

semiautomatic/manual-corrected shape:
  使用人工接受或修正后的 proposal，必须单列
```

人工只做“接受/修正”后再送入 EP1A 的结果不是纯 automatic-boundary 成绩。若用 matched truth endpoint 替换 proposal endpoint，结果本质上回到了 oracle boundary，禁止标为 automatic。

EP1B oracle-boundary shape、EP2A boundary localization、automatic proposal + frozen shape、semiautomatic corrected + frozen shape 必须分别报告。

## 9. 测试要求

先写测试，再实现。最窄测试至少覆盖：

1. proposal bundle、summary 和 human boundary exact loader；
2. proposal schema 不能被 accepted boundary loader 接受；
3. proposal/truth 完整 scope 绑定，跨 source/track/session/setup 不匹配；
4. 实际调用共享 `match_episodes_one_to_one()`，最大基数优先用例保持一致；
5. `proposed + uncertain` 主视图与 `proposed_only` 条件视图分母不同且可复算；
6. ready truth 被 uncertain 覆盖时在 proposed-only recall 中仍是 FN；
7. `rejected_by_qc` 不参加 candidate TP，但保留 coverage/failure；
8. human `boundary_uncertain` 可见但不进入确定性能分母；
9. matched tIoU/onset/offset error 的正负号和聚合；
10. unmatched、subthreshold、split、merge 和 fragmentation diagnostics；
11. 分组与时长 support；
12. 空 matched cohort 返回 `not_computable`；
13. 重复 ID、重叠人工 ready boundary、identity drift 和非有限数 fail closed；
14. 原 proposal/truth 文件在运行前后字节不变；
15. deterministic output、fresh directory 和 non-overwrite；
16. CLI `--help`。

不得为了测试方便读取 EP1B shape truth、模型概率或现有 AI 预标注。

## 10. 验证顺序

所有 Python/pytest 使用 WSL `eldercare-ai`，先确认 editable install 指向当前仓库。

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary_evaluation.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py \
  tests/test_wandering_camera_development.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_primary_inference.py

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_*.py
```

CLI smoke 只能使用 synthetic fixture 或明确标记为 human-reviewed 的 boundary。H02/H03 当前只有 truth-free proposal，禁止给它们制造假 boundary 后报告真实成绩。

## 11. 建议文件

最小实现优先放在徘徊模块内：

```text
configs/modules/wandering_camera_episode_boundary_eval_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_episode_boundary_evaluation.py
scripts/wandering/evaluate_camera_episode_boundaries.py
tests/test_wandering_camera_episode_boundary_evaluation.py
reports/mental_health/wandering_camera_episode_boundary_eval_v1/
```

若现有 helper 足够，不修改 `camera_development.py`。确需提升公共适配 helper 时必须保持原 matching 行为和现有测试全部通过。不要修改 `camera_qc.py` 或复制 hard-break 算法。

## 12. 状态与完成门

S1A 实现前状态：

```text
m0cam_ep2a_s1a_implementation_status=not_started
m0cam_ep2a_human_boundary_status=pending
m0cam_ep2a_evaluation_status=pending
m0cam_ep2a_status=not_completed
```

只有 config/module/CLI/tests、synthetic evaluator bundle、非覆盖输出、共享 matching 复用和 camera 回归全部完成后，才可写：

```text
m0cam_ep2a_s1a_implementation_status=completed
m0cam_ep2a_human_boundary_status=pending
m0cam_ep2a_evaluation_status=pending
m0cam_ep2a_data_status=synthetic_contract_only
m0cam_ep2a_status=not_completed
```

**完成记录（2026-08-16，功能复审修正）：** 上述完成门已满足。固定 config、只读 module、CLI、16 条聚焦 tests、synthetic evaluator bundle、非覆盖原子输出和共享 matcher 复用均已落地。复审修复了跨时长带时错误合成 band F1、fragmentation 跨 technical hard break，以及 independent-human 分支缺少 E2E 覆盖三项问题；全部 wandering camera 回归为 `602 passed, 1 warning`，两轮 synthetic CLI 的 8 个输出文件逐字节一致。没有独立人工 continuous boundary，因此这里只关闭 S1A implementation，matching/failure policy 仍 `validated=false`，人工 boundary/evaluation 与 EP2A 总状态保持 pending/not completed。详见 [S1A 报告](../../../../reports/mental_health/wandering_camera_episode_boundary_eval_v1/README.md)。

没有独立人工 continuous boundary 时，不得写：

```text
m0cam_ep2a_evaluation_status=completed
m0cam_ep2a_status=completed
```

EP2A 总完成仍要求：真实人工 boundary、development 参数冻结、独立 session/setup、boundary 指标、failure cases、automatic proposal + frozen-shape 结果和可复跑报告全部到位。

## 13. 禁止事项

- 不重训、不调冻结 shape 模型、不改 0.5 threshold 或类别定义；
- 不根据 proposal 或 shape prediction 修改人工 boundary；
- 不把 AI 预标注、目录名、拍摄脚本或 synthetic fixture 当 truth；
- 不修改原 XML/truth/proposal，不实现 acceptance converter；
- 不运行 automatic shape、alert、risk 或 `AlgorithmEvent`；
- 不启动 EP2B、EP2C、EP3 或 sealed camera；
- 不扩展 receipt、PORTABLE、逐文件 hash、字节对齐或 exact-schema 攻击审计；
- 不修改其他模块或共享依赖，不 commit/push；
- 不 reset、checkout、clean、stash 或覆盖当前 dirty/untracked 工作树和既有输出。

## 14. 已执行的开发 AI 目标提示词（历史）

以下提示词已于 2026-08-16 执行完毕；当前状态以上文“完成记录”为准，不应据此重复启动 S1A 或自动进入 S1B。

```text
你是本项目 M0-CAM-EP2A-S1A 的主开发 AI。请持续完成 boundary evaluator implementation-only 工具线。当前没有独立人工 continuous boundary，因此你的任务是完成代码、合成测试、synthetic evaluator smoke、报告和文档同步；不得生成或宣称真实 boundary 成绩，也不得自动进入 S1B。

AI 设置：
- 模型：优先 GPT-5.6 Codex / gpt-5.6-sol
- 智商（推理强度）：最高（max）
- 工作方式：以功能实现和可复跑验证为优先；普通局部代码、路径、timeout 和测试问题自主排查并修复。

开始前完整阅读并严格执行：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/README.md
4. docs/modules/mental_health/plans/徘徊识别技术文档2.md
5. docs/modules/mental_health/plans/M0-CAM-EP1B执行任务书.md
6. docs/modules/mental_health/plans/M0-CAM-EP2A-S0执行任务书.md
7. docs/modules/mental_health/plans/M0-CAM-EP2A-S1A执行任务书.md
8. reports/mental_health/wandering_camera_episode_boundary_proposal_v1/README.md
9. reports/mental_health/wandering_camera_episode_boundary_proposal_v1/VERIFICATION.md
10. camera_episode_boundary.py、camera_episode_import.py、camera_development.py 及对应测试

当前事实：
- EP1A oracle-boundary shape inference 已实现。
- 本段目标提示词记录 S1A 开发时点的历史前置状态；后续 EP1B 已使用 B01/B02 manual CVAT 完成限定范围的 oracle-boundary descriptive pilot，旧 15 条 AI 预标注未进入正式成绩。
- EP2A-S0 producer 已实现。审计修复后 H02/H03 truth-free smoke 共 4 条 proposal，全部 uncertain；这不是自动边界性能。
- S1A evaluator implementation-only 已完成；后续 S1B 已冻结 B01 policy 并完成 B02 一次性视频级 holdout，实时状态只看任务表和证据交接。

必须实现：
1. 新增 boundary-eval config/module/CLI/tests，读取 S0 proposal bundle、独立 wandering-camera-episode-boundary-v1、media sidecar 和 batch group binding。
2. 复用 camera_development.match_episodes_one_to_one()。只做 proposal_id/prediction_id、episode_id/annotation_id 和 scope 字段适配；不得复制最大基数 matching。
3. 主视图 all_locomotion_candidates 使用 proposed+uncertain；条件视图 proposed_only_conditional 只用 proposed。rejected_by_qc 和 human boundary_uncertain 必须可见，不能静默删除。
4. 输出 ready truth/candidate support、TP/FP/FN、P/R/F1、matched tIoU、onset/offset signed+absolute errors、状态 coverage、分组/时长 support、split/merge/subthreshold 诊断和失败清单。
5. ready truth 只被 uncertain/rejected 覆盖时，在 proposed-only 视图仍必须是未召回；不得因状态不理想从分母删除。
6. 原 proposal/truth 只读且字节不变；输出新目录、原子构建、拒绝覆盖。
7. 输出明确声明 synthetic_contract_only、matching policy validated=false、shape truth/prediction 未读取、模型未训练、automatic shape 未评价。
8. 先写最窄 synthetic tests，再跑 S0/matching/EP1A/EP1B 相邻回归，最后跑全部 tests/test_wandering_camera_*.py。
9. 更新 docs/tasks/README.md、mental-health README、徘徊识别技术文档2.md、docs 索引和新 report。没有人工 boundary 时只写 S1A implementation completed + human_boundary/evaluation pending + EP2A not_completed。

证据边界：
- 不得给 H02/H03 伪造人工 boundary 或输出真实 P/R/F1/tIoU。
- S1A 不运行 frozen shape inference。
- 后续 automatic proposal + shape 必须使用 producer 原始端点；人工修正端点只能单列 semiautomatic/oracle，不能冒充 automatic。

硬限制：
- 保护 dirty/untracked 工作树，不 reset、checkout、clean、stash，不覆盖既有输出。
- 所有 Python/pytest 使用 WSL eldercare-ai，并确认 editable install 指向当前仓库。
- 不重训、不调参、不改冻结模型、0.5 threshold、类别或人工 truth/XML。
- 不实现 acceptance converter、alert、risk、AlgorithmEvent、EP2B/EP2C/EP3，不访问 sealed 数据，不扩展哈希/PORTABLE 审计，不修改其他模块。
- 不 commit/push。

最终报告实际文件、测试命令和结果、synthetic 覆盖、matching/metric 分母、状态可见性、证据边界，以及 S1B 仍缺的真实人工 boundary、参数冻结和独立 session/setup。
```
