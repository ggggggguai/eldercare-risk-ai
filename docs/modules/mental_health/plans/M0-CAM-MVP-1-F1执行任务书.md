# M0-CAM-MVP-1-F1 执行任务书

版本：1.1

更新时间：2026-08-14

状态：`complete`；MVP-1 exact-schema 接受门已通过，证据仍仅为 synthetic contract

建议工作 AI 推理等级：**极高（xhigh）**。本任务需要精确审计多层 JSON/JSONL 契约和失败路径，但范围只有一个产品 validator；不需要用最高推理扩大到 PORTABLE、身份系统、日级基线或风险策略。

## 1. 事务定位

`M0-CAM-MVP-1-F1` 是 MVP-1 的一次有界产品契约修复：

> 将“schema_version + hash + 部分语义校验”收口为完整 v1 exact-schema 与决策边界校验，确保 primary 内任何缺字段、未知字段或非空风险/事件字段都不能被包装为合格 `WanderingEvidence`。

它不重做 MVP-1，不继续 M0-CAM-PORTABLE，不开始日级聚合、个人基线、风险评分或 `AlgorithmEvent`。

实时状态只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；MVP-1 原实现看 [MVP-1 任务书](M0-CAM-MVP-1执行任务书.md)；复现证据看 [MVP-1 后续复审](../../../../reports/mental_health/wandering_camera_mvp1_v1/MVP1_REAUDIT_20260814.md)。

## 2. 接任现场与状态

制定本文时现场为：

```text
branch=feat/wandering-data-pipeline
HEAD=bfc3329ad14d145e6b4c3fb836a0a525f30e970b
stage=empty

status=wandering_m0cam_mvp1_complete
current_code_task=none
m0cam_mvp1_implementation_complete=true
m0cam_mvp1_acceptance_status=passed
m0cam_mvp1_review_blocker=none
m0cam_mvp1_f1_status=complete
m0cam_mvp1_f1_complete=true
session_evidence_contract_ready=true
mvp1_evidence_scope=synthetic_contract_only
algorithm_event_emitted=false

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

Git 状态只是制定时快照。执行 AI 必须重新现场核对，保留全部 dirty/untracked 文件；禁止 stash、reset、checkout、clean、恢复或覆盖用户修改，未经当前用户明确授权不得 stage、commit、push。

## 3. 已复现缺陷

当前 `camera_product.py` 对 primary 各 artifact 会验证 canonical encoding、descriptor、schema version、scope、count 和部分语义，但 `_require_schema()` / `_require_rows_schema()` 不核对 exact key set。

已经用 synthetic fixture 复现：

1. 删除 prediction 的必需 `model_purpose`，更新 `predictions.jsonl` descriptor 后，`summarize_primary_camera_bundle()` 仍返回 `ready`；
2. 向 prediction 注入非空 `risk_level=high`、`algorithm_event=mental_health`，更新 descriptor 后，完整 product build 仍成功；final primary 保留非空字段，而 evidence 为 null、manifest 写 `algorithm_event_emitted=false`。

因此 `17/90/60` 通过数不等于 exact-schema 门已通过。它们继续证明已覆盖行为，没有证明未覆盖的字段闭包。

## 4. 唯一目标

在不改变现有 v1 producer bytes 和科学语义的前提下，为 MVP-1 消费的完整 primary bundle 建立 exact-schema closure：

- primary run manifest；
- media sidecar，包括 detector/tracker 嵌套对象；
- normalized tracking row；
- bbox tracklet row；
- camera window row；
- primary prediction row及 binary/subtype/four-class probability object；
- episode candidate row及 binary/four-class probability summary；
- QC summary；
- model bindings及 active-source preflight；
- execution及 runtime/latency/environment 嵌套对象；
- `wandering-session-evidence-v1`；
- `wandering-camera-product-manifest-v1`；
- final 顶层集合只能是 `primary/`、`wandering_evidence.json`、`manifest.json`。

所有 exact field set 必须从当前固定 v1 producer 和现有真实 synthetic builder 输出现场枚举，不凭报告手抄。消费者可以声明本地常量，但不得修改 producer schema 来迁就 validator。

## 5. 必须保持的边界

产品输入中：

- `risk_level`、`risk_score`、`recommended_action`、`medical_diagnosis`、`algorithm_event` 等决策字段不得出现在 primary artifact；
- episode 的 `alert_decision` 必须存在且只能为 `null`；
- run manifest 的 `algorithm_event_emitted` 与 `risk_or_alert_decision_emitted` 必须精确为 `false`；
- `data_access` 必须与固定 synthetic-only primary config 精确一致；
- probability/summary 对象必须 exact-key、class order 固定、数值有限且归一化、predicted label 与 argmax/二分类阈值一致；
- 建议复用 `aggregate_episode_candidates()` 对 predictions 重算 episode，并与保存的 canonical episode bytes 比较，避免复制 episode 语义；若现场证明不可行，必须说明并采用等价的窄校验。

F1 不得：

- 修改 `camera_primary_inference.py`、`camera_episode.py`、`camera_qc.py`、`camera_inference.py` 或 fixed candidate；
- 修改模型、阈值、标签、split、class order、calibration、uncertain、matching 或 episode policy；
- 修改 PORTABLE、camera collection、shared tracker、通用 pipeline/baseline/config、common schema、service 或 `environment.yml`；
- 读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；
- 运行 detector/tracker/M0-CAM-D；
- 把 synthetic 测试写成 camera、产品性能、老人域或临床证据。

## 6. 先红后绿

编码前先补能在当前实现上失败的参数化回归：

1. 每类 primary artifact 删除一个必需字段，descriptor 同步更新，仍必须 fail closed；
2. 每类 artifact 增加一个未知字段，descriptor 同步更新，仍必须 fail closed；
3. prediction、episode、execution 或 manifest 注入非空 risk/event/diagnosis/action，必须无 final；
4. probability 和 probability summary 嵌套对象增删字段、class order/label/probability 漂移，必须拒绝；
5. `data_access` 缺字段、增字段或从 synthetic-only 漂移，必须拒绝；
6. product evidence/manifest 字段增删和 final staging 增加 rogue 顶层文件，必须拒绝；
7. 所有篡改均重算 canonical bytes、size 和 SHA，证明不是只被旧 descriptor 挡住；
8. 失败不留下 final 或本次 staging，不删除无关目录；
9. 当前合法 fixed-primary synthetic E2E 继续成功，产品 evidence bytes/schema 不因修复发生无关变化。

测试必须验证具体 blocker，不只笼统断言抛出异常。不得通过降低现有断言、接受 unknown 字段或修改 fixture 规避红灯。

## 7. 允许文件面

默认生产与测试只允许：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_product.py
tests/test_wandering_camera_product.py
```

CLI 行为预计无需修改；只有失败回归证明错误映射不符合既有 CLI contract 时，才允许最小修改：

```text
scripts/wandering/run_wandering_camera_product.py
```

文档与证据允许：

```text
docs/modules/mental_health/plans/M0-CAM-MVP-1-F1执行任务书.md
docs/modules/mental_health/plans/M0-CAM-MVP-1执行任务书.md
docs/tasks/README.md
docs/modules/mental_health/README.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/README.md
README.md
reports/mental_health/wandering_camera_mvp1_v1/README.md
reports/mental_health/wandering_camera_mvp1_v1/VERIFICATION.md
reports/mental_health/wandering_camera_mvp1_v1/MVP1_REAUDIT_20260814.md
```

需要 allowlist 外文件时立即暂停并报告，不得自行扩大。

## 8. 验证门

所有命令只使用 `eldercare-ai`：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_product.py -q

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_product.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_inference.py -q

conda run -n eldercare-ai python -m pytest \
  tests/test_mental_health_daily_aggregation.py \
  tests/test_mental_health_baseline.py \
  tests/test_mental_health_pipeline.py \
  tests/test_mental_health_cli.py -q

conda run -n eldercare-ai python \
  scripts/wandering/run_wandering_camera_product.py --help
```

随后在 Git 外 fresh temporary root 重跑 fixed-candidate synthetic E2E，并检查：

- product evidence 关键摘要与原合法行为一致；
- 任何 exact-schema 攻击都没有 final/staging residue；
- strict UTF-8 和本任务新增/修改链接；
- `git diff --check`；
- editable 安装指向当前 checkout；
- stage 为空，实际 diff 严格位于 allowlist。

不预设修复后测试计数；以实际收集数和结果为准。完整仓库 pytest 不是硬门。

## 9. 完成状态

只有所有红测先失败、最小实现后通过，合法 E2E 与相邻回归无退化，报告口径修正后才可写：

```text
status=wandering_m0cam_mvp1_complete
m0cam_mvp1_implementation_complete=true
m0cam_mvp1_acceptance_status=passed
m0cam_mvp1_f1_complete=true
m0cam_mvp1_review_blocker=none
session_evidence_contract_ready=true
mvp1_evidence_scope=synthetic_contract_only
algorithm_event_emitted=false
m0cam_d_started=false
```

即使 F1 通过，也不得写 `product_ready`、`camera_performance_validated`、`real_person_validated`、`clinical_validated`、`AlgorithmEvent integrated` 或 `PORTABLE passed`。

F1 后下一产品功能只登记为：

- `M0-CAM-MVP-2S`：使用一个或多个已验收 MVP-1 bundle 与显式 synthetic binding manifest，输出 binding-ready 的 `WanderingDailySummary`；不生成基线、风险或事件；
- `M0-CAM-MVP-2R`：在真实 stable person/time/presence binding 到位后做真实接受；
- `MVP-3/4`：分别处理个人基线和负责人冻结的风险/动作映射。

本任务不得顺手实现这些后续功能。

### 2026-08-14 完成记录

- 旧实现先出现 `42 failed, 17 passed`：所有攻击都重算 canonical bytes、size 与 SHA，证明不是 descriptor 旧值造成的拒绝；
- 最小修复只修改产品 consumer 与其测试，没有修改 primary/QC/episode/adapter producer；
- exact closure 覆盖 primary 十类 artifact、detector/tracker、probability/summary、active-source preflight、runtime/latency/environment、产品 evidence/manifest 与 final 三项顶层集合；episode 由 `aggregate_episode_candidates()` 重算并比较 canonical bytes；
- 产品聚焦 `59 passed`，camera 相邻 `132 passed`，日级隔离 `60 passed, 16 subtests passed`，CLI help exit 0，Git 外 fresh fixed-candidate synthetic E2E `1 passed`；
- 所有非法输入均 fail closed，未留下 final 或本次 staging；没有读取真人或授权媒体，也没有启动 PORTABLE、MVP-2S 或 M0-CAM-D。

## 10. 可直接交给工作 AI 的目标提示词

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-MVP-1-F1 产品契约修复任务。

reasoning=极高（xhigh）。不要只用 high；也不要以 max 为理由扩大范围。

唯一目标：关闭 MVP-1 primary bundle 同 schema_version 字段漂移缺口。为 primary 全部 v1 artifact、嵌套 probability/summary、产品 evidence/manifest 和 final 顶层集合建立 exact-schema 校验；禁止 primary 内出现非空 risk/diagnosis/action/AlgorithmEvent。不要重做 MVP-1，不要继续 PORTABLE，不要开始 MVP-2/3/4。

第一步完整阅读并现场核对：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/plans/M0-CAM-MVP-1-F1执行任务书.md
4. docs/modules/mental_health/plans/M0-CAM-MVP-1执行任务书.md
5. docs/modules/mental_health/plans/徘徊识别技术文档2.md
6. docs/modules/mental_health/README.md
7. reports/mental_health/wandering_camera_mvp1_v1/README.md
8. reports/mental_health/wandering_camera_mvp1_v1/VERIFICATION.md
9. reports/mental_health/wandering_camera_mvp1_v1/MVP1_REAUDIT_20260814.md
10. camera_product.py、camera_primary_inference.py、camera_episode.py、camera_qc.py、camera_adapter.py 和相关测试。

先执行只读检查：git status --short、git diff --cached --name-only、branch、HEAD、git diff --check，以及 eldercare-ai editable 位置。保留 dirty worktree；禁止 stash/reset/checkout/clean，未经明确授权不 stage/commit/push。

已复现：删除 prediction.model_purpose 并更新 descriptor 后旧 summarize 仍成功；注入非空 risk_level/algorithm_event 并更新 descriptor 后旧完整 build 仍成功，primary 保留非空字段，而外层 evidence 为空、manifest 声称未发出事件。先把这两条变成当前实现必失败的正式红测。

随后参数化覆盖每类 primary artifact 的 missing/extra field、嵌套 probability/summary 漂移、data_access 漂移、非空决策字段和 rogue product 顶层文件。所有攻击必须同步更新 canonical bytes/size/SHA，修复后仍 fail closed、无 final、无本次 staging。

只做使这些红测转绿的最小修改。exact key set 以当前固定 v1 producer 和合法 synthetic builder 输出为准；不得改变 producer schema、fixed candidate、阈值、标签、split、class order、calibration、uncertain、matching 或 episode policy。优先复用 aggregate_episode_candidates 重算 episode，不复制现有算法。

默认只修改 camera_product.py 与 test_wandering_camera_product.py。不得修改 primary/QC/episode/adapter producer、PORTABLE、collection、shared tracker、pipeline/baseline/config、service、environment.yml 或共享依赖。需要 allowlist 外文件立即停止。

不读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；不运行 detector/tracker/M0-CAM-D。fixed-candidate synthetic E2E 可以按既有测试运行，但仍只证明 software/synthetic contract。

全部验证只使用 eldercare-ai：先产品聚焦，再 primary/episode/QC/inference 相邻回归，再日级/pipeline 隔离回归、CLI help、Git 外 synthetic E2E、UTF-8/链接、git diff --check、editable 和最终 Git 现场。不要预设测试数。

最终分开报告：复现的旧失败、exact-schema 实现、自动化/synthetic 证据、未获得的真实证据、MVP-1 接受状态、PORTABLE/C0/C1/M0-CAM-D 不变状态和 Git 现场。只有全部门通过才写 m0cam_mvp1_acceptance_status=passed；不得顺手实现 MVP-2S。
```
