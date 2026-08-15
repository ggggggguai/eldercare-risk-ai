# M0-CAM-MVP-1 执行任务书

版本：1.2

更新时间：2026-08-14

状态：主功能与 F1 exact-schema 修复完成；`m0cam_mvp1_acceptance_status=passed`

> **2026-08-14 F1 关闭：** 后续复审发现的同版本字段缺失与非空 risk/event 注入缺口已由 [M0-CAM-MVP-1-F1](M0-CAM-MVP-1-F1执行任务书.md)关闭。现行 consumer 会核对完整 v1 exact key set、嵌套 probability/summary、synthetic-only `data_access`、产品 evidence/manifest 与 final 顶层集合，并重算 episode canonical bytes；证据见 [MVP-1 复审与 F1 关闭记录](../../../../reports/mental_health/wandering_camera_mvp1_v1/MVP1_REAUDIT_20260814.md)。本文第 12 节原执行提示词已经退役，不得复用。

建议工作 AI 推理等级：**极高（xhigh）**。本任务需要较强的源码盘点、契约设计和测试能力，但必须保持薄集成，避免重新滑回打包治理或扩大科学范围。

## 1. 事务定位

`M0-CAM-MVP-1` 的目标是把已经存在的徘徊 camera 研发链包装成第一段产品可消费的离线垂直切片：

```text
synthetic tracking.jsonl + wandering-media-v1 sidecar
  → 现有 Camera adapter / QC / preprocessing
  → 现有 fixed-primary forward
  → 现有 window predictions / episode candidates
  → session-level WanderingEvidence
  → 一条可重复的离线产品 CLI
```

它不是新的模型任务，不重做 PORTABLE，不启动真实 camera development，也不把单次 session 直接变成心理健康风险结论。

实时状态只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；已经实现的能力和限制看[模块 README](../README.md)。PORTABLE 后续复审见 [F1 复审记录](../../../../reports/mental_health/wandering_camera_portable_v1/F1_REAUDIT_20260814.md)。

## 2. 接任时现场与状态

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
m0cam_mvp1_f1_complete=true
m0cam_mvp1_started=true
session_evidence_contract_ready=true
mvp1_evidence_scope=synthetic_contract_only

m0cam_portable_f1_implementation_attempt_complete=true
m0cam_portable_f1_acceptance_status=rework_deferred
portable_software_audit_status=rework_required
portable_review_blocker=post_f1_findings_open
m0cam_portable_status=blocked
portable_runtime_blocker_code=asset_source_unavailable
source_integration_pending=true
portable_blocks_mvp1=false
portable_blocks_m0cam_d=true

C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

以上 Git 状态只是制定任务时快照。执行 AI 必须重新现场核对，不得覆盖、恢复、stash、reset 或清理现有 dirty worktree；未经用户当前明确授权，不 stage、commit、push。

PORTABLE 首次实现和 F1 修复确实形成过 `100` 项聚焦、`79 + 4 subtests`、`297` 项 camera、隔离 overlay 与 candidate safe-load 自动化证据；后续独立复审仍发现 source anchor、runtime import closure、restore 假 `ready` 和 rollback ownership 等未关闭问题。因此其接受状态是 `rework_deferred`，不是 passed。它不阻塞本任务的 synthetic 产品集成，但仍阻塞 fresh-clone 正式部署和 M0-CAM-D。

## 3. 当前代码已经具备什么

以下能力必须复用，不能复制第二套：

| 层 | 当前实现 | MVP-1 行为 |
|---|---|---|
| tracking/sidecar adapter | `camera_adapter.py` | 直接复用 |
| Camera QC 与窗口门控 | `camera_qc.py` | 直接复用 |
| 14 维特征与窗口准备 | `camera_inference.py` | 直接复用 |
| 固定候选加载与 forward | `camera_primary_inference.py` | 只调用现有 builder |
| `ready/unavailable/inference_error` | primary prediction schema | 原样保留，不把失败写成负类 |
| episode candidate | `camera_episode.py` | 原样透传 `development_unfrozen` |
| primary bundle | `build_primary_camera_inference_bundle()` | 作为产品切片的受信子产物 |
| generic 日级聚合/基线/风险事件 | `mental_health/daily_aggregation.py`、`baseline.py`、`pipeline.py` | 本任务不接线 |

现有 primary bundle 已包含 normalized tracking、tracklet、window、prediction、episode、QC、model binding、execution 和 manifest。其 manifest 明确写着：

```text
algorithm_event_emitted=false
risk_or_alert_decision_emitted=false
```

真正缺失的是一个面向普通调用者的、版本化的 session 结果，而不是另一套 detector、QC、模型或 episode 算法。

## 4. 为什么 MVP-1 不直接输出 AlgorithmEvent

当前仍缺少四项不能由执行 AI猜定的条件：

1. 稳定业务 `person_id`；`track_id` 和 `parent_tracklet_id` 都不是人员身份；
2. 可信绝对时间、timezone、presence 和跨午夜规则；
3. episode 到日级徘徊次数、时长、夜间比例和质量覆盖的聚合协议；
4. 日级偏离到风险等级、升级/降级和关怀动作的冻结策略。

所以 MVP-1 只能形成 session-level 工程证据。不得从未校准概率发明风险分数，不得用 track ID 生成 person ID，不得把 pacing/lapping/random 直接写成心理健康告警。

## 5. 目标输出

建议新增薄适配层和单命令入口：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_product.py
scripts/wandering/run_wandering_camera_product.py
tests/test_wandering_camera_product.py
```

CLI 从 synthetic tracking+sidecar 开始，内部只调用一次现有 fixed-primary builder。最终 fresh output 建议为：

```text
<output>/
  primary/                    # 完整保留现有 primary bundle
  wandering_evidence.json    # wandering-session-evidence-v1
  manifest.json              # wandering-camera-product-manifest-v1
```

`wandering_evidence.json` 至少包含：

- `schema_version=wandering-session-evidence-v1`；
- `product_stage=session_evidence_prototype`；
- `session_status=ready|unavailable|inference_error`；
- `degraded`；
- `evidence_scope`、`validation_scope` 原样传播；
- source group/video、device、setup、stream epoch 的明确 scope；
- `person_id=null`、`person_binding_verified=false`；
- observation、tracklet、window 三态计数和 ready ratio；
- QC reason/quality flag 汇总；
- episode candidate 数量、按 direct/pacing/lapping/random 的计数与持续时间总和；
- duration 字段必须明确命名为 `*_duration_sum_seconds`，不能冒充无重叠 presence time；
- fixed candidate ID、manifest/model identity；
- `probability_calibrated=false`；
- episode 的 `policy_status=development_unfrozen` 和显式 merge gap 只作透传；
- `risk_level=null`、`risk_score=null`、`recommended_action=null`；
- `alert_decision=null`、`medical_diagnosis=null`；
- `algorithm_event=null`、`algorithm_event_status=not_ready_session_only`；
- 缺失条件清单：stable person binding、absolute time/presence、daily evidence/baseline、frozen risk policy。

session 状态固定为：

```text
至少一个 ready window                    → ready
ready 与 unavailable/error 混合          → ready + degraded=true
没有 ready，且至少一个 inference_error   → inference_error
其余（只有 unavailable 或零有效窗口）    → unavailable
```

原始三态计数和 reason 必须完整保留。`unavailable`/`inference_error` 绝不能计为 direct、ordinary negative 或 wandering。

产品 manifest 至少绑定 `wandering_evidence.json` 和 `primary/manifest.json` 的相对路径、size 与 SHA-256；primary manifest 继续负责其内部全部 artifact。产品层必须在汇总前验证 primary manifest、必需 artifact、size/SHA、schema/scope 和计数一致性。

## 6. 工程边界

MVP-1 必须：

- 保持 seed `20260731`、best epoch `5` 和固定 candidate 不变；
- 保持 binary `sigmoid >= 0.5`、类别顺序和现有 episode merge 输入语义不变；
- 只新增薄产品适配层；默认不修改 primary/QC/episode 源码；
- 使用 canonical JSON/JSONL、fresh output 和 sibling staging；
- 对本进程内可捕获失败清理本次 staging，不覆盖已有 final；
- 明确不声称 PORTABLE 等级的跨进程竞争、断电或 crash atomicity；
- 允许固定本机 candidate 在现有 synthetic fixture 上 forward，但不得搜索、下载或替换 detector/权重；
- 把所有测试和 demo 标为 `synthetic_contract_only` 或 `synthetic_demo_only`。

MVP-1 不得：

- 修改 `camera_portability.py`、PORTABLE CLI/test 或继续打包治理；
- 读取、搜索、hash 或运行真人 camera、receipt、collection、annotation、WP raw、SmartCare official/raw 或 sealed 数据；
- 运行 detector/tracker 或 M0-CAM-D；
- 修改 candidate、模型、阈值、标签、split、calibration、uncertain、matching 或 episode policy；
- 修改共享 tracker、其他模块、`common/schemas.py`、`mental_health/pipeline.py`、`baseline.py`、`environment.yml` 或共享依赖；
- 输出 accuracy、F1、FAR、camera performance、老人域或临床结论；
- 生成非空 `AlgorithmEvent` 或风险/告警决策；
- 为了“一条命令”复制现有 adapter、QC、preprocessing、forward 或 episode 实现。

## 7. 先红后绿的测试顺序

编码前先完成复用矩阵，并回答是否可以完全不改现有 primary/QC/episode 源码。随后先写失败回归：

1. 全 ready → `ready/degraded=false`；
2. ready + unavailable/error → `ready/degraded=true`；
3. 无 ready 且有 inference error → `inference_error`；
4. 全 unavailable 或零窗口 → `unavailable`；
5. unavailable/error 不进入 direct 或 wandering 计数；
6. source/setup/epoch/scope 不一致时 fail closed，不跨 scope 聚合；
7. `track_id` 不成为 `person_id`；
8. risk、alert、medical diagnosis 和 AlgorithmEvent 始终为空；
9. primary manifest/artifact 缺失、size/SHA 篡改、schema 或计数漂移时无 final；
10. evidence 对一个固定 primary bundle 使用 canonical encoding，重复汇总得到相同 bytes；完整 one-command run 可保留真实执行延迟，不要求不同运行的整个 primary bundle 逐字相同；
11. 已存在 final 拒绝覆盖，失败路径不留下 final 或本次 staging；
12. product builder 只调用一次现有 primary builder，不调用 RF/TCN comparison-only、detector 或 tracker；
13. CLI `--help`、参数错误和 synthetic E2E；
14. CLI 不提供 candidate 选择、threshold、calibration、risk、download、video 或 tracker bypass 参数。

不得先实现后补只顺从新实现的测试；不得降低断言或扩大 schema 宽容度制造通过。

## 8. 允许文件面

生产代码默认只允许新增：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_product.py
scripts/wandering/run_wandering_camera_product.py
tests/test_wandering_camera_product.py
```

任务和实现证据允许：

```text
docs/modules/mental_health/plans/M0-CAM-MVP-1执行任务书.md
docs/tasks/README.md
docs/modules/mental_health/README.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/README.md
README.md
reports/mental_health/wandering_camera_mvp1_v1/README.md
reports/mental_health/wandering_camera_mvp1_v1/VERIFICATION.md
```

默认禁止修改：

```text
camera_primary_inference.py
camera_episode.py
camera_qc.py
camera_inference.py
camera_portability.py
camera_collection.py
model.py
release.py
common/schemas.py
mental_health/pipeline.py
mental_health/baseline.py
configs/modules/wandering_camera_primary_v1.yaml
environment.yml
```

如果失败回归证明必须修改 allowlist 外文件，执行 AI 必须先停止并说明精确原因，不能自行扩大范围。

## 9. 验证门

全部 Python 命令只使用 `eldercare-ai`：

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

再执行：

- 一个 Git 外 fresh 临时目录中的 synthetic E2E；
- `git diff --check`；
- Markdown UTF-8 与本地链接检查；
- editable 安装指向当前 checkout；
- 最终 `git status --short`、stage 和精确 diff allowlist 复核。

完整仓库 pytest 不是本任务硬门。若没有修改既有 camera 主链，按风险运行上述相邻回归；若实际修改扩大，必须相应扩大回归。

## 10. 原定完成状态与后续复审

任务执行中：

```text
status=wandering_m0cam_mvp1_in_progress
m0cam_mvp1_status=in_progress
mvp1_evidence_scope=synthetic_contract_only
algorithm_event_emitted=false
m0cam_d_started=false
```

首次实现按当时完成门写出以下状态；后续复审曾重开 exact-schema 接受门，F1 已以 descriptor-aware 红测和 fresh synthetic E2E 关闭该门，因此顶部 `passed` 状态现为实时依据：

```text
status=wandering_m0cam_mvp1_complete
m0cam_mvp1_complete=true
session_evidence_contract_ready=true
product_stage=session_evidence_prototype
mvp1_evidence_scope=synthetic_contract_only
algorithm_event_emitted=false
```

即使完成也不得写 `product_ready`、`camera_performance_validated`、`real_person_validated`、`clinical_validated`、`AlgorithmEvent integrated` 或 `PORTABLE passed`。

## 11. 后续任务只登记、不展开

1. `M0-CAM-MVP-1-F1`——已完成；关闭 exact-schema 与决策字段边界。
2. `M0-CAM-MVP-2S：WanderingDailySummary`——下一产品功能，仅登记未启动；用显式 synthetic binding 验证日级次数、时长、夜间比例、类型和覆盖率的软件聚合；不叫真实 evidence。
3. `M0-CAM-MVP-2R`——在真实 stable person/session/time/presence binding 到位后执行真实日级接受。
4. `M0-CAM-MVP-3/4`——分别处理个人基线与负责人冻结的风险/动作映射；此前 `AlgorithmEvent` 保持空。
5. `M0-CAM-MVP-5`——PORTABLE formal、C0/C1 到位后运行 M0-CAM-D 工程联调。

## 12. 原执行提示词（已执行且退役）

下列提示词只保存 MVP-1 首次实现上下文，已经退役。[MVP-1-F1 任务书](M0-CAM-MVP-1-F1执行任务书.md)也已执行完成；后续工作不得复用任一旧提示词或重复实现 MVP-1。

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-MVP-1 产品代码任务。

建议 reasoning=极高（xhigh）。不要只用 high；本任务也不需要以 max 为理由扩大治理或科学范围。

你的唯一目标是：复用现有 tracking/sidecar → Camera QC → fixed-primary → window prediction → episode candidate 主链，新增一个薄的 session-level WanderingEvidence 产品适配层和单命令 synthetic 演示 CLI。不要继续 M0-CAM-PORTABLE 治理，不要直接生成心理风险或 AlgorithmEvent。

第一步完整阅读并现场核对：
1. AGENTS.md
2. docs/tasks/README.md（唯一实时任务源）
3. docs/modules/mental_health/plans/M0-CAM-MVP-1执行任务书.md
4. docs/modules/mental_health/plans/徘徊识别技术文档2.md
5. docs/modules/mental_health/README.md
6. reports/mental_health/wandering_m0cam_engineering_v1/README.md
7. reports/mental_health/wandering_camera_portable_v1/F1_REAUDIT_20260814.md
8. camera_primary_inference.py、camera_episode.py、camera_qc.py、camera_inference.py
9. mental_health/daily_aggregation.py、baseline.py、pipeline.py
10. 相关 CLI、测试和当前 Git 状态。

先执行只读检查：
- git status --short
- git diff --cached --name-only
- git rev-parse --abbrev-ref HEAD
- git rev-parse HEAD
- conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
- 用 eldercare-ai 确认导入模块位于当前 checkout。

如果 Windows shell 无 conda，使用已验证的 WSL Ubuntu-22.04 + eldercare-ai。禁止裸 python、裸 pytest 或 base 环境。保护 dirty worktree：不 stash/reset/checkout/clean，不覆盖用户修改，未经明确授权不 stage/commit/push。

编码前先盘点现有 primary bundle 的 artifact、manifest、QC/三态和 episode 字段，形成“直接复用/真正缺失”矩阵。默认完全不修改现有 primary/QC/episode 源码；新增：
- src/elderly_monitoring/modules/mental_health/wandering/camera_product.py
- scripts/wandering/run_wandering_camera_product.py
- tests/test_wandering_camera_product.py

目标 CLI 从 synthetic tracking+sidecar 开始，只调用一次现有 fixed-primary builder，在 fresh output 中保留完整 primary bundle，并新增 wandering-session-evidence-v1 和产品 manifest。session evidence 必须保留 scope、evidence/validation scope、窗口 ready/unavailable/inference_error 计数、QC reasons、episode 类型/数量/持续时间总和、固定模型身份，并固定 person_id=null、person_binding_verified=false、probability_calibrated=false、policy_status=development_unfrozen、risk/alert/medical_diagnosis/AlgorithmEvent 全部为空。

session_status 固定为：有 ready 即 ready；ready 混合失败时 degraded=true；无 ready 且有 inference_error 时 inference_error；其余 unavailable。失败窗口不得计为 direct 或 negative，track_id 不得当 person_id，episode duration sum 不得冒充 presence time。

先写失败回归，再做最小实现。覆盖三态、混合 degraded、scope 隔离、manifest/hash/schema/count 漂移、fresh output、失败清理、canonical evidence、CLI help/synthetic E2E，以及风险和 AlgorithmEvent 始终为空。不要复制 adapter/QC/preprocessing/forward/episode，不调用 RF/TCN comparison-only、detector 或 tracker。

固定候选保持 seed=20260731、best epoch=5、candidate_id=topowander-m0s-seed20260731-epoch0005。不得重训、换候选或修改模型、阈值、标签、split、class order、calibration、uncertain、matching、episode policy、shared tracker、common schema、mental-health pipeline/baseline、environment.yml 或共享依赖。

M0-CAM-PORTABLE F1 当前是 rework_deferred，formal 仍 blocked。它不阻塞 MVP-1，但仍阻塞 fresh-clone 部署和 M0-CAM-D。不要修改 portability 源码/CLI/test，不生成 detector、canonical contract、READY.json，不把 PORTABLE 写成 passed。

本任务不读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、WP raw、SmartCare official/raw 或 sealed 数据，不运行真实 tracker或 M0-CAM-D。synthetic 测试只证明软件契约，不是 camera、老人域、产品性能或临床证据。

只使用任务书允许文件。若必须改变 scientific contract、读取受限数据、修改 allowlist 外文件、删除/覆盖证据或 stage/commit/push，立即暂停并报告。

最终分开报告：计划目标、实际代码、自动化/synthetic 证据、未获得的真实证据、Git 现场，以及 MVP-2/3/4 尚未开始的边界。不得用代码或测试冒充真实产品效果。
```
