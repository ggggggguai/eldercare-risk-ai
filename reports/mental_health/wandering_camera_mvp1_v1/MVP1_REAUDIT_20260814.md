# M0-CAM-MVP-1 后续复审与 F1 关闭记录

日期：2026-08-14

状态：`passed`；复审发现的 exact-schema 产品契约缺口已由 M0-CAM-MVP-1-F1 关闭。

## 1. 当前结论

```text
m0cam_mvp1_implementation_complete=true
m0cam_mvp1_acceptance_status=passed
m0cam_mvp1_review_blocker=none
session_evidence_contract_ready=true
m0cam_mvp1_f1_complete=true
current_code_task=none
mvp1_evidence_scope=synthetic_contract_only
algorithm_event_emitted=false
C0=false
C1=false
m0cam_d_started=false
```

MVP-1 的薄产品适配层、单命令 CLI、三态/degraded 汇总、QC/episode/model identity、fresh output 和风险/事件空值输出均已实现。F1 先在旧 validator 上得到 `42 failed, 17 passed`，最小修复后产品聚焦 `59 passed`、camera 相邻 `132 passed`、日级隔离 `60 passed, 16 subtests passed`，CLI help exit 0，Git 外 fresh synthetic E2E `1 passed`。这些仍只是自动化与 synthetic contract 证据。

后续源码红队发现并在 Git 外 `/tmp` 使用 `eldercare-ai` 复现一个 P1。它不推翻主功能，但使报告中的“严格 schema 校验”和当前 complete/ready 接受结论过强。

## 2. P1：同 schema_version 的字段漂移被接受

`camera_product.py` 的 `_require_schema()` 与 `_require_rows_schema()` 只核对 `schema_version`，没有核对每类 v1 artifact 的完整字段集合。产品层虽会复核文件 hash、计数、scope 和若干语义，但攻击者若同步更新 primary manifest descriptor，仍可：

- 删除 prediction 中契约必需但产品汇总未读取的字段；
- 为 primary artifact 增加未知字段；
- 注入非空 `risk_level` 或 `algorithm_event`，同时产品 evidence 仍输出空值、产品 manifest 仍声明没有发出事件。

现有 schema 回归只修改 `schema_version`，没有覆盖同版本字段增删或决策字段注入。

最小复现结果：

```text
MISSING_REQUIRED_FIELD_SUMMARIZE=SUCCESS status=ready
NONNULL_RISK_EVENT_BUILD=SUCCESS
retained_risk=high
retained_event=mental_health
evidence_risk=None
evidence_event=None
manifest_event_emitted=False
calls=1
TMP_REMOVED=True
```

复现只使用 synthetic primary fixture；没有读取真人媒体、receipt、collection、annotation、WP raw、SmartCare official/raw 或 sealed camera，没有运行 detector/tracker，也没有修改仓库文件。

## 3. 未发现的阻断

本轮没有发现 P0。下列行为与交接一致：

- primary builder 只调用一次；
- `ready/unavailable/inference_error` 与 mixed degraded 语义正确；
- failure window 不进入 direct/wandering/negative；
- person binding、risk、alert、diagnosis 和 `AlgorithmEvent` 输出保持空；
- scope/hash/count、episode contributor 和固定模型身份校验存在；
- final 已存在时拒绝，普通可捕获异常清理本次 staging；
- fixed primary/QC/episode/inference、PORTABLE、pipeline、baseline 和 `environment.yml` 没有因 MVP-1 改动。

“先红后绿”的时间顺序只能由原执行报告佐证，最终工作树不能独立重放历史顺序。本轮也不能从代码树证明整个历史期间从未访问真人数据；只能确认本轮复验没有访问。

## 4. F1 关闭与接受决定

F1 只修改 `camera_product.py` 及其产品测试：从当前合法 fixed v1 producer 与 synthetic builder 输出枚举 exact key set，对 primary 十类 artifact 与 detector/tracker、probability/summary、active-source preflight、runtime/latency/environment 建立闭锁；固定 `data_access` 为 synthetic-only，递归拒绝 primary 内非空风险/诊断/动作/`AlgorithmEvent` 字段，并把 episode aggregator 重算的 canonical bytes 与保存 artifact 比较。产品 evidence/manifest 和 final 顶层集合也执行 exact 校验。没有修改 primary/QC/episode/adapter producer、fixed candidate 或科学策略。

因此 MVP-1 当前可以表述为“实现完成、synthetic 自动化通过、exact-schema 接受门通过、session evidence contract ready”。这里的 ready 仅指 MVP-1 synthetic 软件契约，不是 `product_ready`、camera performance、真实人/老人域或临床证据。

下一产品功能登记为 `M0-CAM-MVP-2S`：使用显式 synthetic binding 构建 binding-ready 的 `WanderingDailySummary`；本次没有启动。真实 person/time/presence 接受、个人基线和风险策略仍分别留给 MVP-2R、MVP-3、MVP-4。PORTABLE 仍为 `rework_deferred`，`C0=false/C1=false`、`authorized_camera_data_consumed=false`、`m0cam_d_started=false` 均未改变。
