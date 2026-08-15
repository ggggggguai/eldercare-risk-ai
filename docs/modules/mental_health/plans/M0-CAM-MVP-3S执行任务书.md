# M0-CAM-MVP-3S 执行任务书

版本：1.0

更新时间：2026-08-14

状态：主体与 [MVP-3S-F1](M0-CAM-MVP-3S-F1执行任务书.md) 接受门均已完成；`m0cam_mvp3s_acceptance_status=passed`

建议工作 AI 推理等级：**极高（xhigh）**。本任务同时涉及多 bundle 回读、按人/日排序、冷启动、滚动窗口、版本重热、统计量和 exact-schema；`high` 容易漏掉历史泄漏或身份漂移，`max/最高` 暂无必要，也不得借此扩大到真人数据、风险策略、PORTABLE 或模型改动。

## 1. 事务定位

`M0-CAM-MVP-3S` 是 MVP-2S 后的第三个 synthetic 产品切片：

> 消费一个或多个已经通过 MVP-2S 验收的日级 bundle，为每个 synthetic person 构建可回读、可重放的 `WanderingBaselineProfilePreview`。

它解决“日级徘徊统计怎样进入个人历史参考结构”的软件问题，不声称得到真实个人基线，不计算当前日偏离，不输出风险、动作、诊断或 `AlgorithmEvent`。

实时状态只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；上游契约看 [MVP-2S 任务书](M0-CAM-MVP-2S执行任务书.md)；上游实际证据看 [MVP-2S 报告](../../../../reports/mental_health/wandering_camera_mvp2s_v1/README.md)和[验证记录](../../../../reports/mental_health/wandering_camera_mvp2s_v1/VERIFICATION.md)。

## 2. 接任现场与入口状态

制定本文时已独立复核：

```text
branch=feat/wandering-data-pipeline
HEAD=bfc3329ad14d145e6b4c3fb836a0a525f30e970b
stage=empty

status=wandering_m0cam_mvp3s_ready_to_start
current_code_task=M0-CAM-MVP-3S
m0cam_mvp2s_acceptance_status=passed
daily_summary_contract_ready=true
mvp2s_evidence_scope=synthetic_contract_only
m0cam_mvp3s_status=ready_to_start
m0cam_mvp3s_started=false
m0cam_mvp3s_evidence_scope=none

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

MVP-2S 的独立复验为：聚焦 `37 passed`，MVP-1/product/daily/primary/episode/QC/inference 相邻回归 `169 passed`，generic daily/baseline/pipeline/CLI 隔离回归 `60 passed, 16 subtests passed`，CLI help 和 Git 外跨午夜 multi-session synthetic E2E 通过。代码审计确认 full product/daily 回读、IANA/epoch 时间、presence/status interval union、episode 跨日语义和空决策边界均已实现。

这些只证明代码和 synthetic contract，不证明真人身份、目标摄像头性能、真实个人基线、产品风险效果或临床价值。

Git 状态只是制定时快照。执行 AI 必须重新检查并保留所有 dirty/untracked 文件；禁止 stash、reset、checkout、clean、恢复或覆盖用户改动。未经当前用户明确授权，不得 stage、commit、push。

## 3. 唯一目标

实现一个薄、独立、可回读的 synthetic baseline-profile builder：

```text
一个或多个已验收 MVP-2S daily bundle
  + configs/modules/mental_health.yaml 中现有 baseline 天数与 upper_quantile
  ↓
baseline_profiles.jsonl
manifest.json
```

输出目录必须 fresh，固定只含两项：

```text
baseline_profiles.jsonl
manifest.json
```

初版固定 schema 名如下；F1 完成后 production builder/consumer 已升级为 v2，v1 只保留为需重新审计的历史输出：

```text
wandering-baseline-preview-v1
wandering-baseline-preview-manifest-v1
wandering-baseline-preview-v2
wandering-baseline-preview-manifest-v2
```

本阶段产物名称必须是 `WanderingBaselineProfilePreview`，不得命名为真实 `PersonalBaseline`、`RiskProfile`、`DailyRisk` 或 `AlgorithmEvent`。

## 4. 明确不做

本任务不得：

1. 读取、搜索、hash 或运行真人视频、receipt、collection、annotation、detector 权重、WP raw、SmartCare official/raw 或 sealed 数据；
2. 启动 detector、tracker、M0-CAM-D、PORTABLE F1 返工或 FORMAL-CLOSURE；
3. 修改 fixed candidate、模型、阈值、标签、split、class order、概率校准、uncertain、matching 或 episode merge policy；
4. 从 synthetic person token 推断真实身份，或把 `person_binding_verified`、`eligible_for_baseline` 改为 true；
5. 把本任务的机械可用日定义写回 MVP-2S 或冒充真实 baseline eligibility；
6. 计算当前日 `baseline_deviation`、异常分数、风险级别、升级/降级、动作、诊断、告警或 `AlgorithmEvent`；
7. 使用 `baseline.abnormal_score_threshold`、scoring 权重或 action policy；
8. 修改共享 `baseline.py`、`daily_aggregation.py`、`pipeline.py`、通用事件接口、service、`configs/modules/mental_health.yaml`、`environment.yml` 或共享依赖；
9. 顺手修复 PORTABLE 的 source/restore/rollback P0，或把 synthetic profile 当作 fresh-clone 部署证据。

## 5. 输入契约

### 5.1 MVP-2S daily bundle

每个输入必须是通过完整回读的 MVP-2S final，而不只是存在 `manifest.json`：

- 顶层精确为 `daily_summary.jsonl + manifest.json`；
- JSON/JSONL canonical bytes、exact field set、稳定排序、唯一 person/date/timezone key、descriptor、count、night/config/model/policy identity全部通过；
- 每一行仍为 `synthetic_contract_only`、`person_binding_verified=false`、`eligible_for_baseline=false`；
- baseline/risk/action/diagnosis/`AlgorithmEvent` 继续为空；
- 同一 daily manifest 不得重复输入。

在 `camera_daily_summary.py` 增加最小公开只读 loader：

```python
load_validated_wandering_daily_summary(path: str | Path) -> ValidatedWanderingDailySummary
```

它必须复用现有 staging/final validator 和 row/manifest 校验，不复制或削弱 MVP-2S 契约，不改变 MVP-2S producer、输出 bytes 或 schema。

### 5.2 跨 bundle 合并约束

1. 输入顺序不能影响输出；按 manifest digest 和 row key 确定性处理；
2. 重复 `(synthetic_person_id, local_date, timezone)` 必须拒绝，不做 last-write-wins 或静默去重；
3. 同一 synthetic person 必须保持一个 IANA timezone；timezone 漂移必须拒绝；
4. 同一 person profile 内，candidate ID/hash、model state、seed/epoch、class order、binary threshold、calibration status、episode policy/merge gap、night window、duration semantics 和 effective baseline config 必须一致；
5. 不同 synthetic person 可以独立形成 profile，不能共享或混合历史；
6. 不接受 caller 传入 threshold、天数、quantile、metric list 或 readiness override。

## 6. Synthetic 基线预览语义

### 6.1 可用日与缺口

MVP-2S 的 `eligible_for_baseline=false` 必须原样保留。本任务另定义纯机械字段 `usable_for_synthetic_profile`，只在以下条件同时满足时为 true：

```text
presence_seconds > 0
any_window_covered_seconds > 0
```

不新增或猜测 coverage 百分比门槛。缺少窗口覆盖的日期排除，并记录原因与计数；自然日缺口保持缺口，不前向填充、不补零、不把 unavailable/error 当作正常日。

每个 synthetic person 只使用时间上最近的 `max_window_days` 个可用日。现有配置固定提供：

```text
initial_days=3
stable_days=7
max_window_days=14
upper_quantile=0.90
```

代码从当前 config loader 读取并在 manifest 绑定有效值与配置 SHA，不在代码重复维护第二套默认值。

### 6.2 冷启动预览状态

按 profile window 中可用日数固定输出：

```text
0-2 days  -> warming_up
3-6 days  -> initial_ready_preview
7-14 days -> stable_ready_preview
```

无论处于哪一状态，都必须保持：

```text
person_binding_verified=false
real_baseline_ready=false
eligible_for_risk=false
baseline_deviation=null
risk_level=null
risk_score=null
recommended_action=null
alert_decision=null
medical_diagnosis=null
algorithm_event=null
algorithm_event_emitted=false
```

`initial_ready_preview` 和 `stable_ready_preview` 只表示 synthetic history 数量达到软件预览门，不是对真人的 baseline acceptance。

### 6.3 固定 profile metrics

按日先派生以下固定顺序的 scalar：

```text
trajectory_coverage
presence_hours
direct_episode_count_per_presence_hour
pacing_episode_count_per_presence_hour
lapping_episode_count_per_presence_hour
random_episode_count_per_presence_hour
wandering_like_episode_count_per_presence_hour
wandering_like_candidate_duration_seconds_per_presence_hour
night_wandering_like_ratio
```

约束：

- `presence_hours = presence_seconds / 3600`；
- count rate 和 candidate duration rate 都以 presence hours 为分母；
- duration rate 继承 `candidate_interval_sum_not_presence_time`，候选重叠时可以大于 3600，不能写成占比或互斥 person-time；
- 没有 wandering-like candidate 时，相关 count/duration rate 合法为 0，但 `night_wandering_like_ratio` 继续为 null；
- `trajectory_coverage` 和 `presence_hours` 是 profile 质量/观察量，不是风险分数。

每个 metric 输出 exact stats：

```text
count
median
p90
min
max
```

无可用值时 `count=0` 且其余为 null。`p90` 使用与现有 baseline quantile 相同的 `(n-1)*p` 线性插值公式，`p=upper_quantile`；不导入 `baseline.py` 私有函数，也不顺手计算 mean/std/deviation。

### 6.4 版本与重热

每个 profile 必须包含稳定 `profile_contract_id`，至少绑定：

- schema 与 metric 顺序；
- candidate/model/seed/epoch/class/threshold/calibration identity；
- episode policy/merge gap 与 duration semantics；
- night window；
- baseline config SHA 与 effective initial/stable/max-window/upper-quantile。

`profile_contract_id` 不包含输入日期或统计值。一个 synthetic person 的输入出现上述任一身份漂移时，必须以结构化错误 `baseline_version_reset_required` 失败且无 final，不跨版本混合。未来调用方应以新版本重新开始 warming-up；本阶段不自动迁移或拼接旧 profile。

## 7. 输出契约

### 7.1 `baseline_profiles.jsonl`

每个 synthetic person 恰好一行，按 `synthetic_person_id, timezone` 稳定排序。行至少精确包含：

- schema/product stage/status/evidence scope；
- synthetic identity、timezone 与 `person_binding_verified=false`；
- `profile_contract_id` 和固定 model/policy/config identity；
- source/usable/excluded day counts、排除原因计数；
- profile window 的 first/last local date 和 window day count；
- cold-start preview status；
- 固定 metric order 与每项 exact stats；
- real baseline/risk/action/diagnosis/event 的 false/null 边界。

### 7.2 `manifest.json`

manifest 至少精确绑定：

- `baseline_profiles.jsonl` byte count + SHA-256；
- 每个 input daily manifest 的 byte count + SHA-256，按 digest 稳定排序；
- input bundle/daily row/synthetic person/profile counts；
- metric order、quantile method 与 effective baseline config；
- `mental_health.yaml` SHA、builder source SHA；
- model/policy/night/duration identity；
- `real_baseline_emitted=false`、`baseline_deviation_emitted=false`、`risk_or_alert_decision_emitted=false`、`algorithm_event_emitted=false`、`real_human_media_consumed=false`、`m0cam_d_started=false`。

输出 JSON/JSONL 必须 canonical；final 目录已存在时拒绝覆盖；使用 sibling staging，回读 exact bytes/schema/hash/count/semantics 全部通过后再提交。只清理本次 staging，不删除竞争者或既有证据。本任务不把普通 fresh-only 提交升级成 PORTABLE 并发/崩溃原子性项目，也不得在报告中声称获得该证据。

## 8. 建议实现面

默认 allowlist：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_baseline_preview.py
src/elderly_monitoring/modules/mental_health/wandering/camera_daily_summary.py
scripts/wandering/build_wandering_baseline_preview.py
tests/test_wandering_camera_baseline_preview.py
tests/test_wandering_camera_daily_summary.py
reports/mental_health/wandering_camera_mvp3s_v1/README.md
reports/mental_health/wandering_camera_mvp3s_v1/VERIFICATION.md
docs/tasks/README.md
docs/modules/mental_health/README.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/modules/mental_health/plans/M0-CAM-MVP-3S执行任务书.md
docs/README.md
README.md
```

`camera_daily_summary.py` / daily-summary test 只允许为 public validated loader 做最小改动；不得改变 MVP-2S producer/output bytes。不得修改 generic `baseline.py`、`daily_aggregation.py`、`pipeline.py` 或 config。若需要 allowlist 外 Python/config/shared 文件，停止并报告，不自行扩大范围。

建议 API/CLI：

```python
build_wandering_baseline_preview(
    daily_bundles: Sequence[str | Path],
    output_dir: str | Path,
) -> WanderingBaselinePreviewBuildResult
```

```bash
conda run -n eldercare-ai python scripts/wandering/build_wandering_baseline_preview.py \
  --daily-bundle <mvp2s-daily-1> \
  --daily-bundle <mvp2s-daily-2> \
  --output <fresh-output>
```

CLI 不提供 person、timezone、metric、天数、quantile、threshold、risk 或 config override。

## 9. TDD 顺序

先让 production/CLI 不存在的红测失败，再做最小实现：

1. public validated-daily loader 对合法 MVP-2S final 完整回读；
2. 1/2/3/7/14/15 个可用日的 warming/initial/stable 与 14 日滚动窗口；
3. 输入顺序打乱仍产出相同 canonical bytes；
4. 两个 synthetic person 隔离、日期缺口不补齐；
5. 零 wandering 日合法为 0，night ratio 保持 null；
6. count/duration per-presence-hour 公式、candidate duration 语义和 p90 线性插值；
7. no-window day 排除及原因计数，不引入隐藏 coverage threshold；
8. 重复 daily manifest、重复 person-day、timezone drift；
9. candidate/model/class/threshold/calibration/episode-policy/night/config/duration 漂移返回 `baseline_version_reset_required`；
10. daily final/manifest/row missing/extra、descriptor/hash/count、rogue top-level、非空 decision 字段；攻击修改 artifact 时同步重算 canonical bytes、size 和 SHA；
11. profile/manifest missing/extra、metric order/stats 漂移、非空 baseline deviation/risk/event；
12. fresh-only、稳定排序、staging 回读和失败清理；
13. CLI help/参数错误/Git 外多 bundle synthetic E2E；
14. MVP-2S/MVP-1/camera 相邻回归不退化；既有 generic daily/baseline/pipeline/CLI 隔离回归继续通过。

## 10. 验证命令

全部使用 `eldercare-ai`，先确认 editable：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

先窄后宽，不预设通过数：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_baseline_preview.py -q

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_baseline_preview.py \
  tests/test_wandering_camera_daily_summary.py \
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
  scripts/wandering/build_wandering_baseline_preview.py --help

git diff --check
```

另做一个 Git 外 fresh synthetic E2E，至少包含 15 个可用日、一个日期缺口和一个零 wandering 日；检查只保留最近 14 个可用日、final 精确两项、回读 bytes/hash/count/profile stats/空风险边界一致。不得读取视频或启动 tracker。

### 执行完成记录（2026-08-14）

红测先在旧代码上因缺失 validated daily loader/MVP-3S producer 而 collection 失败；交付文档红测也先因报告不存在而失败。最小实现完成后，执行时门为 MVP-3S + daily 聚焦 `84 passed`、MVP-2S/MVP-1/camera 相邻组 `216 passed`、generic daily/baseline/pipeline/CLI 隔离组 `60 passed, 16 subtests passed`、Git 外 15 日 synthetic CLI E2E `1 passed`；CLI help、UTF-8/相关链接、`git diff --check`、eldercare-ai editable、空 stage 与 Git branch/HEAD 现场均通过。完整执行收据见 [MVP-3S 报告](../../../../reports/mental_health/wandering_camera_mvp3s_v1/README.md)和[验证记录](../../../../reports/mental_health/wandering_camera_mvp3s_v1/VERIFICATION.md)。后续独立复审复现 producer source identity 自证明与 rehashed reference stats 假通过，现行接受状态由[复审记录](../../../../reports/mental_health/wandering_camera_mvp3s_v1/MVP3S_REAUDIT_20260814.md)覆盖：主体实现保留，接受门转为 `rework_required`。

```text
status=wandering_m0cam_mvp3s_complete
current_code_task=none
m0cam_mvp3s_status=complete
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
synthetic_baseline_profile_contract_ready=true
mvp3s_evidence_scope=synthetic_contract_only
person_binding_verified=false
real_baseline_ready=false
eligible_for_risk=false
algorithm_event_emitted=false

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

上段是初版执行完成时快照。F1 关闭独立复审缺口后的现行状态为：

```text
status=wandering_m0cam_mvp3s_f1_complete
current_code_task=none
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_execution_tests_status=passed
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
m0cam_mvp3s_f1_status=complete
m0cam_mvp3s_f1_acceptance_status=passed
```

## 11. 原执行完成门与后续路线

以下是首次实现时使用的完成门快照；后续独立复审已经证明这些门不足以关闭 validated readback，因此不得再据此恢复 `passed`。现行恢复条件只看 [MVP-3S-F1 任务书](M0-CAM-MVP-3S-F1执行任务书.md)：

```text
status=wandering_m0cam_mvp3s_complete
current_code_task=none
m0cam_mvp3s_status=complete
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
synthetic_baseline_profile_contract_ready=true
mvp3s_evidence_scope=synthetic_contract_only
person_binding_verified=false
real_baseline_ready=false
eligible_for_risk=false
algorithm_event_emitted=false
```

同时必须保持：

```text
m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

MVP-3S 完成后不得自动输出风险或事件：

- `MVP-2R`：用真实、稳定的 person/session/time/presence binding 完成日级 acceptance；
- `MVP-3R`：在 MVP-2R 与足量真实观察日到位后，接受真实个人基线并定义版本重热；
- `MVP-4`：负责人冻结风险级别、偏离解释、升级/降级和动作映射后，才输出风险与 `AlgorithmEvent`；
- PORTABLE rework 仍在正式部署/M0-CAM-D 前恢复，不由 MVP-3S 顺手修复。

## 12. 首次实现提示词（已执行，禁止复用）

以下提示词只保留历史追溯。当前 AI 必须使用 [MVP-3S-F1 任务书](M0-CAM-MVP-3S-F1执行任务书.md)中的新提示词，不得重新执行 MVP-3S 首次实现。

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-MVP-3S synthetic 个人基线预览任务。

reasoning=极高（xhigh）。不要只用 high；当前不需要 max/最高，也不得借此扩大到真人身份、风险策略、PORTABLE、模型或视频数据。

唯一目标：消费一个或多个已经通过 MVP-2S 完整回读的 synthetic daily bundle，按 synthetic person 构建 fresh、canonical、exact-schema 的 baseline_profiles.jsonl + manifest.json。产物名称固定为 WanderingBaselineProfilePreview，不是真实 PersonalBaseline、Risk 或 AlgorithmEvent。

第一步完整阅读并现场核对：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/plans/M0-CAM-MVP-3S执行任务书.md
4. docs/modules/mental_health/plans/M0-CAM-MVP-2S执行任务书.md
5. docs/modules/mental_health/plans/徘徊识别技术文档2.md
6. docs/modules/mental_health/README.md
7. reports/mental_health/wandering_camera_mvp2s_v1/README.md
8. reports/mental_health/wandering_camera_mvp2s_v1/VERIFICATION.md
9. camera_daily_summary.py、config.py、baseline.py（只读参考）及相关测试。

先执行只读检查：git status --short、git diff --cached --name-only、branch、HEAD、git diff --check、eldercare-ai editable 位置。保留 dirty/untracked 现场；禁止 stash/reset/checkout/clean，未经当前用户明确授权不 stage/commit/push。

先写失败回归，再做最小实现。先在 camera_daily_summary.py 增加复用现有 validator 的最小 public read-only loader，不改变 MVP-2S producer、schema 或 bytes；随后新增独立 camera_baseline_preview.py、CLI 和测试。不要修改 generic baseline.py、daily_aggregation.py、pipeline.py、mental_health.yaml、风险评分或 AlgorithmEvent 接口。

只使用完整 validated MVP-2S final。输入顺序必须不影响输出；同一 person 的重复日、timezone 或 candidate/model/class/threshold/calibration/episode/night/config/duration identity 漂移必须 fail closed。漂移使用结构化 baseline_version_reset_required，绝不跨版本混合。

机械可用日只要求 presence_seconds>0 且 any_window_covered_seconds>0，不新增 coverage 门槛，不把 eligible_for_baseline 改为 true，不填补缺失日期。按配置使用最近 max_window_days=14 个可用日；<3 warming_up、3-6 initial_ready_preview、>=7 stable_ready_preview。它们只表示 synthetic 软件预览。

固定生成 trajectory coverage、presence hours、四类及 wandering-like episode count per presence hour、wandering-like candidate duration seconds per presence hour、night wandering-like ratio；每项输出 count/median/p90/min/max。p90 沿用现有 linear quantile 公式。零 wandering 是 0，night ratio 无分母时为 null；candidate duration rate 不是占比，可以大于 3600。

所有真实基线、偏离、风险、动作、诊断、告警和 AlgorithmEvent 边界必须保持 false/null：person_binding_verified=false、real_baseline_ready=false、eligible_for_risk=false、algorithm_event_emitted=false。证据只能写 synthetic_contract_only。

默认只修改任务书 allowlist。不得读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；不得运行 detector/tracker/M0-CAM-D；不得修 PORTABLE、改 candidate/model/threshold/label/split/shared dependency。

全部验证只用 eldercare-ai：先 MVP-3S 聚焦，再 MVP-2S/MVP-1/camera 相邻回归，再 generic daily/baseline/pipeline/CLI 隔离回归、CLI help、Git 外 15 日 synthetic E2E、UTF-8/链接、git diff --check、editable 和最终 Git 现场。不要预设测试数。

最终分开报告：实现代码、自动化证据、synthetic E2E、未获得的真实证据、MVP-3S 接受状态、PORTABLE/C0/C1/M0-CAM-D 不变状态和 Git 现场。只有全部门通过才写 m0cam_mvp3s_acceptance_status=passed；不得自动开始 MVP-2R/MVP-3R/MVP-4。
```
