# M0-CAM-MVP-2S 执行任务书

版本：1.0

更新时间：2026-08-14

状态：`completed`；`m0cam_mvp2s_acceptance_status=passed`

建议工作 AI 推理等级：**极高（xhigh）**。本任务包含跨 session、自然日、时区/DST、presence、interval union 和多层 bundle 绑定，`high` 容易漏边界；`max/最高` 没有必要，且不得借此扩大到真实身份、基线、风险策略、PORTABLE 或摄像头数据。

## 1. 事务定位

`M0-CAM-MVP-2S` 是 MVP-1 之后的第二个产品垂直切片：

> 消费一个或多个已经通过 MVP-1-F1 验收的 synthetic session product bundle，以及一份显式 synthetic person/session/time/presence binding，形成按 synthetic person + 本地自然日汇总的 `WanderingDailySummary`。

它解决“session 工程证据怎样进入日级产品结构”的软件问题，不解决真实身份、真实摄像头性能或风险判断。

实时状态只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；上游产品契约看 [MVP-1 任务书](M0-CAM-MVP-1执行任务书.md)和 [MVP-1-F1 任务书](M0-CAM-MVP-1-F1执行任务书.md)；上游实际证据看 [MVP-1 报告](../../../../reports/mental_health/wandering_camera_mvp1_v1/README.md)。

## 2. 接任现场与入口状态

制定本文时已独立复核：

```text
branch=feat/wandering-data-pipeline
HEAD=bfc3329ad14d145e6b4c3fb836a0a525f30e970b
stage=empty

status=wandering_m0cam_mvp2s_ready_to_start
current_code_task=M0-CAM-MVP-2S
m0cam_mvp1_acceptance_status=passed
session_evidence_contract_ready=true
mvp1_evidence_scope=synthetic_contract_only
m0cam_mvp2s_status=ready_to_start
m0cam_mvp2s_started=false
m0cam_mvp2s_evidence_scope=none

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

MVP-1-F1 的独立复验为：产品聚焦 `59 passed`，primary/episode/QC/inference 相邻回归 `132 passed`，既有日级/baseline/pipeline/CLI 隔离回归 `60 passed, 16 subtests passed`，CLI help 与 fresh fixed-candidate synthetic E2E 通过。它们只证明代码和 synthetic contract，不证明 camera、真人、产品性能或临床效果。

Git 状态只是制定时快照。执行 AI 必须重新检查并保留所有 dirty/untracked 文件；禁止 stash、reset、checkout、clean、恢复或覆盖用户改动。未经当前用户明确授权，不得 stage、commit、push。

## 3. 唯一目标

实现一个薄、独立、可回读的 daily-summary builder：

```text
一个或多个已验收 MVP-1 product bundle
  + 一份 explicit synthetic binding manifest
  + 现有 mental_health.yaml 的 night_start/night_end
  ↓
daily_summary.jsonl
manifest.json
```

输出目录必须 fresh，固定只含两项：

```text
daily_summary.jsonl
manifest.json
```

固定 schema 名：

```text
wandering-daily-binding-v1
wandering-daily-summary-v1
wandering-daily-summary-manifest-v1
```

本阶段产物名称必须是 `WanderingDailySummary`，不得命名为 `DailyEvidence`、`DailyRisk` 或 `AlgorithmEvent`。

## 4. 明确不做

本任务不得：

1. 读取、搜索、hash 或运行真人视频、receipt、collection、annotation、detector 权重、WP raw、SmartCare official/raw 或 sealed 数据；
2. 启动 detector、tracker、M0-CAM-D、PORTABLE F1 返工或 FORMAL-CLOSURE；
3. 修改 fixed candidate、模型、阈值、标签、split、class order、概率校准、uncertain、matching 或 episode merge policy；
4. 从 `track_id`、文件名、轨迹形状或 source scope 推断 person identity；
5. 生成真实 `person_id`，或把 synthetic binding 写成 `person_binding_verified=true`；
6. 训练或接入个人基线，输出 baseline deviation、risk level、action、diagnosis、alert 或 `AlgorithmEvent`；
7. 把 `unavailable`、`inference_error` 或无覆盖区间当作 direct/normal/negative；
8. 修改共享 `daily_aggregation.py`、`baseline.py`、`pipeline.py`、通用事件接口、service、`configs/modules/mental_health.yaml`、`environment.yml` 或共享依赖。

## 5. 输入契约

### 5.1 MVP-1 product bundle

每个输入必须是通过 MVP-1-F1 完整回读的 final，而不只是存在 `manifest.json`：

- 顶层必须精确为 `primary/ + wandering_evidence.json + manifest.json`；
- product evidence/manifest exact-schema、canonical bytes、descriptor 和空风险/事件边界必须通过；
- `primary/` 必须重新走 MVP-1-F1 的完整 exact-schema、hash、scope、count、probability、episode 重算和 synthetic-only `data_access` 校验；
- `mvp1_evidence_scope` 必须保持 `synthetic_contract_only`；
- 同一 product manifest 不得重复输入。

如果下游需要读取 validated primary rows，可以在 `camera_product.py` 增加一个最小、只读、公开的 validated-product loader。它必须复用现有 `_load_and_validate_primary_bundle()` 与产品 staging/final 校验，不得复制或削弱 F1 validator，也不得改变 MVP-1 输出 bytes/schema。

### 5.2 synthetic binding manifest

binding 是测试/软件输入，不是 C3、授权或真实身份事实。建议 exact-schema 结构：

```json
{
  "schema_version": "wandering-daily-binding-v1",
  "evidence_scope": "synthetic_binding_fixture",
  "sessions": [
    {
      "binding_id": "SYN-BIND-001",
      "product_manifest_sha256": "<64 lowercase hex>",
      "session_id": "SYN-SESSION-001",
      "session_started_at": "2026-08-14T23:59:30+08:00",
      "timezone": "Asia/Shanghai",
      "source_scope": {
        "source_group_id": "...",
        "source_video_id": "...",
        "device_id": "...",
        "setup_id": "...",
        "stream_epoch": "..."
      },
      "person_tracks": [
        {
          "synthetic_person_id": "SYN-PERSON-001",
          "track_ids": [1],
          "presence_intervals_sec": [
            {"start_sec": 0.0, "end_sec_exclusive": 80.0}
          ]
        }
      ]
    }
  ]
}
```

最小硬约束：

1. manifest、session、source_scope、person_tracks、presence interval 均为 exact key set；JSON 必须 UTF-8、finite、canonical；
2. `product_manifest_sha256` 绑定实际 product manifest bytes；`source_scope` 与 validated MVP-1 bundle 精确一致；
3. `binding_id`、`session_id` 在 manifest 内唯一；同一 product 只能绑定一次；
4. `synthetic_person_id` 只允许匿名 synthetic token，禁止空值、路径、URL、邮箱、电话号码或自由文本姓名；final 固定 `person_binding_verified=false`；
5. `track_ids` 必须非空、无重复；一个 session 内每个 primary track 恰好分配一次且不能跨 person 重叠；不得自动补齐或按连续性推断；
6. `session_started_at` 必须是带 UTC offset 的 ISO-8601；`timezone` 必须是可用 IANA zone，且该 instant 映射后的本地 offset/wall time 与声明一致；不得用系统本地时区或文件 mtime 猜时间；
7. presence 使用 session-relative 半开区间，数值有限、`0 <= start < end`；可重叠输入先做确定性 union；所有被分配 track 的 window/episode 必须被 presence 完整覆盖，否则 fail closed，不静默裁切；
8. 同一 synthetic person 可以跨多个 session；同一 person-day 必须使用同一 IANA timezone。timezone 漂移直接拒绝，不自动换日归并。

## 6. 日级汇总语义

### 6.1 时间与自然日

- absolute instant = `session_started_at + relative_seconds`，按 epoch elapsed seconds 计算；
- 自然日边界使用 binding 的 IANA timezone；必须覆盖跨午夜、DST spring-forward 和 fall-back；
- duration 在午夜处分片到各日；episode candidate count 只计入 episode start 所在本地日一次；
- night window 复用当前 `configs/modules/mental_health.yaml` 的 `aggregation.night_start/night_end`，不改配置；manifest 记录有效 night window 和配置 SHA；
- 无明确 absolute time、timezone 或 presence 时拒绝构建，不生成“假 0 点”或“全天 presence”。

### 6.2 window/QC coverage

对每个 person-day 至少输出：

- `presence_seconds`：union 后 presence 与该自然日的交集；
- `any_window_covered_seconds`：所有已绑定 window interval 的 union；
- `ready_covered_seconds`、`unavailable_covered_seconds`、`inference_error_covered_seconds`：各状态分别做 union 后与 presence/自然日相交；
- `trajectory_coverage = any_window_covered_seconds / presence_seconds`；
- `ready_coverage`、`unavailable_coverage`、`inference_error_coverage` 分别以 presence 为分母；
- `status_overlap_seconds`：不同状态 union 之间的重叠，用 quality flag 明示。各状态秒数不强制互斥，不靠未经冻结的优先级改写；
- session/source/setup/track/window 数量和 QC reason/quality flag 计数。

`presence_seconds=0`、窗口越界、重复绑定或未绑定 track 必须 fail closed。`unavailable/inference_error` 只进入质量/覆盖字段，不能进入 direct 或 wandering-like episode 统计。

### 6.3 episode candidate 汇总

固定四类顺序继续为：

```text
direct, pacing, lapping, random
```

- `wandering_like` 仅等于 `pacing + lapping + random`，`direct` 单独报告；
- 输出四类 episode candidate count、四类 duration sum、wandering-like count/duration、night wandering-like duration/ratio；
- duration 保持 `candidate_interval_sum_not_presence_time`，不伪装成互斥 person-time；
- episode 跨午夜时 duration 分日，count 只放 start day；
- 只消费 `prediction_status=ready` 形成的、由 MVP-1 已重算通过的 episode candidates；
- 本阶段没有模型 uncertain 输出，`uncertain_episode_count` 与 `uncertain_ratio` 固定为 `null`，不得写 0；
- 同一 person-day 跨 bundle 聚合时，candidate/model identity、class order、binary threshold、calibration status 和 episode policy/merge gap 必须一致；漂移时 fail closed。

### 6.4 产品决策空边界

每一行 summary 固定：

```text
person_binding_verified=false
eligible_for_baseline=false
baseline_status=not_ready_synthetic_binding
baseline_deviation=null
risk_level=null
risk_score=null
recommended_action=null
alert_decision=null
medical_diagnosis=null
algorithm_event=null
algorithm_event_status=not_ready_daily_summary_only
```

manifest 固定：

```text
baseline_emitted=false
risk_or_alert_decision_emitted=false
algorithm_event_emitted=false
real_human_media_consumed=false
m0cam_d_started=false
```

任何输入或 staging 中出现非空 risk/diagnosis/action/`AlgorithmEvent` 都必须拒绝。

## 7. 输出契约

### 7.1 `daily_summary.jsonl`

每行唯一键建议为：

```text
(synthetic_person_id, local_date, timezone)
```

行按 `local_date, synthetic_person_id, timezone` 稳定排序。每行至少包含：

- schema/product stage/evidence/validation scope；
- synthetic identity 与 `person_binding_verified=false`；
- local date、timezone、night window；
- session/source/setup/track/window 数量；
- presence、window/QC coverage 与 status overlap；
- episode 四类 count/duration、wandering-like/night 字段与 duration semantics；
- fixed candidate/model/class/threshold/calibration/episode-policy identity；
- baseline/risk/action/diagnosis/event 固定空边界；
- 排序后的 quality flags。

生产代码必须集中定义 exact field set 和关键嵌套 field set；不能只验证 schema version 或被使用字段。

### 7.2 `manifest.json`

manifest 至少精确绑定：

- `daily_summary.jsonl` byte count + SHA-256；
- binding manifest byte count + SHA-256；
- 每个 input product manifest 的 byte count + SHA-256，按 digest 稳定排序；
- input product/session/person/date/summary counts；
- effective night window 与 `mental_health.yaml` SHA；
- fixed candidate/model/policy identity；
- builder source hash；
-全部 false decision/data-access boundary。

输出 JSON/JSONL 必须 canonical；final 目录已存在时拒绝覆盖；只允许 sibling staging，在回读 exact bytes/schema/hash/count/semantics 全部通过后用单目录原子 rename 提交。失败时只清理本次 staging，不删除竞争者或既有证据。

## 8. 建议实现面

默认 allowlist：

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_daily_summary.py
scripts/wandering/build_wandering_daily_summary.py
tests/test_wandering_camera_daily_summary.py
src/elderly_monitoring/modules/mental_health/wandering/camera_product.py
tests/test_wandering_camera_product.py
reports/mental_health/wandering_camera_mvp2s_v1/README.md
reports/mental_health/wandering_camera_mvp2s_v1/VERIFICATION.md
docs/tasks/README.md
docs/modules/mental_health/README.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/modules/mental_health/plans/M0-CAM-MVP-2S执行任务书.md
docs/README.md
README.md
```

`camera_product.py` / product test 只有在提供最小 validated final loader 时才允许修改；不得改变 MVP-1 producer/output bytes。若需要 allowlist 外 Python/config/shared 文件，停止并报告，不自行扩大范围。

建议 API/CLI：

```python
build_wandering_daily_summary(
    product_bundles: Sequence[str | Path],
    binding_manifest: str | Path,
    output_dir: str | Path,
) -> WanderingDailySummaryBuildResult
```

```bash
conda run -n eldercare-ai python scripts/wandering/build_wandering_daily_summary.py \
  --product-bundle <mvp1-product-1> \
  --product-bundle <mvp1-product-2> \
  --binding-manifest <synthetic-binding.json> \
  --output <fresh-output>
```

CLI 不提供 threshold、class order、episode merge、risk、baseline 或任意 config override。

## 9. TDD 顺序

先让下列红测在旧代码中因功能不存在而失败，再做最小实现：

1. 单 session/单 synthetic person/单日合法路径；
2. 多 session 合并到同一 person-day，以及两个 synthetic person 隔离；
3. 跨午夜：duration 分日，episode count 只在 start day；
4. DST spring-forward/fall-back：按 epoch elapsed seconds，不按 wall-clock 相减；
5. presence 重叠 union、window coverage union、不同 status overlap 明示；
6. direct 与 pacing/lapping/random 分离，wandering-like 不含 direct；
7. unavailable/error 不进入 direct/wandering-like，uncertain 字段为 null；
8. product final/primary exact readback、descriptor/hash/source scope、binding product hash；
9. binding missing/extra/nested drift、重复 session/product、track 漏绑/重绑/跨 person 重叠；
10. 缺 absolute time、无 offset、未知 IANA zone、offset/zone 不一致、无 presence、episode/window 超出 presence；
11. candidate/model/class/threshold/calibration/episode-policy 跨 bundle 漂移；
12. summary/manifest missing/extra、非空 risk/event、rogue top-level、同长度篡改；
13. deterministic order/canonical bytes、fresh-only、竞争 final、失败清理本次 staging；
14. CLI help/参数错误/Git 外 multi-session synthetic E2E；
15. MVP-1 聚焦与相邻 camera 回归不退化；既有 generic daily/baseline/pipeline/CLI 测试继续隔离通过。

攻击测试修改 artifact 时必须同步重算 canonical bytes、size 和 SHA，避免只测试旧 descriptor。

## 10. 验证命令

全部使用 `eldercare-ai`，先确认 editable：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

先窄后宽，实际文件数可以增长但不得预设通过数：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_daily_summary.py -q

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_product.py \
  tests/test_wandering_camera_daily_summary.py \
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
  scripts/wandering/build_wandering_daily_summary.py --help

git diff --check
```

另做一个 Git 外 fresh multi-session synthetic E2E，至少跨午夜，并检查 final 只有两项、回读 bytes/hash/count/空风险边界一致。不得为了 E2E 读取视频或启动 tracker。

### 执行完成记录（2026-08-14）

红测先在旧代码上因缺失 validated-product loader/daily-summary 功能而 collection 失败。实现后，最终门为 daily 聚焦 `37 passed`、MVP-1/product/daily/primary/episode/QC/inference 相邻组 `169 passed`、generic daily/baseline/pipeline/CLI 隔离组 `60 passed, 16 subtests passed`、Git 外 multi-session cross-midnight CLI E2E `1 passed`；CLI help、UTF-8/相关链接、`git diff --check`、eldercare-ai editable、空 stage 与 Git branch/HEAD 现场均通过。完整收据见 [MVP-2S 验证记录](../../../../reports/mental_health/wandering_camera_mvp2s_v1/VERIFICATION.md)。

```text
status=wandering_m0cam_mvp2s_complete
current_code_task=none
m0cam_mvp2s_status=complete
m0cam_mvp2s_implementation_complete=true
m0cam_mvp2s_acceptance_status=passed
m0cam_mvp2s_review_blocker=none
daily_summary_contract_ready=true
mvp2s_evidence_scope=synthetic_contract_only
person_binding_verified=false
eligible_for_baseline=false
algorithm_event_emitted=false

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

## 11. 完成门与后续路线

只有 production、红绿测、相邻/隔离回归、CLI、fresh E2E、报告、UTF-8、链接与 Git 现场全部通过，才可写：

```text
status=wandering_m0cam_mvp2s_complete
current_code_task=none
m0cam_mvp2s_status=complete
m0cam_mvp2s_implementation_complete=true
m0cam_mvp2s_acceptance_status=passed
m0cam_mvp2s_review_blocker=none
daily_summary_contract_ready=true
mvp2s_evidence_scope=synthetic_contract_only
person_binding_verified=false
eligible_for_baseline=false
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

MVP-2S 完成后：

- `MVP-2R`：用真实、合法且稳定的 person/session/time/presence binding 做 acceptance；它不是本任务自动启动项；
- `MVP-3`：只有真实日级 evidence 和最低观察量/冷启动规则明确后才接个人基线；
- `MVP-4`：只有负责人冻结风险级别、升级/降级和动作映射后才输出风险与 `AlgorithmEvent`；
- PORTABLE rework 仍在正式部署/M0-CAM-D 前恢复，不由 MVP-2S 顺手修复。

## 12. 可直接交给工作 AI 的目标提示词

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-MVP-2S synthetic 日级产品切片任务。

reasoning=极高（xhigh）。不要只用 high；也不要以 max/最高为理由扩大到真实身份、PORTABLE、基线、风险策略或视频数据。

唯一目标：消费一个或多个已通过 MVP-1-F1 的 synthetic product bundle，以及一份显式 wandering-daily-binding-v1 synthetic person/session/time/presence binding，输出 fresh、canonical、exact-schema 的 daily_summary.jsonl + manifest.json。产物名称是 WanderingDailySummary，不是 DailyEvidence、Risk 或 AlgorithmEvent。

第一步完整阅读并现场核对：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/plans/M0-CAM-MVP-2S执行任务书.md
4. docs/modules/mental_health/plans/M0-CAM-MVP-1执行任务书.md
5. docs/modules/mental_health/plans/M0-CAM-MVP-1-F1执行任务书.md
6. docs/modules/mental_health/plans/徘徊识别技术文档2.md
7. docs/modules/mental_health/README.md
8. reports/mental_health/wandering_camera_mvp1_v1/README.md
9. reports/mental_health/wandering_camera_mvp1_v1/VERIFICATION.md
10. reports/mental_health/wandering_camera_mvp1_v1/MVP1_REAUDIT_20260814.md
11. camera_product.py、camera_primary_inference.py、camera_episode.py、daily_aggregation.py、config.py 和相关测试。

先执行只读检查：git status --short、git diff --cached --name-only、branch、HEAD、git diff --check、eldercare-ai editable 位置。保留 dirty/untracked 现场；禁止 stash/reset/checkout/clean，未经当前用户明确授权不 stage/commit/push。

按任务书先写失败回归，再做最小实现。输入 product 必须完整重走 MVP-1-F1 final + primary exact validator；需要读取 validated rows 时，只在 camera_product.py 增加最小公开只读 loader，复用既有 validator，不复制算法、不改变 MVP-1 bytes/schema。

binding 必须显式绑定 product manifest SHA、session、带 offset 的 session start、IANA timezone、完整 source scope、synthetic person、track set 和 presence half-open intervals。不得从 track_id/文件名/轨迹推断 person；每个 track 恰好显式分配一次；presence 做 union，所有 window/episode 必须被覆盖。

按 epoch elapsed seconds 跨本地自然日和 DST 切分；episode duration 分日，count 只进 start day。window coverage 各状态分别 union，状态重叠显式报告；unavailable/error 不当 direct/negative。wandering-like 只含 pacing/lapping/random，direct 单列。uncertain 固定 null。

所有 person/baseline/risk/action/diagnosis/AlgorithmEvent 边界保持任务书规定的 false/null：person_binding_verified=false、eligible_for_baseline=false、algorithm_event_emitted=false。证据只能写 synthetic_contract_only。

默认只修改任务书 allowlist。不得修改 generic daily_aggregation.py、baseline.py、pipeline.py、config、service、PORTABLE、collection、primary/QC/episode producer、shared tracker、environment.yml 或共享依赖。需要 allowlist 外文件立即停止并报告。

不读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；不运行 detector/tracker/M0-CAM-D。

全部验证只用 eldercare-ai：先 daily-summary 聚焦，再 MVP-1/camera 相邻回归，再 generic daily/baseline/pipeline/CLI 隔离回归、CLI help、Git 外跨午夜 multi-session synthetic E2E、UTF-8/链接、git diff --check、editable 和最终 Git 现场。不要预设测试数。

最终分开报告：实现代码、自动化证据、synthetic E2E、未获得的真实证据、MVP-2S 接受状态、PORTABLE/C0/C1/M0-CAM-D 不变状态和 Git 现场。只有全部门通过才写 m0cam_mvp2s_acceptance_status=passed；不得自动开始 MVP-2R/3/4。
```
