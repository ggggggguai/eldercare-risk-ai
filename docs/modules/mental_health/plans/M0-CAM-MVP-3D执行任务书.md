# M0-CAM-MVP-3D 执行任务书

版本：1.1

更新时间：2026-08-14

状态：`complete`

建议工作 AI 推理等级：**极高（xhigh）**。本任务是有界产品功能，但同时涉及观察日泄漏、跨 bundle 身份匹配、warming/unavailable 语义、九项指标的 null 传播、exact schema 和已冻结上游源码边界；`high` 容易把同日数据混进参考窗口或把缺测当成 0，`max/最高` 暂无必要。

## 1. 事务定位

`M0-CAM-MVP-3D` 是 MVP-3S/F1 之后的第四个 synthetic 产品切片：

> 使用一个已完整回读的 MVP-3S v2 baseline profile bundle，加上一个或多个已完整回读的 MVP-2S observation daily bundle，为每个 synthetic person/day 生成 strictly-prior-day、可解释的 per-metric deviation preview。

它回答“今天的日级观察值与此前个人参考窗口相差多少”，但不回答“是否异常、是否有心理风险、是否应告警”。输出仍不是实际个人基线、风险评分或 `AlgorithmEvent`。

实时状态只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；上游接受证据见 [MVP-3S/F1 报告](../../../../reports/mental_health/wandering_camera_mvp3s_v1/README.md)和[验证记录](../../../../reports/mental_health/wandering_camera_mvp3s_v1/VERIFICATION.md)。

## 2. 入口状态

```text
status=wandering_m0cam_mvp3d_complete
current_code_task=none
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_f1_acceptance_status=passed
synthetic_baseline_profile_contract_ready=true
m0cam_mvp3d_status=complete
m0cam_mvp3d_started=true
m0cam_mvp3d_acceptance_status=passed
baseline_deviation_preview_contract_ready=true
m0cam_mvp3d_evidence_scope=synthetic_contract_only

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

MVP-3D 完成复验为聚焦 `155 passed`、相邻 camera `287 passed`、generic `60 passed, 16 subtests passed`、Git 外 7 日 reference + 后续 observation E2E `1 passed` 和 CLI help 通过。完整记录见 [MVP-3D 验证报告](../../../../reports/mental_health/wandering_camera_mvp3d_v1/VERIFICATION.md)。这些只证明 synthetic 软件契约。

Git 工作树仍有大量用户未提交内容。禁止 stash、reset、checkout、clean、恢复、覆盖、stage、commit 或 push，除非当前用户另有明确授权。

## 3. 唯一目标

新增独立 builder 和 CLI，消费：

```text
一个 validated MVP-3S v2 baseline bundle
+
一个或多个 validated MVP-2S observation daily bundle
```

输出 fresh：

```text
baseline_deviation_preview.jsonl
manifest.json
```

产品名固定为 `WanderingBaselineDeviationPreview`，证据范围固定为 `synthetic_contract_only`。

## 4. 明确不做

本任务不得：

1. 修改 `camera_daily_summary.py`、`camera_baseline_preview.py` 或其既有 producer/schema/合法输出 bytes；
2. 修改 generic `baseline.py`、`daily_aggregation.py`、`pipeline.py`、`mental_health.yaml`、service 或 `AlgorithmEvent`；
3. 读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；
4. 启动 detector、tracker、M0-CAM-D、MVP-2R、MVP-3R、MVP-4 或 PORTABLE 返工；
5. 定义 abnormal、risk、alert、action、diagnosis、升级/降级策略或阈值；
6. 计算 z-score、概率、置信度或单一综合 deviation score；当前参考日太少且没有冻结尺度；
7. 把同日或未来 observation、缺测日、unavailable 窗口当成历史参考或正常 0 值；
8. 把 synthetic person token 写成已验证真人身份。

## 5. 冻结上游边界

任务启动时只读复算并要求以下已接受源码 bytes 不变：

```text
camera_daily_summary.py
size=63082
sha256=22b79b815cc8bb042b36e786ded932b29eb2946a72ed092a1e7d7d0ef62b2f08

camera_baseline_preview.py
size=52439
sha256=99b2219ea4ff7adc2c95ab5cab1013175718d1a6451746b595c5108f8d83b1e6
```

若现场不一致，先判断是否已有其他用户改动；不得覆盖。没有明确的新上游接受记录时停止并报告冲突。

MVP-3D 新模块从第一版就分开 producer prefix 与 consumer-only loader extension。manifest 的 `builder_source_sha256` 对 marker 前实际 prefix bytes 计算；public loader 重算并比较。不得嵌入一个未经计算的自报 digest，也不得因以后追加 loader 而改变 producer identity。

## 6. 输入与无泄漏匹配

### 6.1 完整回读

1. baseline 输入只通过 `load_validated_wandering_baseline_preview()` 取得；只接受 v2；
2. observation 输入只通过 `load_validated_wandering_daily_summary()` 取得；
3. CLI 不接受外部 stats、reference date、threshold、risk policy 或任意 JSON patch；
4. baseline bundle 的 `(synthetic_person_id, timezone)` profile 必须唯一；每个 observation row 必须精确匹配一个 profile；
5. observation `(synthetic_person_id, timezone, local_date)` 在全部输入中必须唯一。

### 6.2 Strictly-prior-day

对每个 observation：

```text
profile_window_last_local_date < observation_local_date
```

同日、未来 reference、空 reference window 均不得计算 deviation。observation daily manifest SHA 还不得出现在 baseline manifest 的 `input_daily_manifests` 中，避免把同一 daily bundle 同时作为 reference 与 observation。

### 6.3 版本身份

observation 的 candidate/model/seed/epoch/class order/binary threshold/calibration/episode policy/night window/daily config/duration semantics 必须与 profile identity 精确一致；漂移统一返回：

```text
baseline_version_reset_required
```

不得跨版本比较，也不得自动重置或重建 baseline。

## 7. Observation 与 deviation 语义

### 7.1 Observation metric

沿用 MVP-3S 的九项 `METRIC_ORDER` 和同一日级计算公式：

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

不要从 profile 的 private helper 偷跑生产逻辑；MVP-3D consumer 在新模块中按已验收 daily row 明确计算，并用独立测试复算 parity。candidate-duration rate 不是占比，可以大于 3600；night ratio 可以为 null。

### 7.2 总状态

每个 person/day 的 `deviation_preview_status` 只允许：

```text
observation_unavailable
profile_warming_up
ready_preview
```

判定顺序：

1. `any_window_covered_seconds <= 0` → `observation_unavailable`，不得把 episode/rate 当成可信 0；
2. profile readiness 为 `warming_up` → `profile_warming_up`；
3. profile readiness 为 `initial_ready_preview|stable_ready_preview` 且 observation 可用 → `ready_preview`。

这些状态都不是风险等级。

### 7.3 每项 deviation

`baseline_deviation_preview` 按 `METRIC_ORDER` 固定九项。每项 exact 字段：

```text
status
observation_value
reference_count
reference_median
reference_p90
delta_from_median
delta_from_p90
```

`status` 只允许：

```text
ready
observation_unavailable
profile_warming_up
observation_value_unavailable
reference_unavailable
```

只有 `ready` 才计算：

```text
delta_from_median = observation_value - reference_median
delta_from_p90    = observation_value - reference_p90
```

保持 signed delta，不输出 `abnormal`、`above_threshold`、风险分或动作。若 night observation 为 null、reference count 为 0、profile warming 或 observation unavailable，对应 delta 必须为 null，不能补 0。

### 7.4 真实决策边界

每行继续固定：

```text
person_binding_verified=false
real_baseline_ready=false
baseline_deviation=null
eligible_for_risk=false
risk_level=null
risk_score=null
recommended_action=null
alert_decision=null
medical_diagnosis=null
algorithm_event=null
algorithm_event_emitted=false
```

非空的只有 `baseline_deviation_preview`，它明确属于 synthetic descriptive preview。

## 8. 输出契约

建议 schema：

```text
wandering-baseline-deviation-preview-v1
wandering-baseline-deviation-preview-manifest-v1
```

final 精确只有两个顶层文件，canonical UTF-8、稳定排序、fresh-only。manifest 至少绑定：

- baseline manifest descriptor；
- observation daily manifest descriptors；
- profile contract IDs；
- observation person/day count；
- metric order 与 signed-delta semantics；
- builder producer-prefix SHA；
- `real_human_media_consumed=false`、`risk_or_alert_decision_emitted=false`、`algorithm_event_emitted=false`。

所有 staging 写完后回读 public loader，再提交 final；失败无 final，只清理本次 staging。无需把本任务扩大成 PORTABLE 级 crash/concurrency 工程。

结构化错误至少区分：

```text
invalid_baseline_bundle
invalid_observation_daily_bundle
baseline_profile_not_found
duplicate_observation_day
reference_window_not_prior
observation_manifest_reused_as_reference
baseline_version_reset_required
invalid_deviation_preview_bundle
active_builder_source_mismatch
```

## 9. 默认 allowlist

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_baseline_deviation_preview.py
scripts/wandering/build_wandering_baseline_deviation_preview.py
tests/test_wandering_camera_baseline_deviation_preview.py
reports/mental_health/wandering_camera_mvp3d_v1/README.md
reports/mental_health/wandering_camera_mvp3d_v1/VERIFICATION.md
docs/modules/mental_health/plans/M0-CAM-MVP-3D执行任务书.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/modules/mental_health/README.md
docs/tasks/README.md
docs/README.md
README.md
```

`camera_daily_summary.py`、`camera_baseline_preview.py` 及其既有测试只读。若发现必须修改上游 producer、generic pipeline/config 或 allowlist 外 Python 文件，停止并报告，不自行扩大范围。

## 10. TDD 顺序

1. 先写不存在 MVP-3D builder/loader/CLI 的失败回归；
2. 用 3 日 initial 与 7 日 stable reference profile 加一个严格后续 observation day，独立复算九项 signed delta；
3. same-day、future reference、空 reference window 和 observation manifest 被 reference 复用全部拒绝；
4. missing profile、duplicate person/day、timezone 与完整 profile identity 漂移拒绝；
5. profile warming 时全部 delta null；
6. observation 无 window coverage 时不把 episode/rate 当 0；
7. night observation null 与 reference count 0 的 per-metric 状态正确；
8. candidate-duration rate >3600 合法；
9. 输入顺序不改变 canonical output；
10. row/manifest missing/extra、nested metric missing/extra、非空 risk/event、rogue 顶层文件在重算 descriptor 后仍拒绝；
11. builder producer-prefix source 漂移拒绝，consumer-only extension 不改变 producer identity；
12. fresh-only、owned staging cleanup、validated readback 和 CLI 参数边界；
13. 现有 MVP-3S/F1、MVP-2S、MVP-1/camera 与 generic baseline/pipeline 回归不退化。

## 11. 验证

全部使用 `eldercare-ai`，不预设最终通过数：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_baseline_deviation_preview.py \
  tests/test_wandering_camera_baseline_preview.py \
  tests/test_wandering_camera_daily_summary.py -q

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_baseline_deviation_preview.py \
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
  scripts/wandering/build_wandering_baseline_deviation_preview.py --help

git diff --check
```

另做 Git 外 synthetic CLI E2E：历史 bundle 至少 7 个 reference days，observation 至少 1 个严格后续日期；不用 production deviation helper，独立复算全部 ready metric 的 signed delta。再做同日泄漏、reference-manifest 复用和 descriptor-aware nested deviation 篡改，均须失败且不影响合法 final。

## 12. 完成门与后续

全部门通过后才可写：

```text
status=wandering_m0cam_mvp3d_complete
current_code_task=none
m0cam_mvp3d_status=complete
m0cam_mvp3d_acceptance_status=passed
baseline_deviation_preview_contract_ready=true
mvp3d_evidence_scope=synthetic_contract_only

person_binding_verified=false
real_baseline_ready=false
eligible_for_risk=false
algorithm_event_emitted=false
```

同时保持 PORTABLE/C0/C1/M0-CAM-D 不变。MVP-3D 通过后产品代码线先停在 deviation preview；没有负责人冻结风险映射时，不自动启动 MVP-4，也不把 synthetic preview 接入通用评分主链。

## 13. 可直接交给工作 AI 的目标提示词

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-MVP-3D synthetic baseline deviation preview 任务。

reasoning=极高（xhigh）。不要只用 high；当前不需要最高/max，也不得扩大到真人数据、风险策略、PORTABLE、模型或共享 pipeline。

第一步完整阅读并现场核对：AGENTS.md、docs/tasks/README.md、M0-CAM-MVP-3D执行任务书.md、M0-CAM-MVP-3S-F1执行任务书.md、徘徊识别技术文档2.md、模块 README、MVP-3S README/VERIFICATION，以及 camera_daily_summary.py、camera_baseline_preview.py 和对应测试。

先执行 git status --short、git diff --cached --name-only、branch、HEAD、git diff --check 和 eldercare-ai editable 检查。保留全部 dirty/untracked 现场；禁止 stash/reset/checkout/clean，未经明确授权不 stage/commit/push。

唯一目标：消费一个 fully validated MVP-3S v2 baseline bundle 和一个或多个 fully validated MVP-2S observation daily bundle，生成 fresh、canonical、exact-schema 的 baseline_deviation_preview.jsonl + manifest.json。输出叫 WanderingBaselineDeviationPreview，只是 synthetic descriptive preview。

不得修改 camera_daily_summary.py、camera_baseline_preview.py、generic baseline/daily/pipeline/config、fixed candidate、模型、阈值、标签、split 或 PORTABLE。先只读确认两个上游 source size/SHA 与任务书一致。

先红后绿。每个 observation 必须匹配唯一 profile；baseline 最后日期必须严格早于 observation 日期；observation manifest 不能出现在 baseline input manifests；candidate/model/class/threshold/calibration/episode/night/config/duration identity 必须一致。禁止同日/未来泄漏。

沿用九项 MVP-3S metric。profile warming 或 observation 无 coverage 时不得计算 delta；night/current/reference 不可用时对应 metric delta 为 null。ready metric 只计算 signed delta_from_median 和 delta_from_p90，不输出异常判断、综合分、概率或告警。

继续固定 person_binding_verified=false、real_baseline_ready=false、baseline_deviation=null、eligible_for_risk=false，risk/action/diagnosis/AlgorithmEvent 全部为空。非空的只能是 baseline_deviation_preview。

新模块从第一版就把 producer prefix 与 consumer-only loader extension 分开；manifest 绑定 marker 前实际 producer bytes SHA，loader 重算。默认只修改任务书 allowlist。

不得读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；不得运行 detector/tracker/M0-CAM-D。

全部验证只用 eldercare-ai：先 MVP-3D/MVP-3S/daily 聚焦，再 camera 相邻回归、generic baseline/pipeline 隔离、CLI help、Git 外 synthetic E2E、strictly-prior leakage 攻击、UTF-8/链接、git diff --check、editable 和最终 Git 现场。不要预设测试数。

最终分开报告：实现代码、红绿测、自动化、synthetic E2E、未获得的真实证据、MVP-3D 状态、PORTABLE/C0/C1/M0-CAM-D 不变状态和 Git 现场。只有全部门通过才写 m0cam_mvp3d_acceptance_status=passed；不得自动启动 MVP-2R、MVP-3R 或 MVP-4。
```
