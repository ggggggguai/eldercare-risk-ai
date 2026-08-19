# M0-CAM-MVP-3S-F1 执行任务书

> 历史软件契约：本任务记录 synthetic baseline loader 修复，不再阻塞真实授权 camera 的薄 adapter。当前任务见 [M0-CAM-5D-T4](M0-CAM-5D-T4真实日报与个人基线.md)。

版本：1.0

更新时间：2026-08-14

状态：`complete`；`m0cam_mvp3s_f1_acceptance_status=passed`

建议工作 AI 推理等级：**极高（xhigh）**。任务范围很窄，但同时涉及冻结 source prefix、descriptor-aware 攻击、reference-day provenance、统计重算、schema 版本和既有 bundle 兼容；`high` 容易只补表面范围检查，`max/最高` 暂无必要，也不得借此扩大到风险、真人数据或 PORTABLE。

## 1. 事务定位

`M0-CAM-MVP-3S-F1` 是 MVP-3S 的有界接受门修复，不是新的产品阶段：

> 让 MVP-2S producer identity 真正绑定冻结 prefix bytes，并让 MVP-3S final 携带 selected reference-day evidence，使 public loader 能独立重算 profile window、readiness 和 reference stats。

MVP-3S 的 builder、3/7/14 日语义、版本重热和空风险边界主体保留。完整独立发现见 [MVP-3S 复审](../../../../reports/mental_health/wandering_camera_mvp3s_v1/MVP3S_REAUDIT_20260814.md)。

实时状态只看[任务表](../../../tasks/README.md)；稳定路线看[技术文档2](徘徊识别技术文档2.md)；原任务范围看 [MVP-3S 任务书](M0-CAM-MVP-3S执行任务书.md)。

## 2. 入口状态

```text
status=wandering_m0cam_mvp3s_f1_complete
current_code_task=none
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_execution_tests_status=passed
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
m0cam_mvp3s_f1_status=complete
m0cam_mvp3s_f1_acceptance_status=passed
m0cam_mvp3s_f1_evidence_scope=synthetic_contract_only
synthetic_baseline_profile_contract_ready=true

m0cam_portable_f1_acceptance_status=rework_deferred
C0=false
C1=false
authorized_camera_data_consumed=false
m0cam_d_started=false
```

制定本文时独立复现：`84 passed`、`216 passed`、`60 passed, 16 subtests passed`、Git 外 E2E `1 passed` 和 CLI help；同时用临时内存/目录稳定复现 source mutation 仍返回旧 SHA、rehashed negative stats 被 loader 接受。执行 AI 必须先把这两个攻击固化为失败回归。

Git 现场仍有大量用户未提交工作。禁止 stash、reset、checkout、clean、恢复、覆盖、stage、commit 或 push，除非当前用户另有明确授权。

## 3. 唯一目标

关闭两个 blocker：

1. `MVP2S_PRODUCER_SOURCE_SHA256` 只有在 marker 前冻结 producer prefix 的真实 SHA 完全一致时才能返回或接受；
2. MVP-3S final 必须包含足以重算 selected-window reference stats 的 exact reference-day evidence，public loader 不再相信 profile 中自报的 median/P90/min/max。

修复后仍只输出：

```text
baseline_profiles.jsonl
manifest.json
```

不增加第三个顶层文件，不计算 baseline deviation 或风险。

## 4. 明确不做

本任务不得：

1. 读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；
2. 启动 detector、tracker、M0-CAM-D、MVP-2R、MVP-3R、MVP-4 或 PORTABLE 返工；
3. 修改 generic `baseline.py`、`daily_aggregation.py`、`pipeline.py`、`mental_health.yaml`、service 或 `AlgorithmEvent`；
4. 修改 fixed candidate、模型、阈值、标签、split、class order、calibration、uncertain、matching 或 episode policy；
5. 只给 stats 加 `>=0` 范围检查后宣称完成；范围内伪造统计也必须被重算拒绝；
6. 修改 marker 前 MVP-2S producer bytes 来迎合新校验；冻结 prefix 必须保持原 SHA；
7. 把 synthetic reference-day evidence 写成真实身份、真实 baseline 或风险证据。

## 5. F1-P0：冻结 producer prefix 的真实身份

当前冻结边界为：

```text
marker = # === M0-CAM-MVP-3S READ-ONLY LOADER EXTENSION ===
expected producer SHA = f9b2b29e560c557dfa7d356fbccbd907113476e22e2fe7b67013ab1aa80e9ca3
```

实现要求：

1. 从当前 `camera_daily_summary.py` bytes 中查找唯一 marker；缺失、重复或位置异常全部 fail closed；
2. 明确定义 extension 插入时多出的单个 LF，不靠 `rstrip()`、换行归一化或文本 decode 猜边界；
3. 对精确 producer prefix bytes 计算 SHA-256，必须等于冻结值；
4. 只有校验通过后，MVP-2S builder 才可继续把冻结 SHA 写入原 v1 manifest；
5. `load_validated_wandering_daily_summary()` 也必须执行同一 prefix identity 校验；
6. loader extension 和后续 F1 consumer code 可以变化，但不能使原 MVP-2S 合法输出 bytes/schema 漂移；
7. 不在 marker 前插入 import、helper、注释或格式化改动。

失败回归至少覆盖：producer prefix 同长度变更、marker 缺失、marker 重复、边界多/少一个 byte；所有情况都不得返回冻结 SHA，不得生成 final。

## 6. F1-P1：reference-day evidence 与统计重算

### 6.1 Schema 版本

MVP-3S v1 执行证据保留，但其 public readback 不能验收。F1 使用新 schema：

```text
wandering-baseline-preview-v2
wandering-baseline-preview-manifest-v2
```

不得把已报告的 v1 bytes 静默解释成 v2。是否保留 v1 只读解析由实现便利决定，但 production builder 与后续 consumer 只接受 v2；v1 必须明确返回 `mvp3s_v1_reaudit_required` 或等价结构化 blocker，不能进入后续 deviation。

### 6.2 `reference_days`

每个 profile 新增 exact、稳定排序的 `reference_days`，只包含实际 selected window，最多 `max_window_days` 条。每条固定包含：

```text
local_date
metric_values
```

`metric_values` 的 exact key order 与 `METRIC_ORDER` 一致：

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

- 日期严格递增、无重复；
- `reference_days` 长度精确等于 `profile_window_day_count`；首尾日期精确等于 profile window 字段；
- coverage 与 night ratio 必须在 `[0,1]`；presence hours 必须 `>0`；count/duration rates 必须 `>=0`；night ratio 可以为 null；
- 零 wandering 日的 count/duration rates 为 0，night ratio 为 null；
- candidate duration rate 可以大于 3600；
- 所有数值 finite，不接受 bool、NaN 或 infinity。

### 6.3 Public loader 必须重算

`load_validated_wandering_baseline_preview()` 对 v2 必须从 `reference_days` 独立重算并逐字义比较：

1. profile window day count、first/last date；
2. 3/7 日 readiness；
3. 每个 metric 的 `count/median/p90/min/max`；
4. P90 继续使用 `(n-1)*p` 线性插值和 effective `upper_quantile`；
5. metric order、null count、rounding 与 builder 完全一致；
6. profile contract ID 必须绑定 v2 schema、metric order、profile identity 和 reference-day contract version，但不绑定统计结果本身；
7. manifest contract/profile/count/config 必须与重算结果一致。

攻击者即使同步重算 profile JSONL、manifest descriptor、byte count 和 SHA，只要 reference days 与 stats 不一致，loader 就必须拒绝。

### 6.4 Builder source identity

MVP-3S v2 manifest 的 `builder_source_sha256` 必须是当前 `camera_baseline_preview.py` 的真实 bytes SHA。public loader 必须与 active source 重新计算比较；任意自报 64 位 digest 不再足够。

## 7. 输出与失败语义

final 继续精确只有两个文件，canonical、fresh-only。失败时无 final，只清理本次 staging。

结构化错误至少区分：

```text
producer_source_identity_mismatch
invalid_reference_day_evidence
reference_stats_mismatch
active_builder_source_mismatch
baseline_version_reset_required
```

不要求在本任务解决已登记的普通 check-then-rename 极端竞争窗口；不得声称获得 PORTABLE 级并发、崩溃或 fresh-clone 原子性证据。

## 8. 默认 allowlist

```text
src/elderly_monitoring/modules/mental_health/wandering/camera_daily_summary.py
src/elderly_monitoring/modules/mental_health/wandering/camera_baseline_preview.py
tests/test_wandering_camera_daily_summary.py
tests/test_wandering_camera_baseline_preview.py
reports/mental_health/wandering_camera_mvp3s_v1/README.md
reports/mental_health/wandering_camera_mvp3s_v1/VERIFICATION.md
reports/mental_health/wandering_camera_mvp3s_v1/MVP3S_REAUDIT_20260814.md
docs/modules/mental_health/plans/M0-CAM-MVP-3S执行任务书.md
docs/modules/mental_health/plans/M0-CAM-MVP-3S-F1执行任务书.md
docs/modules/mental_health/plans/徘徊识别技术文档2.md
docs/modules/mental_health/README.md
docs/tasks/README.md
docs/README.md
README.md
```

CLI 参数和文件名不变；通常不需要修改 CLI。若需要 allowlist 外 Python/config/shared 文件，停止并报告，不自行扩大范围。

## 9. TDD 顺序

1. 将独立复审的 producer mutation 与 negative/rehashed stats 变成失败回归；
2. 增加范围内但错误的 median/P90/min/max 攻击，证明不能只修非负范围；
3. producer marker 缺失/重复/边界漂移；
4. prefix 合法时 MVP-2S producer 输出相对 F1 前 exact bytes 不变；
5. v2 reference days 合法单日、3 日、7 日、15→14 日；
6. reference day missing/extra key、乱序、重复日期、wrong first/last/count；
7. coverage/night 超界、presence 非正、negative rate、null 漂移、candidate duration >3600 合法；
8. stats 和 reference days 分别篡改并重算全部 descriptors 后拒绝；
9. active builder source mismatch；
10. v1 bundle 不得进入 v2 production consumer；
11. model/policy/config identity drift 继续返回 `baseline_version_reset_required`；
12. 风险/事件空边界、fresh-only、staging cleanup、CLI/E2E 不退化；
13. MVP-2S/MVP-1/camera 相邻回归和 generic baseline/pipeline 隔离回归通过。

## 10. 验证

全部使用 `eldercare-ai`，不预设最终通过数：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms

conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_baseline_preview.py \
  tests/test_wandering_camera_daily_summary.py -q

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

另做 Git 外 15 日 v2 synthetic E2E，回读 reference days 并独立重算全部 stats。再用同一 E2E final 做至少两次 descriptor-aware 攻击：一个负数 stats，一个范围内错误 stats；两者都必须失败且不影响合法 final。

## 11. 完成门与后续

2026-08-14 已按本任务书关闭两个 blocker：marker 前 producer prefix 的真实 SHA 校验保持冻结值 `f9b2b29e...e9ca3`，MVP-3S production schema 升为 v2 并由 public loader 从 exact `reference_days` 重算窗口、readiness 与全部统计。descriptor-aware 负数和范围内错误统计攻击均被拒绝；聚焦、camera 相邻、generic 隔离、CLI help 与 Git 外 15 日 E2E 全部通过。完整收据见 [验证记录](../../../../reports/mental_health/wandering_camera_mvp3s_v1/VERIFICATION.md)。

全部门通过后才可写：

```text
status=wandering_m0cam_mvp3s_f1_complete
current_code_task=none
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
m0cam_mvp3s_f1_status=complete
m0cam_mvp3s_f1_acceptance_status=passed
synthetic_baseline_profile_contract_ready=true
mvp3s_evidence_scope=synthetic_contract_only
```

同时保持真实边界 false/null，PORTABLE/C0/C1/M0-CAM-D 不变。

F1 通过后的下一产品功能为 `M0-CAM-MVP-3D`：使用严格早于 observation day 的 synthetic reference window，计算可解释的 per-metric deviation preview，不输出 risk/action/`AlgorithmEvent`。F1 没有自动启动它；后续用户已单独激活，现行协议见 [MVP-3D 任务书](M0-CAM-MVP-3D执行任务书.md)。

## 12. 可直接交给工作 AI 的目标提示词

```text
你接任 C:\Users\lenovo\Desktop\心理算法 的 M0-CAM-MVP-3S-F1 接受门修复。

reasoning=极高（xhigh）。不要只用 high；当前不需要 max/最高，也不得扩大到真人数据、风险、PORTABLE、模型或共享 pipeline。

第一步完整阅读：AGENTS.md、docs/tasks/README.md、M0-CAM-MVP-3S-F1执行任务书.md、M0-CAM-MVP-3S执行任务书.md、徘徊识别技术文档2.md、模块 README、MVP-3S README/VERIFICATION/MVP3S_REAUDIT_20260814.md，以及 camera_daily_summary.py、camera_baseline_preview.py 和两份测试。

先现场核对 git status、stage、branch、HEAD、git diff --check 和 eldercare-ai editable。保留全部 dirty/untracked 文件；禁止 stash/reset/checkout/clean，未经明确授权不 stage/commit/push。

唯一目标只有两个：
1. marker 前 MVP-2S producer prefix 必须真实计算 SHA 并等于冻结 f9b2b29e...e9ca3，缺失/重复 marker 或任意 prefix byte 漂移均 fail closed；不要修改 marker 前 producer bytes。
2. MVP-3S v2 profile 必须携带 selected reference_days，public loader 从它们重算 window/count/readiness 和全部 count/median/P90/min/max；同步重算 profile/manifest bytes、size、SHA 后的负数或范围内伪造 stats 都必须被拒绝。

先把复审中的两个攻击写成失败回归，再最小修复。MVP-3S production schema 升为 wandering-baseline-preview-v2 / manifest-v2；v1 不能进入 v2 下游。reference_days 只含最近 max_window_days 个 selected days，日期严格递增，metric values exact；coverage/night ratio 在[0,1]，presence>0，rates>=0，night 可 null，candidate duration rate>3600 合法。

MVP-3S v2 manifest 的 builder_source_sha256 必须与 active camera_baseline_preview.py bytes 一致。不要只补数值范围；loader 必须重算统计。继续保持 person_binding_verified=false、real_baseline_ready=false、eligible_for_risk=false，deviation/risk/action/diagnosis/AlgorithmEvent 全部为空。

默认只修改任务书 allowlist。不得读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；不得运行 detector/tracker/M0-CAM-D；不得修改 generic baseline/daily/pipeline/config、candidate/model/threshold/label/split 或 PORTABLE。

全部验证只用 eldercare-ai：先两份聚焦测试，再 camera 相邻回归、generic baseline/pipeline 隔离、CLI help、Git 外 15 日 v2 E2E、两类 rehashed stats attack、UTF-8/链接、git diff --check、editable 和最终 Git 现场。不要预设测试数。

最终分开报告：修复代码、红绿测、自动化、synthetic E2E、未获得的真实证据、MVP-3S/F1 接受状态、PORTABLE/C0/C1/M0-CAM-D 不变状态和 Git 现场。只有全部门通过才恢复 m0cam_mvp3s_acceptance_status=passed；不得自动启动 MVP-3D、MVP-2R、MVP-3R 或 MVP-4。
```
