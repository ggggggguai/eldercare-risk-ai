# M0-CAM-5D-T4 真实日报与个人基线

状态：`completed_real_development_with_replay_readiness_evidence`

计划：Day 4

上游：[T2 自动 Episode 与轨迹识别闭环](M0-CAM-5D-T2自动Episode与轨迹识别闭环.md)

下游：[T5 算法对接包与整体验收](M0-CAM-5D-T5算法对接包与整体验收.md)

## 目标

把真实授权 camera episode results 聚合为 person/day 报告，并复用既有 3/7/14 日统计规则形成个人滚动基线和偏离输出。

W5D-04 的硬上游是已经完成的 T2 episode result 契约。顺序上在 W5D-03A core 后执行，但不依赖居家 MP4、居家人工标签或可用的真实多模态 provider；当前可直接使用 B01+B02 real-development result 和显式身份/时间/presence binding 开发与验收。

## 完成事实（2026-08-18）

- B01 36 条、B02 12 条视频已展开为 48 个显式 `person/session/timezone/presence` binding；人员来自 owner-bound development index，不从 track ID 或文件名推断；
- 129 条 T2 result 聚合为 `P-OFFICE-01` 的两个自然日，95 条可用 shape、81 条 uncertain、34 条 unavailable、1 条 high-confidence wandering-like 和 617.8 秒 wandering-like duration 输入/输出守恒；
- sidecar 没有 capture clock，故 manifest 明示 declared development schedule；真实两日只支持 day-level development aggregation，不支持实测日内节律结论；
- 两个真实 profile/deviation 都是 `warming_up`，严格使用 observation day 之前的 0/1 个 usable day；
- 独立 15 日 deterministic replay 验证 3/7/14 日状态机，得到 3/4/8 个 warming/initial/stable profile，第 15 日最多引用严格 prior 的 14 日；不得把 replay 写成实际长期观察；
- public loader 会核验 descriptor/schema/hash 并重新计算 profile/deviation；fresh/non-overwrite、跨午夜、多人隔离、低覆盖 null 和 context 不反改 shape 均有回归。

最终证据见 [B01+B02 real-development bundle](../../../../reports/mental_health/wandering_camera_daily_baseline_w5d04_v1/README.md) 和 [15-day deterministic replay](../../../../reports/mental_health/wandering_camera_daily_baseline_w5d04_replay_v1/README.md)。

## 输入优先级

1. B01+B02 的 T2 real-development episode results，用于真实 camera row 到 person/day 的连接。
2. W5D-03 context rows；provider 不可用或尚未生成时允许空/`unavailable`，不得阻断日报。
3. 显式 `person_id/session_id/timezone/presence intervals` binding；不能从 `track_id` 推断。
4. 确定性多日 fixture/replay，只用于验证跨午夜和 3/7/14 日状态机，必须标记为 fixture/replay。
5. 居家视频 result 到位后作为补充输入复放，不改变既有 schema。

## 实现原则

- 新增或扩展薄 real-development adapter，不复制整套 synthetic daily/baseline 算法。
- 显式输入 `person_id/session_id/timezone/presence intervals`，不能从 track ID 或文件名推断 person。
- 使用绝对时间和 IANA timezone 切自然日；正确处理跨午夜 episode。
- `unavailable/uncertain/error` 分开计数；低覆盖日不能自动填零。
- high-confidence 与 uncertain wandering candidate 分开聚合。
- context 只作为解释字段，不反向改变 shape count。
- 缺少居家输入时记录实际 `validation_scope=b01_b02_development|deterministic_replay`，不把缺失写成零事件日。

## 连续执行步骤

1. 固定 real-development binding manifest 和 schema。
2. 将 T2 episode result 与可选 context row 转成真实 session evidence。
3. 按 IANA timezone/presence 聚合 daily report，保留 high-confidence、uncertain、unavailable 和 error。
4. 使用严格早于 observation day 的记录构建 baseline profile/deviation。
5. 用多日 fixture/replay 验证 1/3/7/14 日与跨午夜边界。
6. 输出 fresh bundle、运行摘要、输入范围和已知限制；随后推进 W5D-05。

## 日报最小字段

```text
module=mental_health
person_id
local_date
timezone
presence_seconds
tracking_coverage_seconds
episode_count_by_shape
episode_duration_by_shape
wandering_like_count
wandering_like_duration_seconds
wandering_like_count_per_presence_hour
wandering_like_duration_per_presence_hour
night_wandering_like_ratio
high_confidence_candidate_count
uncertain_candidate_count
context_counts
unavailable_count
error_count
quality_flags
baseline_readiness
```

## 基线规则

```text
usable days 1-2: warming_up
usable days 3-6: initial_ready
usable days >=7: stable_ready
maximum reference window: latest 14 usable days
```

当历史不足、presence 为零、coverage 不足或 reference 为空时输出 null/status，不制造零偏离。

## 完成门

- 同一 person 的多 session 正确合并，不串人；
- 跨午夜正确分日；
- 一批 B01+B02 real-development episode 能生成 daily report；
- 多日 fixture/replay 能验证 3/7/14 日 readiness；
- baseline profile 和 deviation 可由 public loader 重新计算；
- 单日真实视频显示 `warming_up`，不伪造 stable baseline；
- 居家输入缺席不产生 run error；到位后可按同一 binding/adapter 补充复放；
- 输出不包含医学诊断。

## 验证命令

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_daily_baseline.py \
  tests/test_wandering_camera_daily_summary.py \
  tests/test_wandering_camera_baseline_preview.py \
  tests/test_wandering_camera_baseline_deviation_preview.py \
  tests/test_mental_health_daily_aggregation.py \
  tests/test_mental_health_baseline.py -q
```
