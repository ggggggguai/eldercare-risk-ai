# M0-CAM-EP2A-S0 continuous episode boundary proposal

日期：2026-08-16

状态：

```text
m0cam_ep2a_s0_implementation_status=completed
m0cam_ep2a_human_boundary_status=available_b01_b02
m0cam_ep2a_evaluation_status=completed_development_plus_b02_historical_boundary
m0cam_ep2a_data_status=initial_truth_free_smoke_plus_b01_b02_manual_evaluation
m0cam_ep2a_status=not_completed
```

初始 S0 报告范围：`partial_truth_free_smoke_only`；后续 B01/B02 人工评价见文末 current addendum。

## 实现结果

S0 已形成独立的 proposal-only 工具线：

```text
tracking JSONL + wandering-media-v1 sidecar
  -> CameraObservation + full-scope grouping
  -> accepted observations + weighted 0.5 s buckets
  -> existing Camera QC technical hard breaks
  -> idle/open/closing/closed/uncertain state machine
  -> wandering-camera-episode-boundary-proposal-v1
  -> mandatory manual review
```

实现文件：

```text
configs/modules/wandering_camera_episode_boundary_proposal_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_episode_boundary.py
scripts/wandering/propose_camera_episode_boundaries.py
tests/test_wandering_camera_episode_boundary.py
```

producer 复用 `CameraObservation`、`group_observations()`、`weighted_bucket_observations()`，并直接调用 `camera_qc._split_observations()`。因此 `long_internal_gap`、`suspected_id_switch` 和 `height_position_discontinuity` 仍只有 Camera QC 中的一份权威算法。没有修改冻结的 `camera_qc.py` 字节，也没有为 proposal 复制 gap/jump/height 判定。

状态机语义：

- `idle`：只累积 accepted bucket 的持续位移证据，低置信度边缘点和原地抖动不打开 episode；
- `open`：候选 locomotion bout 已打开，方向改变、端点反转、闭环、random 换向和起终点接近均不切段；
- `closing`：连续 bucket 的 body-height-normalized 位移进入静止候选；15 秒前恢复运动返回 `open`；
- `closed`：持续静止达到候选值后，以 stationary run 开始处形成软 offset；后续持续位移可以打开下一条；
- `uncertain`：track/stream end、技术 hard break、短候选或证据不足时保留区间和 reason，并要求人工复核。

2026-08-16 功能复审补充了两个此前窄测未覆盖的边界条件：同一 0.5 秒 bucket 内出现技术 hard break 时，相邻技术段现在以两侧 accepted observation 的时间中点作为共同边界，避免 proposal 重叠；`closing` 现在累计相对静止锚点的最大漂移，持续缓慢平移不能再被当成高置信 stationary dwell。后者保留候选区间，但状态降为 `uncertain` 并记录 `stationary_candidate_drift`。

默认候选参数为 0.5 秒 bucket、confidence `>=0.25`、3 bucket 内累计位移 `>=0.25` body heights、单步静止位移 `<=0.08` body heights、持续静止 15 秒、最短候选 2 秒。所有值都只由 protocol 与合成行为测试固定，`movement_start_candidate_validated=false`、`stationary_dwell_candidate_validated=false`；没有在 H02/H03 上调参或搜索最优值。

## Proposal 与 accepted boundary

每条输出包含 `start_reason/end_reason/reason_codes`、`hard_break_reasons`、`uncertain`、`manual_review_required=true`、source/accepted observation 数和 bucket 数。状态为 `proposed|uncertain|rejected_by_qc`。

proposal schema 固定为：

```text
wandering-camera-episode-boundary-proposal-v1
```

它不是 EP1A 接受的：

```text
wandering-camera-episode-boundary-v1
```

单测确认把 `proposals.jsonl` 直接交给正式 boundary loader 会因字段/schema 隔离而拒绝。S0 没有 acceptance/export 工具，没有修改 CVAT XML、EP1B truth 或现有 boundary，也没有读取 shape/purpose truth、加载冻结模型、训练或推理。

## H02/H03 truth-free smoke

输入为现有 Git-ignored development tracking/sidecar：H02 三段、H03 一段。N01 没有可用 tracking/sidecar，未伪造。

| source | duration s | proposal interval(s) s | status | end reason | accepted observations | observed buckets | hard-break segments |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| H02-01 | 44.733 | `[8.500,44.733)` | uncertain | track_or_stream_end | 671/671 | 90/90 | 0 |
| H02-02 | 28.000 | `[11.500,28.000)` | uncertain | track_or_stream_end | 420/420 | 56/56 | 0 |
| H02-03 | 22.133 | `[15.500,22.133)` | uncertain | track_or_stream_end | 332/332 | 45/45 | 0 |
| H03-01 | 80.733 | `[1.500,50.000)` | uncertain | sustained_stationary + stationary_candidate_drift | 1211/1211 | 162/162 | 0 |

四个输入的 accepted-observation coverage 与 observed-bucket coverage 均为 `1.0`，technical-segment locomotion coverage 均为 `1.0`。合计形成 4 条 locomotion proposal：0 条 `proposed`、4 条 `uncertain`、0 条 `rejected_by_qc`；4 条全部 `manual_review_required=true`。

这些数量、区间和 reason 只描述 producer 在无 truth 输入上的行为。H03 的单段 `uncertain` 不能解释为正确切成一段，H02 输出一段也不能解释为没有漏切；本报告不使用“漏切/多切”语言评价它们。

两轮分别写入：

```text
tmp/m0cam_ep2a_s0_truth_free_smoke_20260816_v2/run_a/
tmp/m0cam_ep2a_s0_truth_free_smoke_20260816_v2/run_b/
```

每个 source 的 `README.md/diagnostics.jsonl/proposals.jsonl/summary.json` 在两轮间逐文件 SHA-256 一致。对已有 `run_a/h02-01` 的第三次 CLI 调用退出非零并返回 `FileExistsError`，没有覆盖原 bundle。

## Current addendum（2026-08-17）

上面的 H02/H03 段落只证明 S0 初始 implementation/truth-free smoke。后续 B01/B02 manual CVAT 已提供人工 continuous boundary：B01 development 冻结 producer/matching，B02 boundary 主视图为 P/R/F1=`1.0/0.944444/0.971429`、mean tIoU=`0.731254`。这些后续结果不改变 H02/H03 的 truth-free 性质，完整指标见 [B01/B02 报告](../wandering_camera_b01_development_v1/README.md)。

后继 S1A/S2A/S2B implementation、人工 boundary 和 B01 policy freeze 已完成。负责人授权后，candidate-inclusive S2A v2 在复用 B02 上得到 `17/18` ready，但该结果是 post-hoc development/exploratory，不是新 holdout。当前只保留独立泛化证据线：

1. 获取并预声明新 participant 和/或新 camera setup；
2. 原样运行 frozen S0/matching 与 candidate-inclusive S2A v2，不再用 B01/B02 调 v2；
3. 分开报告 boundary、automatic candidate shape 与 EP1B oracle shape；
4. uncertain 仍不是 accepted boundary，人工修正端点只能单列 semiautomatic/oracle。

S1A 的详细代码契约见 [M0-CAM-EP2A-S1A 执行任务书](../../../docs/modules/mental_health/plans/M0-CAM-EP2A-S1A执行任务书.md)，当前独立证据顺序见[EP1B / EP2A 证据交接](../../../docs/modules/mental_health/plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md)。

EP1B 后续已在 B01/B02 manual-CVAT oracle-boundary descriptive pilot 范围 completed；旧 15 条 AI 预标注未进入正式成绩。proposal 仍不能替代人工 truth。

详细验证命令和实际结果见 [VERIFICATION.md](VERIFICATION.md)。
