# M0-CAM-MVP-3S 独立复审

日期：2026-08-14

结论：实现与既有自动化结果真实存在，但接受门重新打开。

```text
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_execution_tests_status=passed
m0cam_mvp3s_acceptance_status=rework_required
m0cam_mvp3s_review_blocker=producer_source_identity_and_reference_stats_readback
```

## 独立复验

全部使用当前 checkout 的 `eldercare-ai`：

| 门 | 独立结果 |
|---|---|
| MVP-3S + daily 聚焦 | `84 passed` |
| MVP-2S/MVP-1/camera 相邻回归 | `216 passed` |
| generic daily/baseline/pipeline/CLI 隔离 | `60 passed, 16 subtests passed` |
| Git 外 15 日 synthetic E2E | `1 passed` |
| CLI help | 通过 |

第一次相邻回归只因外层 64 秒 timeout 被终止，未产生 pytest 结论；随后使用更长 timeout 原命令重跑并得到 `216 passed`。这不是代码失败。

上述结果证明既有实现路径和测试集可运行，但没有覆盖下列两个可构造缺口。

## 发现 1：冻结的 MVP-2S producer SHA 没有核验 producer bytes

`camera_daily_summary.py` 为保持 MVP-2S 合法输出 bytes 不变，在 loader extension 后重新定义 `_sha256_file()`。当目标是当前 source 时，函数只检查 marker 存在，随后直接返回常量 `MVP2S_PRODUCER_SOURCE_SHA256`；没有计算 marker 前 producer prefix 的真实 SHA 并与常量比较。

独立只读核对确认：marker 前去除 extension 引入的一个 LF 后，prefix SHA 确实等于冻结值 `f9b2b29e...e9ca3`，因此可以在不改变 MVP-2S producer identity 的前提下做真实 prefix 校验。

临时内存 monkeypatch 把 marker 前的 `PRODUCT_NAME` 改为另一同长度值，仓库文件未修改；当前函数仍返回原冻结 SHA：

```text
SOURCE_MUTATION_CHANGED=True
RETURNED_SHA=f9b2b29e560c557dfa7d356fbccbd907113476e22e2fe7b67013ab1aa80e9ca3
```

这意味着 producer 语义可以漂移却继续自称原身份。当前报告中的“锁定 MVP-2S producer 身份”结论过强。

## 发现 2：public preview loader 不重算 reference stats

`load_validated_wandering_baseline_preview()` 会检查 exact field set、descriptor、count 和 stats 大小顺序，但没有保存或回读 selected reference-day values，因此无法从 profile final 重算 median/P90/min/max。

独立临时目录攻击把 `trajectory_coverage` 的四个统计值全部改为 `-1`，同步重算 canonical JSONL、byte count 和 SHA；当前 loader 仍接受：

```text
NEGATIVE_STATS_ACCEPTED={count: 1, median: -1.0, p90: -1.0, min: -1.0, max: -1.0}
```

只补非负范围检查仍不能关闭缺口，因为攻击者仍可写入范围内但不等于 selected days 的伪造统计。F1 必须让 final 携带有界、exact-schema 的 selected reference-day metric evidence，并由 loader 重新计算窗口、readiness 和全部 stats。

## 接受判断

这两项不否定 builder 主路径、3/7/14 日逻辑、synthetic E2E 或空风险边界，但会让 source identity 和 public validated readback 产生假通过，因此不能保持 `m0cam_mvp3s_acceptance_status=passed`。

下一任务固定为 `M0-CAM-MVP-3S-F1`。F1 只关闭上述缺口，不读取视频，不修改通用 baseline/pipeline/config，不输出 deviation/risk/`AlgorithmEvent`，也不恢复 PORTABLE。

证据范围仍为 `synthetic_contract_only`；`C0=false`、`C1=false`、`authorized_camera_data_consumed=false`、`m0cam_d_started=false`。

## F1 关闭记录

2026-08-14，M0-CAM-MVP-3S-F1 先用同长度 producer-prefix mutation 与 descriptor-aware negative stats 重现上述两个假通过，再完成最小修复：

- daily loader/builder 现在要求 marker 唯一、边界精确，并对 marker 前去除 extension 单个 LF 后的实际 bytes 计算 SHA-256；冻结值仍为 `f9b2b29e560c557dfa7d356fbccbd907113476e22e2fe7b67013ab1aa80e9ca3`，marker 前 producer bytes 未改变；
- production profile/manifest 升为 v2，每个 profile 携带 selected `reference_days`；public loader 从这些日值重算 window/count/readiness 和全部统计，不再相信自报 median/P90/min/max；
- 同步重算 canonical profile bytes、artifact size 与 SHA 的负数统计和范围内错误统计攻击均返回 `reference_stats_mismatch`；active builder source 也与实际源码 bytes 重新比较。

F1 验证门通过后，原复审 blocker 已关闭，现行状态恢复为 `m0cam_mvp3s_acceptance_status=passed`、`m0cam_mvp3s_review_blocker=none`。这只接受 synthetic contract，不产生真人身份、真实 baseline、风险、camera 性能或临床证据。
