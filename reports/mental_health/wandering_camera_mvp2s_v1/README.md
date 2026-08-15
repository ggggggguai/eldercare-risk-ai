# M0-CAM-MVP-2S synthetic 日级摘要

状态：`m0cam_mvp2s_acceptance_status=passed`

本目录记录 `M0-CAM-MVP-2S` 的软件契约实现与 synthetic 验证。产物名称固定为 `WanderingDailySummary`；它不是 `DailyEvidence`、风险结论、医学诊断或 `AlgorithmEvent`。

## 实现范围

- `camera_product.py` 新增只读 `load_validated_wandering_camera_product()`：完整回读 MVP-1 final，复用既有 primary exact-schema/hash/scope/count/probability/episode/data-access validator，并由 validated primary 重算 session evidence 与 product manifest。
- `camera_daily_summary.py` 消费一个或多个已完整验收的 MVP-1 synthetic product，以及 canonical exact-schema `wandering-daily-binding-v1`。
- binding 显式给出 product manifest SHA、session、带 offset 的 start、IANA timezone、完整 source scope、synthetic person、track set 与 presence half-open intervals；实现不从 track、文件名或轨迹推断 identity。
- 时间按 epoch elapsed seconds 前进并按 IANA 本地自然日切分；episode duration 跨日分片，count 只进入 start day；night window 固定读取 `configs/modules/mental_health.yaml`。
- presence、any-window、ready、unavailable、inference-error 分别求确定性 union；不同 status 的重叠显式进入 `status_overlap_seconds` 和 quality flag。
- `wandering_like` 只含 pacing/lapping/random，direct 单列；unavailable/error 只进入质量覆盖，不进入 direct 或 wandering-like episode。
- final 固定仅含 `daily_summary.jsonl + manifest.json`，canonical、fresh-only、sibling staging、回读通过后原子 rename。

## 固定边界

每行保持：

```text
evidence_scope=synthetic_contract_only
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
algorithm_event_emitted=false
```

manifest 保持：

```text
baseline_emitted=false
risk_or_alert_decision_emitted=false
algorithm_event_emitted=false
real_human_media_consumed=false
m0cam_d_started=false
```

## 证据边界

本任务没有读取、搜索、哈希或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；没有启动 detector、tracker、M0-CAM-D、MVP-2R/3/4，也没有修改 fixed candidate、阈值、标签、split、class order、calibration、uncertain、matching、episode policy、generic daily/baseline/pipeline/config 或 PORTABLE。

自动化命令、红测和 synthetic E2E 收据见 [VERIFICATION.md](VERIFICATION.md)。接受结论只覆盖 software contract 与 synthetic evidence；真实 MVP-2R/3/4 未启动。
