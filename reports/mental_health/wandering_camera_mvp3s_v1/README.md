# M0-CAM-MVP-3S synthetic 个人基线预览

状态：`implementation_complete`；独立复审发现的两个缺口已由 M0-CAM-MVP-3S-F1 关闭，现行 `m0cam_mvp3s_acceptance_status=passed`。见 [MVP3S_REAUDIT_20260814.md](MVP3S_REAUDIT_20260814.md) 与 [验证记录](VERIFICATION.md)。

## 交付

MVP-3S 消费一个或多个已经通过 MVP-2S 完整回读的 synthetic daily bundle，按 `synthetic_person_id` 输出 fresh、canonical、exact-schema 的：

```text
baseline_profiles.jsonl
manifest.json
```

产物名固定为 `WanderingBaselineProfilePreview`，production schema 固定为 `wandering-baseline-preview-v2` 与 `wandering-baseline-preview-manifest-v2`。v1 final 不得进入 v2 downstream，必须返回 `mvp3s_v1_reaudit_required`。它不是生产 `PersonalBaseline`、Risk 或 `AlgorithmEvent`。

实现包含：

- 在 `camera_daily_summary.py` 的 consumer-only marker 后增加 public read-only validated loader；它会真实计算 marker 前精确 producer prefix SHA，marker 缺失/重复、边界或任意前缀 byte 漂移均 fail closed，同时 MVP-2S producer schema 与合法输出 bytes 保持不变；
- 新增独立 `camera_baseline_preview.py` 和仅接受 repeatable `--daily-bundle`、`--output` 的 CLI；
- 机械可用日只要求 `presence_seconds > 0` 且 `any_window_covered_seconds > 0`，不设置额外 coverage 门槛、不填补日期；
- 使用现行配置的 3/7 日 readiness、最近 14 个可用日和 P90，按人稳定排序并产生 canonical bytes；
- profile 携带最近最多 14 个 selected `reference_days`，固定输出 trajectory coverage、presence hours、四类与 wandering-like episode count per presence hour、wandering-like candidate duration per presence hour、night wandering-like ratio；public loader 从 reference days 重算窗口、3/7 readiness 与全部 `count/median/p90/min/max`；
- 对同一 synthetic person 的 timezone、candidate/model/class/threshold/calibration/episode/night/config/duration identity 漂移返回结构化 `baseline_version_reset_required`，不跨版本混合。

零 wandering 的 count/duration rate 合法为 `0`；night ratio 无分母时为 `null`；candidate-duration rate 是可重叠 duration sum 对 presence hour 的速率，可以大于 `3600`。

## 固定边界

所有 profile 均固定：

```text
person_binding_verified=false
eligible_for_baseline=false
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

证据范围仅为 `synthetic_contract_only`。本任务没有读取真人媒体、receipt、collection、annotation、WP raw、SmartCare official/raw 或 sealed 数据；没有运行 detector、tracker 或 M0-CAM-D；没有修改 generic baseline/daily/pipeline/config、fixed candidate、模型、阈值、标签、split、PORTABLE 或共享依赖。

完整自动化与 synthetic E2E 收据见[验证记录](VERIFICATION.md)。[独立复审](MVP3S_REAUDIT_20260814.md)中的两个假通过先被固化为红测，再由 F1 最小修复关闭。

```text
status=wandering_m0cam_mvp3s_f1_complete
current_code_task=none
m0cam_mvp3s_status=complete
m0cam_mvp3s_implementation_complete=true
m0cam_mvp3s_acceptance_status=passed
m0cam_mvp3s_review_blocker=none
synthetic_baseline_profile_contract_ready=true
mvp3s_evidence_scope=synthetic_contract_only
m0cam_mvp3s_f1_status=complete
m0cam_mvp3s_f1_acceptance_status=passed
```
