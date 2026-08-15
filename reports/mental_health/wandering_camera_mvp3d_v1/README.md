# M0-CAM-MVP-3D synthetic baseline deviation preview

状态：`implementation_complete`；`m0cam_mvp3d_acceptance_status=passed`。自动化与 Git 外 synthetic 收据见[验证记录](VERIFICATION.md)。

## 交付

MVP-3D 完整回读一个 MVP-3S v2 baseline bundle 与一个或多个 MVP-2S observation daily bundle，输出 fresh、canonical、exact-schema 的：

```text
baseline_deviation_preview.jsonl
manifest.json
```

产物名固定为 `WanderingBaselineDeviationPreview`，schema 固定为 `wandering-baseline-deviation-preview-v1` 与 `wandering-baseline-deviation-preview-manifest-v1`。它只是 synthetic descriptive preview，不是真实 baseline deviation、Risk 或 `AlgorithmEvent`。

实现包含：

- 新增独立 `camera_baseline_deviation_preview.py`、public validated loader 与只接受 baseline/observation bundle/output 的 CLI；
- 生产段从首版止于固定 marker，manifest 绑定 marker 前实际 producer bytes SHA；consumer-only loader 重算 marker count、边界、prefix bytes 和 SHA；
- 只消费两个上游 public validated loader；baseline profile 必须按 synthetic person/timezone 唯一匹配，reference window 非空且最后日期严格早于 observation day，observation manifest 不得复用 baseline reference manifest；
- candidate/model/seed/epoch/class/threshold/calibration/episode/night/config/duration identity 精确一致，漂移返回 `baseline_version_reset_required`；
- 明确重算 MVP-3S 九项 observation metric；只有 ready metric 输出 signed `delta_from_median` 与 `delta_from_p90`，不复用 profile private helper；
- warming、无 window coverage、night observation null 或 reference count 0 均传播 null；candidate-duration rate 不封顶，允许大于 `3600`；
- final 精确只有两个文件，所有 staging 写完后由 public loader 回读，再原子提交；失败无 final，只清理本次 staging。

## 固定边界

每行固定：

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

非空描述对象只使用 `baseline_deviation_preview`，且不包含 abnormal、above-threshold、综合 deviation score、概率、风险或告警。

本任务没有读取、搜索、hash 或运行真人媒体、receipt、collection、annotation、detector、WP raw、SmartCare official/raw 或 sealed 数据；没有运行 detector、tracker 或 M0-CAM-D；没有修改 MVP-2S/MVP-3S producer、generic baseline/daily/pipeline/config、fixed candidate、模型、阈值、标签、split、PORTABLE 或共享依赖。

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
