# 跌倒事件训练阻塞清单

当前状态：`infrastructure_only`。仅列出未通过或仍需独立复核的门禁。

## Formal blockers

- `formal_manifest_ineligible`：27
- `formal_uncertain`：258

## P0 blockers

- `v2_formal`：errors=0，blockers=285
- `frozen_split`：split 状态=provisional_unfrozen
- `validation_sample_scale`：validation primary 标签记录=7；要求至少 30 个独立事件且仍需保护组复核
- `hard_negative_group_scale`：每类需至少 20 个独立保护组
- `continuous_background`：0.0/100.0 camera-hour
- `elderly_adl_domain`：0/20 人，0.0/30.0 camera-hour
- `evaluation_protocol`：status=development_provisional，缺失预注册字段=bootstrap_grouping,rejection_count_policy,test_custodian_role,test_release_policy
- `test_release`：需要独立保管人和一次性发布授权
- `candidate_report_current_split`：候选报告 split=['splitv3_f89832f6cad5b9c5f630d00b']，当前 split=splitv3_e71a045eb58489f43dc5fd11

## 停止边界

- 不创建 `fall_event_v2.frozen.yaml` 或 frozen split。
- 不读取 test 真值，不训练、选模、调阈值或拟合校准器。
- 不把 candidate-clip 指标解释为连续事件指标。
- 不修改规则主路径或 `AlgorithmEvent` 契约。
