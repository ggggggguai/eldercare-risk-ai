# 跌倒事件训练 P0 审计

状态：`infrastructure_only`。本报告不是训练结果、冻结数据发布或 test 评估。

## 当前事实

- 当前 v3 split：`splitv3_e71a045eb58489f43dc5fd11`。
- v3 结构校验：`valid=true`。
- 连续背景：`0.0` camera-hour。
- 显式老人 ADL：`0` 人、`0.0` camera-hour。
- 评估协议：`development_provisional`。
- 本审计不解析 test 标签内容；test 只允许独立保管人执行一次性发布。

## 门禁结果

| 门禁 | 状态 | 说明 |
|---|---|---|
| `hash_bindings` | `passed` | 输入 hash 一致 |
| `v2_formal` | `blocked` | errors=0，blockers=285 |
| `fall_event_supervision` | `passed` | 缺失 hard-negative 类型=[] |
| `split_leakage` | `passed` | 机器报告 leakage issue=0 |
| `frozen_split` | `blocked` | split 状态=provisional_unfrozen |
| `validation_sample_scale` | `blocked` | validation primary 标签记录=7；要求至少 30 个独立事件且仍需保护组复核 |
| `hard_negative_group_scale` | `independent_group_audit_required` | 每类需至少 20 个独立保护组 |
| `continuous_background` | `blocked` | 0.0/100.0 camera-hour |
| `elderly_adl_domain` | `blocked` | 0/20 人，0.0/30.0 camera-hour |
| `evaluation_protocol` | `blocked` | status=development_provisional，缺失预注册字段=bootstrap_grouping,rejection_count_policy,test_custodian_role,test_release_policy |
| `test_release` | `custodian_required` | 需要独立保管人和一次性发布授权 |
| `candidate_report_current_split` | `rebuild_required` | 候选报告 split=['splitv3_f89832f6cad5b9c5f630d00b']，当前 split=splitv3_e71a045eb58489f43dc5fd11 |

## 推断

P0 未通过，因此当前只允许审计、工具、测试、合成数据和短 smoke。不得启动正式训练、生成 frozen 协议、读取 test 真值或替换 `FallStateDetector` 主路径。

## 建议

由数据与协议负责人依次解除 formal、独立保护组规模、连续背景、老人域、frozen split/协议和 test 保管门禁；解除前保持候选状态为 `provisional_shadow`。
