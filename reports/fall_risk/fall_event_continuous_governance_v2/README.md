# 跌倒连续模型训练数据治理 v1

状态：`provisional_training_data_governed`。本产物不是模型效果、冻结发布或主链路替换证据。

## 数据结果

- 开发候选监督：7701 条。
- 可物化训练/验证监督：7701 条。
- 唯一开发视频：5338 个。
- 姿态缺失视频：0 个。
- 封锁 test 标签：3730 条；未解析语义、未读取姿态。

## 采样与监督

- 事件标签优先；动作标签只补充普通背景与恢复型近跌倒负监督。
- train 权重按监督桶、数据来源、主体三级逆频率归一。
- 只有 primary LE2I 官方精确边界正例训练 onset；辅助整段边界只训练 presence。
- 窗口扩增后必须按 `normalization_group_id` 做事件内归一，防止长事件支配 loss。
- 清洗姿态中的插值坐标及其派生运动必须屏蔽，不能作为在线因果输入。

## 门禁

| 门禁 | 状态 |
|---|---|
| `hash_bindings` | `passed` |
| `v3_fall_event_supervision` | `passed` |
| `test_isolation` | `passed` |
| `development_pose_coverage` | `passed` |
| `frozen_split` | `blocked` |
| `continuous_background` | `blocked` |
| `elderly_adl_domain` | `blocked` |
| `online_causal_pose_cleaning` | `blocked` |
| `main_path_replacement` | `blocked` |

## 边界

本治理扩大了可用显式监督，但没有补齐连续背景 camera-hour、老人域、冻结 split/协议或在线因果姿态清洗，因此不能据此替换规则主路径。
