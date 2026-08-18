# 步态模型训练审计

- 状态：`development_provisional`
- split：`splitv3_3342705b7b1ac51570148337`（`provisional`）
- 协议：`development_provisional`
- train/validation 姿态命中：4589；test sealed：834
- validation 独立正动作段：15
- test pose/tensor/metrics：均未读取或生成。

本报告只授权有界 train/validation 开发；规则步态分支继续作为主路径和 fallback。

运行决策更新（2026-08-18）：项目负责人批准 pretrained seed 43 进入比赛默认步态主分支；本报告中的 provisional、test 未读取和泛化限制仍然有效，规则分保留为 fallback。
