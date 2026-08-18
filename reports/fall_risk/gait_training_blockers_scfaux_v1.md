# 步态模型训练阻塞清单

- `validation_scale`：未通过。
- `frozen_split`：未通过。
- `frozen_evaluation_protocol`：未通过。
- `action_type_supervision`：未通过。
- `test_release`：未通过。

正式效果验收仍依赖人工补数、B03 primary 监督、split/协议冻结和独立 test 保管人签发；这些 blocker 不撤销 2026-08-18 的比赛受控运行启用，运行中继续保留规则 fallback。
