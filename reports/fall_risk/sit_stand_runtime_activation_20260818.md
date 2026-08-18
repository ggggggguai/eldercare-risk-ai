# 坐站 TCN 比赛期主链启用记录

状态：`development_provisional`。决定日期：2026-08-18。

## 决定

项目最高执行人明确接受当前模型未达到正式替换门禁的风险，批准在比赛交付阶段将坐站默认运行分支从规则 baseline 改为 TCN-first。规则未删除：TCN 无事件、输入不足或推理失败时自动回退，并在分支诊断中记录 `score_source` 与 `fallback_reason`。

## 固定产物

- checkpoint：`reports/fall_risk/sit_stand_continuous_tcn_v2/sitstandsplit_11e163684e3bb898a5c3508d/seed42-pilot-loss-scale-fixed/best_model.pt`
- checkpoint SHA-256：`efcc12dbb2c243214f89348a991ebf595061b918e4fda04f4215707443ce756b`
- dataset SHA-256：`83c218eca416d9dda566d29f25cf7cc080e196a5f58625693c846d5b00453d98`
- 运行设备：CPU
- 运行策略：每个 track 只推理最新 causal cutoff

## 已知边界

同一 development validation 上，当前 TCN 的 Event F1/Recall/FP-hour 为 `0.449/0.419/158.8`，规则为 `0.168/0.382/1142.8`。TCN 相对规则更好，但仍低于 Event F1 `0.75`、Recall `0.85` 的正式门禁；显式背景只有 `0.353 camera-hour`，老人域、冻结协议和正式 test 证据仍不完整。本决定只改变比赛期默认运行分支，不把模型状态改写为正式验收通过。
