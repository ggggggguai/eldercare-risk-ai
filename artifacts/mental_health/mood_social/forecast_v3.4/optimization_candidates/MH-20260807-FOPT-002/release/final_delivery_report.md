# FORECAST-OPT-001G 交付报告

本报告整理 001F 正式评价结果并完成交付隔离复核。候选不进入后端，不替换 V3.3 在线模型。

- 运行：`MH-20260807-FOPT-002`；评价 manifest：`b6a9809fd237a5aa83251f7ca82e506e92f4fafb2c911926e70a640bb5027a30`。
- 晋级结论：`retain_v3_4_baseline`；推荐回退：`MH-20260805-FCAST-001`。
- 当前在线模型：`mood-fusion-v3.3.3` / `MH-20260802-013`。

## 结果摘要

| Horizon | 基线 AUPRC | 候选 AUPRC | 差值 | 候选 AUROC | 候选 Brier | 候选 ECE | horizon 决定 |
|---|---:|---:|---:|---:|---:|---:|---|
| forecast_1m | 0.482186 | 0.482799 | +0.000614 | 0.671453 | 0.201506 | 0.019376 | retain_baseline |
| forecast_2m | 0.486916 | 0.494514 | +0.007598 | 0.678700 | 0.200531 | 0.013655 | promote_candidate |

## 验收与隔离

- 001F 已完成 2,000 次参与者 bootstrap、共同窗口配对、逐折稳定性、模型/校准消融和 Conscious 影子候选评价。
- 两个 horizon 必须同时通过冻结晋级门；1m 未通过，因此整体保留原 V3.4 离线基线。
- V3.3 acceptance、在线包 manifest/SHA256SUMS、算法管线/包选择代码和后端情绪接口文件均完成哈希登记。
- 后端静态 Forecast 命中数：`0`；V3.3 在线模型：`mood-fusion-v3.3.3`。
- 状态固定为 `experimental / offline_only / shadow_only / product_visible=false`。

## 限制

PSYCHE-D 是名义月份条件实验，不是中国老年人或项目摄像头/睡眠仪设备域验证；输出不是医学诊断。Conscious/OBF 结果只作辅助证据，不能扩大为产品效果。
