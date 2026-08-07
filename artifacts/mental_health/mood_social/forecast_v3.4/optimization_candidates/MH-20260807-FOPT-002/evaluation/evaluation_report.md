# FORECAST-OPT-001F 评价报告

本报告比较冻结 V3.4 基线与 001D 非负融合候选。所有预测来自严格 outer OOF；本阶段只做离线评价，不接入后端或替换 V3.3。

- 运行：`MH-20260807-FOPT-002`；bootstrap：2000 次参与者聚类重采样。
- 晋级结论：`retain_v3_4_baseline`。若任一 horizon 未通过全部门槛，继续保留原 V3.4 离线基线。

## 全量结果

| Horizon | 方案 | AUPRC | AUROC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| forecast_1m | 冻结基线 | 0.4822 | 0.6641 | 0.2051 | 0.0128 |
| forecast_1m | 001D融合候选 | 0.4828 | 0.6715 | 0.2015 | 0.0194 |
| forecast_2m | 冻结基线 | 0.4869 | 0.6691 | 0.2036 | 0.0138 |
| forecast_2m | 001D融合候选 | 0.4945 | 0.6787 | 0.2005 | 0.0137 |

## Conscious 影子结果

Conscious 仅以任务内百分位排名作为无标签拟合的影子分数，不参与晋级门。
- forecast_1m：AUPRC `0.3893`，AUROC `0.5675`；资格：`False`。
- forecast_2m：AUPRC `0.4011`，AUROC `0.5787`；资格：`False`。

## 边界

OBF 已在 001E 停止迁移，CoronaHealth 延后；本报告不把外部代理或负结果写成产品效果。
