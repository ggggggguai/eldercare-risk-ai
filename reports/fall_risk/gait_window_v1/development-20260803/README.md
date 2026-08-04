# 步态模型开发训练报告

日期：2026-08-03

状态：开发训练完成，候选未通过替换门槛，测试集未解锁。

## 数据与协议

- 数据集：`data/processed/fall_risk/gait_window_v1/dataset.npz`
- 数据集 SHA-256：`33590a640e8033c3deaf12ad16dd8a46eaa3fa465262fcc99bbf16d095f25fea`
- split：`splitv3_32db8736e890c9fad95a8292`
- 输入：`4 FPS x 4 秒 = [16,14,5]`
- Fold A：LE2I + CaucaFall/UR Fall hard negative 训练，Pre_VFallp 验证
- Fold B：Pre_VFallp + CaucaFall/UR Fall hard negative 训练，LE2I 验证
- TCN：80 epochs 上限、patience 15、hidden channels 48、dilations 1/2/4/8、seed 42/43/44
- test：未评估；所有报告均为 `test_evaluated=false`，没有 test prediction 文件

当前 `reports/fall_risk/training-labels-v3-validation.json` 仍为 `training_ready.action_type=false`。本报告只能用于开发选模和错误分析，不是正式模型评估。

## 表格基线

以下均为验证动作段级指标，阈值只由对应验证折选择。

| 模型 | Fold | Balanced accuracy | Recall | Specificity | ROC-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Rule | A | 0.517 | 0.034 | 1.000 | 0.433 | 0.257 | 0.327 |
| Rule | B | 0.507 | 1.000 | 0.014 | 0.420 | 0.251 | 0.304 |
| LightGBM | A | 0.588 | 0.415 | 0.760 | 0.581 | 0.634 | 0.694 |
| LightGBM | B | 0.645 | 0.607 | 0.683 | 0.620 | 0.169 | 0.162 |
| EBM | A | 0.582 | 0.203 | 0.960 | 0.505 | 0.589 | 0.659 |
| EBM | B | 0.594 | 0.893 | 0.295 | 0.576 | 0.170 | 0.155 |

LightGBM 是当前最强结构化对照，两折 balanced accuracy 平均约为 0.616。Fold A 的 LightGBM/EBM 概率校准明显较差，不能直接接入运行时。

## TCN 三 seed 结果

| Seed | Fold A BA | Fold B BA | 两折均值 | 最差折 | 主要失败项 |
|---:|---:|---:|---:|---:|---|
| 42 | 0.548 | 0.639 | 0.593 | 0.548 | A recall=0.297；B recall=0.464 |
| 43 | 0.669 | 0.641 | 0.655 | 0.641 | B specificity=0.388 |
| 44 | 0.638 | 0.620 | 0.629 | 0.620 | A recall=0.517 |

完整分项：

| Seed | Fold | Recall | Specificity | ROC-AUC | Brier | ECE | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | A | 0.297 | 0.800 | 0.407 | 0.205 | 0.149 | 6 |
| 42 | B | 0.464 | 0.813 | 0.622 | 0.150 | 0.101 | 1 |
| 43 | A | 0.619 | 0.720 | 0.574 | 0.157 | 0.051 | 10 |
| 43 | B | 0.893 | 0.388 | 0.669 | 0.151 | 0.098 | 5 |
| 44 | A | 0.517 | 0.760 | 0.568 | 0.184 | 0.114 | 23 |
| 44 | B | 0.679 | 0.561 | 0.621 | 0.137 | 0.079 | 2 |

按最差折 balanced accuracy 排序，seed 43 最好；但它不能成为部署 checkpoint，因为 Fold B specificity 不合格，且两折均值低于 0.70。不同 seed 的阈值和错误方向变化很大，结论不稳定。

## Checkpoint 哈希

| Seed | Fold A SHA-256 | Fold B SHA-256 |
|---:|---|---|
| 42 | `a0656e20d1827395233e231f2916c1171b5d73c9c7f1aa178300c91c97f92f97` | `ec11d58b058df27caf449d6a76adee65efed0dff97ae27157ab3eefbedfd05f2` |
| 43 | `cba10bebf092a0095f6d7897e364bfc3cfb92b63e55701aa5001c89b64d6372c` | `43a8fc831cbe275bc27d0ec9f0015320e9f0cd97ae770bc1ef8bc05e331edacc` |
| 44 | `fa8da0e2a71767a45e7874ac2b9c1be291a0f797299e14d56a8871b54fe9631f` | `4de3624d0abb2cdc5756dae3d78887aecca75d3e3794e555b762f0c6d2b890ff` |

这些 checkpoint 仅作开发复现，不得写入 `configs/modules/fall_risk.yaml` 的默认 `checkpoint`。

## 门槛判断

| 门槛 | 结果 |
|---|---|
| 两折平均 balanced accuracy >= 0.70 | 未通过；最好 0.655 |
| 最差折 balanced accuracy >= 0.60 | seed 43/44 通过，seed 42 未通过 |
| 每折 recall/specificity 均 >= 0.55 | 三个 seed 均未通过 |
| 相对规则平均提升 >= 0.05 | 通过，但规则基线很弱 |
| 相对最强结构化 baseline 平均提升 >= 0.05 | 未通过；最好约 +0.039 |
| 三 seed 结论方向一致 | 未通过 |
| 锁定测试、误报时长、延迟和外部测试 | 尚未执行正式验收 |

结论：当前 TCN 不能替换规则主路径，也不应挑选单个最好 seed 对外报告。下一轮应优先检查 B01 语义、Fold A 来源偏移和 source/class 权重，再做预注册消融；不得查看测试集后调参。
