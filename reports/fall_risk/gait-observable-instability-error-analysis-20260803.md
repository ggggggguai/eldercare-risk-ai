# 可观察异常步态目标与错误分析

日期：2026-08-03

状态：开发实验完成；停止继续调参，测试集未解锁，模型不得部署。

## 目标契约

本轮将步态任务显式定义为 `observable_instability_b02_b04`：

- 正类：B02 `dragging_walk`、B03 `shuffling_walk`、B04 `swaying_walk`。
- 负类：A01-A12 正常活动与 hard negative。
- 独立功能 proxy：B01 `slow_walk`，不再作为异常步态正类。

数据集为 `data/processed/fall_risk/gait_window_v3_observable_instability/dataset.npz`，包含 1,777 个窗口，SHA-256 为 `b0fbf265528c9da1af8f371251fa9bfe2d6ccdfa156d293c10c1aafe47d2b2b4`。元数据中的 `target_contract` 记录上述边界；B01 不存在于训练数组。该 profile 没有生成新的根标签，也不把 B01 冒充 `functional_proxy_v1` 的参与者级真值。

## 受控特征消融

以下均为 LightGBM 验证动作段指标，阈值只由对应验证折选择。

| 特征 profile | Fold A BA | Fold B BA | 两折均值 | 最差折 | Fold A R/S | Fold B R/S |
|---|---:|---:|---:|---:|---:|---:|
| 全部 25 个特征 | 0.822 | 0.619 | 0.721 | 0.619 | 0.804 / 0.840 | 0.375 / 0.863 |
| 移除 5 个质量字段 | 0.681 | 0.670 | 0.676 | 0.670 | 0.522 / 0.840 | 0.938 / 0.403 |
| 移除质量字段和 `hip_width_mean` | 0.692 | 0.690 | 0.691 | 0.690 | 0.543 / 0.840 | 0.625 / 0.755 |

全部特征模型在两折都把 `mean_core_keypoint_quality` 排为第一重要特征，说明存在姿态质量/数据源捷径。只移除质量字段后，Fold A 又把 `hip_width_mean` 排为第一，且 Fold B 产生大量误报。最终 profile 同时移除这六个字段，跨来源最差折明显提高，recall/specificity 也更一致，但仍未达到：

- 两折平均 balanced accuracy >= 0.70；当前 0.691。
- 每折 recall/specificity 均 >= 0.55；Fold A recall=0.543。
- 稳定校准；Fold A Brier=0.460、ECE=0.506。

不能为了 0.009 的均值差或一个动作段继续围绕验证集调阈值、树深和特征组合。

## 动作级错误

最终 profile 的错误数：

| Fold/验证源 | 正类漏检 | 负类误报 |
|---|---|---|
| Fold A / Pre_VFallp | B02 5/20；B03 1/1；B04 15/25 | A01 4/25 |
| Fold B / LE2I | B02 1/4；B03 2/6；B04 3/6 | A01 12/68；A02 10/24；A03 4/28；A04 8/19 |

Fold A 的主要失败是 Pre_VFallp B04 漏检；Fold B 的主要失败是 LE2I 正常动作误报，尤其 A02 和 A04。B03 的跨来源覆盖严重不对称：LE2I 有 6 个 primary 验证动作段，Pre_VFallp 只有 1 个。

逐动作段预测已经包含 `action_segment_id`、`video_id`、`sample_group_id`、`action_id`、数据源、概率和阈值结果：

- Fold A：`reports/fall_risk/gait_window_v3_observable_instability/development-20260803/tabular-exclude-quality-scale/fold-a/lightgbm_validation_action_segment_predictions.jsonl`
- Fold B：`reports/fall_risk/gait_window_v3_observable_instability/development-20260803/tabular-exclude-quality-scale/fold-b/lightgbm_validation_action_segment_predictions.jsonl`

两份文件中共有 65 个 `label != predicted_label` 的动作段需要优先双人复核：Fold A 25 个，Fold B 40 个。复核重点是动作边界、遮挡/出画、错误人物轨迹、B03/B04 区分和 A02-A04 是否实际包含异常步态片段。

## 当前独立动作段覆盖

| 来源 | A01 | A02 | A03 | A04 | B02 | B03 | B04 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pre_VFallp primary | 25 | 0 | 0 | 0 | 20 | 1 | 25 |
| LE2I primary | 68 | 24 | 28 | 19 | 4 | 6 | 6 |

这不是简单的类别不平衡，而是“动作类型和数据源几乎绑定”：Pre_VFallp 没有 A02-A04，LE2I 的 B02-B04 又很少。模型无法区分动作信号和采集域信号。

## 可执行补标清单

下面数量是下一轮开发的工程收集下限，不是统计功效保证：

| 优先级 | 来源/场景 | 动作 | 当前 primary | 至少新增 | 目的 |
|---|---|---|---:|---:|---|
| P0 | Pre_VFallp 同类采集域 | B03 | 1 | 19 | 补齐最缺失的异常 subtype |
| P0 | LE2I/家庭监控域 | B02 | 4 | 16 | 让拖步跨来源可验证 |
| P0 | LE2I/家庭监控域 | B03 | 6 | 14 | 让小碎步跨来源可验证 |
| P0 | LE2I/家庭监控域 | B04 | 6 | 14 | 降低摇摆步域绑定 |
| P0 | Pre_VFallp 同类采集域 | A02 | 0 | 20 | 补正常转身 hard negative |
| P0 | Pre_VFallp 同类采集域 | A03 | 0 | 20 | 补受控坐下 hard negative |
| P0 | Pre_VFallp 同类采集域 | A04 | 0 | 20 | 补正常坐站 hard negative |

每个动作应尽量覆盖至少 5 个独立人员/保守来源组，不用同一长视频切出大量相邻窗口凑数。新增数据完成双人复核后，重新冻结 split；不能把当前 validation 动作段回流训练后继续沿用本报告指标。

## 决策

- 保留 `exclude_quality_and_scale` LightGBM 作为下一轮结构化开发对照，不配置到运行时。
- 停止在当前数据上继续调 TCN、LightGBM 超参数或阈值。
- 先复核 65 个现有错误，再完成 P0 跨来源补标。
- 新标签冻结后重新生成 Fold A/B；只有开发门槛通过，才讨论校准和一次性 test 解锁。

当前 `training_ready.action_type=false`，默认步态 checkpoint 必须保持 `null`。

## 无新增数据约束下的最终审计

在明确没有新增数据后，又增加了带有效样本权重的标准化 Logistic Regression，以检查低方差线性模型能否改善跨来源泛化。固定 `C=1.0`，没有做参数搜索：

| 特征 profile | Fold A BA | Fold B BA | 两折均值 | 主要失败项 |
|---|---:|---:|---:|---|
| 全部特征 | 0.704 | 0.669 | 0.687 | Fold B specificity=0.338 |
| 移除质量字段 | 0.704 | 0.633 | 0.669 | Fold B specificity=0.266 |
| 移除质量与尺度 | 0.679 | 0.656 | 0.667 | Fold A recall=0.478；Fold B specificity=0.374 |

线性模型同样失败，且全部特征模型仍把 `mean_core_keypoint_quality` 作为最大系数之一。这进一步确认瓶颈是数据源与动作类型绑定，不是 TCN 或树模型容量。

冻结分区也不能作为替代开发 split：

| Partition | Negative windows | Positive windows | Positive action segments |
|---|---:|---:|---:|
| train | 658 | 15 | B02=3、B03=3、B04=2 |
| validation | 689 | 2 | B04=2 |
| test | 231 | 146 | B02=23、B03=5、B04=34 |

validation 只有 2 个正窗口，无法可靠选择模型、阈值或校准器。test 中虽然有更多正例，但将其回流训练或调参会破坏唯一独立测试证据。因此在“不新增数据、保持 test 锁定”的约束下，当前数据已经没有合规的进一步训练空间。

最终工程处置：

- `exclude_quality_and_scale` LightGBM 仅保留为离线 shadow/错误分析候选。
- 线上和演示主路径继续使用可解释规则 fallback，不配置开发模型。
- 不再运行新的特征子集、阈值、模型容量或集成权重搜索。
- 若未来仍无新数据，只能报告 Fold A/B 开发结果和限制，不能宣称正式准确率或模型已替换主路径。
