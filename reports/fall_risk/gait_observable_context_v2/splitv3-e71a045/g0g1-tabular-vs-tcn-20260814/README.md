# 步态分支 G0/G1 表观基线 vs 轻量 TCN：固定 validation 对照实验

日期：2026-08-14
状态：`development_provisional`，候选保留判定为 **No-Go**
范围：只读现有 `splitv3_e71a045eb58489f43dc5fd11`，未读取 test，未修改标签/split/主路径

## 1. 结论先行

1. 数据审计确认步态 validation 存在**确定性的来源捷径**：主口径 340 个动作段中正类仅 9 段且 100% 来自 `le2i_imvia` 单一 source group（`le2i_home_02_unknown_subject_pool`），负类 331 段中 314 段来自 `ntu_rgbd`。任何模型都可能在学"数据来源/场景风格"而非"步态稳定性"。
2. 规则 fallback（G0）在该 validation 上**与标签反相关**：`gait_risk_score` 对正常坐/起（A03/A04）均值 0.55-0.57，高于真实不稳步态（B02/B03/B04）的 0.29-0.38，ROC-AUC=0.084，F1=0.052。规则分把坐站竖直运动误判为步态风险。
3. 结构化特征上限（G1）：Logistic F1=0.310、LightGBM 0.121、EBM 0.112，均远低于保留线；且无 walking gate，误报集中在 A03/A04 坐站负类。
4. 轻量 TCN 三 seed 复现为 F1 `0.774±0.040`（最佳 seed44=0.818），复现了历史 README 单 seed 0.720 的结论；其 walking gate 把坐站误报从 Logistic 的 40 段压到 4-7 段，是唯一有相对增益的设计。
5. 判定 **No-Go**：TCN 未达保留线——最差来源（NTU）为 single_class、无法计算 Recall/F1；FP/hour 16-28 > 目标 10/小时；9 个正类来自 1 个 source group，无法声称跨来源/老人域泛化。

## 2. 数据量、split、seed、配置与代码版本

| 项 | 值 |
|---|---|
| dataset | `data/processed/fall_risk/gait_observable_context_v2/splitv3-e71a045-build-a/dataset.npz` |
| dataset SHA-256 | `6af44dfac9820e6e7eaf393e36039e50fbff560cc4e889c2f4e0641364304872` |
| split | `splitv3_e71a045eb58489f43dc5fd11`（provisional，未冻结） |
| 窗口 | 2,112（train 1,589 / validation 523；test=0，未读取） |
| 类别 | 正类(gait_instability) 202 / 负类(normal_activity) 1,910 |
| 训练监督段 | train 正类段 101（primary 60 / weak 32 / representation 9，其中 92 施加二分类 loss） |
| validation 主口径 | 343 窗 → 340 段（9 正 / 331 负） |
| validation 弱证据口径 | 126 窗（1 正 / 125 负，不用于选模） |
| 协议 | `development_provisional`；`primary_evaluation_mask` 为选模口径 |
| seeds | 42 / 43 / 44 |
| 设备 | CPU（Darwin arm64，无 CUDA/MPS） |
| 代码版本 | git `18e38ed95c9b41211c39c103a8197251144c87ba`（+ 本实验对 `gait_tabular.py` 的评估口径对齐改动） |
| test | `test_pose_read=false`、`test_evaluated=false`，未读取 test 姿态/标签/指标 |

训练正类段来源分布（train）：`pre_vfallp` 64、`le2i_imvia` 25、`fall_tiktok` 12；`caucafall`/`ntu_rgbd`/`toaga`/`ur_fall` 正类=0。validation 正类段来源分布：`le2i_imvia` 10（9 primary + 1 weak），其余来源 0。

## 3. 指标总表（validation 主口径，动作段级，3 seed）

口径：只统计 `primary_evaluation_mask` 内的 validation 动作段（340 段 = 9 正 + 331 负），段概率取同段全部窗口概率均值，阈值由各模型在 validation 上按 balanced-accuracy 选取。

| 模型 | seed 数 | F1 (mean±sd) | Precision | Recall | PR-AUC | Balanced Acc | FP 段数 | FP/hour | 最佳单模型 F1 |
|---|---|---|---|---|---|---|---|---|---|
| rule（G0，现行 fallback） | 3（确定） | 0.052 | 0.027 | 1.000 | 0.016 | 0.502 | 330 | 1324.0 | 0.052 |
| logistic（G1） | 3（确定） | 0.310 | 0.184 | 1.000 | 0.459 | 0.940 | 40 | 160.5 | 0.310 |
| lightgbm（G1） | 3（确定） | 0.121 | 0.065 | 0.889 | 0.043 | 0.771 | 115 | 461.4 | 0.121 |
| ebm（G1） | 3 | 0.112±0.001 | 0.061±0.001 | 0.630±0.052 | 0.048 | 0.684±0.014 | 75-94 | ~301 | 0.113 |
| 轻量 TCN（对照） | 3 | 0.774±0.040 | 0.632±0.054 | 1.000 | 0.755±0.091 | 0.992±0.002 | 4-7 | 16.0-28.1 | 0.818（seed44） |

TCN 逐 seed：seed42 F1=0.720 / P=0.562 / FP=7 / FP-h=28.1；seed43 F1=0.783 / P=0.643 / FP=5 / FP-h=20.1；seed44 F1=0.818 / P=0.692 / FP=4 / FP-h=16.0。logistic/lightgbm 在当前数据+配置下为确定性结果（lbfgs 与无随机子采样的 boosting），故三 seed 数值一致。

说明：`FP/hour` 分母只统计 validation 中负类段（label=0）的标注时长；`rule` 的 330 个误报主要来自 NTU 坐站负类。PR-AUC 为动作段级 `average_precision_score`。

## 4. 按来源/动作分层

validation 主口径只有 `le2i_imvia`（9 正 + 17 负）与 `ntu_rgbd`（0 正 + 314 负）两个来源；**NTU 为 single_class，Recall/F1 无法计算**，这正是跨来源泛化无法验证的根因。

| 子集 | rule F1 | logistic F1 | lightgbm F1 | ebm F1 | TCN F1 (seed44) |
|---|---|---|---|---|---|
| le2i_imvia（唯一含正类来源，26 段） | 0.529 | 0.783 | 0.667 | 0.500 | 0.900（2 FP） |
| ntu_rgbd（314 段全负类） | single_class | single_class | single_class | single_class | single_class |

动作类型分层（validation 主口径，label 语义见标签字典）：正类只有 B02=2 段、B03=4 段、B04=3 段（全 LE2I）。负类 A01=6、A03=207、A04=119。规则分在 A03/A04 上均值 0.57/0.55，在 B02/B03/B04 上仅 0.37/0.38/0.29——坐站被规则判为"步态风险"高于真实拖步/小碎步/摇晃。

弱证据口径（126 段，1 正）：TCN F1=0.15（11 FP），tabular 同样极低，不用于选模。

## 5. False positive / false negative 案例

**TCN seed44（最佳单模型，4 FP / 0 FN）**：
- FP `le2i_imvia A01 normal_walk`（`le2i_home_02_unknown_subject_pool`，prob 0.663 / 0.521）：与正类同一来源、同一房间的正常行走被判为不稳，说明在唯一有正类的来源内仍难区分"正常走"与"不稳走"。
- FP `ntu_rgbd A03 controlled_sit_down`（`ntu_rgbd_subject_p018`，prob 0.490 / 0.442）：walking gate 仍让 2 个快速/异常坐下漏过。

**logistic（40 FP / 0 FN）**，误报构成：23× NTU A03 坐下、12× NTU A04 起身、5× LE2I A01 正常行走。无 walking gate，坐站竖直运动被大量判为步态不稳。

**lightgbm（1 FN / 115 FP）**：FN `le2i_imvia B03 shuffling_walk`（prob 0.078，dur 3.0s）——真实小碎步被漏。

**ebm（4 FN / 75 FP）**：FN 依次为 `B02 dragging_walk`(prob 0.210)、`B03`(0.058)、`B03`(0.051)、`B04 swaying_walk`(0.118)，全为 LE2I 正类段。

共 40+ 误报、5 漏报案例可复核（样本充足）。误报主签名是"坐站运动/正常行走被判为步态不稳"，漏报主签名是"同一来源内的轻量步态异常被判为正常"。

## 6. 延迟、资源与失败降级

| 模型 | 参数量/体积 | CPU 推理（单窗 batch1） | 备注 |
|---|---|---|---|
| logistic | 2 KB joblib | 0.08 ms | 需 StandardScaler |
| lightgbm | 166 KB joblib | 0.16 ms | 训练/推理均 CPU |
| ebm | 148 KB joblib | 0.09 ms | interpret 依赖 |
| 轻量 TCN | 14,212 参数 / ~68 KB checkpoint | 2.37 ms | 不含姿态提取/窗口重采样 |

失败降级：所有候选的默认 checkpoint 仍为 `null`，运行主路径仍为规则 fallback。但本次审计发现规则 fallback 在该 validation 上与标签反相关（ROC-AUC=0.084），说明"低质量输入降级到规则"这一安全网本身在步态语义上不可靠——这是需要单独复核的运行侧风险，本实验不据此改动主路径。

## 7. 是否达到候选保留线

保留线（计划 §5.5）：Recall≥0.80、F1≥0.65、PR-AUC≥0.70、Balanced Acc≥0.75、最差来源 F1≥0.55、连续正常片段误报<10/小时。

| 项 | TCN 3-seed 均值 | 达标 |
|---|---|---|
| Recall | 1.000 | 是（但仅 9 个正段，1 个来源） |
| F1 | 0.774 | 是 |
| PR-AUC | 0.755 | 是 |
| Balanced Acc | 0.992 | 是 |
| 最差来源 F1 | NTU 为 single_class，**无法计算** | **否** |
| FP/hour | 16.0-28.1 | **否（>10）** |

因最差来源 F1 与误报率两项不达标，且 validation 正类规模/来源多样性远低于计划要求的"≥50 独立正段、覆盖多来源"，判定 **未达到候选保留线**。

## 8. Go / No-Go 结论

**No-Go**。不保留任何新 checkpoint、不接入运行主路径、不修改默认配置或 `AlgorithmEvent` 契约。规则 fallback 继续作为主路径与降级安全网；步态 observable-context v2 TCN 继续作为 validation-only provisional 候选。

不保留的依据不是模型容量，而是监督不足：9 个正段、1 个 source group、NTU single_class。在补齐跨来源正例前，任何 F1 都可能是来源/场景捷径，不是步态稳定性识别能力。

## 9. 修改文件、运行命令与产物路径

**代码修改**（唯一）：`src/elderly_monitoring/modules/fall_risk/gait_tabular.py`
- 将 tabular 基线的 validation 评估对齐 TCN 协议：主口径只用 `primary_evaluation_mask`，弱证据窗口单列为 `validation_sensitivity` 口径（不参与选模）；train 只保留 `sample_weights>0` 的窗口（排除 audit_only/representation_only）。
- 向后兼容：无 mask 的旧 dataset 回退到全 validation 窗口行为。
- 相关窄测试通过：`tests/test_gait_model_training.py`（26 passed）、`tests/test_fall_risk_gait.py`+`tests/test_gait_training_audit.py`（16 passed）。

**运行命令**（均在 `eldercare-ai` 环境）：
```bash
# G0/G1 表观基线，3 seed
python scripts/train/train_gait_tabular_baselines.py \
  --data data/processed/fall_risk/gait_observable_context_v2/splitv3-e71a045-build-a/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/tabular-baselines/seed42 \
  --models rule,logistic,lightgbm,ebm --seed 42   # 43 / 44 同理

# 轻量 TCN 复现，3 seed
python scripts/train/train_gait_tcn.py \
  --data data/processed/fall_risk/gait_observable_context_v2/splitv3-e71a045-build-a/dataset.npz \
  --output-dir reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/tcn-rerun/seed42 \
  --epochs 80 --patience 15 --batch-size 16 --seed 42 --device cpu   # 43 / 44 同理
```

**产物**：
- `reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/tabular-baselines/seed{42,43,44}/metrics.json` + 各模型 `*_validation_action_segment_predictions.jsonl` + `*_model.joblib`
- `reports/fall_risk/gait_observable_context_v2/splitv3-e71a045/tcn-rerun/seed{42,43,44}/{best_model.pt,metrics.json,history.jsonl}`

**完整测试**：`python -m pytest -q` → 607 passed / 3 failed / 1 skipped。3 个失败均为 `tests/test_fall_risk_baseline_longitudinal.py` 内嵌套 `subprocess.run(["conda", ...])` 因 `conda` 不在 PATH 而 `FileNotFoundError`，属预存环境问题，与本改动无关（该模块未触及步态）。

## 10. 下一项最值得投入的数据/实验（不继续堆 epoch）

1. **最高优先级：给 validation/test 补跨来源正类步态段**（拖步/小碎步/摇晃的正面/侧面/斜侧视角，来自非 LE2I 来源，目标各 ≥50 独立段、≥2-3 个保守来源组）。没有它，任何步态模型都无法通过"最差来源 F1"与"跨来源泛化"门槛。
2. **复核运行侧规则 fallback 的步态语义**：本实验发现 `gait_risk_score` 在 validation 上与标签反相关（坐站高于真实步态不稳），应先确认 `gait.py` 规则分是否需要 walking gate 或坐站抑制，否则"低质量降级"安全网本身不可靠。
3. 数据补足后，把 walking gate 引入表观模型（在 walking 段上训练二分类）与 TCN 做同口径对照，验证 gate 是否就是 TCN-vs-tabular 增益的单一来源。

在完成第 1 项前，不再训练新的步态深度模型，不重跑已判定 No-Go 的实验。
