# M0-CAM B02 candidate-inclusive shape development evaluation

日期：2026-08-17

```text
shape_inference_policy_id=m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2
evidence_scope=post_hoc_development_exploratory_not_held_out
m0cam_ep2a_candidate_inclusive_implementation_status=completed
m0cam_ep2a_candidate_inclusive_evaluation_status=completed_on_reused_b02_development
m0cam_ep2a_status=not_completed
```

## 决策与范围

负责人明确授权在已经查看过结果的 B02 上调整 S2A 非模型策略。旧策略把
`uncertain` 直接写成 `boundary_uncertain` 并跳过模型；新策略把“边界是否可自动
接受”和“该原始候选能否送入冻结 shape 模型”分离：

| S0 status | v2 行为 |
| --- | --- |
| `proposed` | 原 endpoint 通过 episode QC 后调用冻结模型 |
| `uncertain` | 保持 `proposal_status=uncertain`、原 endpoint/reason 和人工复核要求；通过 episode QC 后调用冻结模型，结果角色为 `boundary_uncertain_candidate_diagnostic` |
| `rejected_by_qc` | 保持 `prediction_status=unavailable`，继续跳过模型 |

本次没有重训、调参、改模型、改 `0.5` threshold、改类别或改人工 truth，也没有
把 `uncertain` 提升为 `proposed`/accepted boundary。旧 B02 frozen-video holdout v1
仍作为历史快照保留；因为 v2 是看过 B02 结果后建立并再次在 B02 上运行，本报告
只能称为 post-hoc development/exploratory，不能再称为 held-out。

## B02 结果

B02 继续使用原来的 11 条长视频、18 条人工 shape-eligible episode 和 S1A
一对一匹配。boundary 主视图未修改：17/18 matched，precision/recall/F1 为
`1.0/0.944444/0.971429`，mean tIoU=`0.731254`。唯一 boundary miss 仍是
`vid-b02-0007-cvat-11`；其对应自动区间是 `rejected_by_qc`，没有强行送模。

| shape 视图 | support | ready | binary accuracy | binary macro-F1 | subtype accuracy | subtype macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all shape eligible（主视图，boundary miss 算 pipeline miss） | 18 | 17 | 0.944444 | 0.970588 | 0.833333 | 0.638889 |
| candidate-ready conditional | 17 | 17 | 1.000000 | 1.000000 | 0.882353 | 0.666667 |
| uncertain-only diagnostic | 15 | 15 | 1.000000 | 1.000000 | 0.866667 | 0.666667 |
| proposed-only conditional | 2 | 2 | 1.000000 | not_computable | 1.000000 | not_computable |

主要 camera shape 结果是独立 binary head。四分类只是 subtype diagnostic；两条
pacing 仍预测为 lapping，符合既知摄像头投影坍缩，不触发重训、改标签或改阈值。

21 条 S0 row 的状态为 `proposed/uncertain/rejected_by_qc=2/15/4`。新策略实际
forward/skip=`17/4`，17 条 boundary-matched candidate 全部 ready；按 18 条人工
shape-eligible episode 计算，ready coverage 从历史旧策略的 `2/18=0.111111`
提高到 `17/18=0.944444`，绝对增加 `0.833333`。这说明旧短板确实主要来自
`uncertain` 的送模门禁，而不是冻结 binary classifier。

## 完整性校验

- 17 条 match 的 proposal/prediction 视频、track、status 和原 start/end 全部一致；endpoint/binding mismatch=`0/0`。
- 15 条 uncertain 的 `proposal_status=uncertain` 与 `uncertain=true` 均保留，并带 `boundary_uncertain_candidate` flag。
- 4 条 `rejected_by_qc` 全部 `unavailable + model_invocation_skipped=true`。
- candidate/model identity 分别为 `topowander-m0s-seed20260731-epoch0005` 与 `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031`。
- binary class order、four-class order 与 threshold `0.5` 未变；未读取 oracle prediction，未生成 accepted boundary。

## 证据边界与下一步

本结果关闭了“代码仍只允许 proposed 送模”和“B02 exploratory coverage 只有
2/18”两个当前事实，但不关闭 EP2A：B01/B02 是同一 participant、同一机位，且
v2 是使用已经查看过的 B02 建立并评价，没有独立验证。下一证据应在预声明的
新 participant 和/或新 camera setup 上原样运行 v2；在该证据到位前，不继续用
B01/B02 调 v2，不启动 EP2B/EP2C/EP3、alert、risk 或 `AlgorithmEvent`。

机器指标见 [metrics.json](metrics.json)，逐 episode 结果见
[automatic_shape_results.jsonl](automatic_shape_results.jsonl)，实际命令与 hash 见
[VERIFICATION.md](VERIFICATION.md)。
