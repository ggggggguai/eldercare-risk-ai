# M0-CAM-EP2A-S1A continuous boundary evaluator

日期：2026-08-16

状态：

```text
m0cam_ep2a_s1a_implementation_status=completed
m0cam_ep2a_human_boundary_status=available_b01_b02
m0cam_ep2a_evaluation_status=completed_development_plus_b02_historical_boundary
m0cam_ep2a_data_status=initial_synthetic_contract_plus_b01_b02_manual_evaluation
m0cam_ep2a_status=not_completed
```

初始 S1A 报告范围：`synthetic_contract_only`；后续 B01/B02 人工评价见文末 current handoff。

## 实现结果

S1A 已形成独立、只读的 boundary evaluator：

```text
S0 proposal bundle
  + independent boundary-only JSONL
  + wandering-media-v1 sidecar
  + participant/session/camera-setup/clock binding
  -> exact input validation and full-scope isolation
  -> shared camera_development.match_episodes_one_to_one()
  -> all_locomotion_candidates = proposed + uncertain
  -> proposed_only_conditional = proposed
  -> metrics, coverage, diagnostics and row-level failures
```

实现文件：

```text
configs/modules/wandering_camera_episode_boundary_eval_v1.yaml
src/elderly_monitoring/modules/mental_health/wandering/camera_episode_boundary_evaluation.py
scripts/wandering/evaluate_camera_episode_boundaries.py
tests/test_wandering_camera_episode_boundary_evaluation.py
```

evaluator 复用 `camera_episode_import.load_episode_boundaries()` 读取独立 boundary-only view，复用 `camera_development.match_episodes_one_to_one()` 完成最大基数优先的一对一匹配。没有修改 `camera_development.py`、`camera_qc.py` 或 S0 producer，也没有复制 matching 或 technical hard-break 算法。

输入按 source group/video/device/setup/stream epoch/track、participant、session、camera setup 和 clock domain 完整隔离。重复 proposal/episode ID、跨 batch 物理重复、source identity drift、同 scope 重叠 ready truth、非有限数、非法 authorization 或 schema 漂移均 fail closed。原 proposal、human boundary、sidecar 和 batch index 在构建前后做字节快照检查；输出只允许 fresh directory，并以临时目录原子提交。

## 功能复审修复

2026-08-16 的完成后审计发现并关闭了三处功能缺口：

1. 原时长分层把“按 proposal duration 计算的 precision”和“按 truth duration 计算的 recall”直接合成 F1；跨时长带匹配时两个总体不同，可能产生没有统计意义的完美 F1。现在显式输出 same/cross-band match counts、precision/recall population；只要该时长带存在跨带匹配，band F1 固定为 `not_computable`。
2. 原 fragmentation 诊断只按 camera/track scope 分组，可能把技术 hard break 两侧的 proposal 当成相邻碎片。现在强制按 `technical_segment_index` 隔离，并把该字段纳入 bound proposal 的验证契约。
3. 原测试全部使用 `truth_source=synthetic_fixture`，没有执行正式的 `independent_human` loader 分支。现在新增 authorized independent-human E2E，并确认 continuous boundary 评价拒绝 `boundary_source=whole_clip`。

这些修复不改变 S0 proposal、共享 matcher、EP1A oracle inference 或 EP1B shape evaluator；完整 camera 回归已重跑。

## 评价语义

主视图 `all_locomotion_candidates` 使用 `proposed + uncertain`，条件视图 `proposed_only_conditional` 只使用 `proposed`。两个视图分别报告 ready truth support、candidate support、matched/unmatched、precision/recall/F1 和 matched localization；条件视图不能单独冒充最终 automatic-boundary 成绩。

matched localization 包含 temporal IoU、onset/offset signed error 和 absolute error。总体之外保留 participant/session/camera-setup/clock-domain 分组支持及 short/medium/long 时长支持。时长分层显式区分 proposal-duration precision population 与 truth-duration recall population；跨时长带匹配时不合成 band F1。空 matched cohort 返回 `not_computable`。

`rejected_by_qc` 不进入 locomotion candidate TP 分母，但保留在 coverage、unmatched 和 failure 证据中；human `boundary_uncertain` 可见但不进入 ready truth 性能分母。split、merge、subthreshold overlap 和 fragmentation 只输出诊断行，不自动合并、修改或接受 proposal。

当前 matching policy 与 failure thresholds 均为 synthetic contract：

```text
policy_id=m0cam-ep2a-s1a-synthetic-unvalidated-v1
matching_policy_validated=false
failure_thresholds_validated=false
```

这些参数没有在 H02/H03 或其他无 truth tracking 上优化。

## Synthetic CLI smoke

smoke 使用测试内构造的两个 synthetic source。输入共有 9 条 `ready` boundary、1 条 `boundary_uncertain` 和 10 条 proposal，其中 7 条 `proposed`、2 条 `uncertain`、1 条 `rejected_by_qc`。它覆盖双视图分母、最大基数 matching、完整 scope、localization error、状态 coverage、分组/时长支持、split/merge/subthreshold/fragmentation 和失败清单。

正式 CLI 在两个全新输出目录各运行一次：

```text
tmp/m0cam_ep2a_s1a_audit_20260816/
  test_deterministic_non_overwri0/cli-output-a/
  test_deterministic_non_overwri0/cli-output-b/
```

测试 builder 的 `output-a/output-b` 与正式 CLI 的 `cli-output-a/cli-output-b` 均为 fresh output。两轮 CLI 各生成 8 个文件，逐文件内容完全一致。CLI 两次均返回 `ready_truth_count=9`、`uncertain_truth_count=1`、`proposal_count=10`、主视图 `match_count=5`、`failure_count=28`。主视图 synthetic support 为 TP/candidate/truth=`5/9/9`，条件视图为 `4/7/9`。

这些 synthetic 数字只证明实现可以复算预设合同。它们不是 H02/H03 或真人数据的 boundary precision/recall/F1、tIoU、起止误差、漏切、多切或 automatic-boundary 性能，不能用于参数选择或科学结论。

## Evidence boundary

输出 summary 固定声明：

```text
truth_source=synthetic_fixture
evidence_scope=synthetic_contract_only
matching_policy_validated=false
proposal_modified=false
human_boundary_modified=false
shape_predictions_consumed=false
shape_truth_consumed=false
frozen_model_loaded=false
automatic_shape_evaluated=false
models_trained_or_updated=false
automatic_boundary_evaluation_completed=false
```

本任务没有读取 EP1B shape truth/AI 预标注，没有给 H02/H03 制造 boundary，没有运行 shape inference、训练、阈值搜索、alert/risk/`AlgorithmEvent`，也没有修改 XML、truth 或 proposal。

## Current evidence handoff

后继 [M0-CAM-EP2A-S2A](../../../docs/modules/mental_health/plans/M0-CAM-EP2A-S2A执行任务书.md) 和 [S2B](../../../docs/modules/mental_health/plans/M0-CAM-EP2A-S2B执行任务书.md) implementation 均已完成。B01/B02 manual CVAT 后续已进入 S1B：B01 development 冻结 policy，B02 boundary P/R/F1=`1.0/0.944444/0.971429`、mean tIoU=`0.731254`。当前 `current_code_task=none (waiting_for_independent_camera_generalization_evidence)`。

EP2A 总完成当前仍缺：

1. 在预声明的新 participant/setup 上原样验证 frozen S0/matching 与 candidate-inclusive S2A v2；
2. 报告独立 boundary 指标和 failure cases；
3. 使用 producer 原始端点单列 automatic candidate shape，uncertain 不得写成 accepted boundary；
4. 与 EP1B oracle-boundary shape 结果保持分开。

EP1B 后续已在 B01/B02 manual-CVAT oracle-boundary descriptive pilot 范围 completed；旧 15 条 P01/L01/R01/H01/H04/H05 AI 预标注未进入正式成绩。S1A synthetic fixture 和 proposal 仍不能替代人工 truth。

新数据的人工标注仍必须先盲于 proposal/prediction 观看完整连续视频并覆盖所有 locomotion episode；S2B review pack 只能在人工记录锁定后作定位和差异复核。具体见[EP1B / EP2A 证据交接](../../../docs/modules/mental_health/plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md)。

详细命令和实际结果见 [VERIFICATION.md](VERIFICATION.md)。
