# M0-CAM-EP2A-S2A proposal-shape bridge

日期：2026-08-16

```text
m0cam_ep2a_s2a_implementation_status=completed
m0cam_ep2a_human_boundary_status=available_b01_b02
m0cam_ep2a_evaluation_status=post_hoc_development_completed_on_reused_b02
m0cam_ep2a_s2a_data_status=implementation_plus_b02_post_hoc_development_evaluation
m0cam_ep2a_status=not_completed
```

## 实现

本阶段只增加最小 automatic-proposal + frozen-shape bridge：

- `wandering_camera_episode_proposal_shape_v1.yaml` 固定 EP1A preprocessing 与同一 frozen candidate；
- `camera_episode_proposal_inference.py` 读取最小 batch index、S0 proposal bundle、原 tracking 和 sidecar，一条 proposal 输出一条独立 proposal-shape result；
- `run_camera_episode_proposal_inference.py` 只暴露 config、batch index 和 fresh output 参数；
- `camera_episode_inference.py` 抽出 interval QC/preprocessing 与 ready-interval forward 两个低层 helper。EP1A 仍先验证 accepted boundary，并保持原 prediction/summary 与 `oracle_boundary` flags；S2A 只写 `automatic_boundary_proposal`。

当前 v2 状态映射：

| S0 status | S2A 行为 |
| --- | --- |
| `proposed` | 原始 start/end 进入 EP1A 同数学的 accepted-observation、episode QC、80 点预处理；QC ready 才调用冻结模型 |
| `uncertain` | 保持原 status/start/end/reason/manual-review；进入同一 episode QC，ready 后调用冻结模型，并标为 `boundary_uncertain_candidate_diagnostic`；不升级为 proposed/accepted boundary |
| `rejected_by_qc` | `prediction_status=unavailable`，加入 `proposal_rejected_by_qc`，跳过模型且不丢行 |

输出只有 `proposal_shape_predictions.jsonl`、`summary.json` 和短 README；没有 accepted boundary、人工修正端点、truth、XML、purpose、风险、告警或 `AlgorithmEvent`。

## 2026-08-16 历史 proposed-only smoke

以下内容记录初始 proposed-only v1，不再描述当前 v2 行为。synthetic bundle 含 3 条 proposal：`proposed/uncertain/rejected_by_qc=1/1/1`。未 monkeypatch 的 CLI 加载固定 candidate，历史结果为：

```text
proposal_count=3
result_count=3
ready/unavailable/boundary_uncertain/inference_error=1/1/1/0
model_forward_invocation_count=1
model_invocation_skipped_count=2
```

唯一 `proposed` 行真实到达 frozen model，输入 tensor 为 `1x80x14`、`1x80x2`、`1x80`；结果携带固定 manifest/model-state identity。这里的预测类别只证明 forward 跑通，不是 camera shape 性能。

## H02/H03 历史 truth-free skip smoke

以下表格也是 2026-08-16 旧策略 smoke。使用 S0 修复后的 v2 原 proposal，未修改 status、endpoint 或 reason：

| bundle | 原区间（秒） | S0 status | S2A status | model |
| --- | --- | --- | --- | --- |
| H02-01 | 8.5–44.7333 | uncertain | boundary_uncertain | skipped |
| H02-02 | 11.5–28.0 | uncertain | boundary_uncertain | skipped |
| H02-03 | 15.5–22.1333 | uncertain | boundary_uncertain | skipped |
| H03-01 | 1.5–50.0 | uncertain | boundary_uncertain | skipped |

覆盖仅为 4/4 原 proposal 的 identity/endpoint/status/reason 保留与 skip 行为；`model_forward_invocation_count=0`、`model_invocation_skipped_count=4`。run A/B 的三个输出逐文件 SHA-256 一致。

## Candidate-inclusive v2 B02 development

2026-08-17 负责人明确授权复用已看过的 B02 调整 S2A 非模型送模门禁。新增 policy：

```text
m0cam-ep2a-s2a-proposed-plus-uncertain-development-v2
```

B02 21 条 S0 row 为 `2 proposed + 15 uncertain + 4 rejected_by_qc`，v2 实际
forward/skip=`17/4`。17 条 boundary match 全部 ready；18 条 all-shape-eligible
主视图 ready=`17/18`，binary accuracy/macro-F1=`0.944444/0.970588`；
candidate-ready binary=`17/17`。17 条 match 的 endpoint/binding mismatch=`0/0`。

uncertain 仍保持 uncertain，只是其 shape prediction 可见；没有 accepted boundary，
没有修改 model/0.5 threshold/classes/producer/matching/truth。因为策略是在查看 B02
后建立并复跑，结果固定为 post-hoc development/exploratory，不是 held-out。完整
结果见 [B02 v2 报告](../wandering_camera_b02_candidate_inclusive_development_v1/README.md)。

## 证据边界

S2A runner 自身不生成 truth 或 boundary 指标。B02 v2 的 shape 指标由独立只读评估连接已有人工 truth/S1A match 后生成；runner 仍不接受 truth 输入。proposal、synthetic forward 和 uncertain ready prediction 都不是 accepted boundary。没有调整 S0/matching、0.5 threshold、标签或模型，也没有计算 alert/FAR/临床指标。

后继 S2B review pack、B01/B02 人工 boundary 和 S1B producer/matching freeze 均已完成。当前没有未阻塞代码任务；下一缺口是在预声明的新 participant/setup 上原样验证 candidate-inclusive v2。EP2A 保持 `not_completed`，具体输入见[证据交接](../../../docs/modules/mental_health/plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md)。

完整命令与实际测试结果见 [VERIFICATION.md](VERIFICATION.md)。
