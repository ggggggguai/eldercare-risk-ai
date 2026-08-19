# M0-CAM-EP2A-S2A verification

日期：2026-08-16

环境：全部 Python/pytest 使用 WSL `eldercare-ai`；editable project location 为 `/mnt/c/Users/lenovo/Desktop/心理算法`。

## 测试

最窄 S2A：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_proposal_inference.py -q
```

历史 v1 结果：`2 passed in 8.30s`。当时覆盖 proposed forward、uncertain/rejected skip、一对一行数、端点/status/reason 保留、独立 schema、无 oracle/accepted boundary、CLI help 和 non-overwrite。该 uncertain-skip 断言已在 v2 中替换。

EP1A 聚焦：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_inference.py -q
```

结果：`11 passed in 3.39s`。

S0/S1A/EP1B/importer/primary 相邻回归：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_primary_inference.py -q
```

结果：`113 passed in 28.67s`。

共享 helper 后全 camera：

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_*.py -q
```

结果：`604 passed, 1 warning in 210.19s`；warning 仍来自既有 PORTABLE 重复 ZIP 条目拒绝测试。

`compileall` 退出 0，CLI `--help` 退出 0。D01 12 条为 `12/12 ready`，S00 三条为 `3/3 ready`；13 份当前 `episode_predictions.jsonl` 与任务开始前已有 EP1B 回归产物逐文件 SHA-256 一致，EP1A oracle flags 与概率未变。

## Smoke

历史 v1 真实 frozen synthetic CLI：3 proposal -> 3 result，状态 `ready/unavailable/boundary_uncertain=1/1/1`，forward/skipped=`1/2`。两次独立生成的 README、predictions、summary SHA-256 一致。

历史 v1 H02/H03 CLI：4 proposal -> 4 `boundary_uncertain` result，forward/skipped=`0/4`。run A/B 三个文件逐文件 SHA-256 一致。再次写同一 output 由自动化 non-overwrite 测试拒绝。

## Candidate-inclusive v2（2026-08-17）

先写 v2 断言后运行旧生产代码，得到 `3 failed, 5 passed`；三项失败均由旧
uncertain-skip 行为触发。实现 v2 后：

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_episode_proposal_review.py -q
```

结果：`8 passed in 11.21s`。覆盖 proposed+uncertain forward、rejected skip、
uncertain status/flag 保留、summary policy/count 和 S2B uncertain+ready join。

B02 fresh-output CLI：

```text
proposal_count=21
ready/unavailable/boundary_uncertain/inference_error=17/4/0/0
model_forward_invocation_count=17
model_forward_invocation_count_by_status=proposed:2, uncertain:15, rejected_by_qc:0
```

只读 evaluation 得到 all-shape-eligible `17/18` ready，binary
accuracy/macro-F1=`0.944444/0.970588`；17 条 candidate-ready binary=`17/17`。
proposal/prediction endpoint/binding mismatch=`0/0`。完整命令、hash 和证据边界见
[B02 v2 verification](../wandering_camera_b02_candidate_inclusive_development_v1/VERIFICATION.md)。

S2A runner 没有读取或生成 truth；指标由独立评估连接既有人工 truth 后计算。没有
生成 accepted boundary，也没有修改模型、0.5 threshold、类别、S0 或 matching。
