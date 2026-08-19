# M0-CAM B02 candidate-inclusive verification

日期：2026-08-17

全部 Python/pytest 使用 WSL `eldercare-ai`。`pip show
elderly-monitoring-algorithms` 的 editable project location 为：

```text
/mnt/c/Users/lenovo/Desktop/心理算法
```

## 聚焦测试

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_episode_proposal_review.py -q
```

结果：`8 passed in 11.21s`。

覆盖 proposed+uncertain forward、rejected skip、uncertain 状态/flag 保留、summary
policy/count，以及 S2B 对 uncertain+ready 的 fail-closed join。首次生产代码修改前的
红测为 `3 failed, 5 passed`，失败均来自旧的 uncertain-skip 行为。

相邻 S0/S1A/S2A/S2B/EP1A/EP1B/importer/primary 回归：

```text
133 passed in 37.22s
```

全部 camera 回归：

```text
611 passed, 1 warning in 168.11s
```

唯一 warning 是既有 portability 测试故意构造的重复 ZIP entry
`candidate/candidate_manifest.json`。

## B02 S2A

使用旧 B02 tracking/S0 bundle，写入 fresh output：

```bash
conda run -n eldercare-ai python \
  scripts/wandering/run_camera_episode_proposal_inference.py \
  --batch-index tmp/m0cam_b02_frozen_video_heldout_20260817_v1/s2a_batch_index.jsonl \
  --output-dir tmp/m0cam_b02_candidate_inclusive_shape_20260817_v1/s2a_proposal_shape
```

结果：

```text
proposal_count=21
ready/unavailable/boundary_uncertain/inference_error=17/4/0/0
model_forward_invocation_count=17
proposed forward/ready=2/2
uncertain forward/ready=15/15
rejected_by_qc forward/ready=0/0
```

## 只读评估

评估脚本为 `tmp/evaluate_m0cam_candidate_inclusive_shape_20260817.py`。它读取旧
人工 truth/S1A matches 和新 S2A bundle，校验 binding/endpoint 后生成本目录，
并拒绝覆盖既有 output。评估成功退出 0；endpoint/binding mismatch=`0/0`。

关键 SHA-256：

```text
metrics.json                         2e8f5e966fdb03ac562b50c047504b825f21f41556b94cebea5beb95e6ac9866
automatic_shape_results.jsonl        edd465aa58ab994643e9c8867437df6561a4292dc5c96378b8ab173b21c13c46
proposal_shape_predictions.jsonl     664c4b0d08ec7d6ce3dc9fea7ec44d38c2369c6b9810da8bf5ba5c30ecc14961
proposal_shape_summary.json          f301ffcaa9a7d278649ce46c3f229f02896ecb67df6b137891ee6399bcf81e8d
```

后两项是报告 `input_snapshot/` 中的 canonical 输入快照。旧输出未覆盖，原视频、
tracking、S0 proposal、人工 XML/truth 和 frozen model 均未修改。

第二次 fresh-output B02 S2A 复跑同样得到 `17 ready / 4 unavailable`；两次的
`README.md`、`proposal_shape_predictions.jsonl`、`summary.json` 逐文件 SHA-256
完全一致。再次指向已存在的 v2 output 时 CLI 拒绝并返回：

```text
status=rejected
successful_output_written=false
```

`compileall` 与 S2A CLI `--help` 退出 0。`git diff --check` 无 whitespace error，
仅打印既有 LF/CRLF working-copy warning。针对 23 份本次 M0-CAM 入口/任务书/报告
的相对链接扫描检查 282 个链接，`missing=0`；显式忽略 `docs/README.md` 中 3 个
此前已确认存在的 fall-risk JSON 缺口，没有跨模块修改它们。
