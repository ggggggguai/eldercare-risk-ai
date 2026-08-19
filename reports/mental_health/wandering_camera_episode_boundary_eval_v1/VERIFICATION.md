# M0-CAM-EP2A-S1A verification

日期：2026-08-16

环境：所有 Python/pytest 均通过 WSL `eldercare-ai`。`elderly-monitoring-algorithms` editable project location 已确认为：

```text
/mnt/c/Users/lenovo/Desktop/心理算法
```

## TDD and focused tests

合成 contract 测试先于实现加入；首次运行因 `camera_episode_boundary_evaluation` 尚不存在而在 collection 阶段失败。初版实现后为 `5 passed, 8 failed`，失败暴露了原始 S0 proposal 错用已绑定 entity 排序键的问题。增加 raw-proposal 稳定排序键并保持 entity scope 排序独立后，初版达到 13 条测试。完成后功能复审又新增 independent-human、跨时长带和 technical hard-break 三条回归，最终窄测为：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_boundary_evaluation.py -q
```

结果：`16 passed in 8.82s`。

覆盖 proposal/summary/human-boundary loader、proposal 与 accepted boundary schema 隔离、实际 independent-human 分支、continuous whole-clip 拒绝、最大基数共享 matcher、跨 scope 隔离、双视图分母、localization signed/absolute error、uncertain/rejected/human-uncertain 可见性、split/merge/subthreshold/fragmentation、technical hard-break 隔离、分组/时长 support、跨时长带 F1 不可计算、空 cohort、重复/重叠/identity drift/非有限数拒绝、原输入字节不变、deterministic output、non-overwrite 和 CLI help。

## Completion audit

审计确认并修复：

```text
duration-band F1:
  precision population = proposal duration band
  recall population    = truth duration band
  cross-band match     -> F1=not_computable

fragmentation:
  camera/track scope + technical_segment_index isolation

human branch:
  authorized independent_human E2E covered
  whole_clip rejected for continuous boundary evaluation
```

## Regressions

S0/S1A/shared matcher：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py \
  tests/test_wandering_camera_development.py -q
```

结果：`63 passed in 12.53s`。

EP1A/EP1B 与相邻 camera 链：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_primary_inference.py -q
```

结果：`168 passed in 27.36s`。

全部 wandering camera 回归：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_*.py -q
```

结果：`602 passed, 1 warning in 158.83s`。唯一 warning 来自既有 portability 测试刻意写入重复 ZIP entry，不属于本次 evaluator。

## Synthetic CLI smoke

先用聚焦测试在全新 Git-ignored 目录生成 synthetic proposal/boundary/sidecar/index，并验证输入字节不变与 non-overwrite：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_boundary_evaluation.py::test_deterministic_non_overwrite_and_source_files_unchanged \
  -q --basetemp=tmp/m0cam_ep2a_s1a_audit_20260816
```

结果：`1 passed in 5.91s`。

随后以正式 CLI 分别写入两个 fresh output：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python \
  scripts/wandering/evaluate_camera_episode_boundaries.py \
  --batch-index tmp/m0cam_ep2a_s1a_audit_20260816/test_deterministic_non_overwri0/batch-index.jsonl \
  --output-dir tmp/m0cam_ep2a_s1a_audit_20260816/test_deterministic_non_overwri0/cli-output-a
```

第二轮只把输出目录改为 `cli-output-b`。两轮 CLI 输出相同计数：

```json
{"failure_count":28,"match_count":5,"proposal_count":10,"ready_truth_count":9,"uncertain_truth_count":1}
```

两轮以下 8 个文件的 SHA-256 全部一致：

```text
summary.json
metrics.json
matches.jsonl
unmatched_truth.jsonl
unmatched_proposals.jsonl
split_merge_diagnostics.jsonl
failures.jsonl
README.md
```

对既有 `cli-output-a` 的第三次同路径 CLI 调用被结构化拒绝：`error_code=output_exists`、`successful_evaluation_written=false`；`summary.json` SHA-256 在拒绝后仍为 `B4BA487984F8F204A6F3AD00D0DE352A9BA1960CC7251CABB6117E27DFF360D3`。

synthetic summary 为 `wandering_m0cam_ep2a_s1a_boundary_evaluator_implemented / synthetic_contract_only`；proposal 状态为 proposed/rejected_by_qc/uncertain=`7/1/2`，human boundary 状态为 ready/boundary_uncertain=`9/1`。`matching_policy_validated=false`、`automatic_boundary_evaluation_completed=false`、shape truth/prediction 均未读取、模型未训练。

该 smoke 未读取 H02/H03，也未给 H02/H03 伪造 truth。synthetic TP/F1/tIoU/error 只验证计算合同，不是 automatic boundary 性能。
