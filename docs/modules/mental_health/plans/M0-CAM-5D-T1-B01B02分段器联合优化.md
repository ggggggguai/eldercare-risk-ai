# M0-CAM-5D-T1 B01+B02 分段器联合优化

状态：`completed`

计划：Day 1

上游：[T0 交付契约与运行基线](M0-CAM-5D-T0交付契约与运行基线.md)

下游：[T2 自动 Episode 与轨迹识别闭环](M0-CAM-5D-T2自动Episode与轨迹识别闭环.md)

## 目标

把 B01+B02 作为一个可反复使用的 camera development 集，优先提高完整 locomotion episode 的候选覆盖率和 boundary recall，同时控制严重 fragmentation、跨技术硬断和不可解释合并。

## 可调整范围

- `movement_start_min_buckets`；
- movement displacement threshold；
- stationary displacement threshold；
- stationary dwell；
- maximum internal gap；
- opening/closing hysteresis；
- 相邻候选 merge/split 规则；
- 粗边界附近基于原 tracking timestamp 或 0.25 秒局部 bucket 的边界 refinement；
- proposal status/acceptance policy，但必须保留 uncertain 和 rejected 的可追溯性。

## 执行步骤

1. 用现有 frozen policy 对 B01、B02 和 pooled cohort 形成基线。
2. 从 false negative、split、merge、low-tIoU、large onset/offset 中提取最主要的 3-5 类失败。
3. 建立小网格，每轮只改变一个参数组；每轮写入 fresh output。
4. 先比较 boundary recall、candidate coverage，再比较 F1、tIoU 和 split/merge。
5. 对最佳 2-3 个配置做完整 B01+B02 回放和失败图审。
6. 选出唯一 development segmenter profile，记录理由和已知限制。
7. 保留旧 v1/v2 报告为历史，不覆盖。

## 最低工程门

| 指标 | 最低目标 | 伸展目标 |
|---|---:|---:|
| pooled boundary recall | 0.75 | 0.80 |
| pooled boundary F1 | 0.72 | 0.78 |
| matched mean tIoU | 0.65 | 0.72 |
| known-episode candidate coverage | 0.85 | 0.90 |

另外必须满足：

- B01/B02 分开报告，不用 pooled 指标掩盖单批崩溃；
- technical hard break 跨越数为 0；
- 每个漏检和严重 split/merge 有失败记录；
- 若指标不能全部达到，选择 recall-first 且 coverage 最高的稳定版本，明确记录未达项，不让分段调参无限阻塞闭环。

## 输出

```text
segmenter_search_manifest.json
candidate_configs/
runs/<config_id>/metrics.json
runs/<config_id>/failures.jsonl
selected_segmenter_profile.yaml
README.md
VERIFICATION.md
```

## 实现与证据

- 实现：`camera_episode_boundary.py` 增加独立 development profile 和基于原 tracking timestamp 的 `0.25` 秒局部边界 refinement；冻结 v1 proposal 路径保持逐字节 golden 回归。
- 编排：`camera_episode_boundary_development.py` 和 `scripts/wandering/tune_camera_episode_boundaries.py` 固定 48 条 index、候选 profile、B01/B02/pooled 三视图、recall-first 排名、失败完整性审计、技术硬断审计和 top-3 全量复放。
- 第一轮：`wandering_camera_segmenter_search_v1` 比较 13 个 baseline/单参数组/组合候选；`closing-s012-d08` 达到 pooled recall/F1/tIoU/coverage `0.705128/0.705128/0.761088/0.961538`。
- 第二轮：`wandering_camera_segmenter_search_v2` 对首轮 stationary-closing 最优区间做 16 候选有界搜索；按预声明顺序选择唯一 fallback `closing-s008-d04`，manifest SHA-256 为 `1a1cc9202cfbb90ec03c568db9708b9b576966a018645a4fa91c2932994c489d`。

最终 development 指标：

| cohort | recall | F1 | matched mean tIoU | known-episode coverage |
|---|---:|---:|---:|---:|
| pooled | 0.730769 | 0.622951 | 0.716007 | 0.935897 |
| B01 | 0.672414 | 0.537931 | 0.704619 | 0.948276 |
| B02 | 0.900000 | 0.947368 | 0.740680 | 0.900000 |

pooled tIoU、coverage 和 technical hard-break crossing `0` 通过；recall `0.730769 < 0.75`、F1 `0.622951 < 0.72` 未达最低目标。第二轮把 pooled/B01 recall 从首轮最优的 `0.705128/0.637931` 提高到 `0.730769/0.672414`，但 candidate support 增至 105，B01 fragmentation/split 增加。按本任务“未全达标时选择 recall-first 稳定 fallback，不无限阻塞闭环”的规则停止扩网格；这些数字只属于 B01+B02 development，不是 sealed、跨人或临床效果。

最终 top-3 `closing-s008-d04`、`closing-s012-d04`、`closing-s014-d04` 均完成 48 视频全量复放，metrics/failures payload 和 proposal tree 一致；全部 primary miss 与严重 split/merge 保留在 `runs/<config_id>/failures.jsonl`。下游 W5D-02 只使用 `reports/mental_health/wandering_camera_segmenter_search_v2/selected_segmenter_profile.yaml`。

## 验证命令

```bash
conda run -n eldercare-ai python scripts/wandering/propose_camera_episode_boundaries.py --help
conda run -n eldercare-ai python scripts/wandering/evaluate_camera_episode_boundaries.py --help
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_boundary_evaluation.py \
  tests/test_wandering_camera_qc.py -q
```

## 停止条件

只有以下情况停止并请求负责人决策：需要改人工 truth/标签定义、跨 technical hard break 合并、覆盖既有证据或修改其他算法模块。普通参数、路径、timeout、局部实现和测试失败由执行 AI 继续排查。

## 完成验证

```text
editable_project_location=/mnt/c/Users/lenovo/Desktop/心理算法
development_record_count=48
candidate_runs=29 (round1=13, round2=16)
selected_candidate_id=closing-s008-d04
selected_manifest_sha256=1a1cc9202cfbb90ec03c568db9708b9b576966a018645a4fa91c2932994c489d
manifest_artifacts=51/51 hash_and_byte_count_verified
finalist_full_replays=3/3 identical
technical_hard_break_crossings=0
focused_boundary_tests=40 passed
second_round_config_tests=5 passed
full_camera_tests=624 passed, 1 expected duplicate-ZIP negative-fixture warning
```
