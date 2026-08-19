# M0-CAM-5D-T2 自动 Episode 与轨迹识别闭环

状态：`completed_with_known_limits`

计划：Day 2

上游：[T1 B01+B02 分段器联合优化](M0-CAM-5D-T1-B01B02分段器联合优化.md)

下游：[T3 居家视频与多模态上下文](M0-CAM-5D-T3居家视频与多模态上下文.md)、[T4 真实日报与个人基线](M0-CAM-5D-T4真实日报与个人基线.md)

## 目标

形成一个无需人工修改中间 JSON 的 `tracking/sidecar -> proposal -> episode QC -> 80 点 -> shape prediction` 入口，并为每个 proposal 输出唯一、可聚合的 episode result。

## 必须完成

1. 使用 T1 选定的 segmenter profile。
2. `proposed+uncertain` 在 interval QC ready 后送入轨迹模型；`rejected_by_qc` 保持 unavailable 并跳过。
3. 每个 proposal 恰好有一条 result，包括原 endpoint、status、reason、QC、binary/four-class、duration 和 identity。
4. 增加运行决策 `auto_accepted/uncertain/rejected`，但不伪造人工 accepted boundary。
5. binary 作为主要识别结果；四分类完整保留为诊断。
6. 输出可以直接被真实日报 adapter 消费。

## 模型调整顺序

1. 先用 fixed primary 在 B01+B02 oracle boundary 上复算 camera binary。
2. 再比较 automatic boundary 下的 end-to-end binary 和 pipeline miss。
3. 若主要问题来自 boundary/QC，不重训模型。
4. 若 oracle boundary 下 binary 仍明显不足，依次允许：camera normalization 修正、binary threshold calibration、轻量 binary-head fine-tune、camera-domain adapter。
5. 任何新候选不得覆盖 fixed primary；必须保留 fallback、video-grouped development 复核和 model/config identity。
6. 不为 phone call/searching/cleaning/exercise 分别建立困难样本模型。

## 最低工程门

- B01+B02 所有 proposal 都有 result 或明确 error；
- known-episode `ready+uncertain` prediction coverage 至少 0.85，目标 0.90；
- automatic binary 结果、pipeline miss 和 oracle binary 分开报告；
- EP1A oracle 回归没有未解释退化；
- 模型、阈值、类别或 preprocessing 若改变，必须新建 identity 并保留旧 fallback；
- 端到端运行不需要人工改 JSON。

## 完成证据与已知限制

最终 development 证据为 [W5D-02 v4 pipeline bundle](../../../../reports/mental_health/wandering_camera_episode_pipeline_w5d02_v4/README.md)，对应运行摘要和完整核验见同目录的 `run_summary.json`、`VERIFICATION.md`。该次 fresh、non-overwrite 运行覆盖 B01=36、B02=12，共 48 条视频，生成 129 条 proposal/result，129 个 `record_id` 唯一且无 error；其中 `proposed=15`、`uncertain=90`、`rejected_by_qc=24`，模型实际 forward 95 次、跳过 34 次。

known-episode `ready+uncertain` prediction coverage 为 `70/78=0.897436`，通过最低门 `>=0.85`，但未达到目标 `0.90`；因此 run status 保持 `uncertain`，且 segmenter selection gate 未满足。automatic boundary 条件二分类 support=51、accuracy=`0.941176`、macro-F1=`0.936383`；fresh oracle boundary 二分类 support=77、accuracy=`0.857143`、macro-F1=`0.875629`，四分类仅作 diagnostic（macro-F1=`0.468998`）。自动边界的 known-episode match/pipeline miss 与 oracle pipeline miss 分开记录（分别 22/21 与 5）。

基于 oracle 证据，保留 fixed primary `topowander-m0s-seed20260731-epoch0005` 和 binary threshold `0.5`，不做 camera binary/head 调整；当前主要问题是 automatic boundary/QC，而不是 threshold。所有 automatic boundary 均标记为 `automatic_boundary_not_human_accepted`。这是一份 B01+B02 development evidence，不代表 sealed、跨人/跨机位泛化或临床效果，也不发出 `AlgorithmEvent`；后端、前端和真实日报/个人基线仍由 T3-T5 完成。

## 输出

```text
episode_results.jsonl
run_summary.json
handoff_manifest.partial.json
README.md
VERIFICATION.md
```

## 验证命令

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_episode_proposal_inference.py --help
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode_pipeline.py -q
```
