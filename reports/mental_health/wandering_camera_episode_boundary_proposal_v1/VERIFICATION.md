# M0-CAM-EP2A-S0 verification

日期：2026-08-16

环境：所有 Python/pytest 均通过 WSL `eldercare-ai`；`elderly-monitoring-algorithms` 的 editable project location 为 `/mnt/c/Users/lenovo/Desktop/心理算法`。

## 测试驱动记录

首次添加合成测试后，聚焦命令按预期在 collection 阶段失败：新 `camera_episode_boundary` module 和共享 splitter 接口尚不存在。随后实现 proposal module/config/CLI；为了保持历史 Step8 对 `camera_qc.py` 的冻结源码 SHA，不修改其字节，而是直接复用已有 `_split_observations()` 唯一算法。

最终 boundary 窄测：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py
```

审计前结果为 `14 passed in 8.64s`。功能复审新增“同 bucket 技术 hard break 不得形成重叠区间”和“慢速连续平移不得被高置信关闭”两条回归测试，并修复实现。

最终复验结果：`16 passed in 7.28s`。

覆盖静止到移动、短暂停回开、pacing 反转、lapping 闭环、random 换向、两段移动、long gap、ID switch、height-position discontinuity、低置信度边缘、多 track 隔离、track-end uncertain、静止 rejection、deterministic replay、schema 隔离、CLI help 和 non-overwrite。

EP1A/EP1B 与相邻 camera 回归：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_evaluation.py \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_episode_inference.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_inference.py
```

审计后结果：`180 passed in 58.74s`。

全部 camera 回归：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_*.py
```

审计前结果为 `584 passed, 1 warning in 150.00s`。审计后完整 camera 回归结果见本文件末尾的“最终复验”；旧数字不再作为当前代码的完成证据。

## Truth-free smoke

四个 source 分别以相同 CLI 在全新 `run_a` 与 `run_b` 目录运行：

```bash
/home/lenovo/miniconda3/bin/conda run -n eldercare-ai python \
  scripts/wandering/propose_camera_episode_boundaries.py \
  --tracking-jsonl tmp/m0cam_ep1b_inputs_20260815_v1/tracking/<source>/tracking.jsonl \
  --media-sidecar tmp/m0cam_ep1b_inputs_20260815_v1/tracking/<source>/media_sidecar.json \
  --output-dir tmp/m0cam_ep2a_s0_truth_free_smoke_20260816_v2/<run>/<source-id>
```

`<source>` 为 `h02/01`、`h02/02`、`h02/03`、`h03/01`；`<run>` 为 `run_a`、`run_b`。两轮的四文件 bundle 均逐文件 SHA-256 一致。已有 output 的第三次运行退出码为 1，错误为 `FileExistsError`，原输出保持不变。

smoke 只读取 tracking/sidecar 并输出 proposal/diagnostic/summary。`summary.json` 均声明 `truth_labels_consumed_by_producer=false`、`shape_predictions_consumed_by_producer=false`、`frozen_model_loaded=false`、`automatic_boundary_validated=false`、`stationary_dwell_candidate_validated=false`。

没有运行 boundary evaluator、shape inference、训练、threshold search 或 sealed 数据访问。没有生成 boundary F1/tIoU/onset-offset/漏切/多切/automatic-shape 指标。

修复后的 truth-free smoke 为：H02-01 `[8.500,44.733)`、H02-02 `[11.500,28.000)`、H02-03 `[15.500,22.133)`、H03-01 `[1.500,50.000)`；四条均为 `uncertain`。H03 记录 `stationary_candidate_drift`，不再输出旧版本的两条 `proposed`。两轮共 16 个文件逐字节一致。

## 最终复验

完整 `tests/test_wandering_camera_*.py` 回归在 S0 审计修复后重新运行：`586 passed, 1 warning in 152.64s`。唯一 warning 仍来自既有 PORTABLE 重复 ZIP 条目拒绝测试。editable project location 已重新确认为 `/mnt/c/Users/lenovo/Desktop/心理算法`。

`propose_camera_episode_boundaries.py --help` 与本次 boundary module/CLI/test 的 WSL `eldercare-ai` compileall 均通过。
