# M0-CAM-EP2A-S2B proposal-shape review pack

日期：2026-08-16

```text
m0cam_ep2a_s2b_implementation_status=completed
m0cam_ep2a_s2b_data_status=implementation_plus_v2_uncertain_ready_compatibility
m0cam_ep2a_human_boundary_status=available_b01_b02
m0cam_ep2a_evaluation_status=post_hoc_development_completed_on_reused_b02
m0cam_ep2a_status=not_completed
```

## 实现

S2B 只增加一条 truth-free review-pack 代码线：

- `wandering_camera_episode_proposal_review_v1.yaml` 固定最小输入/输出契约，不包含模型、阈值或评价参数；
- `camera_episode_proposal_review.py` 只读连接 S0 proposal、S2A proposal-shape result、tracking 与 media identity；
- `build_camera_episode_proposal_review_pack.py` 只暴露 config、batch index 和 fresh output；
- `test_wandering_camera_episode_proposal_review.py` 覆盖一对一连接、无概率状态保留、空白人工列、轨迹图和 non-overwrite。

一个 S2A batch 可以对应多个 S0 视频。builder 先汇总所有 S0 `proposal_id`，再与一个或多个 S2A bundle 做全局一对一连接；缺失、重复、scope/binding/endpoint/status 不一致都会在输出前失败。它不调用冻结模型，也不修改任何输入。

## Bundle

```text
README.md
review_index.csv
human_review_template.csv
per_video_summary.md
per_proposal_trajectory_plots/
  <source_video_id>__<proposal_id>__<identity>.svg
```

`review_index.csv` 一行对应一条 proposal，字段分为：

- batch/video identity：`bundle_id`、participant/session/setup/clock、`source_video_id`、`media_ref`、device/stream/track；
- proposal identity：`proposal_id`、technical segment、原 start/end/duration/status/reason；
- S2A result：`prediction_status`、model skipped、candidate ID，以及 ready 时的 binary/pattern label、class order 和 probabilities；
- review navigation：trajectory point count、plot path、`needs_human_review=true`。

每张 SVG 使用该 proposal 的完整 camera scope、原始 `[start_sec,end_sec_exclusive)` 和 track ID 选择 bbox bottom-point 轨迹。图中包含全部匹配点、绿色起点、红色终点、蓝色方向箭头、时间范围、proposal/prediction status 和 reason codes。文件名与 CSV path 一对一，图内同时写出完整 `source_video_id` 和 `proposal_id`。

`human_review_template.csv` 只复制定位上下文。以下人工列全部为空：

```text
human_boundary_decision
corrected_start_sec
corrected_end_sec
observable_pattern
purpose_context
review_notes
```

模型答案没有写入这些列；该模板不是 truth。

## Synthetic contract

初始两条 synthetic proposal 覆盖一个 ready S2A result 和一个 boundary-uncertain result，生成 2 行 index、2 张 SVG、逐视频摘要、空白模板和 README。另一个两行 fixture 覆盖 `boundary_uncertain/unavailable` 均无概率但不丢行。缺失、重复和错 scope 三种 join 均 fail closed；输出目录已存在时拒绝覆盖。2026-08-17 v2 又增加 uncertain+ready join：review row 显示概率，同时原 `proposal_status=uncertain` 和人工复核要求保持可见。

实际命令与结果：

```bash
conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_proposal_review.py
# 6 passed in 13.41s

conda run -n eldercare-ai python -m pytest -q \
  tests/test_wandering_camera_episode_boundary.py \
  tests/test_wandering_camera_episode_proposal_inference.py \
  tests/test_wandering_camera_episode_inference.py
# 29 passed in 17.70s
```

CLI `--help` 与 `compileall` 均退出 0。editable project location 为当前 WSL 仓库 `/mnt/c/Users/lenovo/Desktop/心理算法`。

## H02/H03 truth-free smoke

使用既有 S0 v2 proposal、S2A run A、原 tracking/sidecar 构建两个 fresh review pack；未修改 status、endpoint 或 reason：

| video | 原区间（秒） | proposal | S2A result | plot points |
| --- | ---: | --- | --- | ---: |
| H02-01 | 8.5–44.733333 | uncertain | boundary_uncertain / skipped（历史 v1） | 543 |
| H02-02 | 11.5–28.0 | uncertain | boundary_uncertain / skipped（历史 v1） | 247 |
| H02-03 | 15.5–22.133333 | uncertain | boundary_uncertain / skipped（历史 v1） | 99 |
| H03-01 | 1.5–50.0 | uncertain | boundary_uncertain / skipped（历史 v1） | 727 |

结果为 4 个 source video、4 行 review index、4 张 SVG、4 行空白 human template，S2A model invocation/skipped 状态为 `0/4`。run A/B 共 8 个文件的集合和逐文件内容一致，人工列预填单元格为 0。

该真实 smoke 只证明 H02/H03 的视频、区间、轨迹、proposal 状态和 S2A skip 结果能被人工定位。四条均为 uncertain，因此没有覆盖真实 proposed forward，更没有验证 boundary 或 shape 性能。

## 证据边界

S2B 没有生成 truth、accepted boundary、XML 或人工决定，没有训练、调参、修改 0.5 threshold，也没有计算 P/R/F1、tIoU、起止误差、漏切、多切、alert、FAR 或临床指标。S2A prediction 只作为只读复核上下文，不能替代独立人工标签。

EP2A 仍为 `not_completed`。下一缺口仍是独立人工 continuous boundary、S1B development 参数冻结和独立 participant/session/setup 评价；EP2B、EP2C、EP3 未启动。

当前 `current_code_task=none (waiting_for_independent_camera_generalization_evidence)`。S2B 已兼容 candidate-inclusive v2 的 uncertain+ready result；模板暴露 proposal/prediction，只能在盲标完成后用于定位和差异复核，不能直接作为 `independent_human` truth。下一证据入口见[EP1B / EP2A 证据交接](../../../docs/modules/mental_health/plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md)。
