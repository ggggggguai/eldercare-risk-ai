# M0-CAM B01/B02 cohort freeze

负责人确认日期：2026-08-17

后续说明：本文件记录旧 proposed-only v1 首次运行前的 cohort 状态，原冻结事实
不回写。负责人在查看 v1 结果后明确授权将 B02 复用为 candidate-inclusive S2A v2
的 development/exploratory 数据，因此下文 11 条视频只对历史 v1 保持
`frozen_unexposed_video_holdout` 身份，不能用于宣称 v2 held-out。v2 结果见
[B02 candidate-inclusive 报告](../wandering_camera_b02_candidate_inclusive_development_v1/README.md)。

状态：

```text
identity_binding_status=confirmed_by_owner
annotation_blinding_status=confirmed_blind_by_owner
cohort_assignment_status=frozen_before_unexposed_b02_inference
participant_generalization_available=false
camera_setup_generalization_available=false
independent_session_generalization_available=false
unexposed_video_holdout_available=true
```

## 1. 固定身份绑定

| batch | participant_id | session_id | camera_setup_id | clock_domain_id |
| --- | --- | --- | --- | --- |
| B01 | P-OFFICE-01 | S-20260814 | C6C-OFFICE-01 | CLOCK-OFFICE-01 |
| B02 | P-OFFICE-01 | S-20260816 | C6C-OFFICE-01 | CLOCK-OFFICE-01 |

B01/B02 为同一 participant、同一摄像头机位和同一画面范围。B02 不能证明跨 participant 或跨 setup 泛化。

负责人确认 CVAT 第一遍标注只查看原视频和标注手册，未查看 S0 proposal、S2A prediction、S2B review pack 或冻结模型结果。旧 P01/L01/R01/H01/H04/H05 的 15 条 AI 预标注不进入本 cohort 的正式 EP1B/EP2A 成绩。

## 2. Development cohort

B01 除 frame-check 外的 36 条视频属于 development：

```text
VID-B01-0002_tracking-smoke.mp4
VID-B01-0003_direct-R01.mp4
VID-B01-0004_direct-R02.mp4
VID-B01-0005_direct-R03.mp4
VID-B01-0006_direct-R04.mp4
VID-B01-0007_direct-R05.mp4
VID-B01-0008_direct-R06.mp4
VID-B01-0009_direct-R07.mp4
VID-B01-0010_direct-R08.mp4
VID-B01-0011_direct-R09.mp4
VID-B01-0012_direct-R10.mp4
VID-B01-0013_direct-R11.mp4
VID-B01-0014_direct-R12.mp4
VID-B01-0015_pacing-R01.mp4
VID-B01-0016_pacing-R02.mp4
VID-B01-0017_pacing-R03.mp4
VID-B01-0018_pacing-R04.mp4
VID-B01-0019_lapping-R01.mp4
VID-B01-0020_lapping-R02.mp4
VID-B01-0021_lapping-R03.mp4
VID-B01-0022_random-R01.mp4
VID-B01-0023_random-R02.mp4
VID-B01-0024_random-R03.mp4
VID-B01-0025_phone-call-R01.mp4
VID-B01-0026_phone-call-R02.mp4
VID-B01-0027_search-object-R01.mp4
VID-B01-0028_search-object-R02.mp4
VID-B01-0029_search-object-R03.mp4
VID-B01-0030_cleaning-R01.mp4
VID-B01-0031_carrying-R01.mp4
VID-B01-0032_carrying-R02.mp4
VID-B01-0033_walking-exercise-R01.mp4
VID-B01-0034_natural-office-R01.mp4
VID-B01-0035_occlusion-R01.mp4
VID-B01-0036_out-of-frame-R01.mp4
VID-B01-0037_dim-light-R01.mp4
```

`VID-B01-0001_frame-check.mp4` 固定为 excluded。

`VID-B01-0036_out-of-frame-R01.mp4` track 56 的负责人裁决为：

```text
purpose_context=unknown
purpose_evidence=unknown
observable_pattern=unknown
evaluation_role=uncertain
tracking_issue=out_of_frame
```

裁决理由是出画后目的不可判。不得覆盖原始 `annotations.xml` 或 per-video XML；只在新的受审/导入视图中应用该裁决，也不得把 unknown 改成可评分类别。

## 3. 已暴露 B02 development diagnostic

```text
VID-B02-0008_window-direct-pacing-R01.mp4
```

该视频已在 cohort freeze 前运行 tracking、EP1A、EP1B、S0 和 S1A，并观察到 S0 没有 locomotion candidate。因此它固定为 exposed development diagnostic，不进入 held-out 指标。

## 4. Frozen unexposed-video held-out

以下 11 条 B02 视频在本次 freeze 时尚未运行 M0-CAM prediction，固定为 held-out：

```text
VID-B02-0002_tracking-smoke-R01.mp4
VID-B02-0003_window-direct-R01.mp4
VID-B02-0004_window-direct-R02.mp4
VID-B02-0005_window-pacing-R01.mp4
VID-B02-0006_window-lapping-R01.mp4
VID-B02-0007_window-random-R01.mp4
VID-B02-0009_window-direct-random-R01.mp4
VID-B02-0010_window-direct-lapping-R01.mp4
VID-B02-0011_episode-pacing-R01.mp4
VID-B02-0012_episode-random-R01.mp4
VID-B02-0013_episode-random-R02.mp4
```

`VID-B02-0001_frame-check-R01.mp4` 固定为 excluded。

在 S0/matching policy 冻结前，不得对以上 11 条视频运行 tracking 后的 proposal/shape/evaluation，不得查看模型预测，也不得用其结果修改同一版本。文件名和既有 manual CVAT 标签已知，因此该集合只能称为 `frozen_unexposed_video_holdout`，不是 sealed test。

## 5. 可声明与不可声明

该划分可以检查：

- 同一 participant、同一 setup 下未暴露视频的 frozen-policy 稳定性；
- B01 development 调整后是否在 B02 剩余视频上保持 boundary/shape 行为；
- 是否存在明显的同机位跨录制日期退化。

该划分不能证明：

- 跨 participant 泛化；
- 跨摄像头、跨机位或跨画面范围泛化；
- 完整独立 session 泛化，因为 B02-0008 已作为 development diagnostic 暴露；
- sealed/official test 性能、camera 95%、alert/FAR 或临床效果。

后续仍需要新的 participant 或至少全新、未暴露的完整 session 才能关闭跨分组评价门。
