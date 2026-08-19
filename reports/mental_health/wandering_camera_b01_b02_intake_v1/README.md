# M0-CAM B01/B02 人工 CVAT 数据接入审计

日期：2026-08-17

状态：

```text
source_data_status=available_external_read_only
human_annotation_status=reviewed_manual_cvat
group_binding_status=confirmed_by_owner
blinding_status=confirmed_blind_by_owner
development_holdout_assignment=frozen_unexposed_video_holdout
m0cam_ep1b_status=completed
m0cam_ep2a_s1b_policy_status=frozen_on_b01_development
m0cam_ep2a_s1b_video_holdout_status=completed_once_on_b02
m0cam_ep2a_status=not_completed
```

证据范围：本报告记录 `human_cvat_intake + cohort_freeze`；后续 B01/B02 评价见 [B01 development / B02 frozen-video holdout](../wandering_camera_b01_development_v1/README.md)。

## 1. 数据位置与保护边界

负责人指出真实视频和人工标注位于：

```text
D:\徘徊数据集\自采数据\B01
D:\徘徊数据集\自采数据\B02
```

本轮只读检查该目录；没有修改视频、项目级 `annotations.xml` 或派生的 per-video XML。所有运行产物写入仓库忽略的全新目录：

```text
tmp/m0cam_b01_b02_intake_20260817_v1/
```

## 2. 只读数据盘点

| batch | video | manual CVAT track | duration |
| --- | ---: | ---: | ---: |
| B01 | 37 | 59 | 33.14 min |
| B02 | 13 | 21 | 11.51 min |
| total | 50 | 80 | 44.65 min |

50 条视频均为 15 FPS、2560 x 1440；实际视频帧数与 CVAT `job.size` 全部一致，共 40,188 帧。80 条 track 均使用 `label=wandering_episode`、`source=manual`，同一 track 内人工属性保持不变。

原始人工 `observable_pattern` 分布：

| batch | direct | pacing | lapping | random | unknown | not_set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B01 | 36 | 9 | 3 | 9 | 1 | 1 |
| B02 | 10 | 3 | 2 | 5 | 0 | 1 |
| total | 46 | 12 | 5 | 14 | 1 | 2 |

这些是 raw annotation counts，不是 EP1B eligible support，也不是模型成绩。

## 3. 现有 importer 兼容性

使用现有 `import_cvat_episode_xml()` 对 50 份 per-video XML 做只读兼容性检查：

```text
directly_importable_video_count=47/50
imported_episode_count=75
boundary_status_ready=70
boundary_status_uncertain=5
direct/pacing/lapping/random=46/12/5/12
```

三条未直接进入 importer 成功路径的视频分别是：

1. `VID-B01-0001_frame-check.mp4`：人工明确为 `not_set + excluded`，只用于取景复核，不应进入可评分 episode。
2. `VID-B02-0001_frame-check-R01.mp4`：同样为 `not_set + excluded`，不应进入可评分 episode。
3. `VID-B01-0036_out-of-frame-R01.mp4`：track 56 为 `observable_pattern=unknown`、`evaluation_role=uncertain`、`tracking_issue=out_of_frame`，原 XML 同时填写 `purpose_context=unknown`、`purpose_evidence=scripted`。负责人随后裁决为 `purpose_context=unknown + purpose_evidence=unknown`；该裁决只应用于新的派生 reviewed XML，原 XML 未修改。

前两条是预期排除，不是标签缺陷。第三条在 intake 当时使 importer 对整条视频 fail closed；负责人裁决后已在派生受审视图解除，未静默修改或部分接纳原 XML。

## 4. 首条真实链路 smoke

本轮选用：

```text
D:\徘徊数据集\自采数据\B02\VID-B02-0008_window-direct-pacing-R01.mp4
```

tracking 完成 900/900 帧，得到单一 `track_id=1`；检测置信度 minimum/mean=`0.6585/0.8659`。本机 CUDA driver 与运行时不兼容，实际使用 CPU 完成；没有改环境或共享依赖。

人工 CVAT 边界为：

```text
direct: [0.000000, 9.400000)
pacing: [28.666667, 60.000000)
```

### 4.1 EP1A / EP1B

EP1A 两条 episode 均为 `ready`：

| human shape | binary head | four-class head |
| --- | --- | --- |
| direct | direct_or_non_wandering | direct |
| pacing | wandering_like | lapping |

单视频 EP1B intake smoke 的 shape eligible 和 QC ready 均为 `2/2`；binary accuracy/macro-F1=`1.0/1.0`，four-class accuracy=`0.5`，four-class macro-F1 因类别不全不可计算。该结果只验证真实人工 boundary/truth 能进入现有 evaluator，并再次观察到 pacing 在 camera subtype head 上坍缩为 lapping；它不是完整 pilot，也不能外推 camera accuracy。

### 4.2 S0 / S1A

S0 对完整视频输出一条覆盖 `[0,60)` 的 `rejected_by_qc` interval：

```text
reason=insufficient_locomotion_evidence
locomotion_proposal_count=0
```

S1A 使用两条人工 ready boundary 后，主视图 candidate support/match=`0/0`，recall=`0.0`，failure count=`6`。两条人工 episode 都只被 rejected-by-QC interval 覆盖。

这是当前未冻结 S0 movement-start policy 在一条真实 B02 视频上的首个失败证据。不得据此单视频调参或改变 proposal 状态；应先声明 development cohort，再在多个代表视频上确认失败模式。

## 5. 负责人确认与 cohort freeze

负责人于 2026-08-17 确认：

1. B01/B02 为同一 participant `P-OFFICE-01`，session 分别为 `S-20260814`、`S-20260816`。
2. 两批均使用 `camera_setup_id=C6C-OFFICE-01`、`clock_domain_id=CLOCK-OFFICE-01`，机位和画面范围未变化。
3. CVAT 第一遍标注盲于 S0/S2A/S2B 和冻结模型结果。
4. B01 可用视频属于 development；已暴露的 B02-0008 固定为 development diagnostic。
5. B02 除 frame-check 与 B02-0008 外的 11 条可用视频固定为 unexposed-video held-out。

逐文件清单和可声明范围见 [COHORT_FREEZE.md](COHORT_FREEZE.md)。由于 B02-0008 已暴露，剩余 B02 只能形成同一 participant/setup 下的视频级 held-out，不证明完整独立 session、跨人或跨机位泛化。

## 6. 后续执行记录

上述 cohort freeze 之后已按顺序完成：

1. track 56 裁决只写入派生 reviewed XML，两条 frame-check 继续 excluded；
2. B01 36 条可用视频完成 importer/tracking/EP1A/EP1B/S0/S1A development；
3. 只用 B01 比较 movement opening 与 dwell，冻结 `movement_start_min_buckets=5`、`stationary_dwell_candidate_seconds=15.0` 和既有 matching；
4. policy freeze 文档存在后，一次性运行 11 条未暴露 B02 视频；
5. EP1B oracle-boundary shape、S1A boundary 和 S2A automatic proposal + shape 分开报告。

完整指标、冻结 ID、失败清单与产物 SHA 见 [B01/B02 评价报告](../wandering_camera_b01_development_v1/README.md)。旧 producer/matching 与 proposed-only v1 结果不得覆盖。2026-08-17 负责人随后明确授权复用已查看的 B02，建立独立 ID 的 candidate-inclusive S2A v2；该新结果固定降格为 post-hoc development/exploratory，见 [v2 报告](../wandering_camera_b02_candidate_inclusive_development_v1/README.md)。

## 7. 当前结论

intake、owner binding、盲标确认、track 56 裁决、cohort freeze、B01 development 与 B02 历史一次性视频级 holdout 均已完成。EP1B 在 manual-CVAT oracle-boundary descriptive pilot 范围为 `completed`；旧 15 条 AI 预标注未进入正式成绩。负责人授权后的 candidate-inclusive S2A v2 在复用 B02 上 ready=`17/18`，关闭了旧 `2/18` 送模缺口，但它不是 holdout。EP2A 仍为 `not_completed`，因为 B01/B02 是同一 participant/setup 且 v2 缺独立验证。下一输入必须优先是新 participant/setup；当前不启动 EP2B、EP2C、EP3、alert、risk 或 `AlgorithmEvent`。
