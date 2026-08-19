# M0-CAM-EP1B oracle-boundary batch evaluation

日期：2026-08-15；功能复审更新：2026-08-16

状态：

```text
m0cam_ep1b_implementation_status=completed
m0cam_ep1b_human_truth_status=pending
m0cam_ep1b_evaluation_status=pending
m0cam_ep1b_data_status=pending
m0cam_ep1b_status=not_completed
```

证据范围：`ai_preannotation_provisional_diagnostic_pending_human_review`

成绩名称：**oracle-boundary shape classification**

## 实现结果

EP1B 已形成轻量的多 bundle evaluator：

- 每个 batch row 显式绑定一个 EP1A prediction bundle、truth-view JSONL 和匿名 participant/session/setup；正式评分要求该 truth view 来自独立人工复核；
- 只按 `episode_id + source_video_id + track + start + end` 做严格一对一连接；`episode_id` 重复、同一物理 episode 换 ID 重复计数、同 source/track 区间重叠、缺项、错视频、错 track 和区间漂移都在写输出前拒绝；
- 全部 shape-eligible truth 是主分母；`unavailable/boundary_uncertain/inference_error/abstention` 作为 `pipeline_miss` 影响 recall/F1；
- 另报 `ready_only_conditional`，不把它冒充主成绩；
- 以 shape-binary 作为 camera 主要形态结果，四分类作为 subtype diagnostic；同时报告 QC coverage、15/40 秒时长分层、participant/session/setup support、purposeful-hard-negative 诊断和逐 episode failure。shape-binary 直接使用冻结 binary head 的 `p(wandering_like) >= 0.5` 决策，不从 four-class argmax 反推；
- `QC coverage` 固定为 `qc_status=ready` 的技术 QC 通过率，并另报 prediction-ready coverage；模型 forward 的 `inference_error/abstention` 仍是主指标 pipeline miss，但不再误算成 QC 未通过；
- 每个分组另报数量、逐类 eligible support、QC/prediction-ready coverage 与 runtime status；同一 source 的 session/setup、同一 source/track 的 participant 映射跨 bundle 必须一致；failure 行携带 participant/session/setup、purpose、原 truth/QC 和时长上下文；
- EP1A summary 必须显式保持 `models_retrained=false`、`legacy_40_second_diagnostic_included=false`；ready prediction 的 binary/four-class 概率必须有限、归一化，分别满足冻结 0.5 threshold/fixed-order argmax，并保持冻结层级关系 `P(direct)=P(binary direct)`。身份、概率或 enum 输入错误由 CLI 以稳定 JSON rejection 返回，退出非零且不生成成功指标目录；
- shape 与 purpose 分离。P/L/R 的 purpose unknown 和 `annotation_status/evaluation_role=uncertain` 没有排除清楚的 shape；purposeful pacing/lapping 也没有改写为 direct；
- probability 保持 `calibrated=false`，alert metrics 保持 unavailable。

两个 EP1A P1 同步关闭：CVAT import 与 episode inference builder 都有真实调用及 non-overwrite 回归；低置信度边缘 observation 不再虚假满足 raw detection、track split 或 boundary-gap QC，输出增加 `accepted_observation_count`。

## AI 预标注与执行顺序

现有 P01/L01/R01/H01-H05 共 19 条视频先生成 20 帧 contact sheet 和 truth-free target-track 轨迹图。执行 AI 随后为 P01/L01/R01/H01/H04/H05 生成了 15 条 whole-clip 标签；这些标签是在模型 shape forward 前写入，因而没有根据预测回改，但它们仍然是 AI 预标注，不是独立人工 truth。目录名和拍摄脚本没有直接作为标签；H02/H03 的混合活动未进入诊断。

当前 15 条 AI 预标注为：

| 来源 | AI preannotation shape | episode | purpose | 处理 |
| --- | --- | ---: | --- | --- |
| P01 | pacing | 4 | unknown | 全部进入 shape-only；P01-02 保留真实 fragmented track |
| L01 | lapping | 3 | unknown | 全部进入 shape-only |
| R01 | random | 3 | unknown | 全部进入 shape-only |
| H01 | lapping | 2 | purposeful phone-call walking | purposeful hard negative |
| H04 | pacing | 2 | purposeful carrying | purposeful hard negative |
| H05 | pacing | 1 | purposeful exercise | purposeful hard negative |

S00 的三个既有 CVAT direct truth 和 EP1A prediction bundle 一并进入 batch。D01 只有 engineering prediction、没有独立 truth 文件，因此没有反向生成 direct 标签，也没有进入指标。

本机文件时间记录显示 15 个 AI 预标注文件在 `14:44:33-14:45:39 UTC` 写入，首个 prediction summary 在 `14:47:31 UTC` 写入。15 个新 inference summary 与 S00 summary 都声明 `truth_labels_consumed_by_inference=false`、`automatic_boundary_inference=false`。这证明推理未消费标签，但不能把 AI 预标注升级为人工 truth。prediction 与 label view 只由 evaluator 在推理完成后连接。

人工复核必须不看模型预测，按完整视频重新确认 boundary、shape、purpose 和可判性；现有 XML 与预标注文件保持不变，在新 reviewed truth view 中记录确认、修改或排除。逐条队列见 [HUMAN_TRUTH_REVIEW.md](HUMAN_TRUTH_REVIEW.md)，原视频、contact sheet、truth-free 轨迹和 tracking 文件的一对一画廊见 [HUMAN_TRUTH_REVIEW_GALLERY.md](HUMAN_TRUTH_REVIEW_GALLERY.md)。

## AI 预标注 provisional diagnostic

按当前预标注视图，支持为 `direct/pacing/lapping/random = 3/7/5/3`，共 18 条 evaluator-eligible 记录。17 条 ready，P01-02 因 `boundary_tracking_gap` 为 unavailable；该条仍在 all-eligible pacing support 和 pipeline-miss 分母中。以下数值只用于验证 evaluator 语义和观察 camera subtype 坍缩，不能作为模型真实性能或 EP1B 完成证据。

| 指标 | all shape-eligible provisional | ready-only conditional |
| --- | ---: | ---: |
| support | 18 | 17 |
| four-class accuracy | 0.333333 | 0.352941 |
| four-class macro-F1 | 0.260417 | 0.260417 |
| shape-binary accuracy | 0.777778 | 0.823529 |
| shape-binary macro-F1 | 0.756410 | 0.773333 |
| pipeline miss | 1 | 0 |

all-eligible 四分类 subtype diagnostic：

| truth class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| direct | 0.500000 | 1.000000 | 0.666667 | 3 |
| pacing | 0.000000 | 0.000000 | 0.000000 | 7 |
| lapping | 0.272727 | 0.600000 | 0.375000 | 5 |
| random | 0.000000 | 0.000000 | 0.000000 | 3 |

all-eligible shape-binary 主要形态诊断：

| truth class | precision | recall | F1 | support |
| --- | ---: | ---: | ---: | ---: |
| direct_or_non_wandering | 0.500000 | 1.000000 | 0.666667 | 3 |
| wandering_like | 1.000000 | 0.733333 | 0.846154 | 15 |

## Coverage 与分层

overall QC-pass coverage 为 `17/18 = 0.944444`，prediction-ready coverage 也是 `17/18`；两者在本批相同，是因为唯一 non-ready P01-02 同时未通过 boundary QC，并不表示两个口径可以合并。逐类 QC coverage 为 direct `3/3`、pacing `6/7`、lapping `5/5`、random `3/3`。

| duration band | support | ready | coverage | four-class macro-F1 |
| --- | ---: | ---: | ---: | --- |
| short, `[0,15)` | 1 | 1 | 1.000000 | not_computable |
| medium, `[15,40)` | 10 | 9 | 0.900000 | not_computable |
| long, `[40,+inf)` | 7 | 7 | 1.000000 | not_computable |

当前索引共有 2 个匿名 participant group、7 个 session group 和 1 个 camera setup。15 个 AI 预标注 episode 来自同一 participant/setup；S00 作为独立 development group。该分组只描述输入结构，不能证明跨人或跨机位泛化。

purposeful-hard-negative AI 预标注共 5 条，shape 为 pacing 3、lapping 2；four-class prediction 分布为 direct 2、lapping 3，独立 binary head 为 direct/non-wandering 2、wandering-like 3。5 条的原 `evaluation_role` 均一致，`role_mismatch_count=0`。evaluator 以 purposeful + P/L/R 的独立语义派生诊断资格，因此未来 role 错填不会静默漏掉，而会单列 mismatch；它不改写原 label view。这里 `alert_metrics_available=false`：purposeful 只用于诊断，不能从 shape prediction 推导告警误报。

## 失败清单

`failures.jsonl` 有 12 条，并直接携带匿名分组、purpose/evaluation role、annotation、tracking/QC 与 duration context：

- P01-02：1 条 `pipeline_miss`，原因为 `boundary_tracking_gap`；
- pacing：6 个 ready 中 5 个预测 lapping、1 个预测 direct，另有 1 个 pipeline miss；
- random：3 个全部预测 lapping；
- purposeful H01 lapping：2 个均预测 direct；
- 3 个 S00 direct 与 3 个 core L01 lapping 正确。

按指标头拆分，相对 AI 预标注有 11 条 ready four-class mismatch，其中 3 条同时是 binary-head mismatch，另有 1 条 pipeline miss；当前 17 条 ready 上 binary head 与“four-class 是否为 direct”的派生结果恰好 0 分歧，所以历史 binary 数值未变化。该 0 分歧只是本批观察，不再作为 evaluator 的计算捷径。

失败只是一轮基于 AI 预标注的 camera-domain 诊断，不授权调 threshold、改标签或重训。现有图像平面轨迹显示 P01 的往返路线形成宽回环，R01 在狭小空间也大量复用同一区域，而 L01 是更清楚的重复闭环；这支持“采集动作与透视使 subtype 坍缩”的假设，但必须先由人工 truth 和更多 setup 数据验证。binary 结果可用于判断 wandering-like shape 是否保留，四分类只用于定位 subtype 问题。

## 可复跑位置

Git 忽略的本机输入、预测和 machine-readable evaluation 位于：

```text
tmp/m0cam_ep1b_visual_review_20260815_v1/
tmp/m0cam_ep1b_inputs_20260815_v1/
tmp/m0cam_ep1b_pilot_20260815_v1/
```

2026-08-16 完成独立 binary-head、层级概率语义、跨 bundle overlap/group identity 和稳定 enum 输入拒绝后的当前代码复跑证据位于 `tmp/m0cam_ep1b_pilot_20260815_v1/evaluation_current_code_20260816_03/`。它的六个输出文件与此前 `evaluation_final_audited_20260816_01/` 完全一致，说明加固只拒绝损坏、重复、重叠或矛盾输入，没有改变当前预标注支持、主/条件 diagnostic 或 12 条 mismatch/pipeline 记录。

批量 evaluator 的标准入口：

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_episode_evaluation.py \
  --batch-index tmp/m0cam_ep1b_pilot_20260815_v1/batch_index.jsonl \
  --output-dir tmp/m0cam_ep1b_pilot_20260815_v1/evaluation_replay_20260816_04
```

每次复跑都必须把 `--output-dir` 换成一个尚不存在的新目录（例如递增末尾序号）；evaluator 会拒绝已有目录，且不得删除或覆盖既有证据。输出包含 `summary.json`、`metrics.json`、`confusion.json`、`episode_results.jsonl`、`failures.jsonl` 和 `README.md`。详细测试见 [VERIFICATION.md](VERIFICATION.md)。

## 未完成门与下一缺口

EP1B 不能标 completed：15 条 P/L/R/H 还没有独立人工 truth，因而当前没有合法的真实 D/P/L/R camera pilot。用户已明确把 camera 四分类降为 subtype diagnostic，所以每类约 10 条保留为后续扩样目标，不再单独作为 binary 工具线的硬完成门；但人工复核后仍必须实际覆盖 D/P/L/R，否则继续 data pending。

EP1B 的下一缺口仍是 human truth：先按 [人工复核队列](HUMAN_TRUTH_REVIEW.md)逐条复核当前 15 条预标注，在新的 reviewed truth view 中确认、修改或排除；若复核后缺某个 subtype，再补拍清楚样本。新拍 pacing 优先横跨画面并形成约 3 次反转，lapping 约 2 圈，random 访问至少 3 个分离区域；这些只是采集提示，不是标签或模型阈值。补样前不重训、不调 0.5 阈值、不改冻结候选，也不覆盖现有 truth/XML。

EP2A-S0、S1A、S2A 和 S2B 的后继 implementation-only 工具现均已完成，当前没有未阻塞代码任务。EP1B 下一步仍是按[人工证据交接](../../../docs/modules/mental_health/plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md)形成新的 reviewed truth；proposal、S2A prediction 和 S2B review template 均不能作为 EP1B truth。

本报告不证明 automatic boundary、连续视频端到端效果、camera 95%、告警/FAR、真实老人或临床效果，也没有输出风险或 `AlgorithmEvent`。
