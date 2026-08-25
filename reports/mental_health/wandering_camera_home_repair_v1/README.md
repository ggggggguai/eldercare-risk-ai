# 居家徘徊分段与分类修复报告

日期：2026-08-22

状态：`development_repair_completed`

本报告记录负责人点名问题的算法修复及 70 视频/135 人工时间段复评。人工标注只提供时间段与行为类别；`evaluation_role` 也只作行为/评价分组语义。CVAT rectangle 和 `evaluation_role` 均未作为检测或跟踪真值。

## 修复内容

1. 新增 `home-recall-confidence070-v1`：只有 `track_confidence >= 0.70` 的 bbox-bottom 观测可以形成正式轨迹、分段、QC 和模型输入；低置信点只保留诊断。
2. 将旧 5 bucket/0.25 body-height 起步改为 3 bucket/0.06，加入 0.015 静止步长与 2 秒滞回；短停不立即碎裂，慢走/短直行可启动。
3. proposal 摘要将可信门槛和运动证据传到分类桥；不再用旧 0.25 motion extent 对已确认运动做第二次无解释拒绝。
4. `boundary_status` 与 `prediction_status` 分开；`uncertain` proposal 仍可得到 `ready` 分类。同一高置信 Track 的单纯位置跳变保留 `suspected_id_switch` reason，但不单独强制 boundary uncertain。
5. 新增 camera geometry logistic 头与层级输出：父 `behavior_episode` 判 pacing/lapping/random；端点短停形成 `motion_bout`。direct 子行程共识可纠正取物往返；父段 pacing 概率达到 0.70 时禁止子段覆盖。最终同时输出父 proposal 结果和 resolved episode 结果。

## 70 视频 development 结果

| 视图 | Precision | Recall | F1 | 说明 |
|---|---:|---:|---:|---|
| 修复前 automatic parent | 0.566038 | 0.666667 | 0.612245 | `closing-s008-d04` 基线 |
| 修复后 behavior parent | 0.802920 | 0.814815 | 0.808824 | 137 candidates，110/135 一对一匹配，median tIoU 0.857143 |
| 修复后 resolved hierarchy | 0.504098 | 0.938931 | 0.656000 | 244 candidates，123/131 已知行为 truth 匹配；召回视图，不与 parent precision 混报 |

最终 `all70_v5` 的 137 个父 proposal 全部 `prediction_status=ready`，`unavailable=0`；resolved 输出 244 段，其中 137 段为 motion bout。0.70 门槛下可信观测覆盖为 48666/58296=`0.834809`，135 条人工时间段中 131 条达到 0.80 时序覆盖，4 条明确保留低覆盖状态。

geometry 头使用 oracle 时间段和事后匹配的自动段共 239 个 development 样本，按 source video 做 5 折分组交叉验证：macro-F1=`0.757405`，direct/pacing/lapping/random recall=`0.942105/1.000000/0.625000/0.666667`。

resolved 全链把未匹配 truth 计为 pipeline miss 后：accuracy=`0.885496`，macro-F1=`0.762286`，direct/pacing/lapping/random recall=`0.933962/0.888889/0.875000/0.250000`。pacing 和 direct 已达到本轮目标；random 仍是主要剩余分类短板。

## 点名视频核对

| 视频 | 修复后结果 | 状态 |
|---|---|---|
| `wand_R01_base_2` | 0.5–121.4 秒父段，`random/ready` | closed |
| `wand_P01_hall_2` | 单一连续父段，`pacing/ready` | closed |
| `wand_P01_hall_1` | 单一连续父段，`pacing/ready` | closed |
| `wand_P01_dining_1` | 0–60.4 秒父段，base random 经 geometry 改为 `pacing/ready` | closed |
| `wand_D01_dining_1_M1` | 4 个父段全部 `direct/ready`，boundary proposed | closed |
| `wand_D01_dining_1_S1` | 从无 proposal 改为 0.5–7.0 秒 `direct/ready` | closed |
| `wand_H01_base_2` | base lapping 经 13 个 direct motion bout 修正为 direct | closed |
| `wand_H01_dining_1` | 6 个父段全部 `direct/ready`，resolved 为 8 个 direct | closed |
| `wand_H02_base_1` | 7 段全部 `direct/ready/proposed`；保留 `suspected_id_switch` 技术 reason | closed/non-regression |
| `wand_H02_base_1_exercise` | 3 个父段全部送模；resolved 13 段，12 direct、1 pacing | closed，仍可人工查看末段 |
| `wand_H02_base_1_sweep` | 长轨迹送模，父段 direct/ready；人工 shape 为 unknown，不用于效果结论 | closed for forwarding |
| `wand_H02_base_2` | `<0.70` 假 Track 不成正式段；5 个高置信父段全部 ready，resolved 11 direct | closed，保留人工视频复核 |
| `wand_H02_dining_1_exercise` | 2.5–68.27 秒 `pacing/ready` | closed for shape；evaluation role 仍待人工裁决 |
| `wand_H02_dining_2` | 5 个父段全部 `direct/ready` | closed |

## 证据边界与剩余问题

- 所有指标来自同一 participant/home setup 的 development 调参与复评，不是 independent test、跨 participant/setup 泛化、检测/跟踪准确率或临床证据。
- behavior parent 是精度/F1 主视图；resolved hierarchy 是问题优先的召回视图。下游必须按 `resolution_level` 选择，不能把两层当作同一互斥候选集。
- random support 只有 8，resolved 全链 recall 为 0.25，尚未达到可用水平。
- `wand_H02_dining_1_exercise` / CVAT track 100 的 `evaluation_role` 仍由负责人裁决；算法没有修改它，也未用预测反向修改任何人工标签。

主要产物：

- 最终运行：`../wandering_camera_home_repair_all70_v5/`
- parent boundary 评价：`../wandering_camera_home_repair_boundary_eval_v2/`
- resolved/classification 评价：`../wandering_camera_home_repair_classification_eval_v5.json`
- geometry 头：`../wandering_camera_home_geometry_head_v1/`
- 分段参数搜索：`../wandering_camera_home_boundary_tuning_v1.json`
