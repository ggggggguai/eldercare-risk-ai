# M0-CAM-5D 快速交付总任务书

版本：1.1

更新时间：2026-08-18

状态：`completed_core_awaiting_home_smoke`

本任务书是当前徘徊摄像头算法闭环的唯一执行入口。历史 EP1B、EP2A、MVP、PORTABLE 任务书继续保留为实现与证据追溯，但其中“等待独立 participant/setup”“B01/B02 不得继续调参”“C4 sealed 先于日级闭环”等限制不再作为当前门禁。

## 1. 五日目标

在五个工作日内交付一个可在当前机器运行、能处理真实摄像头视频、并可向后端交接结构化结果的算法 MVP：

```text
camera MP4
  -> person detection/tracking
  -> automatic episode segmentation
  -> episode QC + 80-point trajectory preparation
  -> trajectory shape classification
  -> optional multimodal purpose/context review
  -> session evidence
  -> daily aggregation
  -> rolling personal baseline
  -> backend-facing JSON/JSONL handoff
```

本轮不实现后端、前端、账号、权限、设备管理、消息推送或业务处置。算法交付到“稳定文件契约、可重复运行命令、明确失败状态和示例输出”即满足对接准备。

## 2. 最新负责人决策

以下决策覆盖此前相冲突的研究门禁：

1. B01 和 B02 均为可反复使用的 camera development 数据，可用于分段器、送模策略、camera 预处理、binary 识别和必要的轻量模型调整。
2. 不再保留 B02 作为当前 holdout。旧 B02 frozen-video 结果只作为历史快照，不限制新版本开发。
3. 下一段居家视频用于补充真实场景 truth-free smoke 和问题定位；未标注时也可直接运行，尚未提供视频或标签时不阻塞 context、日报、基线和 handoff 的工程实现。
4. 全部现有和后续提供的视频、画面、标注及内部研发使用权益默认已经完整授权；不再以 consent、receipt 或隐私审批阻塞当前开发。
5. 仍保留匿名 `person_id/session_id/setup_id/source_video_id`、原视频不进 Git、track ID 不等于 person ID等工程约束。
6. 困难样本允许存在，不为每一种打电话、找东西、清洁或锻炼行为单独训练轨迹模型。
7. 轨迹模型只判断 `direct/pacing/lapping/random` 与 shape-binary；目的和上下文由 episode 截图/联系图的多模态模型补充，失败时输出 `unknown`，不得覆盖轨迹类别或人工标签。
8. 当前优先级是闭环可运行和真实视频可识别；C4、sealed、跨机 PORTABLE 和长期统计研究均为后续增强，不是五日交付门。
9. W5D-03 分成 input-independent core 和居家补充 smoke。core 以 B01+B02 真实开发视频及确定性 fixture 验收；居家素材到位后只追加 fresh evidence，不回滚或暂停 W5D-04/W5D-05。

## 3. 当前可复用底座

已经实现并应直接复用：

- YOLO/ByteTrack tracking 与 camera sidecar；
- Camera QC、technical hard break、0.5 秒 bucket 和 bbox-height compensation；
- EP1A accepted/manual boundary 到 80 点轨迹和冻结模型的推理路径；
- EP1B oracle-boundary evaluator；
- EP2A-S0 automatic proposal state machine；
- EP2A-S1A boundary matching/evaluator；
- EP2A-S2A proposal-to-shape bridge；
- EP2A-S2B review pack；
- TopoWander-MPT fixed primary candidate；
- synthetic session evidence、daily summary、baseline profile 和 deviation preview 的计算/校验原语。

现有 daily/baseline producer 仍带 synthetic scope，不能直接把它们写成真实闭环已完成。任务 `W5D-04` 必须新增或扩展一个很薄的 real-development adapter，显式绑定真实 `person_id/session/timezone/presence` 后再复用现有统计和校验逻辑。

## 4. 当前性能起点

历史结果只作为开发起点：

| 项目 | B01 | B02 | 解释 |
|---|---:|---:|---|
| oracle-boundary binary macro-F1 | 0.817375 | 1.000000 | 正确边界下的轨迹分类 |
| automatic boundary F1 | 0.598131 | 0.971429 | 现行分段器差异较大，优先优化 |
| automatic shape ready coverage | 历史较低 | 17/18 | candidate-inclusive v2 已解决 B02 送模覆盖 |

公开 WP 分数继续证明轨迹模型底座有效，但不替代 camera development。五日任务只用 development/MVP 口径描述结果。

截至 2026-08-18，W5D-00 至 W5D-05 已完成。W5D-02 v4 覆盖 B01+B02 共 48 条视频、129 条 result，known-episode prediction coverage 为 `0.897436`，通过 `>=0.85` 最低门但未达 `0.90` 目标，整体状态保留 `uncertain`。W5D-03A 已为 82/82 个 eligible episode 生成 context row 和 246/246 个起中末时点；fake/disabled 只验证工程契约。W5D-04 将 48 个显式 binding 汇总为两个真实 development day/profile/deviation，均保持 `warming_up`；15 日 deterministic replay 验证 3/7/14 日 readiness 和最新 14 日窗口，不冒充真实长期观察。W5D-05 已生成严格 8 文件 handoff，完成完整 cached-tracking E2E、public loader、跨阶段 identity/hash/count/duration 校验和 provider-disabled 非阻塞复放。当前没有在途五日代码任务；居家视频到位后补 W5D-03B truth-free smoke。

## 5. 连续任务队列

| 顺序 | 任务 | 计划时间 | 依赖 | 完成后解锁 |
|---|---|---|---|---|
| `W5D-00` | [交付契约与运行基线](M0-CAM-5D-T0交付契约与运行基线.md) | Day 1 上午 | 无 | T1 |
| `W5D-01` | [B01+B02 分段器联合优化](M0-CAM-5D-T1-B01B02分段器联合优化.md) | Day 1 | T0 | T2 |
| `W5D-02` | [自动 episode 与轨迹识别闭环](M0-CAM-5D-T2自动Episode与轨迹识别闭环.md) | Day 2 | T1 | T3、T4 |
| `W5D-03A` | [多模态上下文 input-independent core](M0-CAM-5D-T3居家视频与多模态上下文.md) | Day 3 | T2；B01+B02/fixture | T4、T5 |
| `W5D-04` | [真实日报与个人基线](M0-CAM-5D-T4真实日报与个人基线.md) | Day 4 | T2；顺序上接 T3A，不依赖居家输入 | T5 |
| `W5D-05` | [算法对接包与整体验收](M0-CAM-5D-T5算法对接包与整体验收.md) | Day 5 | T3A core、T4 | 后端对接准备 |
| `W5D-03B` | [居家视频补充 smoke](M0-CAM-5D-T3居家视频与多模态上下文.md) | 素材到位后 | 居家 MP4；人工标签可选 | 补充真实场景证据，不阻塞上述任务 |

W5D-00 至 W5D-05 均已完成。W5D-03B 是外部素材到位后的条件任务，当前为 `awaiting_input`，不是 `in_progress` 或算法失败。

居家输入缺席时，manifest/run summary 固定记录 `home_input_status=awaiting_input`、`home_smoke_status=not_run_input_unavailable` 和实际 `validation_scope`。这属于外部输入待补，不是算法 run error，也不是 W5D-03A、W5D-04 或 W5D-05 的失败状态。

## 6. 总体技术约束

### 6.1 分段器

- 优先提高 locomotion episode recall；多余候选可由 QC、轨迹模型和上下文层继续过滤。
- 保留 0.5 秒 bucket 作为稳定状态机主尺度。
- 允许在粗边界附近使用原始 tracking timestamp 或 0.25 秒局部 bucket 做边界细化。
- 可调整 movement opening、位移阈值、stationary threshold、dwell、gap、merge/split 和边界 refinement。
- technical hard break 继续硬隔离，不能为提高指标跨 ID switch、长 gap、stream epoch 或 camera movement 合并。
- 所有参数实验使用 fresh output，记录配置、输入、指标和失败案例；不覆盖旧报告或历史输出。

### 6.2 轨迹模型

- camera 主目标是 shape-binary：`direct_or_non_wandering` 对 `wandering_like`。
- 四分类继续输出，用于解释和 subtype 诊断，不作为五日闭环的唯一阻断门。
- 首先使用固定 candidate 做 oracle/automatic 对照；不要先为困难行为建立新模型。
- 如果正确 oracle boundary 下 B01+B02 binary 仍明显不足，可依次尝试 camera 归一化、阈值校准、轻量 binary-head fine-tune 或 camera-domain adapter。
- 任何新模型必须保留原 fixed primary 作为 fallback，使用 video-grouped development 复核，记录 model/config identity，不覆盖原候选。
- 不根据多模态模型结果反向训练或修改 shape truth。

### 6.3 自动 episode 状态

当前历史 `proposed/uncertain/rejected_by_qc` 必须可追溯。五日闭环允许新增运行决策层：

```text
auto_accepted: boundary/QC/shape 条件满足，可进入高置信日报统计
uncertain:     仍运行 shape，进入候选统计和人工/多模态复核
rejected:      QC 不可用或技术硬断，不调用 shape
```

`uncertain` 不能伪装成已人工接受边界；日报必须分开报告 high-confidence 与 uncertain candidate。

### 6.4 多模态上下文

- 只对 `wandering_like` 或需要复核的候选触发。
- 默认抽取 episode 开始、中间、结束三帧，建议同时提供人物裁剪和完整场景联系图。
- 最小标签：`phone_call/searching/cleaning/exercise/social/other/unknown`。
- 输出 provider/model、frame refs/hash、label、confidence、rationale、status。
- API key 缺失、超时、网络错误或模型拒答统一降级为 `unknown/unavailable`，轨迹链和日报继续运行。
- 截图和原视频保存在 Git 外受控输出目录；本轮默认已获授权，不再重复审批。

### 6.5 日报与个人基线

每日最少输出：

- presence duration、tracking/QC coverage；
- episode count/duration；
- direct/pacing/lapping/random 与 wandering-like count/duration；
- high-confidence 与 uncertain candidate 分开计数；
- 次数/小时、candidate duration/小时、night ratio；
- context label 计数；
- unavailable/error/quality flags；
- baseline readiness 和偏离量。

基线状态固定为：

```text
1-2 usable days: warming_up
3-6 usable days: initial_ready
>=7 usable days: stable_ready
rolling window: latest 14 usable days
```

五日内可以证明真实数据聚合器和 readiness 状态可运行，但单段居家视频不能被描述为已经形成稳定七日个人基线。测试可使用多日 fixture 或可控 replay 验证 3/7/14 日规则。

## 7. 算法侧交付契约

最终交付至少包含：

```text
handoff_manifest.json
episode_results.jsonl
context_reviews.jsonl
daily_reports.jsonl
baseline_profiles.jsonl
baseline_deviations.jsonl
run_summary.json
README.md
```

每个可对接记录至少携带：

```text
schema_version
module=mental_health
person_id
session_id
source_video_id
start_time/end_time or local_date
status
quality_flags
model/config/policy identity
source episode/result references
```

后端可以根据这些文件或同构对象实现持久化/API；本轮算法侧不实现 HTTP 服务或 UI。

## 8. 五日总完成门

必须同时满足：

1. B01+B02 可以从既有 tracking/sidecar 无人工修改中间 JSON 地重复运行。
2. 分段器达到 T1 的最低工程门，或按 T1 规定选择 recall-first fallback 并显式记录未达项。
3. 每条 proposal 都有唯一 episode result 或明确 rejected/error 行。
4. B01+B02 能完成 context trigger -> 三帧/引用 -> context row；居家视频缺席时状态显式为 pending，到位后可用同一入口完成 truth-free smoke。
5. 真实 person/session/timezone binding 能生成日报和 baseline readiness。
6. 一条命令或一个清楚的顺序命令可从 MP4/已有 tracking 生成完整 handoff bundle；无居家输入时用 B01+B02 完成 E2E。
7. 所有输出 fresh/non-overwrite；失败可定位；模型、配置和数据身份可追溯。
8. 聚焦测试与完整 wandering 回归通过；居家视频未到位时记录具体输入缺口，不将补充 smoke 伪造为已运行，也不阻塞核心验收。
9. 文档、命令和示例输出一致，不宣称临床诊断、跨人泛化或后端/前端完成。

## 9. 环境与验证

所有 Python 和 pytest 必须使用项目环境：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_episode_boundary.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_episode_proposal_inference.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_daily_summary.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py -q
```

editable location 必须指向当前仓库。不得使用裸 `python`、裸 `pytest` 或临时 `PYTHONPATH` 掩盖环境问题。

## 10. 历史与后续

- EP1B/EP2A 历史报告和旧 B02 holdout 快照保留，不覆盖、不删除。
- C4 sealed、跨 participant/setup 的统计评价、真实老人/长期居家研究属于后续增强，不阻塞本轮 MVP。
- 居家 MP4 truth-free smoke 是素材到位后的必补证据，但不是五日代码闭环和算法对接包的前置门；没有人工标签时不计算性能指标。
- PORTABLE/F1 只在跨机部署或正式发布前关闭。
- 后端和前端由其他团队在算法交付包之后接入。
- 当前进度只看 `docs/tasks/README.md`；技术路线看 `徘徊识别技术文档2.md`；单任务执行看本任务书链接的 T0-T5。
