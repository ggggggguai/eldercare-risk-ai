# 心理健康风险算法模块

更新时间：2026-08-18

本模块输出行为与睡眠变化的工程特征和独立心理健康风险事件，不输出医学诊断。心理健康与跌倒模块分别评分、验证和输出；本模块只使用 `module=mental_health`。

## 当前入口

- 现行技术路线：[徘徊识别技术文档2](plans/徘徊识别技术文档2.md)
- 当前五日交付：[M0-CAM-5D 快速交付总任务书](plans/M0-CAM-5D快速交付总任务书.md)
- 当前任务状态：[任务表](../../tasks/README.md)
- 拍摄手册：[萤石 C6C 办公室视频拍摄与标注手册](拍摄与标注/萤石C6C办公室视频拍摄与标注手册.md)
- CVAT 教程：[Windows 本地部署 CVAT 徘徊识别标注员教程](拍摄与标注/Windows本地部署CVAT徘徊识别标注员教程.md)
- 算法交接边界：[徘徊模块协作交接与职责边界](徘徊模块协作交接与职责边界.md)

旧 [EP1B/EP2A 独立证据交接](plans/M0-CAM-EP1B-EP2A-S1B人工证据交接.md)已被五日路线取代，只保留历史数据和指标追溯。历史实验事实以 `reports/mental_health/` 为准，不因当前路线调整而改写。

## 1. 当前目标

五日内完成算法侧闭环：

```text
camera MP4
  -> detection/tracking
  -> automatic episode segmentation
  -> trajectory shape classification
  -> optional multimodal purpose/context
  -> daily report
  -> rolling personal baseline
  -> backend-facing JSON/JSONL
```

本轮不实现后端、前端、账号、权限、设备管理、推送或工单。最终交付稳定输出契约、运行命令、模型/配置身份、质量状态和示例 bundle，供后端团队后续接入。

全部现有和后续提供的视频、截图、标注和内部研发使用默认已经完整授权；不以授权、隐私或 receipt 审批阻塞开发。仍要求匿名 ID、原视频不进 Git、输出不覆盖和显式 person/session/video binding。

## 2. 通用心理健康能力

当前代码已经支持：

- 行为、睡眠和自评数据适配；
- 日级聚合与缺失模态处理；
- 个人基线与持续异常判断；
- 独立心理健康风险评分；
- 离线日级 CLI；
- `AlgorithmEvent(module=mental_health)` 输出。

徘徊专项当前尚未接入通用心理风险主链；五日任务先形成可对接 evidence、日报和个人基线，不直接生成医学诊断。

## 3. 徘徊已实现能力

### 3.1 数据和模型

- WanderingPatterns/SmartCare 来源适配、split、80 点 14 通道预处理；
- RF、纯 TCN 对照；
- `TopoWanderMPT` TCN + relation-aware Transformer；
- M0/M0-S 监督训练、fresh reload 和固定 primary；
- M0-R/RH release bundle 与 M0-RS public-WP 计分；
- public WP 四分类/binary macro-F1 分别约 `0.983331/0.989010`，仅代表公开轨迹 benchmark。

### 3.2 Camera 和 episode

- bbox-bottom camera adapter、Camera QC 和 40 秒 legacy diagnostic；
- EP1A 人工/CVAT boundary -> episode QC -> 80 点 -> frozen shape；
- EP1B oracle-boundary batch evaluator；
- EP2A-S0 automatic boundary proposal；
- EP2A-S1A boundary evaluator；
- EP2A-S2A proposal -> interval QC -> frozen shape；
- EP2A-S2B proposal/prediction/tracking review pack；
- technical hard break、三态/失败保留和 non-overwrite 输出。

### 3.3 日级软件底座

- session-level synthetic evidence；
- `WanderingDailySummary`；
- 3/7/14 日 `WanderingBaselineProfilePreview`；
- `WanderingBaselineDeviationPreview`。

这些 producer/loader 的历史接受证据仍是 synthetic contract。当前任务不是重写它们，而是增加真实授权 camera episode 的薄 adapter，显式绑定 `person_id/session_id/timezone/presence`，再复用其统计和校验原语。

### 3.4 W5D-00 交付契约

- `camera_delivery_contract.py` 和 CLI 使用 fresh staging + atomic commit，拒绝覆盖已有目录；
- B01 36 条、B02 12 条联合 development index 已逐条校验 video/tracking/sidecar/truth/CVAT 路径和 SHA-256；
- fixed primary、80 点/14 通道/2D shape、`0.5` binary threshold 和 CPU runtime identity 已固定；
- episode/context/daily/baseline profile/deviation/handoff/summary 使用九份 JSON Schema；
- 统一顶层状态为 `ready/uncertain/unavailable/error`，shape 与 purpose/context 分离；
- 交付证据见 [W5D-00 contract bundle](../../../reports/mental_health/wandering_camera_5d_contract_v3/output_contract.md)。

### 3.5 W5D-01 分段器联合优化

- B01 36 条和 B02 12 条在两轮共 29 个 full-cohort 候选中共同用于 development；B01、B02 和 pooled 指标分开输出；
- development profile 支持声明式 state-machine 参数和 `0.25` 秒局部边界 refinement，冻结 v1 proposal 路径保持不变；
- 唯一 recall-first fallback 为 `closing-s008-d04`，pooled recall/F1/tIoU/coverage 为 `0.730769/0.622951/0.716007/0.935897`；
- B01 recall/F1 为 `0.672414/0.537931`，B02 为 `0.900000/0.947368`；technical hard-break crossing 为 `0`；
- recall 和 F1 未达到 `0.75/0.72` 最低目标，B01 fragmentation 和 false candidate 仍是 W5D-02 的显式限制，不把 fallback 写成达标模型；
- top-3 已做确定性全量复放，全部 miss 与严重 split/merge 已保留；证据见 [W5D-01 segmenter search](../../../reports/mental_health/wandering_camera_segmenter_search_v2/README.md)。

### 3.6 W5D-02 自动 episode 与轨迹识别闭环

- 已形成无需人工改中间 JSON 的 `tracking/sidecar -> proposal -> QC -> 80 点 -> shape` 自动入口；B01+B02 的 48 条 development 视频生成 129 条一一对应的 episode result，`proposed+uncertain` 送模型，`rejected_by_qc` 跳过；
- 最终 v4 运行的 `ready+uncertain` prediction coverage 为 `70/78=0.897436`，最低门 `>=0.85` 通过但 `0.90` 目标未达，故整体状态仍为 `uncertain`；结果无 error，模型 forward 95 次、QC skip 34 次；
- binary 是主结果，四分类保留为诊断；automatic boundary binary support/accuracy/macro-F1=`51/0.941176/0.936383`，oracle binary=`77/0.857143/0.875629`，pipeline miss 分开记录；
- fresh oracle 证据支持保留 fixed primary `topowander-m0s-seed20260731-epoch0005` 和 threshold `0.5`，不做 camera binary 调整，主要短板落在 automatic boundary/QC；
- automatic boundary 统一标记为 `automatic_boundary_not_human_accepted`，产物只是 B01+B02 development evidence，不是 sealed 或跨人/跨机位效果，也不发出 `AlgorithmEvent`；证据见 [W5D-02 v4 pipeline](../../../reports/mental_health/wandering_camera_episode_pipeline_w5d02_v4/README.md)。

### 3.7 W5D-03A 三帧多模态 context core

- `camera_context_review.py` 与 CLI 从 W5D-02 结果选择 `binary=wandering_like` 或顶层 `uncertain` episode；每个 trigger 恰好输出一条 context row，非 eligible 不调用 provider；
- 每条 row 固定包含 episode 起/中/末三个 frame ref；每个时点记录请求/实际时间、frame index、dominant-track bbox、完整场景与人物 crop 的 Git 外路径及 SHA-256；缺帧或缺 bbox 仍保留 `unknown/unavailable` 行；
- provider 层已有统一 request/response、disabled、deterministic fake 和 OpenAI-compatible multimodal adapter；真实 adapter 的六张图序列化、严格 JSON 解析以及 missing-key/connection/HTTP/timeout/非法响应降级均有 stubbed-client 回归，不要求 live key 或外网；
- context v2 schema 增加结构化 `error_code`、trigger、三帧 scene/crop refs、upstream episode identity 和 snapshot SHA；context 不能修改 shape、boundary、truth、QC 或 threshold；
- 全量 B01+B02 development evidence 从 129 条 episode result 选出 82 个 trigger，生成 82/82 row、246/246 个 ready 时点和 492 个 JPEG artifact，逐引用 hash 与 episode snapshot 全部复核；fake 标签只验证 ready contract，不是 context accuracy；
- 独立 disabled evidence 覆盖 B01/B02 两个登记视频的 3 个 trigger，成功退出并生成 `3/3 unknown+unavailable+provider_disabled`，provider invocation 为 0；
- 核心状态为 `completed_core_awaiting_home_smoke`，`home_input_status=awaiting_input`、`home_smoke_status=not_run_input_unavailable`，不发 `AlgorithmEvent`。最终证据见 [W5D-03A full context core](../../../reports/mental_health/wandering_camera_context_core_w5d03a_v2/README.md) 和 [disabled degradation](../../../reports/mental_health/wandering_camera_context_core_w5d03a_disabled_v2/README.md)。

### 3.8 W5D-04 真实 development 日报与个人基线

- `camera_daily_baseline.py` 和 CLI 复用既有 IANA 自然日/区间切分与线性分位数原语，新增 real-development binding、daily v2、严格 prior profile/deviation 和可独立重算的 public loader；不把旧 synthetic producer 改名为真实证据；
- binding manifest 显式展开 B01 36 条、B02 12 条视频到 `P-OFFICE-01`、两个 session、`Asia/Shanghai` 和 full authorized protocol presence，人员身份来自 development index owner binding，不从 `track_id` 推断；
- sidecar 的 `capture_started_at` 全为 null，因此使用声明式 development schedule，只确定所属自然日与可重复 clip 顺序，并携带 `capture_clock_unavailable_declared_schedule`；该时间不能作为实测日内节律证据；
- 129 条 episode 的聚合守恒通过：95 条可用 shape、81 条 uncertain、34 条 unavailable、1 条 auto-accepted wandering-like high-confidence，context bucket 总数 129，wandering-like 总时长输入/输出均为 617.8 秒；fake context 统一进入 `unknown`，不反改 shape；
- 真实两日 presence 为 1960.2/667.333333 秒，tracking coverage 为 0.995102/1.0；两个 observation profile 分别使用严格 prior 的 0/1 个 usable day，均为 `warming_up`，没有伪造 stable baseline；
- 独立 15 日 deterministic replay 得到 3 个 `warming_up`、4 个 `initial_ready`、8 个 `stable_ready`，第 15 日 reference 为严格 prior 的 14 日窗口。replay 只证明状态机，不是实际长期观察；
- producer 使用 staging + atomic commit，拒绝已有目录；manifest 绑定 schema/输入/输出 SHA-256，public loader 会重新计算全部 profile/deviation。证据见 [B01+B02 daily/baseline](../../../reports/mental_health/wandering_camera_daily_baseline_w5d04_v1/README.md) 与 [15-day replay](../../../reports/mental_health/wandering_camera_daily_baseline_w5d04_replay_v1/README.md)。

### 3.9 W5D-05 算法对接包与 E2E

- `camera_handoff.py` 和 CLI 将 W5D-02/03A/04 组装为严格 8 文件后端对接包；五类 JSONL 原样透传，不静默丢弃 `uncertain/unavailable/error`；
- manifest/run-summary v2 同时记录 delivery、validation 和 home input/smoke 状态；固定不发 `AlgorithmEvent`、不输出医学诊断，也不实现后端或前端；
- producer 验证三阶段 manifest SHA、artifact descriptor/record count、现行 row schema、context→episode snapshot/identity、daily→episode/context hash、daily/profile/deviation identity 和汇总守恒；public loader 会独立重读最终包；
- 正式 v2 包包含 129 episode、82 context、2 daily、2 profile、2 deviation；95 条可用 shape、34 条 unavailable、0 条 error、19 条 wandering-like 和 617.8 秒 duration 全部守恒；
- 单命令 cached-tracking E2E 已重跑 48 条 B01+B02 视频并复得 129/82/2/2/2；disabled 降级包保留 3 条 `unknown/unavailable` context，同时继续输出完整 episode/daily/baseline；
- 输出采用 fresh staging + atomic commit；stable ID、non-overwrite、descriptor/stage/cross-stage tamper 均有回归；
- 最终状态为 `algorithm_ready_with_home_smoke_pending`，`home_input_status=awaiting_input`、`home_smoke_status=not_run_input_unavailable`，不是 `home_validated`。证据见 [B01+B02 handoff v2](../../../reports/mental_health/wandering_camera_handoff_w5d05_v2/README.md) 与 [disabled degradation v2](../../../reports/mental_health/wandering_camera_handoff_w5d05_disabled_v2/README.md)。
- 最终验收为 handoff 聚焦 `10 passed`、全 camera `660 passed`、wandering + mental_health 联合 `929 passed, 205 subtests passed`。

## 4. B01/B02 最新使用规则

B01 和 B02 现在统一为可反复使用的 camera development 数据：

- 可以共同调分段器参数、边界 refinement、送模策略和 camera binary；
- 可以反复回放、做失败分析和必要的轻量模型/threshold 校准；
- 不再把 B02 作为当前不可回调 holdout；
- 旧 B02 holdout/v1/v2 数字继续保留在报告中，只作历史快照；
- 任何 B01/B02 新指标必须写 development，不写跨人泛化或临床效果。

历史起点：

| 指标 | B01 | B02 |
|---|---:|---:|
| oracle binary macro-F1 | 0.817375 | 1.000000 |
| automatic boundary F1 | 0.598131 | 0.971429 |
| candidate-inclusive ready coverage | - | 17/18 |

当前主要短板是 B01 分段 F1/fragmentation、真实拍摄 clock 缺失、真实长期个人观察和居家场景 smoke。W5D-02/W5D-03A/W5D-04/W5D-05 主链已完成，后续不因居家素材暂缺回退到分段、context、日报或 handoff 重做。

## 5. 居家视频与上下文

W5D-03A 已用 B01+B02 和确定性 fixture 完成候选三帧、context adapter、provider 失败降级及下游契约：

```text
T2 episode result -> frame refs -> optional context -> daily
```

居家 MP4/人工标签不是上述核心的前置门。MP4 未到位时记录 `home_input_status=awaiting_input`；MP4 到位但未标注时直接运行 truth-free `MP4 -> tracking -> segmenter -> shape -> context -> daily` smoke，不计算准确率；有独立标签后才补 development 定量评价。

轨迹模型只判断 shape。对 `wandering_like` 或 uncertain episode 抽取开始/中间/结束三帧，由可插拔多模态模型判断：

```text
phone_call | searching | cleaning | exercise | social | other | unknown
```

多模态调用不得修改 shape、boundary 或人工 truth；无 key、超时或失败时输出 `unknown/unavailable`，主链继续。

## 6. 日报与基线

五日目标日报至少包含：

- presence、tracking/QC coverage；
- episode count/duration；
- shape/binary 分布；
- high-confidence 与 uncertain candidate；
- wandering-like count/hour、duration/hour、night ratio；
- context counts；
- unavailable/error/quality；
- baseline readiness 与 deviation。

基线状态：

```text
1-2 usable days: warming_up
3-6 usable days: initial_ready
>=7 usable days: stable_ready
latest 14 usable days: rolling window
```

单段居家视频只能形成当日报告和 `warming_up`，不能写成稳定个人基线。多日 fixture/replay 用于验证机制，不能冒充真实七日观察。

## 7. 当前连续任务

| 顺序 | 任务 | 作用 |
|---|---|---|
| W5D-00 | 交付契约与运行基线 | 已完成：48 条 index、CPU identity、九份 schema、fresh/non-overwrite |
| W5D-01 | B01+B02 分段器联合优化 | 已完成：两轮 29 候选、唯一 recall-first fallback、top-3 全量复放、失败留存 |
| W5D-02 | 自动 episode 与轨迹识别闭环 | 已完成 development 主链；coverage 达最低门但未达 0.90，状态 `uncertain`，已保留 boundary/QC 限制 |
| W5D-03A | input-independent 多模态上下文 | 已完成 core：82/82 trigger rows、246/246 三帧时点、真实 adapter contract 与 disabled/error 降级；居家 smoke pending |
| W5D-04 | 真实日报与个人基线 | 已完成：48 个显式 binding、2 个真实 development day/profile/deviation 均 warming-up；15 日 replay 验证 3/7/14 与最新 14 日窗口 |
| W5D-05 | 算法对接包与整体验收 | 已完成：严格 8 文件、manifest/run-summary v2、单命令 E2E、public loader、disabled 非阻塞和 tamper/non-overwrite 回归 |
| W5D-03B | 居家视频补充 smoke | `awaiting_input`：素材到位后 truth-free 复放，人工标签只在需要定量评价时补；不是当前代码任务 |

精确输入、输出、完成门和命令见 [五日总任务书](plans/M0-CAM-5D快速交付总任务书.md)。实际状态只在 [任务表](../../tasks/README.md) 更新。

## 8. 常用验证

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_delivery_contract.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_episode_boundary.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_episode_proposal_inference.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_context_review.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_daily_baseline.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera_daily_summary.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py tests/test_mental_health_*.py -q
```

所有 Python/pytest 必须使用 `eldercare-ai`，editable location 必须指向当前仓库。

## 9. 不可误报的结论

- public-WP 高分不等于 camera/居家效果；
- B01/B02 development 指标不等于跨人泛化；
- 单个 wandering-like shape 不等于临床徘徊；
- 多模态 purpose/context 不等于医学判断；
- 单日数据不等于稳定个人基线；
- algorithm handoff ready 不等于后端或前端完成。
