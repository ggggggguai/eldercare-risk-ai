# M0-CAM-5D-T3 居家视频与多模态上下文

状态：`completed_core_awaiting_home_smoke`

计划：Day 3 完成 input-independent core；居家视频到位后补充 smoke

上游：[T2 自动 Episode 与轨迹识别闭环](M0-CAM-5D-T2自动Episode与轨迹识别闭环.md)（已完成）

下游：[T4 真实日报与个人基线](M0-CAM-5D-T4真实日报与个人基线.md)、[T5 算法对接包与整体验收](M0-CAM-5D-T5算法对接包与整体验收.md)

## 1. 目标

先完成与居家素材到位时间无关的 context 核心：从 T2 episode result 选择 `wandering_like` 或 `uncertain` 候选，抽取开始/中间/结束三帧，调用可插拔多模态 adapter，并为每个触发生成可追溯的 context row。

负责人提供居家 MP4 后，使用同一入口补做真实场景 truth-free smoke。居家视频及其人工标签都是补充输入，不是 W5D-03 核心实现、W5D-04 或 W5D-05 的启动门。

## 2. 当前输入事实与决策

当前没有可用于定量验收的已标注居家视频。按实际输入分三种情况处理：

| 输入情况 | 立即执行 | 可得结论 | 是否阻塞 |
|---|---|---|---|
| 居家 MP4 和标签都未到位 | 用 B01+B02 和确定性 fixture 完成 context/core/E2E；记录 `home_input_status=awaiting_input` | 工程接口、失败降级和闭环可运行 | 否 |
| 居家 MP4 到位但未标注 | 直接运行 truth-free `MP4 -> tracking -> episode -> shape -> context` | 真实场景工程 smoke；不能计算边界或分类准确率 | 否 |
| 居家 MP4 和独立人工标签都到位 | 在 truth-free smoke 之外补 development 定量复核 | 有限的居家 development 误差和失败案例 | 否；属于补充证据 |

不得把模型输出、多模态回答或文件名当作人工 truth。需要量化边界/shape/context 效果时再补人工标签；标签未完成不能阻止算法链运行。

## 3. 数据与身份规则

- 视频、人物画面、截图、标注和内部模型调用默认全部已经授权。
- 保留匿名 `person_id/session_id/setup_id/source_video_id`；原视频、截图和 provider 原始响应不进入 Git。
- B01+B02 是本阶段的替代工程 smoke/development 输入，不冒充居家场景或跨场景泛化证据。
- 居家视频到位后先按当前配置运行；若随后据其调参，记录为 development，不再把同一次结果称为独立验收。
- 多模态结果不能修改 shape prediction、boundary endpoint、人工 truth、QC 决策或模型阈值。

## 4. 连续执行任务

### W5D-03A0：输入状态与输出契约

1. 在运行摘要或 handoff manifest 中显式记录：

```text
home_input_status=awaiting_input|available
home_annotation_status=not_provided|unlabeled|available
home_smoke_status=not_run_input_unavailable|ready|uncertain|error
validation_scope=b01_b02_development|deterministic_fixture|home_truth_free_smoke|home_labeled_development
```

2. 缺少居家输入时使用 `not_run_input_unavailable`，不得把整个 run 标记为 `error`。
3. 固定 eligible trigger、三帧选取、context label、provider status 和 source reference 语义。

### W5D-03A1：候选选择与三帧抽取

1. 读取 T2 `episode_results.jsonl`，只对 `wandering_like` 或需要复核的 `uncertain` episode 触发。
2. 以 episode 有效起止时间选择开始、中间、结束三帧；时间越界时夹紧到可解码范围并写 quality flag。
3. 每个时点同时支持完整场景帧和人物 bbox 裁剪；缺 bbox、缺帧或解码失败时仍生成一条 `unavailable` context row。
4. frame reference 记录时间戳、frame index、SHA-256 和 Git 外 artifact path；不得把媒体二进制写入 JSONL 或 Git。

### W5D-03A2：可插拔 context adapter

1. 定义统一 provider request/response，不让具体云模型字段泄漏到主 pipeline。
2. 最小标签固定为：

```text
phone_call
searching
cleaning
exercise
social
other
unknown
```

3. 必须实现 `provider-disabled` 和确定性 fake adapter，用于离线回归。
4. 至少实现一个可配置的真实多模态 provider adapter 请求/响应路径；本地核心验收允许用 stubbed client 证明序列化、解析和错误映射，不以真实 key、外网或 live call 为完成门。
5. API key 缺失、无网络、超时、拒答、解析失败和非法标签统一降级为 `context_label=unknown`、`status=unavailable`，并保留 `error_code`。
6. live provider smoke 是环境允许时的补充验收；provider 不可用时 episode、daily 和 baseline 主链继续成功。

### W5D-03A3：context producer 与闭环连接

1. 每个 eligible episode 恰好输出一条 context row；不得静默丢行。
2. 保留 episode/result/model/config/source refs，确保下游可按 `episode_id` 连接。
3. 非 eligible episode 不调用 provider；在 run summary 中记录 skipped count 和原因。
4. 输出 record count、ready/unavailable/error count、provider 调用数、解码失败数和主要 error code。
5. context 只增加解释字段，不改变 T2 的 binary/four-class、boundary 或 run status。

### W5D-03A4：无居家视频工程验收

1. 从 B01 和 B02 各选择至少一个已登记视频/result，覆盖 `wandering_like` 或 `uncertain` 触发；不足的错误分支使用确定性 fixture。
2. 运行真实视频帧抽取、fake/provider-disabled、超时/异常降级和 fresh/non-overwrite 回归。
3. 证明输入缺少居家 MP4 时仍能生成 schema-valid `context_reviews.jsonl` 和下游可消费的 partial handoff。
4. 证据明确标记为 B01+B02 development 或 fixture，不写成居家 smoke、context accuracy 或跨场景效果。

### W5D-03A5：交接与状态推进

1. 更新 CLI `--help`、输入要求、输出字段、失败状态和示例命令。
2. 核心完成后状态改为 `completed_core_awaiting_home_smoke`，立即解锁 W5D-04 和 W5D-05。
3. 后续补做居家 smoke 时只新增 fresh evidence，不覆盖核心证据或历史报告。

### W5D-03B：居家视频到位后的补充复放

1. 登记匿名 person/session/setup/video/timezone 和 media identity。
2. 未标注也直接运行一次完整 `MP4 -> tracking -> segment -> shape -> context`。
3. 检查可解码性、目标覆盖、episode/result/context 行数、错误降级和媒体引用。
4. 未标注时只报告 truth-free 工程现象；不输出 accuracy/F1。
5. 有独立人工标签时再做边界、shape 或 context development 复核，并与 truth-free smoke 分目录保存。
6. 补充 smoke 通过后状态可改为 `completed_with_home_smoke`；失败只形成待定位问题，不回滚已经完成的核心与 T4/T5 交付。

## 5. Context schema

```text
episode_id
source_video_id
frame_refs
frame_sha256
provider
model
context_label
confidence
rationale
status
error_code
```

`confidence` 和 `rationale` 可以为空；不得为 provider 不可用伪造数值或解释。

## 6. 核心完成门

- 居家输入缺席被明确记录为 `awaiting_input/not_run_input_unavailable`，而不是 run failure；
- B01+B02 或确定性 fixture 能走通 episode result -> 三帧/引用 -> context row；
- 每个 eligible trigger 恰好有一条 context row，缺帧和 provider 失败也不丢行；
- provider-disabled、fake、timeout/error 降级均有回归，主 pipeline 退出码成功；
- 至少一个真实 provider adapter 的请求/响应路径通过 stubbed-client contract 测试；
- shape、boundary、truth 和阈值在 context 前后保持不变；
- 媒体留在 Git 外，引用和 hash 可定位，输出 fresh/non-overwrite；
- schema、CLI 和下游连接测试通过。

达到这些门即可完成 W5D-03 core 并推进 W5D-04/W5D-05，不等待居家视频或人工标签。

## 7. 居家补充验收门

- 至少一段居家 MP4 完整运行一次 MP4 -> tracking -> episode -> shape -> context；
- 未标注时报告 `home_truth_free_smoke` 和可观察工程问题，不报告性能指标；
- 有标签时才报告相应 development 指标；
- 所有结果保留输入、模型、配置和输出身份，不覆盖核心证据。

本节未完成只表示 `home_smoke_pending`，不表示五日算法包不可运行。

## 8. 验证

现有底座先回归：

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_context_review.py --help
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_context_review.py \
  tests/test_wandering_camera_episode_pipeline.py -q
```

实现时新增 context producer/adapter 的 contract、frame selection、provider-disabled、fake、timeout/error、one-trigger-one-row、non-mutation 和 non-overwrite 聚焦测试。所有 Python 与 pytest 继续使用 `eldercare-ai`。
