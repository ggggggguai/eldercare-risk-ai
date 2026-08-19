# M0-CAM-EP2A-S2B 执行任务书

> 历史执行记录：本任务保留 review-pack 契约，不再阻止 context、日报或基线开发。现行三帧多模态 context 任务见 [M0-CAM-5D-T3](M0-CAM-5D-T3居家视频与多模态上下文.md)。

日期：2026-08-16

任务：`proposal-shape review pack / truth-free triage`

当前状态：

```text
historical_next_task=independent participant/setup validation
current_task=W5D-03A input-independent context core; W5D-03B home smoke pending input
m0cam_ep2a_s2b_implementation_status=completed
m0cam_ep2a_s2b_data_status=synthetic_contract_plus_partial_truth_free_review_smoke_only
m0cam_ep2a_s1b_policy_status=frozen_on_b01_development
m0cam_ep2a_s1b_video_holdout_status=completed_once_on_b02
m0cam_ep2a_automatic_shape_coverage_status=insufficient
m0cam_ep2a_status_at_historical_snapshot=not_completed
```

完成记录（2026-08-16）：最小 config/module/CLI/tests/report 已落地。builder 按 `proposal_id + full scope` 只读连接 S0 proposal、S2A result、tracking 和视频身份，输出一 proposal 一 review row、一张 SVG、逐视频摘要和人工字段全空的模板；缺失、重复、错 scope/binding/endpoint/status 均 fail closed。聚焦测试 `6 passed`，S0/S2A/EP1A 相邻回归 `29 passed`，CLI help、compileall、`git diff --check` 与 4/4 smoke SVG XML 解析均通过。H02/H03 4 条 uncertain proposal 形成 4 行/4 图，旧 S2A invocation/skipped=`0/4`，run A/B 八个对应文件一致，人工字段预填单元格为 0。2026-08-17 candidate-inclusive v2 增加 uncertain+ready 合法 join，review row 展示冻结模型概率但仍保留 uncertain/manual-review；S2A/S2B 聚焦测试为 `8 passed`。详见 [S2B 报告](../../../../reports/mental_health/wandering_camera_episode_proposal_review_v1/README.md)。

该完成记录只关闭 S2B implementation。后续 B01/B02 owner binding、旧 cohort/holdout 和 v2 结果仍保留历史价值；S2B 模板仍不是真值。现行路线允许继续使用 B01+B02，并以 [W5D-T3](M0-CAM-5D-T3居家视频与多模态上下文.md)实现三帧 context 和失败降级。

## 1. 目标

S0 已输出独立 boundary proposal，S2A 已在不改变 proposal endpoint 的前提下输出一对一 proposal-shape result。S2B 只补一条最小人工复核桥接线：

```text
S0 proposal
  + S2A proposal-shape prediction
  + tracking trajectory
  + media/video identity
  -> truth-free human review bundle
```

它必须让人工复核者能直接对应：

- 哪个视频；
- 哪个时间段；
- 哪条轨迹；
- 哪个模型输出。

S2B 不修改任何上游产物，也不判断 proposal 或 prediction 是否正确。

## 2. 打薄范围

S2B 是 review-pack 生成器，不是新的 implementation-only 审计包。只实现只读连接、必要的可视化和空白人工记录入口，不增加 evaluator、acceptance converter、发布治理或产品决策层。

S2B 不是：

- truth 生成工具；
- boundary evaluator；
- 性能报告；
- EP2B、EP2C 或 EP3；
- alert、risk 或 `AlgorithmEvent`；
- 新的 receipt、PORTABLE、逐文件 hash 或大型 exact-schema 审计包。

## 3. 只读输入

最小输入为：

- S0 `wandering-camera-episode-boundary-proposal-v1` bundle；
- S2A proposal-shape prediction bundle；
- 对应 tracking JSONL；
- media sidecar 与最小 batch/video identity index。

连接必须使用 `proposal_id` 和完整 scope，并保留 proposal 原始 start/end、status、reason、technical segment 与人工复核要求。不得读取 shape truth、purpose truth、人工 boundary、AI 预标注或 XML。

## 4. 最小产物

```text
review_index.csv 或 review_index.jsonl
per_proposal_trajectory_plots/
per_video_summary.md
human_review_template.csv
README.md
```

`review_index` 至少把 video identity、proposal identity/scope、原时间端点、proposal status/reasons、prediction status、model invoked/skipped 与轨迹图路径连接在同一行。逐 proposal 图只展示对应轨迹和区间身份，不推断或改写标签。

`human_review_template.csv` 只能提供待人工填写的空白列。模型预测、proposal status 或 reason 可以作为独立只读上下文列展示，但绝不能预填到人工 boundary/shape/acceptance/decision 列，也不能把空白列自动派生为模型答案。

## 5. H02/H03 证据边界

现有 H02/H03 四条真实 proposal 全部为 `uncertain`。truth-free smoke 只能检查：

- 四条 review row 均存在；
- proposal/status/reason/prediction 状态可见；
- 对应轨迹图与视频身份可定位；
- 空白人工列未被模型结果预填；
- fresh 输出可复跑且不覆盖。

该 smoke 不覆盖 `proposed` 路径性能，不能证明 boundary 或 shape 正确，也不能报告自动切段性能。

## 6. 最低验收

1. 每条输入 proposal 恰好对应一条 review row，proposal、prediction、tracking 和视频身份连接无歧义。
2. 每条 proposal 都有可定位的 trajectory plot；缺失或不一致的只读输入必须结构化失败，不能静默错连。
3. `human_review_template.csv` 的人工判断列全部为空，且测试明确防止预填模型答案。
4. 输入 proposal、prediction、tracking、sidecar、XML 和 truth 均未修改。
5. 输出 fresh、deterministic、non-overwrite；不生成 accepted boundary、truth 或性能结果。
6. synthetic fixture 覆盖至少一种 model-invoked result 和一种 skipped result 的 review 可见性；H02/H03 只覆盖四条 uncertain skip 的真实 truth-free smoke。
7. CLI help、聚焦测试和 Markdown/报告同步完成。修改 Python 后按 `AGENTS.md` 使用 WSL `eldercare-ai` 先跑窄测，再按共享面决定相邻 camera 回归。

## 7. 硬边界

- 不修改 proposal、prediction、tracking、sidecar、XML 或 truth；
- 不修改 proposal endpoint/status/reason，不把 S2A prediction 写成 truth；
- `human_review_template` 只能保留空白人工列，不得预填模型答案为人工标签；
- 不计算 P/R/F1、tIoU、起止误差、漏切、多切、alert、FAR 或临床指标；
- H02/H03 全 uncertain 的真实 smoke 只能证明 review rows/plots 和状态可见，不能证明 proposed path 性能；
- 不启动 EP2B、EP2C 或 EP3，不实现 alert、risk 或 `AlgorithmEvent`；
- 不扩展 receipt、PORTABLE、逐文件 hash、字节对齐或大型 exact-schema 审计；
- 不访问 sealed 数据，不修改其他模块或共享依赖；
- 不覆盖既有输出，不 commit/push，不 reset/checkout/clean/stash。

## 8. 已执行的开发 AI 目标提示词（历史）

以下提示词对应已经完成的 S2B，不再是当前代码任务，不应据此重复实现或扩展 S2B。当前执行入口是[人工证据交接](M0-CAM-EP1B-EP2A-S1B人工证据交接.md)。

```text
你是本项目 M0-CAM-EP2A-S2B 的主开发 AI。请实现最小 proposal-shape review pack / truth-free triage，不要把它扩展成 evaluator、truth 工具或大型审计包。

模型：GPT-5.6 Codex / gpt-5.6-sol
智商（推理强度）：最高（max）

开始前完整阅读并严格执行：
1. AGENTS.md
2. docs/tasks/README.md
3. docs/modules/mental_health/README.md
4. docs/modules/mental_health/plans/徘徊识别技术文档2.md
5. docs/modules/mental_health/plans/M0-CAM-EP2A-S0执行任务书.md
6. docs/modules/mental_health/plans/M0-CAM-EP2A-S2A执行任务书.md
7. docs/modules/mental_health/plans/M0-CAM-EP2A-S2B执行任务书.md
8. reports/mental_health/wandering_camera_episode_boundary_proposal_v1/README.md
9. reports/mental_health/wandering_camera_episode_proposal_inference_v1/README.md 与 VERIFICATION.md
10. S0/S2A module、CLI 和对应测试源码

当前事实：
- S0 proposal producer、S1A boundary evaluator 和 S2A 薄 proposal-shape bridge 均已完成；EP2A 总状态仍 not_completed。
- 当前 S2A v2 的 proposed+uncertain 使用原始 endpoint 做 episode QC/80 点预处理并在 ready 后调用 frozen model；uncertain 保持原 status/manual-review，只作 candidate diagnostic；rejected_by_qc 输出 unavailable 并跳过模型。
- S2A 输出是独立 proposal-shape schema，不生成 accepted boundary，不含 oracle_boundary；EP1A oracle schema、flags 和概率未变。
- synthetic 三行 fixture 的旧 v1 ready/unavailable/boundary_uncertain=1/1/1，真实 frozen model 调用 1 次、跳过 2 次；当前 v2 测试要求 proposed+uncertain 都可 ready。
- H02/H03 四条真实 proposal 的 boundary_uncertain/skipped 结果是 2026-08-16 历史 v1 smoke，不描述当前 v2；它只验证旧 uncertain skip。
- 当前证据任务仍是 M0-CAM-EP1B + M0-CAM-EP2A-S1B；下一证据缺口是独立人工 continuous boundary、S1B 参数冻结和独立 participant/session/setup 评价。

必须实现：
1. 只读输入 S0 proposal bundle、S2A proposal-shape prediction、tracking JSONL、media sidecar 和最小 batch/video identity index。
2. 按 proposal_id 与完整 scope 一对一连接，保留原 endpoint/status/reason/technical segment 和 prediction 状态，回答哪个视频、时间段、轨迹和模型输出。
3. 最小输出仅为 review_index.csv 或 JSONL、per_proposal_trajectory_plots/、per_video_summary.md、human_review_template.csv 和 README.md。
4. human_review_template 的人工 boundary/shape/acceptance/decision 列必须为空；不得把 prediction 或 proposal reason 预填为人工答案。
5. 输出 fresh、deterministic、non-overwrite；不得修改任何输入，不生成 truth 或 accepted boundary。
6. synthetic 测试覆盖 model-invoked 与 skipped result 的 review 可见性、一 proposal 一 row、连接错误拒绝、空白人工列和 non-overwrite。
7. H02/H03 truth-free smoke 保持四条 uncertain 原状态，只验证 review rows/plots/status/video identity 可见和可复跑性。
8. 更新最小报告与现行文档。没有人工 boundary 时保持 m0cam_ep2a_status=not_completed。

禁止：
- 不读取或修改 truth/XML，不改 proposal、prediction、tracking 或 sidecar，不把 S2A prediction 写成 truth。
- 不计算 P/R/F1、tIoU、起止误差、漏切、多切、alert、FAR 或临床指标。
- 不训练、调参、改阈值、改标签、改 proposal endpoint，不实现 alert/risk/AlgorithmEvent。
- 不启动 EP2B/EP2C/EP3，不扩展 receipt、PORTABLE、逐文件 hash 或大型 exact-schema 审计。
- 不 commit/push，不 reset/checkout/clean/stash，保护现有 dirty/untracked 工作树。

所有 Python/pytest 必须使用 WSL eldercare-ai，并先确认 editable install 指向当前仓库。最终明确报告实现文件、测试与 truth-free smoke、空白人工模板保证、没有计算的指标、H02/H03 只覆盖 uncertain review 可见性，以及仍缺的人工 continuous boundary 和 S1B 评价。
```
