# fall-risk-data-v2 发布候选状态

更新时间：2026-07-25

结论：manifest、v2 转换器、根标签发布器、严格校验器、四任务 split builder 和事件评估器已实现；根标签已从 v2 来源批次重建，但 formal 校验仍有 blocker，因此当前仍是 `release_candidate`，不是 frozen 数据版本。

| 能力 | 状态 |
|---|---|
| 6,530 条资产 / 5,516 条视频 manifest | 已重建；含 2,976 条 NTU，但其旧解压媒体路径当前不可用 |
| v2 CVAT/LE2I 转换标签 | 已生成 |
| Coffee 原始来源 | 已找回，hash 匹配 |
| Lecture room CVAT 来源 | 250 条动作/事件候选已脱敏转换 |
| Office CVAT 来源 | 158 条动作/事件候选已脱敏转换 |
| Pre_VFallp 媒体资格与标注 | 108/108 可用；5 个 CVAT 压缩包授权组覆盖 36+18+18+18+18 条视频，全部 108 条已有导入标签，公开来源仍未验证 |
| 根标签 v2 formal | `errors=5,952`（NTU 媒体路径缺失）、`blockers=179`，尚未通过 |
| 根标签发布 | 4,759 action、1,783 event；纳入 2,976 条 NTU 人工精确动作和 268 条 UR Fall 动作/事件，排除 71 条重叠 CVAT `fall`；A043 的 948 条不导入 |
| 四任务 split | 开发版 fall=248、near-fall=13 为 ready，功能/纵向仍 blocked；均未 frozen |
| 真实正式指标 | 未生成 |

下一步以[工作流 A 当前阻塞](workflow_a_blockers.md)为准。
