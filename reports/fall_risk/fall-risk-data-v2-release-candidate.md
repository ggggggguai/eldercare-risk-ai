# fall-risk-data-v2 发布候选状态

更新时间：2026-08-08

结论：manifest、v2 转换器、根标签发布器、严格校验器、四任务 split builder 和事件评估器已实现；Fall Detection 2017 人工批次已完成 v2/v3 候选链重建，但 formal 校验仍有 blocker，因此当前仍是 `release_candidate`，不是 frozen 数据版本。

| 能力 | 状态 |
|---|---|
| 7,534 条资产 / 6,520 条视频 manifest | 已重建；3,924 个 NTU AVI 均可访问，主 manifest 纳入 3,914 个已接受 NTU 视频，其中 938 个为人工标注 A043；2,014 个 Fall Detection 2017 视频也已纳入；UR Fall `adl-07-cam0.mp4` 已恢复并完整读取 180/180 帧 |
| 标注来源归档 | raw 目录当前共 124 个 CVAT ZIP，其中 123 个缺失文件已恢复并逐个匹配标签固定 SHA-256；含身份信息的原包未直接复制，相关包均按历史脱敏流程重建 |
| v2 CVAT/LE2I 转换标签 | 已生成 |
| Coffee 原始来源 | 已找回，hash 匹配 |
| Lecture room CVAT 来源 | 250 条动作/事件候选已脱敏转换 |
| Office CVAT 来源 | 158 条动作/事件候选已脱敏转换 |
| Pre_VFallp 媒体资格与标注 | 108/108 可用；5 个 CVAT 压缩包授权组覆盖 36+18+18+18+18 条视频，全部 108 条已有导入标签，公开来源仍未验证 |
| Fall Detection 2017 人工 CVAT | 2,011 个源导入，生成 2,977 action/event；候选来源仍需 QC 和来源治理 |
| 根标签 v2 formal | `errors=0`、`blockers=285`，媒体访问错误已清零但治理门禁尚未通过 |
| 模型训练标签 v3 | `valid=true`、`issues=[]`；fall/near-fall 事件监督门禁通过，动作类型因稀有类别 split 覆盖不足仍未通过 |
| 根标签发布 | 9,314 action、6,338 event；纳入 4,404 条 NTU 动作、1,428 条 A043 映射事件、2,977 条 Fall Detection 2017 动作/事件和 268 条 UR Fall 动作/事件，排除 71 条重叠 CVAT `fall`，隔离 1,428 条目录名与 `batch_id` 不一致的候选动作/事件 |
| 四任务 split | 开发版 fall=2,291、near-fall=14 为 ready，功能/纵向仍 blocked；均未 frozen |
| 真实正式指标 | 未生成 |

下一步以[工作流 A 当前阻塞](workflow_a_blockers.md)为准。
