# 自采视频与 CVAT 标注增量整理报告

审查日期：2026-08-11
审查对象：`自采视频.zip`、`自采标注.zip`
结论状态：已完成安全解包、内容哈希对账、媒体探测、CVAT 结构校验、身份元数据脱敏和本地分区整理；未接入统一 manifest、正式标签或 split。

## 1. 结论

新交付包含 262 个 MP4 和与之一一对应的 262 个 CVAT 任务，覆盖 P01–P05。其中 P01–P03 的 156 个视频与 2026-08-10 已整理副本逐字节一致，未重复覆盖；P04 的 50 个和 P05 的 56 个视频新增到 `review_pending/`。

| 参与者 | 视频 | CVAT 轨迹 | 当前状态 | 可用边界 |
|---|---:|---:|---|---|
| P01 | 50 | 123 | `engineering_replay_only` | 仅工程回放、误报复现和演示 |
| P02 | 50 | 84 | `engineering_replay_only` | 同上；另有 CVAT 坐标缩放问题 |
| P03 | 56 | 297 | `review_pending` | 4 组重复条件冲突和帧数差未裁决 |
| P04 | 50 | 91 | `review_pending` | 缺授权引用、subject profile 和双人复核 |
| P05 | 56 | 191 | `review_pending` | 仅解除部分旧文件的来源未知，未解除训练门禁 |

全部 262 条记录仍为 `training_eligible=false`。本次没有修改 `data/manifests/fall_risk_video_manifest.jsonl`、v2/v3 标签或任何 split。

## 2. 来源与完整性

| 原始包 | 字节数 | SHA-256 | 校验 |
|---|---:|---|---|
| `自采视频.zip` | 1,639,162,021 | `3859ae64fb947574e05a95edede54339628847144e419e9fe4e57fdd6235b577` | ZIP 完整，262 个 MP4 |
| `自采标注.zip` | 846,797 | `7aceb317420bf49280b3f7bff748ae306fdca90fefe1b487c4e1ea2f63f53625` | ZIP 完整，5 个 CVAT 项目子包 |

262 个视频中有 258 个唯一内容，`ffprobe` 全部成功；总时长 1,821.716 秒。编码为 H.264 247 条、HEVC 15 条，138 条带音频。原片原样保留；后续如生成训练派生版，应确定性去音轨并记录派生哈希。

CVAT 1.1 项目导出共含 262 个任务、786 条轨迹、44,809 个框，其中 44,285 个为活动框。所有任务都有轨迹，文件名中的计划动作 ID 均在对应任务标签中出现。账号、邮箱、owner 和 assignee 节点已从仓库内副本移除，原子包未复制进仓库。

## 3. 增量对账与异常

### P05 反向映射

P05 中 49/56 个视频与旧 `quarantine/unassigned/` 中的 49/50 个视频内容哈希一致。这为旧文件提供了可审计的 P05 文件名映射，但不能替代授权和人工复核。

尚未与旧隔离文件匹配的 7 个新 P05 视频为 `P05_S03_E001_Q01_A01.mp4` 和全部 6 个 S05 视频；旧隔离目录中唯一未被新包匹配的文件为 `N03_rear.mp4.mp4`。同时，P05 S03 的哈希映射与旧 `Nxx_rear` 序号不是自然一一对齐，因此旧隔离副本未删除。

### P03 重复和帧数

2026-08-10 报告中的 4 组 P03 字节级重复仍然存在，共涉及 8 个文件。它们的动作 ID 相同，但被赋予了不同 session/条件语义，在条件映射裁决前仍按 `quarantine_duplicate_condition_conflict` 处理。

P03 全部 56 个视频的容器帧数比 CVAT 任务 `size` 多 3–7 帧。项目导出内的轨迹在 CVAT 自身时间轴内合法，但在未确认多出帧位置前，不能宣称与原视频帧级精确对齐。

### P02 坐标和转换器兼容性

P02 原视频是 2560×1440，CVAT 项目的 `original_size` 和框坐标是 640×360，边界框在用于原视频前需要按 4 倍缩放并专门验证。P03 由旋转元数据呈现为 1080×1920，容器编码尺寸是 1920×1080且 rotation=-90。

现有通用 CVAT 转换器要求 `fall_risk__...` 任务命名，会拒绝本批 `P01_S05_E001_Q01_A01.mp4` 类型的任务名。这是显式兼容性门禁：后续应增加来源专用导入器和测试，不应手工改名或使用临时 FPS 强行发布。

## 4. 整理结果

```text
data/raw_videos/fall_risk/self_collected/SCF_MVP_V1/
  engineering_replay/clips/P01/          50
  engineering_replay/clips/P02/          50
  review_pending/P03/                     56
  review_pending/P04/                     50
  review_pending/P05/                     56
  quarantine/unassigned/                  50  # 保留旧名供追溯

data/annotations/fall_risk/cvat_exports/raw/self_collected_scf_mvp_v1/
  P01_cvat_redacted.zip
  P02_cvat_redacted.zip
  P03_cvat_redacted.zip
  P04_cvat_redacted.zip
  P05_cvat_redacted.zip

data/documents/fall_risk/self_collected/SCF_MVP_V1/audit/ingest_20260811/
  delivery_inventory.jsonl
  delivery_summary.json
```

视频、脱敏 CVAT 导出和逐文件审查清单均按 `.gitignore` 作为本地敏感数据保存；含身份元数据的原始子包只保留在用户交付位置。机器清单记录每个视频的 SHA-256、媒体参数、CVAT 任务/轨迹/标签摘要、旧隔离文件映射、重复组和治理状态。

## 5. 进入正式标签链前的门禁

1. 补齐 P01–P05 的授权/同意引用、脱敏 `subject_profiles` 和 session 场景映射，明确 S06 含义。
2. 人工裁决 P03 四组重复条件冲突，并确认 P03 CVAT 少 3–7 帧的具体位置和对齐规则。
3. 核实 P05 S03 与旧 `Nxx_rear` 的非自然映射，裁决 `N03_rear.mp4.mp4` 及 7 条新视频的来源关系。
4. 实现来源专用、manifest-backed CVAT 导入器，处理 P02 坐标缩放和 P03 旋转/帧数差，并先生成 candidate，不覆盖根标签。
5. 对动作边界、目标人、质量和 near-fall 恢复证据做双人复核；当前所有 `target_subject` 仍为 `unknown`。
6. 只有在来源、哈希、复核、去泄漏分组和冻结决策全部齐备后，才能修改 manifest、v2/v3 标签和 split。

机器事实源为本地 `data/documents/fall_risk/self_collected/SCF_MVP_V1/audit/ingest_20260811/`；本报告只保留可读结论，不替代逐文件清单。
