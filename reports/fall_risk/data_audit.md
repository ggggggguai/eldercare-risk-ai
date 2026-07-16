# 跌倒风险数据审计（Workflow A）

核验日期：2026-07-15

候选版本：`fall-risk-data-v1`

状态：数据与评估自动化候选；未冻结，不可作为正式比赛或临床指标来源。

## 环境与范围

- editable 安装：已由 `pip show` 确认绑定当前仓库
- conda 环境：`eldercare-ai`
- manifest：`data/manifests/fall_risk_video_manifest.jsonl`
- manifest SHA-256：`6cbb45d7f5f11aabe0239426022e9df31ba9c4383078ae05dfce870bb5a7e049`
- 资产总数：3,454；视频：2,440；表格：322；时序资产：692。
- 资产级 `eligibility=true` 共 2,725 条，`false` 共 729 条。该字段只表示 manifest 来源/许可/重复检查结果，不代表标签已经通过人工复核。

## 数据集清点

| 数据集 | 资产 | 当前可用 | 当前隔离/排除 | 主要说明 |
|---|---:|---:|---:|---|
| Fall Detection 2017 | 2,014 | 2,012 | 2 | 重复内容保守排除 |
| GSTRIDE | 475 | 475 | 0 | 视频、IMU、步态与分段资产均纳入 |
| LE2I/IMViA | 190 | 0 | 190 | manifest 保守记录为待法律确认，不进入正式 split |
| LTMM | 2 | 2 | 0 | 仅本地表格/报告；长期原始信号缺失 |
| Pre_VFallp | 108 | 0 | 108 | 来源、许可与标签语义未确认，整体隔离 |
| TOAGA | 423 | 0 | 423 | 许可状态待确认；394 份姿态 CSV 与参与者表已索引 |
| UR Fall | 242 | 236 | 6 | 重复内容保守排除；142 份同步 CSV 已索引 |

manifest 排除原因按资产计数：`license_unknown=721`、`dataset_quarantined=108`、`source_unknown=108`、`duplicate_content=14`；同一资产可有多个原因。已恢复 2,910 条资产的公开数据集伪名 subject，544 条保持 `unknown`；共有 282 个保守 `source_group_id`。

## 时间轴与官方标注

- manifest 保存每个视频的 `fps_num/fps_den/fps`、帧数、时长和画幅；转换器按 `video_id` 使用逐视频有理 FPS。
- Coffee/Lecture/Office 使用视频实际 25 FPS，不再沿用 Home 的约 24 FPS。
- LE2I 130 份官方 TXT 的结构化复核结果为：99 个官方跌倒窗口、31 个明确 `0/0` 无跌倒窗口、0 个 bbox-only 文件。
- Lecture room 与 Office 共 60 个视频没有官方 TXT，官方事件导入器明确排除，不进入官方有监督事件指标。
- Home_02 保留原始 `video (31)` 至 `video (60)` 编号。

官方 LE2I 候选位于 `data/annotations/fall_risk/generated/v1/le2i_official/`，事件文件 SHA-256 为 `d69ad426a818e6e18d0105aa0d365b28a5f0794fd5f4a2500b8901144b5dc28a`。其 `label_source=le2i_txt`，不会覆盖 CVAT 人工来源。

## 标签审计

根目录现有 `action_labels.jsonl` 和 `event_labels.jsonl` 各 922 条，均为 `pending`；动作标签的 `subject_id` 全为 `unknown`。动作分布中 C/D 高风险动作共 204 条，`C03/C05` 为 0，`C04` 仅 1 条。

根标签当前不是 v1 正式输入。严格 audit 报告为 2,766 个 schema 错误：1,844 次缺失新来源链/资格字段，922 次存在旧字段。输入哈希保持为：

| 文件 | SHA-256 |
|---|---|
| `action_labels.jsonl` | `e72eeed3067d8c1bc75ddca732deffc5a5defefe37fbd618e9ba2e05bccc885a` |
| `event_labels.jsonl` | `a200a39b7119d7468232e487ef45c0a4e8dc728c2dc0610a15add1dbe64bede9` |
| `risk_labels.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `annotation_review_log.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `subject_profiles.json` | `aa92c571bf1316e58af81866c6a3028b99f7ad3d8c03686f7333a513ec68c8f2` |

`quarantine/` 另有 Home_02 缺来源链的 action/event 各 133 条；它们只作为待找回原始导出的证据，不属于训练或评估输入。

候选重导没有替换根标签：

| 候选 | action / event | schema errors | 人审 blocker | action SHA / event SHA |
|---|---:|---:|---:|---|
| CVAT Home_01 | 8 / 8 | 0 | 6 | `09d168a9...e509` / `2fb62960...35d5` |
| `cvat_coffee_01_02` 源导出 | 514 / 514 | 0 | 211 | `feabb732...2741` / `5f4adcce...2883` |
| LE2I official | 0 / 99 | 0 | 99 | 空文件 / `d69ad426...2c28` |

目录名按源导出文件命名，但该 514 条导出实际覆盖 `Coffee_room_01=233`、`Coffee_room_02=150`、`Home_02=131` 条动作，共关联 100 个视频；不能把整组简称为纯 Coffee 数据。Coffee 使用 25 FPS，Home_02 使用各自约 24 FPS 的 manifest 时间轴。

blocker 来自尚无双人独立复核的高风险/跌倒候选。`risk_labels.jsonl` 与 `annotation_review_log.jsonl` 是空模板，`subject_profiles.json` 是零 subject 模板；没有生成任何虚构风险标签、人员档案或签字记录。

## 身份与许可风险

两个受 Git 跟踪的 CVAT 原始导出和一个未跟踪导出均检测到非空用户名/邮箱类元数据；本报告不记录具体值。转换器确认这些字段未复制到候选标签，但原件的脱敏、移出版本库或受控留存仍需项目负责人决定。

原始导出校验和：

- `annotations.xml`：`117e1e13d4c18a5d56f550403fb1524c72cdabf4b1ccc05b45c7d0a04a0611dd`
- Home_01 ZIP：`24c06453391ac8f856fde9f2ca45353a754130874093d7ac9965242ee74dbdbf`
- Coffee ZIP：`130a09aecc8ef68fbbccda2f5341280387857967afd17c626e1a76253cfeac1c`

## Split 与评估分母

四类任务均生成了真实 blocked artifact，`eligible_sample_count=0`、`split_id=null`，没有制造空的 ready/frozen split。当前 manifest 中 `continuous_monitoring_eligible=true` 为 0，合法连续监控时长为 0；因此正式 `FP/摄像机小时`、`FP/家庭日` 和传统 FPR 均没有合法分母。

评估器只使用 `eligibility=true` 的真值。`reports/fall_risk/workflow_a_synthetic_evaluation/` 已跑通一条合成 perfect-match 链路；其中 `F1=1.0` 仅为基础设施烟测，不是模型性能或比赛结果。

## 复现

```bash
conda run -n eldercare-ai python scripts/annotation/build_fall_risk_manifest.py \
  --repo-root . \
  --output /tmp/fall_risk_video_manifest_rebuild.jsonl \
  --ffprobe-bin ffprobe

conda run -n eldercare-ai python scripts/annotation/validate_fall_risk_labels.py \
  --manifest data/manifests/fall_risk_video_manifest.jsonl \
  --action-labels data/annotations/fall_risk/action_labels.jsonl \
  --event-labels data/annotations/fall_risk/event_labels.jsonl \
  --risk-labels data/annotations/fall_risk/risk_labels.jsonl \
  --subject-profiles data/annotations/fall_risk/subject_profiles.json \
  --review-log data/annotations/fall_risk/annotation_review_log.jsonl \
  --config configs/data/fall_risk_label_validation_v1.yaml \
  --mode audit \
  --report-output /tmp/fall_risk_label_validation_audit.json
```

默认写入拒绝覆盖已有 manifest 和报告。需要验证确定性时，应输出到新的临时路径并比较 SHA-256，不要覆盖候选版本。
