# M0-CAM-EP1A oracle-boundary episode inference

日期：2026-08-15

状态：`m0cam_ep1a_status=completed`

证据范围：`authorized_development_smoke`

成绩名称：**oracle-boundary shape classification**

## 实现

本阶段把已经跑通的 D01/S00 临时路径提升为正式、truth-separated episode 入口：

- `wandering_camera_episode_v1.yaml` 固定复用旧 Camera QC 的 0.5 秒 bucket、短缺口、运动范围和同一 80×14 输入语义，不修改 legacy 40 秒配置；
- `camera_episode_import.py` 支持 CVAT for video 1.1 XML 和简化 boundary JSONL。CVAT `outside=1` 作为 exclusive end；视频末尾无 outside 时使用最后可见帧后一帧；CVAT track ID 与机器 `target_track_id` 分层；
- importer 分开写 `episode_boundaries.jsonl` 与 `episode_truth.jsonl`。推理入口只读取 exact truth-free boundary，拒绝 `observable_pattern/purpose/evaluation_role` 等真值字段；
- `camera_episode_inference.py` 支持 boundary JSONL 与显式 `target_track_id` 的 whole-clip 短片。不会静默选择最长 track；technical track break、长 gap、低运动和不足覆盖返回 `unavailable`，boundary 不可靠返回 `boundary_uncertain`，合法输入 forward 失败返回 `inference_error`；
- episode 内复用 bbox-bottom、置信度加权 bucket、bbox-height compensation、短缺口质量 0.5、`prepare_camera_window()`、train-only feature stats 和原 frozen TopoWander forward；`camera_qc.py` 保持原字节不变，避免重锚旧 corruption trust root；
- 结果保留原 start/end/duration、observation/bucket 数、observed/interpolated coverage、maximum gap、QC、binary/subtype/four-class 概率、candidate/model identity 和 `probability_calibrated=false`。

新增正式入口：

```text
scripts/wandering/import_wandering_cvat_episodes.py
scripts/wandering/run_camera_episode_inference.py
```

## 可复制命令

以下命令从仓库根目录、在 WSL 的项目环境中执行。

S00 CVAT 导入：

```bash
conda run -n eldercare-ai python scripts/wandering/import_wandering_cvat_episodes.py \
  --cvat-xml '/mnt/d/徘徊数据集/自采数据/剪辑/S00/annotations.xml' \
  --media-sidecar tmp/wandering_s00_smoke_20260815_run1/tracking/media_sidecar.json \
  --target-track-id 1 \
  --output-dir tmp/m0cam_ep1a_formal_20260815_v2/s00_import
```

S00 episode inference：

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_episode_inference.py \
  --tracking-jsonl tmp/wandering_s00_smoke_20260815_run1/tracking/tracking.jsonl \
  --media-sidecar tmp/wandering_s00_smoke_20260815_run1/tracking/media_sidecar.json \
  --episode-boundaries tmp/m0cam_ep1a_formal_20260815_v2/s00_import/episode_boundaries.jsonl \
  --output-dir tmp/m0cam_ep1a_formal_20260815_v2/s00_inference
```

D01 12 条 whole-clip：

```bash
for i in {01..12}; do
  conda run -n eldercare-ai python scripts/wandering/run_camera_episode_inference.py \
    --tracking-jsonl "tmp/wandering_d01_smoke_20260815_run1/tracking/$i/tracking.jsonl" \
    --media-sidecar "tmp/wandering_d01_smoke_20260815_run1/tracking/$i/media_sidecar.json" \
    --whole-clip \
    --episode-id "d01-$i-whole-clip" \
    --target-track-id 1 \
    --output-dir "tmp/m0cam_ep1a_formal_20260815_v2/d01/$i"
done
```

所有结果目录均拒绝覆盖。`tmp/m0cam_ep1a_formal_20260815_v2/` 为最终源码重新生成的 Git 忽略本机 development 输出；原视频没有复制进仓库。它与恢复 `camera_qc.py` 原字节前的 `v1` 目录逐文件一致（`diff -rq` 退出 0）。

## D01 实际结果

输入是负责人提供的 12 条短 direct 视频所产生、并由现有 adapter 回读的 tracking/sidecar。每条文件由 operator 明确声明为一个 whole-clip episode，目标 track 固定为 1。

| 项目 | 结果 |
| --- | ---: |
| episode | 12 |
| ready | 12 |
| unavailable / boundary_uncertain / inference_error | 0 / 0 / 0 |
| predicted direct | 12 |
| duration range | 4.4667–10.8 秒 |
| source observation range | 67–162 |
| source bucket range | 9–22 |
| direct probability range | 0.639814–0.997416 |
| 与旧临时脚本 direct probability 最大差 | 0 |

D01-08 的 tracking 同时含少量 track 2 observation；正式入口显式选择 track 1，只消费 93 条 track 1 observation，没有把 track 2 拼接进 episode。

## S00 实际结果

CVAT XML 导出包含三个 `wandering_episode` track；importer 没有读取模型结果创建或修改标签。

| CVAT track | boundary（秒） | duration | QC / status | XML shape | prediction | P(direct) |
| --- | --- | ---: | --- | --- | --- | ---: |
| 0 | 0.0000–18.6000 | 18.6000 | ready / ready | direct | direct | 0.916472 |
| 1 | 18.6000–39.7333 | 21.1333 | ready / ready | direct | direct | 0.981494 |
| 2 | 39.7333–44.7333 | 5.0000 | ready / ready | direct | direct | 0.996004 |

三段的四类概率与旧 `tmp/run_s00_xml_episode_model.py` 诊断逐条一致，最大差为 0。

旧 40 秒路径继续单独保留：

| 路径 | 区间 | prediction | 相关概率 |
| --- | --- | --- | ---: |
| legacy long-context diagnostic | 0–40 秒 | lapping | P(lapping)=0.997431 |

这个 legacy 结果描述三个 direct 组合后的长上下文轨迹，不是单个 episode 的真值对齐结果，也不计为三条 direct 的分类错误。

## 自动化证据

详细命令见 [VERIFICATION.md](VERIFICATION.md)。本轮通过：

- EP1A inference/importer 聚焦：`14 passed`；
- EP1A 加旧 corruption trust-root 聚焦复验：`26 passed`；
- adapter/QC/comparison inference/fixed primary/legacy episode：`105 passed`；
- authorized development/dataset：`80 passed`；
- preprocessing/model：`43 passed, 73 subtests passed`，仅有现存 CUDA driver warning，CPU 测试通过；
- 全部 `test_wandering_camera_*.py`：`525 passed`，唯一 warning 来自 PORTABLE 重复 ZIP 条目攻击测试。

## 不能宣称的结论

- 这不是 automatic/semiautomatic episode boundary；
- 这不是连续视频端到端徘徊识别准确率；
- D01/S00 不是跨 participant/setup 的正式 camera test，不能据此报告 95%；
- 当前没有 pacing/lapping/random 的足量 camera episode，不能计算有意义的四类 camera F1；
- 没有自然负例 person-hour，不能报告 shape-candidate false activations/hour，更不能报告最终 alert FAR；
- 概率未校准；没有 OOD/uncertain 阈值、目的/上下文决策、真实老人或临床证据；
- 没有输出风险、告警或 `AlgorithmEvent`；没有接入日级产品主链。

## 下一数据与唯一优先级

下一批最值得补拍的是自然完整的 pacing、lapping、random episode，各约 10 个；同时保留打电话、找东西、清洁、锻炼和搬运的 purposeful hard negatives。episode 按自然完成结束，不强凑 40 秒。

唯一下一优先级：`M0-CAM-EP1B`，用上述四类完整 episode 计算 oracle-boundary confusion、precision/recall/F1/support、QC coverage 与短/中/长时长分层。完成后再开始 EP2 automatic boundary。
