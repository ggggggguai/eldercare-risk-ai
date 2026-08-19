# M0-CAM-EP1B 人工 truth 复核队列

日期：2026-08-16

状态：`pending_human_review`

本文件是复核清单，不是真值文件。下面的 shape/purpose 是执行 AI 生成的预标注，只用于复核后的差异记录；人工标注者第一遍不得查看模型预测，也不应先采用这些建议。

原视频、contact sheet、truth-free 轨迹图与 tracking JSONL 的一对一入口见 [HUMAN_TRUTH_REVIEW_GALLERY.md](HUMAN_TRUTH_REVIEW_GALLERY.md)。其中还列出了已提取但当前未评分的 H02/H03 视频。

## 复核顺序

1. 先看完整原视频，隐藏 `episode_predictions.jsonl`、evaluation failures 和模型概率。
2. 独立记录目标人物、episode 起止、`observable_pattern`、`purpose_context/evidence`、质量问题和 `accepted/uncertain/excluded`。
3. 再查看 truth-free 轨迹图，确认画面透视或 tracking 是否改变了对轨迹的理解；仍不查看模型预测。
4. 最后才与 AI 预标注对照。相同则确认，不同则保留人工判断和原因，不根据模型预测折中。
5. 不修改现有 XML 或 `tmp/.../truth.jsonl`。在新的 reviewed truth 目录写入人工版本，并保留 reviewer 与复核日期。

## Episode 协议

- 大约连续迈出三步、行走已经成立时开始；入场、站位和准备动作不计入。
- 数秒停顿、犹豫、转向、折返和闭环仍属于同一 locomotion episode。
- 同一地点持续停留约 15 秒、坐下或明确开始另一活动时结束。
- track end、出画、长 gap、ID switch 和 session end 是硬边界。
- 若 A 到 B 后立即返回且没有可观察停留或任务切换，不为了得到两个 direct 强行切开。
- 边界或形态证据不足时使用 `unknown/uncertain`，不为补类别猜测。

## Shape 与 purpose

- `direct`：总体从一个位置到另一个位置，没有稳定往返轴、重复闭环或持续多向低效路径。
- `pacing`：在相似端点或带状区域重复折返。采集 clean sample 时可要求约 3 次清楚反转，但这不是标签阈值。
- `lapping`：闭合或近闭合绕行并回访先前区域。采集 clean sample 时可要求约 2 圈，但这不是标签阈值。
- `random`：持续多方向、不规则、存在回访，但没有稳定往返轴或重复闭环。
- 起终点接近只能作为证据之一，不能单独决定 lapping 或 wandering-like。
- purpose 独立填写。打电话、搬运、锻炼时形成的 pacing/lapping/random 不改成 direct。

## 待复核条目

| source | duration_sec | AI preannotation | AI purpose | 人工 boundary | 人工 shape | 人工 purpose | 人工状态/备注 |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| P01-01 | 36.73 | pacing | unknown | 待填 | 待填 | 待填 | 当前图像平面呈宽回环，重点核对是否真有稳定往返轴 |
| P01-02 | 37.40 | pacing | unknown | 待填 | 待填 | 待填 | track fragmented；边界和可评分性需单独裁决 |
| P01-03 | 44.53 | pacing | unknown | 待填 | 待填 | 待填 | 核对反转次数与端点一致性 |
| P01-04 | 28.20 | pacing | unknown | 待填 | 待填 | 待填 | 核对是否只是曲线路径或单次返回 |
| L01-01 | 58.27 | lapping | unknown | 待填 | 待填 | 待填 | 轨迹图显示重复闭环，仍需完整视频确认 |
| L01-02 | 50.87 | lapping | unknown | 待填 | 待填 | 待填 | 核对闭环是否重复且边界完整 |
| L01-03 | 63.93 | lapping | unknown | 待填 | 待填 | 待填 | 核对是否含活动切换或长停留 |
| R01-01 | 35.47 | random | unknown | 待填 | 待填 | 待填 | 狭小空间复用同一区域，重点排除 pacing/lapping |
| R01-02 | 32.67 | random | unknown | 待填 | 待填 | 待填 | 核对是否有稳定往返轴 |
| R01-03 | 30.13 | random | unknown | 待填 | 待填 | 待填 | 核对多方向性和回访是否足够清楚 |
| H01-01 | 57.80 | lapping | purposeful phone call | 待填 | 待填 | 待填 | shape 与 purpose 分开确认 |
| H01-02 | 51.07 | lapping | purposeful phone call | 待填 | 待填 | 待填 | shape 与 purpose 分开确认 |
| H04-01 | 31.67 | pacing | purposeful carrying | 待填 | 待填 | 待填 | 搬运交互可能形成活动切换，检查是否应分段 |
| H04-02 | 36.60 | pacing | purposeful carrying | 待填 | 待填 | 待填 | 搬运交互可能形成活动切换，检查是否应分段 |
| H05-01 | 81.80 | pacing | purposeful exercise | 待填 | 待填 | 待填 | 检查长片中是否存在持续停留或多个 episode |

## EP1B 复核门

人工复核完成后：

- 新 reviewed truth 必须与原预标注和预测分离保存；
- evaluator 重新连接 reviewed truth 与既有 prediction bundle；
- all-shape-eligible shape-binary 是主要 camera shape 结果；
- 四分类、每类 recall、confusion 和 support 作为 subtype diagnostic；
- 若人工确认后 D/P/L/R 任一类没有清楚支持，EP1B 继续 `data_pending`，只补该类清楚样本；
- 不实现告警，不用 purpose 改 shape，不改冻结模型或 0.5 阈值。
