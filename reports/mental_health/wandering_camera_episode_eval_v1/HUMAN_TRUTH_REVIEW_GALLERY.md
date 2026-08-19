# M0-CAM-EP1B truth-free 轨迹复核画廊

日期：2026-08-16

本文件把原视频、抽帧 contact sheet、模型检测/跟踪得到的 truth-free 轨迹图和底层 tracking JSONL 一对一连接起来。

> **证据边界：** 下列轨迹图只包含 detector/tracker 提取的目标轨迹、时间顺序以及起终点。它们不包含人工 truth、AI 预标注、四分类预测或 binary 预测，不能单独作为 `direct/pacing/lapping/random` 真值。人工复核应先看完整视频，再看 contact sheet 和轨迹图，最后才查看 AI 预标注或模型预测。

原视频根目录（本机 WSL 路径）：`/mnt/d/徘徊数据集/自采数据/剪辑/`

## 一对一索引

| 原视频 `media_ref` | `review_id` | contact sheet | truth-free 轨迹 | tracking JSONL | 当前用途 | AI 预标注（非 truth） |
| --- | --- | --- | --- | --- | --- | --- |
| `P01/01.mp4` | `p01-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/p01/01/tracking.jsonl) | 待人工复核 | pacing |
| `P01/02.mp4` | `p01-02` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_02_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-02.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/p01/02/tracking.jsonl) | 待人工复核；track fragmented | pacing |
| `P01/03.mp4` | `p01-03` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_03_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-03.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/p01/03/tracking.jsonl) | 待人工复核 | pacing |
| `P01/04.mp4` | `p01-04` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_04_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-04.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/p01/04/tracking.jsonl) | 待人工复核 | pacing |
| `L01/01.mp4` | `l01-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/L01_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/l01-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/l01/01/tracking.jsonl) | 待人工复核 | lapping |
| `L01/02.mp4` | `l01-02` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/L01_02_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/l01-02.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/l01/02/tracking.jsonl) | 待人工复核 | lapping |
| `L01/03.mp4` | `l01-03` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/L01_03_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/l01-03.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/l01/03/tracking.jsonl) | 待人工复核 | lapping |
| `R01/01.mp4` | `r01-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/R01_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/r01-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/r01/01/tracking.jsonl) | 待人工复核 | random |
| `R01/02.mp4` | `r01-02` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/R01_02_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/r01-02.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/r01/02/tracking.jsonl) | 待人工复核 | random |
| `R01/03.mp4` | `r01-03` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/R01_03_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/r01-03.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/r01/03/tracking.jsonl) | 待人工复核 | random |
| `H01/01.mp4` | `h01-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H01_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h01-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h01/01/tracking.jsonl) | 待人工复核；purposeful hard negative | lapping + phone call |
| `H01/02.mp4` | `h01-02` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H01_02_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h01-02.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h01/02/tracking.jsonl) | 待人工复核；purposeful hard negative | lapping + phone call |
| `H02/01.mp4` | `h02-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H02_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h02-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h02/01/tracking.jsonl) | 已提取，当前未评分 | 无 |
| `H02/02.mp4` | `h02-02` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H02_02_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h02-02.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h02/02/tracking.jsonl) | 已提取，当前未评分 | 无 |
| `H02/03.mp4` | `h02-03` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H02_03_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h02-03.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h02/03/tracking.jsonl) | 已提取，当前未评分 | 无 |
| `H03/01.mp4` | `h03-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H03_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h03-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h03/01/tracking.jsonl) | 已提取，当前未评分 | 无 |
| `H04/01.mp4` | `h04-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H04_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h04-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h04/01/tracking.jsonl) | 待人工复核；purposeful hard negative | pacing + carrying |
| `H04/02.mp4` | `h04-02` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H04_02_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h04-02.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h04/02/tracking.jsonl) | 待人工复核；purposeful hard negative | pacing + carrying |
| `H05/01.mp4` | `h05-01` | [查看](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H05_01_contact.jpg) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h05-01.png) | [查看](../../../tmp/m0cam_ep1b_inputs_20260815_v1/tracking/h05/01/tracking.jsonl) | 待人工复核；purposeful hard negative | pacing + exercise |

## 待人工复核的 15 条

### `P01/01.mp4` (`p01-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![P01/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_01_contact.jpg) | ![P01/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-01.png) |

### `P01/02.mp4` (`p01-02`)

该视频的 tracker 产生 `track 1` 和 `track 2`，当前 QC 为 fragmented；轨迹图与原 tracking 保持一致，不人工拼接。

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![P01/02.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_02_contact.jpg) | ![P01/02.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-02.png) |

### `P01/03.mp4` (`p01-03`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![P01/03.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_03_contact.jpg) | ![P01/03.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-03.png) |

### `P01/04.mp4` (`p01-04`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![P01/04.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/P01_04_contact.jpg) | ![P01/04.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/p01-04.png) |

### `L01/01.mp4` (`l01-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![L01/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/L01_01_contact.jpg) | ![L01/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/l01-01.png) |

### `L01/02.mp4` (`l01-02`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![L01/02.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/L01_02_contact.jpg) | ![L01/02.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/l01-02.png) |

### `L01/03.mp4` (`l01-03`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![L01/03.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/L01_03_contact.jpg) | ![L01/03.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/l01-03.png) |

### `R01/01.mp4` (`r01-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![R01/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/R01_01_contact.jpg) | ![R01/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/r01-01.png) |

### `R01/02.mp4` (`r01-02`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![R01/02.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/R01_02_contact.jpg) | ![R01/02.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/r01-02.png) |

### `R01/03.mp4` (`r01-03`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![R01/03.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/R01_03_contact.jpg) | ![R01/03.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/r01-03.png) |

### `H01/01.mp4` (`h01-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H01/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H01_01_contact.jpg) | ![H01/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h01-01.png) |

### `H01/02.mp4` (`h01-02`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H01/02.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H01_02_contact.jpg) | ![H01/02.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h01-02.png) |

### `H04/01.mp4` (`h04-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H04/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H04_01_contact.jpg) | ![H04/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h04-01.png) |

### `H04/02.mp4` (`h04-02`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H04/02.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H04_02_contact.jpg) | ![H04/02.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h04-02.png) |

### `H05/01.mp4` (`h05-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H05/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H05_01_contact.jpg) | ![H05/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h05-01.png) |

## 已提取但当前未评分的 4 条

H02/H03 包含混合活动，当前没有为了凑标签而生成 AI shape 预标注，也没有进入 provisional diagnostic。保留轨迹图供后续独立 episode 边界复核。

### `H02/01.mp4` (`h02-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H02/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H02_01_contact.jpg) | ![H02/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h02-01.png) |

### `H02/02.mp4` (`h02-02`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H02/02.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H02_02_contact.jpg) | ![H02/02.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h02-02.png) |

### `H02/03.mp4` (`h02-03`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H02/03.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H02_03_contact.jpg) | ![H02/03.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h02-03.png) |

### `H03/01.mp4` (`h03-01`)

| 20 帧 contact sheet | truth-free target track |
| --- | --- |
| ![H03/01.mp4 contact sheet](../../../tmp/m0cam_ep1b_visual_review_20260815_v1/H03_01_contact.jpg) | ![H03/01.mp4 truth-free track](../../../tmp/m0cam_ep1b_inputs_20260815_v1/truth_free_track_plots/h03-01.png) |

## 人工复核记录位置

人工 boundary、shape、purpose 和可判性应填写到独立 reviewed truth view；不要修改本画廊、原 XML、既有预标注或 prediction bundle 来表达裁决。复核步骤和当前 15 条队列见 [HUMAN_TRUTH_REVIEW.md](HUMAN_TRUTH_REVIEW.md)。
