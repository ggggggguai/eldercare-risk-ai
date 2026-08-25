# W5D-03B 验证记录

日期：2026-08-22

## 环境

- WSL：Ubuntu-22.04
- conda：`eldercare-ai`
- editable package：`/mnt/c/Users/lenovo/Desktop/心理算法`

## 自动化测试

```bash
conda run -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_episode_import.py \
  tests/test_wandering_camera_cvat_project.py \
  tests/test_wandering_camera_home_smoke.py \
  tests/test_wandering_camera_episode_pipeline.py \
  tests/test_wandering_camera_context_review.py -q
```

结果：`31 passed in 24.24s`。

全 wandering camera 回归：

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_camera*.py -q
```

结果：`667 passed, 1 warning in 217.39s`。唯一 warning 来自 portability 攻击测试故意构造
重复 ZIP entry，不是本次实现失败。

## 真实原视频复放

```bash
conda run -n eldercare-ai python scripts/wandering/run_camera_home_smoke.py \
  --input-video /mnt/c/Users/lenovo/Desktop/测试文件夹/wand_R01_hall_1.mp4 \
  --output-dir tmp/wandering_home_w5d03b_20260821_v1/home_smoke_r01_hall1_v2 \
  --run-id w5d03b-home-r01-hall1-20260821-v2 \
  --source-video-id wand_R01_hall_1 \
  --source-group-id HOME-W5D03B-20260821 \
  --device-id HOME-CAMERA-ANON-01 \
  --setup-id hall_1 \
  --stream-epoch full_recording \
  --media-ref external_home/wand_R01_hall_1.mp4 \
  --participant-id P-HOME-01 \
  --session-id S-HOME-CAPTURE-TIME-UNAVAILABLE-01 \
  --clock-domain-id CLOCK-HOME-CAPTURE-TIME-UNAVAILABLE \
  --home-annotation-status available \
  --timezone Asia/Shanghai
```

结果：`status=ready`，487 条 tracking、1 条 proposal、1 条 episode、1 条 context；
tracking mode 为 `fresh_full_mp4_yolov8_bytetrack`，human truth count 为 0。
最终目录无 `shape_batch_index.jsonl`，文本产物中未检出 staging `.tmp-` 路径。

## 产物校验

仓库自带 episode/context 验证器和独立哈希复算结果：

```json
{"frame_artifacts_verified":6,"manifest_artifacts_verified":5,"schema_rows_validated":{"context_reviews.jsonl":1,"episode_results.jsonl":1},"source_video_sha256_verified":true}
```

中间 scene 与 crop 已做一次 AI 可视抽查，均能解码且 crop 包含目标人物。该抽查不替代
负责人/人工视觉验收。

## 标注裁决与导入

- 裁决前 XML SHA-256：`94ad9aee2169e6f0960cb6b740a7b62ad4f0d4f1ea780f9b4235c23a1b911137`
- 裁决后 XML SHA-256：`8aa9911c3b542aee8b9c0ae9eaa2ccdd3cabbed83715dcebb2986f55b3103327`
- 备份：`annotations.pre_track108_adjudication_20260822.sha256-94ad9aee.xml`
- track 108：50/50 重复属性均为 `direct + ordinary_negative`

```bash
conda run -n eldercare-ai python scripts/wandering/import_wandering_cvat_project.py \
  --cvat-xml /mnt/c/Users/lenovo/Desktop/测试文件夹/wandering标注/annotations.xml \
  --video-root /mnt/c/Users/lenovo/Desktop/测试文件夹 \
  --tracking-output-root /mnt/c/Users/lenovo/Desktop/测试文件夹/output \
  --output-dir tmp/wandering_home_w5d03b_20260822_v3/cvat_import \
  --participant-id P-HOME-01 \
  --session-id S-HOME-CVAT-20260822 \
  --clock-domain-id CLOCK-HOME-CAPTURE-TIME-UNAVAILABLE
```

结果：70 task、135 episode、134 temporal-alignment pass、1 low coverage、134 geometry mismatch。
v3 为 fresh labeled-scope proposals；v1/v2 的两次 evaluator 拒绝分别保留
`proposal summary validation scope differs from media` 与
`independent human boundary requires labeled-evaluation authorization`，没有覆盖旧证据。

## Development evaluator

```bash
conda run -n eldercare-ai python scripts/wandering/evaluate_camera_episode_boundaries.py \
  --batch-index tmp/wandering_home_w5d03b_20260822_v3/cvat_import/boundary_evaluation_batch_index.jsonl \
  --output-dir tmp/wandering_home_w5d03b_20260822_v3/boundary_evaluation

conda run -n eldercare-ai python scripts/wandering/run_camera_episode_evaluation.py \
  --batch-index tmp/wandering_home_w5d03b_20260822_v3/oracle_shape_review/oracle_evaluation_batch_index.jsonl \
  --output-dir tmp/wandering_home_w5d03b_20260822_v3/oracle_shape_evaluation_replay
```

boundary evaluator 的 8 个产物与独立 replay 逐文件 SHA-256 相同；oracle-shape evaluator 的
6 个产物也与 replay 逐文件 SHA-256 相同。关键哈希：

- boundary `metrics.json`：`6faeb8f1ecfaf0c2dc187af5ec6167b6c036fbde025138c298c1da4b09c75f2a`
- boundary `summary.json`：`5b54a57c2cae830943fa6389e392d89212e94a3d98c65ac4fb1891578d5eaae6`
- oracle `metrics.json`：`181f6b8843694b0cd75dc1a936000511f2f2432e8afbc5ca26275385fe982e77`
- oracle `summary.json`：`c27a2cd32c4370465736848f74a3a44c143514ac76a3689da9b597ba7a1b20f2`
