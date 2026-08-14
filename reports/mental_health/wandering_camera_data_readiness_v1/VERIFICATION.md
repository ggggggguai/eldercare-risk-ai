# M0-CAM-RD verification

## 环境与初始边界

- Git HEAD：`b258748c14dec2d7801b89a23f3d161d5795e2ff`；初始 staged 集合为空。
- Windows PATH 和常见安装位置没有原生 Conda；所有 Python 命令均从当前 Windows checkout 调用现有 WSL `Ubuntu-22.04` 的 `eldercare-ai`，没有创建、复制或使用另一份 WSL worktree。
- `pip show elderly-monitoring-algorithms` 的 editable location 为当前项目根（终端显示中文路径时有编码噪声）；报告不固化操作者用户名或本机绝对路径。
- 用户原有 `.gitattributes`、Step10 README 和 performance report/artifact 脏改动均保留；未 stash/reset/覆盖/commit/push。

## 测试

聚焦 RD-A/B/C：

```text
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_development.py -q
22 passed
```

新工具与全部既有 camera adapter/QC/inference/primary/episode/corruption 回归：

```text
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_dataset.py \
  tests/test_wandering_camera_development.py \
  tests/test_wandering_camera_adapter.py \
  tests/test_wandering_camera_qc.py \
  tests/test_wandering_camera_inference.py \
  tests/test_wandering_camera_primary_inference.py \
  tests/test_wandering_camera_episode.py \
  tests/test_wandering_camera_corruption.py -q
105 passed
```

覆盖点包括 receipt approved/active/有效期/operation/scope、隐私与绝对路径拒绝、collection 跨引用和分组隔离、完整第 10.3 节标签轴、unknown/uncertain/excluded、purposeful pacing 不改 shape、fresh/atomic/refuse-overwrite、production synthetic 拒绝、receipt 失败零下游调用、私有 synthetic 控制流顺序、一对一 matching tie-break、raw confusion/counts、coverage、区间和 person-hour 分母、显式 policy 与 stage timing。既有测试 `test_primary_entry_rejects_non_synthetic_authorization_before_loader_tracking_or_forward` 继续通过。

## Fresh synthetic artifact

执行入口：

```text
conda run --no-capture-output -n eldercare-ai python \
  scripts/wandering/prepare_camera_session.py \
  --project-root . \
  --tracking reports/mental_health/wandering_m0cam_engineering_v1/fixtures/synthetic_tracking.jsonl \
  --media-sidecar reports/mental_health/wandering_m0cam_engineering_v1/fixtures/synthetic_media_sidecar.json \
  --output-dir reports/mental_health/wandering_camera_data_readiness_v1/artifacts/synthetic_input_readiness_v1
```

结果：240 条 observation；只 canonicalize/validate tracking+sidecar；`media_opened=false`、`detector_run=false`、`tracker_run=false`。三个输出的 bytes/SHA 均已独立复算并写入 `artifact_manifest.json`。没有 authorization receipt；`preparation_summary.json` 明确只是输入规范化摘要，不是 C0 receipt。

## 身份与未触碰项

- 固定 candidate manifest：12642 bytes，SHA-256 `3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7`；本轮未修改、未加载、未 forward。
- 原 primary config/CLI、`model.py`、`release.py` 和既有 camera adapter/QC/inference/episode 文件未修改。
- 没有真人 camera、WP raw、SmartCare official/raw 或 sealed camera 数据访问；没有 M0-CAM-D、训练、调参、阈值选择、模型/配置/标签/split 变更。
- 最终必要验收包括新/既有 camera tests、CLI help、JSON/JSONL/descriptor 复算、文档链接检查和 `git diff --check`。本机机器结果另存于受 `.gitignore` 管理的 `verification.json`，关键结果已摘要在本文件与 README 中。
