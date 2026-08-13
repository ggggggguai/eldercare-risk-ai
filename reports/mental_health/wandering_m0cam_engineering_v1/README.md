# TopoWander-MPT M0-CAM-E primary camera 工程闭环与 M0-CAM-H 加固

更新时间：2026-08-13

## 结论

本任务已完成，统一状态为：

```text
wandering_m0cam_primary_camera_engineering_ready
evidence_scope=synthetic_contract_only
hardening_complete=true
```

固定的 TopoWander-MPT primary candidate 已通过独立 wrapper 接入既有 Step7 bbox tracklet、Camera QC 和 prepared-array 路径。实现只使用 manifest-bound loader 加载 `seed=20260731 / best epoch=5` 候选，对 `ready` camera window 直接调用 `runtime.model`；没有调用 WP 专用 `CandidateRuntime.predict_logits()`，没有使用 WP prefix padding，也没有修改候选、权重、forward/performance config、类别顺序、阈值、split 或 Step7 comparison-only 链。

M0-CAM-H 已完成三项轻量加固：primary 入口在 candidate loader 和 tracking 读取前只接受 `synthetic_fixture / synthetic_camera_contract`；binary window 与 episode validator 统一使用 `p(wandering_like) >= 0.5`，并校验 manifest 的 `sigmoid >= 0.5` 契约；在 candidate loader 前核验当前实际导入的 `model.py`、`release.py` 路径及其 v3 training/release size/SHA 身份。该代号不新增证据等级或发布状态。

本报告只证明 synthetic camera contract 下的工程接线、失败闭锁、制品与延迟烟测。它不是目标摄像头、真人、老人域、产品或临床有效性证据，也不提供 camera accuracy、F1、recall、FAR 或校准结论。

## 固定候选

- candidate ID：`topowander-m0s-seed20260731-epoch0005`
- candidate manifest SHA-256：`3a1e56c37b9b43e340dcd67a3163454da1d37f24b01d8677935f63a069d97ac7`
- model state SHA-256：`94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031`
- binary threshold：`0.5`
- four-class order：`direct / pacing / lapping / random`
- subtype order：`pacing / lapping / random`
- 概率状态：`probability_calibrated=false`

## 新增主路径

- 配置：`configs/modules/wandering_camera_primary_v1.yaml`
- 逐窗主推理：`camera_primary_inference.py`
- 最小 episode candidate：`camera_episode.py`
- CLI：`scripts/wandering/run_topowander_camera_inference.py`
- 单测：`test_wandering_camera_primary_inference.py`、`test_wandering_camera_episode.py`

主路径复用：

1. 复用 sidecar validator 完成 synthetic-only 预检；
2. 用 v3 manifest 完成 active `model.py/release.py` 身份预检；
3. `load_candidate_from_manifest()`；
4. `load_camera_inputs()`、`run_camera_qc()` 与 `prepare_camera_window()`；
5. 直接 primary forward 与 `hierarchical_four_class_probabilities()`；
6. `aggregate_episode_candidates()`。

Step7 的 `run_camera_inference.py`、RF/TCN 四组独立输出、`comparison_only` 配置与报告均保持不变。

## fresh synthetic E2E

当前加固证据目录为 `artifacts/synthetic_e2e_v4/`。输入是 240 条 `synthetic_fixture` tracking rows：两条运动 track 和一条静止 track；没有真人媒体。

结果：

- observation：240；
- window：3；
- ready：2；
- unavailable：1，原因为 `insufficient_motion`；
- inference_error：0；
- model forward invocation：2，证明 unavailable window 为零调用；
- episode candidate：2，分别属于 track 1 与 track 3，没有跨 tracklet 合并；
- episode policy：`development_unfrozen`；
- merge gap：本次 synthetic test 显式传入 `0.0 s`，不是冻结策略；
- alert decision：`null`；
- CPU threads：观测 `intra/inter=8/1`；
- observed batch sizes：`[1, 1]`，maximum batch policy 为 64；
- model-forward-only latency：p50 `93.341036 ms`，p95 `137.4264038 ms`；这只是工程延迟，不是产品阈值。
- active source identity：candidate manifest、`model.py`、`release.py` 的 expected/observed size/SHA 全部一致，且记录 `verified_before_candidate_loader=true`。

v4 run manifest SHA-256：

```text
e90b976fbc88bfd72b0c33488e2e5e769a69a50298c0cb30ad98d4941cd6c1a6
```

`synthetic_e2e_v1/`、`synthetic_e2e_v2/` 与 `synthetic_e2e_v3/` 均按不覆盖原则保留为历史中间证据；三者在本次加固前后的文件数与组合哈希保持不变，现行加固结论只引用 v4。

## 输出契约

v4 bundle 包含：

- `media_sidecar.json`
- `tracking_input.jsonl`
- `bbox_tracklets.jsonl`
- `window_records.jsonl`
- `predictions.jsonl`
- `episode_candidates.jsonl`
- `qc_summary.json`
- `model_bindings.json`
- `execution.json`
- `manifest.json`

manifest 对其余 9 个文件记录 byte count 与 SHA-256。独立检查确认 9/9 一致，JSON/JSONL 均可解析，binary/subtype/four-class 概率可相互复算。

## 数据与科学边界

- 未访问真实人体媒体、WP raw、WP public holdout、SmartCare official/raw 或 sealed camera；
- 未执行 M0-CAM-D、M1、重训、微调、域适配、日级聚合或 `AlgorithmEvent`；
- track ID 只是本段视频中的工程跟踪标识，不是真实身份；
- episode candidate 不输出 risk level、alert 或业务动作；
- synthetic pattern 输出没有真值分母，不得解释为 camera 性能。

## 下一步

下一任务是 M0-CAM-D：等待 C0 授权和 C1/C2/C3 合法 camera development tracking、sidecar、annotation 与分组数据。随后在目标机位 development 数据上验证轨迹/QC coverage、逐窗置信度、purposeful hard negatives、shape episode、eligible negative person-hours、延迟与域差异，并仅形成 provisional 的 uncertain/episode policy。独立泛化结论继续留给 C4 sealed camera。

详细核验见 [VERIFICATION.md](VERIFICATION.md)。
