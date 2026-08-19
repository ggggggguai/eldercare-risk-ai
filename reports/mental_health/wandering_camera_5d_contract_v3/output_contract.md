# 徘徊五日算法交付契约

版本：`m0cam-5d-w5d00-v1`

本契约固定五日闭环的算法侧输入身份、输出文件、统一状态和 source reference。它交付 `module=mental_health` 的徘徊证据，不直接发出 `AlgorithmEvent`，也不实现后端、前端、消息或处置流程。

## 1. 输入与运行身份

- B01 36 条、B02 12 条可用视频统一为 `dataset_role=development`，允许反复分段、边界、送模、预处理、binary threshold 和必要的轻量模型开发。
- B01/B02 同属 `P-OFFICE-01`、`C6C-OFFICE-01`；不能据此声明跨人或跨机位泛化。
- B02-0008 的早期 sidecar 保留旧 setup 别名 `office-setup-01`；index 显式记录 `sidecar_setup_id` 和 `setup_binding_status=owner_confirmed_legacy_alias`，不改写历史 sidecar。其他未登记的 setup 漂移失败关闭。
- 两条 frame-check 视频固定排除。每条 index 记录必须绑定并校验 video、tracking、media sidecar、派生 truth 和原始 per-video CVAT XML 的路径及 SHA-256。
- fixed primary 为 `topowander-m0s-seed20260731-epoch0005`，model state SHA-256 为 `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031`。
- camera 输入固定为 80 点、14 个 model feature、2 个 shape coordinate；binary 规则固定为 `sigmoid >= 0.5`。W5D-00 只验证 CPU safe load，不重训模型。
- 居家 MP4 当前登记为 `awaiting_input`。到位后先作真实场景 smoke/验收；若据其继续调参，身份改为 development。

## 2. Fresh 与不覆盖

所有运行写入 `tmp/wandering_camera_5d_runs/<unique_run_id>/` 下的新目录。最终目录必须不存在；producer 使用同文件系统 staging 后原子提交。历史目录、报告和证据均不得覆盖。

固定 handoff 文件名：

```text
episode_results.jsonl
context_reviews.jsonl
daily_reports.jsonl
baseline_profiles.jsonl
baseline_deviations.jsonl
handoff_manifest.json
run_summary.json
README.md
```

空模态也保留对应空 JSONL 和 manifest descriptor，不通过删除文件表达 unavailable。

## 3. 统一状态

| 状态 | 语义 |
|---|---|
| `ready` | 输入、身份和当前层质量条件满足，可进入下游统计 |
| `uncertain` | 结果可保留和送下游候选统计，但必须保留不确定原因，不得冒充人工接受 |
| `unavailable` | 输入、coverage、provider 或 reference 不足；不填伪造零值 |
| `error` | 可定位的工程异常；保留输入身份和错误码 |

具体层可以增加正交状态，例如 episode 的 `proposal_status`、baseline 的 `readiness_status`，但顶层 `status` 必须使用上述四值。

## 4. Source Reference 与身份

每条可对接记录至少携带：

```text
schema_version
module=mental_health
person_id
session_id
source_video_id
start_time/end_time or local_date
status
quality_flags
identity.model/config/policy
source_refs
```

`source_refs` 中每项包含 `ref_type`、稳定 `ref_id`、可空 `artifact_path` 和可空 SHA-256。日级与 baseline 跨多个 session/video 时，顶层 `session_id/source_video_id` 可以为 null，但必须用 `source_refs` 列出全部来源。

## 5. 分层语义

- `episode_results`：shape/binary 结果；保留原 `technical_segment_index`、起止秒、`duration_seconds` 和 `reason_codes`。`proposal_status`、顶层四态、QC 四态和运行决策分开保存；`run_status` 固定为 `auto_accepted/uncertain/rejected`。`uncertain` 可送 shape，但不改写为人工 accepted boundary。
- `context_reviews`：只描述 phone/searching/cleaning/exercise 等 purpose/context；provider 失败输出 `unknown + unavailable`，不得改 shape、truth 或阈值。
- `daily_reports`：按 `person_id + local_date + timezone` 汇总 presence、coverage、shape/binary、high-confidence/uncertain、context 和质量。
- `baseline_profiles`：1-2 usable days 为 `warming_up`，3-6 日为 `initial_ready`，至少 7 日为 `stable_ready`，最多使用最新 14 个 usable days。
- `baseline_deviations`：只使用 observation day 之前的 reference；reference 不足时输出 null/status，不制造零值。
- `handoff_manifest`：绑定所有文件的 schema、record count、byte count 和 SHA-256；固定 `algorithm_event_emitted=false`。

完整 JSON Schema 位于 `configs/schemas/wandering_5d_v1/`。是否将这些 evidence 映射为业务 `AlgorithmEvent(module=mental_health)`，由后端对接阶段另行确定。

## 6. 构建命令

Windows/WSL 本机使用项目 conda 环境：

```bash
conda run -n eldercare-ai python scripts/wandering/build_camera_delivery_contract.py \
  --source-root "/mnt/d/徘徊数据集/自采数据" \
  --output reports/mental_health/wandering_camera_5d_contract_v3
```

命令会验证 editable module 位于当前仓库、fixed primary 可在 CPU 加载、48 条 index 全部路径/hash 可解析，并生成 `development_index.jsonl`、`run_manifest.json`、本契约快照、schema 快照和 `baseline_verification.txt`。
