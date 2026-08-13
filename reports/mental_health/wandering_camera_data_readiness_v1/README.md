# TopoWander-MPT M0-CAM-RD camera development 数据工具

## 结论

M0-CAM-RD 已完成，统一状态为：

```text
status=wandering_m0cam_development_data_tooling_ready
evidence_scope=synthetic_schema_contract_only
authorized_camera_data_consumed=false
m0cam_d_started=false
sealed_camera_accessed=false
```

本轮只实现并验证 camera development 的数据契约、授权入口控制流和 evaluator 原语。没有读取真人 camera 数据，没有生成或替代 C0 授权 receipt，没有启动 M0-CAM-D，没有加载候选做实际 forward，也没有访问 WP raw、SmartCare official/raw 或 sealed camera。

## 已实现范围

| 分段 | 稳定产物 | 当前证明 |
|---|---|---|
| RD-A | 匿名 collection/session/source/setup/stream-epoch manifest；C0 receipt validator；调用者 tracking+sidecar 规范化；第 10.3 节 episode annotation validator；C0-C3 readiness | schema、隐私拒绝、跨引用、分组隔离、人工标签轴和 `not_ready` 语义 |
| RD-B | 独立 `run_camera_development.py`；固定候选/源身份；receipt-first fail-closed 顺序；fresh staging/final | synthetic 私有测试可验证控制流；production CLI 不含 synthetic、fake runtime 或绕过参数 |
| RD-C | 参数化一对一 temporal matching、四分类/二分类原始 confusion/counts、purposeful hard-negative 诊断、coverage、区间/eligible negative person-hours、stage timing | 纯函数与边界测试；不选择真实 matching、uncertain 或 merge 参数，不输出 F1/FAR/CI/95% |

原 `wandering_camera_primary_v1.yaml` 和 `run_topowander_camera_inference.py` 未修改，原 primary 入口继续只接受 synthetic fixture。新 development 入口必须先取得 approved、active、development-purpose 且作用域匹配的外部 C0 receipt；receipt 失败时不读取 tracking/annotation/cohort，不加载候选，不 forward，也不创建 staging/final。

## Fresh synthetic readiness

fresh 证据位于 `artifacts/synthetic_input_readiness_v1/`，来源仅为既有 synthetic fixture：

- 调用者 tracking JSONL：240 条 observation；source SHA-256 `529b15ad4dd7b6fa3c15f4f8f81a08d612f2f911fb647a052fb4ab2e48011253`；
- canonical tracking SHA-256 `22a99a8bae6b7dc2798d7df2db0e5fd857996e968f1949ab1a804f7be27579de`；
- canonical sidecar SHA-256 `00f5f752df76ad01f1aa60c8eeb98bc1d2e82a710b3ddaf444c44684c03f7727`；
- `media_opened=false`、`detector_run=false`、`tracker_run=false`；
- 没有 C0，故 `readiness_status=not_ready`，C0/C1/C2/C3 均为 false；synthetic 的 C1-shaped 输入不能充当授权 C1。

可提交的验证记录见 [VERIFICATION.md](VERIFICATION.md)。`readiness.json`、`artifact_manifest.json`、`verification.json` 和 fresh synthetic 制品属于本机生成证据，受 `.gitignore` 管理；其关键状态与摘要已固化在本报告中，不把这些本机制品伪装成 fresh-clone 自带文件。

## 后续独立审计

本轮 synthetic schema/tooling 和 `not_ready` 证据保持有效，但后续只读审计发现真实 development 入口尚不能直接启用：primary loader config 与 authorized prediction evidence scope 不兼容；两个数据辅助 CLI 尚未 receipt-gate；C3 没有把实际 tracking 完整绑定到 participant/session/setup；person-hours 会跨 session 合并；truth-masked、abstention、miss 和 unavailable/error 尚未严格分开；episode matching 的贪心实现不能保证最大匹配，且缺少 production authorized success 回归。

这些问题由 `M0-CAM-RD-F` 修复。修复前，本报告只能证明 `synthetic_schema_contract_only` 的工具骨架；三个 RD CLI 不得读取真人 development 输入，也不得生成 authorized evidence。

## 下一步外部输入与入口

M0-CAM-D 无标签 engineering smoke 仍缺：

1. C0：负责人/授权流程提供的 approved、active、development-purpose receipt，覆盖 participant/session/setup/source-group 和 `run_development`；
2. C1：经授权成人、固定 setup 产生的 tracking JSONL + `wandering-media-v1` sidecar，以及与 collection/session manifest 的一致绑定。

只有 RD-F2+C0+C1 后，独立 development 入口才可形成 `authorized_development_smoke`，范围仅为 tracking/QC/coverage/forward/工程 timing，不得输出 accuracy/F1。

有标签 development 评估还缺：

3. C2：人工 episode annotations，完整保留 observable shape、purpose、truth uncertainty/exclusion；
4. C3：显式 tracklet↔匿名 participant、participant-present、setup/session/source-group 和 clock alignment 分组；
5. 在 development 数据上显式形成的 matching、uncertain 和 episode-merge policy。

上述全部满足且 evaluator 成功后，证据范围才可写为 `authorized_labeled_development_evaluated`。本轮没有替后续任务选择 IoU/onset tolerance、uncertain threshold 或 merge gap。

## 科学边界

本报告只证明 synthetic schema 与工程控制契约。它不是 camera accuracy、episode F1、FAR、真实时延、目标机位、真实老人、产品或临床有效性证据。固定候选的 WP public-shape 结果也不能替代 camera development 或 sealed camera 证据。
