# fall-risk-data-v1 发布候选摘要

候选日期：2026-07-15

结论：自动化底座已实现；`fall-risk-data-v1` 整体仍受人工、许可、数据与盲测治理门槛阻塞，不能标记为 frozen。

## 自动化验收

| 条件 | 结果 | 证据 |
|---|---|---|
| editable 安装绑定当前仓库 | 通过 | `pip show` 指向当前工作区 |
| 全量 manifest 与逐视频 FPS | 通过 | 3,454 assets；SHA `6cbb45d7...e049` |
| Coffee/Lecture/Office 不使用 24 FPS | 通过 | manifest 驱动转换测试与 Coffee 候选重导 |
| CVAT 与 LE2I 官方来源独立 | 通过 | source-specific candidate；99 个 `le2i_txt` 事件 |
| 严格 schema/来源/复核/许可门禁 | 通过 | 候选 0 schema errors；根标签 2,766 errors 被拒绝 |
| 四任务独立 split schema | 通过 | 四个真实 blocked artifact，均 `split_id=null` |
| 人员/源组/事件/hash 泄漏检查 | 通过 | split 回归测试覆盖并查集闭包 |
| 事件评估器与证据包 | 通过 | 合成 bundle 含 7 类输出与 95% CI |
| 提前量、分母、一对一匹配回归 | 通过 | 评估测试覆盖；无分母时 fail closed 为 `null` |
| 真实正式 split 与正式指标 | 未通过 | 无合格人审标签、连续分母和 frozen 协议 |
| 身份元数据发布治理 | 未通过 | 三个 CVAT 原始导出检测到身份字段，待负责人决定 |

最终自动化验证：Workflow A 五组窄测 `101 passed, 22 subtests`；仓库全量测试 `268 passed, 67 subtests`；`compileall` 与 `git diff --check` 通过。manifest 第二次重建与已生成 manifest 字节级一致，SHA-256 仍为 `6cbb45d7...e049`。

## 当前可交付物

- `data/manifests/fall_risk_video_manifest.jsonl`
- `data/annotations/fall_risk/generated/v1/` 下的 CVAT/LE2I 候选
- 空的 risk/review/profile 模板
- `data/splits/fall_risk/` 下四类 blocked split 证据
- 严格 validator、split builder、事件评估器及开发配置
- `reports/fall_risk/workflow_a_synthetic_evaluation/` 合成评估证据包
- 数据审计、版本记录和人工阻塞清单

## 不可声明的结论

- 不能把 922 条 pending 根标签称为真值或训练集。
- 不能把公开跌倒片段用于证明长期个体风险预测或临床有效性。
- 不能报告正式 FPR、FP/摄像机小时、FP/家庭日；当前合法分母为零。
- 不能把合成 F1 1.0 当作模型性能。
- 不能声称人员泛化、功能 proxy 或纵向预测已经验证。

下一责任人与解除证据见 `reports/fall_risk/workflow_a_blockers.md`。完成全部人工门槛后，必须重新运行 formal validator，使用经预注册的 frozen 配置创建新 split 版本，再由独立测试集保管人执行一次性评估。
