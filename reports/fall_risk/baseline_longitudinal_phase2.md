# 个体行为基线 Phase 2 状态报告

更新时间：2026-08-12

结论：Phase 2 的算法侧基础设施和合成端到端验收已完成；真实数据与效果验收未完成，当前状态为 `infrastructure_complete / evidence_blocked`。规则/统计 fallback 和实时个人基线失败关闭保持不变。

## 计划审查发现

原计划同时要求“外层人员隔离”和“同一人的未来周期不进入 reference”，但没有定义 test 人员如何取得个人历史。如果 test 人员完全没有测试前 reference，只能测冷启动，无法回答个体化是否优于群体规则。现行实现改为：家庭或保守 `source_group_id` 做外层分区；每个 `person_id + camera_profile_id` 自身按时间先分 reference-only 周期，再分 scoring 周期。分区不读取 outcome，assignment 不含真值或 label ID。

另一个风险是仅用命令行布尔开关开放 test。现行 test 门禁要求 `data_status=formal`、frozen 协议、frozen split、与协议哈希绑定的 assignments，以及保管人 release acknowledgement；代码校验 release 与 split、协议、预测、标签、commit、授权角色和唯一 run ID 的绑定。评估器是无状态的，run ID 是否只消费一次必须由保管人工作流登记和保证。

## 已完成基础设施

- longitudinal observation 与 outcome review JSON schema。
- 稳定身份、授权设备/相机范围、有效监测时长、质量状态和未来 outcome horizon 校验。
- 按保守来源组隔离、同人前向 reference/scoring 的确定性 split；结果绑定 observation、label、profile、review 和协议哈希。
- 评估 CLI 只接受所选 partition 的 observation/label/prediction 文件；把其他 partition 的 test 真值混入 validation 会失败关闭，避免“未评估”不等于“未读取”。
- 评估 bundle 记录 observation、label、assignment、split、prediction 和协议配置的实际文件字节 SHA-256；formal 评估还要求干净 Git 并记录 commit。
- 人员/来源跨分区泄漏、reference/scoring 时间重叠和重复 assignment 审计。
- frozen split CLI 以实际 `risk_labels` 和 subject profiles 文件字节计算 SHA-256，并与 formal v2 校验报告绑定；内存 API 的哈希参数是显式可信边界，不单独充当文件来源证明。
- 固定四组消融：无个人基线、mean/std、median/MAD、鲁棒 EWMA/CUSUM + 防污染。
- 四组完全相同 prediction coverage 门禁，以及 Precision、Recall、F1、PR-AUC、Brier、ECE、FP/观测日、FP/有效摄像头小时和提前量。
- 人员级 bootstrap 95% CI，人员、设备、相机、来源组、场景、质量和曝光分层，以及失败案例 JSONL。

## 合成验收

使用 60 个纯合成身份、每人 10 个 reference 周期和 3 个 scoring 周期构造 fixture；实际落入 validation 的身份和评分周期由确定性来源组分区决定。随后由 outcome-blind 因果回放器生成四组同覆盖预测，并交给评估器完成 validation 烟测。任何合成数值都不是模型或真实老人域效果。

复现命令：

```bash
conda run -n eldercare-ai python scripts/evaluate/build_synthetic_fall_baseline_longitudinal_fixture.py \
  --output-dir /tmp/fall_baseline_longitudinal_synthetic

conda run -n eldercare-ai python scripts/evaluate/generate_fall_baseline_longitudinal_predictions.py \
  --observations /tmp/fall_baseline_longitudinal_synthetic/observations.validation.jsonl \
  --assignments /tmp/fall_baseline_longitudinal_synthetic/assignments.jsonl \
  --split /tmp/fall_baseline_longitudinal_synthetic/split.json \
  --partition validation \
  --output /tmp/fall_baseline_longitudinal_synthetic/replay-predictions.validation.jsonl

conda run -n eldercare-ai python scripts/evaluate/evaluate_fall_baseline_longitudinal.py \
  --observations /tmp/fall_baseline_longitudinal_synthetic/observations.validation.jsonl \
  --risk-labels /tmp/fall_baseline_longitudinal_synthetic/risk_labels.validation.jsonl \
  --assignments /tmp/fall_baseline_longitudinal_synthetic/assignments.jsonl \
  --split /tmp/fall_baseline_longitudinal_synthetic/split.json \
  --predictions /tmp/fall_baseline_longitudinal_synthetic/replay-predictions.validation.jsonl \
  --partition validation \
  --data-status synthetic \
  --output-dir /tmp/fall_baseline_longitudinal_bundle
```

## 真实数据门禁

截至 2026-08-12：

| 输入/门禁 | 当前值 | 状态 |
|---|---:|---|
| longitudinal observations | 0 | blocked |
| `risk_labels.jsonl` | 0 | blocked |
| subject profiles | 0 | blocked |
| 人工 review log | 0 | blocked |
| frozen split | 无 | blocked |
| frozen evaluation protocol | 无 | blocked |
| test custodian release | 无 | blocked |

真实门禁命令预期以退出码 3 结束并写出机器 blocker：

```bash
conda run -n eldercare-ai python scripts/split/build_fall_baseline_longitudinal_split.py \
  --output-dir /tmp/fall_baseline_longitudinal_gate
```

下一步是采集有同意记录的连续个人周期、独立定义并双审未来状态终点，在 validation 前冻结最小样本量、晋级增益、允许误报增加、split 和评估协议。门禁满足前不得生成 formal 指标、读取 test 或替换实时主路径。
