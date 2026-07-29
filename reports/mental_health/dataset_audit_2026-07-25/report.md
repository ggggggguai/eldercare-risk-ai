# 心理健康公开数据集可用性与时序审计报告

审计日期：2026-07-25  
审计版本：`mental-health-dataset-audit-v1`  
范围：PSYCHE-D、RESILIENT、NHANES 2011–2014  
结论边界：只评估数据、标签、时间方向和实验角色，不代表模型已经训练或达到临床有效性。

## 技术摘要

本次审计已经解决三个会直接改变技术路线的问题：

1. **PSYCHE-D 可以作为 P0 主训练数据。**本地有 10,866 个完整季度标签窗口，来自 4,036 名参与者；end PHQ-9≥10 有 3,444 窗、涉及 1,790 人。官方数据构造说明确认标签月 wearable PGHD 来自 end PHQ-9 前 8–14 天，因此可做近两周 `State/Screen`；使用 `m-1`、`m-2` 月记录还可分别建立约 1 月、约 2 月固定提前量实验。公开矩阵没有真实日期，所以这是数据设计级确认，不是逐行时间戳复核。
2. **RESILIENT 既不适合全部拿来训练 P0 主模型，也不是严格的 past-only 外测。**73 人中 PHQ-9≥10 只有 10 人；所有人的传感器都在 PHQ/GDS/ACE 基线量表同日或之后开始，量表前 7/14/28 天四模态窗口均为 0。正确角色是：先在全部 73 人上做一次冻结模型的老人域反向时间敏感性测试；保存结果后，再解锁全体做内部嵌套交叉验证适配。
3. **NHANES 的 DPQ 在 PAM 之前。**本地逐行文件没有真实时间戳，无法计算每人的具体间隔；CDC 官方流程明确 DPQ/PHQ-9 在 MEC 私密访谈中完成，PAM 在 MEC 检查会话结束时启动，并在离开 MEC 后佩戴约 7 天。因此 NHANES 只能证明量表状态与随后活动/睡眠之间的关联，不能证明过去行为对当前或未来心理风险的前置筛查。

由此，P0 证据链应调整为：

```text
PSYCHE-D：主训练 + 参与者级内部测试 + 固定提前量实验
RESILIENT：老人域反向时间敏感性测试 → 之后才做内部适配
NHANES：post-assessment 关联、辅助表征与跨周期稳健性
自采老人小队列：真正的 past-only 目标人群外部验证
```

## 数据版本冻结

正式审计对三个本地目录的全部文件计算了 SHA-256，并用“相对路径、字节数、文件 SHA-256”的排序序列生成集合哈希。

| 数据集 | 文件数 | 字节数 | 集合 SHA-256 |
|---|---:|---:|---|
| PSYCHE-D | 3 | 21,337,360 | `371cf2688e1cbd3df416d93d277f486462770c27a250862a47b688177b7ff001` |
| RESILIENT | 589 | 154,302,372 | `77278afd8d443cf53608ada9cf576b46b1799951ab79149c39af4e5ef50f53c2` |
| NHANES | 22 | 75,241,704 | `cb63c3506a02cbcc493cff6a952d5c14333e0bf8b6abf853bcf0bf53091172b6` |

逐文件清单见 `dataset_file_manifest.csv`。用户已确认许可允许本次比赛使用；但本地仍应归档许可证原件、版本、核验日期、指定引用和模型权重/派生特征发布边界。PSYCHE-D 当前目录尤其缺少 Zenodo 提供的 `LICENSE.pdf`。

## Annotation 1：PSYCHE-D 确认结果

### 时间方向

[PSYCHE-D 官方 Zenodo 记录](https://zenodo.org/records/5085146)和[主论文](https://mhealth.jmir.org/2022/3/e34148)说明：

- PHQ-9 每三个月采集一次；
- 一个季度样本包含 `SM0` 和 `SM3` 的 PHQ-9；
- wearable PGHD 来自标签生成日前 8–14 天；
- Phase 2 使用最终 PHQ 完成前约两周的 SM3 wearable 数据。

因此可以冻结三个任务：

| 任务 | 输入截止 | 标签 | 可用表述 |
|---|---|---|---|
| `sensor_state` | 标签月 `m` 的冻结传感器统计 | end PHQ-9 | 近两周状态筛查 |
| `forecast_1m` | 只使用 `m-1` 月及更早数据 | end PHQ-9/恶化 | 约 1 月固定提前量 |
| `forecast_2m` | 只使用 `m-2` 月及更早数据 | end PHQ-9/恶化 | 约 2 月固定提前量 |

不能把标签月传感器宣传为“提前三个月预测”。公开矩阵只有名义研究月份，没有真实采集/量表日期，因此不能逐行计算精确 30/60/90 天。

### 样本、标签和重复测量

| 指标 | 审计值 |
|---|---:|
| 全部矩阵 | 35,694 行 × 154 列 |
| 矩阵参与者 | 4,948 人 |
| 完整四标签窗口 | 10,866 |
| 有完整标签参与者 | 4,036 人 |
| end PHQ-9≥10 | 3,444 窗、1,790 人 |
| end 类别高于 start 类别 | 2,252 窗 |
| end-start≥5 分 | 1,115 窗 |
| start<10 且 end≥10 | 946 窗 |

每人完整窗口数：

| 窗口数/人 | 人数 |
|---:|---:|
| 1 | 929 |
| 2 | 774 |
| 3 | 943 |
| 4 | 1,390 |

相邻标签窗口名义间隔：

| 月份差 | 窗口对 |
|---:|---:|
| 3 | 6,471 |
| 6 | 293 |
| 9 | 66 |

全部 6,471 个连续季度窗口中，后一窗 `phq9_score_start` 与前一窗 `phq9_score_end` 均一致。这证明季度端点语义可靠，也证明同一人存在多次重复评估，不能随机按行切分。

固定提前量的数据可用性：

- `m-1` 和 `m-2` 均存在：8,494 个标签窗口；
- 只存在其中一个：1,685 个；
- 两者均不存在：687 个。

### 划分契约

推荐立即冻结：

- 以 `participant_id` 为唯一 group；
- 约 800 名参与者作为 untouched internal test；
- 其余约 3,236 人做 5 折 `StratifiedGroupKFold`；
- 参与者级组合分层至少覆盖 `ever_end_ge10 × ever_category_worsened × window_count_bucket`；
- 插补、标准化、缺失率筛列、特征选择、阈值和校准只能在训练折拟合；
- 置信区间采用参与者 cluster bootstrap，不能把 10,866 窗视为独立个体。

主结果至少比较：

1. `sensor-only`；
2. `start-only`；
3. `start+sensor`。

否则变化预测可能只是在复述上一期 PHQ，而不是证明无感睡眠/活动特征有增量价值。

### 冻结特征映射

`psyche_sensor_semantic_v1` 已按字段语义、而不是传感器—标签性能冻结：

- 纳入 27 个有明确字段说明的 Sleep/Steps `Statistics` 特征；
- 隔离 84 个缺少回看窗口和计算定义的 `Linear regression` 字段；
- `phq9_score_end/phq9_cat_end` 只作目标；
- start PHQ 只允许进入 `start-aware` 消融；
- 静态人口学只作公平性/域偏移消融；
- 同月生活方式、药物和非药物治疗字段因时序不明且与部署模态不一致，P0 排除。

完整映射见 `psyche_feature_mapping_v1.csv`。如果后续补齐 84 个派生字段的官方计算定义，应建立 `v2` 映射，不覆盖 `v1`。

## Annotation 2：RESILIENT 应该训练还是外测

### 为什么不能直接全部训练

- 只有 73 人；
- PHQ-9≥10 只有 10 人；
- 固定 20% 测试集约 15 人，期望只有约 2 个 PHQ 阳性；
- 标签是基线量表，而传感器发生在同日或之后，和 PSYCHE-D 的 past-only 输入方向不一致；
- 在此规模上从头训练 TCN、Transformer 或复杂 LightGBM，过拟合和结果波动风险都很高。

### 为什么也不能称为严格外测

四模态中最晚启动的模态相对 PHQ 日期：

| 偏移 | 人数 |
|---:|---:|
| 同一日 | 28 |
| +1 日 | 37 |
| +2 日 | 3 |
| +5/+9/+10/+20 日 | 各 1 |
| 模态缺失 | 1 |

同日只有日期，没有日内先后时间；因此也不能把同日数据当作量表前窗口。严格量表前 7/14/28 天四模态窗口均为 0。

### 推荐的两阶段用法

第一阶段：冻结迁移敏感性测试

1. 不查看 RESILIENT 传感器—标签性能，先冻结共同特征、预处理器、PSYCHE-D 模型和评价指标；
2. 在全部 73 人上运行一次；
3. 命名为“老人域跨队列状态关联验证”或“时间方向不一致条件下的迁移敏感性分析”；
4. 报告置信区间、校准、缺失模态和失败案例，不以单个 AUC 作晋级依据。

第二阶段：解锁后的内部适配

1. 保存并封存第一阶段结果后，才允许使用标签调参；
2. 全部 73 人用于重复嵌套交叉验证或 LOOCV，不再切固定小测试集；
3. 老年适配优先用 GDS-15 连续分数或 `GDS≥5`，因为有 23 个阳性；PHQ 连续分数作辅助，`PHQ≥10` 分类只作探索；
4. 使用 ElasticNet、带强正则 Logistic、浅层 LightGBM；不从头训练深度时序网络；
5. 结果统一标注为“RESILIENT 内部适配验证”，不再称外部测试。

### 可保留的 P2 纵向认知任务

RESILIENT 有 48 人具备 ACE-III 基线与约 6 月随访配对，其中 45 人在两次评估之间四模态各有至少 28 个唯一记录日。ACE 下降至少 3 分有 12 人、至少 5 分有 7 人。

可以探索：

```text
ACE-III 基线 + 基线后预注册的固定早期传感器窗口
→ 约 6 月 ACE-III / 变化量
```

该任务时间方向正确，但样本仍小，只适合 Ridge/ElasticNet 和探索性结论；它属于认知线索 P2，不替代 P0 情绪风险任务。

## Annotation 3：NHANES PAM 与 DPQ 顺序

### 本地逐行证据

- `DPQ_G.xpt`、`DPQ_H.xpt`合计 11,539 行，字段只有 `SEQN`、9 个 PHQ 条目和功能影响字段，没有日期或时间；
- 日级 CSV 有 89,104 行、14,264 人；
- `calendar_date` 全部为 2000-01-02 至 2000-01-15，而调查周期是 2011–2014，属于重置/构造日期；
- `DEMO_G/H`只有 `RIDEXMON` 检查季节码，没有精确检查日期。

因此公开文件不能逐人计算 `PAM_start - DPQ_time`。

### 官方流程证据

- [CDC DPQ 2013–2014 文档](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DPQ_H.htm)：DPQ 在 MEC 私密访谈中完成，回顾前两周症状；
- [CDC PAXMIN 2013–2014 文档](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXMIN_H.htm)：PAM 在 MEC 检查会话结束时开始记录，随后佩戴约 7 天；
- [CDC 2011–2012 调查概览](https://wwwn.cdc.gov/Nchs/Nhanes/ContinuousNhanes/OverviewExam.aspx?BeginYear=2011)：将 PAM 明确归入 post-MEC data collection。

协议顺序为：

```text
MEC 内完成 DPQ/PHQ-9
→ MEC 会话结束
→ PAM 开始记录
→ 离开 MEC 后连续佩戴约 7 天
```

最准确的研究口径是：

> 量表评估后的腕部活动/睡眠模式，与评估时近期抑郁症状负担之间的回顾性横断面关联。

允许用于：

- 睡眠/活动特征方向验证；
- 辅助表征预训练；
- 2011–2012 与 2013–2014 跨周期稳健性；
- 去除 DPQ 睡眠条目的敏感性分析。

禁止表述：

> 根据 PHQ-9 测量前 7 天的活动和睡眠预测随后 PHQ-9。

## 当前需要执行的工作

### 立即冻结，不再讨论

1. 固定本报告中的三个数据集角色；
2. 保存 `dataset_file_manifest.csv` 和三个集合 SHA-256；
3. 采用 `psyche_sensor_semantic_v1` 的 27 个 P0 传感器字段；
4. PSYCHE-D 所有划分按参与者隔离；
5. NHANES 与 RESILIENT 的反向时间结果不得混入 PSYCHE-D past-only 主成绩；
6. 把 V3.1 中“RESILIENT 严格 State/Screen 外测”统一改成“老人域反向时间迁移敏感性测试”。

### 接下来一周的 P0 实施顺序

1. 生成并冻结 PSYCHE-D participant split manifest；
2. 构建 `sensor_state`、`forecast_1m`、`forecast_2m` 三个样本表；
3. 训练并比较 V2-reduced、Logistic/ElasticNet、LightGBM；
4. 做 `sensor-only / start-only / start+sensor` 消融；
5. 完成概率校准、参与者 cluster bootstrap、缺失模态/低质量压力测试；
6. 冻结模型后运行 RESILIENT 全 73 人 one-shot 敏感性测试；
7. NHANES 只运行单独命名的 post-assessment association 实验；
8. 启动老人前瞻小样本采集协议。

## 仍需要项目负责人确认的少量决策

推荐默认值已经给出；只有以下事项会改变正式实验协议，需要在训练前签字冻结：

| 决策 | 推荐值 |
|---|---|
| P0 主标签 | `end PHQ-9 >= 10`；连续分数作共同主结果 |
| Change 主标签 | `end category > start category`；`delta>=5` 和新发 `>=10` 作敏感性 |
| 固定内部测试规模 | 约 800 名 PSYCHE-D 参与者 |
| 随机种子 | `20260725`，生成后保存参与者级 manifest |
| RESILIENT one-shot 主终点 | PHQ 连续分数 + GDS 连续分数；阈值结果仅描述性 |
| 是否启动自采 | 建议立即启动；先采 7–14 天，再填 PHQ-9/GDS |
| 许可归档责任人 | 指定一人把许可证原件与允许范围放入统一 `licenses/` |

如果负责人不另行修改，建议直接按上表作为冻结协议执行，避免继续消耗时间做路线讨论。

## 复现与校验

运行：

```powershell
conda run -n eldercare-ai python scripts/audit/audit_mental_health_datasets.py
conda run -n eldercare-ai python scripts/audit/validate_mental_health_audit.py
conda run -n eldercare-ai python -m pytest tests/test_mental_health_dataset_audit.py -q
```

本次结果：

- 12 项数据一致性/时序检查全部通过；
- 独立校验重新计算关键样本/时序数字，并逐一复核 614 个源文件大小与 SHA-256，全部通过；
- 审计脚本窄范围测试 3 项全部通过；
- notebook 已在 `eldercare-ai` 环境中从头执行。

主要文件：

- `audit_summary.json`：机器可读总结果；
- `validation_checks.json`：检查项与观察值；
- `dataset_file_manifest.csv`：逐文件 SHA-256；
- `dataset_role_decisions.csv`：数据集角色冻结表；
- `psyche_feature_mapping_v1.csv`：PSYCHE-D 冻结特征契约；
- `mental_health_dataset_audit.ipynb`：可复核 notebook。
