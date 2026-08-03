# 徘徊方案步骤 2 转换与复核记录

日期：2026-08-01

本记录只覆盖 WanderingPatterns 与 SmartCare 的只读转换、自动验证和可视化抽检，不代表模型、正式 split、摄像头域或心理健康风险链路已经完成。原始数据未删除、移动或改写。

## 自动验证结论

- 项目环境：`eldercare-ai`，editable 安装指向当前仓库。
- 窄范围测试：`29 passed, 19 subtests passed`。
- 完整测试集：`407 passed, 1 failed, 86 subtests passed`；唯一失败是既有跌倒验收测试缺少 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`，不是本次徘徊转换回归。
- 当前严格 reader 实际读回：WanderingPatterns 1,600 条、SmartCare `train_pool` 190 条、SmartCare `official_validation` 20 条，共 1,810 条。
- 两个全新输出目录各执行一次完整转换；共 9 个 bundle 文件逐文件 SHA-256 完全一致，无不一致文件。
- 转换器在构建与提交后均重新核对来源文件 SHA-256；输出目录已存在时拒绝覆盖。

| 数据源 | manifest SHA-256 | report SHA-256 |
|---|---|---|
| WanderingPatterns | `04bb88849eddd15ef12de7883612f3b03a20004dfbcd9684e1ef059f06875b9e` | `8d62edc4d8048a940d2abf8e31cb66cda0c52d8ae60e5cb58cd76b74ffe2929a` |
| SmartCare | `c74bb729cf2c353311fef2013ae6abf72a59f70dab5a1969ffb48a37262736bc` | `a58f6969abac234fb7cddd13c9ba2ba005546faed2f4aa49987b89265832bdd0` |

## 测试先行记录

实现前分别运行了转换核心、pickle 隔离、CLI 和可视化测试，初始失败依次是缺少 `converters` 模块、隔离提取脚本、两个转换 CLI 和 `visualization` 模块。真实 WanderingPatterns 首次转换又发现 `Coords` 是 80 个 `[x, y]` 对组成的 object 向量，而不是直接的 `(80, 2)` 数值矩阵；先补该真实形态的回归测试并确认失败，再扩展受限解析器。最终所有徘徊窄测试通过。

## WanderingPatterns

两个 pickle 均为 11,144,757 bytes，SHA-256 均为 `f0c6484dbae6ca4df7bb44b9d91f7961e5fa79bd11ee21e86513c89a4dbf9edc`。pickle 只在 WSL 的无网络、只读挂载、空环境、受限资源隔离空间中解包；主转换进程只读取惰性 JSONL，不导入 `pickle`。

- 读取/写出/拒绝：1,600 / 1,600 / 0。
- 类别：`direct=400`、`pacing=400`、`lapping=400`、`random=400`。
- 长度：每条恰好 80 点，共 128,000 点。
- 原坐标范围：x `[-72.37268, 99.9094781519804]`，y `[-73.49258924930601, 80.778786]`；0 个非有限坐标。
- `samples.jsonl` SHA-256：`498a0c39af27240bd38ab82367fdc1a13373ba3a22e8fede915419fea54e6ee7`。

固定哈希文件的实际 DataFrame 列仅为：

```text
pattern, CartesianX, CartesianY, Slope, Coords, Coords_Slope, Path_Efficiency
```

因此事实结论是：该文件没有 `group`、`group_id`、人员、会话或其他受支持的分组字段；`group_fields_found=[]`，全部 1,600 条带 `group_field_unavailable` 警告。后续不能声称 WanderingPatterns 支持真实按人或按会话防泄漏划分。

## SmartCare

- 来源哈希：主数据 `25845a0d66022285e4b631d3101e574f60aa5996275c65222b759bfaf3473b71`；官方验证 `afac7512dfd3903bb87ec006da20937591b699a5eecee7377b7825ae1360bc75`。
- 读取/写出/拒绝：220 / 210 / 10；写出类别为 `normal=100`、`wandering_like=110`。
- 输出角色：`train_pool=190`、`official_validation=20`；官方验证保持独立，未用于可视化抽检或调参。
- 写出长度：最小 4、最大 73、中位数 15、均值 16.7571，共 3,519 点。
- 原画布按 602×369 处理，归一化使用 `x/601`、`y/368`；不裁剪、不补点。写出坐标 x `[0.0098320934, 0.9850249584]`，y `[0.0244565217, 0.9891304348]`。
- `date` 是实际可用的轨迹分组字段；没有逐点时间，因此 210 条均明确记录 `point_time_unavailable`。

极短和越界异常均保存在 `rejections.json`：

| 异常 | 数量 | 事实 |
|---|---:|---|
| 少于 4 点 | 9 条 | 6 条为 1 点、1 条为 2 点、1 条为 3 点；另 1 条为 1 点且同时越界 |
| 含越界点 | 2 条 | `18/09/2020 09:00`（21 点）与 `18/09/2020 06:00`（1 点）；两者首点均为 `(149.90908813476562, -62.07954406738281)` |
| 唯一拒绝轨迹 | 10 条 | 短轨迹与越界集合有 1 条重叠，因此不是 11 条 |

## 分层随机可视化

固定随机种子为 `20260801`，每类抽 20 条：WanderingPatterns 80 条，SmartCare `train_pool` 40 条，共 120 条。绿色点为起点，红色点为终点，标题尾部对应 `selection.json` 中按行优先排列的 sample ID。

| 数据源 | 类别 | 联系表 |
|---|---|---|
| WanderingPatterns | direct | [direct.png](visual_review/wandering_patterns/direct.png) |
| WanderingPatterns | pacing | [pacing.png](visual_review/wandering_patterns/pacing.png) |
| WanderingPatterns | lapping | [lapping.png](visual_review/wandering_patterns/lapping.png) |
| WanderingPatterns | random | [random.png](visual_review/wandering_patterns/random.png) |
| SmartCare | normal | [normal.png](visual_review/smartcare/normal.png) |
| SmartCare | wandering_like | [wandering_like.png](visual_review/smartcare/wandering_like.png) |

机器可读抽样清单分别位于 [WanderingPatterns selection.json](visual_review/wandering_patterns/selection.json) 与 [SmartCare selection.json](visual_review/smartcare/selection.json)。用户于 2026-08-03 确认 6 张联系表均可通过，两个清单现均为 `review_status=human_review_passed`；正式记录见[步骤 2 人工可视化复核记录](HUMAN_REVIEW.md)。

AI 辅助预检只确认图能正常生成、起终点与路径顺序连续，未见明显 x/y 互换、全局翻转、截断或离散跳点。`direct`、`lapping`、`random` 与 SmartCare `wandering_like` 的整体形态符合来源标签；WanderingPatterns `pacing` 中多条呈宽 U 形往返或近闭合轨迹，SmartCare `normal` 中也有少量复杂/重复路径。用户人工复核已覆盖这些边界形态并确认通过；转换器仍只保留来源标签，不擅自重标。

## 人工复核结论

2026-08-03，项目用户在当前 Codex 任务中确认“6张都可通过”。复核范围和结论已经写入 [HUMAN_REVIEW.md](HUMAN_REVIEW.md)：

- 共检查 6 张联系表、120 条轨迹；
- 未报告标签不匹配、路径顺序错误或坐标轴错误；
- 未提出需要排除、重标或重新转换的 sample ID；
- 步骤 2 的自动门槛和人工门槛均已满足，步骤 2 关闭。

该通过结论不扩大证据边界：正式 split、模型、目标摄像头域和临床有效性仍未验证。
