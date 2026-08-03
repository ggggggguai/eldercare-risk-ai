# 心理安全模块目录结构

本目录按《心理健康模块设计 V2》整理为分层工程结构。根目录下既有的
`config.py`、`adapters.py`、`daily_aggregation.py`、`baseline.py`、
`features.py`、`pipeline.py`、`offline.py` 暂作为兼容实现保留，后续可逐步迁移到
下面的新包中。

```text
mental_health/
├── data_sources/              # 摄像头、睡眠仪、通话日志、S10 主动小测等原始数据接入
├── feature_extraction/         # 各模态结构化指标抽取
├── baseline_management/        # 个人参考基线、偏离计算、基线可信度、异常日过滤
├── submodules/                 # 两个业务子模块：情绪/社交关注、认知变化线索
├── scorecards/                 # 领域评分、持续性门槛、强规则、等级映射
├── auxiliary_models/           # Isolation Forest、Change Point、LightGBM 增强位
├── outputs/                    # 输出 schema、家属端文案、建议动作
├── privacy_compliance/         # 非诊断边界、隐私留存、数据最小化约束
├── validation/                 # 数据质量、公开数据集验证、专家审查材料
└── orchestration/              # 日批、实时服务、周报/趋势报告编排
```

## V3.3.3 监督专家进度

`mood_social/experts/` 已完成五个可独立加载的 standalone 专家：

- `M-ACT-001` ActivityExpert：LightGBM + Isotonic，活动 ECDF 和全部预处理只在对应训练折拟合；
- `M-SLP-001` SleepExpert：LightGBM + Isotonic，直接使用 canonical 同语义睡眠字段；
- `M-JNT-001` ActivitySleepJointExpert：LightGBM + Isotonic，只使用同一 canonical 行的真实活动与睡眠证据，双侧均有输入时 `expert_mask=1`。
- `M-PHY-001` PhysiologyExpert：ElasticNet Logistic + Platt，只使用 RESILIENT 的真实生理字段；
- `M-SOC-001` SocialContextExpert：CatBoost + Isotonic，只使用四来源严格同义档案字段。

五者均绑定 `mood_social_feature_schema_v3_3_3` 和 DATA-007 参与者级嵌套五折 split，并由 EVAL-001 冻结为只读 active 基线。MODEL-006 已另行完成 PHQ-9 连续分值回归和五级等级多分类离线辅助 bundle；该 bundle 不 active、不进入融合或 HTTP 推理。TREND-001 已完成活动、睡眠、社会三分支 PersonalTrendExpert、严格 OOF、个人基线回放和 `personal_change_mask`；FUSION-001 已完成 22,191 行严格 OOF 证据表和来源/参与者/外折/标签/窗口对齐审计，公开汇总数据的 day mask 明确是窗口观测代理。社会分支仍仅为无直接 S10/PHQ-9 验证的工程代理。下一任务为 FUSION-002，完整模型包仍待 ART-001。

## V2 对应关系

| V2 章节 | 工程包 |
|---|---|
| 原始数据接入 | `data_sources/` |
| 结构化指标输出 | `feature_extraction/` |
| 个人参考基线 | `baseline_management/` |
| 子模块一/二 | `submodules/` |
| 规则评分卡、强规则、等级映射 | `scorecards/` |
| Isolation Forest / Change Point / LightGBM | `auxiliary_models/` |
| 输出格式和家属端文案 | `outputs/` |
| 隐私与合规边界 | `privacy_compliance/` |
| 验证方案 | `validation/` |

## 设计边界

本模块只输出“行为趋势关注”和“认知功能变化线索”，不输出抑郁、孤独、
认知障碍、痴呆等医学诊断标签。

## 已落地特征入口

- `feature_extraction.activity`：日间活动、久坐/久卧、房间转换、外出与规律性特征。
- `feature_extraction.movement_vitality`：情绪低落/社交退缩关注模块中的运动活力领域分，复用步速、坐站、转身和步态稳定性日级指标，输出 `movement_vitality_score`，不触发紧急安全告警。
- `feature_extraction.physiology`：睡眠仪夜间心率/呼吸趋势辅助特征，基于个人参考基线输出 `night_physiology_score`，只作为情绪低落/社交退缩关注的低权重辅助证据。
- `feature_extraction.wandering`：认知功能变化线索中的徘徊样走动规则检测。当前代码仍接收旧 `x/y` 轨迹；模型化 P0 应新增落脚点轨迹构建器，按“双脚踝中点→单脚踝→人体框底边中心”生成单摄像头 `[0,1]²` 归一化轨迹，再输出 pacing、lapping、random、mixed 等行为线索。规则只作兜底，不保存原始视频、不输出医学诊断。
