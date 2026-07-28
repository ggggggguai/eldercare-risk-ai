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
- `feature_extraction.wandering`：认知功能变化线索中的徘徊样走动规则检测，基于中心点轨迹输出 pacing、lapping、random、mixed 等行为线索，不保存原始视频、不输出医学诊断。
