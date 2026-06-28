# AGENTS.md

## 项目环境

本项目的测试和脚本运行必须使用项目 conda 环境，不要使用默认 shell 里的 Python。

统一使用：

```bash
conda run -n eldercare-ai python -m pytest -q
```

运行项目脚本时也必须进入同一个环境：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py --help
conda run -n eldercare-ai python -m elderly_monitoring.inference.run_features --help
```

除非是在专门排查环境问题，否则不要使用裸 `python`、裸 `pytest` 或 base conda 环境做验证。

标准环境定义文件：

```text
environment.yml
```

本机已验证环境说明：

```text
environment-reference.txt
```

## 开发流程

非平凡功能开发、缺陷修复、重构和发布准备需要遵循较完整的工程流程：

- 需求或目标行为不清楚时，先澄清再实现。
- 多步骤工作先写简短计划。
- 功能和缺陷修复优先采用测试驱动或先补回归测试。
- 遇到失败时做系统化排查，不凭猜测改代码。
- 宣称完成前，必须运行相关测试或验证命令。

措辞调整、文档补充、小配置改动可以保持轻量流程。

## 测试规则

修改 Python 代码后，先运行最相关的窄范围测试；条件允许时再运行完整测试集。

常用命令：

```bash
conda run -n eldercare-ai python -m pytest tests/test_fall_risk_tracking.py -q
conda run -n eldercare-ai python -m pytest -q
```

涉及视觉检测或跟踪模块时，需要额外跑一个真实视频烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_tracking.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_tracks_smoke.jsonl \
  --model yolov8n.pt \
  --scene-region home \
  --max-frames 5
```

涉及姿态关键点模块时，需要额外跑一个真实视频烟测：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_pose.py \
  --input "data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi" \
  --output /tmp/fall_poses_smoke.jsonl \
  --model yolov8n-pose.pt \
  --scene-region home \
  --max-frames 5
```

如果测试无法运行，必须说明失败原因，并写出已经尝试过的具体命令。

## 协作风格

- 优先追求正确性、证据和有用的分歧，不为了表面一致而回避问题。
- 做技术评审时区分事实、推断和观点。
- 评审代码或方案时，先说风险、缺陷、缺失约束和更稳妥的替代方案。
- 没有检查相关证据前，不要宣称某个模块已经满足研发计划要求。

## 项目边界

本仓库只覆盖老年人跌倒风险和心理健康风险预警的算法原型研发。

本仓库不实现：

- 家属端 App、社区端后台或可视化看板。
- 账号、权限、设备管理。
- 消息推送、电话通知、工单流转和线下处置流程。

跌倒风险相关工作需要对齐：

```text
docs/modules/fall_risk/plans/跌倒风险算法研发计划.md
docs/modules/fall_risk/README.md
```

## 跌倒风险固定算法路线

后续跌倒风险模块的实现、文档、测试和代码评审，都需要按以下主线对齐：

```text
萤石设备或开放平台视频流
  ↓
人体检测与跟踪
  ↓
人体姿态关键点提取
  ↓
关键点质量控制与时序平滑
  ↓
步态稳定性分析
  ↓
坐站转换能力分析
  ↓
近跌倒事件检测
  ↓
个体化行为基线建模
  ↓
轻量风险融合模型 + 规则校准
  ↓
跌倒风险等级 + 置信度 + 可解释风险因子 + 预警动作建议
```

实现时不要跳过中间层直接从视频给最终风险结论。若某一层暂时使用规则、轻量 baseline 或占位实现，必须在文档和结果说明中标明当前状态。
