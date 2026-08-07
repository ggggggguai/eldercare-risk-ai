# 老年人多模态风险预警算法工程

本工程只覆盖算法开发部分，面向两个模块：

- `fall_risk`：跌倒风险前置预警算法，承接 `docs/modules/fall_risk/plans/跌倒风险算法研发计划.md`。
- `mental_health`：心理健康风险预警算法，当前已具备行为/睡眠适配、日级聚合、个人基线、量表监督专家、PersonalTrend、融合推理、模型包校验和 HTTP 服务；V3.3.3 主线及 V3.3.4 `OPT-CALIB-001` 至 `REPORT-002` 优化与回退验收均已完成。

系统开发不在本工程范围内。家属端、社区端、账号、消息推送、工单流转、可视化看板等只通过标准 JSON 接口对接。

## 工程结构

```text
configs/                 算法配置
data/                    数据、标注、特征和划分文件
docs/                    当前架构、接口、模块规范、任务和历史归档
reports/                 实验报告、评估结果、复现记录
scripts/                 数据、标注、评估辅助脚本
src/elderly_monitoring/  算法代码
tests/                   算法单元测试
```

## 环境与验证

本项目固定使用 `eldercare-ai` conda 环境。不要直接使用默认 shell 里的 `python` 或 `pytest` 做验证。

完整测试：

```bash
conda run -n eldercare-ai python -m pytest -q
```

环境创建、更新和已验证依赖版本见 `environment-reference.txt`。文档索引见 `docs/README.md`。面向协作代理和自动化工具的项目规则见 `AGENTS.md`。

## 当前研发阶段与优先级

当前跌倒风险模块处于模型化增强阶段。现有 YOLOv8-Pose + ByteTrack、姿态质量控制、规则步态/坐站/近跌倒、个体基线和风险融合仍是可运行 baseline；增强工作将优先把步态、坐站、近跌倒/跌倒事件规则替换为可验证的 TCN、MS-TCN++ 或 ST-GCN 时序模型，同时保留规则安全覆盖和可解释 fallback。模型只有在标签、无泄漏 split、冻结评估协议以及延迟/稳定性证据齐备后才能进入主路径。

当前优先级：

1. 为两个模块分别建立固定验证集、指标脚本和可复现实验报告。
2. 为步态、坐站、近跌倒/跌倒事件模型构建来源完整的时序训练样本，并与现有规则 baseline 做受控对照。
3. 完成真实萤石直播地址、算法会话和后端风险回调联调。
4. 用真实或半真实数据校准模型、规则 fallback、心理健康日级偏离和置信度。
5. 固定两个模块共享的身份、时间、风险等级和事件字段契约，同时保持独立评分与输出。

情绪与社交关注模型已完成 MODEL-001 至 MODEL-006、EVAL-001、TREND-001、FUSION-001/002、ART-001、API-001/002、运行时稳定策略以及 V3.3.4 `OPT-CALIB-001` 至 `REPORT-002`。当前活动包为 ART-001 `MH-20260802-013`，包内默认融合为 `MH-20260802-012 coverage_calibrated`，对外版本为 `mood-fusion-v3.3.3`；MODEL-006 保持 offline-only。V3.3.4 最终候选 `MH-20260804-021` 未通过冻结晋级门，因此 ART-002/API-003 没有创建或接入虚假新包，而是经版本化审计继续使用 ART-001。所有公开数据指标均为参与者级严格 OOF 离线结果；Physiology 新数据仍在申请，当前全局可靠度保持 0，真实摄像头、睡眠仪和 S10 联调仍需另行完成。

## 不做的内容

- 不实现 App、后台、消息推送、账号权限或工单系统。
- 不把心理健康输出定义为医学诊断，只输出风险预警和人工复核建议。
- 模型化增强阶段不把复杂深度模型作为唯一单点依赖；始终保留可解释、可复现的规则安全覆盖和降级路径。

## 跌倒风险 HTTP 服务

安装服务与视觉依赖：

```bash
conda run -n eldercare-ai python -m pip install -e ".[vision,service]"
```

必需环境变量为 `ALGORITHM_API_TOKEN` 和 `CALLBACK_TOKEN`；模型路径由 `MODEL_PATH` 指定，默认是仓库根目录的 `yolov8n-pose.pt`。可选的 `BASELINE_HISTORY_PATH` 指向个体历史 JSONL。启动单 worker 服务：

```bash
ALGORITHM_API_TOKEN=replace-me CALLBACK_TOKEN=replace-me \
conda run -n eldercare-ai uvicorn elderly_monitoring.service.app:app \
  --host 0.0.0.0 --port 8080 --workers 1
```

Docker 镜像不包含模型，运行时只读挂载固定路径：

```bash
docker run --rm -p 8080:8080 \
  -e ALGORITHM_API_TOKEN=replace-me \
  -e CALLBACK_TOKEN=replace-me \
  -v "$PWD/yolov8n-pose.pt:/models/yolov8n-pose.pt:ro" \
  elderly-monitoring-algorithm:0.2.0
```

服务支持单路会话的创建、查询、直播地址更新和停止。输入必须是容器可解码的 `rtsp`、`rtmp`、`http` 或 `https` 地址，不负责转换 `ezopen` 地址。
