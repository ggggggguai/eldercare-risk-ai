# 老年人多模态风险预警算法工程

本工程只覆盖算法开发部分，面向两个模块：

- `fall_risk`：跌倒风险前置预警算法，承接 `docs/modules/fall_risk/plans/跌倒风险算法研发计划.md`。
- `mental_health`：心理健康风险预警算法，当前已具备行为/睡眠适配、日级聚合、个人基线、风险评分和离线 CLI。

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

## 当前优先级

1. 为两个模块分别建立固定验证集、指标脚本和可复现实验报告。
2. 完成真实萤石直播地址、算法会话和后端风险回调联调。
3. 用真实或半真实数据校准跌倒规则、心理健康日级偏离和置信度。
4. 固定两个模块共享的身份、时间、风险等级和事件字段契约，同时保持独立评分与输出。

## 不做的内容

- 不实现 App、后台、消息推送、账号权限或工单系统。
- 不把心理健康输出定义为医学诊断，只输出风险预警和人工复核建议。
- 不把复杂深度模型作为第一版单点依赖，先保留可解释、可复现的轻量主线。

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
