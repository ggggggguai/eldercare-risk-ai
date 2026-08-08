# 老年人多模态风险预警算法工程

更新时间：2026-08-08

本工程只覆盖算法开发部分，面向两个模块：

- `fall_risk`：跌倒风险前置预警算法。规则主路径、实时 HTTP 会话和回调链路可运行；v3 事件监督与步态、坐站、近跌倒/跌倒候选模型开发链已建立，但候选模型仍为 provisional/shadow，尚未替换规则主路径。
- `mental_health`：心理健康风险预警算法。行为/睡眠适配、日级聚合、个人基线、风险评分和离线 CLI 已实现；徘徊专项已完成固定 split、预处理、RF/纯 TCN 对照、bbox-only 合成摄像头链、合成污染兼容性审计，以及步骤 9/9a TopoWander-MPT 纯前向契约与独立 Git 检查点；目标模型尚未训练，也未接入现有评分主链。

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

标准环境定义见 `environment.yml`，本机已验证依赖版本见 `environment-reference.txt`。运行脚本前应确认 editable 安装指向当前仓库：

```bash
conda run -n eldercare-ai python -m pip show elderly-monitoring-algorithms
```

文档唯一总入口见 `docs/README.md`，面向协作代理和自动化工具的项目规则见 `AGENTS.md`。

## 当前研发阶段与优先级

当前跌倒风险模块处于模型化增强阶段。YOLOv8-Pose + ByteTrack、姿态质量控制、规则步态/坐站/近跌倒、个体基线和风险融合仍是运行主路径；步态、坐站、近跌倒和跌倒事件已经具备训练、validation 复评或 shadow 推理的候选实现。现有实验仍受动作类型门禁、连续背景、老人域、跨来源、冻结 test、延迟和稳定性证据限制，不能描述为正式模型效果或已部署能力。

跌倒数据同时存在两个不同层级：v2 是根标签和发布候选契约，formal 校验仍有 blocker；v3 是由 v2 与哈希绑定项目裁决确定性生成的模型训练契约，当前 fall/near-fall 事件监督门禁通过，但 split 尚未冻结，动作类型门禁仍未通过。两者不能互相替代。

心理健康模块的日级 baseline 已可离线运行；徘徊专项步骤 2–8 已关闭，步骤 9/9a 已恢复 `step9_forward_contract_verified` 与 `step9a_exact_config_fail_closed_verified`，最终 `model.py` SHA-256 为 `11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331`，并已形成独立 Git 检查点。步骤 10 已冻结 pair-aware 数据/目标路线、wrapper 调用图、CLI、早停、20-file DAG、config/artifact 字段级 schema 和 canonical bytes；Linux/x86_64 `2.13.0+cu130` runtime 候选已完成工件级重建，但尚待 Git 检查点和新 checkout 复验。TopoWander-MPT 的红测/源码、训练效果、校准/OOD、片段状态机、日级字段、现有评分主链接入及授权摄像头验证仍未完成。

当前优先级：

1. 处理跌倒 v2 formal blocker，复核并冻结 v3 事件标签、split 和一次性 test 发布协议。
2. 为候选模型补充连续背景、老人域、跨来源和困难负样本证据，并完成与规则 baseline 的同协议对照、延迟和稳定性验收。
3. 完成真实萤石直播、算法会话与业务后端风险回调联调，以及固定硬件长时资源验收。
4. 步骤 9+9a 独立检查点已形成；PyTorch `2.13.0+cu130` runtime 候选已完成工件级重建，但仍待 Git 检查点和新 checkout 复验，关闭该门禁后才可按步骤 10 的既定 exact schema 建立红测。前向契约和 runtime 重建仍不得称为识别效果。
5. 继续保持两个模块独立评分、独立验证和独立输出，只共享 `AlgorithmEvent` 字段契约。

详细实现状态以 `docs/architecture/算法工程骨架.md` 和两个模块 README 为准；尚未完成的工作只在 `docs/tasks/README.md` 维护，实验数值以 `reports/` 下对应报告为准。

## 不做的内容

- 不实现 App、后台、消息推送、账号权限或工单系统。
- 不把心理健康输出定义为医学诊断，只输出风险预警和人工复核建议。
- 模型化增强阶段不把复杂深度模型作为唯一单点依赖；始终保留可解释、可复现的规则安全覆盖和降级路径。

## 跌倒风险 HTTP 服务

安装服务与视觉依赖：

```bash
conda run -n eldercare-ai python -m pip install -e ".[vision,service]"
```

必需环境变量为 `ALGORITHM_API_TOKEN` 和 `CALLBACK_TOKEN`；姿态模型路径由 `MODEL_PATH` 指定，默认是仓库根目录的 `yolov8n-pose.pt`。可选的 `BASELINE_HISTORY_PATH` 指向个体历史 JSONL。`GAIT_MODEL_PATH` 默认保持为空，只有通过步态稳定性替换门禁的 checkpoint 才能配置；当前 provisional 模型不应作为服务默认主分支。启动单 worker 服务：

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
