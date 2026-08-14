# 老年人多模态风险预警算法工程

更新时间：2026-08-12

本工程只覆盖算法开发部分，面向两个模块：

- `fall_risk`：跌倒风险前置预警算法。规则主路径、实时 HTTP 会话和回调链路可运行；v3 事件监督与步态、坐站、近跌倒/跌倒候选模型开发链已建立，但候选模型仍为 provisional/shadow，尚未替换规则主路径。
- `mental_health`：心理健康风险预警算法。行为/睡眠适配、日级聚合、个人基线、风险评分和离线 CLI 已实现；徘徊专项完成了隔离数据转换与人工复核步骤，尚未进入正式 split、模型或现有评分主链。

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

当前跌倒风险模块处于模型化增强阶段。YOLOv8-Pose + ByteTrack、姿态质量控制和规则步态/坐站/近跌倒、跌倒状态与风险融合仍是运行主路径；个体行为基线已完成 Phase 0-1 算法和 Phase 2 纵向 schema/前向 split/消融评估基础设施的合成验收，但真实纵向 observation、`risk_labels` 和 subject profiles 为空，实时链路也尚无完整周期聚合来源，因此该分支失败关闭，不能写成在线已启用或已证明有效。步态、坐站、近跌倒和跌倒事件已经具备训练、validation 复评、shadow 推理或连续输入基础设施；现有实验仍受动作类型门禁、连续背景、老人域、跨来源、冻结 test、延迟和稳定性证据限制，不能描述为正式模型效果或已部署能力。

SCF_MVP_V1 自采批次已完成隔离 candidate、262 段四分支回放和近跌倒 E1-E3 九次开发训练。冻结分区为 P01/P02/P04 auxiliary train、P05 challenge、P03 excluded。E2 的负例触发率改善方向最稳定，但同时明显损失正例代理检出，E1-E3 均为 No-Go；规则和原 checkpoint 保持主路径。详见[执行报告](reports/fall_risk/self_collected_scf_mvp_v1/README.md)。

跌倒数据同时存在两个不同层级：v2 是根标签和发布候选契约，formal 校验仍有 blocker；v3 是由 v2 与哈希绑定项目裁决确定性生成的模型训练契约，当前 fall/near-fall 事件监督门禁通过，但 split 尚未冻结，动作类型门禁仍未通过。两者不能互相替代。

心理健康模块的日级 baseline 已可离线运行；徘徊专项当前只完成安全转换、严格契约和人工联系表复核，正式 split、预处理、分类模型、片段状态机、日级字段及摄像头域验证仍未完成。

当前优先级：

1. 处理跌倒 v2 formal blocker，复核并冻结 v3 事件标签、split 和一次性 test 发布协议。
2. 为候选模型补充连续背景、老人域、跨来源和困难负样本证据，并完成与规则 baseline 的同协议对照、延迟和稳定性验收。
3. 在已通过真实萤石算法端 120 秒烟测的基础上，完成业务后端风险回调、直播地址刷新、弱网和固定硬件长时资源验收。
4. 完成徘徊专项固定 split，再进入预处理和模型训练；不得提前把转换产物称为识别能力。
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

必需环境变量为 `ALGORITHM_API_TOKEN` 和 `CALLBACK_TOKEN`；姿态模型路径由 `MODEL_PATH` 指定，默认是仓库内的 `models/yolov8n-pose.pt`。可选的 `BASELINE_HISTORY_PATH` 指向个体历史 JSONL。`GAIT_MODEL_PATH` 默认保持为空，只有通过步态稳定性替换门禁的 checkpoint 才能配置；当前 provisional 模型不应作为服务默认主分支。启动单 worker 服务：

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
  -v "$PWD/models/yolov8n-pose.pt:/models/yolov8n-pose.pt:ro" \
  elderly-monitoring-algorithm:0.2.0
```

服务支持单路会话的创建、查询、直播地址更新和停止。输入必须是容器可解码的 `rtsp`、`rtmp`、`http` 或 `https` 地址，不负责转换 `ezopen` 地址。为兼容萤石 HEVC-over-FLV，服务默认使用 FFmpeg 后端；镜像内置并在 readiness 中检查 `ffmpeg`/`ffprobe`。直播地址属于临时凭据，服务状态和解码错误不得回显其路径、签名或 Token。
