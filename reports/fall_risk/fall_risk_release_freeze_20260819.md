# 跌倒风险正式比赛交付冻结决定

日期：2026-08-19  
状态：`competition_release_frozen`  
release ID：`fall-risk-competition-v1-20260819`

## 决定

项目负责人确认当前模型与规则组合保持不变，并接受现有证据边界。当前跌倒风险模块进入“正式模型替换完成并冻结”阶段，以 `configs/modules/fall_risk_release_v1.yaml` 作为正式比赛交付版本的机器事实源。

本次冻结的运行基线包括：YOLOv8-Pose + ByteTrack；步态 pretrained seed 43；坐站 seed 42；跌倒事件 seed 42/43/44 概率集成；近跌倒规则主路径；个体基线统计分支；规则风险融合。所有模型分支继续保留输入不足、契约不兼容和推理失败时的规则 fallback 或失败关闭行为。

运行配置中的 `experimental_tcn` 名称为既有兼容枚举，不再表示该 checkpoint 可在 v1 内任意替换。正式比赛部署必须与冻结清单中的路径和 SHA-256 一致。

## 冻结边界

- v1 内不再更换 checkpoint、模型路径、特征契约、阈值或服务运行配置。
- 后续优化必须创建新的 release ID，在独立配置和报告中验证后再切换。
- 密钥轮换和部署端点变化不改变算法 release，但不得写入冻结清单。
- 紧急回滚到规则 fallback 属于安全动作，不构成新模型发布。

## 证据解释

负责人批准解决的是比赛交付版本选择，不追溯生成缺失数据，也不改写历史实验结果。v2 formal blocker、v3 动作类型门禁、研究 split/协议、test、连续背景、老人域和纵向真值的既有状态继续按原机器报告记录。历史报告中的 `development_provisional` 或 `No-Go` 仍是当时研究协议下的真实结论。

因此可以表述为“当前混合模型已完成正式比赛交付替换并冻结”，但不能表述为“临床有效”“所有泛化验证已经完成”或“历史未执行的 test 已通过”。

## 验证

冻结完整性由 `tests/test_fall_risk_release_freeze.py` 校验。测试会重新计算服务配置和六个二进制产物的 SHA-256；任一固定文件变化都会失败，要求创建新 release。

