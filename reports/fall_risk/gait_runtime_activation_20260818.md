# 步态 TCN 运行启用记录

日期：2026-08-18

状态：`competition_controlled_activation`

## 决定

项目负责人基于比赛交付时间约束，批准将 action-pretrained hierarchical gait TCN seed 43 接入默认实时步态主分支。该决定只改变运行优先级，不把 `development_provisional` 实验提升为正式效果验收，也不解锁 test。

## 固定产物

- checkpoint：`reports/fall_risk/gait_observable_context_v2/splitv3-3342705-scfaux-v1/pretrained-seed43/best_model.pt`
- SHA-256：`67e6028bfc32cada85bfc3ed440e5f968f96d0f13a3288b6b8c797e8aca8a3fc`
- task：`gait_instability_vs_normal_activity`
- 输入：4 FPS、4 秒、16 帧、14 关节、5 通道，至少 10 个观测帧
- 设备：CPU
- validation：group F1 `0.8571`，walking-gate F1 `0.9189`，正常窗口误报约 `12.04/hour`

## 运行行为

服务启动时常驻加载 checkpoint。兼容且质量合格时，TCN 概率写入 `gait_risk_score` 并进入跌倒模块内部融合；规则分继续写入 `fallback_score` 和解释因子。观测不足、模型输入不兼容、非有限输出或推理异常时，自动输出规则分，并记录 `score_source=rule_fallback` 与 `fallback_reason=model_inference_failed`。

`GET /health/ready` 会在已配置 checkpoint 文件缺失时返回 503。紧急回滚方式是将 `GAIT_MODEL_PATH` 设为空或把服务配置的 `gait_model_path` 改回 `null`，然后重启服务。

## 未解除限制

主评估只有 9 个正动作段，正类来源单一；test、老人域连续背景、冻结 split/协议和长时稳定性均未完成。因此可描述为“步态 TCN 已接入工程链路”，不得描述为“已完成正式泛化验证”或“临床有效”。

## 接入烟测

默认配置加载得到 `task=gait_instability_vs_normal_activity`、`window_frames=16`、`target_fps=4.0`、`min_observed_frames=10` 和模型版本 `gait-tcn-checkpoint-v1:67e6028bfc32`；`GET /health/ready` 返回 200。

真实 `le2i_home_01_video_1` cleaned pose 共 264 条记录，离线步态入口生成 3 个窗口，全部为 `score_source=tcn`。TCN 分数为 `0.0220/0.0337/0.0057`，同步保留的规则对照分为 `0.5703/0.6760/0.6101`。同一输入经实时 `FeatureAssembler` 后，末窗口 `gait_risk_score=0.0057`、`branch_status=valid`、`fusion_enabled=true`、`fallback_reason=null`，证明模型分已进入融合输入而不是仅旁路记录。
