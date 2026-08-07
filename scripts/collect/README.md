# 采集脚本目录

只放算法数据采集辅助脚本，例如视频切片、文件命名检查、数据清单生成。

不放设备管理、账号管理、业务系统接入代码。

## 跌倒候选模型 shadow 推理

`run_fall_event_model.py` 对 cleaned 姿态 JSONL 做 4 秒滑窗，加载 provisional `fall_event_candidate_clip_tcn_v1` checkpoint，输出 `fall_event_score` 和 `fall_event_detected`。它只适合开发阶段的候选片段分析，不替换实时 `fall_state` 规则，也不提供连续事件 onset/offset 评估：

```bash
conda run -n eldercare-ai python scripts/collect/run_fall_event_model.py \
  --input data/processed/fall_risk/pose_quality_y8n_v1/cleaned/<video_id>.jsonl \
  --output /tmp/fall_event_predictions.jsonl \
  --checkpoint reports/fall_risk/fall_event_proxy_v2_v3split/pilot-seed42/best_model.pt
```

当前 v3 split 的三 seed pilot 仍是 presence-only candidate-clip proxy，没有训练跌倒方向头；shadow 输出和评估报告均不提供方向结论。低质量窗口返回 `status=unavailable`，不会把缺失输入当成未跌倒。checkpoint 文件是本地忽略产物，README 路径只描述复现实验目录，不表示仓库发布了模型权重。

## 真实直播算法端烟测

`run_fall_live_smoke.py` 用于在业务后端尚未接入时验证真实直播流。脚本自行启动只监听本机的临时回调接收器，通过现有 HTTP 会话运行真实 `StreamReader`、姿态推理、特征分支和停止流程。

直播地址只通过临时环境变量传入，不作为命令行参数，也不会写入报告：

```bash
EZVIZ_STREAM_URL='rtmp://example.invalid/live?temporary-signature' \
conda run -n eldercare-ai python scripts/collect/run_fall_live_smoke.py \
  --model yolov8n-pose.pt \
  --duration-sec 120 \
  --scene-region living_room \
  --report /tmp/ezviz_live_smoke.json
```

报告将直播来源限制为协议、主机和端口，并分别判断流传输与算法分支。没有触发风险回调不算失败；提前结束、会话失败、发生重连、没有持续出帧、没有主目标姿态、没有特征分析或分支出现 `inference_error` 会使对应检查失败。该烟测只证明工程链路，不评估识别准确率、临床有效性或后端集成。
