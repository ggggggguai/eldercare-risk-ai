# 采集脚本目录

只放算法数据采集辅助脚本，例如视频切片、文件命名检查、数据清单生成。

不放设备管理、账号管理、业务系统接入代码。

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
