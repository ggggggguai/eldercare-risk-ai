# 萤石真实直播算法端烟测（2026-07-30）

## 结论

本次真实 RTMP 算法端烟测失败。地址能够被当前 OpenCV/FFmpeg 后端打开并解码，但不是稳定连续直播：70 秒内产生 4 个 stream epoch，每次连接只瞬时返回约 14 帧后结束，服务随后重连。该结果不能作为真实萤石直播联调通过、算法效果或业务后端闭环证据。

完整机器报告生成于 `/tmp/ezviz_live_smoke_20260730.json`。报告已检查，不包含设备路径、查询签名、直播 Token 或原始帧。本记录仅保留完成诊断所需的聚合数据。

## 输入与运行

- 来源：萤石开放平台临时 RTMP 地址
- 脱敏端点：`rtmp://rtmp09open.ys7.com:1935`
- 姿态模型：`yolov8n-pose.pt`
- 场景：`living_room`
- 请求持续时间：70 秒
- 实际监测时间：70.002 秒
- 本机临时回调：启用；未连接业务后端

复现入口：

```bash
EZVIZ_STREAM_URL='由萤石重新生成的短期地址' \
conda run -n eldercare-ai python scripts/collect/run_fall_live_smoke.py \
  --model yolov8n-pose.pt \
  --duration-sec 70 \
  --poll-interval-sec 0.5 \
  --scene-region living_room \
  --report /tmp/ezviz_live_smoke.json
```

## 聚合诊断

| 指标 | 结果 |
|---|---:|
| 总 stream epoch | 4 |
| frame queue put | 56 |
| frame queue get | 12 |
| 丢弃旧帧 | 44 |
| 单 epoch 最大算法处理帧 | 1 |
| 主目标姿态 | 0 |
| 特征分析次数 | 0 |
| 风险回调 | 0 |
| 最终停止状态 | `stopping` |
| 5 秒停止预算后读取线程仍存活 | 是 |

观察到的会话状态为 `starting -> reconnecting -> stopping`。轮询未捕获稳定的 `running` 状态；最后一个 epoch 的目标状态为 `unbound/no_candidates`。步态、坐站、近跌倒和跌倒状态分支均因 `insufficient_frames` 与 `insufficient_effective_fps` 不可用，没有发生分支推理错误。

帧队列总计收到 56 帧，平均约 14 帧/epoch，与独立 OpenCV 解码检查每次得到 14 帧一致。每次连接返回的是一小段缓冲帧，而不是按真实时间持续到达的直播帧，因此 latest-frame 队列会瞬时丢弃旧帧，姿态窗口无法形成。停止时线程仍阻塞于下一次流打开或读取，超过当前 5 秒停止预算。

## 后续门槛

1. 在萤石侧关闭其他播放器连接、禁用休眠或省电模式，并重新生成直播地址。
2. 优先生成 H.264、720p 或以上的连续 RTMP/HTTP 流，不手工修改签名查询参数。
3. 新地址先通过 120 秒烟测：单一 epoch、`last_frame_at` 持续变化、有效采样率不低于 6 FPS、停止预算内线程退出。
4. 流稳定后再要求测试人员完整入画，验收主目标姿态、特征分析和各分支状态。
5. 算法端通过后再接业务回调，验证 `event_id` 幂等与地址刷新。

## H.264 与协议复测

同日通过萤石开放平台重新生成直播地址，参数为高清主码流、H.264、30 分钟有效期。H.264 RTMP 再次运行 70 秒，结果与 H.265 地址一致：4 个 stream epoch、frame queue `put=56/get=12/dropped=44`、单 epoch 最大算法处理帧为 1，未形成主目标姿态或特征分析，停止读取线程仍超过 5 秒预算。

随后对同批生成的标准流做协议探测：

| 协议 | 结果 |
|---|---|
| H.264 RTMP | 每次连接约返回 14 帧后结束并重连 |
| HLS | 0.881 秒打开，只解码 8 帧后结束 |
| HTTP-FLV | 当前 OpenCV/FFmpeg 后端无法打开 |

HLS 主清单只包含 1 个 4 秒分片，并明确出现 `#EXT-X-ENDLIST`：

```text
#EXTM3U
#EXT-X-VERSION:3
#EXT-X-MEDIA-SEQUENCE:0
#EXT-X-ALLOW-CACHE:YES
#EXT-X-TARGETDURATION:4
#EXTINF:4.000,
#EXT-X-ENDLIST
```

`ENDLIST` 证明开放平台本次返回的是有限播放列表而不是持续直播。编码从 H.265 切换到 H.264 没有改变短流行为，因此不能继续把问题归因于 YOLO、姿态分支或 H.265 解码兼容。下一步应先在萤石 App/开放平台内置播放器验证是否同样只播放约 4 秒；若 App 连续而开放平台短流，应携带设备序列号、生成时间和上述脱敏协议证据提交萤石工单，排查标准流发布、账号权限或设备固件状态。
