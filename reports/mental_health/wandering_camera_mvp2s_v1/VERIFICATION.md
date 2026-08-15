# M0-CAM-MVP-2S 验证记录

状态：`m0cam_mvp2s_acceptance_status=passed`

## 旧缺口红测

在新增 production 代码前运行：

```bash
conda run --no-capture-output -n eldercare-ai python -m pytest \
  tests/test_wandering_camera_daily_summary.py -q
```

结果：collection 失败；旧代码不存在公开的 `load_validated_wandering_camera_product`，也不存在 daily-summary builder。该失败发生在新功能缺失处，不是 descriptor/hash 假失败。

## 已覆盖攻击

- product final 与 primary 全量回读；descriptor-aware 与 same-length 重哈希篡改；rogue product top-level；
- binding 顶层/session/source/person/presence missing/extra，product SHA/source scope/offset/IANA 漂移；
- duplicate binding/session/product、track 漏绑/重绑/跨 person 重叠、presence 缺失或不足；
- candidate/model/class/threshold/calibration/episode-policy 漂移；
- 自然日、跨午夜、DST spring-forward/fall-back、presence/window/status union 与 status overlap；
- direct 与 wandering-like 分离、unavailable/error 隔离、uncertain 固定 null；
- summary/manifest exact-schema、非空 risk/event、rogue staging、fresh-only、竞争 final 与 owned-staging cleanup；
- canonical bytes、稳定排序、CLI help/参数错误与 Git 外 multi-session cross-midnight synthetic E2E。

## 自动化与 synthetic E2E

最终接受命令与结果：

```text
daily-summary focused: 37 passed
MVP-1/product/daily/primary/episode/QC/inference adjacent: 169 passed
generic daily/baseline/pipeline/CLI isolation: 60 passed, 16 subtests passed
Git-external multi-session cross-midnight real CLI E2E: 1 passed
UTF-8 and MVP-2S related Markdown links: 1 passed
daily-summary CLI help: passed
git diff --check: passed (line-ending warnings only)
eldercare-ai editable: current repository
stage: empty
branch: feat/wandering-data-pipeline
HEAD: bfc3329ad14d145e6b4c3fb836a0a525f30e970b
```

`m0cam_mvp2s_acceptance_status=passed` 只在上述门全部通过后写入。

## 未获得的证据

- 无真实 person/session/time/presence binding 接受；
- 无真人或目标摄像头 evidence；
- 无 camera accuracy/F1/FAR、临床效果或老人域结论；
- 无个人 baseline、risk/action/diagnosis/alert/`AlgorithmEvent`；
- 无 PORTABLE rework、C0/C1、M0-CAM-D、MVP-2R/3/4 进展。
