# TopoWander-MPT M0-S 三种子稳定性

## 结论

M0-S development validation 总门禁通过，固定主种子冻结为 `20260731`。本轮严格只运行预注册的 `20260731 / 20260801 / 20260802` 三个种子，没有增加种子、挑选最好种子、集成或重训 RF/TCN baseline。M1 未执行，模型组件未替换。

三个 run 使用同一份最终源码，源码包 SHA-256 为 `70b0b92c9e8aea6b49f500dcaee11e16f266a58da88903e0c4d3dbea7f38feb8`。每个 run 都保存逐 epoch model/AdamW/CPU RNG checkpoint、history、latest/best 指针、独立新进程 fresh reload、逐样本 validation predictions 和 metrics。

| Seed | WP 四分类 macro-F1 | 来源等权二分类 macro-F1 | 最低类别 recall | Best / last epoch | 训练秒数 | 固定 batch=64 平均延迟 | model state |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260731 | 0.987497 | 0.997238 | 0.966667 | 5 / 13 | 432.81 | 1263.45 ms | 838,934 B |
| 20260801 | 0.987497 | 0.983848 | 0.941176 | 4 / 12 | 405.77 | 1238.82 ms | 838,934 B |
| 20260802 | 0.987497 | 0.997238 | 0.966667 | 8 / 16 | 520.92 | 1235.95 ms | 838,934 B |

三种子汇总：

- WP 四分类 macro-F1：mean `0.987497`，population std `0.000000`，min `0.987497`；
- 来源等权二分类 macro-F1：mean `0.992774`，population std `0.006312`，min `0.983848`；
- 全部已报告类别 recall 的跨 seed 最小值为 SmartCare `direct_or_non_wandering` 的 `0.941176`，其余最小值不低于 `0.966667`；
- 训练时长：mean `453.17 s`，population std `49.16 s`，范围 `405.77–520.92 s`；
- 固定 batch=64 CPU 推理平均延迟：整批 mean `1246.07 ms`，折合单样本 mean 约 `19.47 ms`（约 `51 samples/s`）；该数值不包含视频解码、人体检测、tracking、切窗或 episode 聚合，不能当作端到端摄像头延迟。

因此三个种子均满足两项 macro-F1 `>=0.95` 且所有类别 recall `>=0.90` 的硬门禁。

## 错误稳定性

- WP 四分类：三个种子各错 3 条，intersection=union=3，错误集合完全稳定；
- 二分类：错误数为 `1 / 2 / 1`，intersection=1、union=2；唯一额外波动样本只出现在 seed `20260801`。

完整 sample ID、逐 seed 错误集合和代表性错误位于 `evidence/aggregate-v1/stability_report.json` 及各 run 的 `fresh_reload/predictions.jsonl`。

## 与已有 RF/TCN 的同种子配对

下表的 baseline 预测均直接复用 Step 5/6 已保存的同种子 validation predictions，并在聚合报告中绑定了原路径和 SHA-256；没有重训 baseline。

| Seed | Baseline | Task | M0-S error | Baseline error | Both wrong | M0-S wrong / baseline correct | M0-S correct / baseline wrong |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20260731 | RF | four-class | 3 | 6 | 1 | 2 | 5 |
| 20260731 | RF | binary | 1 | 9 | 0 | 1 | 9 |
| 20260731 | TCN | four-class | 3 | 18 | 3 | 0 | 15 |
| 20260731 | TCN | binary | 1 | 1 | 1 | 0 | 0 |
| 20260801 | RF | four-class | 3 | 7 | 1 | 2 | 6 |
| 20260801 | RF | binary | 2 | 9 | 0 | 2 | 9 |
| 20260801 | TCN | four-class | 3 | 11 | 1 | 2 | 10 |
| 20260801 | TCN | binary | 2 | 2 | 1 | 1 | 1 |
| 20260802 | RF | four-class | 3 | 6 | 1 | 2 | 5 |
| 20260802 | RF | binary | 1 | 8 | 0 | 1 | 8 |
| 20260802 | TCN | four-class | 3 | 12 | 1 | 2 | 11 |
| 20260802 | TCN | binary | 1 | 1 | 1 | 0 | 0 |

## 工程加固

- performance config 显式列出固定三种子和 primary seed；`train/evaluate` CLI 必须显式传 `--seed`；
- 实际 seed 写入 resolved config、checkpoint metadata、fresh metrics 和每条 prediction；训练 scratch 初始化、shuffle 和 dropout RNG 均受该 seed 控制；
- 完整 epoch checkpoint 通过临时目录原子 rename 成为恢复事实源；resume 会验证连续 checkpoint 并重建不一致的 history/latest/best；
- 故障注入回归模拟 epoch 2 checkpoint 已提交、history/latest/best 仍停在 epoch 1，恢复后正确对齐至 epoch 2 并可继续加载；
- checkpoint 保存并恢复 CPU torch RNG。正式配置固定 CPU，因此明确记录 `cuda_resume_supported=false`，不声称 CUDA RNG 恢复能力；
- 最终源码的徘徊完整测试为 `229 passed`，其中 M0-S 窄测试为 `12 passed`；overfit smoke 在 epoch 31 通过。

## 数据访问边界

共享 preprocessing container 包含 WP test 行，并在 development materialization 时为全容器完整性校验解析这些行；这不等于向训练开放 test。development accessor 只暴露 train/validation，M0-S 物化包只含 1,257 train + 278 validation。M0-S 的训练、validation、fresh reload 与选点路径没有暴露、物化、tensorize、推理或计分 WP test；WP raw、SmartCare official/raw 和 sealed camera 未进入本次模型流程。

仓库级测试会调用 frozen accessor 检查 240 条 WP test 的访问契约，Step5/Step6 也已在该固定 split 上报告 RF/TCN 结果。因此下一阶段只能称为“TopoWander-MPT 固定候选对 WP public holdout 的一份最终计分结果”，不能称为数据集首次解封或项目级严格盲测。

本结果只证明固定公开轨迹 development validation 的三种子稳定性，不是摄像头、真实老人、产品或临床有效性证据。TopoWander-MPT 的 WP public-holdout 候选级计分尚未执行，必须先完成“新候选模型路径不使用 test”的 Release-Prep，再在独立授权下运行。

## 审计后的 primary 候选身份

后续发布准备必须固定使用 M0-S primary，而不是 validation 分数更高的历史 M0：

- seed：`20260731`；best epoch：`5`；
- best model：`runs/seed-20260731/checkpoints/epoch-0005/model_state.npz`，SHA-256 `94c3c22d4caa38ece347d6a10f440b9fb3f7067b6c791ac1f259ce7efe69c031`；
- checkpoint metadata SHA-256：`47efdb61fa5389d9b23d19ae9b39099f0e27e2eab8a3b30581a1584e0a4d4f48`；
- `best.json` SHA-256：`41ab9495ba42a92c4aae6acb7725533791459c864d52aa0f6b413adaadadb231`；
- source bundle SHA-256：`70b0b92c9e8aea6b49f500dcaee11e16f266a58da88903e0c4d3dbea7f38feb8`。

Release-Prep 仍需生成独立 candidate manifest，把不可变的训练候选身份与新增 loader/evaluator 的发布实现身份分开记录，并绑定上述模型、forward/performance config、预处理/split 身份、类别顺序和输入契约。新 release 模型/evaluator路径不得用 WP test 生成预测或分数；既有 accessor 契约回归不属于候选计分。

## 证据入口

- `evidence/aggregate-v1/stability_report.json`：机器可读总报告、全类别 recall、成本、错误交并集和 paired counts；
- `runs/seed-*/resolved_config.json`：实际 seed、数据身份、环境和源码包哈希；
- `runs/seed-*/checkpoints/`：逐 epoch model/optimizer/RNG checkpoint 及 latest/best；
- `runs/seed-*/fresh_reload/`：独立重载 metrics、predictions、confusion matrix、history 和单样本推理；
- `evidence/overfit-smoke/`：最终源码上的 overfit smoke；
- `VERIFICATION.md`：命令、拷贝校验和原 M0 不变性核验。
