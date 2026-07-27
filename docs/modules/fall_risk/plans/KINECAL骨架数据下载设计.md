# KINECAL 骨架数据下载设计

## 1. 目标

从 KINECAL `v1.0.3` 获取可用于轻量步态 TCN 的最小数据子集：风险组参与者的逐帧 Kinect 骨架、参与者标签和许可证。下载不包含逐帧深度二进制文件，也不改变现有跌倒风险事件、功能 proxy 或数据 split 的状态。

官方来源：

- 数据页：`https://physionet.org/content/kinecal/1.0.3/`
- 文件根目录：`https://physionet.org/files/kinecal/1.0.3/`
- DOI：`10.13026/vvkp-ct80`
- 许可证：CC0 1.0

## 2. 下载范围

参与者以 `register.csv` 的官方风险分组为准：

- `NF`：33 人，无跌倒史组。
- `FHs`：15 人，单次跌倒史组。
- `FHm`：9 人，多次跌倒史组。

共 57 人。仅下载以下动作的 `skel/*.txt`：

- `3m-walk-Front-View`
- `Get-Up-And-Go-Front-View`
- `STS-5`

KINECAL v1.0.3 在这 171 个目标组合中实际提供 154 个骨架目录、55,509 个逐帧文本文件，共 93,005,183 字节；另有 17 个组合的 `skel/` 目录在 PhysioNet 源站返回 404。下载器必须把这些项目记录为源数据缺失，不得把它们归因于本地网络失败或伪造空序列。

源论文把这三个分组定义为老年风险组，但 `register.csv` 中有 4 条记录的年龄为 60 或 64。下载器保留这些官方分组记录，并在摘要中写入 `age_group_mismatch`。后续实验必须同时报告包含和排除这些记录的敏感性分析，不能静默修改官方分组。

明确排除：

- `depth/DepthUshort*.bin`
- 其余 8 类静态平衡动作
- `sample_set/` 示例副本
- 全量 `SHA256SUMS.txt`，因为它包含所有深度帧且文件约 108 MB

## 3. 本地布局

```text
data/external/kinecal/
  README.md
  raw/
    LICENSE.txt
    register.csv
    kinecal/
      <participant_id>/
        <recording_name>/
          skel/
            <clock_tick>.txt
    download_manifest.jsonl
    download_summary.json
```

`data/external/kinecal/raw/` 属于本地原始数据，不提交 Git。仓库只跟踪数据集 README、下载器、测试和必要的文档入口。

## 4. 下载器

新增 `src/elderly_monitoring/modules/fall_risk/kinecal.py` 承载可测试的解析、下载和验证逻辑，`scripts/annotation/download_kinecal.py` 只负责 CLI 参数与调用。职责限定为：

1. 下载并解析 `register.csv`。
2. 选择 `NF/FHs/FHm` 参与者。
3. 枚举三个目标动作的 `skel/` 文件目录。
4. 使用有限并发下载；若环境安装 `httpx`，复用 HTTP 连接，缺少该可选依赖时回退到标准库传输。
5. 已存在且大小一致的文件直接跳过，支持断点续传。
6. 对网络超时和临时 HTTP 错误做有限次数重试。
7. 生成包含来源 URL、本地路径、字节数和本地 SHA-256 的 manifest。
8. 生成参与者、分组、动作、文件数、总字节数、缺失记录和年龄分组异常摘要。

下载器不得猜测缺失目录、自动改写参与者分组或下载深度文件。任何缺失动作目录都写入摘要并使严格验证返回非零状态。

## 5. 使用边界

KINECAL 骨架为 Kinect 25 关节三维坐标，不能直接假设等同于项目的 COCO 17 点姿态。训练前需要建立显式关节映射，保留躯干、肩、髋、膝和踝等关键关节，并按参与者划分训练、验证和测试集。

建议任务：

- 主任务：`NF` 与 `FHs/FHm` 的跌倒史 proxy 分类。
- 次任务：`clinically-at-risk` 分类。
- 辅助动作：TUG、3 米步行和 STS-5 分支或多任务表征。

不得把跌倒史 proxy 描述为未来跌倒预测真值，也不得把同一参与者的不同动作或窗口分到不同数据分区。

## 6. 验证标准

下载完成后必须验证：

- `LICENSE.txt` 和 `register.csv` 存在。
- 选择人数为 57，分组数量为 `33/15/9`。
- 每个下载文件都位于目标动作的 `skel/` 目录，且没有 `.bin` 文件。
- manifest 的文件数量、总字节数和实际目录一致。
- manifest 中每个 SHA-256 可从本地文件重新计算得到。
- 摘要明确记录 17 个官方缺失动作目录；任何可用文件下载失败、空骨架目录或文件大小不一致时明确失败。
- 下载命令和检查命令统一使用项目 `eldercare-ai` conda 环境。
