# 萤石 C6C 办公室徘徊识别视频拍摄与标注手册

版本：v1.3｜更新时间：2026-08-15｜适用场景：办公室平地、固定萤石 C6C、首轮 1 名出镜者

> 本文默认所有视频均已获得拍摄和使用授权，只讲怎样拍出模型需要的视频，以及怎样制定人工标签。

## 1. 最终需要交付什么

拍摄与标注人员最终交付四样东西：

1. 可正常播放的原始视频；
2. 视频清单和拍摄记录；
3. 每段行为的开始时间、结束时间和标签；
4. 画面、遮挡、出画或其他质量问题说明。

本轮视频主要用于：

- 检查萤石 C6C 固定机位能否产生稳定人体轨迹；
- 获得 `direct / pacing / lapping / random` 四种轨迹形态；
- 获得打电话、找东西、清洁、锻炼、搬运等有目的困难负例；
- 获得普通办公室自然活动；
- 为后续 tracking、Camera QC 和模型开发评估准备数据。

拍摄人员不负责训练模型、修改阈值或根据模型预测改标签。

## 2. 最简执行流程

按下面顺序做即可：

1. 固定 C6C，关闭自动跟随、巡航和变焦；
2. 调整到人物全身、双脚和主要行走地面都清楚可见；
3. 录一条 20–30 秒取景检查；
4. 录一条约 90 秒工程样片；
5. 查看样片是否出画、挡脚、模糊或发生云台转动；
6. 样片合格后，再拍四类轨迹、困难负例和自然活动；
7. 每拍一条立即命名并填写拍摄记录；
8. 拍完后按 episode/时间段完成人工标签表。

不要一开始连续拍几十条。先验证一条，再逐步扩大。

### 2.1 开拍前必须先分清三种视频

这是本手册最重要的时长规则。短行为片和连续模型片用途不同，不能用“所有文件加起来的总时长”互相替代。

| 文件类型 | 怎样拍 | 用途 | 在 episode-first 路线中的位置 |
|---|---|---|---|
| `EPISODE` 完整行为片 | 一条文件拍一个自然、完整的行为段；实际走完就结束，direct 可能只有几秒 | episode-first 形态分类、人工标注、四类样本库 | **主输入。** 完整路径按弧长重采样为 80 点；不要求 40 秒 |
| `MODEL-WINDOW` 连续诊断片 | 同一个原始文件连续录像，不暂停、不剪接；至少 40 秒 | 检查 tracking/QC 和旧 40 秒组合轨迹结果 | **仅 legacy 诊断。** 不拿它覆盖 episode 标签，也不直接作告警 |
| `NATURAL` 自然活动长片 | 连续记录 10–30 分钟自然办公室活动 | 自动 episode 边界、普通负例、困难片段和误触发/小时 | 按 episode 与独立 continuous-negative timeline 使用 |

必须记住：

- 12 条 5 秒视频合计 60 秒，仍然不是一条连续 60 秒视频；
- 不要把彼此独立的短片剪接成一条“连续轨迹”；
- episode-first 的 80 点来自一条完整路径的空间弧长重采样，不等于 80 帧，也不等于 40 秒；
- legacy 40 秒诊断仍使用 80 个半秒时间桶，两种 80 点的用途必须分开记录；
- 清楚、完整的短 direct 可以单独进入 episode-first shape 模型；正式 EP1A runner 已完成并复现 D01/S00。当前优先是先标注和检查已有 P01/L01/R01，再只补缺少的 pacing/lapping/random 有效 episode。

### 2.2 本轮现有短 direct 怎样处理

已经拍好的几秒 direct 不要删除，也不必为了凑时长重新演一遍。只要人物全身可见、从一个位置自然走到另一个位置、开始和结束清楚，就保留并按一个 direct episode 标注。

接下来优先补完整 pacing、lapping、random episode 和一段自然活动长片。不要把同一路线反复往返后仍标成 direct；反复折返的实际形态应是 pacing。

### 2.3 现在这批最少补拍什么

先做下面 4 组，拍完检查后再决定是否扩批：

1. `E-P01`：约 2 条自然完整 pacing；
2. `E-L01`：约 2 条自然完整 lapping；
3. `E-R01`：约 2 条自然完整 random；
4. `N01`：1 条 10–30 分钟自然办公室活动。

每条 episode 实际完成就结束，不写死秒数。现有 D01 direct 和 S00 连续样片全部保留；暂时不要求补拍“连续做 40 秒的 direct”。

## 3. 模型需要怎样的画面

### 3.1 必须满足

- 摄像头全程固定，画面不追着人物转动；
- 人物从头到脚完整可见；
- 双脚与地面的接触位置清楚；
- 主要行走区域有连续可见的地面；
- 人物在画面中不能太小，也不能近到头脚被截断；
- 人物在主要行为段内不频繁贴近画面边缘；
- 正常样片中，腿脚不被物体长时间遮挡；
- 光线稳定，人物和背景有足够对比；
- 一条视频中画面方向、清晰度和帧率保持稳定。

### 3.2 理想画面

理想画面应同时看到：

- 人物完整轮廓；
- 双脚落点；
- 人物前后左右可以活动的地面；
- 路线两端和转向区域；
- 绕行时使用的中心区域或参照物；
- 画面四周留有一定余量。

第一批先得到“全身持续可见、轨迹清楚”的干净样片。遮挡、进出画面和较暗光线作为后续单独的困难样片拍摄。

### 3.3 机位怎样调整

你提供的实拍画面中，人物全身和双脚可见，这是可用的起点。正式走动时还要通过取景样片确认：人物走到左侧、中间、右侧和路线转向处时，头脚仍完整，地面轨迹仍连续可见。

机位不写死具体厘米数，以最终画面为准：

- 方向尽量沿房间较长或较开阔的方向取景，让路线在画面里有足够长度；
- 使用适度俯角，同时看到人物全身和主要行走地面；
- 近处站位不能切头切脚，远处站位也不能小到难以辨认；
- 左右两端和转身位置要留余量；
- 如果当前机位已经满足这些条件，就保持不动，不必为了追求固定参数重新调整。

### 3.4 服装建议

当前办公室为浅色墙面和浅色地面，建议穿与背景反差较大的纯色衣服和鞋。不要让上衣、裤子和鞋都与背景接近，否则人体边界和脚部位置不容易观察。

## 4. 萤石 C6C 设置

### 拍摄时固定设置

- 关闭智能跟踪/自动跟随；
- 关闭自动巡航和自动转向；
- 拍摄中不手动控制云台；
- 不使用数字变焦；
- 使用稳定的录像清晰度；
- 直接导出原始视频，不使用手机录屏代替原视频；
- 第一条导出样片先检查实际 FPS、时长和可解码性；
- 实际名义帧率应不低于 15 FPS；
- 一条主要样片中不要反复切换普通画面和夜视画面。

摄像头位置、方向或俯角发生明显变化后，应把后续视频记为新的机位，例如 `SETUP-02`，不能与原机位混在一起。

## 5. A、B 点怎样使用

A、B 只是演员记住的两个路线参考点，不需要在萤石 App 中标记，也不要求出现在最终视频里。

可以用：

- 两个地砖交点；
- 两小段美纹纸；
- 拍摄前临时放置、正式开拍前撤掉的纸张；
- 拍摄者口头记住的两个位置。

A、B 应满足：

- 人站在两个点时都完整可见；
- 两点之间能够走出几个正常步幅；
- 两端都不贴画面边缘；
- 路线中双脚持续可见；
- 转身时身体不会明显出画。

A、B 不是强制落脚点。演员只需要在这个范围内自然活动，不必每次精确踩到同一位置。

## 6. 一个人怎样完成拍摄

一名出镜者就可以完成第一轮：

1. 在 App 中开始录像；
2. 进入画面后站定几秒；
3. 完成宽泛的动作任务；
4. 结束后再次站定几秒；
5. 离开画面并停止录像。

第一轮不要求额外协助人员，也不要求多人同时出现在画面中。

## 7. 第一轮先拍两条样片

### 7.1 `frame_check`：20–30 秒

目的：只检查取景范围。

拍法：

- 人物分别站到活动区左侧、中间、右侧；
- 每个位置自然转身一次；
- 检查头顶、双脚和身体两侧是否都有余量。

文件示例：

```text
VID-B01-0001_frame-check.mp4
```

### 7.2 `tracking_smoke`：60–120 秒

推荐首条约 90 秒。

拍法：

- 开始先站定约 5–10 秒；
- 中间自然走动或自然来回走动约 60–70 秒；
- 结束再站定约 5–10 秒。

这不是要求精确按秒。关键是保留一段连续、清楚、可观察的视频，用于检查 tracking、进出画、停留和 episode 边界。它不是 clean shape episode，也不要求整段只标一种形态。

文件示例：

```text
VID-B01-0002_tracking-smoke.mp4
```

样片通过后再拍后续数据。

### 7.3 `MODEL-WINDOW`：legacy 40 秒组合轨迹诊断片（可选）

#### 所有 `MODEL-WINDOW` 都必须做到

- 一个原始视频文件连续录像至少 40 秒，推荐约 70 秒；
- 录像中途不停止、不暂停、不剪接；
- 同一个人物从头到尾保持在画面内，tracking 不应中断；
- 文件开头和结尾都应能看到人物，不能只在中间出现几秒；
- 拍完后先确认文件时长，再交给模型开发人员；结果只作 long-context diagnostic。

#### pacing、lapping、random 的连续诊断片

如果行为本来自然会持续较长时间，可以按下面的宽泛结构拍；不要为了 legacy 窗口强迫延长动作：

```text
开始站定约 5 秒
持续形成该类轨迹约 50–60 秒
结束站定约 5 秒
```

其中“持续形成”是指整段的主要轨迹特征保持一致，不要求步速、转角、端点或每一圈完全相同。自然的短暂停顿可以保留，但不要在一条 clean-class 模型片里中途改成另一种轨迹。

#### direct 不需要拍成长 clean window

小办公室里，一次自然 direct 通常只有几秒，不能为了填满 40 秒而反复 A↔B。direct 分两种用途处理：

1. **标注用 direct**：拍一个自然、完整的 A→B，走完即结束；几秒完全可以，作为 `EPISODE` 短行为片保留；
2. **工程用 direct 连续片**：连续录约 50–70 秒，人物先站定，再完成一次自然 direct，之后留在画面内站定或进行普通非轨迹动作。只把实际 A→B 的几秒标成 direct，不能把整条文件标成 direct。

第 2 种只能检查连续 tracking、自动边界和 legacy 40 秒组合结果，不算 clean direct episode。正式 shape 评估直接使用第 1 种完整 direct；拍摄者不要伪造动作或修改标签。

## 8. 第一小批拍摄清单

下面是建议的第一小批，不要求一天全部完成。文件名前的 `E` 表示短行为片，`W` 表示连续模型片。

| 镜头组 | 文件类型 | 内容 | 建议数量 | 单条原始文件时长 | 主要用途 |
|---|---|---|---:|---:|---|
| F00 | 取景检查 | 左、中、右取景检查 | 1 | 20–30 秒 | 仅检查画面 |
| W-S00 | MODEL-WINDOW | 自然行走工程样片 | 保留现有 | 60–120 秒连续录像 | tracking / legacy 40 秒诊断 |
| E-D01 | EPISODE | 一次自然 direct | 保留现有短片；再按需要补充 | 按实际走完，通常几秒 | direct episode / ordinary negative |
| E-P01 | EPISODE | 一段清楚 pacing | 2 | 按实际完整行为 | pacing episode |
| E-L01 | EPISODE | 一段清楚 lapping | 2 | 按实际完整行为 | lapping episode |
| E-R01 | EPISODE | 一段清楚 random | 2 | 按实际完整行为 | random episode |
| W-P01 | MODEL-WINDOW | 持续 pacing | 可选 | 自然持续且至少 40 秒 | legacy 长上下文诊断 |
| W-L01 | MODEL-WINDOW | 持续 lapping | 可选 | 自然持续且至少 40 秒 | legacy 长上下文诊断 |
| W-R01 | MODEL-WINDOW | 持续 random | 可选 | 自然持续且至少 40 秒 | legacy 长上下文诊断 |
| W-D01 | MODEL-WINDOW | 一次自然 direct，加前后可见时间 | 可选 | 50–70 秒连续录像 | 自动边界/组合错误诊断，不作 clean direct 评估 |
| H01 | MODEL-WINDOW | 打电话时走动 | 1 | 60–120 秒连续录像 | purposeful hard negative 候选 |
| H02 | MODEL-WINDOW | 找东西 | 1 | 60–120 秒连续录像 | purposeful hard negative 候选 |
| H03 | MODEL-WINDOW | 清洁或整理 | 1 | 60–120 秒连续录像 | purposeful hard negative 候选 |
| H04 | MODEL-WINDOW | 重复搬运轻小物品 | 1 | 60–120 秒连续录像 | purposeful hard negative 候选 |
| H05 | MODEL-WINDOW | 安全的步行式锻炼 | 1 | 60–120 秒连续录像 | purposeful hard negative 候选 |
| N01 | NATURAL | 连续自然办公室活动 | 1 | 10–30 分钟连续录像 | ordinary/natural activity |
| Q01 | MODEL-WINDOW | 轻度遮挡 | 1 | 60–120 秒连续录像 | 质量挑战 |
| Q02 | MODEL-WINDOW | 进出画面 | 1 | 60–120 秒连续录像 | 质量挑战 |
| Q03 | MODEL-WINDOW | 较暗光线 | 1 | 60–120 秒连续录像 | 光线挑战 |

现有 `W-S00` 和 `E-D01` 直接保留。下一批先完成 `E-P01 / E-L01 / E-R01`，再拍 `N01`、困难负例和质量挑战；`W-P01 / W-L01 / W-R01` 只在行为能自然持续时选拍。

表里的 60–120 秒是原始文件连续录像时长。只有标为 `MODEL-WINDOW` 的文件要求连续达到时长；标为 `EPISODE` 的行为按实际自然完成，不能为了凑秒数改变轨迹形态。

## 9. 四种轨迹形态怎样拍

导演只说明要形成的整体轨迹特征，不规定精确步数、速度或转角。

### 9.1 `direct`

含义：人物总体从一个位置走向另一个位置，没有持续反复折返，也没有反复绕圈。

宽泛提示：

> 从一个位置自然走到另一个位置，像平时去取东西、换位置或走向门口一样。

注意：

- 不要为了凑时长反复 A↔B；反复折返会更接近 pacing；
- 小办公室中的 direct 往往较短；
- 单个 direct 没有固定秒数，按正常速度自然走完整段即可；
- 专门拍 `EPISODE` 短行为片时，建议一条文件只放一个完整 direct；
- `MODEL-WINDOW` 或 `NATURAL` 长视频里可以出现多个彼此分开的 direct 段；两段之间的站立不标成 direct；
- 每个 direct 段在标注时分别记录开始和结束时间；
- 如果空间不足，后续可在更长的走廊补拍。

### 9.2 `pacing`

含义：人物在大致相同的带状区域反复来回，方向反转明显。

宽泛提示：

> 在这片区域自然来回走一会儿，端点和转身方式由自己决定。

注意：

- 不要求每次都踩准 A、B；
- 不要求每次步数和速度完全相同；
- 可以有自然停顿；
- 主要形态应能看出重复折返。

### 9.3 `lapping`

含义：人物反复形成闭合或近闭合的绕行路线，经常回到先前经过的区域。

宽泛提示：

> 围绕一个中心区域或固定参照物自然绕行，路线大小可以变化，不要求画标准圆。

注意：

- 一次经过不算 lapping；
- 应能看出重复绕行或多次回到相同区域；
- 绕行过程中大部分时间仍应看见双脚；
- 路线不需要每圈完全相同。

### 9.4 `random`

含义：人物向多个方向不规则移动，没有长期稳定的往返主轴，也没有持续重复的闭环。

宽泛提示：

> 在可用区域里自然访问不同位置和改变方向，不要一直沿同一条线来回，也不要一直绕同一个中心。

注意：

- “自由发挥”不自动等于 random；
- 如果实际明显形成 pacing 或 lapping，必须按实际形态标；
- 如果多种形态边界清楚，就分段标注；
- 如果确实无法判断，标为 `unknown/uncertain`；
- 拍摄清单可以写 `random-like`，正式形态标签使用 `random`。

## 10. 不要把动作拍得太机械

为了让数据更接近真实画面：

- 允许演员使用正常速度；
- 允许自然停顿和自然转身；
- 允许不同 take 的起点、终点略有变化；
- 允许每次路线不完全重合；
- 不要求夸张摆臂、夸张迟疑或模拟疾病；
- 不要求每次正好持续同样时间；
- 不要把公开数据集中的几何图形逐点复刻到地面；
- 只要主要轨迹形态清楚，就不必把每一步写死。

同时也不能只说“随便走”。每条视频仍要有一个清楚的轨迹目标或自然活动目的，否则后续很难标注。

## 11. 有目的困难负例怎样拍

有目的困难负例是指：人物为什么这样走是清楚的，但实际轨迹可能很像 pacing、lapping 或 random。

### 11.1 电话走动

让人物模拟正常通话并自然走动。不要规定一定来回几次。拍完以后看实际轨迹是 pacing、random 还是其他形态。

### 11.2 寻找物品

让人物在几个位置寻找一个已知的普通物品。人物可以自然改变方向、回到之前的位置或停下来观察。

### 11.3 清洁或整理

让人物在可用活动区域内完成简单整理或清洁。活动路线可以形成往返、绕行或不规则移动。

### 11.4 重复搬运

让人物在两个或多个位置之间搬运轻小物品。物品应安全、容易拿取，不影响自然行走。

### 11.5 步行式锻炼

让人物进行安全的室内步行活动。不要奔跑，不要安排跌倒或碰撞动作。

困难负例的关键标签规则：

- 轨迹形态按画面实际填写；
- 行为目的单独填写；
- 有目的不能把 pacing 改成 direct；
- 例如打电话来回走仍是 `pacing + purposeful`。

## 12. 连续自然活动怎样拍

自然活动不需要逐秒脚本。让人物在办公室画面内正常完成一段日常活动，例如：

- 坐着看材料；
- 起身；
- 短距离行走；
- 取放普通物品；
- 整理；
- 停留；
- 返回座位。

建议每个文件连续 10–30 分钟，便于导出和标注。不要把几个小时全部放在一个文件里。

静坐和原地停留本身不是 `direct/pacing/lapping/random` 四类行走形态，可以在标注表中记为未评分区间或 `excluded`，不能硬塞成 random。

## 13. 质量挑战样片

正常清晰样片合格后，再单独拍：

- 短时局部遮挡；
- 人物进入画面、离开画面再返回；
- 较暗但仍能看清人体的光线；
- 与背景对比不同的服装；
- 后续需要时再拍多人短暂交叉。

质量挑战片必须单独编号。它们用来观察 tracking 和标注问题，不能替代清晰基准样片。

## 14. 每条视频怎样检查

### 14.1 可以继续使用

- 文件完整并能正常播放；
- 第一条样片实际 FPS 不低于 15；
- 摄像头没有转动或变焦；
- 主要行为中人物全身和双脚可见；
- 主要轨迹连续且长度足够；
- 光线稳定；
- 实际行为能够判断；
- 行为开始和结束时间能够大致确定。

### 14.2 可以保留，但要记录问题

- 短时局部遮挡；
- 短时出画后返回；
- 脚部偶尔不可见；
- 光线偏暗但仍可判断；
- 行为从一种形态变成另一种；
- 后续 tracking 出现碎片或疑似 ID switch。

### 14.3 建议重拍

- 摄像头自动跟随或明显转动；
- 人物长期切头、切脚或大部分时间出画；
- 主要路线长期被遮挡；
- 文件损坏、卡顿或无法正常解码；
- 计划拍摄的主要行为完全没有形成；
- 整条视频无法确定主要行为时间段。

重拍使用新的文件编号，不覆盖原文件。

## 15. 视频命名

推荐使用匿名、简单、可排序的文件名：

```text
VID-B01-0001_frame-check.mp4
VID-B01-0002_tracking-smoke.mp4
VID-B01-0003_episode-direct-R01.mp4
VID-B01-0004_window-pacing-R01.mp4
VID-B01-0005_window-lapping-R01.mp4
VID-B01-0006_window-random-R01.mp4
VID-B01-0007_window-phone-call-R01.mp4
VID-B01-0008_natural-office-R01.mp4
```

命名规则：

- `B01` 表示拍摄批次；
- 中间数字为唯一视频编号；
- 文件名中写 `episode`、`window` 或 `natural`，让接收者立即知道时长用途；
- 末尾写拍摄脚本和重复编号；
- 不在文件名中写真实姓名；
- 重拍使用 `R02/R03` 或新的唯一视频编号；
- 文件名、拍摄日志和标注表中的 `video_id` 必须一致。

## 16. 拍摄记录表

每拍完一条，填写一行：

| 镜头号 | video_id | 文件类型 | 日期 | 机位 | 光线 | 拍摄脚本 | 给演员的宽泛说明 | 实际时长 | 连续未剪接 | 人物完整可见 | 双脚可见 | 摄像头固定 | 初检结果 | 问题说明 |
|---|---|---|---|---|---|---|---|---:|---|---|---|---|---|---|
| W-S00 | VID-B01-0002 | MODEL-WINDOW | 待填 | SETUP-01 | bright | tracking_smoke | 自然走动 | 待填 | yes/no | yes/no | yes/no | yes/no | pass/redo | 待填 |

如果一条视频中包含多个行为，在“问题说明”中写明大致时间，例如：

```text
00:10-00:45 自然来回走
00:45-00:55 停留
00:55-01:20 不规则走动
```

## 17. 标注工作的核心原则

实际使用 CVAT 进行 episode 标注时，请直接按[Windows 本地部署 CVAT 徘徊识别标注员教程](Windows本地部署CVAT徘徊识别标注员教程.md)操作。本节只保留拍摄交接时需要知道的标签原则。

### 17.1 以实际画面为准

拍摄脚本只是提示，不是真值。

- 计划拍 pacing，但实际一直绕圈，应标 lapping；
- 计划拍 random，但实际反复来回，应标 pacing；
- 计划拍 lapping，但只绕过一次，不应硬标 lapping；
- 形态确实不清楚时，使用 `unknown/uncertain`。

### 17.2 轨迹形态和行为目的分开

必须分别回答两个问题：

1. 人物走出了什么形态？
2. 人物为什么这样走？

例如：

- 打电话时来回走：`pacing + purposeful`；
- 找东西时不规则走动：`random + purposeful`；
- 直接走去拿水：`direct + purposeful`；
- 无任务的自然来回走：`pacing + nonpurposeful`；
- 不知道人物目的：shape 仍按画面标，purpose 写 `unknown`。

不能因为“有目的”就把 pacing、lapping 或 random 改成 direct。

## 18. 人工标注表需要哪些字段

拍摄与初标人员填写下面这些核心字段即可：

| 字段 | 填写内容 |
|---|---|
| `video_id` | 对应视频文件的唯一编号 |
| `episode_id` | 当前视频内唯一的行为段编号 |
| `start_sec` | 行为段开始的相对秒数 |
| `end_sec_exclusive` | 行为段结束的相对秒数；该时刻不属于本段 |
| `observable_pattern` | direct / pacing / lapping / random / unknown |
| `purpose_context` | purposeful / nonpurposeful / unknown |
| `purpose_evidence` | scripted / participant_report / observed_context / unknown |
| `evaluation_role` | wandering_like_positive / purposeful_hard_negative / ordinary_negative / uncertain / excluded |
| `visibility_quality` | good / partial / poor |
| `tracking_issue` | none / occlusion / out_of_frame / fragmented_track / id_switch / not_checked |
| `annotation_status` | accepted / uncertain / excluded |
| `notes` | 中文说明、转场、遮挡或特殊情况 |

推荐直接复制下面的表格使用：

| video_id | episode_id | start_sec | end_sec_exclusive | observable_pattern | purpose_context | purpose_evidence | evaluation_role | visibility_quality | tracking_issue | annotation_status | notes |
|---|---|---:|---:|---|---|---|---|---|---|---|---|
| VID-B01-0004 | E001 | 待填 | 待填 | pacing | nonpurposeful | scripted | wandering_like_positive | good | none | accepted | 自然来回走 |

## 19. 标签怎样选择

### 19.1 `observable_pattern`

```text
direct   整体从一个位置走向另一个位置
pacing   在相似区域反复来回
lapping  重复闭合或近闭合绕行
random   多方向不规则移动，没有稳定往返轴或重复闭环
unknown  画面不足或混合形态无法判断
```

### 19.2 `purpose_context`

```text
purposeful     有明确活动目的
nonpurposeful  拍摄脚本或人物说明明确没有任务目标
unknown        没有足够证据判断目的
```

不要只因为“画面里没看到任务”就填写 nonpurposeful。目的证据不足时写 unknown，但清楚的轨迹形态仍然可以填写。

### 19.3 `purpose_evidence`

```text
scripted            拍摄脚本明确说明了目的
participant_report  参与者拍后说明了目的
observed_context    画面能直接看到活动目的
unknown             没有目的证据
```

`purpose_context` 和 `purpose_evidence` 要配套：目的未知时两者都写 `unknown`；目的明确时必须填写实际证据来源。

### 19.4 `evaluation_role`

```text
wandering_like_positive   形态为 pacing/lapping/random，且有依据确认无明确任务目的
purposeful_hard_negative  有明确目的，但形态像 pacing/lapping/random
ordinary_negative         形态清楚的 direct
uncertain                 形态或目的不足以确定评价角色，暂不作为清楚正负样本
excluded                  画面或行为不可用
```

`purposeful_hard_negative` 不能用于 direct。direct 即使有明确目的，也通常记为 `ordinary_negative`。

静坐、站立和原地停留不属于四种轨迹形态，不要把它们硬标成 direct 或 random；这类时段通常不评分，需要保留记录时可写 `excluded`。

### 19.5 `annotation_status`

```text
accepted   标签清楚，可以使用
uncertain  标签不能稳定确定
excluded   不适合参与正式统计
```

填写时按下面的简单对应关系：

- `evaluation_role=uncertain` 时，`annotation_status=uncertain`；
- `evaluation_role=excluded` 时，`annotation_status=excluded`；
- 其余三种可评分角色填写 `accepted`；
- `observable_pattern=unknown` 不能填写 `accepted`。

## 20. 怎样划分 episode 时间段

一条视频可以包含多个 episode。不要把整条视频强行只标一个标签。

### 开始时间

当目标行为已经清楚开始时记 `start_sec`。入场、站位和准备动作一般不算主要 episode。

### 结束时间

当当前形态停止、变成另一种形态、人物停下或画面无法判断时记 `end_sec_exclusive`。

### 切分示例

假设一条 90 秒视频：

```text
00–08 秒：进入画面并站定，不评分
08–48 秒：来回走，pacing
48–55 秒：停留，不评分
55–82 秒：不规则走动，random
82–90 秒：站定并离开，不评分
```

则可以标成：

| episode_id | start_sec | end_sec_exclusive | observable_pattern |
|---|---:|---:|---|
| E001 | 8.0 | 48.0 | pacing |
| E002 | 55.0 | 82.0 | random |

规则：

- 时间从当前视频的 0 秒开始；
- `start_sec` 不得小于 0，`end_sec_exclusive` 必须大于开始时间且不能超过视频结尾；
- 同一个人的 episode 不要互相重叠；
- 前一个 end 可以等于后一个 start；
- 形态切换不清时缩短清楚区间或标 uncertain；
- 静坐、站立、入场和离场不必硬贴四类形态；
- 遮挡期间无法判断时不要根据前后轨迹猜测。

## 21. 常见标签示例

| 实际画面 | shape | purpose | role |
|---|---|---|---|
| 自然从座位走向门口 | direct | purposeful | ordinary_negative |
| 无任务地持续来回 | pacing | nonpurposeful | wandering_like_positive |
| 打电话时持续来回 | pacing | purposeful | purposeful_hard_negative |
| 无任务地围绕中心区域重复绕行 | lapping | nonpurposeful | wandering_like_positive |
| 重复绕行，但不知道为什么这样走 | lapping | unknown | uncertain |
| 围绕区域反复清洁 | lapping | purposeful | purposeful_hard_negative |
| 找东西时多方向走动 | random | purposeful | purposeful_hard_negative |
| 实际路线混合且边界不清 | unknown | unknown 或有证据的 purpose | uncertain |
| 人物长期出画或被遮挡 | unknown | unknown | excluded |

如果 purpose 不清楚，不要因此把清楚的 pacing/lapping/random 改成 unknown。shape 和 purpose 分开填写。

## 22. 标注自检

完成每条视频后检查：

- [ ] `video_id` 与文件名一致；
- [ ] 每个 `episode_id` 在当前视频中唯一；
- [ ] start 小于 end；
- [ ] 时间没有超过视频时长；
- [ ] episode 之间没有重叠；
- [ ] shape 按实际画面填写，不是按计划填写；
- [ ] purpose 有证据，没有证据时写 unknown；
- [ ] purposeful pacing/lapping/random 没有被改成 direct；
- [ ] 遮挡、出画和不确定区间有记录；
- [ ] 没有根据模型预测修改标签。

## 23. 拍完后交给后续开发人员

交接内容：

```text
原始视频文件
拍摄记录表
人工 episode 标签表
机位与光线说明
质量问题和重拍说明
```

视频、拍摄记录和标签表必须使用同一个 `video_id` 对应。

后续 tracking 生成后，数据开发人员会把人工标签与 `track_id`、机位、session 等技术字段对应。拍摄与初标人员不要提前猜 `track_id`，也不需要自己填写复杂的机器 manifest。

## 24. 后续扩批目标

第一小批通过后，再逐步扩展：

- direct、pacing、lapping、random 各形成至少 10 个清楚 episode；
- 有目的困难负例逐步累计到至少 30 个 episode；
- 普通连续自然活动逐步累计到至少 5 个可用人物小时；这里只计算人物在画面内、标签可判断、tracking 可运行且属于非徘徊的时间，不等于视频文件总时长；
- 增加不同日期、自然速度、转身习惯和光线；
- 条件允许时再增加其他参与者；
- 条件允许时增加更长空间的 direct；
- 最后再增加遮挡、出画和多人交叉。

这些是逐步积累目标，不要求第一天完成。

## 25. 第一日清单

### 拍摄前

- [ ] C6C 自动跟随、巡航和变焦已关闭；
- [ ] 画面可以看到人物全身和双脚；
- [ ] 主要行走地面连续可见；
- [ ] 光线稳定、人物与背景有对比；
- [ ] A、B 或其他路线参考位置已确定；
- [ ] 已在记录表写明本条是 `EPISODE`、`MODEL-WINDOW` 还是 `NATURAL`；
- [ ] 文件命名规则和拍摄记录表已准备。

### 先拍样片

- [ ] 录制 20–30 秒 `frame_check`；
- [ ] 检查左、中、右站位；
- [ ] 录制 60–120 秒 `tracking_smoke`；
- [ ] `tracking_smoke` 是同一个原始文件连续录像，不是多条短片时长相加；
- [ ] 导出原始文件；
- [ ] 检查实际 FPS、时长和可解码性；
- [ ] 检查是否切头、切脚、出画、遮挡或云台移动。

### 样片通过后

- [ ] 已有几秒 direct 短片保留，并按实际 direct episode 标注；
- [ ] pacing、lapping、random 各先拍约 2 条自然完整 episode，实际走完即可结束；
- [ ] 每条 episode 中目标人物在主要行为段内可见、轨迹可判断；
- [ ] 样片通过后再逐步扩到每类约 10 条有效 episode，不机械重复同一路线；
- [ ] 只有为 EP2 或 legacy 诊断另拍连续片时，才按同一个原始文件连续录像，不把独立短片剪接；
- [ ] 没有让 direct 反复往返来凑 40 秒；
- [ ] 困难负例每种先拍 1 条；
- [ ] 拍一条 10–30 分钟自然活动；
- [ ] 最后拍暗光、轻遮挡和进出画面样片；
- [ ] 每条视频立即命名并填写记录。

### 拍完当天

- [ ] 核对所有文件能正常播放；
- [ ] 核对视频编号与记录表一致；
- [ ] 按 episode 填写开始和结束时间；
- [ ] 分别填写 shape 和 purpose；
- [ ] 记录画面质量和 tracking 问题候选；
- [ ] 完成标签自检；
- [ ] 将视频、记录表和标签表一起交接。

## 26. 技术说明

模型正式使用的形态标签固定为：

```text
direct
pacing
lapping
random
```

正式系统后续还需要 `source_video_id`、`track_id`、setup、session 等技术字段。这些字段在 tracking 和数据整理后合并，不要求拍摄者现场填写。

若本手册与现行模型标签定义或机器 validator 冲突，以[徘徊识别技术文档2](../plans/徘徊识别技术文档2.md)和现行代码契约为准。
