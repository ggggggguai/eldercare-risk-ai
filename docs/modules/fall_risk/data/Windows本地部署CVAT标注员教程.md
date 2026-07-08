# Windows 本地部署 CVAT 跌倒风险视频标注教程

## 1. 简述

本教程用于在 Windows 电脑上本地部署 CVAT，并完成跌倒风险视频动作级标注。

本项目标注时必须遵守：

```text
一个视频 = 一个 CVAT task
一个动作片段 = 一个 rectangle track
标注员只标视频中能看到的动作事实
不直接判断医学风险等级
```

CVAT 里的矩形框主要用于承载“动作片段时间段 + 目标对象身份”，不是本项目最终的人体检测真值。

## 2. 标注原则

只记录视频中能看见的事实：

- 不做医学诊断。
- 不判断老人真实健康状态。
- 不把“走得慢”直接标成高风险。
- 看不清、遮挡严重、人物离开画面或目标人物无法确认时，用 `U01_unable_to_judge`，不要猜。
- `U01` 必须填写 `note`，说明无法判断的原因。
- 每个动作片段优先选择一个主要动作标签。

第一阶段主要标动作级标签：

```text
A01-A04：正常动作
B01-B06：轻到中度风险动作
C01-C05：高风险前兆动作
D01-D04：跌倒和跌倒后状态
U01：无法判断
```

## 3. Windows 环境准备

### 3.1 电脑要求

建议配置：

| 项目  | 建议                               |
| --- | -------------------------------- |
| 系统  | Windows 10 2004 及以上，或 Windows 11 |
| 内存  | 16GB 及以上                         |
| 磁盘  | 至少 50GB 可用空间                     |
| 浏览器 | Google Chrome 或 Microsoft Edge   |
| 网络  | 能访问 GitHub 和 Docker 镜像源          |

### 3.2 安装 WSL2

用管理员身份打开 PowerShell，执行：

```powershell
wsl --install -d Ubuntu
```

安装完成后重启电脑。

重启后打开 PowerShell，检查 WSL 状态：

```powershell
wsl -l -v
```

如果看到 Ubuntu，并且版本为 `2`，说明 WSL2 正常。

示例：

```text
  NAME      STATE           VERSION
* Ubuntu    Running         2
```

如果 Ubuntu 不是 version 2，执行：

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

### 3.3 安装 Docker Desktop

1. 下载并安装 Docker Desktop for Windows。
2. 安装时选择使用 WSL2 backend。
3. 打开 Docker Desktop。
4. 进入 `Settings > Resources > WSL Integration`。
5. 勾选 Ubuntu。
6. 点击 `Apply & Restart`。

打开 Ubuntu 终端，检查 Docker 是否可用：

```bash
docker version
docker compose version
```

如果能看到版本号，说明 Docker 正常。

### 3.4 安装 Git

打开 Ubuntu 终端，执行：

```bash
sudo apt update
sudo apt install -y git
```

检查 Git：

```bash
git --version
```

## 4. 本地部署 CVAT

以下命令都在 Ubuntu 终端中执行。

### 4.1 下载 CVAT

```bash
cd ~
git clone https://github.com/cvat-ai/cvat
cd cvat
```

### 4.2 启动 CVAT

```bash
docker compose up -d
```

第一次启动会下载多个 Docker 镜像，可能需要 10-30 分钟。

查看容器状态：

```bash
docker ps
```

如果能看到类似容器，说明服务已启动：

```text
cvat_server
cvat_ui
cvat_db
cvat_redis
```

### 4.3 创建管理员账号

在 Ubuntu 终端执行：

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

按提示输入：

```text
Username: admin_fall_risk
Email address: 可以留空
Password: 自行设置
Password again: 再输入一次
```

密码输入时终端不会显示，这是正常现象。

如果密码太简单，系统可能提示确认，输入：

```text
y
```

### 4.4 打开 CVAT

用 Chrome 或 Edge 打开：

```text
http://localhost:8080
```

使用刚才创建的管理员账号登录。

## 5. 日常启动和关闭

以后每次使用 CVAT，先打开 Docker Desktop，再打开 Ubuntu 终端。

启动：

```bash
cd ~/cvat
docker compose up -d
```

关闭：

```bash
cd ~/cvat
docker compose down
```

查看当前容器：

```bash
docker ps
```

查看 CVAT 服务日志：

```bash
cd ~/cvat
docker compose logs -f cvat_server
```

## 6. 准备本地视频文件

建议把待标注视频放在 Windows 的一个固定目录，例如：

```text
D:\fall_risk_annotation\Home_01\
```

示例文件：

```text
video (1).avi
video (2).avi
video (3).avi
...
video (30).avi
```

注意：

- 不要修改公开视频原始文件名。
- 不要覆盖原始视频。
- 如果 `.avi` 在 CVAT 中无法预览，反馈给我，我来生成 `.mp4` 标注副本。
- 不要自行删除或替换原始视频。

## 7. 创建 CVAT 项目

进入 CVAT 后：

1. 点击顶部 `Projects`。
2. 点击 `+`。
3. 选择 `Create new project`。
4. Project name 填：

```text
fall_risk_action_annotation_v1
```

5. Description 填：

```text
跌倒风险动作级标注。标注员只标视频可见动作事实，不标医学诊断和最终风险等级。
```

6. Labels 使用我提供的 CVAT label JSON 导入。

每个标签都应是 rectangle 类型，并带三个属性：

```text
quality
target_subject
note
```

## 8. 标签配置检查

项目中应该有 20 个动作标签：

```text
A01_normal_walk
A02_normal_turn
A03_normal_sit
A04_normal_stand
B01_slow_walk
B02_dragging_walk
B03_shuffling_walk
B04_swaying_walk
B05_unstable_turn
B06_slow_sit_to_stand
C01_failed_sit_to_stand
C02_wall_support_walk
C03_stumble_recovery
C04_rapid_support_contact
C05_rapid_body_drop_recovery
D01_forward_fall
D02_lateral_fall
D03_backward_fall
D04_long_static_after_fall
U01_unable_to_judge
```

每个标签都应该有以下属性。

### 8.1 `quality`

类型：单选。

可选值：

```text
clear
partial_occlusion
heavy_occlusion
low_light
off_screen
multi_person_uncertain
```

默认值：

```text
clear
```

### 8.2 `target_subject`

类型：文本。

公开数据集不知道身份时填：

```text
unknown
```

自采视频按项目约定填：

```text
p01
p02
```

### 8.3 `note`

类型：文本。

非 `U01` 可为空。`U01` 必须填写原因。

## 9. 批量创建 Task

### 9.1 基本原则

本项目要求：

```text
一个视频 = 一个 task
```

不要把多个视频塞进同一个 task。

正确做法是批量创建多个 task，每个 task 对应一个视频。

### 9.2 在 CVAT 页面批量创建

进入 `Tasks` 页面：

1. 点击右上角 `+`。
2. 进入 `Create a new task` 页面。
3. Name 填：

```text
fall_risk__le2i_imvia__home_02__{{file_name}}
```

4. Project 选择：

```text
fall_risk_action_annotation_v1
```

5. 选择 Project 后，Labels 会自动继承项目标签，不需要重新添加。
6. 在 `Select files > My computer` 中选择多个视频文件。

确认页面底部按钮显示类似：

```text
Submit 5 tasks
```

这表示将创建 5 个 task。

如果显示：

```text
Submit 1 task
```

需要停止提交，检查是否把多个视频放进了一个 task。

## 10. 每条视频的标注流程

每个视频必须按以下顺序：

1. 完整播放一遍，不标注，只理解动作过程。
2. 第二遍观察动作切换点。
3. 第三遍开始创建 action track。
4. 标完后从头回看一次，检查时间边界、漏标和标签混淆。
5. 保存。
6. 提交复核。

不要第一次播放时边看边标，容易漏掉前后动作关系。

## 11. Rectangle Track 标注方法

CVAT 中的每个动作片段用一个 rectangle track 表示。

操作步骤：

1. 跳到动作开始帧。
2. 选择对应动作标签，例如 `B05_unstable_turn`。
3. 使用 rectangle track 框住目标老人可见身体区域。
4. 沿时间轴播放或逐帧移动，保持 track 覆盖同一个目标老人。
5. 在动作结束前最后一帧，保留一个非 outside keyframe。
6. 在下一帧设置 outside，表示该动作片段结束。
7. 动作切换时，新建下一个 action track。

矩形框要求：

- 框住目标老人主要可见身体区域。
- 不要求像人体检测框一样逐帧精确，但不能框到其他人。
- 人物离开画面时不要硬延长 track。
- 多人场景只标目标老人。
- 无法确认目标时标 `U01_unable_to_judge` 并写备注。

## 12. Keyframe 与 outside 的正确用法

### 12.1 普通动作片段

假设 `B05_unstable_turn` 从第 95 帧开始，到第 122 帧结束，第 123 帧切换为跌倒动作。

正确结构：

```text
B05_unstable_turn:
  frame 95   outside=false
  frame 122  outside=false
  frame 123  outside=true
```

错误结构：

```text
B05_unstable_turn:
  frame 95   outside=false
  frame 123  outside=true
```

错误原因：导出转换时可能只得到 `95-95`，变成单帧片段。

### 12.2 跌倒后静止到视频结束

假设 `D04_long_static_after_fall` 从第 176 帧开始，到视频最后一帧 263 结束。

正确结构：

```text
D04_long_static_after_fall:
  frame 176  outside=false
  frame 263  outside=false
```

如果只标：

```text
D04_long_static_after_fall:
  frame 176  outside=false
```

导出转换时可能只得到 `176-176`，变成单帧片段。

### 12.3 如何新增非 outside keyframe

1. 在左侧 Objects 中选中对应 track。
2. 跳到动作结束前最后一帧。
3. 不要点 outside。
4. 轻微移动矩形框，或点击 CVAT 的 keyframe 按钮。
5. 确认时间轴上当前帧出现 keyframe 点。
6. 保存。

## 13. 动作边界规则

| 动作    | 开始                  | 结束                  |
| ----- | ------------------- | ------------------- |
| 行走    | 第一步明显开始移动           | 停止行走或切换到转身、坐下、跌倒等动作 |
| 转身    | 身体或脚步开始改变朝向         | 朝向稳定且不再继续转动         |
| 坐下    | 身体明显开始向下坐           | 坐稳，身体不再继续下降         |
| 起身    | 身体离开座位或开始上升         | 站稳，身体不再明显晃动         |
| 近跌倒   | 失衡、急停、快速下沉或突然扶物开始   | 恢复稳定，或转为真正跌倒        |
| 跌倒    | 身体开始失去支撑并倒向地面、床边或椅旁 | 身体接触并稳定在倒地或倒卧状态     |
| 跌倒后静止 | 跌倒动作结束后开始静止         | 明显起身、移动或视频结束        |

时间边界先精确到相邻几帧或约 `0.1s`。复核时允许小于 `0.5s` 的边界偏差由复核人员统一修正。

## 14. 标签选择规则

| 情况            | 标签                             |
| ------------- | ------------------------------ |
| 稳定连续走路        | `A01_normal_walk`              |
| 正常稳定转身        | `A02_normal_turn`              |
| 正常坐下          | `A03_normal_sit`               |
| 正常起身          | `A04_normal_stand`             |
| 明显慢但稳定        | `B01_slow_walk`                |
| 脚抬不起来、拖着走     | `B02_dragging_walk`            |
| 步幅很小、密集挪动     | `B03_shuffling_walk`           |
| 行走中左右晃但未差点摔倒  | `B04_swaying_walk`             |
| 转身明显晃动或停顿     | `B05_unstable_turn`            |
| 起身慢但最终一次成功    | `B06_slow_sit_to_stand`        |
| 多次尝试起身失败或未成功  | `C01_failed_sit_to_stand`      |
| 持续扶墙、扶桌、扶家具移动 | `C02_wall_support_walk`        |
| 踉跄、脚步错乱但最后站住  | `C03_stumble_recovery`         |
| 快失衡时突然扶物恢复    | `C04_rapid_support_contact`    |
| 身体突然快速下沉但恢复   | `C05_rapid_body_drop_recovery` |
| 向前明确失去支撑并倒下   | `D01_forward_fall`             |
| 侧向明确失去支撑并倒下   | `D02_lateral_fall`             |
| 向后明确失去支撑并倒下   | `D03_backward_fall`            |
| 跌倒后持续不动       | `D04_long_static_after_fall`   |
| 看不清或无法确认      | `U01_unable_to_judge`          |

## 15. 质量属性填写

| 画面情况             | `quality`                |
| ---------------- | ------------------------ |
| 人体主体清楚，动作边界明确    | `clear`                  |
| 身体局部遮挡，但主要动作仍可判断 | `partial_occlusion`      |
| 关键动作被挡住，通常无法判断   | `heavy_occlusion`        |
| 光线过暗，动作判断困难      | `low_light`              |
| 人物部分或完全离开画面      | `off_screen`             |
| 多人场景无法确认目标老人     | `multi_person_uncertain` |

`quality` 不替代动作标签。

画面质量差但仍能判断动作时，动作标签照常标；无法判断时使用 `U01_unable_to_judge`。

## 16. `note` 填写规则

非 `U01` 的 `note` 可以为空。

`U01` 必须填写 `note`，例如：

```text
人物被桌子遮挡，无法确认是否倒地
多人交叉，无法确认目标老人
光线不足，无法判断是否扶墙
人物离开画面，只看到动作后半段
```

对于容易混淆的片段，也建议写简短备注：

```text
疑似扶墙但手部不可见
动作边界不确定，等待复核
```

## 17. 保存和自检

标注过程中经常点击页面上的 Save，或使用快捷键：

```text
Ctrl + S
```

保存后检查：

- 每个动作片段都有标签。
- 标签名称来自标签字典。
- 动作切换处没有明显漏标。
- 每条 track 有开始帧和结束前的非 outside keyframe。
- 该结束的 track 已在下一帧设置 outside。
- `U01` 都有 note。
- `D04` 只覆盖跌倒后静止，不覆盖跌倒过程。
- 近跌倒和跌倒没有混淆。
- 没有把 `D01/D02/D03` 和 `D04` 合并成一段。

## 18. 复核要求

以下片段必须重点复核：

- `C01-C05`
- `D01-D04`
- `U01`
- 起止边界不确定的动作
- 多人或遮挡场景

复核人员检查：

- 是否漏标明显动作片段。
- 标签是否来自标签字典。
- 起止时间是否覆盖完整动作。
- `C03/C04/C05` 是否被误标成普通慢走。
- `D01/D02/D03` 是否确实失去支撑并倒下。
- `D04` 是否只覆盖跌倒后静止，不包含跌倒过程。
- `U01` 是否有明确原因。
- 多人场景是否目标一致。

## 19. 导出标注

复核通过后导出。

在 CVAT task 页面：

1. 点击 `Actions`。
2. 选择 `Export task dataset` 或 `Export annotations`。
3. 格式选择 CVAT 原生视频格式，例如：

```text
CVAT for video 1.1
```

4. 不需要导出图片帧，除非复核人员要求。

导出文件命名：

```text
cvat_export__{task_name}__{labeler_or_reviewed}__v{YYYYMMDD}.zip
```

示例：

```text
cvat_export__fall_risk__le2i_imvia__home_01__le2i_home_01_video_1__reviewed__v20260630.zip
```

## 20. 常见问题

### 20.1 打不开 `http://localhost:8080`

先检查 Docker Desktop 是否正在运行。

再打开 Ubuntu 终端执行：

```bash
cd ~/cvat
docker ps
```

如果没有 CVAT 容器，启动：

```bash
docker compose up -d
```

如果仍打不开，查看日志：

```bash
docker compose logs -f cvat_server
```

### 20.2 Docker Desktop 报 WSL 错误

在 PowerShell 检查：

```powershell
wsl -l -v
```

确认 Ubuntu 是 version 2。

同时检查 Docker Desktop：

```text
Settings > Resources > WSL Integration
```

确认 Ubuntu 已勾选。

### 20.3 创建管理员账号失败

先确认容器存在：

```bash
docker ps
```

进入服务容器：

```bash
docker exec -it cvat_server /bin/bash
```

在容器中执行：

```bash
python3 ~/manage.py createsuperuser
```

### 20.4 视频上传后不能预览

处理方式：

1. 不要删除原始 `.avi`。
2. 反馈给数据管理员。
3. 由数据管理员生成仅用于标注的 `.mp4` 副本。
4. 在 manifest 或批次表中记录原始路径和标注副本路径。

### 20.5 标注片段导出后变成单帧

常见原因：只有开始 keyframe，没有结束前的非 outside keyframe。

正确结构：

```text
开始帧: outside=false
结束前最后一帧: outside=false
下一帧: outside=true
```

如果动作持续到视频最后：

```text
开始帧: outside=false
最后一帧: outside=false
```

### 20.6 批量创建 task 后名字不规范

如果使用：

```text
fall_risk__le2i_imvia__home_01__{{file_name}}
```

生成的 task 名可能类似：

```text
fall_risk__le2i_imvia__home_01__video (2).avi
```

这可以用于 CVAT 批量创建，但导出转换前，工程人员需要把它映射为统一 `video_id`，例如：

```text
le2i_home_01_video_2
```

标注员不要手工乱改文件名。

## 21. 交付物

完成一批任务后，需要交付：

- CVAT task 已保存。
- 复核要求的片段已标记清楚。
- `U01` 均有 note。
- 导出的 CVAT ZIP 文件。
- 如有问题，提供视频名、帧号和简短说明。

示例问题说明：

```text
video (4).avi，frame 132-148，人物被桌子遮挡，无法确认是否扶物，已标 U01。
video (6).avi，frame 210 附近疑似跌倒开始边界不确定，等待复核。
```
