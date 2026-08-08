# 徘徊步骤 9：TopoWander-MPT 纯前向契约复现记录

当前状态：`step9a_exact_config_fail_closed_verified`、`step9_forward_contract_verified`

日期：2026-08-07；第 9a.5 节修正、第 9a.6 节再次终审与第三轮重验：2026-08-08；LF/Git 检查点：2026-08-08

唯一 Git 检查点 commit：`STEP9_CHECKPOINT_COMMIT`  
最终 `model.py` SHA-256：`11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331`

## 完成后独立复核（2026-08-07）

模型主体、参数量、前向、梯度和 deterministic NPZ 证据继续成立，但完成后复核发现步骤 9 初版 exact-config 门禁没有完全失败关闭：`yaml.safe_load()` 加 Python 字典相等会接受 `true→1`、整数→等值浮点数，以及重复 YAML key 的最后一个值。当时已在项目环境独立复现三种情况均被初版 loader/构造器接受。

这违反技术方案原有 exact-fields/value/type 完成门禁，因此当时撤回无条件 `step9_forward_contract_verified`，改为 `step9_contract_correction_required`。随后只按窄范围步骤 9a 补 bool/int/float 与顶层/嵌套重复键红测，再实现拒绝重复键的安全 YAML loader 和递归类型严格比较；没有修改 config 原始字节、模型结构、参数、state、forward、A/V/H/C，也没有读取数据或训练。

继续复核确认，先前补写的裸 forward SHA `572c02e20a0b263ecabe1c466e13743ba0bf2c5b36b47f2a5e8a91f10d30c6ef` 在 HEAD、源码和测试中没有生成器，文档也未定义 fixture/mode/order/serialization；多组常见口径均无法命中。该值正式标记为 `withdrawn_provenance_missing`，不是 forward 漂移证据，步骤 9a 不得把它写成 expected。

项目负责人授权在步骤 9a 中先增加测试私有的规范 `forward_signature_v1`，完整公式和参考实现以[技术方案第 9a.2–9a.3 节](../../../docs/modules/mental_health/plans/徘徊样行为识别技术方案.md)为唯一事实源。当前项目环境中，修改实现前的两个独立进程和修复后窄测均得到 fixture deterministic NPZ `6192 bytes / 584d22203c5df5860e19eca9f8d758d7d33eb40a46096eb2025c7ee58b457c72`，forward deterministic NPZ `1022 bytes / 01415e02b01c71ad1d96fc2d5bc7b1fd8a0e400c45cd480ea8a62092293ef61c`。其余回归锚点仍为 config SHA-256 `debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7`、204,466 参数、77 个 state key、state NPZ 838,934 bytes/SHA-256 `6bd7094845c749a2b73502e9480ec9e6fb3df24d313cb4674b41f3903e96cd40`。

## 步骤 9a 首轮验收记录（2026-08-07，后被终审推翻）

严格执行顺序如下：

1. `model.py` 保持原始 SHA-256 `f3569d587c9a9c93b5402eec2e051d5a0aa942a5524ef5bd60b07fb39feb922a` 不变，先加入规范签名 green baseline；两个独立进程均为 `1 passed`，两项 payload 长度和 SHA 精确匹配；
2. 再加入红测：旧实现对四个顶层/嵌套、同值/异值重复键场景产生 `4 failed`；六类 loader/直接构造器类型等值漂移产生 `12 failed`；安全加载顺序红测证明旧实现会在拒绝错误 config 前读取 state；
3. 红测证据成立后，才在 `model.py` 加入基于 `yaml.SafeLoader` 的重复键拒绝和递归 type-strict 比较；没有修改 forward、encode、Parameter、state key/shape 或 NPZ 逻辑；
4. 首轮修复候选 `model.py` 原始字节 SHA-256 为 `b829293ff7d661e000a45221bcfef6047ea335a2450dabeb8ee01a8499f4d13a`。该 SHA 只记录当前未提交且后来被终审判定仍需修正的源码，禁止绑定到步骤 10。

新增门禁确认重复 YAML key 一律拒绝；`bool/int/float` 与 list 元素的 Python 等值类型漂移在 loader 和直接构造器两条路径均拒绝；稳定文件上的错误 config 在模型建参、state 读取和 `load_state_dict` 前失败关闭。这些结果继续有效，但不足以覆盖下面的对抗性反例，因此两个 verified 标志已再次撤回。

## 步骤 9a 对抗性终审（2026-08-07，修正前事实）

终审独立复现两个阻塞缺陷和一个测试缺口：

1. `safe_load_topowander_model()` 先 `read_bytes()` 校验外部 config SHA，随后 `load_topowander_config(path)` 又 `read_text()`。模拟第一次返回被 SHA 正确绑定但类型漂移的非法字节、第二次返回合法配置时，当前实现会成功加载；哈希与实际解析内容不是同一快照。
2. 构造器保存调用方原 config 引用，`model.config is caller_config`。构造后把 `relation.center_distance_scale` 从 `0.10` 改成 `0.20`，同一 fixture 的 binary/subtype/projection 输出最大绝对变化分别约 `0.011673/0.006150/0.019163`，但参数/state/NPZ 锚点不变。
3. 生产 `forward()` 当前顺序正确，但规范 NPZ 按 ASCII 排序且测试只比较 key set；任意重排输出 mapping 都不会改变 `01415e…f61c`，因此还需独立锁定 `binary_logit → subtype_logits → projection_embedding`。

这仍是原步骤 9a exact-config/forward 契约范围，不另立 9b。下一次实现必须先写 TOCTOU 与 config mutation 两个应失败红测，同时加入当前应通过的 forward 顺序契约断言；随后让 config 原始字节只读一次并由同一 payload 完成 SHA/严格 UTF-8/YAML/type-strict 解析，模型保存递归不可变且无调用方别名的私有 config 快照，只读视图同时拒绝嵌套原地修改和 `model.config = ...` 重绑定。拒绝必须早于构模、state 读取和赋值，稳定合法路径仍须成功。完整规则以[技术方案第 9a.5 节](../../../docs/modules/mental_health/plans/徘徊样行为识别技术方案.md)为准。

修正后必须保持 config SHA、204,466 参数、77 state key、838,934-byte state NPZ/`6bd709…d40`、fixture `6192/584d2220…457c72`、output `1022/01415e02…93ef61c` 和 A/V/H/C 不变，重新记录最终 `model.py` SHA，并复跑窄测、全部徘徊和完整测试。满足前不得恢复 `step9a_exact_config_fail_closed_verified`/`step9_forward_contract_verified`，也不得形成步骤 9+9a 最终检查点或开始步骤 10。

## 第 9a.5 节修正与重验（2026-08-08，后被 9a.6 推翻完成状态）

严格按负责人指定的 green→red→implementation 顺序执行：

1. `model.py` 保持失效候选 SHA-256 `b829293ff7d661e000a45221bcfef6047ea335a2450dabeb8ee01a8499f4d13a` 不变，只在 `test_forward_signature_v1_baseline()` 加入生产 mapping、config descriptor 和固定三字段 tuple 的精确顺序断言；该项先得到 `1 passed`，同时继续确认 fixture `6192/584d2220…457c72` 与 output `1022/01415e02…93ef61c`；
2. 仍不修改 `model.py`，再加入两个红测。TOCTOU 红测以同一路径第一次返回含 `parameter_count: 204466.0` 的非法哈希字节、假想第二次返回合法文件，旧实现因未抛 `TopoWanderStateError` 而失败；config 红测独立复现调用方 mapping 修改导致 binary 输出最大绝对变化 `0.011672735…`，并证明 mapping/list 别名、公开嵌套原地修改和 `model.config` 重绑定都会被旧实现接受。目标选择输出为 `7 failed, 1 passed`，其中六项为 config 不可变性 subtest 失败；
3. 红测证据成立后才修改 `model.py`：新增私有 bytes parser；公共 path loader 只 `read_bytes()` 一次后委托，安全 state loader 对同一 bytes 做外部 SHA、严格 UTF-8、unique-key SafeLoader 与递归 type-strict 解析，不重新打开 config path；
4. 构造器在 exact 校验后递归复制 mapping/list 为私有 `MappingProxyType`/tuple 快照，所有运行时读取均经该快照；无 setter 的 `config` property 只暴露递归只读视图。两个红测随后转为 `2 passed, 6 subtests passed`。

第二轮未提交候选 `model.py` 原始字节 SHA-256：

```text
a5ea680263590c1fcbcc85719a72acaadfe779277b749977187ef5c802c554ab
```

`b829…d13a` 继续只作为已失效首轮候选保留。config SHA、204,466 参数、77 state key、state NPZ 字节/SHA、fixture/output 字节/SHA、原 `forward()`/`encode()` 语义和 A/V/H/C 均由窄测或现场原始字节复核保持不变。未读取任何项目数据、未训练、未生成模型制品、未创建步骤 10 文件。下面的再次终审证明 `a5ea…c554ab` 也不得绑定步骤 10。

## 第 9a.6 节再次终审（2026-08-08，修正前事实）

文件级同字节 loader、普通 dict/list 的构造后无别名只读快照、属性重绑定拒绝和 forward 输出顺序均已真正关闭；但直接构造器仍有一个内存级快照缺口：

1. `TopoWanderMPT.__init__()` 先遍历调用方 `Mapping` 执行 `_exact_config_matches()`，通过后再由 `_freeze_config()` 第二次遍历同一对象；两次读取不构成同一快照。
2. 只读复现使用符合公开 `Mapping[str, Any]` 签名的状态化嵌套 Mapping：第一次为 `center_distance_scale` 返回 exact `0.10`，第二次返回 `0.20`。当前构造器成功创建模型，`model.config` 实际冻结 `0.20`。
3. 相对 exact 模型，同一 fixture 的 binary/subtype/projection 最大绝对变化约为 `0.0116723/0.0061502/0.0191635`；参数量仍为 204,466、state key 仍为 77，因此已有数值锚点不能发现该语义漂移。
4. 现有 25 项窄测只覆盖普通 YAML dict 的构造后 mutation，没有覆盖“校验读取与快照读取呈现不同值”的 Mapping；公共 `load_topowander_config(path)` 当前代码虽只读一次，也缺少独立 read-once 回归。

下一次实现仍属于步骤 9a，不新增 9b。先写只表达目标行为的顶层/嵌套状态化 Mapping 红测，不把旧错误 `model.config=0.20` 固化为 expected：自定义/惰性 Mapping 必须抛 `TopoWanderContractError`，且 spy 证明 `_build_modules()` 未调用。实现只接受可证明稳定的 plain `dict/list` 与冻结标量，对它们做一次捕获，再校验并冻结同一 snapshot；同时把 `TopoWanderMPT.__init__` 和 `create_topowander_model` 的 config 注解收窄为 `dict[str, Any]`，非 exact 值必须在参数创建和 forward 前拒绝。随后增加公共 path loader 绿色 read-once 回归，保持第 9a.5 节全部测试与所有冻结锚点。

当时 `a5ea680263590c1fcbcc85719a72acaadfe779277b749977187ef5c802c554ab` 是第二轮失效候选。修正后必须记录新的源码 SHA、复跑三层测试并由负责人决定唯一检查点；满足前不得恢复两个 verified 标志、不得 stage/commit/push，也不得进入步骤 10。

## 第 9a.6 节第三轮修正与重验（2026-08-08）

严格执行测试先行：

1. 保持 `model.py=a5ea…c554ab` 不变，先加入公共 `load_topowander_config(path)` read-once 绿色回归；spy 确认合法 config path 只以 `rb` 打开一次，结果为 `1 passed`。
2. 仍不修改 `model.py`，加入只表达目标行为的状态化 Mapping 红测：顶层和嵌套自定义 Mapping 都必须在 `_build_modules()` 与参数初始化前抛 `TopoWanderContractError`，并要求构造器/工厂注解为 `dict[str, Any]`。旧候选对两种 Mapping 均成功构造、注解仍为 `Mapping[str, Any]`，结果为 `3 failed`；测试没有把错误的 `model.config=0.20` 固化为 expected。
3. 红测成立后才新增 `_capture_plain_config()`：只接受 exact built-in `dict/list`、built-in `str/bool/int/float` 与 `None`，每个调用方语义值只捕获一次，拒绝自定义 Mapping、容器子类、非字符串 key 与其他可变/未知值。构造器随后只对这个无调用方别名的私有 plain snapshot 执行 exact 校验，再从同一 snapshot 生成递归不可变运行时视图；`TopoWanderMPT.__init__` 与 `create_topowander_model` 的公共 config 注解同步收窄为 `dict[str, Any]`。
4. 定向回归转为 `2 passed, 2 subtests passed`，顶层/嵌套状态化 Mapping 均在 `_build_modules()` 前拒绝；第 9a.5 节全部原测试继续通过。

第三轮最终检查点 `model.py` 原始字节 SHA-256：

```text
11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331
```

现场重验为：窄测 `27 passed, 1 warning, 56 subtests passed`；全部徘徊测试 `217 passed, 1 warning, 160 subtests passed`；完整测试 `697 passed, 2 failed, 1 warning, 239 subtests passed`。两项完整测试失败仍分别为跌倒 v2 工作区 CRLF 原始字节 SHA 与固定 LE2I 视频缺失。config SHA、204,466 参数、77 state key、state NPZ、fixture/output 双 SHA、forward/encode 语义和 A/V/H/C 均未漂移；没有读取项目数据、训练、生成模型制品或进入步骤 10。

## 路径级 LF 与唯一 Git 检查点（2026-08-08，已授权执行）

项目负责人明确授权后，仅向 `.gitattributes` 增加下列三条路径级规则，未扩大全仓匹配、未执行 `git add --renormalize .`、未改写或重签 H：

```gitattributes
configs/modules/wandering_topowander_mpt_v1.yaml text eol=lf
configs/modules/wandering_topowander_pretrain_v1.yaml text eol=lf
reports/mental_health/wandering_step8/visual_review/v3/HUMAN_REVIEW.md text eol=lf
```

验收结果：

- `git check-attr text eol` 对三路径均为 `text=set` / `eol=lf`；
- stage 前后 raw SHA 不变：config=`debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7`，H=`d9390010ec5dbb873945467f30bd4a05715342f9b446510ab1de8309bbc2be3a`，`model.py`=`11fd7732f49d7392f0dab8edaa3023ebb1ba32355b24fc2157349b7fff80b331`；工作区 CRLF 计数均为 0；
- 已跟踪 H 的 `git ls-files --eol` 为 `i/lf,w/lf,attr/text eol=lf`；步骤 9 config 同为 `i/lf,w/lf,attr/text eol=lf`；
- 全新临时 worktree checkout 后上述 raw SHA 仍不漂移（见本检查点形成过程记录）。

因此恢复：

```text
step9a_exact_config_fail_closed_verified
step9_forward_contract_verified
```

唯一检查点 commit 为 `STEP9_CHECKPOINT_COMMIT`。最终 `model.py` SHA 现可绑定步骤 10 exact config；仍不得把失效候选 `a5ea…c554ab` / `b829…d13a` 写入。

## 本步骤实际完成的范围

本步骤只把冻结的 TopoWander-MPT 数学结构实现为可测试的纯前向模块：

- exact config：`configs/modules/wandering_topowander_mpt_v1.yaml`；
- 三输入：`model_features float32[B,80,14]`、`shape_normalized_points float32[B,80,2]`、`point_mask float32[B,80]`；
- mask-aware TCN、19 个 patch、11 维语义、七维对称 relation、37 档 signed offset、两层显式 relation-bias Transformer；
- binary/subtype/projection 三个 head，以及只验证层级概率数学的纯函数；
- 固定初始化、调用方 RNG 隔离、确定性 numeric NPZ 和外部 config SHA 信任门禁。

实现没有读取任何 train/validation/test/official 数据、步骤 8 pair、RF/TCN bundle 或模型制品；没有训练、优化器、loss、校准/OOD、episode、日级心理风险或运行时顶层导出。没有生成或发布正式 checkpoint，也没有报告性能或鲁棒性分数。

## 进入条件复核

开始代码前只读核对：

- 分支 `feat/wandering-data-pipeline` 与远端同步，步骤 8 已由独立提交 `a1451f2` 形成版本库检查点；
- A=`0e255e0f89493c1ad389ac6d183c82843cbe0efc2978d5be7fbae66ce8dc4b1a`；
- V=`31dcff92d3114d237b304953f467f20dd070e46091f08c4dde595f1f5faa6205`；
- H=`d9390010ec5dbb873945467f30bd4a05715342f9b446510ab1de8309bbc2be3a`；
- C=`e27aa0716bb1cd39f6c37ac98c31b3d201d08b7673f61aadaa2ba1355f58bdbf`。

四个现行文件的原始字节 SHA-256 均与技术方案一致，未修改步骤 8 A/V/H/C 或失效归档。

## exact config 与参数手算

config 原始字节 SHA-256：

```text
debd3adcf0c84d266ca594fd9ba2716a5a854e074e6e2092570b1ec656ed80f7
```

参数量按冻结结构逐项手算：

| 组成 | 参数量 |
|---|---:|
| 14→64 input projection | 960 |
| TCN block 1：LN64 + depthwise64 + pointwise64→64 | 4,544 |
| TCN block 2：LN64 + depthwise64 + pointwise64→96 + residual64→96 | 12,768 |
| TCN block 3：LN96 + depthwise96 + pointwise96→96 | 9,888 |
| 11→64→96 semantic projection（含 LN11） | 7,030 |
| learned CLS | 96 |
| relation MLP 7→32→4 + signed offset 37×4 | 536 |
| Transformer layer 1 | 74,784 |
| Transformer layer 2 | 74,784 |
| final LN96 | 192 |
| binary head 96→64→1 | 6,273 |
| subtype head 96→64→3 | 6,403 |
| projection head 96→64 | 6,208 |
| **合计** | **204,466** |

代码构造后的 `sum(parameter.numel())` 也严格为 `204466`；测试不会在不一致时回写配置或文档迁就实现。

## 测试驱动记录

首次只新增 `tests/test_wandering_model.py` 后运行窄测，按预期在收集阶段失败：

```text
ImportError: cannot import name 'model' from
elderly_monitoring.modules.mental_health.wandering
1 error during collection
```

随后按 `exact config loader → 输入/mask → TCN → patch helper → relation helper → Transformer → heads → NPZ` 顺序最小实现。测试全部使用手算或纯合成内存夹具，覆盖：

- exact config 缺、多、枚举、真实数值漂移、Python 等值类型漂移、任意层重复 YAML key、同字节单次快照与拒绝顺序；
- shape/dtype/device/finite/mask/质量/时间通道和有效点门禁；
- masked slot 输出不变性与零输入梯度；
- 19 patch、6/8 门槛、不跨缺口连接，以及直线/折返/闭环/回访的语义真值；
- 对称 relation、signed offset 方向、CLS 零 bias 和 invalid key/value attention 屏蔽；
- 单样本/批处理/批置换一致性、所有可训练参数有限梯度和调用方 RNG 不变；
- 输出 shape/finite、层级概率和为 1；
- 两次 NPZ 字节一致；错误 config SHA、缺/多 key、dtype/endian/shape、对象数组、NaN/Inf 和 ZIP metadata 均在任何 state 赋值前失败；
- 调用方 mapping/list 修改不改变模型 config 或 forward；公开 config 视图递归只读且属性不可重绑定；生产 forward 字段顺序独立于 ASCII-sorted 签名序列化被锁定；
- 源码没有项目数据 loader、训练/评估入口或不安全状态加载路径。

## 环境与命令

Windows PowerShell 未暴露原生 `conda`，实际通过 WSL `Ubuntu-22.04` 的 `/home/lenovo/miniconda3/bin/conda` 运行；editable import 已验证为当前仓库：

```text
file:///mnt/c/Users/lenovo/Desktop/%E5%BF%83%E7%90%86%E7%AE%97%E6%B3%95/src/elderly_monitoring/__init__.py
```

实际环境：Python 3.11.15、PyTorch 2.13.0+cu130、NumPy 2.4.6、PyYAML 6.0.3。验证固定 CPU、AMP=false、intra/inter-op thread=1、deterministic algorithms=true；本机 CUDA driver 不满足当前 PyTorch build，因此没有使用 CUDA。

复现命令：

```bash
conda run -n eldercare-ai python -m pytest tests/test_wandering_model.py -q
conda run -n eldercare-ai python -m pytest tests/test_wandering_*.py -q
conda run -n eldercare-ai python -m pytest -q
```

实际结果：

- 步骤 9/9a 窄测：`27 passed, 1 warning, 56 subtests passed`；该 warning 为 CUDA driver 探测，计算实际在 CPU；
- 全部徘徊测试：`217 passed, 1 warning, 160 subtests passed`；
- 完整测试：`697 passed, 2 failed, 1 warning, 239 subtests passed`。

完整测试的两项失败均在本步骤范围外，且本次未修改相关文件：

1. `test_repository_reviewed_decision_is_bound_to_current_v2_actions`：裁决记录期望 `058540ce76076b373bfa86c70d0b157fa10af5cbb521aa7c4c58a3ac7a1b3af6`；Git `HEAD` blob 的 SHA-256 与该值一致，但 Windows 当前工作文件原始字节 SHA-256 为 `a69058fcdcdf74dc83c29654ea4ed0500b7fa85cd334521c31b79dc8203d4970`，同时 Git diff 为空，属于当前 checkout 的原始字节/换行漂移；没有修改跌倒标签或裁决文件绕过。
2. `test_repository_acceptance_config_has_hashable_fixed_inputs`：缺少固定视频 `data/external/le2i_imvia/raw/FallDataset/Home_01/Videos/video (1).avi`；没有修改跌倒 runtime fingerprint 门禁绕过。

## 证据边界与下一门禁

步骤 9 主体结构、mask、relation、梯度、重复键/type-strict、文件级同字节解析、普通容器只读快照、输出顺序、状态化 Mapping 构模前拒绝和安全数值状态证据均通过分层回归及本轮技术审计。直接构造器已经把调用方 plain 容器的单次捕获、校验和运行时冻结绑定到同一私有快照；路径级 LF 与唯一 Git 检查点已形成，治理状态为两个 verified。现有证据仍不证明模型已经训练、有效、优于 RF/TCN、对合成污染鲁棒、可校准、可拒识、可部署、能识别目标摄像头或真实老人，不构成心理或临床结论。

步骤 10 训练执行仍不得开始，直至可复建 runtime fact source 闭合并把最终 Step 9 commit/`model.py` SHA 写入步骤 10 exact config。技术方案已冻结固定 v3 pair-aware 路线：1,257 个 clean train parent、1,491 条 pair、126 proxy、wrapper exact graph、CLI、早停、20-file DAG、config/artifact schema 和 canonical bytes，不建立 v4。步骤 9/9a 没有在线补 pair，也没有提前实现 consistency/denoise/reconstruction loss。
