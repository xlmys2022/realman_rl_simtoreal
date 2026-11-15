# Evaluation 层重构与兼容性验证

日期：2026-11-15。分支：`refactor/src_rl_evaluation-reconstuct`。基线 commit：`f49faa54241ee778ab3f48958128b0d713322e33`（f49faa5）。开始时工作区干净；未切换分支，未执行 git add / commit / push / merge。

## 1. 重构前的问题与实施结果

原评估分散于 `rl/common/evaluation.py`、`rl/grasp_lift.py`、`imitation/bc.py` 和基线 CLI；通用 Episode 指标、训练日志、运动统计与网络薄适配混在 RL 包中。规则 / BC 数据采集借用 RL 命名空间，基础抓取评估又与 PPO 训练共享来源辅助，职责不易定位。

现已形成独立 `evaluation/`，执行环境回合、收集任务行为、汇总指标；算法创建 / 优化 / checkpoint、训练 callback 和 PPO / BC 参数保护适配保留在 RL / imitation。脚本负责入口组织；测试负责接口与回归。Evaluation 的运行不要求安装或导入 PyTorch / SB3，也不依赖具体算法。

阶段一完成执行器、汇总、规则策略、中立统计 / 来源的迁移；阶段二迁出 BC 和基线循环并新增 callable 接口 / 测试；阶段三更新架构、文件地图和 README。最终完整回归已完成。

Python 文件：99→104；核心包 43→47，脚本保持 32，测试 24→25。新增 17 项 Evaluation 测试，原 396 项全部保留。

## 2. 最新相关目录树

```text
src/realman_rl_simtoreal/
├── simulation/                       # 原实现未修改
│   └── grasp_controller.py            # 阶段控制器产生动作
├── envs/                             # 原 Gymnasium / 物理连接未修改
│   ├── factory.py
│   ├── reach_env.py
│   ├── sequence_reach_env.py
│   └── grasp_lift_env.py
├── tasks/                            # 原配置与奖励未修改
├── rl/
│   ├── common/
│   │   ├── observation.py             # 共享固定观测契约
│   │   ├── env_factory.py             # 原训练包装栈
│   │   ├── artifacts.py               # 原 JSON / 版本 / 资源 / 目录工具
│   │   └── logging.py                 # 训练 logger 消费评估结果
│   ├── ppo.py                         # 训练、checkpoint、只读薄适配
│   ├── sac.py
│   ├── ppo_grasp.py                   # 课程、更新、checkpoint、编码适配
│   ├── sac_monitoring.py              # 原训练监控
│   └── sac_diagnostics.py             # 原 Q/TD / Replay 诊断
├── evaluation/
│   ├── __init__.py                    # 轻量，不主动加载算法 / 环境
│   ├── reach.py                       # 独立 RL / callable / BC 协议
│   ├── grasp_lift.py                  # 策略 / 控制器共享回合执行
│   ├── metrics.py                     # 纯指标汇总
│   └── baselines.py                   # 规则与早期基线协议 / 性能剖析
├── imitation/
│   ├── bc.py                         # 网络 / 监督训练 / 示范 / 权重保护适配
│   └── bc_to_sac.py                   # 原历史初始化保留
├── utils/
│   ├── geometry.py
│   ├── demo_helpers.py
│   ├── metrics.py                     # MotionStats / cosine，训练评估共用
│   └── artifacts.py                   # 抓取来源 / 资源指纹
└── vendor/                            # 未修改
scripts/
├── train/                             # 原入口与训练调度
├── evaluate/
│   ├── evaluate_reach_baselines.py     # 已无评估循环
│   ├── evaluate_ppo_reach.py
│   ├── evaluate_sac_reach.py
│   ├── evaluate_bc_reach.py
│   ├── evaluate_ppo_grasp_lift.py
│   └── analyze_reach_results.py        # 单调用者离线分析保留
├── demos/run_grasp_lift.py             # 使用新执行器
└── diagnostics/                       # 公共导入同步

tests/
├── test_evaluation.py                  # 新增 17 项，无训练
└── test_*.py                          # 原测试与动态导入 / mock 路径同步

docs/
├── architecture.md                    # 边界、接口与依赖图
├── FILE_MAP.md                        # 全部 104 个 Python 文件
└── validation/
    ├── EVALUATION_REFACTOR_VALIDATION.md
    └── evaluation_refactor_checks.json
```

## 3. 文件迁移表

包内路径均相对 `src/realman_rl_simtoreal/`。

| 原路径 / 功能 | 新路径 / 功能 | 处理 |
|---|---|---|
| `rl/common/evaluation.py: evaluate` | `evaluation/reach.py: evaluate` | 原协议、独立环境和种子 / 终止 / JSON 保留；增加可选 raw callable |
| 同文件 `REASONS / summarize_episodes` | `evaluation/metrics.py` | 原原因集合与汇总公式不变 |
| 同文件 `record_evaluation` | `rl/common/logging.py` | 纯训练 logger 写入，标签不变，不放入任务执行器 |
| `rl/common/baselines.py` | `evaluation/baselines.py` | 原规则策略整体移动 |
| `rl/common/metrics.py` | `utils/metrics.py` | MotionStats 同时服务训练与评估，放中立模块 |
| `rl/grasp_lift.py: rollout / evaluate` | `evaluation/grasp_lift.py` | 回合执行循环原样迁移，无独立旧实现 |
| 同文件 `summarize` | `evaluation/metrics.py: summarize_grasp_episodes` | 原字段 / 成功回合分母保留，评估模块导入别名 summarize |
| 同文件 `experiment_metadata` | `utils/artifacts.py` | 训练、演示、评估共用；来源指纹反映新文件 |
| `imitation/bc.py: evaluate_policy` 的循环 | `evaluation/reach.py: evaluate_bc` | 保留 BC records / metrics 协议；BC 模块只保留权重保护薄适配 |
| 同文件 BC 闭环汇总 | `evaluation/metrics.py: summarize_bc_episodes` | 不替换方向阈值、终止键集合或分母 |
| 同文件 `cosine` | `utils/metrics.py` | BC 训练指标与评估共享一份原实现 |
| 基线脚本 `evaluate_policy / validate_transition` | `evaluation/baselines.py` | 原 warm-up、动作、验证与记录不变 |
| 同脚本 `reward_comparison / profile_components` | `evaluation/baselines.py` | 奖励反例与单独剖析流程不变 |
| 同脚本 `latency_summary / goal_distribution / summary` | `evaluation/metrics.py` | 原计算、字段与分母不变 |
| `scripts/evaluate/evaluate_reach_baselines.py` | 原路径 | 只保留 CLI / 报告保存，原函数名直接导入供现有测试调用 |
| `rl/ppo.py: evaluate_ppo` | 原路径 | 保留算法专属适配，无 env.step 循环 |
| `rl/ppo_grasp.py: evaluate_model` | 原路径 | 编码、Box 原动作审计与只读保护薄适配，无独立 Episode 循环 |
| 算法 / CLI / 测试旧导入 | 相应新模块 | 所有调用、隔离进程字符串与 mock 查找位置同步 |

旧 `rl/common/evaluation.py`、`rl/common/baselines.py`、`rl/common/metrics.py`、`rl/grasp_lift.py` 不再存在，没有转发包装文件。工作区使用文件重命名；沙箱把 `.git` 设为只读，因此没有为 git mv 的暂存区写入申请权限或触碰索引，Git 后续可按内容相似度识别历史。

## 4. 关键代码与依赖

Reach 的原 `evaluate()` 仍新建独立环境，按指定种子 reset；离散目标仍逐候选枚举，连续目标仍按原交接与奖励上下文记录。新 `evaluate_policy(predict_action, seeds, ...)` 调用同一循环，raw observation→合法动作的函数可来自规则策略、BC、PPO、SAC 或其他算法；调用者决定预处理与确定性推理，可提供 state_digest 检查参数 / 计数不变。

GraspLift 的原 rollout / evaluate 接收环境配置和种子；无 model 时调用原 `GraspLiftController`，有 model 时只调用确定性 predict。PPO 的 EncodedPolicy 与原动作审计由 RL 薄适配负责，Evaluation 不反向导入 PPO，没有循环依赖。

BC 模型、训练、监督验证 MSE、专家采集和保存加载仍在 imitation；闭环循环进入 Evaluation，权重保护以调用者的指纹函数实现。其示范仍只调用同一份公开观测 rule_action，不引入策略混合、专家兜底或参数更新。

训练 callback / Rollout / loss / checkpoint / SAC Q/TD / Replay 均保留；原 record_evaluation 只是移到 logging，所有 TensorBoard 标签与测量口径不变。MotionStats / cosine 与来源工具放中立 utils，公共训练设施不依赖任务专属执行器。离线 analyze_reach_results.py 只有单一入口，保留以避免多余抽象。

保留三类历史 Reach JSON 协议，而不是强行把它们的运动阈值 / 成功分母合并。新接口复用执行器；旧协议的指标公式分别集中于 metrics，不复制新公式。Evaluation 只通过不依赖学习框架的 `rl/common/observation.py` 共享观测契约，这是本次有意保留的中立耦合。

最新完整依赖图和调用示例见 [architecture](../architecture.md)。Evaluation 衡量策略行为；tests 判断代码是否满足契约，二者职责不同。

## 5. 实测测试结果

使用 `/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python`（Python 3.11.17）执行。

| 检查 | 通过 | 失败 | 跳过 | 秒 |
|---|---:|---:|---:|---:|
| 重构前完整基线 | 396 | 0 | 0 | 374.309 |
| 阶段一包与导入 | 10 | 0 | 0 | 13.555 |
| 阶段二 Evaluation | 17 | 0 | 0 | 2.198 |
| 阶段二 BC | 15 | 0 | 0 | 5.182 |
| 阶段二包与依赖 | 10 | 0 | 0 | 14.598 |
| 最终完整回归 | 413 | 0 | 0 | 378.864 |

实际命令（完整回归在修改前和全部代码修改后各运行一次）：

```bash
python -I -m unittest discover -s tests -v
python -I -m unittest discover -s tests -p test_package.py -v
python -I -m unittest discover -s tests -p test_evaluation.py -v
python -I -m unittest discover -s tests -p test_bc_reach.py -v
python -I -m compileall -q src scripts tests
python -I -m pip check
```

另有一个阶段一独立进程检查，阻断 torch / stable_baselines3 / PPO / SAC / BC 导入后，真实执行了 2 步 Reach 与抓取控制器评估。新增测试保留这个边界，并覆盖独立环境 / NumPy RNG、非法动作、BC / 未知算法参数保护、空统计、成功回合与计划 / 已尝试目标分母等。

实际在 `/tmp` 工作目录以绝对脚本路径执行 29 个 argparse CLI 的 `--help`，全部返回 0；三个无参数模型生成器不伪称支持 help。AST 核对 32 个脚本参数定义不变、55 个算法函数 / 类及 13 个搬移函数实现不变；6 个完整算法模块除导入外的语法树完全相同，包含超参数常量。pip check 报告 No broken requirements found。新包按原 pyproject wildcard 被发现，现有 editable 隔离导入正常，无需重装或更新依赖。

### 固定 seed 的历史模型与物理结果对照

直接使用已有本地 checkpoint，先后执行同一协议；没有训练新模型来代替历史文件。九种协议的动作、目标、物理状态、奖励、结束原因、原 JSON 字段和数值指标逐项完全相同。耗时的字段结构保留，耗时数值不作相等断言；来源路径 / 哈希按迁移后的文件记录，不伪称源码 provenance 不变。

| 协议 | seed | 步数 | 终止 | 结果 |
|---|---:|---:|---|---|
| 共享 Reach 规则 | 88100 | 14 | success | 前后完全一致 |
| 共享 Reach 随机（3 步短验证） | 88100 | 3 | time_limit | 前后完全一致 |
| 历史固定目标 PPO | 88000 | 18 | success | 前后完全一致 |
| 历史正式 SAC | 88100 | 200 | time_limit | 前后完全一致 |
| 历史 BC | 88100 | 14 | success | 前后完全一致 |
| BC records 规则协议（3 步） | 88100 | 3 | time_limit | 前后完全一致 |
| 早期基线协议（6 步） | 88100 | 6 | time_limit | 前后完全一致 |
| GraspLift 阶段控制器 | 230000 | 447 | success | 前后完全一致 |
| 历史 12 mm PPO GraspLift | 270000 | 244 | success | 前后完全一致 |

模型位置（均在现有正式分类目录中）：

- `outputs/reach/ppo/fixed_seed2026_pilot/final/model.zip`
- `outputs/reach/sac/formal_seed2026/final/model.zip`，同时读取原 30000 transition Replay。
- `outputs/imitation/bc/fit_seed2026/best_bc.pt`，仍使用 weights_only=True。
- `outputs/grasp/ppo/pick_lift_12mm_seed2027/final/model.zip`

BC 还以训练 / 验证种子 31000、31001 各采 3 步，21 个示范数组逐元素相同。完整套件继续验证历史 BC→SAC 初始化、真正的短 PPO/SAC 更新和 checkpoint 保存 / 加载；这些是已有必要短测试，未执行正式长训练。

固定协议命令为：

```bash
python -I /tmp/realman-evaluation-refactor-20261115/check_episodes.py before
python -I /tmp/realman-evaluation-refactor-20261115/check_episodes.py after
```

原始测试日志、前后 JSON、示范 NPZ 与源码快照位于 `/tmp/realman-evaluation-refactor-20261115/`；持久化摘要、版本、各协议结果与检查计数见 [evaluation_refactor_checks.json](evaluation_refactor_checks.json)。

### 数据与核心逻辑保护

逐一复核 4116 个文件、2477165716 字节的大小和 SHA-256：其中 outputs 4056 个文件，包含 731 个模型。outputs、全部资产、simulation、envs、tasks、vendor 和原 fixtures 的文件集合与内容均完全相同；未覆盖、删除或新增实验产物。

关键版本：mujoco 3.17.0；numpy 2.4.6；gymnasium 1.4.0；torch 2.14.0+cpu；stable_baselines3 2.9.2；tensorboard 2.21.0；matplotlib 3.11.2；setuptools 84.0.0；Robotic_Arm 1.1.7。

## 6. 兼容性

- 网络结构、训练更新、超参数、环境 / 动作 / 观测、奖励、终止、评估目标种子和预处理未改变。
- PPO / SAC 官方模型和 Replay 格式、BC state_dict 保存 / 加载不变；保留相关算法类的模块路径，不需要不安全的别名反序列化。
- 全部 CLI 参数、默认值、输出保存与退出码语义保持；脚本 helper 的直接导入在原测试中继续有效。
- JSON 顶层 / Episode / 原指标保持；新源码的来源字典自然列出 Evaluation、utils 的真实位置，运行耗时每次重新测量。
- 旧 Python 业务导入路径有意迁移；仓库所有调用已同步，外部调用者按迁移表更新。没有为旧模块保留无意义包装。

## 7. 限制与后续建议

- 当前 Reach 有现代 RL、BC records、早期基线三个历史输出协议；它们有不同的真实采样 / 指标定义，暂不强行归一化。
- GraspLift 的 model / policy='ppo'、controller_phase='ppo' 保留历史命名；其他算法能提供同一 predict 协议，未来若需要新的算法标签，应显式扩展格式。
- 观测处理仍共享 rl/common/observation.py 的中立代码；Evaluation 对该工具的依赖不包含学习框架，已由阻断导入测试验证。
- 本次对四个已有模型做固定 seed 回合对照，并核验所有历史模型文件哈希；未重新逐一评估全部 731 个模型，也未尝试修复更早 schema / 缺配置文件的已有问题。
- 未做正式长训练、真实桌面 Viewer、真机 / 视觉验证或策略效果提升研究。
- 可在实际新增任务后扩展对应 Evaluation 模块和显式策略标签；当前不预建抽象基类、插件注册器或额外依赖。

[架构与接口](../architecture.md) · [完整文件地图](../FILE_MAP.md) · [项目 README](../../README.md)
