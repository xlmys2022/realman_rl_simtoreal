# 工程架构与调用约定

项目使用 src 和 editable 布局。包内应用提供可复用的训练、命令调度和可视化逻辑，`scripts/` 保留用户命令。实际 Python 文件见 [FILE_MAP](FILE_MAP.md)，完整目录与兼容性结果见 [src 重构报告](validation/SRC_LAYOUT_REFACTOR_VALIDATION.md)。2026-11-18 的目录与历史验证见 [渐进式结构重构](validation/STRUCTURE_REFACTOR_VALIDATION.md)。

## 目录与依赖

```text
scripts ──→ apps/cli ──→ apps/training、apps/visualization
        ├─→ apps/training ──→ rl、envs、tasks、evaluation
        └─→ checkpoint 加载与 evaluation

rl ──→ SB3 模型、策略、分布、训练回调与模型状态
evaluation ──→ envs、固定观测编码与通用统计
envs ──→ tasks、simulation、perception
simulation/runtime ──→ physics、motion、devices
simulation/devices/mujoco_rgbd_camera ──→ perception/RgbdFrame
```

- `simulation/physics` 放模型加载、关节步进、执行审计和抓取接触力。`motion` 放法兰和工具 FK/IK、传统末端演示、笛卡尔回溯及分阶段抓取。`devices` 放 MuJoCo 夹爪、相机几何与 RGB-D 读取。`runtime` 组合桌面机器人、观察位预设与启动过程。
- `envs/reach` 提供单目标和连续多目标环境。`envs/grasp` 提供局部夹取、关节桌面、笛卡尔真值和 RGB-D 环境，以及局部和关节桌面的训练观测/奖励包装。`envs/factory.py` 保存按契约创建 Reach 和笛卡尔环境的入口。
- `tasks/reach` 持有 Reach 分布、配置和奖励版本，`tasks/grasp` 持有抓取的三个任务配置和奖励。`ReachEnv` 内保留历史奖励计算，以维持 GoalEnv 重标记、批次输入和原终止语义。
- `rl` 持有 PPO 网络、分布、模型工厂、checkpoint 和完整 rollout 审计。`apps/training` 管理四类训练的任务配置、评估间隔及输出生命周期。局部与关节桌面向量环境共用 `rl.common.env_factory.grasp_training_vector`。
- `evaluation` 接收策略或动作 callable，由独立环境执行回合。无需加载 PyTorch 或 SB3。Reach 和 BC 共用 `rl.common.observation` 固定编码，后者依赖 Gymnasium/NumPy。
- `perception/rgbd.py` 通过 `RgbdFrame` 提供对齐图像、米制深度、内外参与时间戳。目标检测和跟踪消费此契约。
- `hardware` 预留设备适配接口，`deployment` 预留真机策略观测、推理和执行流程。当前两包只有初始化说明。厂家离线 FK/IK 继续位于 `vendor`。
- `utils/metrics.py` 提供运动统计，`utils/artifacts.py` 提供通用产物 I/O、SHA256 和源码归档。局部抓取的来源集合定义在 `evaluation.grasp_lift.experiment_metadata`。

`apps/cli/tabletop_cartesian.py` 统一四个笛卡尔命令的参数与调度。`apps/visualization` 负责 MuJoCo Viewer、关节/笛卡尔回放和视觉交互。

`TabletopPickEnv` 使用独立的 `tabletop_pick.xml`，复用局部抓取的逐物理步保持判据。完整 reset 在仿真时间零初始化六轴零位和 60 mm 开口。课程起点记录与抓取上方姿态的插值比例。七维动作给出六轴绝对角度（度）和归一化夹爪开口，经同步限速及碰撞预检后由执行器执行。私有 IK 用于目标特征、奖励和探索坐标。评估向 `evaluation.grasp_lift.rollout` 提供非空策略并记录原始网络动作。

`rl/ppo_tabletop.py` 的关节空间高斯探索使用完整协方差、可逆动作坐标和保存的刷新周期。v6 的关节动作正则与夹爪开口正则基于动作执行前的目标，以保留子目标交接时的计分口径。

资源和实验目录见 [输出目录说明](OUTPUTS.md)。源码摘要使用实际新路径与文件内容。训练配置、评估指标和输出目录协议沿用原定义。

## 评估实现与入口

- `evaluation/reach.py`：原独立 Reach 评估、指定 seed / 离散目标枚举、连续多目标记录；通用 `evaluate_policy()` 接收原始观测到动作的 callable；`evaluate_bc()` 保留 BC 的逐步 records 协议。
- `evaluation/grasp_lift.py`：原 GraspLift `rollout()` / `evaluate()`。`model=None` 使用阶段控制器；传入具有 `predict(obs, deterministic=True)` 的对象即使用其动作。
- `evaluation/tabletop_cartesian.py`：真值和 RGB-D 桌面共用的独立环境、Episode、full / 随机位置与结果汇总；仅接收 `predict(obs)`、动作形状、状态指纹与来源回调，不依赖学习框架。
- `evaluation/metrics.py`：原 Reach、GraspLift、BC、早期基线的汇总函数，保留各自字段与分母；不依赖环境、训练器或神经网络。
- `evaluation/baselines.py`：原 rule_action、早期随机 / 规则对照、奖励反例和独立 cProfile；BC 数据采集继续调用同一份规则策略。
- `rl/common/logging.py`：原 `record_evaluation()` 及完全相同的 TensorBoard 标签；只消费已有评估结果，不执行 Episode。

`scripts/evaluate/` 保留 CLI 参数、checkpoint 选择和加载、最终 JSON 与控制台输出。基线脚本的执行循环已迁入 Evaluation，脚本直接导入原函数名供测试调用。`analyze_reach_results.py` 是单个调用者的离线 JSON 分析，保留在脚本中。

历史 Reach RL、早期基线、BC 的 JSON 格式并不相同：BC 保存完整逐步 records 与方向分母，早期基线包含 warm-up / 力矩 / 延迟，现代 Reach 支持离散与连续目标。它们共处 Evaluation，但保留原协议，避免把运动方向阈值、结束原因键或成功率分母改成另一套定义。

### Reach 接口

已有 PPO 使用算法薄适配，checkpoint 仍由 RL 加载：

```python
from realman_rl_simtoreal.rl.ppo import load_ppo, evaluate_ppo

model, transform, saved = load_ppo(model_path)
result = evaluate_ppo(model, transform, saved['environment'], [88000])
```

新策略可直接用算法无关的接口。以 BC 为例，它的 predict 已经接收原始观测：

```python
from realman_rl_simtoreal.evaluation.reach import evaluate_policy
from realman_rl_simtoreal.imitation.bc import load_policy, weight_digest

policy, saved = load_policy(bc_path)
result = evaluate_policy(policy.predict, [88100], name='bc',
    config=saved['environment'], transform=policy.transform,
    state_digest=lambda: weight_digest(policy.network))
```

`predict_action(raw_obs)` 返回原动作空间允许的数组。SB3 策略的调用者应编码观测并选择 `predict(..., deterministic=True)[0]`。可选 `state_digest` 在评估前后核对权重 / 计数；它由调用者提供，不要求 Evaluation 导入算法。若需要历史 BC JSON，继续用 `imitation.bc.evaluate_policy` 的薄适配，或传指纹回调给 `evaluation.reach.evaluate_bc`。

### GraspLift 接口

```python
from realman_rl_simtoreal.evaluation.grasp_lift import evaluate
from realman_rl_simtoreal.tasks.grasp.lift import GRASP_LIFT_CONFIG

controller_result = evaluate(GRASP_LIFT_CONFIG, [230000])
```

PPO 用 `rl.ppo_grasp.load_checkpoint()` 加载，再由 `evaluate_model()` 编码原 41 维观测并核验参数 / 计数。共享 GraspLift 执行器只读取传入策略的确定性动作，不导入 PPO。`model` 参数保留历史名称；其对象只需提供 predict 协议，无需训练或优化方法。历史 JSON 的 `policy='ppo'` / `controller_phase='ppo'` 学习分支标签继续保留。

### 笛卡尔桌面接口

笛卡尔训练调用 `apps.training.tabletop_cartesian.train()`。`rl.ppo_cartesian` 提供 `load()` 和 `evaluate()`，单回合执行位于 `evaluation.tabletop_cartesian.rollout()`。真值与 RGB-D 环境由 `envs.factory` 按保存的观测契约创建。GraspLift 和关节桌面继续使用原 GraspLift 评估器。

其他算法或规则可传 `predict(observation)`，返回未裁剪动作；Evaluation 沿用 `[-1,1]` 裁剪与 float32、原 seed、终止规则和统计定义。`evaluate(..., action_shape=(4,), state_fingerprint=..., sources=..., algorithm=...)` 接收完整任务 / startup / contact 配置；可选回调保护调用者自己的只读状态。未提供指纹时通用接口的 `parameters_unchanged` 为 null，避免声称完成未执行的参数核验；旧 PPO 薄适配始终提供指纹，JSON 保持 true。

PPO 指纹覆盖全体参数、num_timesteps、_n_updates、Adam 每参数 step、动量与参数组；不执行训练或切换网络结构。own 环境在 finally 中关闭，外部环境由调用者关闭；full / pose / radius 覆盖不允许与外部环境混用。full 标准恢复后才应用显式 object_radius，避免把随机位置静默恢复为零。历史 `phase='ppo'` / tick 标签作为输出兼容约定保留，不要求预测函数来自 PPO。

## 训练公共工具

`rl/common/observation.py`、`env_factory.py`、`artifacts.py` 保留固定观测、包装、模型状态和来源接口。PPO 回调留在相应的 RL 模块。BC 监督训练与数据集指标留在 imitation，闭环评估委托 evaluation。SE(3)、演示序列和文本格式位于 utils。

## 视觉职责与组合边界

`VisionTabletopCartesianEnv` 继续复用基类的物理动作执行和独立物理审计，组合原 RGB-D Camera / Detector / Tracker。`_observation` 只从机器人本体和传感器读取允许数据；在同一模块内组合两个纯函数：`visual_grip_and_hold` 计算相机夹持置信度 / 连续保持，`encode_visual_observation` 编码原 29 维视觉通道；姿态模式沿用原额外 9 维。函数参数显式限定为相机测量与机器人状态，不传整个特权 info 或 simulator 对象。

物理执行、传感器跟踪、观测编码、视觉进度和最终物理验收由不同现有职责承担；不为单个实现新增 ObservationProvider / CompletionChecker 类或独立文件。视觉环境的 step / reset、vision_lost、visual_false_success、settling margin 和禁止特权成功提前结束的规则保持原实现。真值与视觉观测即便维数相同，也保留各自模型契约和特征类。相机几何、深度单位、坐标变换、RGB-D 检测和 Policy 类保持原有计算与序列化语义。

## 最新机器人运行时的最小接口

笛卡尔环境使用具体的 `simulation.runtime.tabletop.TabletopRuntime`。

- `reset_zero(object_xy)` / `prepare(object_xy, tick)`：明确复位与传统启动；prepare 仍沿用原物理轨迹或显式训练缓存。
- `get_joint_state()`：SI 单位的关节位置、速度和实际参考，返回独立数组副本。
- `get_gripper_state()`：实际开口与编码器推算的开口速度；`Gripper.get_reference_width()` 独立读取实际执行器参考，仅在原本需要它的路径调用，避免新增安全验证时机。
- `tcp_pose()` / `tcp_velocity()` / `tcp_angular_velocity()`：现有真实工具位姿与 Jacobian 运动量。
- `kinematics.solve_ik()` 与 `CartesianController.execute()`：复用原 ToolKinematics / DLS IK 和笛卡尔回溯，不另建 FK / IK。
- `reference_is_safe()` / `path_is_safe()` / `advance(target, width, observer)`：原安全检查与 50 × 1 ms 同步物理执行。
- `forces()` / `raw_forces()`：保持 inward grasp force 与原接触法向力的不同定义和读取顺序。

关节参考限幅、夹爪约束、每个 1 ms 的 recorder → grasp_forces → observer 顺序及任务 hold 计算沿用原实现。任务奖励与完成由环境计算，视觉检测由 perception 处理。Reach 继续使用法兰 TCP 和 25 × 2 ms 控制周期。局部抓取和关节桌面环境保留各自保护和动作契约，共用底层模型、运动学、关节执行与夹爪。

## 任务与奖励的归属

`tasks/grasp/lift.py` 和 `tabletop_joint.py` 保存各自的 REWARD、CURRICULUM 与原 potential/shaped_reward。`tasks/grasp/tabletop_cartesian.py` 定义 EASY/STAGES 和环境配置，`tabletop_cartesian_reward.py` 保存笛卡尔奖励。

`rl/ppo_grasp.py` 与 `rl/ppo_tabletop.py` 保留历史奖励、课程或编码导入。局部抓取和关节桌面的包装与编码实现位于各自环境，训练调度由应用持有。

## 观测、动作和奖励契约

Reach 默认 `observation_mode="legacy"` 的原始 Dict 保持 `observation(27)`、`achieved_goal(3)`、`desired_goal(3)`。27维依次为：q(6)、qvel(6)、实际末端(3)、目标减实际(3)、命令位置(3)、实际ctrl(6)。原始观测为float64，位置单位为米，角度为弧度。

schema 2 固定仿射变换只缩放 27 维 observation，沿用 home 中心、5°、30°/s、10mm 尺度并转为 float32；两个 goal 键仍为 float64 米制，偏移为 0、缩放为 1。网络使用 `achieved_goal → desired_goal → observation` 的 33 维 float32 拼接，保持 SB3 CombinedExtractor 与 BC 的输入顺序一致。不增加 clipping、在线归一化或奖励归一化。

PPO可显式选择 `observation_mode="fetch_like_v1"`：普通observation仅包含ee_site实际世界位置(3)和
`mj_jacSite`完整位置Jacobian乘qvel、再乘control_period的运动特征(3)，加两个3维goal共12维。
运动特征单位为米。schema=3固定变换将位置相对home、运动特征按0.01m缩放；两个goal仍为世界坐标米。
官方MultiInputPolicy/CombinedExtractor将完整Dict的12维传给Actor和Critic，无自定义筛选或额外相对位置拼接。
规则策略和 BC 使用 legacy 契约。运动统计从回合 info 读取命令位置。

包装顺序从内向外严格为：

```text
ReachEnv → ScaledReachObservation → Monitor → DummyVecEnv → VecCheckNan
```

包装器的 `compute_reward` 仍向原环境传入米制 goal。PPO 按官方实现将原始高斯动作裁剪到动作 Box 后交给环境。RolloutBuffer 与 log_prob 仍使用原始动作。所有奖励、失败、拒绝与 timeout 逻辑都留在原 `reach_env.py`。

奖励保留distance、safe_dense和reach_shaped_v1，并可显式选择reach_progress_v1。
新奖励使用真实previous_achieved_goal与当前achieved_goal相对传入desired_goal的距离差，
加持续距离成本、每步成本、成功奖金及真实安全/拒绝成本；不使用命令位移或历史成功标记。
`tasks/reach/reward.py` 集中默认参数和校验；GoalEnv接口支持标量/批量与目标重标记，
`reward_components()`为纯查询，info和现有训练/评估JSON记录六个signed分项及距离元数据，不进入观测。
新checkpoint在环境配置、顶层reward_parameters及SB3普通metadata中保存/核对完整参数；旧checkpoint继续加载。
PPO更新、目标分布和控制/安全逻辑不变。诊断先于新实现完成，旧公式无明确错误；
典型安全排序、往返折扣边界及339项验证见[奖励报告](validation/REACH_REWARD_OPTIMIZATION_VALIDATION.md)。

## PPO 训练数据流

1. `scripts/train/train_ppo_reach.py` 解析任务与观测模式，调用 `apps.training.reach.train`。训练应用用 `tasks.reach.config` 的配置构造独立训练环境，保存环境、对应schema=2/3变换、模式/空间/字段顺序/实际特征维度、依赖和资源指纹。
2. `rl.ppo.make_ppo` 创建官方 `PPO("MultiInputPolicy")`，保留独立 256×256/Tanh 策略和价值网络及原超参数。
3. SB3 从包装后的 Dict 预测动作；经原动作 Box 裁剪和 `env.step` 形成真实轨迹。PPOAudit 只读记录原始 / 实际动作、运动、安全与训练 Episode，不产生动作或修改奖励。
4. SB3 采集完整 512 步 Rollout，计算原优势和 log_prob，按 64 样本 minibatch 完成 10 个 epoch。timeout bootstrap 与优化器逻辑完全使用官方实现。
5. `learn` 返回后读取优化指标，在跨越配置阈值的完整更新点执行评估和 checkpoint；保留第 0 步、最终步、部分更新中断的原标记。
6. `evaluate_ppo` 使用独立环境和保存的预处理，确定性推理并核验参数 / 优化计数不变；`record_evaluation` 写原有 JSON / TensorBoard 指标，绘图脚本只读取这些已测量数据。

04-F2直接复用上述流程及`tasks.reach.config`现有`random_reach`，不增加物理环境或训练器。正式训练只需显式选择随机任务及20个评估目标。PPO评估结果额外保存每个Episode的`raw_actions`、`actual_positions_before_m`和`box_clipping`；原`actions`仍为真正输入环境的Box裁剪动作。记录只读，不改变分布、动作、奖励、网络、Rollout或更新顺序。

04-F3在`tasks.reach.config`增加`axis_pair_reach`和`axis_six_reach`两个显式候选配置。ReachEnv增加`target_mode="discrete"`及米制`target_offsets`；构造时检查全部候选的原IK、home±5°、姿态与路径碰撞条件，任何一个失败都报告并拒绝整个任务，不删除或缩短目标。reset随机选候选，正常step不变。`options={"goal_index": i}`仅供合法reset选目标；`goal_index`只在info中，不进入33维网络观测。候选数组及预检报告通过副本提供，不能用它们修改命令或实际状态。

目前±X5mm合法；六方向5mm中±Y的当前DLS候选解违反原home±5°范围，默认六方向配置会拒绝启动。显式3mm六方向只用于离线诊断，不是暗中改变训练阶段。`scripts/diagnostics/probe_ppo_goals.py`从合法reset获取同一物理状态的观测，检查冻结Actor对目标特征的响应，不执行step或修改权重。旧`probe_reach_env.py`继续只检查原固定/球壳任务。

公共evaluation对离散任务逐一枚举候选，强制Episode数等于不同目标数；不依赖随机seed碰巧覆盖目标。保存`per_goal`、每个Episode的运动统计及结束详情，PPO继续补充raw/applied和实际位置。训练/独立评估CLI对离散任务自动选择全部候选数，显式错误数量报错；旧模式默认1集不变。TensorBoard使用`evaluation/goals/plus_x|minus_x/...`等逐目标标签，原聚合指标保留；绘图从实际per_goal JSON增加成功、距离、首动作XYZ、随机训练成功率和终止分类，不用重复次数伪称泛化。04-F3 早期版本只支持 scratch；当前 Reach 入口已支持经观测 / 奖励契约校验的 --init-model 权重迁移，并重建优化器和 rollout；完整跨进程训练状态续训仍未实现。参见[04-F3验证](validation/PPO_GOAL_CONDITIONED_VALIDATION.md)。

`scripts/evaluate/analyze_reach_results.py`只读取评估JSON，按相对reset home的XYZ符号（六组可重叠）、八象限及3–4/4–5mm半径分组，保留空组、小样本和范围外目标计数。它还比较原始/实际动作方向、裁剪角度及+X偏置；缺少原始轨迹的历史文件明确标记不可计算，不反推数据或重新调用IK。该入口仅依赖NumPy/标准库，不加载SDK、模型或环境。原有公共指标名称保留，不创建语义相同的TensorBoard标签。准备检查、种子划分和手动命令见[随机Reach验证](validation/PPO_RANDOM_REACH_VALIDATION.md)。

历史 SAC 和 BC→SAC 实验的模型、回放数据和报告仍可查阅。现行训练入口支持 PPO 与独立 BC。

## Gymnasium 与 MuJoCo

```text
归一化世界 XYZ 增量动作
  → ReachEnv 独立命令参考
  → 原 DLS IK / 限位 / 位姿 / 51 点路径碰撞预检
  → 原关节命令变化率限制与插值
  → simulation.physics.joint_control.advance_control_cycle
  → ctrl + mj_step（正常周期 25 × 2ms = 50ms）
  → 实际状态观测、奖励、terminated / truncated
```

正常 step 不写实际 qpos；reset 使用官方 home keyframe。每个物理步继续审计原总驱动力、有限数值、速度、限位与碰撞。模型加载、重力补偿、FK/IK、夹爪原执行和接触时序未改变；仿真模块的现行入口对应迁移后的分层路径。`ee_site` 仍是 link_6 原点，不是已标定 TCP。

## 训练、评估、诊断与测试

训练应用按原流程更新网络并写入新 run。独立评估读取指定种子与保存配置并记录动作及结果。Reach 环境探针和奖励诊断继续使用原物理与规则基线。

`tests/test_package.py` 在外部目录发现现行 CLI 并运行 `--help`，模型和场景构建器由资源回归覆盖。全量测试使用 `MUJOCO_GL=egl python -I -m unittest discover -s tests -v`，含原小规模训练更新、checkpoint 读写、相机和算法无关评估。手动完整轨迹回归在 `tests/manual_structure_regression.py`。

## 保存兼容性与边界

PPO 使用官方 SB3 保存格式，配置以 JSON 保存。历史 SAC 归档继续可通过 SB3 读取模型数据。部分桌面 PPO 使用自定义 Policy、分布和特征类，序列化依赖 `rl/ppo_tabletop.py`、`rl/cartesian_policy.py`、`rl/cartesian_reach_policy.py` 与 `rl/vision_policy.py` 中的类路径。BC 保存显式网络标识、配置和 state_dict，继续使用 `torch.load(weights_only=True)`。回归验证使用仓库已有的本地模型。

新PPO配置schema=2保存观测元数据；历史PPO配置schema=1、变换schema=2仍按legacy加载。
加载核对保存模式、固定变换、模型空间与实际features_dim，33/12维模型不得混用或自动迁移权重。
314项双模式开发验收及用户完成的12维随机Reach两档预算结果（16384步、131072步）见[Fetch观测验证报告](validation/REACH_FETCH_OBSERVATION_VALIDATION.md)。

项目加载器继续检查相邻的 config.json、观测版本、动作契约与启动姿态。早期笛卡尔 checkpoint 的兼容 flags 继续用于既有权重迁移路径。SAC 专用训练、评估、诊断与绘图入口已退役。

`pyproject.toml` 原 `include = ["realman_rl_simtoreal*"]` 自动发现新包；安装元数据、依赖版本及 editable 导入在隔离进程验证。未重新安装、升级依赖，也未修改 PYTHONPATH。普通 wheel 的资源打包仍未实现。

模型生成器 `build_dynamics_model.py` 保留写入 MJCF 的旧 provenance 注释字符串，保证资源重生成逐字节相同；它不是可执行路径，当前调用入口是 `scripts/models/build_dynamics_model.py`。

`assets/**/SOURCE.md` 同样作为原版本的来源记录保留，包含历史生成路径；不改写资源目录以保持已有资产指纹。当前生成命令见 README，路径对应关系见重构报告。

## 后续扩展

夹取抬升任务已经由 `envs/grasp/grasp_lift_env.py`、`tasks/grasp/lift.py`、`simulation/motion/staged_grasp_controller.py` 和 `evaluation/grasp_lift.py` 实现，入口为 `scripts/demos/run_grasp_lift.py`。环境提供 41 维 Box 观测、XYZ 加绝对夹爪开口的 4 维动作，通过 `advance_synchronized_cycle` 让机械臂、夹爪和自由物体共享每步 50 次 1 ms 积分。阶段控制器只产生这个动作空间中的动作，使用双侧接触后的有界横向对中反馈；物体的运动由接触、摩擦和重力产生。

成功计时在 1 ms 物理步后读取实际接触力和物体运动，要求持续离台、双侧夹持、低速度及有限滑移。`ContactRecorder` 可选择仅保留最近采样或保存完整物理历史，读取力缓存后才刷新实时几何；数值失败返回明确标记的最近有效观测。抓取评估保存在独立的 `outputs/grasp/controller/`，Reach 仍复用 `rl/common` 的固定观测变换；任务行为评估统一由 Evaluation 组织。这是早期局部 GraspLift 的非视觉接口。当前桌面 RGB-D 观测已实现于独立视觉环境并有本轮逐步回归；真机适配仍未实现。

早期 Evaluation 迁移、实际测试和兼容性证据见 [Evaluation 重构报告](validation/EVALUATION_REFACTOR_VALIDATION.md)。评估结果的来源路径 / 哈希随源码迁移反映新文件，物理量、指标与 JSON 协议保持。

旧版细节完整保存在 [历史架构快照](experiments/ARCHITECTURE_BEFORE_REFACTOR.md)；历史验证报告中的旧路径对应其原版本。现行命令以 README 和 FILE_MAP 为准。
## 桌面末端控制任务

`tabletop_cartesian` 是现有工程中的任务功能。固定观测预设位于 `simulation/runtime/observation_presets.py`，由任务选择 `table_down` 或 `forward`，不读取物体位置来决定启动目标。`simulation/runtime/observation_startup.py` 从零位执行固定关节轨迹，到位后由 PPO 接管；所有课程共享同一预设起点。任务环境提供四维位置控制（29 维状态）或七维笛卡尔姿态控制（38 维状态），策略不读取 IK 关节目标，任务奖励不调用 IK。`simulation/motion/cartesian_controller.py`、工具坐标 FK/IK 适配层及共享执行器执行策略请求，平视起点的转向也来自策略。原 reach、grasp_lift 和七维 tabletop_pick 接口继续保留。详细命令见 [任务说明](tabletop_cartesian.md)。

`rgbd_hsv` 是该任务的独立视觉观测契约，由 `envs/factory.py` 选择视觉环境。`simulation/devices/mujoco_rgbd_camera.py` 只提供原腕部相机的 RGB、对齐深度与机器人相机标定；中立的 `perception/rgbd.py` 从这些传感器数据生成 Mask、基座位置及视觉速度。视觉环境以此构造全部物体观测、目标和完成判断，网络不读取模拟器物体位置、速度、受力或保持进度。物理真值仅保留于动力学、安全、奖励和独立验收。`rl/vision_policy.py` 输出神经末端和夹爪动作，继续复用原 FK/IK；视觉模型与原真值模型严格区分。详见 [视觉抓取接口](tabletop_vision.md)。
