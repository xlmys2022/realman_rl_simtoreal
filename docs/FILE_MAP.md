# Python 文件地图

共 134 个 Python 文件：核心包 65、脚本 41、测试 28。主要文件及新增任务入口列在下方；路径相对仓库根目录，动态进程调用另由测试覆盖。

新增或移动文件时同步维护本表、[架构](architecture.md)和相应测试路径。[Evaluation 重构报告](validation/EVALUATION_REFACTOR_VALIDATION.md)解释本次迁移及接口；[输出导航](../outputs/README.md)用于查实验产物。

水平桌面任务新增文件：

- `scripts/models/build_tabletop_scene.py`：独立水平桌面、零位机械臂和自由方块场景生成器。
- `scripts/train/train_ppo_tabletop.py`：七维 PPO 课程训练、来源记录、完整更新 checkpoint 和停止文件。
- `scripts/evaluate/evaluate_ppo_tabletop.py`：冻结模型独立评估，`--full` 恢复零位完整任务。
- `scripts/demos/render_ppo_tabletop.py`：同一冻结 PPO 策略的桌面全景和夹爪特写回放。
- `src/realman_rl_simtoreal/tasks/tabletop_pick.py`：零位、方块、桌面、抬升及执行器配置。
- `src/realman_rl_simtoreal/envs/tabletop_pick_env.py`：56 维观测、六轴绝对角度和夹爪开口，共享 1 ms 物理时钟。
- `src/realman_rl_simtoreal/rl/ppo_tabletop.py`：固定编码、奖励、持续相关探索、PPO checkpoint 和参数不可变评估适配。
- `tests/test_tabletop_pick.py`：桌面、精确起点、七维动作、真实 PPO 更新、完整协方差和只读推理回归。

## 命令行入口

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `scripts/demos/control_ee.py` | 末端 IK / 控制演示 | CLI / demos | 用户命令行；对应测试 | 是（CLI） |
| `scripts/demos/control_joints.py` | 关节控制演示与验收 | CLI / demos | 用户命令行；对应测试 | 是（CLI） |
| `scripts/demos/demo_gripper.py` | 夹爪开合、带载保持演示 | CLI / demos | 用户命令行；对应测试 | 是（CLI） |
| `scripts/demos/run_grasp_contact_cases.py` | 接触、抬升、释放和失败对照案例 | CLI / demos | 用户命令行；对应测试 | 是（CLI） |
| `scripts/demos/run_grasp_lift.py` | Approach, close, lift and hold a dynamic cube through four-action controls. | CLI / demos | 用户命令行；对应测试 | 是（CLI） |
| `scripts/demos/view_robot.py` | 静态机器人模型 Viewer | CLI / demos | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/compare_controllers.py` | PD 与重力补偿对照 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/compare_fk.py` | 厂家 / MuJoCo FK 对照 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/compare_ik.py` | 厂家离线 IK 自模型与跨模型验证 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/compare_sac_boundary.py` | SAC 边界单因素实验及冻结最终评估 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/diagnose_reach_rewards.py` | 独立Home动作方向、旧奖励解析、同轨迹新旧计分与四张诊断图；不训练 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/diagnose_sac_reach.py` | 冻结 SAC 轨迹、反例与全 Replay TD 诊断 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/probe_ppo_goals.py` | 冻结PPO在合法reset目标下的同状态特征/动作敏感性，不执行step或训练 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/probe_reach_env.py` | 任务分布、奖励与随机探索探针，无学习器 | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/diagnostics/smoke_reach_env.py` | 小规模无界面 Reach 交互 / Gymnasium checker | CLI / diagnostics | 用户命令行；对应测试 | 是（CLI） |
| `scripts/evaluate/analyze_reach_results.py` | 只读评估JSON；逐候选、方向、象限、半径组及动作偏置/裁剪角度，不加载模型 | CLI / evaluate | 用户命令行；对应测试 | 是（CLI） |
| `scripts/evaluate/evaluate_bc_reach.py` | BC 模型加载与配对评估 CLI，经 BC 权重保护薄适配调用 Evaluation | CLI / evaluate | 用户命令行；对应测试 | 是（CLI） |
| `scripts/evaluate/evaluate_ppo_grasp_lift.py` | Independent evaluation of four-action PPO, with no staged-controller fallback. | CLI / evaluate | 用户命令行；对应测试 | 是（CLI） |
| `scripts/evaluate/evaluate_ppo_reach.py` | 加载PPO及保存变换；离散逐目标/随机独立目标评估，重复数拒绝 | CLI / evaluate | 用户命令行；对应测试 | 是（CLI） |
| `scripts/evaluate/evaluate_reach_baselines.py` | 随机/规则评估 CLI；计算与循环直接导入 evaluation.baselines，保留参数/JSON/退出码 | CLI / evaluate | 用户命令行；对应测试 | 是（CLI） |
| `scripts/evaluate/evaluate_sac_reach.py` | 冻结 SAC / 随机 / 规则配对目标评估 | CLI / evaluate | 用户命令行；对应测试 | 是（CLI） |
| `scripts/models/build_dynamics_model.py` | 重建六轴动力学 MJCF | CLI / models | 用户命令行；对应测试 | 是（无参数生成器） |
| `scripts/models/build_grasp_scene.py` | 重建接触代理和动态方块场景 | CLI / models | 用户命令行；对应测试 | 是（无参数生成器） |
| `scripts/models/build_gripper_model.py` | 重建独立 / 组合夹爪模型 | CLI / models | 用户命令行；对应测试 | 是（无参数生成器） |
| `scripts/models/inspect_model.py` | URDF / MJCF、mesh、惯量和关节检查 | CLI / models | 用户命令行；对应测试 | 是（CLI） |
| `scripts/plots/plot_ppo_training.py` | 只读PPO JSON；聚合及逐目标成功/距离/首动作、训练成功率/终止PNG | CLI / plots | 用户命令行；对应测试 | 是（CLI） |
| `scripts/plots/plot_sac_training.py` | 只读 SAC JSON，输出曲线 PNG 和原始序列 | CLI / plots | 用户命令行；对应测试 | 是（CLI） |
| `scripts/train/collect_reach_demonstrations.py` | 公开观测规则策略示范数据采集 | CLI / train | 用户命令行；对应测试 | 是（CLI） |
| `scripts/train/train_bc_reach.py` | 离线 BC MSE / Adam 训练入口 | CLI / train | 用户命令行；对应测试 | 是（CLI） |
| `scripts/train/train_ppo_grasp_lift.py` | Train four-action PPO from scratch or a recorded PPO curriculum checkpoint. | CLI / train | 用户命令行；对应测试 | 是（CLI） |
| `scripts/train/train_ppo_reach.py` | PPO完整更新后评估/保存；离散任务自动全部候选，scratch | CLI / train | 用户命令行；对应测试 | 是（CLI） |
| `scripts/train/train_sac_reach.py` | SAC 训练调度、纯 RL / 显式历史初始化分支 | CLI / train | 用户命令行；对应测试 | 是（CLI） |

## Evaluation 评估层

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `src/realman_rl_simtoreal/evaluation/__init__.py` | 轻量评估包标识；不导入环境或学习框架 | evaluation | `tests/test_evaluation.py`<br>`tests/test_fetch_observation.py` | 否 |
| `src/realman_rl_simtoreal/evaluation/baselines.py` | 原公开观测规则策略；早期随机/规则评估、奖励反例与独立性能剖析 | evaluation | `scripts/evaluate/evaluate_reach_baselines.py`<br>`src/realman_rl_simtoreal/evaluation/reach.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>`src/realman_rl_simtoreal/rl/sac_diagnostics.py`<br>等（共 7 处静态调用） | 否 |
| `src/realman_rl_simtoreal/evaluation/grasp_lift.py` | 原 GraspLift rollout / evaluate；阶段控制器或传入 predict 策略共用物理执行循环 | evaluation | `scripts/demos/run_grasp_lift.py`<br>`src/realman_rl_simtoreal/rl/ppo_grasp.py`<br>`tests/test_evaluation.py`<br>`tests/test_grasp_lift.py` | 否 |
| `src/realman_rl_simtoreal/evaluation/metrics.py` | Reach / GraspLift / BC / 早期基线的原公式、成功率分母、延迟与目标分布统计 | evaluation | `scripts/evaluate/evaluate_reach_baselines.py`<br>`src/realman_rl_simtoreal/evaluation/baselines.py`<br>`src/realman_rl_simtoreal/evaluation/grasp_lift.py`<br>`src/realman_rl_simtoreal/evaluation/reach.py`<br>等（共 9 处静态调用） | 否 |
| `src/realman_rl_simtoreal/evaluation/reach.py` | 独立 Reach 回合、原 RL / BC JSON 协议和原始观测 callable 接口；不加载或训练算法 | evaluation | `scripts/diagnostics/compare_sac_boundary.py`<br>`scripts/evaluate/evaluate_sac_reach.py`<br>`scripts/train/train_sac_reach.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>等（共 10 处静态调用） | 否 |

## 其他核心包模块

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `src/realman_rl_simtoreal/__init__.py` | 轻量包标识，无运行时初始化 | 包基础 | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/envs/__init__.py` | 轻量包标识，无运行时初始化 | envs | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/envs/factory.py` | 按保存配置创建普通或连续目标 Reach 环境，不依赖评估或算法 | envs | `src/realman_rl_simtoreal/evaluation/reach.py`<br>`src/realman_rl_simtoreal/rl/common/env_factory.py`<br>`tests/test_sequence_reach.py` | 否 |
| `src/realman_rl_simtoreal/envs/grasp_lift_env.py` | 抓取抬升 Gymnasium 环境，41 维观测、四维动作、实际接触与安全终止 | envs | `scripts/demos/run_grasp_lift.py`<br>`src/realman_rl_simtoreal/evaluation/grasp_lift.py`<br>`src/realman_rl_simtoreal/rl/ppo_grasp.py`<br>`tests/test_grasp_lift.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/envs/reach_env.py` | 六轴Reach；默认legacy 33维、可选fetch_like_v1 12维；固定/球壳/离散目标，原奖励/step不变 | envs | `scripts/diagnostics/diagnose_reach_rewards.py`<br>`scripts/diagnostics/probe_ppo_goals.py`<br>`scripts/diagnostics/probe_reach_env.py`<br>`scripts/diagnostics/smoke_reach_env.py`<br>等（共 24 处静态调用） | 否 |
| `src/realman_rl_simtoreal/envs/sequence_reach_env.py` | 随机起点、连续多目标 Reach 与物理状态交接，保留 Gymnasium 接口 | envs | `src/realman_rl_simtoreal/envs/factory.py`<br>`tests/test_sequence_reach.py` | 否 |
| `src/realman_rl_simtoreal/imitation/__init__.py` | 轻量包标识，无运行时初始化 | imitation | `tests/test_bc_reach.py` | 否 |
| `src/realman_rl_simtoreal/imitation/bc.py` | 原 BC 网络、数据采集和监督训练；闭环评估仅保留权重保护薄适配 | imitation | `scripts/evaluate/evaluate_bc_reach.py`<br>`scripts/train/collect_reach_demonstrations.py`<br>`scripts/train/train_bc_reach.py`<br>`src/realman_rl_simtoreal/imitation/bc_to_sac.py`<br>等（共 7 处静态调用） | 否 |
| `src/realman_rl_simtoreal/imitation/bc_to_sac.py` | 历史 BC→SAC Actor 均值通路校验、复制与偏差诊断 | imitation | `scripts/train/train_sac_reach.py`<br>`src/realman_rl_simtoreal/rl/sac_monitoring.py`<br>`tests/test_sac_bc_initialization.py` | 否 |
| `src/realman_rl_simtoreal/paths.py` | 源码仓库资源与输出目录定位，不依赖 cwd | 包基础 | `scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/demos/run_grasp_lift.py`<br>`scripts/demos/view_robot.py`<br>等（共 37 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/__init__.py` | 轻量包标识，无运行时初始化 | rl | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/rl/common/__init__.py` | 轻量包标识，无运行时初始化 | common | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/rl/common/artifacts.py` | JSON、依赖版本、资源指纹和空输出目录检查 | common | `scripts/demos/run_grasp_lift.py`<br>`scripts/diagnostics/compare_sac_boundary.py`<br>`scripts/diagnostics/diagnose_sac_reach.py`<br>`scripts/diagnostics/probe_ppo_goals.py`<br>等（共 20 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/env_factory.py` | 原训练包装栈、公开 reset 观测探针 | common | `scripts/train/train_ppo_reach.py`<br>`scripts/train/train_sac_reach.py`<br>`tests/test_bc_reach.py`<br>`tests/test_fetch_observation.py`<br>等（共 11 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/logging.py` | 原 record_evaluation 的训练 logger 写入；TensorBoard 标签不变，不执行 Episode | common | `scripts/train/train_ppo_reach.py`<br>`scripts/train/train_sac_reach.py` | 否 |
| `src/realman_rl_simtoreal/rl/common/observation.py` | legacy schema 2 / fetch schema 3固定变换；33/12维顺序、checkpoint契约与模式检查 | common | `scripts/diagnostics/probe_ppo_goals.py`<br>`scripts/evaluate/evaluate_sac_reach.py`<br>`scripts/train/train_ppo_reach.py`<br>`scripts/train/train_sac_reach.py`<br>等（共 21 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/ppo.py` | 官方 PPO 配置、训练审计、checkpoint、确定性推理和参数保护薄适配；评估循环在 evaluation | rl | `scripts/diagnostics/diagnose_reach_rewards.py`<br>`scripts/diagnostics/probe_ppo_goals.py`<br>`scripts/evaluate/evaluate_ppo_reach.py`<br>`scripts/train/train_ppo_reach.py`<br>等（共 11 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/ppo_grasp.py` | 四维官方 PPO、课程训练/编码/奖励、checkpoint；抓取评估只保留策略薄适配 | rl | `scripts/evaluate/evaluate_ppo_grasp_lift.py`<br>`scripts/train/train_ppo_grasp_lift.py`<br>`tests/test_ppo_grasp.py` | 否 |
| `src/realman_rl_simtoreal/rl/sac.py` | 官方 SAC 配置、模型创建及模型 / Replay checkpoint | rl | `scripts/train/train_sac_reach.py`<br>`tests/test_fetch_observation.py`<br>`tests/test_pure_reach_training.py`<br>`tests/test_sac_bc_initialization.py`<br>等（共 6 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/sac_diagnostics.py` | 冻结策略轨迹、Q / soft TD、参数指纹和 Replay 加载 | rl | `scripts/diagnostics/compare_sac_boundary.py`<br>`scripts/diagnostics/diagnose_sac_reach.py`<br>`scripts/train/train_sac_reach.py`<br>`src/realman_rl_simtoreal/rl/sac_monitoring.py`<br>等（共 8 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/sac_monitoring.py` | 原 SAC 训练/安全审计、只读 Q 与运动监控，消费 evaluation.metrics 与中立统计 | rl | `scripts/train/train_sac_reach.py` | 否 |
| `src/realman_rl_simtoreal/simulation/__init__.py` | 轻量包标识，无运行时初始化 | simulation | `scripts/demos/control_ee.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`scripts/diagnostics/compare_fk.py`<br>`scripts/diagnostics/compare_ik.py`<br>等（共 15 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/control.py` | 关节插值控制、固定周期步进、ForceMonitor 与原保护规则 | simulation | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/diagnostics/compare_controllers.py`<br>等（共 21 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/ee_control.py` | 末端命令范围、碰撞预检、运动执行及误差 | simulation | `scripts/demos/control_ee.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`src/realman_rl_simtoreal/envs/reach_env.py`<br>`src/realman_rl_simtoreal/envs/sequence_reach_env.py`<br>等（共 8 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/grasp_contact.py` | 接触分类、力记录、动态方块及同步控制案例基础 | simulation | `scripts/demos/run_grasp_contact_cases.py`<br>`src/realman_rl_simtoreal/envs/grasp_lift_env.py`<br>`tests/test_grasp_contact.py` | 否 |
| `src/realman_rl_simtoreal/simulation/grasp_controller.py` | 独立阶段抓取控制器；公开反馈对中、接近、闭合、抬升和保持 | simulation | `src/realman_rl_simtoreal/evaluation/grasp_lift.py` | 否 |
| `src/realman_rl_simtoreal/simulation/gripper.py` | 米制夹爪开口、mimic 校验、执行器与物理步进 | simulation | `scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/models/build_grasp_scene.py`<br>`src/realman_rl_simtoreal/envs/grasp_lift_env.py`<br>等（共 8 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/kinematics.py` | 独立 MjData 中的 FK、Jacobian、限位 DLS IK | simulation | `scripts/demos/control_ee.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`scripts/diagnostics/compare_fk.py`<br>等（共 20 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/model.py` | MuJoCo 模型加载、映射、初始化、重力补偿配置与结构检查 | simulation | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>等（共 19 处静态调用） | 否 |
| `src/realman_rl_simtoreal/tasks/__init__.py` | 轻量包标识，无运行时初始化 | tasks | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/tasks/grasp_lift.py` | 抓取任务物体、接近、开口、高度、保持与局部保护配置 | tasks | `scripts/demos/run_grasp_lift.py`<br>`src/realman_rl_simtoreal/envs/grasp_lift_env.py`<br>`src/realman_rl_simtoreal/rl/ppo_grasp.py`<br>`tests/test_evaluation.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/tasks/reach.py` | 算法无关fixed / easy_random / random / axis_pair / axis_six显式目标配置 | tasks | `scripts/diagnostics/diagnose_reach_rewards.py`<br>`scripts/diagnostics/probe_ppo_goals.py`<br>`scripts/diagnostics/probe_reach_env.py`<br>`scripts/train/train_ppo_reach.py`<br>等（共 18 处静态调用） | 否 |
| `src/realman_rl_simtoreal/tasks/rewards.py` | 奖励模式、进度奖励8项实验初值、组件键与checkpoint一致性校验 | tasks | `scripts/diagnostics/diagnose_reach_rewards.py`<br>`scripts/train/train_ppo_reach.py`<br>`src/realman_rl_simtoreal/envs/reach_env.py`<br>`src/realman_rl_simtoreal/evaluation/reach.py`<br>等（共 7 处静态调用） | 否 |
| `src/realman_rl_simtoreal/utils/__init__.py` | 轻量包标识，无运行时初始化 | utils | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/utils/artifacts.py` | 训练/评估/演示共享的抓取资源、源码与运行时指纹；不依赖算法 | utils | `scripts/demos/run_grasp_lift.py`<br>`scripts/train/train_ppo_grasp_lift.py`<br>`src/realman_rl_simtoreal/evaluation/grasp_lift.py`<br>`src/realman_rl_simtoreal/rl/ppo_grasp.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/utils/demo_helpers.py` | FK 样本姿态、原关节演示序列和结果格式化 | utils | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`scripts/diagnostics/compare_fk.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/utils/geometry.py` | 共享 SE(3) 构造与逆变换 | utils | `scripts/diagnostics/compare_fk.py`<br>`scripts/diagnostics/compare_ik.py` | 否 |
| `src/realman_rl_simtoreal/utils/metrics.py` | 训练与评估共享 MotionStats 和原余弦函数；只依赖 NumPy | utils | `scripts/diagnostics/probe_reach_env.py`<br>`src/realman_rl_simtoreal/evaluation/reach.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>`src/realman_rl_simtoreal/rl/ppo.py`<br>等（共 7 处静态调用） | 否 |
| `src/realman_rl_simtoreal/vendor/__init__.py` | 轻量包标识，无运行时初始化 | vendor | `tests/test_fk_comparison.py`<br>`tests/test_ik_comparison.py` | 否 |
| `src/realman_rl_simtoreal/vendor/realman_fk.py` | 可选厂家离线 FK SDK 适配与会话保护 | vendor | `scripts/diagnostics/compare_fk.py`<br>`scripts/diagnostics/compare_ik.py`<br>`src/realman_rl_simtoreal/vendor/realman_ik.py`<br>`tests/test_fk_comparison.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/vendor/realman_ik.py` | 厂家离线 IK、解集与自模型复核 | vendor | `scripts/diagnostics/compare_ik.py`<br>`tests/test_ik_comparison.py` | 否 |
| `src/realman_rl_simtoreal/viewer.py` | 惰性 Viewer 生命周期、后端选择与线程清理 | 包基础 | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>等（共 11 处静态调用） | 否 |

## 自动化测试

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `tests/test_bc_reach.py` | BC 数据、特征、模型、监督更新与闭环 | 测试 | unittest discover | 是（测试） |
| `tests/test_evaluation.py` | 17 项评估接口/指标/独立环境/参数保护/无算法导入及 JSON 协议测试 | 测试 | unittest discover | 是（测试） |
| `tests/test_fetch_observation.py` | 双模式维度、真实Site运动、网络实际输入、物理一致性、旧checkpoint及单512步PPO Smoke | 测试 | unittest discover | 是（测试） |
| `tests/test_fk_comparison.py` | FK 坐标、厂家配置与差异对照 | 测试 | unittest discover | 是（测试） |
| `tests/test_goal_conditioned_reach.py` | 离散候选、旧模式精确回归、一次两方向PPO更新及逐目标评估/监控 | 测试 | unittest discover | 是（测试） |
| `tests/test_grasp_contact.py` | 接触、代理、同步步进与严格失败反例 | 测试 | unittest discover | 是（测试） |
| `tests/test_grasp_lift.py` | 实际抓取接口、物理循环、阶段控制器、失败对照与指标 | 测试 | unittest discover | 是（测试） |
| `tests/test_gravity_compensation.py` | 重力补偿路由、合力矩限幅及控制对照 | 测试 | unittest discover | 是（测试） |
| `tests/test_gripper.py` | 夹爪结构、mimic、开口、模型重生成和演示 | 测试 | unittest discover | 是（测试） |
| `tests/test_ik_comparison.py` | 厂家 IK 自模型 / 边界与跨模型 | 测试 | unittest discover | 是（测试） |
| `tests/test_joint_control.py` | 动力学结构、保持、关节运动、力矩与碰撞 | 测试 | unittest discover | 是（测试） |
| `tests/test_kinematics.py` | FK / Jacobian / DLS IK / 末端控制与资源重生成 | 测试 | unittest discover | 是（测试） |
| `tests/test_package.py` | editable / 包发现 / 导入无环 / 分层边界 / 23 个 CLI help / 退出码与 Viewer 生命周期 | 测试 | unittest discover | 是（测试） |
| `tests/test_ppo_grasp.py` | 四维 PPO 实际更新、课程、奖励、checkpoint 和只读策略评估 | 测试 | unittest discover | 是（测试） |
| `tests/test_ppo_random_reach.py` | 200个随机目标/观测/reset契约、分组与裁剪分析、单512步随机PPO Smoke及20目标评估接口 | 测试 | unittest discover | 是（测试） |
| `tests/test_ppo_reach.py` | 单次完整 PPO Rollout 更新、动作与损失、日志、checkpoint；阻断 SAC / BC 导入 | 测试 | unittest discover | 是（测试） |
| `tests/test_pure_reach_training.py` | 任务配置、奖励 / 探针、纯 SAC 隔离和运动统计 | 测试 | unittest discover | 是（测试） |
| `tests/test_reach_env.py` | Gymnasium、动作、观测、奖励、真实物理步与保护 | 测试 | unittest discover | 是（测试） |
| `tests/test_reach_progress_reward.py` | 25项奖励方向、批量、漏洞/边界、兼容、组件与单次512步PPO Smoke | 测试 | unittest discover | 是（测试） |
| `tests/test_reach_task.py` | 随机可达目标、奖励一致性、基线及 CLI | 测试 | unittest discover | 是（测试） |
| `tests/test_sac_bc_initialization.py` | 历史 BC Actor 映射、受保护参数、短 SAC 更新与监控 | 测试 | unittest discover | 是（测试） |
| `tests/test_sac_diagnostics.py` | 冻结动作轨迹、Q / TD / Replay、随机状态与参数不变性 | 测试 | unittest discover | 是（测试） |
| `tests/test_sac_reach.py` | 官方 SAC 参数更新、预处理、保存 / 加载和确定性评估 | 测试 | unittest discover | 是（测试） |
| `tests/test_sequence_reach.py` | 随机起点、连续目标、奖励上下文与状态交接 | 测试 | unittest discover | 是（测试） |
| `tests/test_wide_reach.py` | 20 cm 目标分布、原物理控制和 PPO 迁移契约 | 测试 | unittest discover | 是（测试） |

## 迁移与共享边界

旧 `rl/common/evaluation.py` 的执行器已进入 `evaluation/reach.py`，汇总进入 `evaluation/metrics.py`，训练日志写入进入 `rl/common/logging.py`；旧 `rl/grasp_lift.py` 已整体拆出评估、统计和中立来源工具，不保留重复实现或转发文件。

跨训练与评估的 MotionStats / cosine 位于 `utils/metrics.py`；`experiment_metadata` 位于 `utils/artifacts.py`。Evaluation 仅复用 `rl/common/observation.py` 的算法无关观测契约，不导入 PPO / SAC / BC 或学习框架。PPO 与 BC 的只读推理薄适配留在模型模块。

`tests/fixtures/` 的原数值基准、`assets/` 的模型与来源、`outputs/` 的模型 / 数据 / 日志均未因本次 Evaluation 重构改写。历史报告中的旧模块路径属于原版本，现行路径以本表为准。
## 新增桌面末端任务

`tabletop_cartesian` 与原有任务并列，配置、环境、训练分别位于 `tasks/tabletop_cartesian.py`、`envs/tabletop_cartesian_env.py`、`rl/ppo_tabletop_cartesian.py`。`simulation/observation_profiles.py` 定义与物体无关的两个固定观测预设；`simulation/cartesian_controller.py` 和 `simulation/observation_pose.py` 负责执行与传统启动，`simulation/tool_kinematics.py` 适配工具坐标并继承原 FK/IK 算法，继续共享夹爪和接触监控。入口位于现有 `scripts/train`、`scripts/evaluate`、`scripts/demos`；详见 [任务说明](tabletop_cartesian.md)。

`simulation/grasp_metrics.py` 从真实接触力提取夹爪内向法向分量，排除顶压和空爪闭合。`rl/cartesian_policy.py` 实现采样及 PPO 似然一致的状态相关探索幅度，以及价值分支/夹爪输出头的独立初始化；动作均值仍完全由网络输出。`tests/test_cartesian_learning.py` 覆盖奖励排序、终止与持续停留的回报界限、接触测量、探索分布及局部参数重置。

`rl/cartesian_reach_policy.py` 复用原 PPO Reach 的神经末端分支，增加独立夹爪网络和可学习的抬升目标门控。输入仍为任务的 29 维笛卡尔状态，目标编码不包含关节目标；训练入口通过 `--init-reach-model` 选择此结构，`--jaw-control delta` 使用增量夹爪参考。
