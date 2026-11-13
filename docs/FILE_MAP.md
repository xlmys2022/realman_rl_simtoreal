# Python 文件地图

路径均相对仓库根目录。共 81 个 Python 文件：核心包 35、脚本 28、测试 18。包内每层的 `__init__.py` 只作轻量包标识。下面列出全部 Python 文件；主要调用者依据实际导入与入口用途整理，不将脚本的文件名当作可导入业务接口。

新增 / 移动 Python 文件时，同步维护本表、[架构](architecture.md)、当前 README 命令和相应测试路径。旧路径见 [迁移对照](validation/PROJECT_ORGANIZATION_REFACTOR.md)。

## 运行入口

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `scripts/demos/control_ee.py` | 末端 IK / 控制演示 | CLI / demos | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/demos/control_joints.py` | 关节控制演示与验收 | CLI / demos | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/demos/demo_gripper.py` | 夹爪开合、带载保持演示 | CLI / demos | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/demos/run_grasp_contact_cases.py` | 接触、抬升、释放和失败对照案例 | CLI / demos | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/demos/view_robot.py` | 静态机器人模型 Viewer | CLI / demos | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/compare_controllers.py` | PD 与重力补偿对照 | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/compare_fk.py` | 厂家 / MuJoCo FK 对照 | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/compare_ik.py` | 厂家离线 IK 自模型与跨模型验证 | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/compare_sac_boundary.py` | SAC 边界单因素实验及冻结最终评估 | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/diagnose_sac_reach.py` | 冻结 SAC 轨迹、反例与全 Replay TD 诊断 | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/probe_reach_env.py` | 任务分布、奖励与随机探索探针，无学习器 | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/diagnostics/probe_ppo_goals.py` | 冻结PPO在合法reset目标下的同状态特征/动作敏感性，不执行step或训练 | CLI / diagnostics | 用户命令行；`tests/test_goal_conditioned_reach.py` | 是（CLI） |
| `scripts/diagnostics/smoke_reach_env.py` | 小规模无界面 Reach 交互 / Gymnasium checker | CLI / diagnostics | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/evaluate/evaluate_bc_reach.py` | 冻结 BC / 未训练 / 规则配对评估 | CLI / evaluate | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/evaluate/analyze_reach_results.py` | 只读评估JSON；逐候选、方向、象限、半径组及动作偏置/裁剪角度，不加载模型 | CLI / evaluate | 用户命令行；`tests/test_ppo_random_reach.py` | 是（CLI） |
| `scripts/evaluate/evaluate_ppo_reach.py` | 加载PPO及保存变换；离散逐目标/随机独立目标评估，重复数拒绝 | CLI / evaluate | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/evaluate/evaluate_reach_baselines.py` | 规则 / 随机、奖励反例与可选性能分析 | CLI / evaluate | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/evaluate/evaluate_sac_reach.py` | 冻结 SAC / 随机 / 规则配对目标评估 | CLI / evaluate | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/models/build_dynamics_model.py` | 重建六轴动力学 MJCF | CLI / models | 用户命令行；对应自动化测试 | 是（无参数生成器） |
| `scripts/models/build_grasp_scene.py` | 重建接触代理和动态方块场景 | CLI / models | 用户命令行；对应自动化测试 | 是（无参数生成器） |
| `scripts/models/build_gripper_model.py` | 重建独立 / 组合夹爪模型 | CLI / models | 用户命令行；对应自动化测试 | 是（无参数生成器） |
| `scripts/models/inspect_model.py` | URDF / MJCF、mesh、惯量和关节检查 | CLI / models | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/plots/plot_ppo_training.py` | 只读PPO JSON；聚合及逐目标成功/距离/首动作、训练成功率/终止PNG | CLI / plots | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/plots/plot_sac_training.py` | 只读 SAC JSON，输出曲线 PNG 和原始序列 | CLI / plots | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/train/collect_reach_demonstrations.py` | 公开观测规则策略示范数据采集 | CLI / train | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/train/train_bc_reach.py` | 离线 BC MSE / Adam 训练入口 | CLI / train | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/train/train_ppo_reach.py` | PPO完整更新后评估/保存；离散任务自动全部候选，scratch | CLI / train | 用户命令行；对应自动化测试 | 是（CLI） |
| `scripts/train/train_sac_reach.py` | SAC 训练调度、纯 RL / 显式历史初始化分支 | CLI / train | 用户命令行；对应自动化测试；diagnostics/compare_sac_boundary.py | 是（CLI） |

## 核心包

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `src/realman_rl_simtoreal/__init__.py` | 轻量包标识，无运行时初始化 | 包基础 | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/envs/__init__.py` | 轻量包标识，无运行时初始化 | envs | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/envs/reach_env.py` | 六轴Reach；原固定/球壳及离散候选reset、全候选预检，原观测/奖励/step不变 | envs | `scripts/diagnostics/probe_reach_env.py`<br>`scripts/diagnostics/smoke_reach_env.py`<br>`scripts/evaluate/evaluate_reach_baselines.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>等（共 17 处静态调用） | 否 |
| `src/realman_rl_simtoreal/imitation/__init__.py` | 轻量包标识，无运行时初始化 | imitation | `tests/test_bc_reach.py` | 否 |
| `src/realman_rl_simtoreal/imitation/bc.py` | 专家示范、离线 BC 网络训练、权重保存加载和闭环评估 | imitation | `scripts/evaluate/evaluate_bc_reach.py`<br>`scripts/train/collect_reach_demonstrations.py`<br>`scripts/train/train_bc_reach.py`<br>`src/realman_rl_simtoreal/imitation/bc_to_sac.py`<br>等（共 6 处静态调用） | 否 |
| `src/realman_rl_simtoreal/imitation/bc_to_sac.py` | 历史 BC→SAC Actor 均值通路校验、复制与偏差诊断 | imitation | `scripts/train/train_sac_reach.py`<br>`src/realman_rl_simtoreal/rl/sac_monitoring.py`<br>`tests/test_sac_bc_initialization.py` | 否 |
| `src/realman_rl_simtoreal/paths.py` | 源码仓库资源与输出目录定位，不依赖 cwd | 包基础 | `scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/demos/view_robot.py`<br>`scripts/diagnostics/compare_controllers.py`<br>等（共 30 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/__init__.py` | 轻量包标识，无运行时初始化 | rl | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/rl/common/__init__.py` | 轻量包标识，无运行时初始化 | common | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/rl/common/artifacts.py` | JSON、依赖版本、资源指纹和空输出目录检查 | common | `scripts/diagnostics/compare_sac_boundary.py`<br>`scripts/diagnostics/diagnose_sac_reach.py`<br>`scripts/diagnostics/probe_reach_env.py`<br>`scripts/evaluate/evaluate_bc_reach.py`<br>等（共 16 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/baselines.py` | 仅使用公开观测的原规则策略 | common | `scripts/evaluate/evaluate_reach_baselines.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>`src/realman_rl_simtoreal/rl/common/evaluation.py`<br>`src/realman_rl_simtoreal/rl/sac_diagnostics.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/env_factory.py` | 原训练包装栈、公开 reset 观测探针 | common | `scripts/train/train_ppo_reach.py`<br>`scripts/train/train_sac_reach.py`<br>`tests/test_bc_reach.py`<br>`tests/test_pure_reach_training.py`<br>等（共 8 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/evaluation.py` | 共享确定性评估；离散逐候选枚举、per_goal、路径及TensorBoard | common | `scripts/diagnostics/compare_sac_boundary.py`<br>`scripts/evaluate/evaluate_sac_reach.py`<br>`scripts/train/train_ppo_reach.py`<br>`scripts/train/train_sac_reach.py`<br>等（共 8 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/metrics.py` | 只读 SI 运动、路径、动作方向统计 | common | `scripts/diagnostics/probe_reach_env.py`<br>`src/realman_rl_simtoreal/rl/common/evaluation.py`<br>`src/realman_rl_simtoreal/rl/ppo.py`<br>`src/realman_rl_simtoreal/rl/sac_monitoring.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/common/observation.py` | 固定 schema 2 观测变换、包装器、33 维特征顺序 | common | `scripts/evaluate/evaluate_sac_reach.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>`src/realman_rl_simtoreal/imitation/bc_to_sac.py`<br>`src/realman_rl_simtoreal/rl/common/env_factory.py`<br>等（共 14 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/ppo.py` | 官方PPO配置、Rollout/裁剪审计、独立或离散目标评估范围、checkpoint | rl | `scripts/evaluate/evaluate_ppo_reach.py`<br>`scripts/train/train_ppo_reach.py`<br>`tests/test_ppo_reach.py`<br>`tests/test_ppo_random_reach.py`<br>`scripts/diagnostics/probe_ppo_goals.py`<br>`tests/test_goal_conditioned_reach.py` | 否 |
| `src/realman_rl_simtoreal/rl/sac.py` | 官方 SAC 配置、模型创建及模型 / Replay checkpoint | rl | `scripts/train/train_sac_reach.py`<br>`tests/test_pure_reach_training.py`<br>`tests/test_sac_bc_initialization.py`<br>`tests/test_sac_diagnostics.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/sac_diagnostics.py` | 冻结策略轨迹、Q / soft TD、参数指纹和 Replay 加载 | rl | `scripts/diagnostics/compare_sac_boundary.py`<br>`scripts/diagnostics/diagnose_sac_reach.py`<br>`scripts/train/train_sac_reach.py`<br>`src/realman_rl_simtoreal/rl/sac_monitoring.py`<br>等（共 7 处静态调用） | 否 |
| `src/realman_rl_simtoreal/rl/sac_monitoring.py` | SAC 训练 / 安全审计、运动与标量监控、显式 BC 参照分支 | rl | `scripts/train/train_sac_reach.py` | 否 |
| `src/realman_rl_simtoreal/simulation/__init__.py` | 轻量包标识，无运行时初始化 | simulation | `scripts/demos/control_ee.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`scripts/diagnostics/compare_fk.py`<br>`scripts/diagnostics/compare_ik.py`<br>等（共 15 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/control.py` | 关节插值控制、固定周期步进、ForceMonitor 与原保护规则 | simulation | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/diagnostics/compare_controllers.py`<br>等（共 19 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/ee_control.py` | 末端命令范围、碰撞预检、运动执行及误差 | simulation | `scripts/demos/control_ee.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`src/realman_rl_simtoreal/envs/reach_env.py`<br>`tests/test_gravity_compensation.py`<br>等（共 7 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/grasp_contact.py` | 接触分类、力记录、动态方块及同步控制案例基础 | simulation | `scripts/demos/run_grasp_contact_cases.py`<br>`tests/test_grasp_contact.py` | 否 |
| `src/realman_rl_simtoreal/simulation/gripper.py` | 米制夹爪开口、mimic 校验、执行器与物理步进 | simulation | `scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/models/build_grasp_scene.py`<br>`src/realman_rl_simtoreal/simulation/grasp_contact.py`<br>等（共 6 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/kinematics.py` | 独立 MjData 中的 FK、Jacobian、限位 DLS IK | simulation | `scripts/demos/control_ee.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`scripts/diagnostics/compare_fk.py`<br>等（共 16 处静态调用） | 否 |
| `src/realman_rl_simtoreal/simulation/model.py` | MuJoCo 模型加载、映射、初始化、重力补偿配置与结构检查 | simulation | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>等（共 19 处静态调用） | 否 |
| `src/realman_rl_simtoreal/tasks/__init__.py` | 轻量包标识，无运行时初始化 | tasks | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/tasks/reach.py` | 算法无关fixed / easy_random / random / axis_pair / axis_six显式目标配置 | tasks | `scripts/diagnostics/probe_reach_env.py`<br>`scripts/train/train_ppo_reach.py`<br>`scripts/train/train_sac_reach.py`<br>`src/realman_rl_simtoreal/imitation/bc.py`<br>等（共 14 处静态调用） | 否 |
| `src/realman_rl_simtoreal/utils/__init__.py` | 轻量包标识，无运行时初始化 | utils | Python 包导入机制 | 否 |
| `src/realman_rl_simtoreal/utils/demo_helpers.py` | FK 样本姿态、原关节演示序列和结果格式化 | utils | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/diagnostics/compare_controllers.py`<br>`scripts/diagnostics/compare_fk.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/utils/geometry.py` | 共享 SE(3) 构造与逆变换 | utils | `scripts/diagnostics/compare_fk.py`<br>`scripts/diagnostics/compare_ik.py` | 否 |
| `src/realman_rl_simtoreal/vendor/__init__.py` | 轻量包标识，无运行时初始化 | vendor | `tests/test_fk_comparison.py`<br>`tests/test_ik_comparison.py` | 否 |
| `src/realman_rl_simtoreal/vendor/realman_fk.py` | 可选厂家离线 FK SDK 适配与会话保护 | vendor | `scripts/diagnostics/compare_fk.py`<br>`scripts/diagnostics/compare_ik.py`<br>`src/realman_rl_simtoreal/vendor/realman_ik.py`<br>`tests/test_fk_comparison.py`<br>等（共 5 处静态调用） | 否 |
| `src/realman_rl_simtoreal/vendor/realman_ik.py` | 厂家离线 IK、解集与自模型复核 | vendor | `scripts/diagnostics/compare_ik.py`<br>`tests/test_ik_comparison.py` | 否 |
| `src/realman_rl_simtoreal/viewer.py` | 惰性 Viewer 生命周期、后端选择与线程清理 | 包基础 | `scripts/demos/control_ee.py`<br>`scripts/demos/control_joints.py`<br>`scripts/demos/demo_gripper.py`<br>`scripts/demos/run_grasp_contact_cases.py`<br>等（共 9 处静态调用） | 否 |

## 自动化测试

| 文件路径 | 功能 | 所属层次 | 主要调用者 | 是否为运行入口 |
|---|---|---|---|---|
| `tests/test_bc_reach.py` | BC 数据、特征、模型、监督更新与闭环 | 测试 | unittest discover | 是（测试） |
| `tests/test_fk_comparison.py` | FK 坐标、厂家配置与差异对照 | 测试 | unittest discover | 是（测试） |
| `tests/test_grasp_contact.py` | 接触、代理、同步步进与严格失败反例 | 测试 | unittest discover | 是（测试） |
| `tests/test_gravity_compensation.py` | 重力补偿路由、合力矩限幅及控制对照 | 测试 | unittest discover | 是（测试） |
| `tests/test_gripper.py` | 夹爪结构、mimic、开口、模型重生成和演示 | 测试 | unittest discover | 是（测试） |
| `tests/test_goal_conditioned_reach.py` | 离散候选、旧模式精确回归、一次两方向PPO更新及逐目标评估/监控 | 测试 | unittest discover | 是（测试） |
| `tests/test_ik_comparison.py` | 厂家 IK 自模型 / 边界与跨模型 | 测试 | unittest discover | 是（测试） |
| `tests/test_joint_control.py` | 动力学结构、保持、关节运动、力矩与碰撞 | 测试 | unittest discover | 是（测试） |
| `tests/test_kinematics.py` | FK / Jacobian / DLS IK / 末端控制与资源重生成 | 测试 | unittest discover | 是（测试） |
| `tests/test_package.py` | editable / 包发现 / 导入无环 / 分层边界 / 23 个 CLI help / 退出码与 Viewer 生命周期 | 测试 | unittest discover | 是（测试） |
| `tests/test_ppo_reach.py` | 单次完整 PPO Rollout 更新、动作与损失、日志、checkpoint；阻断 SAC / BC 导入 | 测试 | unittest discover | 是（测试） |
| `tests/test_ppo_random_reach.py` | 200个随机目标/观测/reset契约、分组与裁剪分析、单512步随机PPO Smoke及20目标评估接口 | 测试 | unittest discover | 是（测试） |
| `tests/test_pure_reach_training.py` | 任务配置、奖励 / 探针、纯 SAC 隔离和运动统计 | 测试 | unittest discover | 是（测试） |
| `tests/test_reach_env.py` | Gymnasium、动作、观测、奖励、真实物理步与保护 | 测试 | unittest discover | 是（测试） |
| `tests/test_reach_task.py` | 随机可达目标、奖励一致性、基线及 CLI | 测试 | unittest discover | 是（测试） |
| `tests/test_sac_bc_initialization.py` | 历史 BC Actor 映射、受保护参数、短 SAC 更新与监控 | 测试 | unittest discover | 是（测试） |
| `tests/test_sac_diagnostics.py` | 冻结动作轨迹、Q / TD / Replay、随机状态与参数不变性 | 测试 | unittest discover | 是（测试） |
| `tests/test_sac_reach.py` | 官方 SAC 参数更新、预处理、保存 / 加载和确定性评估 | 测试 | unittest discover | 是（测试） |

## 非 Python 资源

`tests/fixtures/reach_terminate_04a.json`保留旧terminate逐值基准；`reach_env_legacy_contract.json`冻结04-F3前的旧模式轨迹和step/奖励/安全等7个方法AST摘要，允许新增reset能力而不弱化原保护。`assets/`保留模型来源，`outputs/`保留已有checkpoint、数据、日志与图像；它们不因文件索引更新而重写。

资产内 `SOURCE.md` 属于原版本来源记录，旧生成命令按上表映射到 `scripts/models/`；现行命令见仓库 README。
