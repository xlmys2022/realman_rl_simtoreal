# 工程结构与调用约定（02-D、03-A/B任务、03-C1/C2夹爪、04-A/B SAC、04-C BC、04-D1初始化）

本项目采用 src 布局：可复用能力通过安装后的 `realman_rl_simtoreal` 包调用；脚本负责命令行、实验流程及窗口展示。这样未来环境不必导入一个演示脚本来获得控制器，也不会顺带执行参数解析、打印或打开 Viewer。

02-D只是迁移。03-A在原接口旁增加固定周期推进及最小Gymnasium环境，没有修改模型、DLS、原控制增益、限位、重力补偿、碰撞规则或厂家算法。本包版本为0.1.0，和机器人模型/厂家算法版本没有对应关系。

## 目录

```text
realman_rl_simtoreal/
├── assets/robots/rml63/             # 原始及派生模型、mesh、来源、校验与许可证
├── assets/grippers/eg2_4c2/         # 官方EG2-4C2、独立模型与实验组合模型
├── assets/scenes/grasp_contact.xml  # 独立动态方块/支撑台实验，不改变原资源
├── src/realman_rl_simtoreal/
│   ├── __init__.py                  # 轻量包入口，无运行时初始化
│   ├── paths.py                     # editable 源码仓库中的资源/输出路径
│   ├── viewer.py                    # 惰性的 CLI Viewer 生命周期与后端辅助
│   ├── envs/
│   │   ├── __init__.py              # 轻量标识，不自动注册或创建环境
│   │   └── reach_env.py             # 固定/随机目标、三维动作、奖励、Gymnasium API
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── model.py                 # 加载、模型检查、映射、初始化、纯重力诊断
│   │   ├── control.py               # 原控制段/力审计及新增固定周期推进
│   │   ├── gripper.py               # 米制宽度、mimic约束、独立执行器与物理推进
│   │   ├── grasp_contact.py         # 独立实验的接触分类、力记录及载荷配置
│   │   ├── ee_control.py            # 原有末端检查、碰撞预检、执行和误差
│   │   └── kinematics.py            # 原有 FK、Jacobian、DLS IK
│   ├── vendor/
│   │   ├── __init__.py
│   │   ├── realman_fk.py            # 可选离线 Algo FK
│   │   └── realman_ik.py            # 可选离线 Algo IK、全解与复核
│   └── experiments/
│       ├── __init__.py
│       ├── common.py                # 共享姿态、SE(3)、默认序列、规则策略、纯文本格式化
│       ├── sac_reach.py             # 可选SB3实验封装、固定缩放及只读审计
│       ├── sac_diagnostics.py       # 冻结策略轨迹、命令/实际误差及soft TD诊断
│       ├── bc_reach.py              # 专家数据、独立监督网络、预测误差和闭环对照
│       ├── sac_initialization.py    # 严格空间/语义校验，BC仅移植Actor均值通路
│       └── sac_monitoring.py        # 独立训练/评估标量、冻结BC探针及Q监控
├── scripts/
│   ├── inspect_model.py
│   ├── view_robot.py
│   ├── build_dynamics_model.py
│   ├── control_joints.py
│   ├── control_ee.py
│   ├── compare_controllers.py
│   ├── compare_fk.py
│   ├── compare_ik.py
│   ├── smoke_reach_env.py           # 无界面固定目标演示
│   ├── evaluate_reach_baselines.py  # 随机/规则策略、奖励反例及独立性能剖析
│   ├── build_gripper_model.py       # 校验并生成独立/组合模型
│   ├── demo_gripper.py              # 空载开合、带载保持与Viewer
│   ├── build_grasp_scene.py         # 从原齿面生成代理和动态方块场景
│   ├── test_grasp_contact.py        # 接触、抬升、释放及失败对照/Viewer
│   ├── train_sac_reach.py           # CPU SAC训练、5k分段及独立验证
│   ├── evaluate_sac_reach.py        # 加载模型、独立100目标和三策略对照
│   ├── diagnose_sac_reach.py        # 冻结30k策略/规则对照、奖励反例、全Replay诊断
│   ├── compare_sac_boundary.py      # 单因素10k边界实验及冻结模型最终评估
│   ├── collect_reach_demonstrations.py # 原公开观测规则策略示范
│   ├── train_bc_reach.py            # 离线MSE/Adam，无物理采样或RL更新
│   ├── evaluate_bc_reach.py         # 规则/未训练/BC保留目标闭环
│   └── plot_sac_training.py         # 只读日志导出PNG，不加载模型或环境
├── tests/                          # 234项；含12项BC→SAC及监控验证
├── docs/
│   ├── architecture.md
│   └── validation/                 # 历史实验、重构及环境验证
├── outputs/                        # Git忽略；实验和重构对照产物
├── pyproject.toml
├── requirements.txt
├── requirements-vendor.txt
├── requirements-rl.txt             # 可选CPU训练依赖
├── README.md
└── .gitignore
```

## 迁移关系与职责

| 原位置 | 新位置/保留职责 |
|---|---|
| control_joints 的 load_model / initialize / 映射与结构检查 | simulation.model；函数签名和返回三元组不变 |
| compare_controllers 的 gravity_torque | simulation.model；独立 MjData、零速度 qfrc_bias |
| control_joints 的 validate_target / run_segment / 控制阈值 | simulation.control；插值、步进、限位、接触和统计原样保留 |
| compare_controllers 的 ForceMonitor | simulation.control；补偿和电机合成后限幅及被动力旁路审计 |
| scripts/kinematics.py | simulation.kinematics；整体迁移，不留副本 |
| control_ee 的 check_motion / execute_motion / 阈值 | simulation.ee_control；CLI、打印、Viewer 仍留在脚本 |
| scripts/realman_fk.py、realman_ik.py | vendor 同名模块；不留旧导入转发 |
| 默认13段关节实验、共同FK姿态、SE(3)及关节报告格式 | experiments.common；打印回调仍由脚本提供，保留 JSON 的 joint_sequence.log |
| view_robot 的 passive_viewer / 分散的后端选择 | viewer；只供脚本调用，按需导入 mujoco.viewer |

静态URDF检查与模型生成仍是独立工具，不合并进动力学控制器。生成器只改用统一资源路径，导出算法、参数和注释不变，输出必须与原MJCF逐字节一致。

`require` 在 model 中保留原 ValueError 行为，control 导入同一函数；没有增加新的验证规则。核心 `__init__.py` 不做便利性的大量重导出，也不实例化模型、Kinematics 或 Algo。

## 依赖与数据流

```mermaid
flowchart TD
    CLI[脚本：参数、实验、打印] --> EE[simulation.ee_control]
    CLI --> K[simulation.kinematics]
    CLI --> C[simulation.control]
    CLI --> M[simulation.model]
    CLI --> E[experiments.common]
    CLI --> V[viewer：惰性GUI辅助]
    CLI --> VI[vendor.realman_ik]
    CLI --> VF[vendor.realman_fk]
    E --> C
    EE --> C
    EE --> K
    C --> M
    M --> P[paths]
    VI --> VF
    VI --> K
    VF -. 仅构造时 .-> SDK[可选 Robotic_Arm Algo]
```

核心 simulation 模块不依赖 scripts、experiments、viewer 或 vendor。厂家 IK 沿用 MuJoCo 的四元数辅助和既有旋转误差函数，因此是 vendor 单向使用 simulation；不会由 MuJoCo 控制自动调用厂家 IK。

关节控制流程：`model.load_model → model.initialize → control.validate_target → control.run_segment → ctrl → mj_step → 原有数值/限位/接触检查 → 统计`。`step_observer` 在 mj_step 后、任何 tick 的 mj_forward 前调用，确保观察的是该物理步使用的合力矩。

末端控制流程：`Kinematics.solve_ik → ee_control.check_motion → 原有51点碰撞预检 → execute_motion → control.run_segment → 执行器运动 → 实际末端误差`。IK/FK 使用独立数据；正常运动不赋值实际 qpos。

厂家对照是另一条纯计算流程：`厂家FK目标 → 厂家IK → 厂家FK复核`。RealManIK 继承唯一的 RealManFK 实现，共用进程全局活跃会话保护；配置只能顺序使用。不能混用绕过适配器直接修改原生全局状态的SDK代码。失败仍是 q=None，不自动尝试其他求解器或发送执行器指令。

## 导入与未来环境

先在现有 realman_rl 环境完成 editable 安装。新代码使用：

```python
from realman_rl_simtoreal.simulation.model import load_model, initialize
from realman_rl_simtoreal.simulation.kinematics import Kinematics, IKResult
from realman_rl_simtoreal.simulation.control import validate_target, run_segment, ForceMonitor
from realman_rl_simtoreal.simulation.ee_control import check_motion, execute_motion

model, joint_ids, actuator_ids = load_model(gravity_compensation=False)
data = initialize(model)
kin = Kinematics(model)
```

旧的 `from kinematics ...` 或从 control_joints 导入核心函数不再是支持接口。八个原脚本路径、CLI 参数和退出码继续保留。测试核心时直接导入包；需要测试脚本实验函数时，用标准库 importlib 按文件加载，不修改 sys.path。生产代码不使用这种测试加载方式。

03-A的 `envs.reach_env.ReachEnv` 直接从上述模块获取模型、运动学、目标验证和碰撞预检，不依赖脚本、vendor、experiments或Viewer。`run_segment`仍保留原有统计和收敛检查；环境改用新增的 `control.advance_control_cycle`，不会在step内等待稳态或调用旧演示序列。±5°、10mm和固定home目标朝向等局部实验边界继续保留。

环境数据流：归一化世界XYZ动作 → 独立命令位置 → 以最近合法关节命令为初值的原DLS IK → 限位/位姿复核及51点碰撞预检 → 指令变化率限制与平滑 → 固定物理步 → 总力矩/数值/速度/碰撞审计 → 新状态下的观测和奖励。正常step不写实际qpos；reset通过官方keyframe重置状态。每个物理步先审计驱动力，再刷新运动学缓存，顺序与原控制要求相同。

环境支持 `reset(seed=..., options=None)`、`step`、Dict观测、`compute_reward`、文本 `render` 及幂等 `close`。当前不自动注册Gymnasium ID，直接导入类实例化；未将训练器嵌入环境；独立04-A训练入口见后文。完整形状、失败语义和控制参数见 [03-A验证](validation/REACH_ENV_VALIDATION.md)。

03-B仅扩展envs与实验脚本，没有修改simulation核心或vendor。`target_mode="fixed"`和原米制`distance`奖励继续兼容；`target_mode="random"`在reset中通过np_random采样候选，用原IK、限位和路径预检接受目标，最多尝试64次，失败关闭当前episode并报错。默认候选球壳为home周围3–5mm；实际分布是通过检查后的条件分布，不代表完整工作空间。

随机模式默认`safe_dense`奖励：归一化截断距离成本加独立的安全失败惩罚；step把含`is_safety_failure`的info传给compute_reward。可标量或批量重算，不读取原目标的成功标记。固定目标也可显式选safe_dense用于训练实验。命令边界默认仍终止，04-B增加下述显式拒绝模式；模型及物理安全阈值不变。

规则策略留在评估脚本，只读取27维观测中的位置/目标/命令参考，经env.step驱动；不访问真实关节状态或调用IK绕过环境。评估与独立cProfile记录属于实验层，不向环境info加入非确定性的墙钟计时。任务分布、奖励推导、性能以及未来HER需要重算的字段见 [03-B验证](validation/REACH_TASK_VALIDATION.md)。

## 安装与路径边界

03-C1保持ReachEnv默认6轴场景不变，新增独立资源路径GRIPPER_SCENE。model.load_model按六个命名hinge和执行器建立映射，允许附件坐标；initialize仍使用模型home keyframe。编译前重力补偿遍历link_1后代（包括新载荷），只路由机器人joint，不影响无关场景body。Kinematics仍接收/返回六个机械臂角，Jacobian按六个dof列切片，使用独立MjData；碰撞预检保留输入数据中的附件qpos。没有把夹爪关节加入Reach动作。

Gripper校验6个夹爪hinge、5条激活mimic和一个eg2_position，set_width/open/close只选目标，step只改该执行器ctrl并推进共享物理。实际开口由独立数据中的pad sites读取，不改实时状态/力缓存。机械臂控制仍只写原6个ctrl；两种独立推进函数各自都会过物理时间；03-C2新增的同步周期见下文，仍不新增Pick或完整控制抽象。

ee_site是原link_6原点；实验gripper_mount位于其+Z8mm；grasp_site是mount+Z147.530235mm的固定名义参考，真实指垫中心随开口运动。该装配与真实六维力/相机板栈、TCP和传动动力学尚未校准。转换、接触和限制见 [03-C1验证](validation/GRIPPER_VALIDATION.md)。

pyproject 使用 setuptools 的 src 包发现；基础和 vendor 可选依赖分别动态读取两份原锁定文件。默认安装不要求 Robotic_Arm；任何包初始化都不强制导入它，只有厂家适配器构造时才尝试加载。

本次验证使用 `python -I -m pip install --no-build-isolation -e .`，复用环境中已有构建工具，不升级依赖。通常也可用标准 `python -I -m pip install -e .` 让pip管理构建隔离。无需 PYTHONPATH 或脚本内路径补丁；`-I` 下通过 Conda 环境 site-packages 中的 editable 注册定位源码。

paths 从自身文件位置定位仓库 assets/outputs，不按终端目录搜索。**只支持保留源码仓库的 editable 使用方式，不承诺普通wheel脱离仓库后的资源加载。** 搬移仓库后应重新 editable 安装。

Viewer 的 X11 默认选择在 CLI 导入 MuJoCo/GLFW 前执行，显式 PYGLFW_LIBRARY_VARIANT 优先。核心导入不选择GUI后端；Viewer打开、相机、同步循环和提前退出语义由各脚本保留，退出时继续等待本次Viewer线程清理。

## 单位与保留限制

| 数据 | 约定 |
|---|---|
| 包内关节角、ctrl、IK初值/解 | rad，joint_1..joint_6 顺序 |
| SDK原生关节角 | degree，由厂家适配器转换 |
| 位置、平移 | m；CLI delta-mm和报告mm字段显式转换 |
| 旋转矩阵 | 3×3主动旋转，末端向world/名义base；误差为SO(3)最短旋转向量/角 |
| SDK四元数 | 单位 wxyz；不使用欧拉角差作通用姿态误差 |
| 力矩、质量、惯量、时间 | N·m、kg、kg·m²、s；报告字段标注角度/速度单位 |

ee_site 仍是 link_6 原点，不是标定TCP。型号代际、传感器对应、原始惯量异常和模型差异继续保留。厂家IK边界失败及MuJoCo远初值失败没有被修复或隐藏。历史报告原文移动到 validation，历史代码路径描述当时的结构，当前入口以本文和README为准。


## 03-C2 接触实验与同步物理周期

`paths.GRASP_CONTACT_SCENE`只供独立接触实验使用。原ReachEnv、GRIPPER_SCENE及厂家模块保持原模型和接口。新增freejoint的7个qpos/6个qvel不能被六轴控制覆盖；模型加载/补偿仍按命名关节和link_1子树识别机器人，目标物体不在补偿子树。

`Gripper.update_control()`只更新夹爪ctrl，`observe_step()`只审计刚完成的物理步；原`Gripper.step()`组合两者继续独立运行。`control.advance_synchronized_cycle(m,d,jids,aids,target,gripper,physics_steps=25)`验证共享model/data、原映射、±5°目标与每周期0.5°指令变化上限。循环顺序为：平滑更新六轴ctrl → 更新夹爪ctrl（原0.5rad/s上限）→ **一次mj_step** → 原ForceMonitor及夹爪总驱动力/状态审计 → 可选接触记录器 → mj_forward → 再次状态检查。没有live qpos赋值，没有多跑一个夹爪积分周期。

`Gripper(contact_policy=None)`保留空载默认0.1mm碰撞阈值。独立场景显式传入`grasp_contact.ContactPolicy`，只对按geom/body识别的目标—左右指垫、目标—支撑台设置各自允许数值穿透；其余接触保持原阈值。机械臂/夹爪限位、速度、mimic、总驱动力及外力注入检查均保留。夹爪五个无伺服从动joint也检查补偿路由和总驱动力限幅。

`ContactRecorder`在刷新实时力缓存前调用mj_contactForce，每2ms记录标量，20Hz及新接触类别首次出现时保存接触位置/坐标系力/世界系物体受力。姿态及当前指垫位置使用独立MjData；几何凹槽反例也只是独立初始化的forward诊断。该模块不导入脚本、厂家SDK或环境。

实验脚本负责时序与Viewer：稳定→6秒慢闭合→台面保持→沿支撑法向5mm抬升→2秒离台保持；释放案例先降到名义1mm高度再慢开。运动仍由现有执行器产生。所有案例、负对照和失败原因详见validation/GRASP_CONTACT_VALIDATION.md，未引入抓取环境或训练逻辑。

## 04-A 可选SAC实验层

新增`experiments.sac_reach`只由训练/评估脚本和对应测试导入，依赖官方SB3/PyTorch，不被envs、simulation、vendor或轻量包入口导入。`pyproject.toml`的rl可选组读取requirements-rl.txt；基础依赖和vendor锁定文件保持原样。CPU轮子通过官方CPU索引安装，不隐式升级现有物理依赖。

`ReachTransform`与`ScaledReachObservation`只对27维observation做固定仿射缩放（home中心、5°/30°每秒/10mm量纲）；两个goal键保留SI米制及float64，wrapper.compute_reward仅将这两个仍为SI的goal传给原环境。SB3 CombinedExtractor把Dict展开为33维float32输入，原奖励完全保留。所有偏移和尺度存入schema=2配置，不拟合评估目标或更新运行统计。未来HER仍需要同步重标记相对目标字段，本次没有实现HER。

原规则策略rule_action原样移入experiments.common，旧evaluate_reach_baselines入口继续导入同一函数，便于与新SAC评估公平复用。没有将实验策略引入控制核心。

训练数据流：原ReachEnv.reset/step → Monitor/固定观测变换/VecCheckNan/DummyVecEnv → 官方SAC MultiInputPolicy → 官方DictReplayBuffer和优化器。动作不变换，所有运动仍经环境IK、原控制及mj_step；没有训练脚本直接写qpos、调用厂家SDK或访问夹爪执行器。

TrainingAudit只读取原info、奖励、官方logger和网络；分开记录成功、安全失败与超时，并检查有限数值/明显发散。Q诊断对官方buffer采样后恢复NumPy RNG，确定性Actor不消耗Torch随机状态；验证环境独立，目标种子10000–10009，不污染训练环境和Replay Buffer。on_rollout_end诊断在当前更新前，checkpoint诊断在learn返回后；日志含更新次数以区分。

scripts/train_sac_reach.py每5,000步返回一次learn、检查并独立评估，随后同一进程reset_num_timesteps=False继续；最终存储完整网络、官方Replay Buffer压缩文件、配置、原始日志及版本快照。每次CLI调用从头开始，Pilot与正式实验不合并计数，不实现不完整的跨进程续训。

scripts/evaluate_sac_reach.py独立加载模型和保存的transform；原SI环境只接收合法动作。固定种子100–199在各策略下重置，验证物理目标逐项一致，输出逐Episode和汇总指标，绝不把API通过或训练回报上升当作策略成功。

## 04-B 可选边界模式与冻结诊断

`ReachEnv(boundary_mode="terminate")`保留原默认step/info/reward；改动前的fixture逐值验证。`reject_and_hold`仅将任务级10mm命令越界、模型限位内但超home ±5°的候选关节命令转为拒绝。保留Cartesian命令和最近已发出的合法joint ctrl，复用同一个`advance_control_cycle`推进25步；不修改simulation、IK或模型，也不裁剪候选解。实际关节越限、碰撞、数值、速度、力矩及姿态保护保持终止。拒绝本身不等于物理故障，两者可同时记录。

此模式新增info字段`is_action_rejected`、`rejection_reason`和`boundary_mode`，普通safe_dense基底不变，额外扣`rejection_penalty * is_action_rejected`（默认1）。`compute_reward`要求提供拒绝布尔标记，支持标量和批量；旧distance模式的相同成本单位为0.020m。拒绝计入Episode步数、真实物理时间和超时，不存在无限免费尝试。HER尚未实现，不能仅替换goal就假定整个观测、奖励和终止逻辑已支持重标记。

`experiments.sac_diagnostics`依赖可选SB3，不被envs/simulation导入。轨迹区分`desired_goal-achieved_goal`和`desired_goal-command_position`；使用独立FK读取ctrl对应位置。soft TD按安装版本计算`reward + gamma*(1-done*(1-timeout))*(min(target_Q)-alpha*log_pi)`，用Replay存储动作查询当前Critic。诊断在no_grad与隔离Torch RNG内运行，核验参数/训练计数不变，不把无熵任务回报当作soft Q的精确监督。记录完整数组，成功/物理故障/拒绝/timeout分别统计。

`compare_sac_boundary.py`复用原train入口，从空buffer和原初始化seed训练B；冻结A的10kcheckpoint及未环回buffer前缀只用于评估/诊断。预先保存开发集、预算与复验规则，核验SAC参数、固定观测变换和目标逐项一致。`--final --b ...`只评估已保存模型，不训练。所有新输出写入独立空目录，04-A保持不变。当前开发集未改善成功率，未启动第二seed；最终0%/1%仍不构成可用策略，证据与假设见[04-B验证](validation/SAC_DIAGNOSTICS_VALIDATION.md)。

## 04-C 独立BC实验层

`experiments.bc_reach`只由三个BC脚本及测试导入；不改envs、simulation、vendor、common、sac_reach或历史SAC模型。复用已有可选CPU依赖和`ReachTransform`，不引入第二套预处理。数据流为公开Dict → 原rule_action → env.step → NPZ；每个样本保存步前状态/动作及步后结果，按整个Episode分离监督训练和验证。

网络输入按SB3 CombinedExtractor顺序拼接`achieved_goal(3), desired_goal(3), observation(27)`并转float32。`BCNetwork`的latent_pi和mu对齐SAC确定性均值通路，输出tanh动作；省略log_std及整个RL学习系统。`train_bc`只在NPZ上最小化动作MSE，不构造环境、调用专家、优化Critic或写Replay。模型保存显式BC标识、网络权重、训练参数、环境及固定预处理；推理无需训练数据或专家。

`evaluate_bc_reach.py`加载训练前和最优验证MSE的两个冻结网络，与原规则策略在20000–20099配对目标上分别执行。两个神经分支只调用自己的predict和标准env.step，没有规则兜底、IK标签或实际qpos赋值。15个新测试包含SB3特征/确定性输出逐元素核验、模型更新/加载一致性、无SDK及物理步进检查；成功率作为实验数据，不作为单元测试硬性要求。局部目标分布及无停止样本的覆盖边界见[04-C验证](validation/BC_REACH_VALIDATION.md)。

## 04-D1 Actor初始化和监控

`sac_initialization`单向依赖bc_reach/sac_reach，仅用于全新SAC的Actor初始化。校验feature顺序/空间、CombinedExtractor、ReLU/Linear及SquashedGaussian/tanh，不只比shape；复制latent_pi与mu，log_std、双Critic、Target和温度均作指纹保护，Replay及优化器为空。冻结BC对象无梯度、与Actor不共享参数；缺失/不兼容明确失败。默认scratch不加载BC，scratch显式传bc-model时仅加载冻结参照。

训练入口保留官方SAC_CONFIG、采样、优化器和Replay；新增eval/checkpoint两套间隔，按最近事件在同一进程继续learn。bc_actor强制第0步评估/保存；原scratch缺省不加第0步评估，默认完整网络张量已与原入口128步结果精确比较。原evaluate仅追加可选reference_policy，用实际访问状态比较动作但不修正输出。

`MonitoredTrainingAudit`继承原安全审计，记录训练Episode窗口统计、拒绝率和固定探针偏差；阶段评估使用独立环境及固定seed，BC监督种子重叠会拒绝。当前/Target Q和timeout-aware TD诊断只读、隔离RNG，不改变模型学习。TB写入SB3已有logger（显式开关），CSV/JSON继续保留；训练成功率与独立评估成功率分别命名，不互换。

`plot_sac_training`只使用已保存的JSON，不依赖环境/SDK/模型，输出PNG及原始绘图序列；无测量则skip。只在rl可选依赖中增加TensorBoard/Matplotlib；基础模型和包入口仍不依赖它们。04-D1仅短测试，已记录BC经更新后部分目标退化，没有启动正式长训练。命令、模型指纹、监控标签、回归与限制见[04-D1验证](validation/SAC_BC_INIT_VALIDATION.md)。
