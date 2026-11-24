# Python 文件地图

现行 Python 文件共 158 个，包含核心包 87、脚本 39、测试和手动工具 32。

模块依赖见 [架构](architecture.md)。实际目录树、迁移前后文件和回归结果见 [src 重构报告](validation/SRC_LAYOUT_REFACTOR_VALIDATION.md)。实验原版本的命令与路径见 [历史验证索引](validation/README.md)。

## scripts/demos

- `scripts/demos/calibrate_cube_hsv.py` — 选择实物相机，对照实时掩膜调整 HSV 阈值并保存 JSON。
- `scripts/demos/control_ee.py` — 末端 IK / 控制演示。
- `scripts/demos/control_joints.py` — 关节控制演示与验收。
- `scripts/demos/demo_gripper.py` — 夹爪开合、带载保持演示。
- `scripts/demos/demo_vision_grasp.py` — 解析交互视觉演示参数，调用 apps.visualization.vision_grasp。
- `scripts/demos/detect_tabletop_cube.py` — Capture existing wrist RGB-D, HSV mask and a cube position in robot base。
- `scripts/demos/move_to_observation_pose.py` — setup the Cartesian tabletop task in the existing realman_rl project。
- `scripts/demos/render_ppo_tabletop.py` — 解析关节桌面回放与输出路径，调用 apps.visualization.tabletop_joint。
- `scripts/demos/render_ppo_tabletop_cartesian.py` — render the Cartesian tabletop task in the existing realman_rl project。
- `scripts/demos/run_grasp_contact_cases.py` — 接触、抬升、释放和失败对照案例。
- `scripts/demos/run_grasp_lift.py` — Approach, close, lift and hold a dynamic cube through four-action controls。
- `scripts/demos/view_robot.py` — 静态机器人模型 Viewer。

## scripts/diagnostics

- `scripts/diagnostics/compare_controllers.py` — PD 与重力补偿对照。
- `scripts/diagnostics/compare_fk.py` — 厂家 / MuJoCo FK 对照。
- `scripts/diagnostics/compare_ik.py` — 厂家离线 IK 自模型与跨模型验证。
- `scripts/diagnostics/diagnose_reach_rewards.py` — 独立Home动作方向、旧奖励解析、同轨迹新旧计分与四张诊断图；不训练。
- `scripts/diagnostics/probe_ppo_goals.py` — 冻结PPO在合法reset目标下的同状态特征/动作敏感性，不执行step或训练。
- `scripts/diagnostics/probe_reach_env.py` — 任务分布、奖励与随机探索探针，无学习器。
- `scripts/diagnostics/smoke_reach_env.py` — 小规模无界面 Reach 交互 / Gymnasium checker。

## scripts/evaluate

- `scripts/evaluate/analyze_reach_results.py` — 只读评估JSON；逐候选、方向、象限、半径组及动作偏置/裁剪角度，不加载模型。
- `scripts/evaluate/evaluate_bc_reach.py` — BC 模型加载与配对评估 CLI，经 BC 权重保护薄适配调用 Evaluation。
- `scripts/evaluate/evaluate_ppo_grasp_lift.py` — Independent evaluation of four-action PPO, with no staged-controller fallback。
- `scripts/evaluate/evaluate_ppo_reach.py` — 加载PPO及保存变换；离散逐目标/随机独立目标评估，重复数拒绝。
- `scripts/evaluate/evaluate_ppo_tabletop.py` — Read-only PPO evaluation from the saved pose, or the full tabletop task。
- `scripts/evaluate/evaluate_ppo_tabletop_cartesian.py` — evaluate the Cartesian tabletop task in the existing realman_rl project。
- `scripts/evaluate/evaluate_reach_baselines.py` — 随机/规则评估 CLI；计算与循环直接导入 evaluation.baselines，保留参数/JSON/退出码。

## scripts/models

- `scripts/models/build_dynamics_model.py` — 重建六轴动力学 MJCF。
- `scripts/models/build_grasp_scene.py` — 重建接触代理和动态方块场景。
- `scripts/models/build_gripper_model.py` — 重建独立 / 组合夹爪模型。
- `scripts/models/build_tabletop_cartesian_scene.py` — Derive the Cartesian task scene; additions are visualization frames only。
- `scripts/models/build_tabletop_scene.py` — Horizontal workbench with the fixed arm and a freely supported cube。
- `scripts/models/inspect_model.py` — URDF / MJCF、mesh、惯量和关节检查。

## scripts/plots

- `scripts/plots/plot_ppo_training.py` — 只读PPO JSON；聚合及逐目标成功/距离/首动作、训练成功率/终止PNG。

## scripts/train

- `scripts/train/collect_reach_demonstrations.py` — 公开观测规则策略示范数据采集。
- `scripts/train/train_bc_reach.py` — 离线 BC MSE / Adam 训练入口。
- `scripts/train/train_ppo_grasp_lift.py` — 解析局部夹取课程和迁移参数，调用 apps.training.grasp_lift。
- `scripts/train/train_ppo_reach.py` — 解析 Reach 训练参数并调用 apps.training.reach。
- `scripts/train/train_ppo_tabletop.py` — 解析七维关节桌面 PPO 参数，调用 apps.training.tabletop_joint。
- `scripts/train/train_ppo_tabletop_cartesian.py` — 转发笛卡尔桌面训练参数给 apps.cli.tabletop_cartesian。

## src / realman_rl_simtoreal

- `__init__.py` — 轻量包初始化。
- `paths.py` — 源码仓库资源与输出目录定位，不依赖 cwd。

## src / apps

- `apps/__init__.py` — 轻量包初始化。

## src / apps/cli

- `apps/cli/__init__.py` — 轻量包初始化。
- `apps/cli/tabletop_cartesian.py` — 共享笛卡尔训练、评估、启动和回放命令的参数与调度。

## src / apps/training

- `apps/training/__init__.py` — 轻量包初始化。
- `apps/training/grasp_lift.py` — 运行局部夹取 PPO 的课程任务、权重续训、评估和 checkpoint 调度。
- `apps/training/reach.py` — 组装 Reach 环境与 PPO，按完整 rollout 更新、独立评估并保存实验产物。
- `apps/training/tabletop_cartesian.py` — 运行笛卡尔桌面 PPO、Reach 权重热启动与 RGB-D 迁移训练。
- `apps/training/tabletop_joint.py` — 训练七动作关节桌面 PPO，管理课程、探索参数继承和独立评估。

## src / apps/visualization

- `apps/visualization/__init__.py` — 轻量包初始化。
- `apps/visualization/tabletop_cartesian.py` — Actual MuJoCo replay: overview, top view, and the wrist camera。
- `apps/visualization/tabletop_joint.py` — 回放关节桌面 PPO，生成全景与夹爪近景 GIF、PNG 和评估记录。
- `apps/visualization/viewer.py` — 延迟初始化 MuJoCo Viewer，统一演示程序的渲染后端和线程收尾。
- `apps/visualization/vision_grasp.py` — 在 MuJoCo Viewer 中运行视觉抓取并叠加腕部 RGB-D 检测结果。

## src / deployment

- `deployment/__init__.py` — 轻量包初始化。

## src / envs

- `envs/__init__.py` — 轻量包初始化。
- `envs/factory.py` — 将保存的任务配置分派给对应的 Gymnasium 环境，供训练与独立评估共用。

## src / envs/grasp

- `envs/grasp/__init__.py` — 轻量包初始化。
- `envs/grasp/grasp_lift_env.py` — 局部夹取与抬升的 MuJoCo 环境，提供四维末端动作、接触验收和 41 维观测。
- `envs/grasp/tabletop_cartesian_env.py` — 真值观测的桌面笛卡尔抓取环境，调用运行时执行物理步并计算任务完成状态。
- `envs/grasp/tabletop_cartesian_vision_env.py` — 使用腕部 RGB-D 和机器人本体状态构造桌面策略观测，并保留独立的物理验收。
- `envs/grasp/tabletop_pick_env.py` — 关节空间桌面抓取环境，复用局部抬升任务的接触检查与物理保持判据。

## src / envs/reach

- `envs/reach/__init__.py` — 轻量包初始化。
- `envs/reach/reach_env.py` — 用既有运动学和位置执行器运行固定、离散或可达随机目标的 MuJoCo Reach 任务。
- `envs/reach/sequence_reach_env.py` — 在同一物理回合内串联多个 Reach 目标，并记录各目标的完成情况。

## src / evaluation

- `evaluation/__init__.py` — 轻量包初始化。
- `evaluation/baselines.py` — 原公开观测规则策略；早期随机/规则评估、奖励反例与独立性能剖析。
- `evaluation/grasp_lift.py` — 运行局部夹取和抬升回合，生成独立的物理验收与实验来源记录。
- `evaluation/metrics.py` — Reach / GraspLift / BC / 早期基线的原公式、成功率分母、延迟与目标分布统计。
- `evaluation/reach.py` — 独立 Reach 回合、原 RL / BC JSON 协议和原始观测 callable 接口；不加载或训练算法。
- `evaluation/tabletop_cartesian.py` — 真值与视觉桌面抓取共用的评估循环，接收调用者提供的动作预测与状态指纹。

## src / hardware

- `hardware/__init__.py` — 轻量包初始化。

## src / imitation

- `imitation/__init__.py` — 轻量包初始化。
- `imitation/bc.py` — 原 BC 网络、数据采集和监督训练；闭环评估仅保留权重保护薄适配。

## src / perception

- `perception/__init__.py` — 轻量包初始化。
- `perception/rgbd.py` — 根据对齐的 RGB-D 图像、相机内参和外参估计橙色立方体的位置。

## src / rl

- `rl/__init__.py` — 轻量包初始化。
- `rl/cartesian_policy.py` — 定义笛卡尔 PPO 的连续高斯动作分布，按抓取观测调整各通道探索幅度。
- `rl/cartesian_reach_policy.py` — 为笛卡尔桌面 PPO 提供目标特征、可迁移的 Reach 臂分支和独立夹爪、价值网络。
- `rl/ppo.py` — 通过 SB3 创建 Reach PPO，核验 rollout、模型状态、保存契约和只读评估。
- `rl/ppo_cartesian.py` — 笛卡尔 PPO 模型创建、checkpoint 契约和共享评估薄适配。
- `rl/ppo_grasp.py` — 局部夹取与抬升 PPO 的模型参数、checkpoint、训练回调和评估适配。
- `rl/ppo_tabletop.py` — 关节桌面 PPO 的策略、持续相关噪声、checkpoint 操作及评估适配。
- `rl/vision_policy.py` — 用视觉状态替换目标对象特征，保留 Reach 臂和夹爪 PPO 的网络与迁移接口。

## src / rl/common

- `rl/common/__init__.py` — 轻量包初始化。
- `rl/common/artifacts.py` — 提供模型参数和优化器指纹、Reach 来源摘要，并转发历史文件操作入口。
- `rl/common/env_factory.py` — 构造 Reach 与抓取的 SB3 环境包装栈，复用原 Monitor、向量化与数值检查。
- `rl/common/logging.py` — 将独立评估已产生的指标写入训练日志，沿用原 TensorBoard 标签和分母。
- `rl/common/observation.py` — 定义 Reach 固定观测编码和历史 checkpoint 验证，供 PPO、BC 与评估共用。

## src / simulation

- `simulation/__init__.py` — 轻量包初始化。

## src / simulation/devices

- `simulation/devices/__init__.py` — 轻量包初始化。
- `simulation/devices/camera_geometry.py` — 定义 MuJoCo 相机朝向的几何约定，供桌面运行时设置腕部相机姿态。
- `simulation/devices/mujoco_gripper.py` — 将 EG2-4C2 MuJoCo 模型的开口目标换算为六连杆机构的主关节指令并检查执行状态。
- `simulation/devices/mujoco_rgbd_camera.py` — 从 MuJoCo 腕部相机采样 RGB 和米制深度，生成感知模块共用的 RgbdFrame。

## src / simulation/motion

- `simulation/motion/__init__.py` — 轻量包初始化。
- `simulation/motion/cartesian_controller.py` — 将桌面抓取环境给出的 TCP 增量转为受工作空间、关节速率和碰撞约束的关节参考值。
- `simulation/motion/kinematics.py` — 基于 MuJoCo site 的法兰正运动学、雅可比和阻尼最小二乘逆运动学。
- `simulation/motion/staged_grasp_controller.py` — 局部夹取与抬升的传统分段控制器，供演示、抓取评估基线和物理对照使用。
- `simulation/motion/tool_kinematics.py` — 为抓取 TCP 及腕部附件提供显式工具坐标系，复用法兰 FK/IK 的数值实现。
- `simulation/motion/traditional_ee_control.py` — 固定法兰朝向的末端控制演示与对照实验，组织 IK 预检、插值和稳定性验收。

## src / simulation/physics

- `simulation/physics/__init__.py` — 轻量包初始化。
- `simulation/physics/grasp_contact.py` — 按具名几何体分类抓取接触并记录物理步数据。
- `simulation/physics/grasp_metrics.py` — 从真实接触中提取沿夹持方向的压力，用于桌面任务的双侧夹持和支撑判定。
- `simulation/physics/joint_control.py` — 推进 MuJoCo 关节位置伺服并核验每个物理步的执行状态。
- `simulation/physics/model.py` — 加载并校验固定基座的 RML63 MuJoCo 模型，为执行器和运动学提供模型与关节索引。

## src / simulation/runtime

- `simulation/runtime/__init__.py` — 轻量包初始化。
- `simulation/runtime/observation_presets.py` — 保存固定观察位关节目标和 TCP 姿态，供传统启动及 checkpoint 契约校验。
- `simulation/runtime/observation_startup.py` — 在策略接管前，按固定关节轨迹把机械臂从六轴零位移到指定观察位。
- `simulation/runtime/tabletop.py` — 组合桌面 MuJoCo 模型、工具运动学、位置伺服、夹爪和接触测量，为桌面环境提供机器人接口。

## src / tasks

- `tasks/__init__.py` — 轻量包初始化。

## src / tasks/grasp

- `tasks/grasp/__init__.py` — 轻量包初始化。
- `tasks/grasp/lift.py` — 局部夹取抬升的场景、课程参数和按折扣势差计算的奖励。
- `tasks/grasp/tabletop_cartesian.py` — 定义桌面笛卡尔抓取的观察位、动作与传感器参数，以及课程完成门槛。
- `tasks/grasp/tabletop_cartesian_reward.py` — 计算笛卡尔桌面的接近、围拢、夹持、抬升等奖励项，评估诊断可复用同一计算。
- `tasks/grasp/tabletop_joint.py` — 关节空间桌面抓取的任务、课程和奖励定义，含接近子目标与夹爪开口正则。

## src / tasks/reach

- `tasks/reach/__init__.py` — 轻量包初始化。
- `tasks/reach/config.py` — 定义 Reach 目标分布、SI 动作与时间尺度，以及可保存的环境参数。
- `tasks/reach/reward.py` — 定义 Reach 各奖励版本、进度参数和 checkpoint 一致性检查。

## src / utils

- `utils/__init__.py` — 轻量包初始化。
- `utils/artifacts.py` — 提供 JSON、JSONL、摘要和源码归档，具体实验决定字段和来源范围。
- `utils/demo_helpers.py` — 供演示与对照脚本共用的关节样本、执行序列与结果格式。
- `utils/geometry.py` — 共享 SE(3) 构造与逆变换。
- `utils/metrics.py` — 训练与评估共享 MotionStats 和原余弦函数；只依赖 NumPy。

## src / vendor

- `vendor/__init__.py` — 轻量包初始化。
- `vendor/realman_fk.py` — 可选厂家离线 FK SDK 适配与会话保护。
- `vendor/realman_ik.py` — 厂家离线 IK、解集与自模型复核。

## tests

- `tests/manual_structure_regression.py` — 手动执行真值和视觉回合的逐物理步、策略步及帧摘要对照。
- `tests/test_bc_reach.py` — BC 数据、特征、模型、监督更新与闭环。
- `tests/test_cartesian_learning.py` — Regress the failed full-task reward ranking and PPO contact precision。
- `tests/test_evaluation.py` — 17 项评估接口/指标/独立环境/参数保护/无算法导入及 JSON 协议测试。
- `tests/test_fetch_observation.py` — 双模式维度、真实Site运动、网络实际输入、物理一致性、旧checkpoint及单512步PPO Smoke。
- `tests/test_fk_comparison.py` — FK 坐标、厂家配置与差异对照。
- `tests/test_goal_conditioned_reach.py` — 离散候选、旧模式精确回归、一次两方向PPO更新及逐目标评估/监控。
- `tests/test_grasp_contact.py` — 接触、代理、同步步进与严格失败反例。
- `tests/test_grasp_lift.py` — 实际抓取接口、物理循环、阶段控制器、失败对照与指标。
- `tests/test_gravity_compensation.py` — 重力补偿路由、合力矩限幅及控制对照。
- `tests/test_gripper.py` — 夹爪结构、mimic、开口、模型重生成和演示。
- `tests/test_ik_comparison.py` — 厂家 IK 自模型 / 边界与跨模型。
- `tests/test_joint_control.py` — 动力学结构、保持、关节运动、力矩与碰撞。
- `tests/test_kinematics.py` — FK / Jacobian / DLS IK / 末端控制与资源重生成。
- `tests/test_package.py` — 包发现、无循环 / 分层、全部 argparse / shared CLI help、退出码与 Viewer 生命周期。
- `tests/test_ppo_grasp.py` — 四维 PPO 实际更新、课程、奖励、checkpoint 和只读策略评估。
- `tests/test_ppo_random_reach.py` — 200个随机目标/观测/reset契约、分组与裁剪分析、单512步随机PPO Smoke及20目标评估接口。
- `tests/test_ppo_reach.py` — 单次完整 PPO Rollout 更新、动作与损失、日志、checkpoint；阻断 SAC / BC 导入。
- `tests/test_pure_reach_training.py` — 纯 Reach 任务分布、折扣奖励、SI 约定和运动统计。
- `tests/test_reach_boundaries.py` — Reach 拒绝保持、终止、故障和历史固定种子轨迹。
- `tests/test_reach_env.py` — Gymnasium、动作、观测、奖励、真实物理步与保护。
- `tests/test_reach_progress_reward.py` — 25项奖励方向、批量、漏洞/边界、兼容、组件与单次512步PPO Smoke。
- `tests/test_reach_task.py` — 随机可达目标、奖励一致性、基线及 CLI。
- `tests/test_reach_transform.py` — Reach 原始与包装观测空间、固定编码和模型输入校验。
- `tests/test_refactor_contracts.py` — JSON、来源、现行 CLI、退役模块、策略类、物理方法、奖励及七类历史模型的精确回归。
- `tests/test_rgbd_vision.py` — Camera geometry, true-state isolation, lost-target handling and visual PPO。
- `tests/test_sequence_reach.py` — 随机起点、连续目标、奖励上下文与状态交接。
- `tests/test_tabletop_cartesian.py` — Exercise the policy/controller boundary and real MuJoCo dynamics。
- `tests/test_tabletop_evaluation.py` — Algorithm-free tabletop episodes, ownership and read-only state guards。
- `tests/test_tabletop_pick.py` — Horizontal tabletop geometry, exact zero reset, real PPO and read-only inference。
- `tests/test_tabletop_runtime.py` — Read-only state facade and unchanged latest-runtime physical clock。
- `tests/test_wide_reach.py` — 20 cm 目标分布、原物理控制和 PPO 迁移契约。
