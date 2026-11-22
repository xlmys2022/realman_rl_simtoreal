# Python 文件地图

当前共 152 个 Python 文件：核心包 75、入口 44、测试 33。下列清单来自实际文件与静态导入；函数内导入纳入统计，动态加载和 checkpoint 的类引用另外由测试验证。

当前边界见 [architecture.md](architecture.md)，本轮的基线、确认重复与阶段验证见 [结构审计](validation/STRUCTURE_AUDIT.md)。[此前 Evaluation 报告](validation/EVALUATION_REFACTOR_VALIDATION.md)和其他历史验证报告描述当时版本，不能代替当前地图。

包文件均以 `src/realman_rl_simtoreal/` 为根；scripts 和 tests 路径以仓库根为准。新增、移动文件后更新此清单。

## scripts/demos

- `scripts/demos/calibrate_cube_hsv.py`：选择实物相机，对照实时掩膜调整 HSV 阈值并保存 JSON。
- `scripts/demos/control_ee.py`：末端 IK / 控制演示；入口：用户 CLI。
- `scripts/demos/control_joints.py`：关节控制演示与验收；入口：用户 CLI。
- `scripts/demos/demo_gripper.py`：夹爪开合、带载保持演示；入口：用户 CLI。
- `scripts/demos/demo_vision_grasp.py`：MuJoCo 交互窗口运行视觉 PPO，显示腕部 RGB 和 HSV Mask；入口：用户 CLI。
- `scripts/demos/detect_tabletop_cube.py`：Capture existing wrist RGB-D, HSV mask and a cube position in robot base.；入口：用户 CLI。
- `scripts/demos/move_to_observation_pose.py`：setup the Cartesian tabletop task in the existing realman_rl project.；入口：用户 CLI。
- `scripts/demos/render_ppo_tabletop.py`：Offscreen deterministic PPO replay: full desk and a gripper close-up.；入口：用户 CLI。
- `scripts/demos/render_ppo_tabletop_cartesian.py`：render the Cartesian tabletop task in the existing realman_rl project.；入口：用户 CLI。
- `scripts/demos/run_grasp_contact_cases.py`：接触、抬升、释放和失败对照案例；入口：用户 CLI。
- `scripts/demos/run_grasp_lift.py`：Approach, close, lift and hold a dynamic cube through four-action controls.；入口：用户 CLI。
- `scripts/demos/view_robot.py`：静态机器人模型 Viewer；入口：用户 CLI。

## scripts/diagnostics

- `scripts/diagnostics/compare_controllers.py`：PD 与重力补偿对照；入口：用户 CLI。
- `scripts/diagnostics/compare_fk.py`：厂家 / MuJoCo FK 对照；入口：用户 CLI。
- `scripts/diagnostics/compare_ik.py`：厂家离线 IK 自模型与跨模型验证；入口：用户 CLI。
- `scripts/diagnostics/compare_sac_boundary.py`：SAC 边界单因素实验及冻结最终评估；入口：用户 CLI。
- `scripts/diagnostics/diagnose_reach_rewards.py`：独立Home动作方向、旧奖励解析、同轨迹新旧计分与四张诊断图；不训练；入口：用户 CLI。
- `scripts/diagnostics/diagnose_sac_reach.py`：冻结 SAC 轨迹、反例与全 Replay TD 诊断；入口：用户 CLI。
- `scripts/diagnostics/probe_ppo_goals.py`：冻结PPO在合法reset目标下的同状态特征/动作敏感性，不执行step或训练；入口：用户 CLI。
- `scripts/diagnostics/probe_reach_env.py`：任务分布、奖励与随机探索探针，无学习器；入口：用户 CLI。
- `scripts/diagnostics/smoke_reach_env.py`：小规模无界面 Reach 交互 / Gymnasium checker；入口：用户 CLI。

## scripts/evaluate

- `scripts/evaluate/analyze_reach_results.py`：只读评估JSON；逐候选、方向、象限、半径组及动作偏置/裁剪角度，不加载模型；入口：用户 CLI。
- `scripts/evaluate/evaluate_bc_reach.py`：BC 模型加载与配对评估 CLI，经 BC 权重保护薄适配调用 Evaluation；入口：用户 CLI。
- `scripts/evaluate/evaluate_ppo_grasp_lift.py`：Independent evaluation of four-action PPO, with no staged-controller fallback.；入口：用户 CLI。
- `scripts/evaluate/evaluate_ppo_reach.py`：加载PPO及保存变换；离散逐目标/随机独立目标评估，重复数拒绝；入口：用户 CLI。
- `scripts/evaluate/evaluate_ppo_tabletop.py`：Read-only PPO evaluation from the saved pose, or the full tabletop task.；入口：用户 CLI。
- `scripts/evaluate/evaluate_ppo_tabletop_cartesian.py`：evaluate the Cartesian tabletop task in the existing realman_rl project.；入口：用户 CLI。
- `scripts/evaluate/evaluate_reach_baselines.py`：随机/规则评估 CLI；计算与循环直接导入 evaluation.baselines，保留参数/JSON/退出码；入口：用户 CLI。
- `scripts/evaluate/evaluate_sac_reach.py`：冻结 SAC / 随机 / 规则配对目标评估；入口：用户 CLI。

## scripts/models

- `scripts/models/build_dynamics_model.py`：重建六轴动力学 MJCF；入口：用户 CLI。
- `scripts/models/build_grasp_scene.py`：重建接触代理和动态方块场景；入口：用户 CLI。
- `scripts/models/build_gripper_model.py`：重建独立 / 组合夹爪模型；入口：用户 CLI。
- `scripts/models/build_tabletop_cartesian_scene.py`：Derive the Cartesian task scene; additions are visualization frames only.；入口：用户 CLI。
- `scripts/models/build_tabletop_scene.py`：Horizontal workbench with the fixed arm and a freely supported cube.；入口：用户 CLI。
- `scripts/models/inspect_model.py`：URDF / MJCF、mesh、惯量和关节检查；入口：用户 CLI。

## scripts/plots

- `scripts/plots/plot_ppo_training.py`：只读PPO JSON；聚合及逐目标成功/距离/首动作、训练成功率/终止PNG；入口：用户 CLI。
- `scripts/plots/plot_sac_training.py`：只读 SAC JSON，输出曲线 PNG 和原始序列；入口：用户 CLI。

## scripts/train

- `scripts/train/collect_reach_demonstrations.py`：公开观测规则策略示范数据采集；入口：用户 CLI。
- `scripts/train/train_bc_reach.py`：离线 BC MSE / Adam 训练入口；入口：用户 CLI。
- `scripts/train/train_ppo_grasp_lift.py`：Train four-action PPO from scratch or a recorded PPO curriculum checkpoint.；入口：用户 CLI。
- `scripts/train/train_ppo_reach.py`：PPO完整更新后评估/保存；离散任务自动全部候选，scratch；入口：用户 CLI。
- `scripts/train/train_ppo_tabletop.py`：Train seven-action tabletop PPO; curriculum checkpoints are explicit.；入口：用户 CLI。
- `scripts/train/train_ppo_tabletop_cartesian.py`：train the Cartesian tabletop task in the existing realman_rl project.；入口：用户 CLI。
- `scripts/train/train_sac_reach.py`：SAC 训练调度、纯 RL / 显式历史初始化分支；入口：用户 CLI。

## src

- `__init__.py`：轻量包标识，无运行时初始化；由包导入、运行时或模型反序列化使用。
- `paths.py`：源码仓库资源与输出目录定位，不依赖 cwd；调用者：`scripts/demos/demo_gripper.py`、`scripts/demos/run_grasp_contact_cases.py`、`scripts/demos/run_grasp_lift.py`、`scripts/demos/view_robot.py` 等 48 处。
- `viewer.py`：惰性 Viewer 生命周期、后端选择与线程清理；调用者：`scripts/demos/control_ee.py`、`scripts/demos/control_joints.py`、`scripts/demos/demo_gripper.py`、`scripts/demos/run_grasp_contact_cases.py` 等 12 处。

## src/realman_rl_simtoreal/cli

- `cli/__init__.py`：Command parsing and dispatch for retained script entry points.；由包导入、运行时或模型反序列化使用。
- `cli/tabletop_cartesian.py`：Script argument handling for the Cartesian tabletop task.；调用者：`scripts/demos/move_to_observation_pose.py`、`scripts/demos/render_ppo_tabletop_cartesian.py`、`scripts/evaluate/evaluate_ppo_tabletop_cartesian.py`、`scripts/train/train_ppo_tabletop_cartesian.py` 等 6 处。

## src/realman_rl_simtoreal/envs

- `envs/__init__.py`：轻量包标识，无运行时初始化；由包导入、运行时或模型反序列化使用。
- `envs/factory.py`：按保存配置构造 Reach / 连续目标 / 真值或视觉 Cartesian 环境，不依赖算法；调用者：`scripts/demos/detect_tabletop_cube.py`、`src/realman_rl_simtoreal/evaluation/reach.py`、`src/realman_rl_simtoreal/evaluation/tabletop_cartesian.py`、`src/realman_rl_simtoreal/rl/common/env_factory.py` 等 8 处。
- `envs/grasp_lift_env.py`：抓取抬升 Gymnasium 环境，41 维观测、四维动作、实际接触与安全终止；调用者：`scripts/demos/run_grasp_lift.py`、`src/realman_rl_simtoreal/envs/tabletop_pick_env.py`、`src/realman_rl_simtoreal/evaluation/grasp_lift.py`、`src/realman_rl_simtoreal/rl/ppo_grasp.py` 等 6 处。
- `envs/reach_env.py`：六轴Reach；默认legacy 33维、可选fetch_like_v1 12维；固定/球壳/离散目标，原奖励/step不变；调用者：`scripts/diagnostics/diagnose_reach_rewards.py`、`scripts/diagnostics/probe_ppo_goals.py`、`scripts/diagnostics/probe_reach_env.py`、`scripts/diagnostics/smoke_reach_env.py` 等 24 处。
- `envs/sequence_reach_env.py`：随机起点、连续多目标 Reach 与物理状态交接，保留 Gymnasium 接口；调用者：`src/realman_rl_simtoreal/envs/factory.py`、`tests/test_sequence_reach.py`。
- `envs/tabletop_cartesian_env.py`：Four Cartesian actions and 29 task-space observations after traditional setup.；调用者：`src/realman_rl_simtoreal/envs/factory.py`、`src/realman_rl_simtoreal/envs/tabletop_cartesian_vision_env.py`、`src/realman_rl_simtoreal/evaluation/tabletop_cartesian.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py` 等 6 处。
- `envs/tabletop_cartesian_vision_env.py`：原 RGB-D 生命周期与独立物理验收；组合纯视觉进度与观测编码；调用者：`src/realman_rl_simtoreal/envs/factory.py`、`src/realman_rl_simtoreal/evaluation/tabletop_cartesian.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`tests/test_refactor_contracts.py`。
- `envs/tabletop_pick_env.py`：Seven neural actions for zero/packing-pose to horizontal-table pick and lift.；调用者：`src/realman_rl_simtoreal/rl/ppo_tabletop.py`、`tests/test_tabletop_pick.py`。

## src/realman_rl_simtoreal/evaluation

- `evaluation/__init__.py`：轻量评估包标识；不导入环境或学习框架；调用者：`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`tests/test_evaluation.py`、`tests/test_fetch_observation.py`、`tests/test_tabletop_evaluation.py`。
- `evaluation/baselines.py`：原公开观测规则策略；早期随机/规则评估、奖励反例与独立性能剖析；调用者：`scripts/evaluate/evaluate_reach_baselines.py`、`src/realman_rl_simtoreal/evaluation/reach.py`、`src/realman_rl_simtoreal/imitation/bc.py`、`src/realman_rl_simtoreal/rl/sac_diagnostics.py` 等 7 处。
- `evaluation/grasp_lift.py`：原 GraspLift rollout / evaluate；阶段控制器或传入 predict 策略共用物理执行循环；调用者：`scripts/demos/run_grasp_lift.py`、`src/realman_rl_simtoreal/rl/ppo_grasp.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop.py`、`tests/test_evaluation.py` 等 5 处。
- `evaluation/metrics.py`：Reach / GraspLift / BC / 早期基线的原公式、成功率分母、延迟与目标分布统计；调用者：`scripts/evaluate/evaluate_reach_baselines.py`、`src/realman_rl_simtoreal/evaluation/baselines.py`、`src/realman_rl_simtoreal/evaluation/grasp_lift.py`、`src/realman_rl_simtoreal/evaluation/reach.py` 等 10 处。
- `evaluation/reach.py`：独立 Reach 回合、原 RL / BC JSON 协议和原始观测 callable 接口；不加载或训练算法；调用者：`scripts/diagnostics/compare_sac_boundary.py`、`scripts/evaluate/evaluate_sac_reach.py`、`scripts/train/train_sac_reach.py`、`src/realman_rl_simtoreal/imitation/bc.py` 等 10 处。
- `evaluation/tabletop_cartesian.py`：Algorithm-independent Cartesian tabletop episodes and independent evaluation.；调用者：`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`tests/test_tabletop_evaluation.py`。

## src/realman_rl_simtoreal/imitation

- `imitation/__init__.py`：轻量包标识，无运行时初始化；调用者：`tests/test_bc_reach.py`。
- `imitation/bc.py`：原 BC 网络、数据采集和监督训练；闭环评估仅保留权重保护薄适配；调用者：`scripts/evaluate/evaluate_bc_reach.py`、`scripts/train/collect_reach_demonstrations.py`、`scripts/train/train_bc_reach.py`、`src/realman_rl_simtoreal/imitation/bc_to_sac.py` 等 8 处。
- `imitation/bc_to_sac.py`：历史 BC→SAC Actor 均值通路校验、复制与偏差诊断；调用者：`scripts/train/train_sac_reach.py`、`src/realman_rl_simtoreal/rl/sac_monitoring.py`、`tests/test_sac_bc_initialization.py`。

## src/realman_rl_simtoreal/perception

- `perception/__init__.py`：Camera-only measurements; independent of simulation state and learning.；由包导入、运行时或模型反序列化使用。
- `perception/rgbd.py`：Orange cube localization from aligned RGB-D and calibrated camera geometry.；调用者：`scripts/demos/calibrate_cube_hsv.py`、`src/realman_rl_simtoreal/envs/tabletop_cartesian_vision_env.py`、`src/realman_rl_simtoreal/simulation/rgbd_camera.py`、`tests/test_rgbd_vision.py`。

## src/realman_rl_simtoreal/rl

- `rl/__init__.py`：轻量包标识，无运行时初始化；调用者：`tests/manual_structure_regression.py`、`tests/test_refactor_contracts.py`。
- `rl/cartesian_policy.py`：PPO Gaussian exploration with enough precision for contact and stable hold.；调用者：`src/realman_rl_simtoreal/rl/cartesian_reach_policy.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`tests/test_cartesian_learning.py`。
- `rl/cartesian_reach_policy.py`：Goal-conditioned PPO arm and jaw branches; kinematics stays in the executor.；调用者：`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`src/realman_rl_simtoreal/rl/vision_policy.py`、`tests/test_cartesian_learning.py`。
- `rl/common/__init__.py`：轻量包标识，无运行时初始化；调用者：`tests/test_refactor_contracts.py`。
- `rl/common/artifacts.py`：Reach 原来源范围、模型权重 / Adam 状态工具；原文件操作名称兼容导出；调用者：`scripts/demos/render_ppo_tabletop.py`、`scripts/demos/run_grasp_lift.py`、`scripts/diagnostics/compare_sac_boundary.py`、`scripts/diagnostics/diagnose_sac_reach.py` 等 28 处。
- `rl/common/env_factory.py`：原训练包装栈、公开 reset 观测探针；调用者：`scripts/train/train_ppo_reach.py`、`scripts/train/train_sac_reach.py`、`tests/test_bc_reach.py`、`tests/test_fetch_observation.py` 等 11 处。
- `rl/common/logging.py`：原 record_evaluation 的训练 logger 写入；TensorBoard 标签不变，不执行 Episode；调用者：`scripts/train/train_ppo_reach.py`、`scripts/train/train_sac_reach.py`。
- `rl/common/observation.py`：legacy schema 2 / fetch schema 3固定变换；33/12维顺序、checkpoint契约与模式检查；调用者：`scripts/diagnostics/probe_ppo_goals.py`、`scripts/evaluate/evaluate_sac_reach.py`、`scripts/train/train_ppo_reach.py`、`scripts/train/train_sac_reach.py` 等 21 处。
- `rl/ppo.py`：官方 PPO 配置、训练审计、checkpoint、确定性推理和参数保护薄适配；评估循环在 evaluation；调用者：`scripts/diagnostics/diagnose_reach_rewards.py`、`scripts/diagnostics/probe_ppo_goals.py`、`scripts/evaluate/evaluate_ppo_reach.py`、`scripts/train/train_ppo_reach.py` 等 13 处。
- `rl/ppo_grasp.py`：四维 PPO 训练 / checkpoint / 确定性适配；原奖励与课程名称兼容导出；调用者：`scripts/evaluate/evaluate_ppo_grasp_lift.py`、`scripts/train/train_ppo_grasp_lift.py`、`tests/test_ppo_grasp.py`、`tests/test_refactor_contracts.py` 等 5 处。
- `rl/ppo_tabletop.py`：关节桌面 PPO、原 SmoothJointPolicy / 分布类路径、训练 / checkpoint；原奖励与课程名称兼容导出；调用者：`scripts/demos/render_ppo_tabletop.py`、`scripts/evaluate/evaluate_ppo_tabletop.py`、`scripts/train/train_ppo_tabletop.py`、`tests/test_refactor_contracts.py` 等 5 处。
- `rl/ppo_tabletop_cartesian.py`：笛卡尔 PPO 训练、checkpoint、Reach / 视觉迁移与只读预测；评估仅作薄适配；调用者：`src/realman_rl_simtoreal/cli/tabletop_cartesian.py`、`src/realman_rl_simtoreal/visualization/tabletop_cartesian.py`、`tests/manual_structure_regression.py`、`tests/test_cartesian_learning.py` 等 6 处。
- `rl/sac.py`：官方 SAC 配置、模型创建及模型 / Replay checkpoint；调用者：`scripts/train/train_sac_reach.py`、`tests/test_fetch_observation.py`、`tests/test_pure_reach_training.py`、`tests/test_sac_bc_initialization.py` 等 6 处。
- `rl/sac_diagnostics.py`：冻结策略轨迹、Q / soft TD、参数指纹和 Replay 加载；调用者：`scripts/diagnostics/compare_sac_boundary.py`、`scripts/diagnostics/diagnose_sac_reach.py`、`scripts/train/train_sac_reach.py`、`src/realman_rl_simtoreal/rl/sac_monitoring.py` 等 8 处。
- `rl/sac_monitoring.py`：原 SAC 训练/安全审计、只读 Q 与运动监控，消费 evaluation.metrics 与中立统计；调用者：`scripts/train/train_sac_reach.py`。
- `rl/vision_policy.py`：PPO Cartesian control from RGB-D object estimates and robot proprioception.；调用者：`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`tests/test_rgbd_vision.py`。

## src/realman_rl_simtoreal/simulation

- `simulation/__init__.py`：轻量包标识，无运行时初始化；调用者：`scripts/demos/control_ee.py`、`scripts/diagnostics/compare_controllers.py`、`scripts/diagnostics/compare_fk.py`、`scripts/diagnostics/compare_ik.py` 等 16 处。
- `simulation/camera.py`：Camera-frame convention, independent of any learning task.；调用者：`scripts/models/build_tabletop_cartesian_scene.py`、`src/realman_rl_simtoreal/simulation/tabletop_runtime.py`。
- `simulation/cartesian_controller.py`：Task-independent Cartesian references -> local IK -> bounded joint servos.；调用者：`src/realman_rl_simtoreal/envs/tabletop_cartesian_env.py`。
- `simulation/control.py`：关节插值控制、固定周期步进、ForceMonitor 与原保护规则；调用者：`scripts/demos/control_ee.py`、`scripts/demos/control_joints.py`、`scripts/demos/run_grasp_contact_cases.py`、`scripts/diagnostics/compare_controllers.py` 等 24 处。
- `simulation/ee_control.py`：末端命令范围、碰撞预检、运动执行及误差；调用者：`scripts/demos/control_ee.py`、`scripts/diagnostics/compare_controllers.py`、`src/realman_rl_simtoreal/envs/reach_env.py`、`src/realman_rl_simtoreal/envs/sequence_reach_env.py` 等 8 处。
- `simulation/grasp_contact.py`：接触分类、力记录、动态方块及同步控制案例基础；调用者：`scripts/demos/run_grasp_contact_cases.py`、`src/realman_rl_simtoreal/envs/grasp_lift_env.py`、`src/realman_rl_simtoreal/envs/tabletop_pick_env.py`、`src/realman_rl_simtoreal/simulation/tabletop_runtime.py` 等 5 处。
- `simulation/grasp_controller.py`：独立阶段抓取控制器；公开反馈对中、接近、闭合、抬升和保持；调用者：`src/realman_rl_simtoreal/evaluation/grasp_lift.py`。
- `simulation/grasp_metrics.py`：Measure opposing inward pad pressure without treating top contact as grip.；调用者：`src/realman_rl_simtoreal/envs/tabletop_cartesian_env.py`、`src/realman_rl_simtoreal/evaluation/tabletop_cartesian.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`src/realman_rl_simtoreal/simulation/tabletop_runtime.py` 等 7 处。
- `simulation/gripper.py`：米制夹爪开口、mimic 校验、执行器与物理步进；调用者：`scripts/demos/demo_gripper.py`、`scripts/demos/run_grasp_contact_cases.py`、`scripts/models/build_grasp_scene.py`、`scripts/models/build_tabletop_scene.py` 等 18 处。
- `simulation/kinematics.py`：独立 MjData 中的 FK、Jacobian、限位 DLS IK；调用者：`scripts/demos/control_ee.py`、`scripts/demos/run_grasp_contact_cases.py`、`scripts/diagnostics/compare_controllers.py`、`scripts/diagnostics/compare_fk.py` 等 26 处。
- `simulation/model.py`：MuJoCo 模型加载、映射、初始化、重力补偿配置与结构检查；调用者：`scripts/demos/control_ee.py`、`scripts/demos/control_joints.py`、`scripts/demos/demo_gripper.py`、`scripts/demos/run_grasp_contact_cases.py` 等 22 处。
- `simulation/observation_pose.py`：Traditional zero-to-fixed-observation motion, before policy execution.；调用者：`src/realman_rl_simtoreal/simulation/tabletop_runtime.py`、`tests/test_tabletop_cartesian.py`。
- `simulation/observation_profiles.py`：Fixed robot observation poses, defined without any object or task state.；调用者：`src/realman_rl_simtoreal/envs/tabletop_cartesian_env.py`、`src/realman_rl_simtoreal/evaluation/tabletop_cartesian.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`src/realman_rl_simtoreal/simulation/observation_pose.py` 等 7 处。
- `simulation/rgbd_camera.py`：MuJoCo RGB-D adapter at the existing camera_link optical center.；调用者：`src/realman_rl_simtoreal/envs/tabletop_cartesian_vision_env.py`。
- `simulation/tabletop_runtime.py`：原桌面物理执行 / 启动 / 接触时序，最小只读关节与夹爪状态接口；调用者：`src/realman_rl_simtoreal/envs/tabletop_cartesian_env.py`、`src/realman_rl_simtoreal/visualization/tabletop_cartesian.py`、`tests/test_tabletop_cartesian.py`、`tests/test_tabletop_runtime.py`。
- `simulation/tool_kinematics.py`：Rigid tool-frame selection over the existing, unchanged FK/IK solver.；调用者：`src/realman_rl_simtoreal/simulation/tabletop_runtime.py`、`tests/test_tabletop_cartesian.py`。

## src/realman_rl_simtoreal/tasks

- `tasks/__init__.py`：轻量包标识，无运行时初始化；调用者：`tests/test_refactor_contracts.py`。
- `tasks/grasp_lift.py`：局部抓取配置、原奖励公式及组件、课程；不依赖算法；调用者：`scripts/demos/run_grasp_lift.py`、`src/realman_rl_simtoreal/envs/grasp_lift_env.py`、`src/realman_rl_simtoreal/rl/ppo_grasp.py`、`tests/test_evaluation.py` 等 6 处。
- `tasks/reach.py`：算法无关fixed / easy_random / random / axis_pair / axis_six显式目标配置；调用者：`scripts/diagnostics/diagnose_reach_rewards.py`、`scripts/diagnostics/probe_ppo_goals.py`、`scripts/diagnostics/probe_reach_env.py`、`scripts/train/train_ppo_reach.py` 等 18 处。
- `tasks/rewards.py`：奖励模式、进度奖励8项实验初值、组件键与checkpoint一致性校验；调用者：`scripts/diagnostics/diagnose_reach_rewards.py`、`scripts/train/train_ppo_reach.py`、`src/realman_rl_simtoreal/envs/reach_env.py`、`src/realman_rl_simtoreal/evaluation/reach.py` 等 7 处。
- `tasks/tabletop_cartesian.py`：笛卡尔任务配置、校验与原 EASY / STAGES 完成门槛；调用者：`scripts/demos/detect_tabletop_cube.py`、`src/realman_rl_simtoreal/envs/factory.py`、`src/realman_rl_simtoreal/envs/tabletop_cartesian_env.py`、`src/realman_rl_simtoreal/evaluation/tabletop_cartesian.py` 等 13 处。
- `tasks/tabletop_cartesian_reward.py`：Task-space approach, caging, contact and lift rewards with bounded completion.；调用者：`src/realman_rl_simtoreal/envs/tabletop_cartesian_env.py`、`src/realman_rl_simtoreal/envs/tabletop_cartesian_vision_env.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop_cartesian.py`、`tests/test_cartesian_learning.py` 等 5 处。
- `tasks/tabletop_pick.py`：关节桌面配置、原奖励公式及组件、课程；不依赖算法；调用者：`src/realman_rl_simtoreal/envs/tabletop_pick_env.py`、`src/realman_rl_simtoreal/rl/ppo_tabletop.py`、`tests/test_refactor_contracts.py`。

## src/realman_rl_simtoreal/utils

- `utils/__init__.py`：轻量包标识，无运行时初始化；调用者：`tests/test_refactor_contracts.py`。
- `utils/artifacts.py`：共享 JSON / JSONL、SHA256、版本、指定源码归档、空目录检查；原抓取来源范围保留；调用者：`scripts/demos/render_ppo_tabletop.py`、`scripts/demos/run_grasp_lift.py`、`scripts/evaluate/evaluate_ppo_grasp_lift.py`、`scripts/evaluate/evaluate_ppo_tabletop.py` 等 19 处。
- `utils/demo_helpers.py`：FK 样本姿态、原关节演示序列和结果格式化；调用者：`scripts/demos/control_ee.py`、`scripts/demos/control_joints.py`、`scripts/diagnostics/compare_controllers.py`、`scripts/diagnostics/compare_fk.py` 等 5 处。
- `utils/geometry.py`：共享 SE(3) 构造与逆变换；调用者：`scripts/diagnostics/compare_fk.py`、`scripts/diagnostics/compare_ik.py`。
- `utils/metrics.py`：训练与评估共享 MotionStats 和原余弦函数；只依赖 NumPy；调用者：`scripts/diagnostics/probe_reach_env.py`、`src/realman_rl_simtoreal/evaluation/reach.py`、`src/realman_rl_simtoreal/imitation/bc.py`、`src/realman_rl_simtoreal/rl/ppo.py` 等 7 处。
- `utils/tabletop_cartesian_cli.py`：旧 main 导入的兼容转发；唯一解析实现位于 cli；调用者：`tests/test_refactor_contracts.py`。
- `utils/tabletop_cartesian_render.py`：旧 Recorder / run 导入的兼容转发；唯一回放实现位于 visualization；调用者：`tests/test_refactor_contracts.py`。

## src/realman_rl_simtoreal/vendor

- `vendor/__init__.py`：轻量包标识，无运行时初始化；调用者：`tests/test_fk_comparison.py`、`tests/test_ik_comparison.py`。
- `vendor/realman_fk.py`：可选厂家离线 FK SDK 适配与会话保护；调用者：`scripts/diagnostics/compare_fk.py`、`scripts/diagnostics/compare_ik.py`、`src/realman_rl_simtoreal/vendor/realman_ik.py`、`tests/test_fk_comparison.py` 等 5 处。
- `vendor/realman_ik.py`：厂家离线 IK、解集与自模型复核；调用者：`scripts/diagnostics/compare_ik.py`、`tests/test_ik_comparison.py`。

## src/realman_rl_simtoreal/visualization

- `visualization/__init__.py`：Simulation replay and image generation; no training implementations.；由包导入、运行时或模型反序列化使用。
- `visualization/tabletop_cartesian.py`：Actual MuJoCo replay: overview, top view, and the wrist camera.；调用者：`src/realman_rl_simtoreal/cli/tabletop_cartesian.py`、`src/realman_rl_simtoreal/utils/tabletop_cartesian_render.py`、`tests/test_refactor_contracts.py`。

## tests

- `tests/manual_structure_regression.py`：Opt-in existing-model trajectory regression; never trains or modifies outputs.；入口：手动 python -I（不参与 discover）。
- `tests/test_bc_reach.py`：BC 数据、特征、模型、监督更新与闭环；入口：unittest discover。
- `tests/test_cartesian_learning.py`：Regress the failed full-task reward ranking and PPO contact precision.；入口：unittest discover。
- `tests/test_evaluation.py`：17 项评估接口/指标/独立环境/参数保护/无算法导入及 JSON 协议测试；入口：unittest discover。
- `tests/test_fetch_observation.py`：双模式维度、真实Site运动、网络实际输入、物理一致性、旧checkpoint及单512步PPO Smoke；入口：unittest discover。
- `tests/test_fk_comparison.py`：FK 坐标、厂家配置与差异对照；入口：unittest discover。
- `tests/test_goal_conditioned_reach.py`：离散候选、旧模式精确回归、一次两方向PPO更新及逐目标评估/监控；入口：unittest discover。
- `tests/test_grasp_contact.py`：接触、代理、同步步进与严格失败反例；入口：unittest discover。
- `tests/test_grasp_lift.py`：实际抓取接口、物理循环、阶段控制器、失败对照与指标；入口：unittest discover。
- `tests/test_gravity_compensation.py`：重力补偿路由、合力矩限幅及控制对照；入口：unittest discover。
- `tests/test_gripper.py`：夹爪结构、mimic、开口、模型重生成和演示；入口：unittest discover。
- `tests/test_ik_comparison.py`：厂家 IK 自模型 / 边界与跨模型；入口：unittest discover。
- `tests/test_joint_control.py`：动力学结构、保持、关节运动、力矩与碰撞；入口：unittest discover。
- `tests/test_kinematics.py`：FK / Jacobian / DLS IK / 末端控制与资源重生成；入口：unittest discover。
- `tests/test_package.py`：包发现、无循环 / 分层、全部 argparse / shared CLI help、退出码与 Viewer 生命周期；入口：unittest discover。
- `tests/test_ppo_grasp.py`：四维 PPO 实际更新、课程、奖励、checkpoint 和只读策略评估；入口：unittest discover。
- `tests/test_ppo_random_reach.py`：200个随机目标/观测/reset契约、分组与裁剪分析、单512步随机PPO Smoke及20目标评估接口；入口：unittest discover。
- `tests/test_ppo_reach.py`：单次完整 PPO Rollout 更新、动作与损失、日志、checkpoint；阻断 SAC / BC 导入；入口：unittest discover。
- `tests/test_pure_reach_training.py`：任务配置、奖励 / 探针、纯 SAC 隔离和运动统计；入口：unittest discover。
- `tests/test_reach_env.py`：Gymnasium、动作、观测、奖励、真实物理步与保护；入口：unittest discover。
- `tests/test_reach_progress_reward.py`：25项奖励方向、批量、漏洞/边界、兼容、组件与单次512步PPO Smoke；入口：unittest discover。
- `tests/test_reach_task.py`：随机可达目标、奖励一致性、基线及 CLI；入口：unittest discover。
- `tests/test_refactor_contracts.py`：重构前 JSON / 来源 / CLI / 策略类 / 物理方法 / 24 组奖励 / 7 类旧模型 / 视觉组合 / 任务别名契约；入口：unittest discover。
- `tests/test_rgbd_vision.py`：Camera geometry, true-state isolation, lost-target handling and visual PPO.；入口：unittest discover。
- `tests/test_sac_bc_initialization.py`：历史 BC Actor 映射、受保护参数、短 SAC 更新与监控；入口：unittest discover。
- `tests/test_sac_diagnostics.py`：冻结动作轨迹、Q / TD / Replay、随机状态与参数不变性；入口：unittest discover。
- `tests/test_sac_reach.py`：官方 SAC 参数更新、预处理、保存 / 加载和确定性评估；入口：unittest discover。
- `tests/test_sequence_reach.py`：随机起点、连续目标、奖励上下文与状态交接；入口：unittest discover。
- `tests/test_tabletop_cartesian.py`：Exercise the policy/controller boundary and real MuJoCo dynamics.；入口：unittest discover。
- `tests/test_tabletop_evaluation.py`：Algorithm-free tabletop episodes, ownership and read-only state guards.；入口：unittest discover。
- `tests/test_tabletop_pick.py`：Horizontal tabletop geometry, exact zero reset, real PPO and read-only inference.；入口：unittest discover。
- `tests/test_tabletop_runtime.py`：Read-only state facade and unchanged latest-runtime physical clock.；入口：unittest discover。
- `tests/test_wide_reach.py`：20 cm 目标分布、原物理控制和 PPO 迁移契约；入口：unittest discover。

## 保留边界

- MotionStats / cosine 保留在 `utils/metrics.py`，训练和评估都可复用。
- 自定义 SB3 Policy 与特征类的原模块路径是 checkpoint 契约，不能随目录调整移动。
- Reach、GraspLift、关节桌面和笛卡尔桌面不共享不同语义的奖励、动作或成功条件。
- `tests/fixtures`、`assets`、`outputs` 属于回归基准、物理资源和历史实验，重构不能覆写。
