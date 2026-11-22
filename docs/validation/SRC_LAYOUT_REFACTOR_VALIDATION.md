# src 目录重构与回归验证

2026-11-22 · RealMan RML63II-6F-V

本次将 MuJoCo 仿真按物理、运动、设备、运行时分层，将环境与任务按 Reach 和抓取分组。四套 PPO 调度进入 `apps/training`。现行命令保留原调用方式，SAC 专用入口退役。全量测试 **505 项通过**，**2,355 个本地 PPO 归档**的固定输入推理逐位一致，**4 条历史桌面长回合**复现基准。

## 1. 实际 src 目录树

以下为工作区实际存在的 95 个 Python 文件。

```text
src/
└── realman_rl_simtoreal/
    ├── __init__.py
    ├── apps/
    │   ├── __init__.py
    │   ├── cli/
    │   │   ├── __init__.py
    │   │   └── tabletop_cartesian.py
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── grasp_lift.py
    │   │   ├── reach.py
    │   │   ├── tabletop_cartesian.py
    │   │   └── tabletop_joint.py
    │   └── visualization/
    │       ├── __init__.py
    │       ├── tabletop_cartesian.py
    │       ├── tabletop_joint.py
    │       ├── viewer.py
    │       └── vision_grasp.py
    ├── cli/
    │   ├── __init__.py
    │   └── tabletop_cartesian.py
    ├── deployment/
    │   └── __init__.py
    ├── envs/
    │   ├── __init__.py
    │   ├── factory.py
    │   ├── grasp/
    │   │   ├── __init__.py
    │   │   ├── grasp_lift_env.py
    │   │   ├── tabletop_cartesian_env.py
    │   │   ├── tabletop_cartesian_vision_env.py
    │   │   └── tabletop_pick_env.py
    │   └── reach/
    │       ├── __init__.py
    │       ├── reach_env.py
    │       └── sequence_reach_env.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── baselines.py
    │   ├── grasp_lift.py
    │   ├── metrics.py
    │   ├── reach.py
    │   └── tabletop_cartesian.py
    ├── hardware/
    │   └── __init__.py
    ├── imitation/
    │   ├── __init__.py
    │   └── bc.py
    ├── paths.py
    ├── perception/
    │   ├── __init__.py
    │   └── rgbd.py
    ├── rl/
    │   ├── __init__.py
    │   ├── cartesian_policy.py
    │   ├── cartesian_reach_policy.py
    │   ├── common/
    │   │   ├── __init__.py
    │   │   ├── artifacts.py
    │   │   ├── env_factory.py
    │   │   ├── logging.py
    │   │   └── observation.py
    │   ├── ppo.py
    │   ├── ppo_cartesian.py
    │   ├── ppo_grasp.py
    │   ├── ppo_tabletop.py
    │   ├── ppo_tabletop_cartesian.py
    │   └── vision_policy.py
    ├── simulation/
    │   ├── __init__.py
    │   ├── devices/
    │   │   ├── __init__.py
    │   │   ├── camera_geometry.py
    │   │   ├── mujoco_gripper.py
    │   │   └── mujoco_rgbd_camera.py
    │   ├── motion/
    │   │   ├── __init__.py
    │   │   ├── cartesian_controller.py
    │   │   ├── kinematics.py
    │   │   ├── staged_grasp_controller.py
    │   │   ├── tool_kinematics.py
    │   │   └── traditional_ee_control.py
    │   ├── physics/
    │   │   ├── __init__.py
    │   │   ├── grasp_contact.py
    │   │   ├── grasp_metrics.py
    │   │   ├── joint_control.py
    │   │   └── model.py
    │   └── runtime/
    │       ├── __init__.py
    │       ├── observation_presets.py
    │       ├── observation_startup.py
    │       └── tabletop.py
    ├── tasks/
    │   ├── __init__.py
    │   ├── grasp/
    │   │   ├── __init__.py
    │   │   ├── lift.py
    │   │   ├── tabletop_cartesian.py
    │   │   ├── tabletop_cartesian_reward.py
    │   │   └── tabletop_joint.py
    │   └── reach/
    │       ├── __init__.py
    │       ├── config.py
    │       └── reward.py
    ├── utils/
    │   ├── __init__.py
    │   ├── artifacts.py
    │   ├── demo_helpers.py
    │   ├── geometry.py
    │   ├── metrics.py
    │   ├── tabletop_cartesian_cli.py
    │   └── tabletop_cartesian_render.py
    ├── vendor/
    │   ├── __init__.py
    │   ├── realman_fk.py
    │   └── realman_ik.py
    ├── viewer.py
    └── visualization/
        ├── __init__.py
        └── tabletop_cartesian.py
```

## 2. 文件迁移对照

包内路径相对 `src/realman_rl_simtoreal/`。脚本和测试路径相对仓库根。

| 原文件 | 新文件 | 修改原因 |
|---|---|---|
| `simulation/model.py` | `simulation/physics/model.py` | 物理模型加载、映射与重力补偿校验 |
| `simulation/control.py` | `simulation/physics/joint_control.py` | 关节参考限速、物理步进及执行审计 |
| `simulation/grasp_contact.py` | `simulation/physics/grasp_contact.py` | 接触分类、穿透检查与原接触力记录 |
| `simulation/grasp_metrics.py` | `simulation/physics/grasp_metrics.py` | 单独计算双侧内向夹持量 |
| `simulation/kinematics.py` | `simulation/motion/kinematics.py` | 法兰 FK、Jacobian 与阻尼最小二乘 IK |
| `simulation/tool_kinematics.py` | `simulation/motion/tool_kinematics.py` | 工具 TCP 与附件位姿，复用法兰求解器 |
| `simulation/ee_control.py` | `simulation/motion/traditional_ee_control.py` | 固定朝向末端演示与插值预检 |
| `simulation/cartesian_controller.py` | `simulation/motion/cartesian_controller.py` | 桌面任务的笛卡尔参考、回溯 IK 与限速 |
| `simulation/grasp_controller.py` | `simulation/motion/staged_grasp_controller.py` | 局部抓取传统分阶段基线 |
| `simulation/gripper.py` | `simulation/devices/mujoco_gripper.py` | MuJoCo EG2-4C2 开口、mimic 和执行器 |
| `simulation/camera.py` | `simulation/devices/camera_geometry.py` | MuJoCo 相机方向几何 |
| `simulation/rgbd_camera.py` | `simulation/devices/mujoco_rgbd_camera.py` | 虚拟腕部相机渲染与 RGB-D 帧适配 |
| `simulation/tabletop_runtime.py` | `simulation/runtime/tabletop.py` | 组合桌面机器人、执行器和物理读数 |
| `simulation/observation_pose.py` | `simulation/runtime/observation_startup.py` | 零位到固定观察位的启动轨迹 |
| `simulation/observation_profiles.py` | `simulation/runtime/observation_presets.py` | 固定观察位及 checkpoint 启动契约 |
| `envs/reach_env.py` | `envs/reach/reach_env.py` | 单目标 Reach，保留原步进及奖励方法 |
| `envs/sequence_reach_env.py` | `envs/reach/sequence_reach_env.py` | 连续多目标 Reach，保留状态交接 |
| `envs/grasp_lift_env.py` | `envs/grasp/grasp_lift_env.py` | 局部夹取环境，接收原四动作训练包装及编码 |
| `envs/tabletop_pick_env.py` | `envs/grasp/tabletop_pick_env.py` | 关节桌面环境，接收原七动作训练包装及编码 |
| `envs/tabletop_cartesian_env.py` | `envs/grasp/tabletop_cartesian_env.py` | 笛卡尔真值桌面环境 |
| `envs/tabletop_cartesian_vision_env.py` | `envs/grasp/tabletop_cartesian_vision_env.py` | 继承桌面物理环境的 RGB-D 观测和完成逻辑 |
| `tasks/reach.py` | `tasks/reach/config.py` | Reach 目标分布和环境配置，接收原 PPO 默认配置选择 |
| `tasks/rewards.py` | `tasks/reach/reward.py` | Reach 奖励版本、参数和 checkpoint 检查 |
| `tasks/grasp_lift.py` | `tasks/grasp/lift.py` | 局部夹取配置、课程与势差奖励 |
| `tasks/tabletop_pick.py` | `tasks/grasp/tabletop_joint.py` | 关节桌面配置、课程与势差奖励 |
| `tasks/tabletop_cartesian.py` | `tasks/grasp/tabletop_cartesian.py` | 笛卡尔课程、配置与观测检查 |
| `tasks/tabletop_cartesian_reward.py` | `tasks/grasp/tabletop_cartesian_reward.py` | 笛卡尔接近、夹持、抬升奖励 |
| `cli/tabletop_cartesian.py` | `apps/cli/tabletop_cartesian.py` | 应用层统一处理笛卡尔四个命令，旧路径转发 |
| `viewer.py` | `apps/visualization/viewer.py` | 集中 Viewer 初始化和线程收尾，旧路径转发 |
| `visualization/tabletop_cartesian.py` | `apps/visualization/tabletop_cartesian.py` | 桌面回放及图像输出，旧路径转发 |
| `scripts/train/train_ppo_reach.py` | `apps/training/reach.py` | 迁移完整 Reach 调度，脚本继续解析原参数 |
| `scripts/train/train_ppo_grasp_lift.py` | `apps/training/grasp_lift.py` | 迁移局部抓取训练、课程、输出和权重选择 |
| `scripts/train/train_ppo_tabletop.py` | `apps/training/tabletop_joint.py` | 迁移关节桌面训练与评估调度 |
| `rl/ppo_tabletop_cartesian.py` | `apps/training/tabletop_cartesian.py`、`rl/ppo_cartesian.py` | 拆开训练调度和模型/评估薄适配，旧模块提供兼容接口 |
| `rl/ppo.py` | `tasks/reach/config.py` | 迁移环境默认配置选择，RL 保留模型、回调及同名入口 |
| `rl/ppo_grasp.py` | `envs/grasp/grasp_lift_env.py`、`tasks/grasp/lift.py` | 拆出编码、环境包装与课程配置，RL 提供原名称及 PPO 模型初始化 |
| `rl/ppo_tabletop.py` | `envs/grasp/tabletop_pick_env.py`、`tasks/grasp/tabletop_joint.py` | 拆出编码、环境包装与课程配置，原序列化策略类留在 RL |
| `utils/artifacts.py` | `evaluation/grasp_lift.py` | 迁移局部抓取专用的 experiment_metadata，通用产物 I/O 留在 utils |
| `scripts/demos/render_ppo_tabletop.py` | `apps/visualization/tabletop_joint.py` | 提取关节桌面回放及产物生成，脚本保留原命令 |
| `scripts/demos/demo_vision_grasp.py` | `apps/visualization/vision_grasp.py` | 提取 RGB-D 交互回放及暂停、重启逻辑，脚本保留原命令 |
| `tests/test_sac_reach.py` | `tests/test_reach_transform.py` | 保留公共观测变换和接口验证，移除 SAC 专用用例 |
| `tests/test_sac_diagnostics.py` | `tests/test_reach_boundaries.py` | 保留 Reach 终止、拒绝保持和固定种子轨迹回归，移除 soft TD 用例 |
| `tests/test_pure_reach_training.py`、`tests/test_fetch_observation.py` | 原文件 | 移除对退役入口的调用，保留奖励、观测及 BC/PPO 用例 |
| `rl/sac.py` | 已移除 | 退役 SAC 模型创建及训练 checkpoint 写入 |
| `rl/sac_monitoring.py` | 已移除 | 退役 SAC 训练监控和 Q 探针 |
| `rl/sac_diagnostics.py` | 已移除 | 退役 SAC 轨迹、Replay 和 TD 诊断 |
| `imitation/bc_to_sac.py` | 已移除 | 退役 BC Actor 到 SAC 的权重迁移，BC 独立流程继续使用 |
| `scripts/train/train_sac_reach.py` | 已移除 | 退役 SAC 训练入口 |
| `scripts/evaluate/evaluate_sac_reach.py` | 已移除 | 退役 SAC 专用评估入口 |
| `scripts/diagnostics/compare_sac_boundary.py` | 已移除 | 退役 SAC 边界对照入口 |
| `scripts/diagnostics/diagnose_sac_reach.py` | 已移除 | 退役 SAC Q/TD 诊断入口 |
| `scripts/plots/plot_sac_training.py` | 已移除 | 退役 SAC 专用绘图入口 |
| `tests/test_sac_bc_initialization.py` | 已移除 | 移除针对退役迁移与监控入口的专用用例 |
| 新建 | `hardware/__init__.py`、`deployment/__init__.py` | 分别定义真实设备和策略部署边界 |

## 3. 模块职责

- **simulation** 提供 MuJoCo 物理模型、关节执行、FK/IK、夹爪、相机及桌面组件组合。
- **hardware** 预留真实机械臂、夹爪和相机的设备适配，当前包内只有初始化说明。
- **deployment** 预留部署侧观测、PPO 推理、任务执行与部署检查，当前包内只有初始化说明。
- **envs** 暴露 Gymnasium 契约及环境工厂，按任务保存原继承、训练包装和执行逻辑。
- **tasks** 保存任务分布、配置、课程和奖励。`ReachEnv` 保留原奖励方法，继续支持原标量、批次和目标重标记行为。
- **rl** 提供 SB3 PPO 模型创建、策略、分布、模型/Adam 状态、checkpoint 与算法回调，薄适配把回合执行交给 evaluation。
- **imitation** 保留 BC 示范采集、监督学习、权重加载及评估适配。
- **evaluation** 执行独立回合、规则基线和结果统计，接受调用者注入的策略。
- **perception** 使用 `RgbdFrame`、HSV、米制深度和基座坐标进行检测及跟踪。
- **apps** 组织 CLI、四套 PPO 训练和 MuJoCo Viewer、关节/笛卡尔回放、视觉交互。
- **vendor** 保持原厂离线 FK/IK 适配器。
- **utils** 提供共享几何、演示、运动统计、产物 I/O 和 SHA256。源码详情见 [文件地图](../FILE_MAP.md) 和 [架构](../architecture.md)。

## 4. 功能与兼容性

- **移动范围**包括全部 15 个仿真实现、6 个环境及 6 个任务模块。执行 AST 对照覆盖原 27 个模块和 `perception/rgbd.py`，仅排除文档、导入路径和新增迁入的独立定义。历史物理方法和奖励数值的 fixture 继续由单元测试检查。
- **拆分范围**包括训练编排、课程配置、抓取包装、关节/笛卡尔模型初始化和回放。局部与关节桌面共享同一处 Monitor、SB3 向量环境和 NaN 检查创建逻辑。局部抓取探索标准差及夹爪偏置复用模型初始化函数。各任务保留现有课程、评估节奏和指标字段。
- **旧接口**包括 `cli.tabletop_cartesian`、`visualization.tabletop_cartesian`、`viewer`、两个 `utils.tabletop_cartesian_*` 转发和原 `rl.ppo_tabletop_cartesian` 加载、评估、训练接口。局部/关节桌面的原 RL 奖励、课程和编码名称继续指向现行实现。其他仓库调用者已改用分层后的路径。
- **序列化**继续使用现有的 `rl/cartesian_policy.py`、`rl/cartesian_reach_policy.py`、`rl/vision_policy.py` 和 `rl/ppo_tabletop.py` 类定义。对照覆盖共 9 个原策略、特征、网络和分布类的可执行 AST，其中包括 `SmoothJointPolicy` 和 `CorrelatedPositionNoise`。
- **SAC 退役**包括 3 个专用 RL 模块、BC→SAC 迁移模块和 5 个入口。算法无关的 Reach 评估仍可接收已加载策略，历史 SAC checkpoint 仍由 SB3 只读加载测试覆盖。独立 BC 及共享工具继续使用，原实验报告、训练数据和模型归档逐字节匹配基线。
- **输出**沿用各脚本的原参数、默认目录与产物协议。新输出的源码键、归档路径及 SHA256 对应现行文件。Reach、局部抓取、关节桌面与笛卡尔任务的空间维度、单位、奖励、完成规则和 PPO/SB3 训练配置由回归覆盖。
- **保留边界**包括 Reach 既有奖励方法、各环境的专用控制周期和桌面任务的独立统计。四套训练保留各自的调度。旧笛卡尔模块保留经过验证的启动和观测契约检查，历史权重迁移仍使用原兼容参数。

### 归档中的项目内类引用

- `realman_rl_simtoreal.rl.cartesian_policy.CartesianPPOPolicy` 出现在 383 个模型归档中。
- `realman_rl_simtoreal.rl.cartesian_reach_policy.CartesianGoalFeatures` 出现在 51 个模型归档中。
- `realman_rl_simtoreal.rl.cartesian_reach_policy.ReachGraspPPOPolicy` 出现在 51 个模型归档中。
- `realman_rl_simtoreal.rl.ppo_tabletop.SmoothJointPolicy` 出现在 696 个模型归档中。
- `realman_rl_simtoreal.rl.vision_policy.VisualCartesianGoalFeatures` 出现在 38 个模型归档中。
- `realman_rl_simtoreal.rl.vision_policy.VisualReachGraspPPOPolicy` 出现在 38 个模型归档中。

## 5. 实测结果

环境为 Conda `realman_rl`，Python 3.11、MuJoCo 3.17.0、Gymnasium 1.4.0、SB3 2.9.2、PyTorch 2.14.0+cpu 和 NumPy 2.4.6。以下 `python` 指向该环境的解释器。

| 检查 | 通过 | 失败 | 错误 | 跳过 |
|---|---:|---:|---:|---:|
| 迁移前全量单元测试，676.035 秒 | 532 | 0 | 0 | 0 |
| 仿真分层专项 | 124 | 0 | 0 | 0 |
| 环境和任务专项 | 97 | 0 | 0 | 0 |
| 应用、分层、旧契约和感知专项 | 52 | 0 | 0 | 0 |
| 最终全量单元测试，674.455 秒 | 505 | 0 | 0 | 0 |

迁移前后相差 27 个 SAC 专用测试。Reach 的边界、历史转移和共用观测测试迁入现行测试文件。上述结果为修正阶段性导入和隔离副本设置后的通过记录。历史 fixtures 中的预期数据和容差继续沿用。

```bash
MUJOCO_GL=egl python -I -m unittest discover -s tests -v
MUJOCO_GL=egl python -I -m unittest discover -s tests -v \
  -k test_kinematics -k test_joint_control -k test_gravity_compensation \
  -k test_ik_comparison -k test_gripper -k test_grasp_contact -k test_tabletop_runtime -k test_rgbd_vision
MUJOCO_GL=egl python -I -m unittest discover -s tests -v \
  -k test_reach_env -k test_sequence_reach -k test_reach_task -k PureTaskRewardTests \
  -k test_grasp_lift -k VisualCompositionContracts -k FrozenPolicyAndTaskContracts
MUJOCO_GL=egl python -I -m unittest discover -s tests -v \
  -k test_refactor_contracts -k test_package -k test_reach_boundaries \
  -k test_reach_transform -k RgbdPerceptionTests
python -I -m compileall -q src scripts tests
MUJOCO_GL=egl python -I tests/manual_structure_regression.py \
  --json /tmp/realman-src-refactor.N4Drhdqf/trajectories.json
```

### 导入、入口与结果核验

- 在仓库外以隔离 Python 用 `pkgutil.walk_packages` 和 `importlib.import_module` 导入 95 个包和模块。静态图测试检查包内循环、反向依赖和脚本导入。
- 34 个正式 CLI 从外部目录通过 `--help` 检查。所有现行脚本及共享笛卡尔命令的 222 条 `add_argument` 声明与迁移前相同。5 个无参数构建器在使用独立资产副本的环境中以 `python -I scripts/models/build_*.py` 逐个运行，38 个资产文件的摘要逐字节相同。
- 对保存的真值桌面观察位执行 `scripts/demos/move_to_observation_pose.py`，在临时路径生成启动 JSON。用 `scripts/demos/detect_tabletop_cube.py --object-radius .02 --seed 990010` 生成 RGB、Mask、深度和检测 JSON。视觉交互入口在无显示环境返回原错误信息及退出码 1。
- 训练测试执行各任务原小步数更新，核对配置、初始和最终模型、阶段、评估日志及输出保护。`assets/`、`outputs/` 和 `tests/fixtures/` 共 11,118 个文件与迁移前 SHA256 一致。

### 模型和固定种子仿真

- SB3 层直接加载全部 2,355 个 PPO 归档。每个模型都用 3 组固定观测对照原始动作、裁剪后动作、权重、完整 Adam 状态和训练计数。归档清查和文件完整性检查还覆盖 93 个历史 off-policy 归档。
- 项目加载器的历史契约测试验证 Reach、局部夹取、关节桌面、笛卡尔真值、RGB-D PPO，以及独立 BC 和 1 个历史 SAC 模型的固定输入。全部 7 个参考文件均加载并匹配原基准。
- 手动长轨迹对照逐物理步比较 time、qpos、qvel、ctrl 和执行器总驱动力，逐策略步比较观测、raw/applied 动作、位姿、接触、奖励和终止。RGB-D 同时比较图像、米制深度和时间戳摘要。四个回合共 5,575 个策略步和 372,550 个物理步，优化器、权重及计数均保持一致。

| 历史场景 | seed | 策略步 | 1 ms 物理步 | 对照 |
|---|---:|---:|---:|---|
| 真值固定 | 850000 | 1290 | 87950 | 完全一致 |
| 真值随机半径 0.02 m | 870000 | 1297 | 88300 | 完全一致 |
| RGB-D 固定 | 910000 | 1513 | 99100 | 完全一致 |
| RGB-D 随机半径 0.02 m | 920000 | 1475 | 97200 | 完全一致 |

另以 seed 17721 在单目标 Reach、连续 Reach、局部抓取训练包装、关节桌面训练包装各执行 12 步，前后状态、观测、奖励与诊断的摘要逐位一致。额外对照实际执行了 `/tmp/realman-src-refactor.N4Drhdqf/check_checkpoints.py before|after`、`/tmp/realman-src-refactor.N4Drhdqf/check_source_changes.py` 和 `/tmp/realman-src-refactor.N4Drhdqf/compare_seeded_envs.py before|after`。

图形桌面上的互动操作、实际相机和真机执行尚未实测。本次验证使用历史权重与单元测试短训练预算，长期训练和历史大样本成功率沿用对应实验报告。

### 工作区记录

读取完整附件前执行过一次只读 `git status`，随后对当前工作区实施文件级迁移。隔离验证初次运行时，构建器通过资产链接重写了 `assets/scenes/grasp_contact.xml`，已按迁移前 SHA256 恢复原字节。分阶段快照同步中短暂移除的五个现行说明或依赖文件也从快照恢复，再按本次范围更新现行说明及依赖注释。

## 6. 中文注释与 Docstring

- **仿真**补充 `Kinematics`、`ToolKinematics`、`CartesianController`、`TabletopRuntime`、`Gripper`、接触审计和 RGB-D Camera，说明 rad、m、N、控制周期、参考状态、实际状态和世界/工具/基座坐标转换。
- **环境与任务**补充六个环境、任务工厂、课程配置和奖励函数概述，说明物理保持计时、动作前目标正则及视觉完成的独立物理验收。
- **PPO 与应用**补充原自定义策略、特征提取器、持续相关噪声、Actor/Critic、固定观测缩放及训练/回放入口。解释线性变换下的协方差和用 raw/applied 动作审计环境裁剪。
- **感知**补充 `RgbdFrame`、`OrangeCubeDetector`、`VisualTracker`、去畸变反投影、遮挡下的位置恢复和跟踪超时。
- **开发边界**已有代码对应的控制、感知及 checkpoint 接口已记录。真实设备与部署模块的接口文档随实际实现补充。旧实验命令仍可通过 [历史验证索引](README.md) 追溯。
