# RealMan RL Sim-to-Real 项目结构重构报告

日期：2026-11-10。工作分支：`refactor/project-organization`。基线 commit：`c459787`（Merge pull request #5 from xlmys2022/feat/sac-reach）。开始时工作区干净；始终在用户已有分支工作，没有切换分支、提交或推送。

## 1. 重构结果与范围

阶段 0 基线、阶段 1 脚本分类、阶段 2 公共模块提取和 experiments 迁移、阶段 3 文档整理、阶段 4 完整回归与兼容性验证均已完成；未执行正式长训练。

Python 文件总数从 **66 增至 77**：核心包 24→35，脚本保持 26，测试保持 16。原 8 个 experiments 业务模块被拆分为 15 个职责模块，旧包标识删除，新增 5 个轻量包标识。未创建空业务文件；原有 266 项测试全部保留，并加强导入边界、包发现和 CLI 覆盖。

完整代码与文档树如下；资产和 outputs 的内容原样保留，仅折叠展示这两类已有大量资源 / 产物目录，忽略 Python 缓存、Git 元数据与安装生成文件：

```text
realman_rl_simtoreal/
├── .gitignore
├── README.md
├── assets/
│   ├── grippers/
│   ├── robots/
│   └── scenes/
├── docs/
│   ├── FILE_MAP.md
│   ├── architecture.md
│   ├── experiments/
│   │   ├── ARCHITECTURE_BEFORE_REFACTOR.md
│   │   ├── README.md
│   │   └── README_BEFORE_REFACTOR.md
│   └── validation/
│       ├── BC_REACH_VALIDATION.md
│       ├── CONTROL_VALIDATION.md
│       ├── DYNAMICS_VALIDATION.md
│       ├── GRASP_CONTACT_VALIDATION.md
│       ├── GRIPPER_VALIDATION.md
│       ├── KINEMATICS_VALIDATION.md
│       ├── PPO_REACH_VALIDATION.md
│       ├── PROJECT_ORGANIZATION_REFACTOR.md
│       ├── PURE_REACH_VALIDATION.md
│       ├── REACH_ENV_VALIDATION.md
│       ├── REACH_TASK_VALIDATION.md
│       ├── README.md
│       ├── REALMAN_IK_VALIDATION.md
│       ├── REALMAN_KINEMATICS_VALIDATION.md
│       ├── REFACTOR_VALIDATION.md
│       ├── SAC_BC_INIT_VALIDATION.md
│       ├── SAC_DIAGNOSTICS_VALIDATION.md
│       ├── SAC_REACH_VALIDATION.md
│       ├── VALIDATION.md
│       └── project_organization_checks.json
├── outputs/
├── pyproject.toml
├── requirements-rl.txt
├── requirements-vendor.txt
├── requirements.txt
├── scripts/
│   ├── demos/
│   │   ├── control_ee.py
│   │   ├── control_joints.py
│   │   ├── demo_gripper.py
│   │   ├── run_grasp_contact_cases.py
│   │   └── view_robot.py
│   ├── diagnostics/
│   │   ├── compare_controllers.py
│   │   ├── compare_fk.py
│   │   ├── compare_ik.py
│   │   ├── compare_sac_boundary.py
│   │   ├── diagnose_sac_reach.py
│   │   ├── probe_reach_env.py
│   │   └── smoke_reach_env.py
│   ├── evaluate/
│   │   ├── evaluate_bc_reach.py
│   │   ├── evaluate_ppo_reach.py
│   │   ├── evaluate_reach_baselines.py
│   │   └── evaluate_sac_reach.py
│   ├── models/
│   │   ├── build_dynamics_model.py
│   │   ├── build_grasp_scene.py
│   │   ├── build_gripper_model.py
│   │   └── inspect_model.py
│   ├── plots/
│   │   ├── plot_ppo_training.py
│   │   └── plot_sac_training.py
│   └── train/
│       ├── collect_reach_demonstrations.py
│       ├── train_bc_reach.py
│       ├── train_ppo_reach.py
│       └── train_sac_reach.py
├── src/
│   └── realman_rl_simtoreal/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   └── reach_env.py
│       ├── imitation/
│       │   ├── __init__.py
│       │   ├── bc.py
│       │   └── bc_to_sac.py
│       ├── paths.py
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── common/
│       │   │   ├── __init__.py
│       │   │   ├── artifacts.py
│       │   │   ├── baselines.py
│       │   │   ├── env_factory.py
│       │   │   ├── evaluation.py
│       │   │   ├── metrics.py
│       │   │   └── observation.py
│       │   ├── ppo.py
│       │   ├── sac.py
│       │   ├── sac_diagnostics.py
│       │   └── sac_monitoring.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── control.py
│       │   ├── ee_control.py
│       │   ├── grasp_contact.py
│       │   ├── gripper.py
│       │   ├── kinematics.py
│       │   └── model.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   └── reach.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── demo_helpers.py
│       │   └── geometry.py
│       ├── vendor/
│       │   ├── __init__.py
│       │   ├── realman_fk.py
│       │   └── realman_ik.py
│       └── viewer.py
└── tests/
    ├── fixtures/
    │   └── reach_terminate_04a.json
    ├── test_bc_reach.py
    ├── test_fk_comparison.py
    ├── test_grasp_contact.py
    ├── test_gravity_compensation.py
    ├── test_gripper.py
    ├── test_ik_comparison.py
    ├── test_joint_control.py
    ├── test_kinematics.py
    ├── test_package.py
    ├── test_ppo_reach.py
    ├── test_pure_reach_training.py
    ├── test_reach_env.py
    ├── test_reach_task.py
    ├── test_sac_bc_initialization.py
    ├── test_sac_diagnostics.py
    └── test_sac_reach.py
```

## 2. 文件迁移对照

文件通过工作区重命名完成，保留原内容与 Git 可识别的相似度，没有写入暂存区。当前工具权限将 `.git` 设为只读，因此没有为 `git mv` 的索引写入申请额外权限；用户后续正常暂存时 Git 可以识别重命名。删除旧路径不代表删除功能。

### 全部脚本

| 原文件 | 新文件 | 修改内容 |
|---|---|---|
| `scripts/train_ppo_reach.py` | `scripts/train/train_ppo_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/train_sac_reach.py` | `scripts/train/train_sac_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/train_bc_reach.py` | `scripts/train/train_bc_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/collect_reach_demonstrations.py` | `scripts/train/collect_reach_demonstrations.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/evaluate_ppo_reach.py` | `scripts/evaluate/evaluate_ppo_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/evaluate_sac_reach.py` | `scripts/evaluate/evaluate_sac_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/evaluate_bc_reach.py` | `scripts/evaluate/evaluate_bc_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/evaluate_reach_baselines.py` | `scripts/evaluate/evaluate_reach_baselines.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/view_robot.py` | `scripts/demos/view_robot.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/control_joints.py` | `scripts/demos/control_joints.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/control_ee.py` | `scripts/demos/control_ee.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/demo_gripper.py` | `scripts/demos/demo_gripper.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/test_grasp_contact.py` | `scripts/demos/run_grasp_contact_cases.py` | 重命名为接触案例运行器，区别于同名自动化测试 |
| `scripts/build_dynamics_model.py` | `scripts/models/build_dynamics_model.py` | 移动；保留写入 MJCF 的旧 provenance 文本以保持逐字节资源一致 |
| `scripts/build_gripper_model.py` | `scripts/models/build_gripper_model.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/build_grasp_scene.py` | `scripts/models/build_grasp_scene.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/inspect_model.py` | `scripts/models/inspect_model.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/compare_controllers.py` | `scripts/diagnostics/compare_controllers.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/compare_fk.py` | `scripts/diagnostics/compare_fk.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/compare_ik.py` | `scripts/diagnostics/compare_ik.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/compare_sac_boundary.py` | `scripts/diagnostics/compare_sac_boundary.py` | 同时更新启动 SAC 训练子进程的正式入口路径 |
| `scripts/diagnose_sac_reach.py` | `scripts/diagnostics/diagnose_sac_reach.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/smoke_reach_env.py` | `scripts/diagnostics/smoke_reach_env.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/validate_reach_training.py` | `scripts/diagnostics/probe_reach_env.py` | 重命名为环境 / 奖励探针，明确无网络训练 |
| `scripts/plot_ppo_training.py` | `scripts/plots/plot_ppo_training.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |
| `scripts/plot_sac_training.py` | `scripts/plots/plot_sac_training.py` | 移动并同步包导入 / 测试路径；参数及默认行为不变 |

### 包模块、拆分与删除

下表路径位于 `src/realman_rl_simtoreal/`。

| 原文件 | 新文件 | 修改内容 |
|---|---|---|
| `experiments/ppo_reach.py` | `rl/ppo.py` | 原 PPO、Rollout、裁剪审计、保存加载；公共导入改向 common / tasks |
| `experiments/sac_reach.py` | `rl/sac.py` | 仅保留 SAC_CONFIG、make_model、SAC / Replay 保存 |
| 同上 | `tasks/reach.py` | ENV_CONFIG 与任务参数同处，算法无关 |
| 同上 | `rl/common/observation.py` | ReachTransform、ScaledReachObservation，schema / dtype / 缩放不变 |
| 同上 | `rl/common/env_factory.py` | training_env、make_probe，原包装顺序不变 |
| 同上 | `rl/common/evaluation.py` | REASONS、evaluate、summarize_episodes |
| 同上 | `rl/common/artifacts.py` | write_json、append_json、versions、asset_hashes |
| 同上 | `rl/sac_monitoring.py` | TrainingAudit，与扩展审计集中 |
| 同上 | `rl/sac_diagnostics.py` | q_diagnostic，与原诊断集中 |
| `experiments/sac_monitoring.py` | `rl/sac_monitoring.py` | MonitoredTrainingAudit、probe_q；保留显式 BC 参照的惰性分支 |
| 同上 | `rl/common/evaluation.py` | record_evaluation；PPO 不再借用 SAC 监控模块 |
| `experiments/sac_diagnostics.py` | `rl/sac_diagnostics.py` | 轨迹、指纹、Replay 加载、Q/TD，更新公共导入 |
| `experiments/bc_reach.py` | `imitation/bc.py` | 原 BC 网络 / 训练 / 推理 / 数据及评估 |
| 同上 | `rl/common/observation.py` | FEATURE_KEYS、features，33 维顺序与 float32 转换不变 |
| 同上 | `rl/common/artifacts.py` | empty_directory，复用原禁止覆盖的目录检查 |
| `experiments/sac_initialization.py` | `imitation/bc_to_sac.py` | 原严格 Actor 均值通路迁移与偏差诊断；调用者直接从 common 获取探针 |
| `experiments/reach_tasks.py` | `tasks/reach.py` | TASKS、task_environment；目标分布和默认值不变 |
| 同上 | `rl/common/metrics.py` | MotionStats，只读统计 |
| `experiments/common.py` | `rl/common/baselines.py` | 原公开观测规则策略 |
| 同上 | `utils/geometry.py` | transform、inverse |
| 同上 | `utils/demo_helpers.py` | sample_poses、joint_result_lines、run_joint_demo |
| `experiments/__init__.py` | 删除 | 旧包已无实现 / 调用；无兼容包装 |
| 无 | `tasks/`、`rl/`、`rl/common/`、`imitation/`、`utils/` 的 `__init__.py` | 5 个轻量包标识，不主动导入运行时依赖 |

所有测试留在原 `tests/test_*.py`，同步实际导入、动态脚本加载、mock 位置与隔离检查；没有删除测试或放宽行为断言。特别更新 `test_package.py`、`test_ppo_reach.py`、`test_sac_bc_initialization.py` 和边界对照子进程路径。

### 文档

| 原文档 | 新文档 / 保留位置 | 变化 |
|---|---|---|
| 原 README 全文 | `docs/experiments/README_BEFORE_REFACTOR.md` | 历史正文完整归档，注明旧路径，仅修正归档后的相对链接 |
| 原 architecture 全文 | `docs/experiments/ARCHITECTURE_BEFORE_REFACTOR.md` | 保留历史细节与结论 |
| README.md | 原位置 | 现状、安装、快速开始、训练 / 评估、新路径和限制 |
| docs/architecture.md | 原位置 | 新目录、边界、观测 / 包装契约及 PPO 数据流 |
| 无 | `docs/FILE_MAP.md` | 77 个 Python 文件的职责、层次、主要调用者与入口标记 |
| 无 | `docs/experiments/README.md`、`docs/validation/README.md` | 历史索引和旧路径适用范围 |
| 原 17 份验证报告 | 原路径 | 原文不修改；本次报告另存，未覆盖旧 REFACTOR_VALIDATION.md |
| 无 | 本报告及 `project_organization_checks.json` | 迁移表、实测证据与历史模型清单 |

## 3. 关键架构变化和合理调整

PPO 与 SAC 从 `rl/common` 复用固定观测、训练包装、独立确定性评估、运动统计、JSON 和 TensorBoard 评估标签。算法专属审计与训练更新仍分别保留，未修改官方 loss、RolloutBuffer、Replay 或动作概率。PPO 不再导入任何本项目 SAC 模块，SAC 也不导入 PPO。

观测只存在一份实现：`rl/common/observation.py`。原始 observation 的 27 个字段、两个 3 维 goal、网络 33 维顺序、schema 2 偏移 / 缩放 / dtype 均不变。包装顺序严格为 ReachEnv → ScaledReachObservation → Monitor → DummyVecEnv → VecCheckNan；goal 始终为米，奖励未经归一化。

任务配置与基础 ENV_CONFIG 放在 `tasks/reach.py`，不依赖任何算法；MotionStats 单独放在 common。`record_evaluation` 从 SAC 监控提到共享评估模块，消除原 PPO 对 SAC 的第二条依赖。Q 诊断和 SAC 训练审计分别归入已有诊断 / 监控模块，避免为极少逻辑另建模块。

BC 复用公共工具但不依赖 SAC；历史 Actor 迁移只在 `imitation/bc_to_sac.py`。正常纯 RL 不导入 imitation；显式 BC 监控分支仍惰性导入偏差检查。公共评估的可选 reference_policy 参数仅接收对象，不主动导入 BC。SAC 训练 / BC 转移通过模型对象和官方 SB3 类型校验保持原历史功能。

scripts 保留现有 CLI、训练调度、输出格式、退出码及展示组织，不再把所有入口混在一层。几何与控制演示工具独立于 RL，simulation、envs、vendor 和全部资源无需改动。为保持模型逐字节重生成，生成器输出中的旧 provenance 字符串有意保留；资产内 SOURCE.md 也按历史来源记录原文保留；README 明确其旧命令所属版本并给出当前三个生成入口，路径迁移表逐项对应。其他现行执行命令均指向新路径。

## 4. 测试结果（实际执行）

所有 Python 命令使用 Conda 环境解释器：

```text
/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python
```

完整回归命令在仓库根目录执行：

```bash
python -I -m unittest discover -s tests -v
```

| 阶段 | 通过 | 失败 | 跳过 | 耗时（秒） |
|---|---:|---:|---:|---:|
| 重构前 | 266 | 0 | 0 | 164.929 |
| 阶段 1 | 266 | 0 | 0 | 156.914 |
| 阶段 2 | 266 | 0 | 0 | 166.236 |
| 最终完整回归 | 266 | 0 | 0 | 166.179 |

原始测试日志保存在 `/tmp/realman-refactor-validation/{baseline-tests,phase1-tests,phase2-tests,final-tests}.log`；持久化摘要和逐模型清单保存在本报告及配套 JSON。

基线没有测试失败。阶段 2 额外先运行了 `python -I -m unittest discover -s tests -p test_package.py -v`（10/10）；随后完整运行包含加强后的边界及 CLI 检查。阶段 3 对现行 Markdown 链接、文件索引、命令路径作静态检查，具体计数见配套 JSON。

另执行 `python -I -m compileall -q src scripts tests`、`python -I -m pip check`（No broken requirements found）。原 pyproject 包发现规则正确发现 5 个新包；现有 editable 安装无需重装。隔离进程从其他 cwd 导入核心包、加载资源；没有手工修改 PYTHONPATH、安装或升级依赖。

23 个 argparse 脚本在 `/tmp` 下以绝对路径运行 `--help`，均通过；包含要求的四条命令：

```bash
python -I scripts/train/train_ppo_reach.py --help
python -I scripts/train/train_sac_reach.py --help
python -I scripts/evaluate/evaluate_ppo_reach.py --help
python -I scripts/demos/control_ee.py --help
```

三个无 argparse 的模型生成器由原逐字节重生成测试覆盖，未假称有 `--help` 接口。依赖图测试验证无环、无 scripts 反向导入、无旧 experiments 实现 / 导入、simulation / envs / tasks 边界和 PPO/SAC 隔离。

### 历史模型兼容性

扫描已有 **113 个本地模型文件**。有效模型 **107 个（PPO 17、SAC 87、BC 3）**：按每个保存的环境，以固定种子 88100、88101、88102 reset，并使用固定动作 `[0.1,-0.05,0.02]`、`[-0.05,0.03,0.01]` 获取观测；记录原始观测、变换后观测与确定性动作。前后共 **749 个数组逐元素完全相同**，最大差为 0；配置 / 文件 SHA-256 / 训练计数一致。存在相邻 Replay 的有效 SAC bundle 同时按原协议加载并比较 buffer 大小。未改写任何模型或配置。

BC 仍使用 `weights_only=True`；SB3 使用仓库历史官方类加载，无需旧 experiments 模块别名。没有下载模型，也没有对未知来源模型反序列化。验证脚本与原始数组仅写入 `/tmp/realman-refactor-validation/`，实际命令为：

```bash
python -I /tmp/realman-refactor-validation/checkpoints.py before
python -I /tmp/realman-refactor-validation/checkpoints.py after
```

**6 个历史文件在重构前已有加载限制，未把它们算作兼容通过**；重构后保持同样结果：

| 模型路径 | 重构前及重构后结果 |
|---|---|
| `outputs/rl/sac_reach/smoke_seed2026/checkpoints/step_000160/model.zip` | schema 1：当前 ReachTransform 明确拒绝（Unsupported observation transform） |
| `outputs/rl/sac_reach/smoke_seed2026/final/model.zip` | 缺少相邻 config.json，不能按当前协议评估 |
| `outputs/rl/sac_reach/smoke_verified_seed2026/checkpoints/step_000160/model.zip` | schema 1：当前 ReachTransform 明确拒绝（Unsupported observation transform） |
| `outputs/rl/sac_reach/smoke_verified_seed2026/final/model.zip` | schema 1：当前 ReachTransform 明确拒绝（Unsupported observation transform） |
| `outputs/rl/sac_reach/smoke_seed2026/initial_model.zip` | schema 1：当前 ReachTransform 明确拒绝（Unsupported observation transform） |
| `outputs/rl/sac_reach/smoke_verified_seed2026/initial_model.zip` | schema 1：当前 ReachTransform 明确拒绝（Unsupported observation transform） |

逐模型清单、文件指纹、样本数、Replay / 训练计数见 [机器可读检查结果](project_organization_checks.json)。这不证明未知外部模型或未存储配置的任意旧格式都可兼容。

### 历史模型的新 CLI 实际运行

另外实际运行新入口加载已有模型，输出均位于 `/tmp/realman-refactor-validation/`：

```bash
python -I scripts/evaluate/evaluate_ppo_reach.py --model outputs/rl/ppo_reach/fixed_seed2026_pilot/checkpoints/step_016384/model.zip --episodes 1 --seed-start 88000 --json /tmp/realman-refactor-validation/ppo-evaluation.json
python -I scripts/evaluate/evaluate_sac_reach.py --model outputs/rl/sac_reach/formal_seed2026/final/model.zip --episodes 1 --seed-start 88100 --policies sac --json /tmp/realman-refactor-validation/sac-evaluation.json
python -I scripts/evaluate/evaluate_bc_reach.py --model outputs/bc_reach/fit_seed2026/best_bc.pt --episodes 1 --seed-start 88100 --out /tmp/realman-refactor-validation/bc-evaluation
python -I scripts/plots/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/fixed_seed2026_pilot --out /tmp/realman-refactor-validation/ppo-plots
```

PPO：18 步成功、最终误差 0.9609675836 mm，参数不变；SAC：200 步 timeout、误差 4.0282177124 mm，无安全失败；BC：14 步成功、误差 0.7830662430 mm（配对规则 14 步成功、未训练网络 200 步 timeout）。这些是入口和加载检查的单 Episode 实测，不能概括随机目标整体性能。PPO 图表导出 11 个 PNG，逐一核对文件签名，原日志只读。

### Smoke Test 与无行为改写证据

完整测试套件本身包含实际训练验证，每次完整运行均执行：PPO 单次完整 512 步 Rollout / 10 个 epoch / 80 次 Adam 更新；SAC 小网络 64 步更新与 Replay 保存加载；128 步 SAC / 历史 BC 初始化监控流程；纯 RL 隔离的小规模环境采样。原测试验证 Actor / Critic 参数确实更新、观测 / 奖励 / 动作处理、TensorBoard 标签及 checkpoint 加载。测试输出使用临时目录，不写入已有 outputs。没有额外运行正式训练或超参数搜索。

语法树比较确认迁移的 **72 个函数 / 类 / 常量**除导入外实现完全一致；26 个脚本除导入与入口路径外主体一致，CLI 参数与默认值一致。核心仿真、环境和厂家适配器无代码修改。

已建立并复核 **1118 个文件**的大小与 SHA-256：outputs 1070 个、assets 36 个、simulation / envs / vendor 源文件 12 个。最终复核全部相同，没有新增 / 缺失 / 内容变更；不将源代码哈希因正常模块移动而变化误报为物理参数变化。

### 实际依赖版本

Python 3.11.17；mujoco 3.17.0；numpy 2.4.6；gymnasium 1.4.0；stable-baselines3 2.9.2；torch 2.14.0+cpu；tensorboard 2.21.0；matplotlib 3.11.2；setuptools 84.0.0；realman_rl_simtoreal 0.1.0；Robotic_Arm 1.1.7。全部沿用基线环境。

## 5. 最新运行命令

先 `conda activate realman_rl` 并切换到仓库根目录。下列正式训练命令仅提供给用户手动执行，本次未运行；run 目录必须未使用。GUI 命令需图形桌面。

```bash
# 查看模型
python -I scripts/demos/view_robot.py
# 关节 / 末端 / 夹爪演示；去掉 --viewer 可无界面运行
python -I scripts/demos/control_joints.py --viewer
python -I scripts/demos/control_ee.py --viewer
python -I scripts/demos/demo_gripper.py --viewer
# PPO Reach 正式训练示例
python -I scripts/train/train_ppo_reach.py --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/fixed_seed2026_new --eval-every 2048 --eval-episodes 1 --eval-seed-start 87000 --checkpoint-every 2048 --tensorboard
# 已有 PPO 模型独立评估（另存新 JSON）
python -I scripts/evaluate/evaluate_ppo_reach.py --model outputs/rl/ppo_reach/fixed_seed2026_pilot/checkpoints/step_016384/model.zip --episodes 1 --seed-start 88000 --json /tmp/realman_ppo_evaluation_new.json
# 只读原日志，图片另存
python -I scripts/plots/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/fixed_seed2026_pilot --out /tmp/realman_ppo_plots_new
# SAC Reach 正式训练示例
python -I scripts/train/train_sac_reach.py --task fixed_reach --reward-mode safe_dense --action-scale 0.001 --timesteps 5000 --seed 2026 --init-mode scratch --run-dir outputs/rl/pure_reach/fixed_safe_seed2026_new --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 1 --eval-seed-start 85000 --tensorboard
# 全部自动化测试
python -I -m unittest discover -s tests -v
```

## 6. 剩余问题、历史功能与验证边界

- `rl/common` 当前仍面向 Reach，使用 27+3+3 观测与原 info 契约；未提前抽象为多任务通用训练框架。
- 训练时序仍留在 train 脚本，算法专属 callback 与保存协议各自保留；这有助于保持原更新 / 评估时机，不强行合并。
- 明确选择的历史 BC 参照监控仍有一条惰性调用到 `imitation.bc_to_sac`；普通纯 RL 不导入该模块。BC→SAC、边界实验、失败报告均保留。
- 6 个早期模型的已知加载限制未修复；没有猜测缺失配置，也未改变 schema 1 与 schema 2 的物理语义。
- 保留源码的 editable 安装已验证；普通 wheel 的资产打包、真实图形桌面交互、真机控制、视觉和跨进程完整续训未验证 / 未实现。
- 没有执行新的正式长训练，不能据本次工程回归声称策略学习效果提升或随机目标泛化成功。
- 未来 GraspEnv 建议置于 `envs/grasp_env.py`、任务参数置于 `tasks/grasp.py`，继续复用 simulation 中的夹爪与接触能力；等第二任务实际存在后再提取共性。

本次没有修改、删除或新增已有 outputs 的内容；没有自动 commit 或 push。模型 / 资源与测试详细证据见配套 JSON，现行目录入口见 [FILE_MAP](../FILE_MAP.md)。
