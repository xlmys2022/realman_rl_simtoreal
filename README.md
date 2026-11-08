# realman_rl_simtoreal

基于 MuJoCo 的睿尔曼机械臂仿真项目。长期目标是强化学习控制、抓取及 sim-to-real；当前主动研究纯SAC Reach的奖励、动作尺度与目标难度。模型、控制、离线运动学、夹爪接触、历史SAC及BC研究全部保留；暂停BC与SAC融合。训练流程与策略表现分开验证，最新入口见下文04-E1。

实际模型为官方 **RML63II-6F-V**。它与目标实物 RML63-6F-V 的代际关系尚未核实。静态检查 URDF 保留相机支架转动关节，为 `nq=nv=7, nu=0`；新增动力学 MJCF 将相机刚性固定，为 `nq=nv=nu=6`。

02-C1 已增加厂家 **离线 FK 对照**，结论为当前模型尚不一致：SDK 未提供可确认对应旧版 `6F-V` 的枚举，末端长度与 URDF 近似旋转均有差异。未连接真机，也未据此修改模型、IK 或控制器。完整结果见 [REALMAN_KINEMATICS_VALIDATION.md](docs/validation/REALMAN_KINEMATICS_VALIDATION.md)。

02-C2 已增加厂家 **离线 IK 验证**。II+SF、II+ISFV 的主案例分别通过 36/36 次自模型往返；边界案例分别为 11/12，保留了 J5 负限位附近的初值相关失败。模型不一致结论不变。详见 [REALMAN_IK_VALIDATION.md](docs/validation/REALMAN_IK_VALIDATION.md)。

## 环境

使用已存在的 Conda 环境 `realman_rl`（Python 3.11），不需要新建环境：

```bash
conda activate realman_rl
python -I -m pip install -r /home/mr-beaver/Documents/robot/realman_rl_simtoreal/requirements.txt
python -I -m pip install --no-build-isolation -e /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I -m pip check
```

已验证 MuJoCo 3.17.0、NumPy 2.4.6，所有运行依赖版本见 `requirements.txt`。`-I` 忽略用户包和继承的 `PYTHONPATH`，避免本机 ROS Python 3.14 路径干扰此 Python 3.11 环境。只使用官方 `mujoco`，未安装 `mujoco-py`。

核心功能现在位于标准 src 包中，先完成 editable 安装再运行脚本。上述命令复用当前环境的构建工具；标准写法 `python -I -m pip install -e .` 也使用相同包配置。资源保留在源码仓库，仅支持本次约定的 editable 用法，不提供独立 wheel 资源安装保证。

[工程架构与迁移说明](docs/architecture.md) · [02-D 重构验证](docs/validation/REFACTOR_VALIDATION.md)。旧命令行路径继续可用；Python调用统一使用 `realman_rl_simtoreal.simulation` / `vendor`，不再从scripts导入核心模块。

## 运行

无需图形桌面的检查：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/inspect_model.py
```

在图形桌面的终端运行静态 Viewer：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/view_robot.py
```

左键拖动旋转、右键拖动平移、滚轮缩放；关闭窗口或按 Ctrl+C 退出。可加 `--duration 5` 自动关闭；两个脚本均支持 `--model /absolute/path/model.urdf`。默认资源定位不依赖启动目录。

Viewer 使用官方 `launch_passive()`，不调用 `mj_step()`，机械臂保持 URDF 零位，不因重力塌落。零位未校准为真机零位。Linux 检测到 `DISPLAY` 时默认使用现有 GLFW 的 X11/XWayland 后端；后端选择仅作用于当前进程，可用 `PYGLFW_LIBRARY_VARIANT` 显式覆盖。

本机测试发现 MuJoCo 3.17.0 关闭请求与 GLFW 进程退出之间存在竞态，因此脚本在关闭窗口后等待本次 Viewer 创建的线程完成清理。这一清理方式用于当前无其他后台工作线程的独立脚本。

## 基础关节控制

无界面运行默认验收序列（39 秒仿真时间，实际执行更快），输出每个关节的目标、实际值、误差、超调、末段波动、速度、峰值力矩与接触统计：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_joints.py
```

在 Viewer 中按真实时间观察同一序列：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_joints.py --viewer
```

初始模型关节角为 `[0, -20, 30, 0, 20, 0]` 度；先保持 3 秒，再逐轴增加 3 度并回到初始指令。每次切换用 1 秒平滑过渡。该姿态仅在当前仿真和小幅测试范围内验证，不是真机安全姿态标定。

也可输入六个绝对模型关节角（单位：度，每轴须在初始角的 ±5 度内）：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_joints.py --viewer --target-deg 3 -20 30 0 20 0
```

`--json /absolute/path/results.json` 保存完整数值结果；`--viewer --duration 5` 限制显示时间。提前关闭或 Ctrl+C 正常退出，并明确提示完整测试尚未结束。

控制使用六个官方 `position` 执行器：`ctrl` 为弧度，`tau = kp*(ctrl-q) - kv*qvel`，并有力矩限幅。没有重力补偿或积分项，因此会有小幅重力稳态偏差；默认测试最大末态误差约 0.368°。重力为 −9.81 m/s²，时间步长 0.002 秒，使用 `implicitfast`。完整参数、碰撞诊断及验收结果见 `docs/validation/DYNAMICS_VALIDATION.md` 和模型目录的 `SOURCE.md`。

重新生成 MJCF 和运行持久化的自动测试：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/build_dynamics_model.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests/test_joint_control.py
```

无需新增依赖。生成器复用官方 MuJoCo 导出，再保留双精度的位姿和惯量，避免文本导出舍入引入新的物理参数误差。

## 末端控制与运动学（02-B）

`ee_site` 位于 `link_6` 局部原点，与该连杆同向，随第六连杆运动。它不是相机坐标系，也不代表厂家标定的 TCP。模型生成器会保留此 site。

默认在世界坐标系中，将 home 名义末端位置沿 +X 平移 5 mm，同时将 home 朝向作为 IK 的旋转目标：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --viewer
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --delta-mm 0 0 5
```

也可使用 `--target X Y Z` 输入绝对世界位置（米），例如：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --target 0.010616691 -0.000145159 1.06022729
```

`--target` 与 `--delta-mm` 互斥。`--json /absolute/path/result.json` 保存求解和实际控制结果；`--viewer --duration 5` 可提前停止，但不会报告完整验收通过。Viewer 中绿点是实际 `ee_site`，半透明红球中心是目标点；标记半径为可视化尺寸，不能用球面间距代替数值误差。

运动学模块 `Kinematics(model)` 提供 `fk(q)`、`jacobian(q)` 和 `solve_ik(target_position, initial_q, target_rotation=None)`。关节角单位为弧度；FK 返回世界位置和 site-to-world 的 3×3 旋转矩阵。所有计算使用独立 `MjData`，不修改运行中的仿真状态。默认 IK 同时使用平移/旋转 Jacobian，目标朝向为 home；可显式传入旋转矩阵进行 FK/IK 一致性实验，没有纯位置模式。

IK 成功后，仍需通过 10 mm 位移范围、每轴 home ±5°、位姿复核和 51 点关节路径碰撞预检。符合条件才复用现有关节执行器：先保持 home 3 秒，再用 2 秒平滑关节目标插值及 3 秒保持。正常运动不改写实际 `data.qpos`。IK 失败时不调用 `mj_step()`。

默认实验的 IK 理论误差约 0.00014 mm / 0.000017°，动力学实际误差约 2.39 mm / 0.50°；现有 PD 重力误差仍然存在。固定朝向是 IK 终点约束，关节插值与动力学误差不保证全过程朝向严格不变。验收范围为实际位置误差 ≤5 mm、朝向误差 ≤1°，详见 `docs/validation/KINEMATICS_VALIDATION.md`。

例如 +Y 5 mm 可求解，但 J4 需要变化约 5.37°，超出演示范围，因此拒绝运动。这是演示范围限制，不能据此判定目标物理不可达。局部 IK 失败也不能一概作为全局不可达证明。

运行新增或全部测试：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests/test_kinematics.py
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

IK 解和离散预检均不保证连续轨迹无碰撞；动力学过程中继续检查接触、限位、力矩、有限数值和姿态误差。本次没有障碍物规划、末端闭环误差补偿、真机通信或强化学习。

## 重力补偿对照（02-B+）

任务 02-B+ 的重力误差诊断与 A/B 对照可运行：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_controllers.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_controllers.py --controller pd
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_controllers.py --controller gravity
```

默认使用相同初态、IK 目标、增益、时间、2 ms 步长和限制运行两组：home 保持、世界 ±X 5 mm / +Z 5 mm，以及原来的 13 段关节运动。`--json /absolute/path/results.json` 指定数据文件；默认保存到项目 `outputs/control_comparison.json`。

补偿模式通过 `load_model(gravity_compensation=True)` 在编译前给八个运动连杆/相机 Body 设置 `gravcomp=1`，并给六个 joint 设置 `actuatorgravcomp=true`，使 PD 与补偿合成后再受原关节总力矩限制。缺省 `load_model()` 和已有两个 Demo 仍使用原 PD。原始 URDF、生成的基线 MJCF、质量、惯量、重力与增益都不变；重新生成基线模型不会丢失加载时可选的补偿配置。

必须区分 home 稳态偏移、从稳定 home 出发的相对位移误差和最终绝对误差：基线 +X 5 mm 分别约为 2.210 / 0.211 / 2.392 mm。补偿后的绝对误差约为 0.000140 mm，这是补偿使用与仿真对象完全相同模型参数的理想结果，不能直接当作真机精度。对照表、限幅证据和依赖条件见 `docs/validation/CONTROL_VALIDATION.md`。

`qfrc_bias` 在运动时含重力及速度相关偏置项。诊断程序在独立、零速度的 `MjData` 上计算纯重力力矩；不会把运行中任意 `qfrc_bias` 当作纯重力，也不修改实际 qpos/qvel。补偿的诊断量 `qfrc_gravcomp` 已经路由进总量 `qfrc_actuator`，不能再次相加来计算实际驱动力矩。

运行新增测试可使用 `python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests/test_gravity_compensation.py`；上面的全部测试命令会同时运行原测试和新增测试。

## 文件职责

- `src/realman_rl_simtoreal/envs/reach_env.py`：固定/随机目标、世界坐标增量动作、奖励、Gymnasium API及终止原因；不依赖厂家SDK或脚本。
- `scripts/smoke_reach_env.py`：无界面的固定动作演示及官方环境检查入口。
- `tests/test_reach_env.py`：27项环境、执行器运动、固定周期、安全失败及无SDK验证。
- `scripts/evaluate_reach_baselines.py`：配对目标种子的随机/规则策略、奖励反例及独立性能剖析。
- `simulation/gripper.py`（位于src包）：米制开口、6个夹爪hinge/5条mimic验证、物理推进及限幅审计。
- `scripts/build_gripper_model.py`、`scripts/demo_gripper.py`：可重复转换/实验装配，以及无界面或Viewer开合演示。
- `assets/grippers/eg2_4c2/`：准确型号的官方原文件、清单、兼容模型、组合模型和来源限制。
- `tests/test_reach_task.py`：22项随机目标、奖励一致性、失败处理、策略和API回归验证。
- `src/realman_rl_simtoreal/simulation/model.py`：模型加载、检查、初始化、映射与可选重力补偿配置。
- `src/realman_rl_simtoreal/simulation/control.py`：原有关节控制段、限位/接触检查与总驱动力审计。
- `src/realman_rl_simtoreal/simulation/ee_control.py`：原有末端目标检查、碰撞预检、执行和误差。
- `src/realman_rl_simtoreal/vendor/`：可选的厂家离线FK/IK与配置保护，不自动替换MuJoCo IK。
- `src/realman_rl_simtoreal/experiments/common.py`：共享实验姿态、默认关节序列、SE(3)与纯文本格式。
- `src/realman_rl_simtoreal/paths.py`、`viewer.py`：统一源码资源路径，以及仅供CLI使用的惰性Viewer辅助。
- `pyproject.toml`：src包发现、0.1.0元数据、两份锁定文件动态提供依赖。
- `tests/test_package.py`：隔离导入、无SDK、元数据、依赖图和CLI/Viewer生命周期兼容。
- `scripts/inspect_model.py`：校验来源文件、mesh、结构、物理参数，输出 MuJoCo Body/Joint 及 `nq/nv/nu`，执行 `mj_forward()`，错误时返回非零状态。
- `scripts/view_robot.py`：独立加载模型、静态显示和窗口同步，调用共享线程清理辅助。
- `scripts/build_dynamics_model.py`：从已有兼容 URDF 可重复生成六轴 MJCF，集中定义仿真控制参数。
- `scripts/control_joints.py`：位置控制演示、无界面验收、可选 Viewer 及 JSON 统计。
- `src/realman_rl_simtoreal/simulation/kinematics.py`：独立数据中的 FK、Jacobian、带限位的全位姿 DLS IK 和结构化求解结果。
- `scripts/control_ee.py`：笛卡尔目标CLI、调用核心末端控制、Viewer和实际误差展示。
- `scripts/compare_controllers.py`：匹配条件的 PD/重力补偿对照、独立重力诊断及物理步总驱动力审计。
- `tests/test_joint_control.py`：结构、物理保持、正负小角度运动、映射/限幅及碰撞反例测试。
- `tests/test_kinematics.py`：FK/IK、Jacobian、状态隔离、失败不运动及末端动力学验证。
- `tests/test_gravity_compensation.py`：力通道、合力限幅、多执行器/被动旁路反例和同条件误差对比。
- `assets/robots/rml63/upstream/`：未经修改的官方 URDF、9 个 STL、包元数据和许可证。
- `assets/robots/rml63/rml63_mujoco.urdf`：本地路径及已记录惯量修正的兼容副本，仍直接按 URDF 加载。
- `assets/robots/rml63/rml63_dynamics.xml`：六轴机器人 MJCF、刚性相机、定向碰撞过滤和位置执行器。
- `assets/robots/rml63/scene.xml`：引用机器人并添加灯光和可碰撞地面。
- `assets/robots/rml63/SOURCE.md`：准确型号、固定提交、版本差异、许可证冲突、修改依据及参数限制。
- `assets/robots/rml63/SHA256SUMS`：原始资产校验值。
- `requirements.txt`：本阶段实际使用的依赖锁定版本。
- `.gitignore`：忽略 Python 缓存及临时运行产物，保留模型资产。
- `docs/validation/VALIDATION.md`：实际测试结果、完整检查输出及未验证事项。
- `docs/validation/DYNAMICS_VALIDATION.md`：本次动力学诊断、参数依据、控制测试输出与剩余限制。
- `docs/validation/KINEMATICS_VALIDATION.md`：02-B 的算法、参数、理论与实际误差、完整输出及未解决事项。
- `docs/validation/CONTROL_VALIDATION.md`：02-B+ 重力补偿原理、统一误差定义、对照表、总力矩证据和局限。

## 来源和已知限制

来源为 [RealManRobot/rm_models](https://github.com/RealManRobot/rm_models)，固定提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43`，包版本 `1.0.0`。只获取目标模型必要文件，没有 CAD 或其他型号资源。

原始模型的固定相机连杆惯量存在微小三角不等式偏差，MuJoCo 实测拒绝加载；兼容副本仅修正该项，原始值及推导保留在 `SOURCE.md`，修正值未经厂家确认。上游根许可证与包元数据分别声明 Apache-2.0 和 BSD，尚未消除不一致。

固定模型的数值检查通过不表示已验证真机参数、碰撞几何或动力学。STL 碰撞几何由 MuJoCo 按凸体处理，本阶段未校准；相机和六维力部件也未实现为传感器。URDF effort/velocity 字段不等于 MuJoCo 已配置控制器或限速。

静态 URDF 零位下仍报告 `base_link/link_1` 约 2.73 mm 的装配包络重叠。动力学 MJCF 仅排除该相邻装配对；其他非相邻自碰撞及环境碰撞保留，并有反例测试。相机固定在支架原始零角度，未修改官方原始 URDF 或 STL 尺寸。

后续仍需核实实物版本、零位、惯量、碰撞几何和控制参数。当前已验证小范围关节/末端运动、局部目标环境及独立夹爪接触，并完成首次SAC训练；真机通信未实现，不自动上传 GitHub。

## 最小 Gymnasium 环境（03-A）

现有 editable 安装需要更新新增依赖（仅在 `realman_rl` 中执行）：

```bash
python -I -m pip install --no-build-isolation -e /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/smoke_reach_env.py --check
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/smoke_reach_env.py --action 0 0 0 --steps 10
```

使用 Gymnasium 1.4.0，无需训练框架。默认演示发出5次世界 +X 动作，再保持，目标是名义home位置 +X 5mm。`--json PATH` 保存完整结果；`--action DX DY DZ` 指定恒定归一化动作。

```python
import numpy as np
from realman_rl_simtoreal.envs.reach_env import ReachEnv

with ReachEnv(render_mode="ansi") as env:
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([1, 0, 0], dtype=np.float32))
    print(env.render())
```

动作是 `Box(-1,1,(3,),float32)`，每轴默认缩放为1mm世界平移。动作累积到独立的命令参考，末端目标朝向固定为home。默认启用已有、受总关节力矩限制的重力补偿，可用 `gravity_compensation=False` 做基线实验。每个正常step为50ms，即25个2ms物理步；IK拒绝和安全故障可提前停止，info记录实际步数。

观测为float64的Dict：`observation(27,)`包含q(6)、qvel(6)、实际末端(3)、目标减实际(3)、命令位置(3)、实际ctrl(6)；`achieved_goal(3,)`与`desired_goal(3,)`均为世界坐标米。奖励为负的实际目标距离（m），默认距离≤1mm成功。最多200步；成功或明确安全失败返回terminated，只有超时返回truncated。

复用原IK、51点路径预检和位置执行器；新增 `advance_control_cycle` 替代演示用整段控制，不等待稳态、不重置home。关节指令平均变化率上限10°/s并进行平滑插值，实际速度限制30°/s；仍保留home ±5°和10mm命令区域。正常step不改写实际qpos。

当前仅支持无渲染和`ansi`文本render，没有环境内Viewer/RGB图像；04-A训练由独立脚本实现。官方checker通过，保留无限观测边界及直接构造实例缺少注册spec的提示；文本render、重复close另有测试。03-A固定目标的完整API与历史结果见 [REACH_ENV_VALIDATION.md](docs/validation/REACH_ENV_VALIDATION.md)。03-B后的完整测试为134项（原112项＋新增22项）。

## 随机目标与训练前验证（03-B）

```python
from realman_rl_simtoreal.envs.reach_env import ReachEnv

with ReachEnv(target_mode="random") as env:
    obs, info = env.reset(seed=42)
    # 同一seed可复现；之后reset()不传seed可继续同一随机目标序列。
```

`target_mode="fixed"`是原默认模式，目标、动力学和米制距离奖励保持兼容。`target_mode="random"`默认使用`safe_dense`奖励；固定目标训练实验也应显式选择`reward_mode="safe_dense"`。旧`distance`奖励保留用于回归与诊断，不推荐用于训练，因为提前安全失败可能获得更高累计回报。

默认随机候选是home周围3–5mm的体积均匀球壳，经过原全位姿IK、home ±5°、模型限位及51点路径碰撞检查后接受；仍保持home朝向。只使用Gymnasium的np_random，每次最多64次尝试，耗尽则明确报错。`target_radius`最多10mm，未扩大原控制实验区域。

安全奖励为 `-min(distance / 0.020, 1) - (max_episode_steps + 1) * is_safety_failure`，无量纲。每步普通成本在[-1,0]；未折扣安全Episode总回报≥-H，安全失败总回报≤-(H+1)，避免提前失败的回报捷径。`compute_reward`在此模式下必须收到含布尔`is_safety_failure`的info，支持批量调用；不能复用旧目标的成功标记计算新目标奖励。

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/evaluate_reach_baselines.py --episodes 100 --profile
```

默认JSON输出到仓库`outputs/reach_baselines.json`，可用`--json PATH`指定；`--profile`另跑10个规则Episode，仅剖析step，不污染主要计时。两策略使用相同目标种子0–99。随机策略均匀采样整个动作Box；规则策略根据目标误差减去待执行命令量，并将动作向量范数限制为0.25，避免动态滞后造成命令过度累积。两者都只通过env.step运动，动作幅度分布不同是策略设计的一部分。

本次成功率为随机11/100、规则100/100；平均Episode长度53.66/14.54步，平均回报-192.60246/-1.88148。两组无碰撞和数值失败。实测step均值约15ms，含reset和策略检查的交互速度约65–67步/秒，不按50ms仿真周期推算。命令边界、关节限制、IK和物理安全失败仍终止，超时截断；没有投影非法IK解或放宽安全阈值。

03-B验证为受限局部训练准备了基线；04-A已进行首次SAC实验（见后文），尚未实现HER。未来HER需同步重算27维观测的目标相对字段、距离和成功终止条件，并保存独立安全标记。完整目标分布、失败统计、奖励推导、性能与限制见 [REACH_TASK_VALIDATION.md](docs/validation/REACH_TASK_VALIDATION.md)。

## EG2-4C2夹爪（03-C1）

官方URDF明确名为EG2-4C2，原文件及固定提交、校验值和许可证保存在 [夹爪来源](assets/grippers/eg2_4c2/SOURCE.md)。它包含6个转动关节和5条mimic耦合，使用一个位置执行器真实推进物理开合；原机械臂六个执行器和ee_site不变。组合模型为nq=nv=12、nu=7、neq=5，不能按整个模型只有6自由度处理。

**当前是实验装配和未标定动力学。** 上游质量仅94.2992g（产品参考231g），关节限位对应开口66.217357mm（产品参考70mm）；没有修改尺寸、限位或补齐猜测质量。额外8mm/11g安装板为图纸尺寸的圆柱代理，同轴贴合方向/深度及6F-V附件栈尚未实物核实。相机和原末端几何完整保留。新grasp_site是名义抓取参考，不是标定TCP，实际夹指中点还随开口高度变化。

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/build_gripper_model.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/demo_gripper.py --cycles 2
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/demo_gripper.py --viewer --cycles 1
```

Viewer用末端近景观察夹指；可用`--duration 5`提前退出，未完成时明确提示。`--json PATH`记录完整验收结果。无界面两轮开合、带载home保持、两夹指与外物接触力反例以及原134项回归均已验证；当前共153项测试。完整记录见 [GRIPPER_VALIDATION.md](docs/validation/GRIPPER_VALIDATION.md)。

```python
from realman_rl_simtoreal.paths import GRIPPER_SCENE
from realman_rl_simtoreal.simulation.model import load_model, initialize
from realman_rl_simtoreal.simulation.gripper import Gripper

model, arm_joint_ids, arm_actuator_ids = load_model(GRIPPER_SCENE, gravity_compensation=True)
data = initialize(model)
gripper = Gripper(model, data)
gripper.set_width(0.030)  # m；仅选择目标，不瞬移也不立即改ctrl
gripper.step(1500)        # 3秒物理时间，原六轴保持已有ctrl
print(gripper.get_width())
gripper.close()           # 之后继续step才产生运动；open()选择最大模型开口
```

独立`eg2_4c2.xml`默认带受限幅的重力补偿；组合模型用上面的load_model选项开启。低增益位置伺服不代表厂家20N夹持力，当前未验证不补偿夹爪的跟踪精度。Gripper.step和原关节控制函数都会推进共享物理时间，03-C2同步控制请使用下文的advance_synchronized_cycle，不能把两者顺序调用当作一次物理步。

ReachEnv的模型、动作、奖励及默认资源保持不变；本次未加入物体抓取、Pick环境、力控、真机通信或训练。

## 厂家离线 FK 对照（02-C1，可选 SDK）

基础仿真不需要厂家 SDK。只在 `realman_rl` 中先预检、再安装独立锁定的可选依赖；本次安装仅新增 `Robotic_Arm==1.1.7`，MuJoCo/NumPy 版本未改：

```bash
conda activate realman_rl
python -I -m pip --no-cache-dir install --dry-run -r /home/mr-beaver/Documents/robot/realman_rl_simtoreal/requirements-vendor.txt
python -I -m pip --no-cache-dir install -r /home/mr-beaver/Documents/robot/realman_rl_simtoreal/requirements-vendor.txt
python -I -m pip --no-cache-dir check
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_fk.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_fk.py --generation all --sensor all
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

默认 `II + SF` 是依据官方六维力定义选择的**候选**，不是确认的 `6F-V` 配置；`--sensor ISFV` 可查看新版一体化六维力加视觉候选。`--generation II|III|all`、`--sensor B|ZF|SF|ISF|BV|ISFV|all` 支持多配置调查，不会替换 MuJoCo 机器人。没有厂家 IK 或真机接口调用。

脚本计算 home、六轴分别 +3°、组合和另一姿态，共 9 组 FK；只使用独立计算数据，不执行 `mj_step()`。输出位置与 SO(3) 旋转误差；完整 4×4 变换（含 3×3 旋转）、DH、坐标系、原始及相对误差保存在默认 `outputs/fk_comparison.json`，可用 `--json PATH` 覆盖。**退出码 0：数值一致，2：已完成测量但模型不一致，1：运行或 SDK 错误**。数值一致仍不代表实物/TCP 已标定。SDK 缺失时 9 项可选厂家测试跳过，基础仿真不受影响。

## 厂家离线 IK 验证（02-C2）

沿用可选 SDK，不新增依赖。两个候选配置顺序运行；每套 IK 分别使用自己的 FK 生成目标，原有 MuJoCo DLS、模型和控制器保持不变：

```bash
conda activate realman_rl
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_ik.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_ik.py --sensor SF --repeats 5
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_ik.py --sensor ISFV --repeats 5
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

主集是 9 个姿态 × 4 个初值，默认预热后每例计时 5 次；另测逐轴上下限附近、全解、连续小变化、不可达目标和独立的模型交叉诊断。`--repeats` 范围 1–20；完整结果默认保存到 `outputs/ik_comparison.json`，可用 `--json PATH` 指定位置。02-D时为85项测试，03-A新增27项后为112项；缺少SDK时17项厂家IK测试及9项厂家FK测试跳过。

`realman_rl_simtoreal.vendor.realman_ik` 的 `RealManIK(model_limits, model="II", sensor="SF")` 复用 FK 会话，提供 `solve_ik(position_m, initial_q_rad, rotation_3x3)` 和 `solve_all(...)`。内部使用单位 wxyz 四元数、degree 关节输入，并检查返回码、六轴数量、有限值、双方限位和厂家 FK 残差。只有检查通过时才返回弧度 `q`，失败返回 `q=None`；全解仅允许整数 `2*pi` 等价表示转换，不做关节裁剪或模型零偏修正。

厂家 Algo 使用进程全局配置，不支持并发会话；新建另一配置后旧会话会明确失败。全解是独立诊断，不会自动替代单解失败。当前 J5 边界失败和 MuJoCo 远初值局部失败均保留。

退出码 0 表示实验完成且数值/失败处理审计通过，**不表示所有案例求解成功或两模型一致**；1 表示运行错误，2 表示重复性、成功解复核等审计异常。本程序不调用执行器或 `mj_step()`，不连接真机。IK 成功不保证运动路径无碰撞，厂家关节解不能作为当前 MuJoCo 控制器的可直接替换输入。

## 动态方块接触与夹持（03-C2）

独立场景 `assets/scenes/grasp_contact.xml` 增加20mm、10g的freejoint方块及倾斜支撑台，原机械臂/夹爪资产不变。指垫与目标的碰撞使用从原STL齿顶和凹槽底面提取的薄层代理，保留视觉网格及机器人/支撑台碰撞。组合场景nq=19、nv=18、nu=7；物体不补偿重力，也没有固定到夹爪。

```bash
conda activate realman_rl
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/build_grasp_scene.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/test_grasp_contact.py --case normal
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/test_grasp_contact.py --case release --viewer
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/test_grasp_contact.py --case all
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

`--case`还支持`no_close`、`low_friction`、`heavy`、`hull`；`--no-lift`仅检查台面接触。`--json PATH`输出逐2ms标量及20Hz详细接触记录，默认写入Git忽略的`outputs/grasp_contact/results.json`。Viewer可用`--duration SECONDS`提前停止，结果会标记未完成。正常完成返回0；安全保护触发返回1。因此包含已知低摩擦/30g失败案例的`--case all`当前预期返回1，不能当作全案例成功。

10g、指垫摩擦系数0.7时已验证真实接触、5mm法向抬升、2秒离台保持及降近台面后释放；不闭合时方块仍由台面支撑。低摩擦0.05和30g案例触发原mimic/速度保护，未放宽限制。当前177项测试通过，其中负例测试验证安全停止，**不代表所有物理案例夹持成功**。质量、安装、传动和控制均未实物标定，结果不代表真实EG2-4C2性能。

同步控制使用`simulation.control.advance_synchronized_cycle`：同一周期更新机械臂和夹爪ctrl，再推进25个2ms物理步，总计50ms；不能顺序调用两个各推进25步的接口来替代。ReachEnv保持原接口。完整参数、接触力、失败过程、Viewer及局限见 [GRASP_CONTACT_VALIDATION.md](docs/validation/GRASP_CONTACT_VALIDATION.md)。

## SAC Reach训练（04-A，可选CPU依赖）

训练仅使用原六轴`ReachEnv`，不加载夹爪场景。环境、IK、奖励和安全规则不变。新增SB3/PyTorch独立可选依赖，基础仿真仍无需它们：

```bash
conda activate realman_rl
python -I -m pip install --dry-run --extra-index-url https://download.pytorch.org/whl/cpu -c requirements.txt -r requirements-rl.txt
python -I -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -c requirements.txt -r requirements-rl.txt
python -I -m pip install --no-deps --no-build-isolation -e .
python -I -m pip check
python -I scripts/train_sac_reach.py --run-dir outputs/rl/sac_reach/my_pilot --timesteps 5000 --label pilot
```

采用官方`SAC("MultiInputPolicy", ...)`、CPU单线程、Actor及双Critic各两层256/ReLU。27维observation按固定home和原实验范围缩放；两个goal字段保留米制以保持compute_reward接口，SB3在网络入口转为float32。原safe_dense奖励不缩放、不修改。所有偏移/尺度及环境/训练参数保存在config.json和observation_transform.json，无在线统计需要恢复。

每5,000步用独立种子10000–10009评估并保存checkpoint，调用间在同一进程保留模型、Replay Buffer和环境状态。每次CLI启动都是独立实验，已有非空目录会拒绝覆盖；本次不提供隐式续训。正式30,000步命令须在Pilot检查后运行：

```bash
python -I scripts/train_sac_reach.py --run-dir outputs/rl/sac_reach/my_formal --timesteps 30000 --label formal
python -I scripts/evaluate_sac_reach.py --model outputs/rl/sac_reach/my_formal/final/model.zip --episodes 100 --seed-start 100
```

最终模型旁必须保留config.json/observation_transform.json；评估脚本会使用保存的同一预处理、deterministic=True和独立环境，按相同100个目标对比随机、原规则及SAC策略。final/replay_buffer.pkl.gz为官方SB3 Replay Buffer的无损压缩存档，独立推理不需要读取它。训练日志、网络、配置和依赖快照全部放在已被Git忽略的outputs/rl/sac_reach下，不上传大文件。

本次实验结果、失败分类、奖励尺度与Critic/Q值分析见 [SAC_REACH_VALIDATION.md](docs/validation/SAC_REACH_VALIDATION.md)。少量步数测试只验接口和更新，不要求策略成功；训练回报和任务成功率分别报告。

本次固定seed首次正式训练30,000步（独立于5k Pilot），实际29,900次更新，CPU约37.17步/秒。独立种子100–199：随机4/100、规则100/100、未训练SAC2/100、最终SAC1/100。最终SAC99次超时、无安全失败，平均末端误差6.713mm，**尚未学得可靠Reach策略**；回报改善主要伴随失败减少。保留了Critic乐观偏差和指令往复的证据，未加步数或修改奖励制造成功。原177加新增12项测试共189项通过；详细结果见上述验证报告。已保存产物为`outputs/rl/sac_reach/formal_seed2026/final/model.zip`及相邻配置。

## SAC诊断与边界对照（04-B）

原04-A模型、Replay Buffer、配置和失败报告全部保留。默认`boundary_mode="terminate"`与原行为一致；显式选择`reject_and_hold`时，仅拒绝超出10mm命令区域或home ±5°的候选命令，保持最近合法命令并继续一个50ms物理周期。非法模型限位/IK解、实际碰撞、速度、总驱动力、姿态及数值异常仍终止。拒绝不裁剪IK解、不修改实际qpos；默认增加1个普通距离成本单位，`compute_reward`通过info中的`is_action_rejected`标量/批量重算。

只读重跑原30k诊断，或复现独立10k边界实验（输出目录必须为空，后者会从头训练）：

```bash
conda activate realman_rl
python -I scripts/diagnose_sac_reach.py --out outputs/rl/sac_boundary/new_diagnosis
python -I scripts/compare_sac_boundary.py --out outputs/rl/sac_boundary/new_seed2026
```

已有B模型的最终100目标评估无需训练：

```bash
python -I scripts/compare_sac_boundary.py --final --b outputs/rl/sac_boundary/controlled_seed2026/training/final/model.zip --out outputs/rl/sac_boundary/new_final_evaluation
python -I -m unittest discover -s tests -v
```

原30k在20个开发目标上，动作朝实际目标的比例仅50.5%，相邻反向41.0%；全Replay中安全/成功终止样本的soft TD残差很大，未发现SB3接错终止观测或错误屏蔽timeout。新B从头训练10k、CPU约37.96步/秒；同预算A/B最终100目标成功率0%/1%，平均距离7.458/3.229mm，均无安全失败或拒绝。B降低了误差和往复，但仍不能可靠Reach；开发集均0/20，按预定标准没有追加第二seed或更多步数。原189加新增18项测试共207项通过。完整轨迹、TD公式/数组、奖励反例、限制及运行命令见 [SAC_DIAGNOSTICS_VALIDATION.md](docs/validation/SAC_DIAGNOSTICS_VALIDATION.md)。

## Reach行为克隆（04-C）

独立PyTorch BC网络使用与SAC确定性均值通路相同的33→256/ReLU→256/ReLU→3/tanh容量，拟合原规则策略动作。它没有Critic、Replay或强化学习更新，不是SAC预训练。直接复用04-A固定观测变换，默认terminate环境、奖励和安全检查均不变。

```bash
conda activate realman_rl
python -I scripts/collect_reach_demonstrations.py --out outputs/bc_reach/new_demonstrations
python -I scripts/train_bc_reach.py --data outputs/bc_reach/new_demonstrations/demonstrations.npz --out outputs/bc_reach/new_fit
python -I scripts/evaluate_bc_reach.py --model outputs/bc_reach/new_fit/best_bc.pt --out outputs/bc_reach/new_evaluation
```

采集默认300个专家Episode（240训练/60验证），按完整Episode划分；训练300个epoch，按独立监督验证MSE选模型。最终种子20000–20099保留用于规则、未训练和BC三组配对闭环，神经分支只输出网络动作，不调用专家修正。已有非空输出目录拒绝覆盖。

本轮4447个示范样本全部为移动动作，验证动作MSE约3.67e-6、方向余弦.999954。最终100个新目标：BC100/100、规则100/100、未训练1/100；BC平均误差.86449mm、成功平均15.15步，无安全失败或超时。支持当前网络容量能够表示有效的局部Reach策略，但没有近零停止样本，不能宣称已学会停车/持续保持，也不能据此断定SAC失败的唯一原因。模型和数据保存在Git忽略的`outputs/bc_reach/`，加载模型只需`experiments.bc_reach.load_policy`，无需原数据或专家。完整闭环结果、局限及全部222项测试记录见 [BC_REACH_VALIDATION.md](docs/validation/BC_REACH_VALIDATION.md)。未追加SAC训练。

## BC初始化SAC与监控（04-D1，历史复现；当前暂停）

训练入口默认仍为scratch；显式`--init-mode bc_actor --bc-model PATH`严格校验原33维特征、预处理、激活与动作映射，只复制latent_pi和mu，保留原log_std、随机双Critic和空Replay。缺失/不兼容BC模型报错，不回退。`--bc-model`也可在scratch模式中仅作冻结监控参照。

新增`--tensorboard`、`--checkpoint-every`、`--eval-at-start`；bc_actor自动保存/评估第0步。评估与checkpoint可独立调度，输出独立评估成功率/距离/步数/故障、训练Episode统计、Actor/Critic loss、温度、拒绝率、BC动作偏差及Q/TD摘要。TensorBoard2.21.0、Matplotlib3.11.2和新增运行依赖已锁在可选requirements-rl.txt，既有物理依赖版本不变。

仅已运行128步Smoke Test和必要测试。第0步新20目标20/20、.879mm，动作最大差5.96e-8；另3个配对目标在128步后从3/3退为1/3，记录2次关节命令保护失败，没有继续训练或添加补救机制。全部234项测试通过；这不是正式SAC训练成功结果。

用户手动A/B的完整10k命令、TensorBoard、JSON查看、独立加载评估及PNG导出命令见 [SAC_BC_INIT_VALIDATION.md](docs/validation/SAC_BC_INIT_VALIDATION.md)。本次未执行这些正式训练命令。已有Smoke可只读检查：

```bash
python -I -m tensorboard.main --inspect --logdir outputs/rl/sac_bc_init_validation/smoke_bc_seed2026/sb3
python -I -m json.tool outputs/rl/sac_bc_init_validation/smoke_bc_seed2026/evaluations.json
python -I scripts/plot_sac_training.py --run-dir outputs/rl/sac_bc_init_validation/smoke_bc_seed2026 --out outputs/rl/sac_bc_init_validation/new_smoke_plots
```

绘图仅读JSON，输出成功率、距离、Actor/Critic分图及温度四个PNG；缺评估时不造曲线。模型/配置/日志位于独立新run，原04-A/B/C产物不覆盖。日志可供长期训练监控，但不提供跨进程完整续训；若BC明显退化，先停止并分析保存的更新证据。

## 纯SAC Reach与渐进任务（04-E1，当前方向）

复用同一个ReachEnv，显式`--task fixed_reach|easy_random_reach|random_reach`分别选择原+X5mm固定目标、2–3mm三维可达球壳、原3–5mm三维可达球壳。新任务强制随机scratch、空Replay，拒绝BC初始化和BC参照；不改变物理模型、IK、执行器、安全规则或33维预处理。

`--reward-mode safe_dense`保留旧奖励；新可选`reach_shaped_v1`使用指数距离势的折扣差，配合推导出的成功/失败终值及有限拒绝成本，保持compute_reward与step一致。它不是默认奖励，也不是训练效果已验证的方案。真实轨迹已通过“成功>静止超时>反复拒绝>故意失败”的折扣排序，并保留直接负指数+小失败惩罚的反例。

先由用户手动进行固定目标奖励对照，各自从头5000步；两条命令只改变奖励及输出目录，当前1mm尺度保持一致：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I scripts/train_sac_reach.py --task fixed_reach --reward-mode safe_dense --action-scale 0.001 --run-dir outputs/rl/pure_reach/fixed_safe_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 1 --eval-seed-start 85000 --tensorboard
python -I scripts/train_sac_reach.py --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.001 --run-dir outputs/rl/pure_reach/fixed_shaped_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 1 --eval-seed-start 85000 --tensorboard
```

固定目标评估1集只说明同一固定起点/目标的表现，不冒充随机目标泛化。奖励对照后再单独研究`--action-scale 0.00025`；固定任务未稳定成功时，不继续扩大至随机目标。阶段B/C应保留前阶段选定奖励与尺度，首次均从头训练，不自动迁移参数。

继续使用TensorBoard、独立评估、checkpoint及PNG；新增实际位移、动作相对实际/命令目标的方向比例及路径跟踪统计。纯RL进程测试阻断BC/厂家模块及专家动作调用，训练仍正常执行。开发验收仅运行128步Smoke；用户随后手动完成两组5000步固定目标Pilot，所有独立评估均未成功。新奖励最终200步超时、误差7.448mm，动作主要趋于静止；旧奖励最终36步命令越界、误差12.587mm，两者运动时长不同。暂不扩大至随机目标或30000步。配置核对、完整结果和下一步边界已追加到 [PURE_REACH_VALIDATION.md](docs/validation/PURE_REACH_VALIDATION.md)。
