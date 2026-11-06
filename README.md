# realman_rl_simtoreal

基于 MuJoCo 的睿尔曼机械臂仿真项目。长期目标是强化学习控制、抓取及 sim-to-real；当前已完成模型、控制、离线运动学验证，以及固定/随机局部目标的 Gymnasium reaching 环境和训练前基线评估。尚未实现强化学习算法或训练。

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

后续仍需核实实物版本、零位、惯量、碰撞几何和控制参数。当前只验证小范围关节/末端运动及局部目标环境；未实现强化学习、夹爪或真机通信，不自动上传 GitHub。

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

当前仅支持无渲染和`ansi`文本render，没有环境内Viewer/RGB图像或正式训练。官方checker通过，保留无限观测边界及直接构造实例缺少注册spec的提示；文本render、重复close另有测试。03-A固定目标的完整API与历史结果见 [REACH_ENV_VALIDATION.md](docs/validation/REACH_ENV_VALIDATION.md)。03-B后的完整测试为134项（原112项＋新增22项）。

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

可开始受限局部任务的小规模训练实验，但尚未训练或实现HER。未来HER需同步重算27维观测的目标相对字段、距离和成功终止条件，并保存独立安全标记。完整目标分布、失败统计、奖励推导、性能与限制见 [REACH_TASK_VALIDATION.md](docs/validation/REACH_TASK_VALIDATION.md)。

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
