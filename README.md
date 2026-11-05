# realman_rl_simtoreal

基于 MuJoCo 的睿尔曼机械臂仿真项目。长期目标是强化学习控制、抓取及 sim-to-real；当前已完成模型导入、静态检查、基础六关节位置控制，以及固定末端朝向的数值逆运动学和小幅末端控制。

实际模型为官方 **RML63II-6F-V**。它与目标实物 RML63-6F-V 的代际关系尚未核实。静态检查 URDF 保留相机支架转动关节，为 `nq=nv=7, nu=0`；新增动力学 MJCF 将相机刚性固定，为 `nq=nv=nu=6`。

## 环境

使用已存在的 Conda 环境 `realman_rl`（Python 3.11），不需要新建环境：

```bash
conda activate realman_rl
python -I -m pip install -r /home/mr-beaver/Documents/robot/realman_rl_simtoreal/requirements.txt
python -I -m pip check
```

已验证 MuJoCo 3.17.0、NumPy 2.4.6，所有运行依赖版本见 `requirements.txt`。`-I` 忽略用户包和继承的 `PYTHONPATH`，避免本机 ROS Python 3.14 路径干扰此 Python 3.11 环境。只使用官方 `mujoco`，未安装 `mujoco-py`。

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

控制使用六个官方 `position` 执行器：`ctrl` 为弧度，`tau = kp*(ctrl-q) - kv*qvel`，并有力矩限幅。没有重力补偿或积分项，因此会有小幅重力稳态偏差；默认测试最大末态误差约 0.368°。重力为 −9.81 m/s²，时间步长 0.002 秒，使用 `implicitfast`。完整参数、碰撞诊断及验收结果见 `DYNAMICS_VALIDATION.md` 和模型目录的 `SOURCE.md`。

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

默认实验的 IK 理论误差约 0.00014 mm / 0.000017°，动力学实际误差约 2.39 mm / 0.50°；现有 PD 重力误差仍然存在。固定朝向是 IK 终点约束，关节插值与动力学误差不保证全过程朝向严格不变。验收范围为实际位置误差 ≤5 mm、朝向误差 ≤1°，详见 `KINEMATICS_VALIDATION.md`。

例如 +Y 5 mm 可求解，但 J4 需要变化约 5.37°，超出演示范围，因此拒绝运动。这是演示范围限制，不能据此判定目标物理不可达。局部 IK 失败也不能一概作为全局不可达证明。

运行新增或全部测试：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests/test_kinematics.py
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

IK 解和离散预检均不保证连续轨迹无碰撞；动力学过程中继续检查接触、限位、力矩、有限数值和姿态误差。本次没有障碍物规划、末端闭环误差补偿、真机通信或强化学习。

## 文件职责

- `scripts/inspect_model.py`：校验来源文件、mesh、结构、物理参数，输出 MuJoCo Body/Joint 及 `nq/nv/nu`，执行 `mj_forward()`，错误时返回非零状态。
- `scripts/view_robot.py`：独立加载模型、静态显示、同步窗口和清理 Viewer 线程。
- `scripts/build_dynamics_model.py`：从已有兼容 URDF 可重复生成六轴 MJCF，集中定义仿真控制参数。
- `scripts/control_joints.py`：位置控制演示、无界面验收、可选 Viewer 及 JSON 统计。
- `scripts/kinematics.py`：独立数据中的 FK、Jacobian、带限位的全位姿 DLS IK 和结构化求解结果。
- `scripts/control_ee.py`：笛卡尔目标输入、IK 执行前检查、现有执行器运动和实际末端误差统计。
- `tests/test_joint_control.py`：结构、物理保持、正负小角度运动、映射/限幅及碰撞反例测试。
- `tests/test_kinematics.py`：FK/IK、Jacobian、状态隔离、失败不运动及末端动力学验证。
- `assets/robots/rml63/upstream/`：未经修改的官方 URDF、9 个 STL、包元数据和许可证。
- `assets/robots/rml63/rml63_mujoco.urdf`：本地路径及已记录惯量修正的兼容副本，仍直接按 URDF 加载。
- `assets/robots/rml63/rml63_dynamics.xml`：六轴机器人 MJCF、刚性相机、定向碰撞过滤和位置执行器。
- `assets/robots/rml63/scene.xml`：引用机器人并添加灯光和可碰撞地面。
- `assets/robots/rml63/SOURCE.md`：准确型号、固定提交、版本差异、许可证冲突、修改依据及参数限制。
- `assets/robots/rml63/SHA256SUMS`：原始资产校验值。
- `requirements.txt`：本阶段实际使用的依赖锁定版本。
- `.gitignore`：忽略 Python 缓存及临时运行产物，保留模型资产。
- `VALIDATION.md`：实际测试结果、完整检查输出及未验证事项。
- `DYNAMICS_VALIDATION.md`：本次动力学诊断、参数依据、控制测试输出与剩余限制。
- `KINEMATICS_VALIDATION.md`：02-B 的算法、参数、理论与实际误差、完整输出及未解决事项。

## 来源和已知限制

来源为 [RealManRobot/rm_models](https://github.com/RealManRobot/rm_models)，固定提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43`，包版本 `1.0.0`。只获取目标模型必要文件，没有 CAD 或其他型号资源。

原始模型的固定相机连杆惯量存在微小三角不等式偏差，MuJoCo 实测拒绝加载；兼容副本仅修正该项，原始值及推导保留在 `SOURCE.md`，修正值未经厂家确认。上游根许可证与包元数据分别声明 Apache-2.0 和 BSD，尚未消除不一致。

固定模型的数值检查通过不表示已验证真机参数、碰撞几何或动力学。STL 碰撞几何由 MuJoCo 按凸体处理，本阶段未校准；相机和六维力部件也未实现为传感器。URDF effort/velocity 字段不等于 MuJoCo 已配置控制器或限速。

静态 URDF 零位下仍报告 `base_link/link_1` 约 2.73 mm 的装配包络重叠。动力学 MJCF 仅排除该相邻装配对；其他非相邻自碰撞及环境碰撞保留，并有反例测试。相机固定在支架原始零角度，未修改官方原始 URDF 或 STL 尺寸。

后续仍需核实实物版本、零位、惯量、碰撞几何和控制参数。当前只验证小范围关节/末端运动；未实现任务环境、强化学习、夹爪或真机通信，不自动上传 GitHub。
