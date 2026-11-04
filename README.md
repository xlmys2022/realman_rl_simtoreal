# realman_rl_simtoreal

基于 MuJoCo 的睿尔曼机械臂仿真项目。长期目标是强化学习控制、抓取及 sim-to-real；当前仅完成第一阶段的模型导入、结构检查和静态可视化。

实际模型为官方 **RML63II-6F-V**。它与目标实物 RML63-6F-V 的代际关系尚未核实。六个机械臂关节之外还保留一个相机支架转动关节，因此 `nq=nv=7`；未配置执行器，`nu=0`。

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

## 文件职责

- `scripts/inspect_model.py`：校验来源文件、mesh、结构、物理参数，输出 MuJoCo Body/Joint 及 `nq/nv/nu`，执行 `mj_forward()`，错误时返回非零状态。
- `scripts/view_robot.py`：独立加载模型、静态显示、同步窗口和清理 Viewer 线程。
- `assets/robots/rml63/upstream/`：未经修改的官方 URDF、9 个 STL、包元数据和许可证。
- `assets/robots/rml63/rml63_mujoco.urdf`：本地路径及已记录惯量修正的兼容副本，仍直接按 URDF 加载。
- `assets/robots/rml63/SOURCE.md`：准确型号、固定提交、版本差异、许可证冲突、修改依据及参数限制。
- `assets/robots/rml63/SHA256SUMS`：原始资产校验值。
- `requirements.txt`：本阶段实际使用的依赖锁定版本。
- `.gitignore`：忽略 Python 缓存及临时运行产物，保留模型资产。
- `VALIDATION.md`：实际测试结果、完整检查输出及未验证事项。

## 来源和已知限制

来源为 [RealManRobot/rm_models](https://github.com/RealManRobot/rm_models)，固定提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43`，包版本 `1.0.0`。只获取目标模型必要文件，没有 CAD 或其他型号资源。

原始模型的固定相机连杆惯量存在微小三角不等式偏差，MuJoCo 实测拒绝加载；兼容副本仅修正该项，原始值及推导保留在 `SOURCE.md`，修正值未经厂家确认。上游根许可证与包元数据分别声明 Apache-2.0 和 BSD，尚未消除不一致。

固定模型的数值检查通过不表示已验证真机参数、碰撞几何或动力学。STL 碰撞几何由 MuJoCo 按凸体处理，本阶段未校准；相机和六维力部件也未实现为传感器。URDF effort/velocity 字段不等于 MuJoCo 已配置控制器或限速。

零位下，MuJoCo 检测到 `base_link` 与 `link_1` 的两个约 2.73 mm 穿透接触。检查脚本持续报告该警告；本阶段未修改碰撞几何或关闭碰撞检测来掩盖它。

后续首先核实实物版本、零位、惯量和碰撞几何，再独立规划控制、任务环境与强化学习。本次不实现这些后续功能，不自动上传 GitHub。
