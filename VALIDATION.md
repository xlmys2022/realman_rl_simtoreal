# 第一阶段实际验证记录

验证日期：2026-11-04（Asia/Shanghai）。所有 Python 安装和验证均在现有 Conda `realman_rl` 环境中完成；使用 `python -I` 隔离用户包与继承的 ROS `PYTHONPATH`。未修改其他 Conda 环境、相邻项目或系统 Python，未执行 Git commit/push。

## 环境和依赖

- Python：3.11.17。
- 解释器：`/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python`。
- MuJoCo 和 NumPy 的实际导入位置均位于该环境的 `lib/python3.11/site-packages/`。
- `conda run -n realman_rl python -I -m pip check`：`No broken requirements found.`，退出码 0。
- 下列每项精确版本均通过 `importlib.metadata.version()` 与 `requirements.txt` 比对。pip、setuptools、wheel、packaging 是原环境已有工具，不是本次新增项目依赖。

```text
# Direct dependencies, verified in Python 3.11 / realman_rl.
mujoco==3.17.0
numpy==2.4.6
glfw==2.11.0

# Required runtime dependencies (including etils[epath] dependencies).
absl-py==2.5.1
etils==1.15.0
fsspec==2026.9.0
PyOpenGL==3.1.10
typing_extensions==4.16.0
websockets==17.4
zipp==4.1.2

```

## 模型加载与兼容修改

实际来源为官方 `RML63II-6F-V`，固定提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43`。12 个原始文件下载后均与该提交的 Git blob 校验值一致。模型的 9 个 STL 均存在、长度与三角形数量一致，且经 MuJoCo 编译解析成功。

使用原始 URDF 内容加载的实际错误：

```text
Error: Error opening file 'package:/RML63II-6F-V/meshes/base_link.STL'
```

仅解析 mesh 路径并保留原始物理参数后的实际错误：

```text
Error: inertia must satisfy A + B >= C; use 'balanceinertia' to fix
Element name 'camera_link', id 9
```

按照已批准方案，仅在兼容副本修正 `camera_link.izz`，随后加载、结构检查及 `mj_forward()` 均成功。原始值、推导、上游许可证不一致和型号差异详见 `assets/robots/rml63/SOURCE.md`。

## 无界面检查：完整标准输出

运行命令：

```bash
conda run --no-capture-output -n realman_rl python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/inspect_model.py
```

退出码：0。完整输出如下，警告没有被过滤：

```text
Python: 3.11.17 (/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python)
MuJoCo: 3.17.0; NumPy: 2.4.6
Model: /home/mr-beaver/Documents/robot/realman_rl_simtoreal/assets/robots/rml63/rml63_mujoco.urdf
[PASS] Upstream SHA-256 checksums: 12 files
[WARN] Original camera_link inertia violates A+B>=C: margin=-3.000000073e-13 kg*m^2; see SOURCE.md
[WARN] RML63II-6F-V hardware generation and physical parameters are not verified against the real arm.
[PASS] URDF: 9 links, 8 joints; six arm revolute joints + camera revolute + camera fixed
  Mesh base_link.STL: 242184 bytes, 4842 triangles
  Mesh camera_link.STL: 19484 bytes, 388 triangles
  Mesh camera_rolink.STL: 7662584 bytes, 153250 triangles
  Mesh link_1.STL: 236384 bytes, 4726 triangles
  Mesh link_2.STL: 171084 bytes, 3420 triangles
  Mesh link_3.STL: 250584 bytes, 5010 triangles
  Mesh link_4.STL: 203484 bytes, 4068 triangles
  Mesh link_5.STL: 168284 bytes, 3364 triangles
  Mesh link_6.STL: 1404084 bytes, 28080 triangles
[PASS] All 9 referenced binary STL files are complete
  Link base_link: mass=1.862 kg, COM=[0.0014213, -0.00010164, 0.050087] m, principal inertia=[0.0006790174429643876, 0.0009961831447939935, 0.0010169494122416179] kg*m^2
  Link link_1: mass=1.837 kg, COM=[-0.068442, -0.023913, -0.006938] m, principal inertia=[0.0016910526437423244, 0.00419084075181575, 0.004313819604441927] kg*m^2
  Link link_2: mass=2.006 kg, COM=[0.1667, -2e-06, -0.09259] m, principal inertia=[0.0014646266661970983, 0.030618537041331294, 0.030972818292471633] kg*m^2
  Link link_3: mass=1.961 kg, COM=[0.033399, -0.029498, -0.017697] m, principal inertia=[0.0016431521869763187, 0.005964517844575161, 0.0061621609684485235] kg*m^2
  Link link_4: mass=1.201 kg, COM=[0.0, -0.035177, -0.1844] m, principal inertia=[0.0006080178644797742, 0.010697086000312307, 0.010710430135207921] kg*m^2
  Link link_5: mass=1.026 kg, COM=[-3.1e-05, 0.030146, -0.012341] m, principal inertia=[0.0005160235013257967, 0.0016094078496641164, 0.0016439986490100868] kg*m^2
  Link link_6: mass=0.248 kg, COM=[-0.000481, 0.000365, -0.010605] m, principal inertia=[3.6176251908374674e-05, 4.329070017038236e-05, 6.428904792124295e-05] kg*m^2
  Link camera_rolink: mass=0.21281 kg, COM=[0.010432, -0.036654, -0.0024431] m, principal inertia=[3.168747664543617e-05, 0.00014208964273480646, 0.00015933788061975735] kg*m^2
  Link camera_link: mass=9.2946e-05 kg, COM=[-0.03617, 1.9588e-08, -6.501e-05] m, principal inertia=[2.302699999983072e-09, 4.749400000001324e-08, 4.979669000000367e-08] kg*m^2
[PASS] Selected URDF mass/inertia checks; total mass including fixed base=10.353902946 kg
  URDF joint_1: axis=[0.0, 0.0, 1.0] (joint frame), effort=60 N*m, velocity=3.14 rad/s
  URDF joint_2: axis=[0.0, 0.0, 1.0] (joint frame), effort=60 N*m, velocity=3.14 rad/s
  URDF joint_3: axis=[0.0, 0.0, 1.0] (joint frame), effort=30 N*m, velocity=3.14 rad/s
  URDF joint_4: axis=[0.0, 0.0, 1.0] (joint frame), effort=30 N*m, velocity=3.14 rad/s
  URDF joint_5: axis=[0.0, 0.0, 1.0] (joint frame), effort=10 N*m, velocity=3.14 rad/s
  URDF joint_6: axis=[0.0, 0.0, 1.0] (joint frame), effort=10 N*m, velocity=3.14 rad/s
  URDF camera_rojoint: axis=[0.0, 0.0, 1.0] (joint frame), effort=0 N*m, velocity=0 rad/s
  URDF camera_joint: fixed (no MuJoCo motion joint)
MuJoCo: nbody=10 (includes world), njnt=7, nq=7, nv=7, nu=0, nmesh=9, ngeom=18
  Body 0 world: parent=-, mass=0 kg, COM=[0.0, 0.0, 0.0] m, principal inertia=[0.0, 0.0, 0.0] kg*m^2
  Body 1 base_link: parent=world, mass=1.862 kg, COM=[0.0014213, -0.00010164, 0.050087] m, principal inertia=[0.0010169494122416194, 0.0009961831447939944, 0.0006790174429643882] kg*m^2
  Body 2 link_1: parent=base_link, mass=1.837 kg, COM=[-0.068442, -0.023913, -0.006938] m, principal inertia=[0.004313819604441924, 0.004190840751815748, 0.0016910526437423224] kg*m^2
  Body 3 link_2: parent=link_1, mass=2.006 kg, COM=[0.1667, -2e-06, -0.09259] m, principal inertia=[0.03097281829247165, 0.03061853704133129, 0.0014646266661970987] kg*m^2
  Body 4 link_3: parent=link_2, mass=1.961 kg, COM=[0.033399, -0.029498, -0.017697] m, principal inertia=[0.006162160968448523, 0.00596451784457516, 0.0016431521869763174] kg*m^2
  Body 5 link_4: parent=link_3, mass=1.201 kg, COM=[0.0, -0.035177, -0.1844] m, principal inertia=[0.010710430135207918, 0.010697086000312307, 0.0006080178644797745] kg*m^2
  Body 6 link_5: parent=link_4, mass=1.026 kg, COM=[-3.1e-05, 0.030146, -0.012341] m, principal inertia=[0.0016439986490100879, 0.0016094078496641177, 0.0005160235013257971] kg*m^2
  Body 7 link_6: parent=link_5, mass=0.248 kg, COM=[-0.000481, 0.000365, -0.010605] m, principal inertia=[6.4289047921243e-05, 4.329070017038239e-05, 3.6176251908374694e-05] kg*m^2
  Body 8 camera_rolink: parent=link_6, mass=0.21281 kg, COM=[0.010432, -0.036654, -0.0024431] m, principal inertia=[0.00015933788061975744, 0.00014208964273480649, 3.16874766454362e-05] kg*m^2
  Body 9 camera_link: parent=camera_rolink, mass=9.2946e-05 kg, COM=[-0.03617, 1.9588e-08, -6.501e-05] m, principal inertia=[4.97966900000037e-08, 4.749400000001329e-08, 2.3026999999830737e-09] kg*m^2
  Joint 0 joint_1: mjJNT_HINGE, arm, axis=[0.0, 0.0, 1.0] (child body frame), range=[-3.106, 3.106] rad / [-177.960691, 177.960691] deg
  Joint 1 joint_2: mjJNT_HINGE, arm, axis=[0.0, 0.0, 1.0] (child body frame), range=[-3.106, 3.106] rad / [-177.960691, 177.960691] deg
  Joint 2 joint_3: mjJNT_HINGE, arm, axis=[0.0, 0.0, 1.0] (child body frame), range=[-3.106, 2.5307] rad / [-177.960691, 144.998429] deg
  Joint 3 joint_4: mjJNT_HINGE, arm, axis=[0.0, 0.0, 1.0] (child body frame), range=[-3.106, 3.106] rad / [-177.960691, 177.960691] deg
  Joint 4 joint_5: mjJNT_HINGE, arm, axis=[0.0, 0.0, 1.0] (child body frame), range=[-3.106, 3.106] rad / [-177.960691, 177.960691] deg
  Joint 5 joint_6: mjJNT_HINGE, arm, axis=[0.0, 0.0, 1.0] (child body frame), range=[-6.28, 6.28] rad / [-359.817495, 359.817495] deg
  Joint 6 camera_rojoint: mjJNT_HINGE, camera accessory, axis=[0.0, 0.0, 1.0] (child body frame), range=[0.0, 1.05] rad / [0.0, 60.160568] deg
[PASS] Six arm hinges verified by name, type, topology, axis and limits; one additional camera hinge
[WARN] nu=0: URDF effort/velocity do not create actuators or enforce runtime velocity limits
[PASS] mj_forward: finite state, poses, mass matrix and forces; time=0, qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[INFO] Initial contacts=2; contact geometry and dynamics are not validated for training
  Contact base_link / link_1: distance=-0.002726524 m
  Contact base_link / link_1: distance=-0.002726676 m
[WARN] Initial collision penetration detected; geometry is unchanged and needs review before dynamic simulation
[PASS] Model import and structural checks complete (warnings above remain applicable)

```

`nbody=10` 包含 world 和 9 个模型 Body；`njnt=nq=nv=7` 包含六个机械臂关节及相机支架关节；固定连接不占运动自由度，`nu=0` 表示没有执行器。程序同时检查名称、类型、父子链、轴、限位及编译前后的质量、质心和惯量，不仅检查数量。

## 其他自动测试

通过目标环境的 Python `subprocess` 执行真实 CLI；使用 `/tmp` 临时目录中的模型副本构造故障，未修改正式资产。检查返回码及明确错误文字：

```text
PASS normal inspection: exit=0; found '[PASS] Model import and structural checks complete'
PASS inspection from /tmp: exit=0; found '[PASS] Model import and structural checks complete'
PASS missing model: exit=1; found 'Model file not found'
PASS missing mesh: exit=1; found 'Missing mesh:'
PASS wrong joint name: exit=1; found 'Unexpected or duplicate URDF joints'
PASS original camera inertia rejected: exit=1; found 'Non-physical inertia: camera_link'
PASS invalid XML: exit=1; found '[FAIL]'
PASS viewer without display: exit=1; found 'No desktop display'
PASS viewer missing model: exit=1; found 'Model file not found'
PASS invalid viewer duration: exit=2; found 'duration must be finite'
PASS XML semantic diff: only mesh paths, compiler options and the authorized camera_link izz correction
PASS exact requirements versions
PASS syntax: both scripts
PASS invalid DISPLAY: clear GLFW initialization error, exit=1, no hang
```

项目目录和 `/tmp` 启动的正常检查均返回 0。临时模型覆盖缺失文件、缺失 mesh、错误关节名称、原始不合法惯量和 XML 解析失败；Viewer 覆盖缺失显示变量、不可连接 DISPLAY、缺失模型及非法 duration。没有为这些测试额外安装测试框架。

## Viewer 图形与退出验证

桌面环境存在 `DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0`，实际测试通过 X11/XWayland 打开官方 MuJoCo Viewer。捕获并查看了本次 Viewer 窗口截图，确认完整机械臂和相机附件可见；截图位于 `outputs/viewer_initial.png`（本地生成产物，Git 忽略）。

初次测试发现退出阶段崩溃：Wayland 下出现 libdecor 初始化警告及退出码 139，X11 下也曾出现 GLFW TLS 清理断言及退出码 134。检查固定版本 viewer.py 后发现 `close()` 只请求关闭，渲染线程为 daemon；解释器退出可与渲染清理竞争。修复为：关闭后用标准库 `Thread.join()` 等待该独立脚本本次 Viewer 创建的线程结束，不修改安装包、不使用固定延时掩盖竞态。X11/XWayland 选择使用已有 pyGLFW 的官方环境变量，避免本机 native Wayland 的窗口装饰问题。

最终测试结果：

```text
PASS actual X11 viewer window; screenshot captured
GUI input injection: rotation_changed=False; zoom_changed=False; physical mouse interaction remains unverified
PASS window-manager WM_DELETE_WINDOW close from /tmp: process exit=0
PASS Ctrl+C equivalent SIGINT: process exit=0
PASS timed shutdown repeat 1: process exit=0
PASS timed shutdown repeat 2: process exit=0

```

图形验收使用临时测试程序，只查找属于本次 Python 子进程的 X11 窗口；截图由该窗口像素获取。向该窗口发送 `WM_DELETE_WINDOW` 检查窗口关闭，用 SIGINT 检查 Ctrl+C 处理，并两次独立运行 `--duration 1` 检查退出竞态；所有最终关闭测试均核查了操作系统退出码 0。

自动注入拖动和滚轮后截图未出现可确认变化，因此**没有宣称旋转、缩放的人工交互测试通过**。`viewer_rotated.png`、`viewer_zoomed.png` 仅是注入尝试后的截图。用户使用真实鼠标操作仍待人工确认；脚本保留官方 Viewer 的交互机制。

窗口关闭测试的完整输出（退出码 0）：

```text
Model: /home/mr-beaver/Documents/robot/realman_rl_simtoreal/assets/robots/rml63/rml63_mujoco.urdf
GLFW backend preference: x11
Static URDF zero pose; no physics stepping. Real hardware match remains unverified.
[PASS] Viewer initialized. Rotate: left drag; pan: right drag; zoom: scroll.
[PASS] Viewer closed normally; qpos unchanged, simulation time=0

```

Ctrl+C 等效 SIGINT 测试的完整输出（退出码 0）：

```text
Model: /home/mr-beaver/Documents/robot/realman_rl_simtoreal/assets/robots/rml63/rml63_mujoco.urdf
GLFW backend preference: x11
Static URDF zero pose; no physics stepping. Real hardware match remains unverified.
[PASS] Viewer initialized. Rotate: left drag; pan: right drag; zoom: scroll.
[PASS] Viewer stopped by Ctrl+C

```

定时关闭测试的完整输出（两次退出码均为 0）：

```text
Model: /home/mr-beaver/Documents/robot/realman_rl_simtoreal/assets/robots/rml63/rml63_mujoco.urdf
GLFW backend preference: x11
Static URDF zero pose; no physics stepping. Real hardware match remains unverified.
[PASS] Viewer initialized. Rotate: left drag; pan: right drag; zoom: scroll.
[PASS] Viewer closed normally; qpos unchanged, simulation time=0

```

脚本没有 `mj_step()` 调用；静态循环恢复检查姿态并执行 `mj_forward()`，仿真时间保持 0，避免无执行器机械臂因重力运动。

## 尚未验证及保留问题

- 当前模型与目标实物的代际、零位、尺寸及动力学参数是否完全一致。
- 相机惯量修正的厂家认可；目前只是明确记录的数学修正。
- 初始两个约 2.73 mm 的底座/第一连杆碰撞穿透接触；未修改 STL、碰撞配置或过滤来隐藏问题。
- 真实鼠标的旋转、缩放操作；修复清理逻辑后 native Wayland 后端未重新验收，当前通过的是 X11/XWayland。
- 上游 Apache-2.0 与包元数据 BSD 声明的最终解释。
- 执行器、控制器、传感器、动力学稳定性、训练任务及真机部署均未实现或验证。
