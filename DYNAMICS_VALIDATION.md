# 基础关节控制验证记录

日期：2026-11-05（Asia/Shanghai）。环境：现有 `realman_rl`，Python 3.11.17、MuJoCo 3.17.0、NumPy 2.4.6。没有新增依赖，没有修改其他 Conda 环境或官方原始资产，没有提交或推送 GitHub。

本文件保留基础关节控制阶段的结果。后续任务 02-B 已增加临时末端 site、FK/IK 及小幅末端控制，记录见 `KINEMATICS_VALIDATION.md`；下文“未实现逆运动学”指本文件原阶段结束时的状态。六关节物理参数与控制器未改变。

## 模型和控制方式

- `scene.xml` 引用 `rml63_dynamics.xml`，另加入灯光及可碰撞地面。机器人保留九个 Body，含 world 共十个；底座无自由度，两个相机 Body 刚性连接在第六连杆，只有六个机械臂 hinge。
- 六个官方 `position` 执行器通过关节名一一映射，`nq=nv=nu=6`，输入 `ctrl` 单位为 rad。实际积分由 `mj_step()` 推进，只有初始化时直接设置状态。
- `tau = kp*(ctrl-q) - kv*qvel`，无积分项、无重力补偿。稳态误差来自 PD 对重力的平衡，不通过重置状态或取消重力伪造保持。
- kp：`[500,1000,800,200,120,80] N*m/rad`；kv：`[50,90,70,15,10,5] N*m*s/rad`；joint damping：`[1,1,0.5,0.2,0.1,0.05] N*m*s/rad`。参数选择依据、模型惯量精度和完整兼容性修改见 `assets/robots/rml63/SOURCE.md`。
- 沿用关节角限位，六轴输出力矩限幅为 `[60,60,30,30,10,10] N*m`。这些上游 effort 字段未通过真机能力核实。
- `gravity=[0,0,-9.81] m/s^2`、时间步 0.002 s、`implicitfast`。不增加未知电机惯量或摩擦模型。

## 穿透和相机关节的诊断

原静态模型的 `base_link/link_1` 接触复现为两个约 −0.0027265 m 的穿透接触，坐标约 `(0.0395,-0.0122,0.10936) m`。原始网格的世界 Z 包络分别约为 `[0,0.1113] m` 与 `[0.1080,0.21672] m`，重叠处位于 J1 装配接口。原始模型的底座 `body_weldid=0`，因此 world 固定组的相邻过滤例外使装配口凸碰撞包络仍进入接触求解；J1 在 −90、−10、0、10、90 度的前向抽样均有约 2.73 mm 穿透。

派生 MJCF 只增加 `base_link/link_1` 这一对的 exclude，恢复常见的相邻装配过滤。没有更改机器人尺寸、移动原点、全局关闭接触或删除其他碰撞几何。静态 URDF 的历史警告保留；动力学默认测试中接触数为 0。

通过两项反例确认碰撞保留：

- 前向计算一个折叠姿态，`base_link_collision/link_2_collision` 仍有约 2.0 mm 的穿透接触；该姿态只用于碰撞检测，没有进行大幅动力学运动。
- 在第二连杆内部放置独立球体，检测到约 47.8 mm 穿透，运行控制检查立即报 `Unexpected collision`。球体仅存在于测试临时模型中。

相机支架原定义为 revolute，且没有保持驱动；当前初始姿态下该自由度的重力广义力约 0.0772 N*m，因此不能依赖原始零位自然保持。动力学模型移除该 joint，在原角度 0 rad 刚性连接，保留两个相机连杆的全部几何、质量和惯量。底座与相机的固定关系通过 20 秒物理积分和相对位姿检查验证。

## 验收标准与数值结果

测试初始指令为 `[0,-20,30,0,20,0]` 度。默认序列：保持 3 s，逐轴 +3°（1 s 平滑切换 +3 s 保持），各次返回用 1 s 切换 +1 s 保持；共 13 段、39 s 仿真时间。

采用三次平滑曲线 `3s^2-2s^3` 调整位置指令，不跨步直接修改实际角度。命令路径的最大速度为 4.5°/s；实测关节峰值速度约 4.541°/s。自定义命令限制在 home 每轴 ±5°内，拒绝越限、NaN 和 Inf。

每段验收：所有关节最终绝对误差 ≤0.5°；最后 0.5 s 的每轴峰峰波动 ≤0.05°、最大速度 ≤0.1°/s；运动轴对最终目标的超调 ≤0.5°；未指令轴相对该段开始时的最大运动 ≤0.5°；穿透深度超过 0.1 mm 即失败；监控全部步骤的有限数值、数值警告、时间推进、角限位和力矩限幅。阈值只针对本阶段小幅仿真验收，不是真机精度或安全指标。

默认序列的逐轴 +3°段结果：

```text
J1: target=3.000000 deg, actual=3.000000 deg, error=-0.000000 deg
J2: target=-17.000000 deg, actual=-17.052088 deg, error=-0.052088 deg
J3: target=33.000000 deg, actual=33.367067 deg, error=+0.367067 deg
J4: target=3.000000 deg, actual=3.025908 deg, error=+0.025908 deg
J5: target=23.000000 deg, actual=23.309070 deg, error=+0.309070 deg
J6: target=3.000000 deg, actual=2.996117 deg, error=-0.003883 deg
```

汇总：

- 最大末态绝对误差：0.367748688°（包括所有未指令轴）。
- 最大对目标超调：0.367066944°，包含静态重力偏差，未将其扣除后报告。
- 小幅运动段中未指令关节的最大运动：0.044116973°。初始保持段的自然重力收敛最大运动为 0.328393°。
- 末 0.5 s 最大峰峰波动：0.000817543°；没有持续振荡。
- 所有段最大峰值力矩：5.306065603 N*m，低于对应关节限幅；逐关节值见完整输出和 JSON。
- 全部正常段接触数为 0，无 NaN/Inf、MuJoCo 数值警告或仿真自动重置。

完整数值日志保留在本文件后面；`outputs/joint_control_results.json` 为本地 JSON 产物，Git 忽略。

## 持久化自动测试

命令：

```bash
conda run --no-capture-output -n realman_rl python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests/test_joint_control.py
```

以下测试从 `/tmp` 启动，退出码为 0；正负运动测试包含六轴各 ±3° 共 12 个子场景。除了默认 2 ms 外，另以 1 ms 时间步验证组合小角度运动。模型重新生成两次的 SHA-256 一致。

```text
test_base_assembly_exclusion_is_specific (__main__.JointControlTests.test_base_assembly_exclusion_is_specific) ... ok
test_conversion_preserves_geometry_inertia_and_limits (__main__.JointControlTests.test_conversion_preserves_geometry_inertia_and_limits) ... ok
test_each_joint_positive_and_negative_small_motion (__main__.JointControlTests.test_each_joint_positive_and_negative_small_motion) ... ok
test_environment_collision_detected_and_rejected (__main__.JointControlTests.test_environment_collision_detected_and_rejected) ... ok
test_half_timestep_and_combined_small_motion (__main__.JointControlTests.test_half_timestep_and_combined_small_motion) ... ok
test_hold_base_camera_and_uncommanded_joints (__main__.JointControlTests.test_hold_base_camera_and_uncommanded_joints) ... ok
test_invalid_commands_and_bad_mapping_rejected (__main__.JointControlTests.test_invalid_commands_and_bad_mapping_rejected) ... ok
test_nonadjacent_self_collision_is_still_detected (__main__.JointControlTests.test_nonadjacent_self_collision_is_still_detected) ... ok
test_original_assets_and_static_inspection (__main__.JointControlTests.test_original_assets_and_static_inspection) ... ok
test_position_actuator_force_mapping_and_clamps (__main__.JointControlTests.test_position_actuator_force_mapping_and_clamps) ... ok

----------------------------------------------------------------------
Ran 10 tests in 7.500s

OK

```

另通过真实 CLI 验证了 NaN 目标和超过 home 5° 的目标均返回退出码 1。原始 URDF、STL 等 12 文件的 SHA-256 均一致；已有 `inspect_model.py` 在同一测试中仍成功返回 0，保持第一阶段功能。

## 图形验证

通过 X11/XWayland 实际运行 `control_joints.py --viewer --target-deg 3 -20 30 0 20 0`。完整 7 s 仿真结束后退出码为 0；截取本次 Viewer 的初始化和运动后图像，中心显示区约 3.03% 的 RGB 分量变化，并已查看截图确认模型正常显示。截图保存在 `outputs/control_viewer_initial.png`、`outputs/control_viewer_moved.png`（本地生成产物）。

另外实际测试窗口管理器关闭、SIGINT（Ctrl+C 等效）及 `--duration 1`，均退出码 0，并报告测试序列未完整结束，没有把提前结束当作完整验收通过。清理复用第一阶段的 Viewer 线程等待逻辑，避免解释器退出与 GLFW 清理竞争。鼠标旋转/缩放未重新进行人工验收；native Wayland 未验收。

## 剩余边界

- 当前验证覆盖指定初始姿态、小幅单轴/组合目标及 20 s 保持，不代表全工作空间、任意目标或大幅运动稳定。
- 约 0.37° 的 PD 重力稳态误差仍存在；没有将仿真稳定等同于与真机一致。
- 实物代际、实际零位、真实惯量、齿轮/摩擦及执行器带宽仍未标定；相机的刚性零角度是假定的仿真安装方式。
- 仍使用原始 mesh 的凸碰撞包络，非相邻和环境碰撞保留，但没有验证全工作空间碰撞精度；相邻关节装配过滤也不是对 CAD 实体无干涉的证明。
- 第一阶段的相机惯量微小修正和上游许可证声明不一致仍未获得厂家确认。
- 未实现逆运动学、末端位姿控制、强化学习、Gymnasium、真机通信或夹爪任务。

## 默认控制程序完整输出

命令：

```bash
conda run --no-capture-output -n realman_rl python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_joints.py
```

退出码：0；默认输出与带 `--json` 的数值输出相同。

```text
MuJoCo 3.17.0: nq=6 nv=6 nu=6; gravity=[0.0, 0.0, -9.81]; dt=0.002
Position PD control, no gravity compensation; model gains are not real-arm parameters.
  joint_1 <- position_1: kp=500, kv=50, torque range=[-60.0, 60.0] N*m
  joint_2 <- position_2: kp=1000, kv=90, torque range=[-60.0, 60.0] N*m
  joint_3 <- position_3: kp=800, kv=70, torque range=[-30.0, 30.0] N*m
  joint_4 <- position_4: kp=200, kv=15, torque range=[-30.0, 30.0] N*m
  joint_5 <- position_5: kp=120, kv=10, torque range=[-10.0, 10.0] N*m
  joint_6 <- position_6: kp=80, kv=5, torque range=[-10.0, 10.0] N*m

[PASS] hold home under gravity
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000000 deg; tail range=9.839e-14 deg; tail speed=1.568e-12 deg/s; contacts=0
  max uncommanded joint travel=0.328393 deg
  peak torque [N*m]: [0.13418 2.69014 4.72626 0.09598 0.62621 0.005  ]

[PASS] J1 +3 deg
  J1: target=   3.0000 deg actual=   3.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000000 deg; tail range=1.019e-12 deg; tail speed=1.066e-11 deg/s; contacts=0
  max uncommanded joint travel=0.000928 deg
  peak torque [N*m]: [9.64877e-02 2.47431e+00 4.64463e+00 9.61165e-02 6.11215e-01 1.14740e-03]

[PASS] J1 return
  J1: target=   0.0000 deg actual=   0.0000 deg error=+0.000005 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000000 deg; tail range=8.175e-04 deg; tail speed=8.519e-03 deg/s; contacts=0
  max uncommanded joint travel=0.000980 deg
  peak torque [N*m]: [9.64884e-02 2.47698e+00 4.64399e+00 9.59224e-02 6.11135e-01 1.19722e-03]

[PASS] J2 +3 deg
  J1: target=   0.0000 deg actual=   0.0000 deg error=+0.000000 deg
  J2: target= -17.0000 deg actual= -17.0521 deg error=-0.052088 deg
  J3: target=  30.0000 deg actual=  30.3677 deg error=+0.367749 deg
  J4: target=   0.0000 deg actual=   0.0354 deg error=+0.035403 deg
  J5: target=  20.0000 deg actual=  20.3097 deg error=+0.309739 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000734 deg
  max final error=0.367749 deg; overshoot=0.000000 deg; tail range=8.583e-14 deg; tail speed=1.518e-12 deg/s; contacts=0
  max uncommanded joint travel=0.044116 deg
  peak torque [N*m]: [2.06989e-02 2.88116e+00 5.30607e+00 1.32075e-01 6.65588e-01 1.06286e-03]

[PASS] J2 return
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140761 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332317 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332317 deg; overshoot=0.140761 deg; tail range=1.620e-04 deg; tail speed=1.882e-03 deg/s; contacts=0
  max uncommanded joint travel=0.044117 deg
  peak torque [N*m]: [2.05460e-02 2.88818e+00 5.28589e+00 1.31096e-01 6.63568e-01 1.05426e-03]

[PASS] J3 +3 deg
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1124 deg error=-0.112414 deg
  J3: target=  33.0000 deg actual=  33.3671 deg error=+0.367067 deg
  J4: target=   0.0000 deg actual=   0.0352 deg error=+0.035243 deg
  J5: target=  20.0000 deg actual=  20.3094 deg error=+0.309397 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000733 deg
  max final error=0.367067 deg; overshoot=0.367067 deg; tail range=6.964e-15 deg; tail speed=1.526e-12 deg/s; contacts=0
  max uncommanded joint travel=0.035142 deg
  peak torque [N*m]: [4.66451e-03 2.60923e+00 5.24548e+00 1.26690e-01 6.59856e-01 1.04733e-03]

[PASS] J3 return
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291764 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000000 deg; tail range=2.623e-04 deg; tail speed=3.078e-03 deg/s; contacts=0
  max uncommanded joint travel=0.035241 deg
  peak torque [N*m]: [4.76092e-03 2.62962e+00 5.25102e+00 1.26377e-01 6.58307e-01 1.04322e-03]

[PASS] J4 +3 deg
  J1: target=   0.0000 deg actual=   0.0000 deg error=+0.000000 deg
  J2: target= -20.0000 deg actual= -20.1392 deg error=-0.139151 deg
  J3: target=  30.0000 deg actual=  30.3343 deg error=+0.334291 deg
  J4: target=   3.0000 deg actual=   3.0259 deg error=+0.025908 deg
  J5: target=  20.0000 deg actual=  20.2917 deg error=+0.291691 deg
  J6: target=   0.0000 deg actual=  -0.0018 deg error=-0.001789 deg
  max final error=0.334291 deg; overshoot=0.025908 deg; tail range=3.127e-15 deg; tail speed=1.489e-12 deg/s; contacts=0
  max uncommanded joint travel=0.002124 deg
  peak torque [N*m]: [1.17316e-03 2.46476e+00 4.67133e+00 9.66368e-02 6.11151e-01 2.49802e-03]

[PASS] J4 return
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000000 deg; tail range=9.959e-05 deg; tail speed=1.356e-03 deg/s; contacts=0
  max uncommanded joint travel=0.002135 deg
  peak torque [N*m]: [1.23613e-03 2.46586e+00 4.67089e+00 1.09269e-01 6.11094e-01 2.49802e-03]

[PASS] J5 +3 deg
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1386 deg error=-0.138602 deg
  J3: target=  30.0000 deg actual=  30.3350 deg error=+0.334964 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027213 deg
  J5: target=  23.0000 deg actual=  23.3091 deg error=+0.309070 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000725 deg
  max final error=0.334964 deg; overshoot=0.309070 deg; tail range=2.684e-15 deg; tail speed=1.520e-12 deg/s; contacts=0
  max uncommanded joint travel=0.003235 deg
  peak torque [N*m]: [1.56927e-04 2.47199e+00 4.68890e+00 9.50874e-02 6.52498e-01 1.01695e-03]

[PASS] J5 return
  J1: target=   0.0000 deg actual=   0.0000 deg error=+0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332317 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332317 deg; overshoot=0.000000 deg; tail range=2.347e-04 deg; tail speed=2.875e-03 deg/s; contacts=0
  max uncommanded joint travel=0.003237 deg
  peak torque [N*m]: [1.62582e-04 2.47405e+00 4.68732e+00 9.50381e-02 6.55867e-01 1.01667e-03]

[PASS] J6 +3 deg
  J1: target=   0.0000 deg actual=   0.0000 deg error=+0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140779 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332296 deg
  J4: target=   0.0000 deg actual=   0.0268 deg error=+0.026753 deg
  J5: target=  20.0000 deg actual=  20.2916 deg error=+0.291630 deg
  J6: target=   3.0000 deg actual=   2.9961 deg error=-0.003883 deg
  max final error=0.332296 deg; overshoot=0.000000 deg; tail range=6.279e-16 deg; tail speed=1.284e-12 deg/s; contacts=0
  max uncommanded joint travel=0.000449 deg
  peak torque [N*m]: [2.89998e-04 2.45728e+00 4.64007e+00 9.49517e-02 6.11076e-01 7.05273e-03]

[PASS] J6 return
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000667 deg; tail range=1.743e-05 deg; tail speed=2.871e-04 deg/s; contacts=0
  max uncommanded joint travel=0.000449 deg
  peak torque [N*m]: [2.90070e-04 2.45725e+00 4.64007e+00 9.49517e-02 6.11078e-01 5.37196e-03]

[PASS] All 13 segments completed; simulation time=39.000 s

```
