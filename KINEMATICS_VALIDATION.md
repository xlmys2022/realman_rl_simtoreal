# 任务 02-B：FK、固定朝向 IK 与末端动力学控制验证

日期：2026-11-05（Asia/Shanghai）。全部开发和 Python 测试使用现有 Conda `realman_rl`：Python 3.11.17、MuJoCo 3.17.0、NumPy 2.4.6。没有新增依赖、修改其他环境或提交/推送 GitHub；没有加入强化学习、逆动力学控制器、真机通信或路径规划。

## 文件与末端坐标定义

- 新增 `scripts/kinematics.py`：FK、世界系 Jacobian、旋转误差和结构化全位姿 DLS IK。
- 新增 `scripts/control_ee.py`：笛卡尔输入、IK/路径预检、复用执行器运动、实际末端误差和 Viewer。
- 新增 `tests/test_kinematics.py`：16 项运动学/末端控制测试；原有 10 项关节测试保留并回归通过。
- 修改 `scripts/build_dynamics_model.py`，同步生成 `assets/robots/rml63/rml63_dynamics.xml` 的 `ee_site`；重新生成结果一致。
- 更新 README、SOURCE 和历史 DYNAMICS_VALIDATION 的阶段说明；本文件保存详细结果。

`ee_site` 直接位于 `link_6` 上，局部位置 `[0,0,0] m`，局部四元数 `[1,0,0,0]`。它与第六连杆同向，不通过相机支架定义，不是厂商标定的真实 TCP。site 没有质量、惯量或碰撞；机器人已有 Body、geom、动力学参数、六个执行器、固定相机及底座接触 exclude 不变。

`mj_forward()` 后的 `site_xpos` 为世界位置，`site_xmat` 是 site-to-world 的 3×3 旋转矩阵，其列向量为 site 的三个轴在世界系中的表达。home 名义位置约为 `[0.00561669109,-0.000145158656,1.06022729] m`。姿态用矩阵和最短旋转角报告，不用欧拉角相减。

FK 接口 `Kinematics(model).fk(q)` 输入 `joint_1..joint_6` 顺序的六个弧度值，返回位置/旋转矩阵的独立副本。`jacobian(q)` 返回同一顺序的 `Jp, Jr`（各 3×6）。每个 Kinematics 实例拥有独立 MjData，内部赋值 qpos 只作用于该计算数据，不读取或改写正在运行的动力学 MjData；同一实例按单调用线程使用。

API 依据：[MuJoCo 3.17.0 的 mj_forward、mj_jacSite 与四元数辅助函数](https://mujoco.readthedocs.io/en/3.17.0/APIreference/APIfunctions.html)。没有手写 DH 表或使用相机坐标替代末端。

## 数值 IK：算法、参数与失败语义

接口为 `solve_ik(target_position, initial_q, target_rotation=None, ...)`。位置以世界米表示，初值以关节弧度表示；缺省旋转目标是名义 home 的旋转矩阵。显式传入目标旋转用于全位姿一致性实验。没有纯位置求解模式，也不在姿态约束失败后自动降级。

每次迭代在私有数据上用 MuJoCo 正运动学和 `mj_jacSite()` 计算误差及平移、旋转 Jacobian：

```text
e_p = p_target - p(q)
e_R = Log(R_target @ R(q).T)^vee      # world-frame rotation vector, radians
e_w = [e_p; ell*e_R]
J_w = [J_p; ell*J_r]

(J_w.T @ J_w + (lambda^2 + mu^2)*I) @ delta_q
    = J_w.T @ e_w + mu^2*(q_seed - q)
```

旋转差矩阵先通过 MuJoCo 转换为四元数，选择最短代表，再用 `mju_quat2Vel(..., dt=1)` 得到轴角向量。角速度 Jacobian 与旋转误差均使用世界坐标系。上述更新采用局部线性近似；不是任意大姿态变化的全局求解保证。

- `ell=0.2 m/rad`：把旋转残差缩放到与平移相同的米尺度；平移和旋转仍分别满足收敛阈值才判成功。
- 位置容差 `1e-5 m`（0.01 mm）；姿态容差 `1e-4 rad`（约 0.00573°）。这是 IK 数值验收，不能当作实际控制精度。
- `lambda=0.01`：DLS 阻尼，抑制小奇异值放大；参数检查拒绝非正值、NaN/Inf，阻尼至少为 `1e-6`。
- `mu=1e-4`：弱初值正则项，配合从初值开始的局部搜索偏向附近解；不进行随机重启或 ±2π 跳转，不保证找到了全局最近解。
- 单次每轴最大变化 `0.05 rad`（约 2.865°），先整体缩放步长，再将候选限制在模型硬限位内。
- 最多 150 次迭代，API 允许的迭代预算范围为 0–1000。每次最多尝试 8 个回溯比例 `1 ... 1/128`，只有加权残差加初值正则的代价下降才接受。
- 由父子链平移长度之和建立世界原点保守可达球，仅用于明显远目标的拒绝，不当作真实工作空间。
- 无下降、关节受限或局部不可达时给出 stalled；预算耗尽、非法输入及数值异常分别给出原因。局部失败不构成全局不可达证明。

`IKResult` 包含 success、q、iterations、position_error（米）、orientation_error（弧度）、reason。失败时 `q=None`，不向调用者暴露可直接执行的失败候选；尚未进行有效残差评估的远目标/非法输入，其误差字段为 None。FK 输入越限时明确抛出 ValueError，IK 初值越限时返回失败。

## 从 IK 到真实动力学运动

默认目标是名义 home 位置沿世界 +X 5 mm，朝向仍为 home。程序先完成求解和执行检查，再进行任何 mj_step：

1. 失败 IK 不改 ctrl、不推进仿真；成功结果再次经 FK 检查目标位置与固定朝向。
2. 末端目标限定在名义 home 的 10 mm 球内，每轴目标限定在 home ±5°内，并检查模型原始限位。
3. 在独立计算数据中对从当前角度到目标的关节线段采样 51 点，检查大于 0.1 mm 的非预期穿透，保留原有碰撞过滤。
4. 复用 `control_joints.load_model()`、`initialize()`、`validate_target()` 和 `run_segment()`。先保持 home 3 s，再用 2 s 的三次平滑曲线切换六个 ctrl 目标，保持 3 s，总计 8 s、4000 次 mj_step。
5. 每个物理步骤后调用 mj_forward 刷新新 qpos 对应的 site/接触，然后记录实际位姿；持续检查限位、力矩限幅、接触、数值警告、NaN/Inf、时间推进和朝向误差。

初始化允许通过既有 home keyframe 放置机器人。正常运动中不直接写实际 `data.qpos`。测试在全部 4000 个积分调用之间检查 qpos 与上次 mj_step 的结果完全一致，确认没有通过状态改写或瞬移伪造结果。FK、IK 和碰撞预检所写的是独立的计算 MjData。

**固定朝向约束真正参与 IK**：同时使用 Jp 和 Jr，终点满足位置与 SO(3) 朝向容差。但关节空间插值不等同于严格固定朝向的笛卡尔轨迹，实际 PD 也存在重力误差。默认名义插值最大朝向漂移约 0.000019°；实际末态朝向误差约 0.499543°，全程最大约 0.500271°，并非实际朝向完全不变。

实际控制验收采用终点位置误差 ≤5 mm、全程朝向误差 ≤1°，同时沿用关节控制的 ≤0.5° 最终关节误差、末段波动/速度、超调与力矩检查。该尺度参考既有 PD 的约 0.37° 关节稳态偏差和本次实测结果，没有修改增益、关闭重力或用实际状态赋值来满足测试。

**IK 可达不意味着整条运动轨迹无碰撞**。51 点离散预检既不是连续碰撞检测，也不涵盖实际跟踪偏差；运行时检查可中止异常，但不能保证任意场景的路径安全。本次没有障碍物规划。

## 实际结果与回归

FK 姿态矩阵有限、正交、行列式为 +1；世界平移和角速度 Jacobian 均通过中心有限差分验证。旋转误差函数在接近 0 和 π 时也通过有限数值测试，这不代表已验证大姿态 IK 或大幅运动。

FK/IK 一致性使用三组 home 附近已知角度，通过 FK 构造完整位姿，再从 home 求解。比较重新 FK 的位姿，而非要求关节角与原始配置完全相同。另以 +X、−X、+Z 各 5 mm、固定 home 朝向进行真实动力学测试：

```text
offsets_deg=[1, -1, 1, 1, -1, 1]; success=True; iterations=2; position_error=0.009463425 mm; orientation_error=0.000240014 deg
offsets_deg=[-1, 1, -2, 1, 2, -1]; success=True; iterations=2; position_error=0.002292222 mm; orientation_error=0.000621674 deg
offsets_deg=[2, 0, 0, -1, 0, 1]; success=True; iterations=2; position_error=0.004741892 mm; orientation_error=0.000350892 deg
x5: IK iterations=2, IK position error=0.000140351 mm, IK rotation error=0.000017462 deg; actual error=2.391544403 mm / 0.499542925 deg
xminus5: IK iterations=2, IK position error=0.000156523 mm, IK rotation error=0.000032235 deg; actual error=2.036601371 mm / 0.468520013 deg
z5: IK iterations=2, IK position error=0.001872496 mm, IK rotation error=0.000318042 deg; actual error=2.251888513 mm / 0.485840292 deg

```

默认 +X 5 mm：

- IK 两次迭代成功，理论误差约 0.000140 mm / 0.000017°。
- 实际末端误差约 2.391544 mm / 0.499543°；未做末端闭环补偿，反映已有 PD 控制误差。
- 同时输出初始名义位置、重力收敛后的运动起点、目标、IK 理论终点和实际终点，避免混淆名义 home 与重力作用下的实际 home。
- 两段控制均无接触、超限力矩或 NaN/Inf；六轴目标和实际角度、误差、峰值力矩及末段统计见后面的完整日志。

明显超出范围的 `[10,10,10] m` 在第 0 次迭代返回保守距离界失败，q=None。API 和真实 CLI 均验证了失败结果不进入运动，仿真时间保持 0。非法输入、迭代预算、奇异初值及接近 J1 上限的内部候选也进行了边界检查。

另一个实际限制是 +Y 5 mm：全位姿 IK 可成功，但 J4 需要约 5.36986° 变化，超过演示每轴 ±5° 的门槛，因此在任何物理步之前拒绝执行。这是保守演示范围限制，不是物理不可达结论。

运行全部测试：

```bash
conda run --no-capture-output -n realman_rl python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

从 `/tmp` 启动，退出码 0；26 项测试通过（旧 10 + 新 16）。包括原始资产 SHA-256、静态检查、几何/惯量/限位一致性、正负小幅关节运动、20 s 保持、碰撞反例、site 重新生成、FK 隔离、完整位姿往返、奇异/限位/预算边界，以及失败不执行和真实末端运动。

另实际从 `/tmp` 运行下文的绝对 XYZ 命令，退出码为 0；`pip check` 报告 `No broken requirements found.`。Python/XML 检查及 `git diff --check` 通过，原控制器、静态脚本、兼容 URDF 和场景文件未变。

```text
test_base_assembly_exclusion_is_specific (test_joint_control.JointControlTests.test_base_assembly_exclusion_is_specific) ... ok
test_conversion_preserves_geometry_inertia_and_limits (test_joint_control.JointControlTests.test_conversion_preserves_geometry_inertia_and_limits) ... ok
test_each_joint_positive_and_negative_small_motion (test_joint_control.JointControlTests.test_each_joint_positive_and_negative_small_motion) ... ok
test_environment_collision_detected_and_rejected (test_joint_control.JointControlTests.test_environment_collision_detected_and_rejected) ... ok
test_half_timestep_and_combined_small_motion (test_joint_control.JointControlTests.test_half_timestep_and_combined_small_motion) ... ok
test_hold_base_camera_and_uncommanded_joints (test_joint_control.JointControlTests.test_hold_base_camera_and_uncommanded_joints) ... ok
test_invalid_commands_and_bad_mapping_rejected (test_joint_control.JointControlTests.test_invalid_commands_and_bad_mapping_rejected) ... ok
test_nonadjacent_self_collision_is_still_detected (test_joint_control.JointControlTests.test_nonadjacent_self_collision_is_still_detected) ... ok
test_original_assets_and_static_inspection (test_joint_control.JointControlTests.test_original_assets_and_static_inspection) ... ok
test_position_actuator_force_mapping_and_clamps (test_joint_control.JointControlTests.test_position_actuator_force_mapping_and_clamps) ... ok
test_a_fk_home_and_site_definition (test_kinematics.KinematicsTests.test_a_fk_home_and_site_definition) ... ok
test_b_fk_ik_full_pose_roundtrip (test_kinematics.KinematicsTests.test_b_fk_ik_full_pose_roundtrip) ... ok
test_c_dynamics_uses_actuators_without_live_qpos_overwrites (test_kinematics.KinematicsTests.test_c_dynamics_uses_actuators_without_live_qpos_overwrites) ... ok
test_collision_path_is_rejected_before_execution (test_kinematics.KinematicsTests.test_collision_path_is_rejected_before_execution) ... ok
test_d_unreachable_target_returns_failure_without_angles (test_kinematics.KinematicsTests.test_d_unreachable_target_returns_failure_without_angles) ... ok
test_default_ik_really_constrains_home_orientation (test_kinematics.KinematicsTests.test_default_ik_really_constrains_home_orientation) ... ok
test_e_failed_or_excessive_goal_never_steps (test_kinematics.KinematicsTests.test_e_failed_or_excessive_goal_never_steps) ... ok
test_fk_ik_do_not_mutate_live_data_or_model (test_kinematics.KinematicsTests.test_fk_ik_do_not_mutate_live_data_or_model) ... ok
test_headless_cli_unreachable_target (test_kinematics.KinematicsTests.test_headless_cli_unreachable_target) ... ok
test_invalid_inputs_have_structured_failure (test_kinematics.KinematicsTests.test_invalid_inputs_have_structured_failure) ... ok
test_iteration_at_a_joint_limit_stays_bounded (test_kinematics.KinematicsTests.test_iteration_at_a_joint_limit_stays_bounded) ... ok
test_iteration_budget_and_near_singularity (test_kinematics.KinematicsTests.test_iteration_budget_and_near_singularity) ... ok
test_jacobian_matches_translation_and_rotation_finite_differences (test_kinematics.KinematicsTests.test_jacobian_matches_translation_and_rotation_finite_differences) ... ok
test_rotation_error_near_zero_and_pi (test_kinematics.KinematicsTests.test_rotation_error_near_zero_and_pi) ... ok
test_site_survives_model_regeneration (test_kinematics.KinematicsTests.test_site_survives_model_regeneration) ... ok
test_step_and_joint_bounds_are_respected (test_kinematics.KinematicsTests.test_step_and_joint_bounds_are_respected) ... ok

----------------------------------------------------------------------
Ran 26 tests in 9.340s

OK

```

## Viewer 与运行命令

实际通过 X11/XWayland 运行默认末端控制，8 s 仿真正常结束，退出码为 0。已捕获并查看窗口，确认机械臂、绿色末端参考点和半透明红色目标球；最后一次标记外观截图在 `outputs/ee_viewer_final.png`。标记仅用于显示，球的半径不代表误差容差。

```text
PASS actual X11 EE Viewer: before/after window screenshots; changed RGB fraction=0.02386194; full 8s run exit=0
PASS Viewer close: exit=0; incomplete validation explicitly reported
PASS Viewer sigint: exit=0; incomplete validation explicitly reported
PASS Viewer duration: exit=0; incomplete validation explicitly reported
PASS final target marker and full Viewer run: exit=0
```

窗口关闭、SIGINT（Ctrl+C 等效）、duration 提前结束均正常退出，明确报告完整验收未完成。沿用原 Viewer 清理逻辑；没有重新验收真实鼠标旋转缩放或 native Wayland。

```bash
conda activate realman_rl
# 默认 +X 5 mm，无界面
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py
# Viewer
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --viewer
# 世界坐标系增量，单位 mm
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --delta-mm 0 0 5
# 绝对世界位置，单位 m
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --target 0.010616691 -0.000145159 1.06022729
# 重新生成模型；ee_site 会保留
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/build_dynamics_model.py
```

## 保留的问题与边界

- 临时 site 不是厂家标定 TCP；实物代际、零位、惯量和相机安装方式仍待核实。
- 仅验证 home 附近小范围目标；DLS 是有界局部求解器，不保证任意目标成功或得到全局最近解。
- 完整位姿 IK 的朝向数值约束已验证，实际动力学仍有约 0.5° 朝向偏差和 2–2.4 mm 位置偏差；没有加入新的补偿器来隐藏它们。
- 仍使用已有凸碰撞包络及相邻装配过滤；离散路径筛查与运行时停止不是连续安全规划。
- 第一阶段原始相机惯量异常的修正、上游许可证声明不一致尚未获得厂家确认。
- 没有末端实时闭环、路径优化、夹爪、强化学习、ROS2 或真机通信；本任务到此停止。

## 默认 Demo 完整输出

以下是实际从 `/tmp` 运行 `control_ee.py --json ...` 的输出，退出码为 0；JSON 不改变控制过程。生成产物位于 Git 忽略的 `outputs/`，关键输出在本文件保留：

```text
Temporary ee_site: link_6 local origin, not a calibrated TCP.
Home world position [m]: [ 5.61669109e-03 -1.45158656e-04  1.06022729e+00]
Fixed target site-to-world rotation:
 [[ 8.66016975e-01  1.58358158e-03  5.00012091e-01]
 [-1.36918178e-03  9.99998746e-01 -7.95670621e-04]
 [-5.00012724e-01  4.45681933e-06  8.66018057e-01]]
IK: success=True, iterations=2, position_error_m=1.4035134426844543e-07, orientation_error_rad=3.0477053550513374e-07, reason=None
IK joint target [deg]: [-6.20453897e-03 -1.95987976e+01  2.99689622e+01  9.25239940e-03
  1.96298184e+01 -3.34113139e-03]
initial_position_m: [ 5.61669109e-03 -1.45158656e-04  1.06022729e+00]
motion_start_position_m: [ 7.24656699e-03 -1.20312470e-04  1.05873463e+00]
target_position_m: [ 1.06166911e-02 -1.45158656e-04  1.06022729e+00]
theoretical_position_m: [ 1.06166835e-02 -1.45222716e-04  1.06022717e+00]
actual_position_m: [ 1.24540130e-02 -1.20251543e-04  1.05869657e+00]
Actual terminal position error: 2.391544 mm
Actual terminal orientation error: 0.499543 deg
Maximum actual orientation error during motion/hold: 0.500271 deg
Nominal joint-interpolation orientation drift: 0.000019 deg
Actual site-to-world rotation:
 [[ 8.61632042e-01  1.10938364e-03  5.07532259e-01]
 [-9.10691582e-04  9.99999381e-01 -6.39766053e-04]
 [-5.07532655e-01  8.90375741e-05  8.61632518e-01]]

[PASS] settle home
  J1: target=   0.0000 deg actual=  -0.0000 deg error=-0.000000 deg
  J2: target= -20.0000 deg actual= -20.1408 deg error=-0.140762 deg
  J3: target=  30.0000 deg actual=  30.3323 deg error=+0.332318 deg
  J4: target=   0.0000 deg actual=   0.0272 deg error=+0.027202 deg
  J5: target=  20.0000 deg actual=  20.2918 deg error=+0.291765 deg
  J6: target=   0.0000 deg actual=  -0.0007 deg error=-0.000667 deg
  max final error=0.332318 deg; overshoot=0.000000 deg; tail range=9.839e-14 deg; tail speed=1.568e-12 deg/s; contacts=0
  max uncommanded joint travel=0.328393 deg
  peak torque [N*m]: [0.13418 2.69014 4.72626 0.09598 0.62621 0.005  ]

[PASS] end-effector target
  J1: target=  -0.0062 deg actual=  -0.0062 deg error=+0.000000 deg
  J2: target= -19.5988 deg actual= -19.7283 deg error=-0.129483 deg
  J3: target=  29.9690 deg actual=  30.3054 deg error=+0.336425 deg
  J4: target=   0.0093 deg actual=   0.0375 deg error=+0.028215 deg
  J5: target=  19.6298 deg actual=  19.9217 deg error=+0.291855 deg
  J6: target=  -0.0033 deg actual=  -0.0040 deg error=-0.000668 deg
  max final error=0.336425 deg; overshoot=0.028256 deg; tail range=3.565e-15 deg; tail speed=1.469e-12 deg/s; contacts=0
  max uncommanded joint travel=0.000000 deg
  peak torque [N*m]: [7.25937e-04 2.47170e+00 4.70298e+00 9.87976e-02 6.11745e-01 9.34727e-04]
[PASS] End-effector dynamics validation; simulation time=8.000 s

```
