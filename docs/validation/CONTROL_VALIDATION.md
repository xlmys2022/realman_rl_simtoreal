# 任务 02-B+：重力补偿与末端控制精度对照

验证日期：2026-11-05（Asia/Shanghai）。现有 Conda `realman_rl`，Python 3.11.17、MuJoCo 3.17.0、NumPy 2.4.6。无新增依赖，无 Git commit/push。下文全部数值来自实际物理积分和保存的诊断，不是示意数据。

## 误差定义与基线诊断

设 `p0` 为名义 home FK 位置，`ps` 为 home 指令保持 3 s 后的实际位置，`pt` 为共同的绝对目标，`pf` 为实际末态。每组实验共享 `p0`、`pt` 和目标关节角；不把基线偏移加到目标上来掩盖绝对误差。

```text
home_offset = ps - p0
expected_displacement = pt - p0
actual_displacement = pf - ps
relative_error = actual_displacement - expected_displacement
absolute_error = pf - pt = home_offset + relative_error
```

等式适用于向量；三个向量的模长不能简单相加。home-only 场景没有后续运动，定义 pf=ps，故相对运动误差为 0，而绝对误差仍是 home 偏移。姿态误差为实际旋转到目标旋转的 SO(3) 最短角。

基线 +X 5 mm：home 偏移向量为 `[1.6298758932005997, 0.024846186866523855, -1.4926623352065072]` mm，模长 **2.210238360 mm**。
从稳定 home 到终点的实际位移为 `[5.207445990723356, 6.0926486453280644e-05, -0.038060552838503625]` mm；相对误差向量为 `[0.2074459907233546, 6.0926486453280644e-05, -0.038060552838503625]` mm，模长 **0.210908626 mm**；最终绝对误差 **2.391544403 mm**。
home 的实际姿态偏移为 0.484050549°；+X 目标终点为 0.499542925°。这三类位置误差及两个姿态结果分别记录，没有混用。

稳态 home 分轴结果（误差均为实际角−目标角；力矩正号为保持姿态所需的广义驱动力）：

| 关节 | 目标角 ° | PD 实际角 ° | PD 角误差 ° | 名义 home 重力 N·m | PD 稳态重力 N·m | PD 稳态输出 N·m | 补偿后角误差 ° | 补偿模式 PD 输出 N·m | 补偿模式总驱动 N·m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J1 | 0 | -6.01177186e-16 | -6.01177186e-16 | 0 | 0 | 6.99036218e-16 | -3.48525384e-19 | -3.06529558e-17 | -3.06529558e-17 |
| J2 | -20 | -20.1407617 | -0.140761669 | 2.44382627 | 2.45675458 | 2.45675458 | 0 | 0 | 2.44382627 |
| J3 | 30 | 30.3323175 | 0.332317513 | -4.6053958 | -4.64002781 | -4.64002781 | 0 | 0 | -4.6053958 |
| J4 | 0 | 0.0272016719 | 0.0272016719 | -0.0932244615 | -0.0949517474 | -0.0949517474 | -7.90017749e-18 | 3.99278901e-17 | -0.0932244615 |
| J5 | 20 | 20.2917647 | 0.29176471 | -0.605113864 | -0.61107058 | -0.61107058 | 0 | 0 | -0.605113864 |
| J6 | 0 | -0.000667250781 | -0.000667250781 | 0.000905245986 | 0.000931657845 | 0.000931657845 | -1.67646587e-17 | 2.39004326e-17 | 0.000905245986 |

名义与稳态重力不同是因为姿态发生了偏移，并非改变了质量。基线稳态速度接近 0、阻尼力可忽略，PD 输出与该实际姿态的纯重力平衡；因此 PD 需要非零角误差来产生保持力矩。补偿后主要由重力前馈提供保持力矩，PD 的稳态输出接近 0，但关节总驱动力并不为 0。
home 位移的 Jacobian 一阶预测为 `[1.633242030306006, 0.024462715785215812, -1.484300003709123]` mm，与直接 FK 的实际偏移接近；精确结果始终使用 FK/实际 site，不用线性近似替代测量。

### qfrc_bias 的解释

模型满足 `M(q)*qacc + c(q,qvel) + g(q) = tau_drive + tau_passive + tau_contact + tau_external`。`qfrc_bias` 是左侧的速度相关偏置加重力，不包含 M*qacc；运动时不能当作纯重力。
`gravity_torque()` 创建独立 MjData，将诊断用 qpos 设为待测姿态、诊断 qvel 设为 0，再执行 mj_forward 读取 bias，得到 g(q)。它从不清零实际速度或改写实际 qpos。用全部连杆质心 Jacobian 的 `sum(Jcom.T @ (-mass*gravity))` 独立核对该结果。
在速度 `[0.6,-0.5,0.4,-0.3,0.2,-0.1] rad/s` 的前向诊断中，`norm(qfrc_bias-g(q))≈0.266009489 N·m`，说明运动偏置不等于纯重力。该诊断没有推进大幅物理运动。

## 唯一选定的补偿方案与合力限幅

保留原六个 position PD 执行器。启用时在 MjSpec 编译前给六个运动连杆及两个固定相机 Body 设置 `gravcomp=1`；固定底座不补偿。给六个 joint 设置 `actuatorgravcomp=true`，保留原 joint actuatorfrcrange / actuatorfrclimited。
力通道与计算顺序如下：

```text
每个 position 产生 PD 力矩，并保留原执行器级限幅
各执行器力矩映射到关节，与原生重力补偿合成
tau_drive = clip(sum(mapped_servo_torque) + gravity_compensation, joint_total_limits)
实际总驱动力矩读取 qfrc_actuator
```

`qfrc_gravcomp` 是补偿诊断量，开启 actuatorgravcomp 后已经加到 qfrc_actuator，不能再加一次。此时 qfrc_passive 仅含原物理阻尼等被动力，不含补偿。程序同时监控两个外力入口为 0，防止使用 qfrc_applied/xfrc_applied 绕过电机限幅。被动阻尼与接触反力不属于电机驱动力矩；不能据电机限幅声称任何外部受力都被限幅。
依据：[MuJoCo 3.17.0 actuatorgravcomp / actuatorfrcrange](https://mujoco.readthedocs.io/en/3.17.0/XMLreference.html#body-joint-actuatorgravcomp)、[该版本引擎的补偿合成与关节级限幅顺序](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_forward.c)。
仅修改已经编译的 body_gravcomp 数组不会可靠更新编译时重力补偿元数据，实测原基线仍没有产生补偿。因此实现通过 MjSpec 配置后编译，得到 ngravcomp=8，并核对六个路由标志与实际通道。
加载接口：`load_model(gravity_compensation=False)` 保持原 PD；True 确定性地从同一场景编译补偿配置。原 build_dynamics_model.py、基线 MJCF、URDF、mesh 均不变，重新生成原模型后仍能使用该加载选项。不是运行中切换；每个实验重新初始化。

力矩边界沿用 `[±60,±60,±30,±30,±10,±10] N·m`。每个物理步在 mj_step 后、任何 mj_forward 刷新前观测用于该步的总驱动力与补偿通道，避免只记录单个执行器或时间阶段不一致的数据。

### 限幅与旁路反例（仅前向计算，无极端运动）

| 反例 | 关节 | PD/其他电机生成 N·m | 原始补偿 N·m | 未限幅合计 N·m | 实际总驱动 N·m |
|---|---|---:|---:|---:|---:|
| 单 PD + 补偿 | J3 | -30.000000 | -4.605396 | -34.605396 | -30.000000 |
| 单 PD + 补偿 | J2 | 60.000000 | 2.443826 | 62.443826 | 60.000000 |
| 两个电机 + 补偿 | J2 | -60 + -60 | 2.443826 | -117.556174 | -60.000000 |
| 两个电机 + 补偿 | J2 | 60 + 60 | 2.443826 | 122.443826 | 60.000000 |

测试用额外电机只存在于临时限幅模型中，正常 A/B 模型仍为六个执行器。上表证明多个来源先合成、再按关节总限幅；没有仅依据 actuator_force 是否超限来判断。
另一个故意错误的配置只启用 Body 补偿、关闭 joint actuatorgravcomp：J2 电机力矩 60 N·m，另有被动补偿约 2.443826 N·m，合计 62.443826 N·m，超过 60。监控器明确拒绝这一配置；它从未用于实际对照积分。缺少关节总限幅或非零外力注入的配置也会被拒绝。

## 同条件 A/B 结果

共同条件：同一 home keyframe、相同名义 home、同一组 IK 目标角、相同 kp/kv/damping、重力 −9.81 m/s²、implicitfast、2 ms 时间步、原碰撞过滤与角度/力矩边界。模型参数数组一致性有自动测试。
每个控制器分别运行 home 3 s、三个末端实验各 8 s（3 s home、2 s 平滑切换、3 s 末态保持），以及原关节序列 39 s，共 66 s 仿真、33000 个物理步。每个场景都重新从同一初态开始，不复用另一控制器的状态。表内峰值覆盖该场景全时段，包括 home 收敛。

| 场景 | 控制器 | home 偏移 mm | 相对误差 mm | 最终绝对误差 mm | 姿态误差 ° | 最大总驱动力 N·m | 最大关节速度 °/s |
|---|---|---:|---:|---:|---:|---:|---:|
| home 保持 | A: PD | 2.21023836 | 0 | 2.21023836 | 0.484050549 | 4.72626235 | 3.36233669 |
| home 保持 | B: PD+重力补偿 | 0 | 0 | 0 | 1.38905009e-17 | 4.6053958 | 2.33119155e-14 |
| 世界 +X 5 mm | A: PD | 2.21023836 | 0.210908626 | 2.3915444 | 0.499542925 | 4.72626235 | 3.36233669 |
| 世界 +X 5 mm | B: PD+重力补偿 | 0 | 0.000140351345 | 0.000140351345 | 1.74620653e-05 | 4.66583924 | 0.299434415 |
| 世界 −X 5 mm | A: PD | 2.21023836 | 0.209963212 | 2.03660137 | 0.468520013 | 4.72626235 | 3.36233669 |
| 世界 −X 5 mm | B: PD+重力补偿 | 0 | 0.000156523207 | 0.000156523207 | 3.22353548e-05 | 4.61024989 | 0.29393676 |
| 世界 +Z 5 mm | A: PD | 2.21023836 | 0.123083082 | 2.25188851 | 0.485840292 | 4.72626235 | 3.36233669 |
| 世界 +Z 5 mm | B: PD+重力补偿 | 0 | 0.00187249561 | 0.00187249561 | 0.000318042072 | 4.61423701 | 1.63466721 |

小于显示精度的 home 偏移记作 0，含义是当前精确匹配仿真中的浮点量级，不是真机零误差。三个补偿实验均达到探索性 <1 mm 终点误差目标，没有改变原安全阈值、增益或目标定义。
补偿后末态已接近 IK 目标关节角，残留绝对末端误差基本由原来的 IK 数值残差决定（约 0.000140 / 0.000157 / 0.001872 mm）。没有收紧 IK 容差或重新求一组对 B 更有利的角度。

### 实际相对位移向量

| 任务 | A 实际位移 [x,y,z] mm | B 实际位移 [x,y,z] mm | 期望位移 mm |
|---|---|---|---|
| x_plus_5mm | [5.20744599, 6.09264865e-05, -0.0380605528] | [4.99999238, -6.40594077e-05, -0.00012464651] | [5, 0, 0] |
| x_minus_5mm | [-5.20687406, -0.000225195374, 0.0358834412] | [-4.99999675, -7.43362422e-06, -0.000156312891] | [-5, 0, 0] |
| z_plus_5mm | [0.107411242, -0.00175522205, 5.06007653] | [-5.34929902e-06, -3.01847821e-08, 4.99812751] | [0, 0, 5] |

### 六关节目标与实际终值

各单元依次为 J1..J6，单位度。更多力通道、坐标、姿态矩阵、每轴峰值和完整误差向量在 JSON 中保留。

| 场景 | 控制器 | 目标关节角 ° | 实际关节角 ° |
|---|---|---|---|
| home | pd | [0.0000000, -20.0000000, 30.0000000, 0.0000000, 20.0000000, 0.0000000] | [-0.0000000, -20.1407617, 30.3323175, 0.0272017, 20.2917647, -0.0006673] |
| home | gravity | [0.0000000, -20.0000000, 30.0000000, 0.0000000, 20.0000000, 0.0000000] | [-0.0000000, -20.0000000, 30.0000000, -0.0000000, 20.0000000, -0.0000000] |
| x_plus_5mm | pd | [-0.0062045, -19.5987976, 29.9689622, 0.0092524, 19.6298184, -0.0033411] | [-0.0062045, -19.7282808, 30.3053877, 0.0374678, 19.9216738, -0.0040095] |
| x_plus_5mm | gravity | [-0.0062045, -19.5987976, 29.9689622, 0.0092524, 19.6298184, -0.0033411] | [-0.0062045, -19.5987976, 29.9689622, 0.0092524, 19.6298184, -0.0033411] |
| x_minus_5mm | pd | [0.0055088, -20.3938192, 30.0168019, -0.0079206, 20.3769852, 0.0026556] | [0.0055088, -20.5457620, 30.3449263, 0.0182474, 20.6686590, 0.0019894] |
| x_minus_5mm | gravity | [0.0055088, -20.3938192, 30.0168019, -0.0079206, 20.3769852, 0.0026556] | [0.0055088, -20.3938192, 30.0168019, -0.0079206, 20.3769852, 0.0026556] |
| z_plus_5mm | pd | [0.0000000, -18.8581439, 27.8039597, -0.0000202, 21.0538661, 0.0000215] | [0.0000000, -18.9854436, 28.1250584, 0.0243998, 21.3456422, -0.0006438] |
| z_plus_5mm | gravity | [0.0000000, -18.8581439, 27.8039597, -0.0000202, 21.0538661, 0.0000215] | [0.0000000, -18.8581439, 27.8039597, -0.0000202, 21.0538661, 0.0000215] |

### 原有 13 段关节运动对照

保持原来 3 s home、逐轴 +3°/返回的 39 s 序列，插值和保持时间不变。JSON 中逐段保留目标/实际关节角、目标/实际 site 位姿、home 偏移、相对/绝对误差、姿态误差及控制统计。

| 控制器 | 通过段数 | 最大末态关节误差 ° | 全程峰值总驱动 N·m | 全程峰值速度 °/s | 正常饱和步数 |
|---|---:|---:|---:|---:|---:|
| pd | 13/13 | 0.367748688 | 5.3060656 | 4.54068697 | 0 |
| gravity | 13/13 | 4.99942089e-06 | 5.24979653 | 4.43142086 | 0 |

### 每轴总驱动力峰值与限制

以下峰值为该模式所有场景的最大绝对关节总驱动，不是单个电机输出。

| 关节 | 原总力矩限制 ±N·m | A 全场景峰值 N·m | B 全场景峰值 N·m |
|---|---:|---:|---:|
| J1 | 60 | 0.134178438 | 0.0963661418 |
| J2 | 60 | 2.88817576 | 2.86918773 |
| J3 | 30 | 5.3060656 | 5.24979653 |
| J4 | 30 | 0.132075192 | 0.129137654 |
| J5 | 10 | 0.665588449 | 0.657972767 |
| J6 | 10 | 0.00705273228 | 0.00700282432 |

所有正常实验无饱和、无非预期接触、无 NaN/Inf 或数值警告。最后 0.5 s 的每轴峰峰波动和速度均满足原阈值 ≤0.05°、≤0.1°/s；未放宽原 ≤0.5° 关节末态误差、超调或穿透检查。B 的关节返回段仍有有限收敛残差，不能声称任意时刻零误差。

## 实现、复现与测试

- 新增 `scripts/compare_controllers.py`：共同目标的 A/B 实验、纯重力诊断、统一指标和逐步力通道审计。
- 新增 `tests/test_gravity_compensation.py`：13 项测试，包括静态力平衡、模型参数不变、多执行器合力限幅、被动力旁路拒绝、运动偏置分离、实际 qpos 连续性和 <1 mm 探索目标。
- 修改 `control_joints.py`：加载时可选补偿、要求关节总限幅、复用段执行的观测钩子；原 PD 缺省不变。修改 `control_ee.py`：仅向已有段执行透传可选步观测器。
- README/SOURCE 与历史验证记录添加入口和说明；基线模型生成脚本、所有机器人资源、原有增益和 FK/IK 求解参数均未改。

```bash
conda activate realman_rl
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_controllers.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_controllers.py --controller pd
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_controllers.py --controller gravity
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

`--json /absolute/path/results.json` 可指定输出，默认保存到 `outputs/control_comparison.json`。原始数据、前向饱和诊断及测试日志在 outputs/（Git 忽略）。新增 13 项与原有 26 项全部通过，共 39 项；补偿运行核对了 4000 个末端物理步的 qpos 连续性，没有使用直接改写实际状态的办法。

```text
test_all_normal_steps_are_bounded_quiet_and_collision_free (test_gravity_compensation.GravityCompensationTests.test_all_normal_steps_are_bounded_quiet_and_collision_free) ... ok
test_compensated_execution_never_rewrites_live_qpos (test_gravity_compensation.GravityCompensationTests.test_compensated_execution_never_rewrites_live_qpos) ... ok
test_error_definitions_are_vector_consistent (test_gravity_compensation.GravityCompensationTests.test_error_definitions_are_vector_consistent) ... ok
test_home_gravity_equilibrium_and_improvement (test_gravity_compensation.GravityCompensationTests.test_home_gravity_equilibrium_and_improvement) ... ok
test_matched_ik_targets_and_below_one_mm_exploratory_goal (test_gravity_compensation.GravityCompensationTests.test_matched_ik_targets_and_below_one_mm_exploratory_goal) ... ok
test_missing_joint_limits_and_external_injection_rejected (test_gravity_compensation.GravityCompensationTests.test_missing_joint_limits_and_external_injection_rejected) ... ok
test_moving_bias_is_not_pure_gravity_and_probe_is_isolated (test_gravity_compensation.GravityCompensationTests.test_moving_bias_is_not_pure_gravity_and_probe_is_isolated) ... ok
test_multiple_actuators_and_compensation_share_one_joint_limit (test_gravity_compensation.GravityCompensationTests.test_multiple_actuators_and_compensation_share_one_joint_limit) ... ok
test_only_compensation_configuration_changes (test_gravity_compensation.GravityCompensationTests.test_only_compensation_configuration_changes) ... ok
test_passive_bypass_configuration_is_detected_and_rejected (test_gravity_compensation.GravityCompensationTests.test_passive_bypass_configuration_is_detected_and_rejected) ... ok
test_servo_plus_compensation_saturates_at_joint_total_bounds (test_gravity_compensation.GravityCompensationTests.test_servo_plus_compensation_saturates_at_joint_total_bounds) ... ok
test_static_compensation_is_in_actuator_not_passive_channel (test_gravity_compensation.GravityCompensationTests.test_static_compensation_is_in_actuator_not_passive_channel) ... ok
test_zero_velocity_gravity_matches_com_jacobians (test_gravity_compensation.GravityCompensationTests.test_zero_velocity_gravity_matches_com_jacobians) ... ok
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
Ran 39 tests in 25.994s

OK
```

## 实际局限与真机复用

这些极小终点误差依赖补偿模型与被控仿真对象完全匹配：两者共享连杆质量、质心、几何/关节坐标和重力方向。纯静态重力项取决于质量和质心、运动学与重力，**不直接依赖转动惯量张量**；惯量仍影响瞬态与仿真动力学可信度。没有验证参数偏差、载荷变化、柔性、噪声、时延或真实摩擦。
这里只补偿重力，不补偿运动时的科里奥利/离心项，也不是末端实时闭环、连续碰撞规划或全工作空间精度保证。速度反馈和有限增益仍产生运动过程中的跟踪滞后；终点良好不能代表整条轨迹达到同样精度。
不能直接把 body/joint 的 MuJoCo 配置部署到真实机械臂。重力前馈思路和总力矩限幅原则可沿用，但需要核实真机控制接口、厂家内部重力补偿、工具/载荷参数、安装方向和实际增益/安全机制，避免重复补偿。当前临时 ee_site 也未标定为真实 TCP。没有加入真机 API、ROS2、夹爪或强化学习。
本任务未新增图形程序或重新验证 Viewer；结论来自可重复的无界面物理积分及力通道测试。实验到此结束。

## 对照程序摘要输出

```text
Errors are norms [mm]; relative error is measured from settled home, absolute error from nominal target.
scenario        controller   home_offset  relative_err  absolute_err  rotation_deg  peak_total_Nm  peak_speed_deg/s
home            pd             2.210238      0.000000      2.210238      0.484051       4.726262          3.362337
home            gravity        0.000000      0.000000      0.000000      0.000000       4.605396          0.000000
x_plus_5mm      pd             2.210238      0.210909      2.391544      0.499543       4.726262          3.362337
x_plus_5mm      gravity        0.000000      0.000140      0.000140      0.000017       4.665839          0.299434
x_minus_5mm     pd             2.210238      0.209963      2.036601      0.468520       4.726262          3.362337
x_minus_5mm     gravity        0.000000      0.000157      0.000157      0.000032       4.610250          0.293937
z_plus_5mm      pd             2.210238      0.123083      2.251889      0.485840       4.726262          3.362337
z_plus_5mm      gravity        0.000000      0.001872      0.001872      0.000318       4.614237          1.634667
pd: original joint sequence 13/13 passed; max joint error=0.367748688 deg; peak total drive=5.306066 N*m; saturated steps=0
gravity: original joint sequence 13/13 passed; max joint error=0.000004999 deg; peak total drive=5.249797 N*m; saturated steps=0
```
