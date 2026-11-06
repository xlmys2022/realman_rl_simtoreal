# 03-A 最小Gymnasium环境验证

日期：2026-11-06。分支：`feat/gymnasium-env`。全部安装及Python验证均在现有 `realman_rl` 环境中执行；没有训练、真机连接、提交或推送。

## 文件与依赖

新增：

- `src/realman_rl_simtoreal/envs/__init__.py`：轻量包入口，不注册或创建环境。
- `src/realman_rl_simtoreal/envs/reach_env.py`：`ReachEnv`、固定目标、动作/观测、奖励及终止语义。
- `scripts/smoke_reach_env.py`：无界面固定动作演示、可选官方检查及JSON结果。
- `tests/test_reach_env.py`：27项API、实际控制、安全失败和SDK隔离测试。
- 本报告：运行证据和限制。

修改：`simulation/control.py`仅在原实现后新增 `advance_control_cycle` 与 `ControlCycleError`；requirements锁定新增依赖；README和architecture更新使用约定。原 `load_model`、`initialize`、DLS、`check_motion`、`execute_motion`、`run_segment`、`ForceMonitor`及厂家模块未改写。

安装前预检仅需新增3个包，随后更新editable安装和元数据：

```bash
conda activate realman_rl
python -I -m pip install --no-build-isolation -e /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I -m pip --no-cache-dir check
```

实际新增 `gymnasium==1.4.0`、`cloudpickle==3.1.2`、`Farama-Notifications==0.0.6`。既有包版本没有变化，包括MuJoCo 3.17.0、NumPy 2.4.6与可选SDK。`pip check`通过，未安装任何训练框架或Gymnasium额外环境依赖。pyproject沿用动态读取requirements的配置，版本仍为0.1.0。安装来源与旧模型一样依赖editable源码仓库。

## 环境接口与坐标

直接使用 `from realman_rl_simtoreal.envs.reach_env import ReachEnv`；当前没有自动注册Gymnasium ID。默认参数：

| 参数 | 默认值及意义 |
|---|---|
| action_scale | 每轴0.001m；世界XYZ平移增量，单位动作对应1mm |
| control_period | 0.05s，必须为模型2ms时间步的正整数倍 |
| max_episode_steps | 200，正常全周期最多10s仿真时间 |
| success_threshold | 实际位置距固定目标≤0.001m |
| gravity_compensation | True，沿用现有编译前补偿及关节总力矩限制；False可保留PD基线 |
| render_mode | None或`ansi`；无窗口、无RGB图像 |

动作空间 `Box(-1, 1, shape=(3,), dtype=float32)`。每轴线性缩放，三轴同时为1时增量范数是√3mm；没有隐含的单位球投影。形状错误、NaN/Inf或超出归一化范围的动作直接抛ValueError，状态、时间和episode计数不变。

命令更新为 `p_command_next = p_command + action_scale * action`，不把实际测量位置作为下一次积分起点。姿态目标始终是home对应的site-to-world旋转矩阵。`ee_site`仍是link_6原点，未改成厂家TCP。

Dict观测全部为float64的独立副本：

| 字段/切片 | 形状 | 单位与内容 |
|---|---|---|
| observation[0:6] | 6 | 六关节实际q，rad |
| observation[6:12] | 6 | 六关节实际qvel，rad/s |
| observation[12:15] | 3 | 实际ee_site世界位置，m |
| observation[15:18] | 3 | 固定目标减实际末端，m |
| observation[18:21] | 3 | 独立的笛卡尔命令参考，m |
| observation[21:27] | 6 | 当前执行器ctrl，rad |
| achieved_goal | 3 | 实际末端世界位置，m |
| desired_goal | 3 | 固定目标世界位置，m |

增加最后9个参考/控制量，避免仅提供机器人物理量却隐藏累积命令状态。空间使用无限边界以容纳终止故障状态，不作为物理安全保证；运行时另行检查有限值及实际限位。

## Reset与固定周期控制

`reset(seed=..., options=None)`调用 `super().reset(seed=seed)`，然后在现有MjData上执行官方 `mj_resetDataKeyframe(home)` 和 `mj_forward`。清空计数、速度/控制状态与失败标记，命令参考置home，仿真时间为0。目标固定为home世界位置 +X 5mm；构造时已用当前模型的IK和原路径检查验证。没有随机化、额外稳态等待或Viewer。

每个正常step：

1. 验证动作、实际关节状态和局部命令区域。
2. 用上个周期实际合法关节命令作初值，调用原MuJoCo全位姿DLS IK。独立计算数据不影响运行状态。
3. 复核IK状态、有限值、模型限位、home ±5°、理论位姿及51点离散关节路径碰撞。
4. 以原位置执行器运动；每周期关节指令变化上限为 `10°/s * control_period`，默认0.5°。只限制合法解的命令变化率，不把非法IK解裁剪为合法解。
5. 使用 `3a²-2a³` 平滑插值ctrl，默认执行25次 `mj_step`。曲线内最大指令变化率约15°/s；实际速度另限30°/s，均为实验参数，不是厂家额定值。
6. 每个物理步检查有限数值、警告、时间、位置、速度、接触和总驱动力。复用ForceMonitor，在mj_step后、mj_forward前审计力通道；之后刷新最新site位置/接触再返回观测。

正常step不赋值实际qpos或qvel，也不调用run_segment。一次完整step恰好推进0.05s，即20Hz**仿真控制频率**；无界面程序不按墙钟限频。

## 奖励与结束原因

奖励为 `-||achieved_goal-desired_goal||₂`，数值单位m；没有成功奖励或失败惩罚附加项。`compute_reward`也支持批量XYZ数组，为后续目标接口预留能力。

| termination_reason | terminated / truncated | 处理 |
|---|---|---|
| success | True / False | 实际位置进入1mm目标球，不使用IK理论位置判断 |
| ik_failure | True / False | 有界局部求解或位姿复核失败；不代表全局不可达 |
| unreachable_target | True / False | 保守几何reach bound可证明超界；与普通局部求解失败区分 |
| workspace_limit | True / False | 超过home周围10mm实验命令球，不等于真实工作空间边界 |
| joint_limit | True / False | 非法关节目标、home ±5°命令边界或实际模型关节限位异常 |
| collision | True / False | 离散预检或实际物理步检测到穿透超过0.1mm，沿用原阈值 |
| velocity_limit / orientation_limit | True / False | 实际速度超过30°/s或周期末朝向偏差超过1° |
| numerical_failure / drive_failure | True / False | 非有限数值、警告、异常时间或驱动力/外力旁路审计失败 |
| time_limit | False / True | 没有成功或失败时达到最大环境步数 |

成功/安全失败优先于同一步的超时。结束后必须reset；不允许继续执行。IK或预检拒绝时不提交命令参考、不改ctrl、不推进物理时间；物理阶段失败可提前停止。info中的`physics_steps`和控制摘要明确报告0或部分步数，不伪称完成固定周期。

若故意注入非有限状态，环境终止，返回上次有限观测的副本并设 `observation_valid=False`、`orientation_error_rad=None`；它是明确标记的旧快照，不是当前有效测量。运行状态不被修复或瞬移。正常测试的观测、状态和奖励均有限。

保留全部原碰撞过滤及力矩限制。补偿已经合入 `qfrc_actuator`，监控不会重复相加。关节模型限位检查沿用原控制的1e-4rad软约束数值容差；正常演示与真实限位相距很远。

## 实际运行与测试

基线：修改前原85项全部通过，30.978秒。最终完整回归：**112/112通过，33.703秒**（原85项源文件和断言未改，新增27项）。命令：

```bash
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/smoke_reach_env.py --check --json /home/mr-beaver/Documents/robot/realman_rl_simtoreal/outputs/reach_env/smoke.json
```

[Gymnasium官方check_env](https://gymnasium.farama.org/api/utils/#environment-checking)默认检查选项通过。保留3条提示：观测Box最小/最大值为无限，以及直接构造实例无注册spec、无法自动实例化替代render模式。没有关闭checker检查来掩盖问题；另有测试验证ansi render和重复close。API遵循 [官方Env约定](https://gymnasium.farama.org/api/env/)。

默认动作序列为5次 `[1,0,0]`，然后 `[0,0,0]`：

| 项目 | 实测值 |
|---|---|
| 初始世界位置m | [0.005616691095, -0.000145158656, 1.060227293452] |
| 目标世界位置m | [0.010616691095, -0.000145158656, 1.060227293452] |
| 成功时实际世界位置m | [0.009714122782, -0.000144299199, 1.060226323605] |
| 环境步 / 物理步 / 仿真时间 | 6 / 150 / 0.300s |
| 实际X位移 | 4.097432mm |
| 最终绝对目标误差 | 0.902569mm，低于本任务1mm阈值 |
| 最终姿态误差 | 3.379276e-5rad，约0.001936° |
| 六轴峰值总驱动力矩N·m | [0.020435, 2.943614, 4.790414, 0.103647, 0.615562, 0.000989] |
| 六轴峰值速度°/s | [0.027525, 1.593449, 0.155004, 0.053140, 1.478327, 0.015089] |
| 接触 / 力矩饱和物理步 | 0 / 0 |

第五步的命令参考已经达到理论目标，但实际误差仍为1.808213mm，没有提前宣布成功；第六步实际状态进入目标球才结束。此处是运动中越过成功阈值，不是稳态精度测量或驻留验证。

新增测试还验证：同seed的多步状态完全相同、正常step的每次qpos变化均来自mj_step、参考不累积实测偏差、非法成功IK解被拒绝、超时与成功优先级、外力旁路、限速、无SDK/无桌面及其他工作目录运行。真实碰撞反例在第5物理步停止（10ms）；预检碰撞拒绝为0步。初次碰撞夹具修改静态地面后未重算模型常量，导致未生成接触；已在独立scratch数据上调用mj_setConst修正夹具，正式模型未改。

本地证据在Git忽略的 `outputs/reach_env/`：baseline_tests.log、all_tests.log、env_tests.log、smoke.log、smoke.json、smoke_summary.json和audit.json。审计确认17个资产SHA256一致；原model、kinematics、ee_control、vendor及原85项测试源文件不变；旧control实现仍为当前文件完全一致的前缀。既有依赖版本无变化。SDK缺失验证通过导入拦截完成，没有卸载或调用厂家接口。

## 已知限制

这是局部固定目标环境，没有随机目标、障碍物规划、GUI/RGB render、厂家控制或训练。姿态是IK终点约束，关节插值和实际动态不保证全过程严格不变。离散路径预检也不保证连续轨迹无碰撞，实际物理步检查仍有必要。

成功只按距离判断，没有速度、驻留时间或稳态要求。奖励没有专门惩罚失败；超出实验区域会终止，因此尚未评价任意策略的学习行为。Dict和批量reward为HER预留接口，但未实现HER；未来重标目标时还需同步处理observation中目标相对位置的字段。

重力补偿依赖当前仿真模型的质量参数。真实型号、TCP及厂家/MuJoCo完整运动学差异仍未解决。本次结果不能外推为真实机械臂精度或安全保证。
