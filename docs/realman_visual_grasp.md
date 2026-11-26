# 真机传统视觉抓取验收

固定俯视流程沿同一末端姿态完成张爪、水平对准、预抓取、直线下降、力控闭合和垂直抬升。视觉给出 XY、方块边方向及绝对 Z 校核值。下降终点取经过现场验收的 `Grip_plus` 夹持 TCP 高度。现有仿真场景和 PPO 仍使用各自的配置。

## 现场读回与绝对 Z

- 用户确认桌面水平，基座安装面到方块桌面的实测垂距为 40 mm。桌面到第二关节中心约 214 mm，和安装高度加实机 DH 的 `d1=172 mm` 所得约 212 mm 接近。按现有基座轴定义，执行平面采用水平桌面 `Z=-0.040 m`，20 mm 方块在桌面时的参考中心是 `Z=-0.030 m`。
- `calibration.20261126.json` 使用 `Grip_plus` 为父系。A/B/C 回放中方块中心的 Z 仍偏高 12.44、12.92、12.53 mm，周边支撑点分别偏高 12.66、13.53、13.01 mm。方块中心相对局部深度支撑面的距离为 9.42–9.79 mm。高度差出现在方块和周围点云，中心减半边长的补偿只能解释约 0.2–0.6 mm。[三组 Z 诊断](../outputs/validation/realman_installation/hand_eye_20261126_height.json)
- 以 A/B 首帧中靠近桌面的两个固定黑点做纯 RGB 双视图三角化，得到 Z 约 −34.0、−33.8 mm，重投影误差小于 0.35 px。对应像素周围的 RGB-D 读数约 −30.4 至 −28.5 mm。三角化共用当前 FK 和手眼外参，独立于深度读数。结果提示深度链和两种方法共用的坐标链均需核查，样本不足以分解误差或把偏差应用到其他位置。局部点云斜率作为诊断量记录。部署保留水平平面和原始手眼矩阵。
- 本轮真机只读返回 `Base / Grip_plus`、末端最大线速度 `0.25 m/s` 和动力学碰撞等级 `0`。首次动作验收由操作员在示教器按现场要求启用碰撞防护并复查。当前回包记录夹住方块时 `opening_raw=241`、`force_g=787`、`mode=6`，夹爪在线、使能且无报错。张满读回 `1001` 和力反馈 `787 g` 属于不同的反馈通道。力控指令值按厂家定义单独设定。[现场状态快照](../outputs/validation/realman_installation/control_readonly_20261126.json) · [睿尔曼夹爪接口](https://develop.realman-robotics.com/robot/apipython/classes/gripperControl/) · [碰撞等级与线速度接口](https://develop.realman-robotics.com/robot/apipython/classes/tipVelocityParameters/)

绝对 Z 验收需要在抓取区域放置可独立测到基座 XYZ 的参考物。在至少两个 XY 位置比对其相机返回高度，同时使用已知高度差的量块复核 D435 的相对深度。记录基座安装面水平度并保留同帧 RGB-D、机械臂位姿和参考物实测值。当前 12 mm 残差会在规划阶段中止下降。

## 现场配置

从项目根目录在 `realman_rl` 环境运行。现场文件已填入确认的水平桌面基准，剩余参数按验收进度填写。未填写的 `null` 会显示字段名并中止对应命令。新站可从模板创建自己的文件。[本机现场配置](../assets/calibration/realman/grasp_site.local.json) · [配置模板](../assets/calibration/realman/grasp_site.example.json)

```bash
conda activate realman_rl
ARM_IP=192.168.1.18
CALIB=assets/calibration/realman/calibration.20261126.json
SITE=assets/calibration/realman/grasp_site.local.json
```

- `motion` 填写现场可用的基座系 TCP 工作空间、单段位移、速度百分比和线速度上限、反馈路径偏离上限及运动时限。Z 下界应覆盖现场核验的最小工具间隙，包含张开和闭合的夹指与附件。`T_flange_tcp` 从手眼配置的实测 TCP 字段读取。先用小段直线独立验收，控制器的最大线速度是换算上限。
- `gripper` 需要多次稳定读回的张满原始范围、空夹闭合值、实物力控停止时的原始开口和力读回范围。现场确认低速开合、允许的闭合力控指令及动作时限。`open_speed_raw`、`close_speed_raw` 为 SDK 速度量，`close_force_raw` 为力控阈值指令，`held_force_g` 为设备压力反馈范围。张满使用可显式限速的 SDK 松开接口，闭合使用单次限力夹取。读取到 `mode=6` 且开口和力反馈均落在验收范围才放行抬升。[EG2-4C2 控制及读回字段](https://develop.realman-robotics.com/robot/json/endTool/)
- `grasp` 需要实测张满净间距、能包住方块且两指避开支撑面的 `Grip_plus` 夹持 Z、实测夹紧轴 `x` 或 `y`。另外填入已验收的预抓取净高、试抬高度、速度和视觉容差。固定俯视下工具倾角容差至多 5°，夹指对边容差至多 15°。采样需有至少 5/8 帧提供位置和边方向，方块边按 90° 等价方向核对。`max_visual_z_error_m` 和 `max_target_shift_m` 上限均为半个方块边长。

张满实测以夹指内侧同一基准为准。在静止状态下读回夹爪并用量具测量物理间距。确认 `241` 对应稳态有物夹持后，通过多次读回确定上下限。现场首次下降前，需独立验收夹持高度、最低工具点净空与工件 XY。控制器的碰撞检测、限位和现场防护装置应保持可用。

## 终端验收顺序

1. 先核对设备、模式、碰撞防护、工具配置和相机只读定位。

   ```bash
   python -I scripts/diagnostics/check_realman_arm.py --ip "$ARM_IP" --gripper --kinematics --collision
   python -I scripts/demos/locate_real_cube.py --ip "$ARM_IP" --calibration "$CALIB"
   ```

   使用已保存的 A/B/C 复查水平桌面参照下的误差。

   ```bash
   SAMPLES=outputs/validation/realman_installation
   python -I scripts/diagnostics/audit_realman_vision.py compare \
     "$SAMPLES/overhead_A.npz" "$SAMPLES/overhead_B.npz" "$SAMPLES/oblique_C.npz" \
     --calibration "$CALIB" --capture-calibration assets/calibration/realman/calibration.local.json \
     --table-z-base-m -0.040 --output "$SAMPLES/height_recheck.json"
   ```

2. 独立验收小段直线及夹爪张开。`move` 只读取 `motion` 节，`open` 读取 `motion` 和 `gripper` 两节。先运行只读命令，核对目标与预检。确认现场条件后，在交互终端追加 `--execute`，按提示输入 `MOVE` 或 `OPEN`。张开验收应在夹爪下方净空且工件有承托时进行。

   ```bash
   read -rp '输入现场验收的 TCP X Y Z（m）和速度百分比 ' TARGET_X TARGET_Y TARGET_Z SPEED_PCT
   python -I scripts/demos/grasp_real_cube.py move --ip "$ARM_IP" --site "$SITE" \
     --calibration "$CALIB" --tcp "$TARGET_X" "$TARGET_Y" "$TARGET_Z" --speed-pct "$SPEED_PCT"
   python -I scripts/demos/grasp_real_cube.py open --ip "$ARM_IP" --site "$SITE" --calibration "$CALIB"
   ```

3. 使用摆在桌面上、无遮挡的单个绿方块。从经人工验收的俯视观察位置开始。先离线校验现场文件，再运行只读规划。规划打印抓取点、起点、预抓取点、下降点和抬升点。路径预检沿直线取样，用现有 MuJoCo TCP 运动学核对 FK、IK、关节限位及分支连续性。现场操作员和控制器碰撞防护负责实际扫掠空间与障碍物。

   ```bash
   python -I scripts/demos/grasp_real_cube.py grasp --validate-site --site "$SITE" --calibration "$CALIB"
   python -I scripts/demos/grasp_real_cube.py grasp --ip "$ARM_IP" --site "$SITE" --calibration "$CALIB"
   ```

4. 实测和只读规划均验收后，在交互终端分段确认。首次可在 `pregrasp` 停住核对净空，再从验收过的观察位重新启动完整流程。每段有独立确认，确认后以实时关节起点复查本段路径。接近期间还会重新取图，若方块移动、遮挡、工具姿态变化或 Z 超限，立即中止后续动作。

   ```bash
   python -I scripts/demos/grasp_real_cube.py grasp --ip "$ARM_IP" --site "$SITE" --calibration "$CALIB" \
     --execute --until pregrasp
   python -I scripts/demos/grasp_real_cube.py grasp --ip "$ARM_IP" --site "$SITE" --calibration "$CALIB" \
     --execute --until lift
   ```

机械臂接受的工作系为 `Base`，工具系为核对过的 `Arm_Tip` 或 `Grip_plus`。实际动作前后对照实时 FK。运行中监测 TCP 工作空间、路径偏离、相邻回包速度、关节限位、姿态、错误码和碰撞防护。抬升中持续核对夹爪夹持状态。SDK 回执异常也视为命令可能已发出，机械臂会尝试发送轨迹停止并锁定本次会话。物理停止和夹爪异常由现场人员按设备流程处置，单独核查夹爪停稳和工件状态。[轨迹停止接口](https://develop.realman-robotics.com/robot/apipython/classes/motionControl/)

`lift_verified` 表示机械臂、夹爪和视觉抬升位移均符合验收条件。`lift_needs_visual_confirmation` 对应遮挡或视觉位移不符，进程返回 2，由操作员现场复核。流程结束时夹爪保持当前状态，收工和释放由操作员单独确认。出错、取消或分段停止后，使用示教器检查机械臂和工件并按现场流程回到已验收的起始位置。

## 离线回归

```bash
MUJOCO_GL=egl python -I -m unittest discover -s tests -p 'test_real*.py' -v
MUJOCO_GL=egl python -I -m unittest discover -s tests
```

测试覆盖缺少确认、碰撞保护关闭、确认后位置变化、法兰/TCP 指令转换、越界、路径偏离、异常停止、夹爪空夹、SDK 超时、只读入口、视觉 Z 拒绝、目标移动、逐段执行以及抬升后视觉丢失。

本轮全量回归通过 549 项。增加确认后的实时关节起点路径复查后，当前版本通过 43 项真机接口、安装与视觉抓取专项测试。EGL 下另核对了 17 项 RGB-D 测试。MuJoCo 渲染的 RGB-D 通过抓取规划和各段 IK 预检，A/B/C 真实回放均在 Z 校核处中止。真机测试为只读状态采集。[全量回归日志](../outputs/validation/realman_installation/full_regression_visual_grasp.log)
