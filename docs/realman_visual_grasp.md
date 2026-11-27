# 真机传统视觉抓取验收

固定俯视流程沿同一末端姿态完成张爪、水平对准、预抓取、直线下降、力控闭合和垂直抬升。视觉给出 XY、方块边方向及绝对 Z 校核值。下降终点取经过现场验收的 `Grip_plus` 夹持 TCP 高度。现有仿真场景和 PPO 仍使用各自的配置。

## 2026-11-27 视觉基座水平校正

未校正的 RGB 棋盘中心偏高 14.5–16.9 mm，固定棋盘的朝上法向在基座系呈现约 1.3–2.3° 的倾斜。RGB-D 与纯 RGB 的同像素光轴深度中位差为 −1.2～+1.8 mm。使用实测平行桌面拟合得到的水平校正为 1.84°，独立 Z 平移仅 +0.21 mm。沿 Base +X 的倾斜项能解释抓取区域的主要高度偏差。原始光学外参与机械臂实际位姿各自贡献的误差仍需外部位姿基准分离。[棋盘拟合与留出报告](../outputs/validation/realman_installation/base_leveling_fit_20261127.json)

现场默认配置 `calibration.20261127.json` 在完成 `T_base_flange @ T_flange_tcp @ T_tcp_color` 之后左乘 `base_leveling`，把视觉重建对齐现场实测的 Base 水平基准。逐次留出一个棋盘采样位时，RGB 中心 Z 误差最大 1.8 mm。独立的 A/B/C 方块回放中，中心 Z 偏差从 +12.44、+12.92、+12.53 mm 降至 −1.45、−1.00、−1.49 mm。三个位置的 XY 校正量均小于 0.8 mm。固定俯视抓取用校正后的视觉坐标对准 XY，并从现场配置读取实测的 `Grip_plus` 夹持 Z。更换站点或拆装相机后应重新测量参考面和采样。[现场标定文件](../assets/calibration/realman/calibration.20261127.json) · [独立方块回放](../outputs/validation/realman_installation/base_leveling_cube_replay_20261127.json)

真机和 D435 重新只读采集的棋盘 8/8 帧通过校核。使用同一组帧对比，RGB-D 棋盘中心 Z 误差从 +15.87 mm 降至 +0.74 mm，纯 RGB 从 +15.03 mm 降至 −0.13 mm。该批采样未参与拟合。夹爪读回为离线，主动抓取仍待现场验收。[真机只读复测](../outputs/validation/realman_installation/onsite_base_leveling_report_20261127.json)

在 `realman_rl` 环境离线复现。至少采集两组不同位置或腕姿的静止棋盘录像，填写现场实测跨度和图案面高度。命令输出新标定与逐组误差报告。

```bash
SAMPLES=outputs/validation/realman_installation
RAW_CALIB=assets/calibration/realman/calibration.20261126.json
python -I scripts/diagnostics/audit_realman_vision.py reference \
  "$SAMPLES/onsite_reference_A.npz" "$SAMPLES/onsite_reference_B.npz" "$SAMPLES/onsite_reference_C.npz" \
  --calibration "$RAW_CALIB" --board-corners 11 8 --board-spans-mm 300 210 \
  --board-top-mm 3.5 --table-z-base-m -0.040 \
  --write-leveling-calibration "$SAMPLES/calibration.leveling_replay.json" \
  --output "$SAMPLES/base_leveling_rerun.json"
```

## 现场读回与绝对 Z

- 用户确认桌面水平，基座安装面到方块桌面的实测垂距为 40 mm。桌面到第二关节中心约 214 mm，和安装高度加实机 DH 的 `d1=172 mm` 所得约 212 mm 接近。按现有基座轴定义，执行平面采用水平桌面 `Z=-0.040 m`，20 mm 方块在桌面时的参考中心是 `Z=-0.030 m`。
- `calibration.20261126.json` 使用 `Grip_plus` 为父系。A/B/C 回放中方块中心的 Z 仍偏高 12.44、12.92、12.53 mm，周边支撑点分别偏高 12.66、13.53、13.01 mm。方块中心相对局部深度支撑面的距离为 9.42–9.79 mm。高度差出现在方块和周围点云，中心减半边长的补偿只能解释约 0.2–0.6 mm。[三组 Z 诊断](../outputs/validation/realman_installation/hand_eye_20261126_height.json)
- 以 A/B 首帧中靠近桌面的两个固定黑点做纯 RGB 双视图三角化，得到 Z 约 −34.0、−33.8 mm，重投影误差小于 0.35 px。对应像素周围的 RGB-D 读数约 −30.4 至 −28.5 mm。三角化共用当前 FK 和手眼外参，独立于深度读数。结果提示深度链和两种方法共用的坐标链均需核查，样本不足以分解误差或把偏差应用到其他位置。局部点云斜率作为诊断量记录。部署保留水平平面和原始手眼矩阵。
- 首次真机只读返回 `Base / Grip_plus`、末端最大线速度 `0.25 m/s` 和动力学碰撞等级 `0`。动作验收前由操作员在示教器按现场要求启用碰撞防护并复查。历史回包记录夹住方块时 `opening_raw=241`、`force_g=787`、`mode=6`，夹爪在线、使能且无报错。力控指令值按厂家定义单独设定。[现场状态快照](../outputs/validation/realman_installation/control_readonly_20261126.json) · [睿尔曼夹爪接口](https://develop.realman-robotics.com/robot/apipython/classes/gripperControl/) · [碰撞等级与线速度接口](https://develop.realman-robotics.com/robot/apipython/classes/tipVelocityParameters/)

绝对 Z 验收需要在抓取区域放置可独立测到基座 XYZ 的参考物。在至少两个 XY 位置比对其相机返回高度，同时使用已知高度差的量块复核 D435 的相对深度。记录基座安装面水平度并保留同帧 RGB-D、机械臂位姿和参考物实测值。未校正配置的 12 mm 残差会在规划阶段中止下降。

2026-11-26 现场复查时，操作员已在示教器设置碰撞防护等级 **4**，只读回包同步确认了该等级。这一姿态的十次静止读回为开口度 237、模式 6、力反馈 0 g。开口或模式的单项读回不足以验收夹持。[复查状态与十次读回](../outputs/validation/realman_installation/onsite_collision_4_20261126.json)

### 用实测棋盘区分两类 Z 误差

把原标定棋盘格平放在方块区域，避免弯曲和遮挡。`11×8` 指内角点数量。横向量第 1 到第 11 个内角点，竖向量第 1 到第 8 个内角点，再量图案平面高出桌面的毫米数。软件配置的 30 mm 格长只给出名义跨度 300×210 mm，现场回放填实测值。

由操作员让整块棋盘进入相机视野并保持静止。我方通过现有 `capture` 入口记录只读帧。第一组使用已验收的观察位。第二组保持棋盘不动，由操作员改变相机距棋盘的距离后重新静止。单批至少要有三帧同时通过棋盘角点、重投影质量和深度覆盖率检查。

```bash
SAMPLES=outputs/validation/realman_installation
CALIB=assets/calibration/realman/calibration.20261126.json
ARM_IP=192.168.1.18
python -I scripts/diagnostics/audit_realman_vision.py capture --ip "$ARM_IP" \
  --calibration "$CALIB" --frames 8 --output "$SAMPLES/onsite_reference_A.npz"
# 第二次在另外一个静止采样位执行 capture，输出 onsite_reference_B.npz
read -rp '输入横向跨度、竖向跨度、图案面高度（mm） ' WIDTH_MM HEIGHT_MM BOARD_TOP_MM
python -I scripts/diagnostics/audit_realman_vision.py reference \
  "$SAMPLES/onsite_reference_A.npz" "$SAMPLES/onsite_reference_B.npz" --calibration "$CALIB" \
  --board-corners 11 8 --board-spans-mm "$WIDTH_MM" "$HEIGHT_MM" --board-top-mm "$BOARD_TOP_MM" \
  --table-z-base-m -0.040 --output "$SAMPLES/onsite_reference_report.json"
```

`depth_minus_rgb_axial_mm` 比较同一像素的对齐深度和 RGB PnP 平面轴深度，供判断深度链是否随距离偏移。`rgb_paired_base_z_error_mm` 由 RGB、FK 和手眼外参得到，供核对共同坐标链和水平基准。`rgbd_paired_base_z_error_mm` 是两者叠加到基座系的观测结果。报告保留每帧结果和剔除原因，现场安全阈值仍由验收配置管理。[OpenCV 相机位姿接口](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) · [RealSense 光轴与反投影](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

实机 A/B 各 8 帧均有效。使用现场确认的 30 mm 格长和约 3.5 mm 图案高度，RGB 估计距离分别为 0.732 和 0.637 m。B 的 TCP 下降了 96.31 mm，腕姿与 A 相差 0.005°。在同像素比较中，对齐深度分别比 RGB 平面多 1.79 和 1.24 mm。纯 RGB 得到的基座 Z 偏高 15.36 和 16.68 mm，RGB-D 偏高 13.95 和 15.41 mm。A/B 的纯 RGB 棋盘面相对基座 XY 面表观倾角为 1.97° 和 2.33°，沿 X 每米升高约 34–41 mm。该斜率用于诊断坐标链。抓取下降继续受实测高度保护。[现场 A/B 棋盘报告](../outputs/validation/realman_installation/onsite_reference_AB_report.json)

C 的腕姿相对 A 改变 29.04°，8 帧同样全部有效。RGB 估计距离 0.581 m，基座 Z 偏高 14.18 mm，RGB-D 偏高 14.98 mm。同像素对齐深度比 RGB 平面少 1.16 mm。棋盘固定时，RGB 重建中心的 A→C 横向位置差约 11.3 mm，平面法向差约 0.86°。[现场 A/B/C 棋盘报告](../outputs/validation/realman_installation/onsite_reference_ABC_report.json)

### 2026-11-27 原始手眼重放与在线深度复核

现场已确认安装面与木板平行、高差 40 mm，板面高出桌面约 3.5 mm。预期桌面、棋盘图案面与 20 mm 方块中心分别为基座 Z −40、−36.5 和 −30 mm。整个平台相对地面的朝向会同时旋转基座和桌面。离线候选仅进入诊断报告，现场 40 mm 高差及 `calibration.20261126.json` 作为重放基准。

- **方块和实际支撑面。** A/B/C 的方块中心相对支撑面高 9.79、9.42、9.48 mm。局部支撑面的朝上法向分别为 `[-0.0271, 0.0016, 0.9996]`、`[-0.0252, 0.0026, 0.9997]`、`[-0.0296, 0.0021, 0.9996]`。表观倾角 1.45–1.70° 在方块所在的 Base X 约 0.44 m 处对应约 11–13 mm 的斜率项，接近绝对高度偏差。拟合来自局部区域，外推到基座原点只作量级诊断。方块边长复测无法解释支撑面与棋盘的同向偏差。
- **独立棋盘的 RGB 与深度。** 24/24 帧通过。RGB PnP 的基座朝上法向分别为 `[-0.0343, -0.0003, 0.9994]`、`[-0.0408, -0.0005, 0.9992]`、`[-0.0212, 0.0071, 0.9998]`。同像素中位偏差在 RGB 单独计算时为 +15.38、+16.67、+14.19 mm，加入对齐深度后为 +13.95、+15.41、+14.98 mm。深度轴中位差为 +1.80、+1.23、−1.15 mm，A 位同像素 P10–P90 约 −8.7 至 +12.4 mm，因此单点深度可显著波动。A→C 同一实体中心仍漂移约 11.3 mm，法向差 0.86°。同帧法兰 FK 加 TCP 与 SDK `Grip_plus` 的最大差约 0.010 mm、0.079°，反映两套软件位姿表达的一致性。实物位姿误差仍需独立基准分离。
- **原始手眼残差。** 20 张 1280×960 JPG 和 `poses.txt` 逐编号配对，复现原脚本的自由 K/D 和 Tsai 求解，平移与存档相差小于 0.001 mm。重投影 RMS 按原生 640×480 换算为 0.174 px。固定棋盘中心在 Base 中的离散 RMS 为 8.47 mm，其中 Z 的 RMS 为 2.86 mm。相对运动 `AX=XB` 的平移与转角 RMS 约 7.00 mm、1.00°。Tsai 逐张留出的 3D/Z RMS 为 9.41/3.09 mm。留出时以其余 19 张重求 K/D 与手眼矩阵，棋盘在历史数据中的绝对 Base 高度未独立记录。
- **算法与内参敏感性。** 自由 K/D 下，Park、Horaud、Andreff、Daniilidis 的三维留出 RMS 分别为 9.60、9.53、11.82、9.04 mm。Andreff 让独立 A/B/C 的 Z 误差降至 +6.61～+8.53 mm，同时恶化历史留出残差。Daniilidis 的留出指标略好，但 A→C 独立漂移增至 16.89 mm。自由拟合焦距约 601.06/600.76 px，D435 读回为 605.11/605.16 px。固定存档外参、仅切换 K/D 时，独立三组棋盘中心的基座 Z 只改变约 1.32–2.52 mm。五种求解的改善取决于内参和验证数据，尚无通过绝对高度和跨姿态一致性双重验收的候选。
- **在线同帧光学核对。** 两批各 8 帧。原生点云经设备流间外参投到 RGB 光学系后，相对 RGB PnP 平面的组中位轴差分别为 −0.51、+0.33 mm。SDK 对齐后按部署代码换算光轴的结果分别为 −1.14、−0.29 mm，两次光轴换算本身的中位影响均约 +0.28 mm。实测只连接相机，比较的是棋盘距离约 0.597 m 时的相对光学误差。[首批在线同帧记录](../outputs/validation/realman_installation/native_depth_online_20261127.json) · [复跑同帧记录](../outputs/validation/realman_installation/native_depth_online_repeat_20261127.json) · [RealSense 投影与流间外参](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

综合证据将首要排查点定位到纯 RGB 与 RGB-D 共享的 Base 重建链。近距离原生深度与对齐后的差值，以及已试 K/D 的敏感性，均未达到 12–13 mm。手眼验收还需要在已知 Base 高度的物理靶上，分别用两处相距足够的 Base X 位置及两种腕姿复核。空间移动和腕姿变化分开做，可比较空间斜率和腕姿相关项，靠近桌面的主动抓取沿用现有 Z 联锁。[离线原始重放报告](../outputs/validation/realman_installation/hand_eye_z_offline_20261127.json) · [OpenCV 眼在手上变换定义及求解器](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

从项目根目录重跑。`hand-eye` 依赖原 `handeye` 环境的 OpenCV 4.x，项目 `realman_rl` 环境的 OpenCV 5 构建未导出 `calibrateHandEye`。`--legacy-dir` 指向实机采集照片与 `poses.txt`，根目录的程序压缩包未包含这批实机数据。

```bash
conda activate handeye
SAMPLES=outputs/validation/realman_installation
CALIB=assets/calibration/realman/calibration.20261126.json
LEGACY_DIR=/home/mr-beaver/Documents/robot/hand_eye_calibration-main/eye_hand_data/data20261126
PYTHONPATH="$PWD/src" python -B scripts/diagnostics/audit_realman_vision.py hand-eye \
  "$SAMPLES/onsite_reference_A.npz" "$SAMPLES/onsite_reference_B.npz" "$SAMPLES/onsite_reference_C.npz" \
  --legacy-dir "$LEGACY_DIR" --calibration "$CALIB" \
  --board-corners 11 8 --board-spans-mm 300 210 --board-top-mm 3.5 --table-z-base-m -0.040 \
  --cube-samples "$SAMPLES/overhead_A.npz" "$SAMPLES/overhead_B.npz" "$SAMPLES/oblique_C.npz" \
  --cube-capture-calibration assets/calibration/realman/calibration.local.json \
  --output "$SAMPLES/hand_eye_z_rerun.json"

conda activate realman_rl
python -I scripts/diagnostics/audit_realsense_board_depth.py --serial 152122072347 \
  --board-corners 11 8 --board-spans-mm 300 210 --frames 8 \
  --output "$SAMPLES/native_depth_rerun.json"
```

### 基础动作的现场结果

- 操作员在 C 位通过交互终端确认一次向 Base +Z 的空爪直线。设定速度 1%，现场线速度上限 5 mm/s，控制器碰撞防护为 4。停稳回读 Z 上移 4.945 mm，X/Y 分别变化 −0.029 和 −0.002 mm，TCP 距目标 0.021 mm，腕姿变化 0.004°，控制器无报错。该配置仅覆盖 C 位附近的试验空间。[验收记录](../outputs/validation/realman_installation/onsite_C_lift_verified.json) · [试验配置](../assets/calibration/realman/grasp_site.onsite_C_lift.json)
- 操作员实测张满夹指内侧净距 71 mm，示教器开口度 1001；SDK 在 C 位连续十次返回 1001、模式 1、力反馈 0 g。[张满采样](../outputs/validation/realman_installation/onsite_open_C_20261126.json)
- 在同一高位用示教器空夹闭合，示教器开口度为 −2；SDK 十次均返回 −2、模式 6、力反馈 702 g。空夹到端也会出现力控停止和显著力反馈，软件必须同时核对实物开口范围与力反馈。现场配置按原始有符号值记录空夹读数。[空夹采样](../outputs/validation/realman_installation/onsite_empty_closed_20261126.json)

## 现场配置

从项目根目录在 `realman_rl` 环境运行。现场文件已填入确认的水平桌面基准，剩余参数按验收进度填写。未填写的 `null` 会显示字段名并中止对应命令。新站可从模板创建自己的文件。[本机现场配置](../assets/calibration/realman/grasp_site.local.json) · [配置模板](../assets/calibration/realman/grasp_site.example.json)

```bash
conda activate realman_rl
ARM_IP=192.168.1.18
CALIB=assets/calibration/realman/calibration.20261127.json
SITE=assets/calibration/realman/grasp_site.local.json
```

- `motion` 填写现场可用的基座系 TCP 工作空间、单段位移、速度百分比和线速度上限、反馈路径偏离上限及运动时限。Z 下界应覆盖现场核验的最小工具间隙，包含张开和闭合的夹指与附件。`T_flange_tcp` 从手眼配置的实测 TCP 字段读取。先用小段直线独立验收，控制器的最大线速度是换算上限。
- `gripper` 需要多次稳定读回的张满原始范围、空夹闭合值、实物力控停止时的原始开口和力读回范围。现场确认低速开合、允许的闭合力控指令及动作时限。`open_speed_raw`、`close_speed_raw` 为 SDK 速度量，`close_force_raw` 为力控阈值指令，`held_force_g` 为设备压力反馈范围。张满使用可显式限速的 SDK 松开接口，闭合使用单次限力夹取。读取到 `mode=6` 且开口和力反馈均落在验收范围才放行抬升。[EG2-4C2 控制及读回字段](https://develop.realman-robotics.com/robot/json/endTool/)
- `grasp` 需要实测张满净间距、能包住方块且两指避开支撑面的 `Grip_plus` 夹持 Z、实测夹紧轴 `x` 或 `y`。另外填入已验收的预抓取净高、试抬高度、速度和视觉容差。固定俯视下工具倾角容差至多 5°，夹指对边容差至多 15°。采样需有至少 5/8 帧提供位置和边方向，方块边按 90° 等价方向核对。`max_visual_z_error_m` 和 `max_target_shift_m` 上限均为半个方块边长。

张满实测以夹指内侧同一基准为准。在静止状态下读回夹爪并用量具测量物理间距。确认 `241` 对应稳态有物夹持后，通过多次读回确定上下限。现场首次下降前，需独立验收夹持高度、最低工具点净空与工件 XY。控制器的碰撞检测、限位和现场防护装置应保持可用。

基础小直线先验收一段远离桌面的上移。操作员提供 `Base / Grip_plus` 中起点与终点的 XYZ 和 RX/RY/RZ，注明示教器单位，并给出包住该段的安全 TCP 范围、批准速度以及允许的位移、路径偏差和超时。只有这段通过后再扩大工作范围。夹爪现场依次记录张满、空夹闭合、夹住方块三种静止状态。每种状态可用 `--gripper --gripper-samples 10 --collision` 采集原始开口、力反馈和模式范围，同时记录示教器上已验收的速度与力控指令量。

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

2. 独立验收小段直线及夹爪张开。`move` 使用 `motion` 节，`open` 另需实测的 `gripper.open_feedback_raw`、现场确认的 `gripper.open_speed_raw` 和 `gripper.completion_timeout_s`。先运行只读命令，核对目标与预检。确认现场条件后，在交互终端追加 `--execute`，按提示输入 `MOVE` 或 `OPEN`。张开验收应在夹爪下方净空且工件有承托时进行。

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

现场阶段全量回归通过 556 项。加入 Z 偏差分析后，51 项真机与视觉专项测试通过，新增用例覆盖深度飞点剔除、法向方向、手眼闭环和缩放像素中心约定。MuJoCo 渲染的 RGB-D 通过抓取规划和各段 IK 预检，A/B/C 真实回放均在 Z 校核处中止。真实机械臂的基础动作记录在本文件的现场结果小节。[上一轮软件交付的回归日志](../outputs/validation/realman_installation/full_regression_visual_grasp.log)
