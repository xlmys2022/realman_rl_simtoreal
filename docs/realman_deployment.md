# RML63II-6FB 真机只读部署

本次设备已确认三代控制器、RML63II 本体以及 `Grip_plus` 工具。现场手眼解释和仿真实机安装适配的最新记录见 [实机安装适配](realman_installation.md)。

三个入口分别检查机械臂、检查 RealSense 和显示方块基座坐标。机械臂状态包含关节角、控制器当前位姿、错误码、查询的主机单调时间，以及由本次控制器 DH 和关节角计算的 `T_base_flange`。夹爪可按需通过控制器的 EG2-4C2 协议读回，`opening_raw` 为控制器原始开口量。[RealMan 连接和超时接口](https://develop.realman-robotics.com/robot/apipython/classes/roboticArm/)

## 终端命令

从仓库根目录运行。预览依赖桌面会话和 Tkinter，已有 `realman_rl` 环境且依赖齐全时可直接运行检查命令。

```bash
conda activate realman_rl
python -m pip install -e '.[hardware]'

read -rp '输入示教器上显示的控制器 IPv4 地址 ' ARM_IP
python -I scripts/diagnostics/check_realman_arm.py --ip "$ARM_IP"
# 实机确认装有 EG2-4C2 且使用控制器内置夹爪协议时
python -I scripts/diagnostics/check_realman_arm.py --ip "$ARM_IP" --gripper

python -I scripts/diagnostics/check_realsense.py
python -I scripts/diagnostics/check_realsense.py --no-preview --frames 30
```

相机检查独立于机械臂。首次输出设备序列号、彩色内参 `K`、原始畸变模型和系数、流规格与深度比例。`--no-preview` 输出逐帧 JSONL，包含两路帧号、设备时间戳（ms）、时钟域、主机收帧时间（s）、深度有效率和中值。彩色和深度使用同一针孔像素网格，深度值沿彩色光轴、单位为米，黑色表示无效值。预览可以用 `Q`、`Esc` 或关闭窗口退出。[RealSense 投影与对齐](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0) · [RealSense SDK 对齐实现](https://github.com/realsenseai/librealsense/blob/master/src/proc/align.cpp)

## 实测标定文件

2026-11-26 实机手眼矩阵的父系是 `Grip_plus` TCP。只读验收使用 `--calibration assets/calibration/realman/calibration.20261126.json` 显式加载。静止方块跨视角一致性已复核，抓取高度仍需用实测支撑面完成验收。[新批次证据和回放命令](realman_installation.md#2026-11-26-实机手眼标定)

模板位于 [assets/calibration/realman](../assets/calibration/realman/)。其中的 `cube_hsv.json` 是 `cube_hsv_20261124_210413_368026/hsv.json` 的副本，绿色阈值为 `[49, 80, 66]` 到 `[84, 255, 255]`。复制模板后填写现场值，`hsv_file` 支持相对配置文件的路径。

```bash
cp -i assets/calibration/realman/calibration.example.json assets/calibration/realman/calibration.local.json
python -I scripts/demos/locate_real_cube.py --validate-calibration
```

压缩包包含标定程序、棋盘格规格、真机采集照片和求解截图。截图中可读出一组外参，使用现场样本的离线复核结果见实机安装文档。追溯该批次需要与原图逐张对应的 `poses.txt`、当时的工具系记录及相机标识。可用 `unzip -p hand_eye_calibration-main.zip hand_eye_calibration-main/README.md` 查看采样步骤。[手眼标定压缩包](../hand_eye_calibration-main.zip) · [截图外参与 A/B/C 回放](realman_installation.md#手眼矩阵与视觉诊断)

- `hand_eye.parent_frame` 填 `flange`、`tcp` 或 `base`。腕部相机用手眼求解时实际采样的末端坐标系，固定相机用 `base`。采集时记录活动工具系、工作系和相机序列号，一一对应保留角点识别成功的图像与位姿。
- `hand_eye.T_parent_color` 填实际彩色光学系到指定父坐标系的齐次矩阵。压缩包的 `compute_in_hand.py` 输出相机到采样末端的旋转和平移，`compute_to_hand.py` 输出相机到基座的旋转和平移。把实测 `R` 和以米为单位的 `t` 放入前三行，第四行为 `[0, 0, 0, 1]`。
- `T_flange_tcp` 填从实际 TCP 到法兰的齐次矩阵。用固定参考点和多组法兰姿态标定 TCP 偏移，并明确工具轴向和对应夹爪开口。矩阵必须包含测得的平移和确认的轴方向。
- `cube_edge_m` 填用卡尺测得的立方体边长，单位为米。`camera_serial` 可填写同一台 RealSense 的序列号，以启用设备校验。

`T_A_B` 把 B 系坐标变换到 A 系。腕部相机的外参拼接为 `T_base_flange @ T_flange_color` 或 `T_base_flange @ T_flange_tcp @ T_tcp_color`。固定相机直接使用 `T_base_color`。缺少 HSV、手眼、TCP 或方块实测尺寸时，定位入口在打开设备前退出并列出缺项。

```bash
CALIB=assets/calibration/realman/calibration.20261126.json
# 腕部相机
python -I scripts/demos/locate_real_cube.py --calibration "$CALIB" --ip "$ARM_IP"

# 固定相机可另行运行下列命令，FIXED_CAMERA_CALIB 为该相机的实测 parent_frame=base 配置
# python -I scripts/demos/locate_real_cube.py --calibration "$FIXED_CAMERA_CALIB"

# 腕部相机的无界面采样
mkdir -p outputs/calibration
python -I scripts/demos/locate_real_cube.py --calibration "$CALIB" --ip "$ARM_IP" \
  --no-preview --frames 60 > outputs/calibration/real_cube_samples.jsonl
```

预览显示 RGB、深度、HSV 最大连通域和基座系方块中心 XYZ（m）。腕部模式先等 RGB-D 启动配对稳定，再用前后关节查询夹住相机采样，默认窗口 250 ms、关节最大变化 0.05°。当 RGB 和深度时间戳与主机墙钟同域时，窗口按曝光区间校验。设备硬件时钟模式使用新帧的主机收包时刻做静止采样配对，报告中保留设备原始时间戳。

## 现场验收

1. 在示教器确认当前为真实模式，保持机械臂静止。运行状态检查，核对 `RML63II-6FB`、6 轴及示教器关节角，错误码应全为 0。若启用夹爪查询，核查在线状态、原始开口和错误码。
2. 打开相机预览，核对 RGB 和深度边缘，在已知距离的平面上检查米制深度。确认两路帧号递增。若要切换流规格，用 `--width`、`--height`、`--fps` 指定相机支持的组合。
3. 放置实测方块，保持正立，方块边与基座 X/Y 轴对齐。用真实工作距离和灯光检查 HSV Mask。分别在数个基座系已知参考位置采样，记录 XYZ 偏差和静态抖动，按现场任务容差验收。腕部相机完成一次采样后，可由操作员在示教器上更换观测姿态、保持静止并再次采样，核对同一方块的基座坐标是否一致。
4. 移开方块或遮住视线，无可靠深度检测的帧应显示未检出，JSONL 的 `cube_center_base_m` 应为 `null`。在只读模式检查断开相机时的退出行为。机械臂回包超时由 `--arm-timeout-ms` 配置，相机等待配对帧超时由 `--camera-timeout-ms` 配置。

## 运动保护与离线回归

三个终端入口使用只读会话。硬件 API 的 `move_linear_tcp` 仅在调用者同时显式传入 `enable_motion=True` 和现场 `MotionLimits` 后可用。限值包括基座系 TCP 工作空间、单次位移、速度百分比、线速度（m/s）、运动时限以及实测 `T_flange_tcp`。接口仅做保持姿态的单次直线平移，要求控制器活动工作系和工具系分别对应真实基座与法兰。运动过程中检查实测 TCP、相邻回包的平均线速度、关节限位、姿态、轨迹类型及错误码。工作空间边界针对 TCP，整机避障由现场设施与控制器负责。[RealMan 直线规划接口](https://develop.realman-robotics.com/en/robot/apipython/classes/movePlan/) · [控制器末端速度读回](https://develop.realman-robotics.com/en/robot/apipython/classes/tipVelocityParameters/)

下发后若通信、状态或时限异常，接口尝试 `rm_set_arm_stop`，并锁定该对象的后续运动。停止未确认时应立即使用现场硬件急停，再复核状态。停止回执表示控制器接受轨迹停止指令。[RealMan 轨迹停止](https://develop.realman-robotics.com/en/robot/apipython/classes/motionControl/)

```bash
python -m unittest discover -s tests -p 'test_real*.py' -v
MUJOCO_GL=egl PYTHONPATH=tests python -m unittest test_rgbd_vision test_fk_comparison test_ik_comparison test_package
MUJOCO_GL=egl python -I -m unittest discover -s tests
```

本轮全量离线回归 529 项通过，其中新增真机部署测试 22 项，覆盖从控制器 DH 推算法兰、只读调用边界、限值拦截与异常停止、实机标定加载、RGB-D 对齐与时间窗、绿色方块基座坐标及丢失处理。

2026-11-25 使用本机 D435 进行了只读复测。640×480、30 fps 配置连续收到 5 组有效配对帧，两路帧号均递增，有效深度像素占比约 94%–95%。机械臂与夹爪接口使用离线 SDK 替身验证，基座定位精度待补齐实测配置后现场验收。
