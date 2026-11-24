# 腕部 RGB-D 视觉抓取

这是 `RML63II-6FB` 场景中 `tabletop_cartesian` 的 `rgbd_hsv` 观测模式，使用 `realman_rl` 环境。机械臂从六轴零位经传统控制到固定观察位，再由 PPO 输出末端 XYZ 和夹爪增量，独立 FK/IK 执行。

## 交互演示

在图形桌面终端中，从仓库根目录启动：

```bash
conda activate realman_rl
python -I scripts/demos/demo_vision_grasp.py
```

窗口从六轴零位开始运行 MuJoCo 仿真。到达固定观测姿态后，右侧显示 PPO 使用的腕部 RGB 和 HSV Mask，左上角显示视觉估计的基座坐标与稳定保持进度。用鼠标左键拖动旋转、右键拖动平移、滚轮缩放。按空格暂停或继续，按 `R` 用同一 seed 从零位重跑。任务结束后画面保留，可继续调整视角。

默认请求 2 倍仿真速度，`--speed 1` 可按实时速度观看。用 `--seed 997100 --object-radius .02` 可复现已验证的随机位置，用 `--model` 可指定其他视觉 PPO 检查点。

## 图像与三维定位

读取 `camera_link` 原点处的 `wrist_rgb` 相机。6FB 视觉场景将 `camera_link` 作为法兰的刚性安装坐标系，位置沿用仿真中的法兰到光心变换。RGB 与深度由同一相机在同一个物理时刻采集，相同分辨率。深度是光轴 Z 方向的米制测量，不使用物体 ID、MuJoCo 分割标签或物体位姿。[法兰、相机及 TCP 坐标系](../assets/robots/rml63/SOURCE.md)

OpenCV 按 RGB → HSV 转换，默认阈值为 H `[3,25]`、S `[175,255]`、V `[50,255]`。输出单通道 `uint8` Mask：目标白色 255，背景黑色 0。选择橙色连通区域后，对相应像素的深度反投影：

`Xc = (u-cx)*D/fx, Yc = (v-cy)*D/fy, Zc = D`

再以 `Pbase = Rbase_camera * Pc + tbase_camera` 得到机器人基座坐标。OpenCV 光学坐标为右、下、前；适配器显式转换 MuJoCo 的右、上、后坐标，不能直接混用。

夹爪接近时会遮挡方块，Mask 质心可能偏离方块中心。当前定位器使用深度点云中的可见近侧表面，以及已知 20 mm 正立方块尺寸，补偿半边长后恢复中心；高度由实时深度测量，不会在抬升阶段继续假定物体停在桌面。该几何先验适用于当前正立方块，未声称支持任意物体或大角度旋转。

默认图像 640×480、测量周期 0.1 秒，控制周期仍为 0.05 秒。物体速度由连续相机测量在基座系中差分、滤波获得。短暂遮挡时保留带年龄标记的最后一次视觉测量，不把目标附着到末端，不回退到物体真值；超过 0.25 秒没有有效检测就终止为 `vision_lost`。

独立检查 RGB、Mask、深度和基座坐标：

```bash
conda activate realman_rl
pip install -r requirements-vision.txt
python -I scripts/demos/detect_tabletop_cube.py \
  --object-radius .02 --seed 990010 --output-dir /tmp/realman_rgbd_detection
```

输出 `rgb.png`、`mask.png`、`depth_m.npy`、`depth_mm.png` 和 `detection.json`。JSON 包含相机内参、相机到基座变换和视觉估计坐标，不包含用于控制的物体真值。

## 策略观测与信息边界

视觉模式有独立的 29 维观测契约，不能因为维数相同而直接混用旧真值模型：

- 机器人自身状态：TCP 基座位置/速度、夹爪宽度/速度/参考、末端跟踪误差和 IK 接受比例。
- 相机物体状态：物体相对 TCP 位置、相对视觉目标的误差、视觉速度、可见表面法向、检测置信度和测量年龄。
- 视觉任务状态：相机目标与夹爪几何包围、编码器开口跟踪误差共同生成的夹持置信度，以及视觉稳定保持进度。

旧观测中的物体真值位置、线/角速度、接触力、桌面支撑力和真值保持进度全部退出策略输入。目标位置由固定观测姿态下第一帧有效的相机估计加上 30 mm 抬升量生成，不读取 reset 中采样的目标坐标或 MuJoCo 物体位置。

视觉结束判断也来自相机：目标 XY 误差小于 3 mm、Z 误差小于 1 mm、视觉线速度小于 1.5 mm/s、视觉夹持置信度合格，并连续保持 3 秒（原 2 秒保持加 1 秒视觉稳定余量）。策略不会因为 MuJoCo 的成功标记提前结束。相机判定完成后，独立物理验收仍要求原完整接触、离台、速度、滑移及连续保持标准；不满足则立即记为 `visual_false_success`，不会等待真值满足后再宣布成功。

MuJoCo 物体真值继续用于仿真动力学、碰撞安全、训练奖励和独立物理验收。评估记录中的 `object_*`、真实受力等字段是审计数据，网络只接收单独构造的视觉观测数组。视觉回放记录每次实际送入网络的 `policy_observation`，以便核对。测试会改写调试物体位置、速度、目标、接触力和保持时间，确认策略观测不变。

## PPO 与模型迁移

`--observation-source rgbd_hsv` 选择视觉模式。视觉 PPO 保留已训练的神经末端网络，使用独立夹爪分支及可学习的视觉夹持门控；原力输入对应的夹爪权重在跨契约迁移时清除，价值分支重新初始化。训练仍由 PPO 更新，没有示范动作、关节角教师或交接后的抓取脚本。

夹持后，末端网络的目标特征使用相机测得的物体目标误差，补偿不同夹持深度和物体相对夹爪位置变化，不再只把 TCP 移到固定高度。所有末端和夹爪动作均值由神经网络输出，IK/FK 仍是独立执行模块。

当前 `wrist_rgbd_hsv_cube_v4` 使用同一个末端网络分别计算当前请求及保持参考时的响应，两者之差作为末端动作均值，使零参考误差对应零增量，消除继承网络的小幅静态漂移。相机目标误差进入 XY 1 mm、Z 0.5 mm 精度带后保持已发出的参考；夹爪动作仍由独立网络产生。这个结构参与 PPO 采样、似然计算和梯度更新，没有在执行器中替换网络动作。

加载器校验视觉观测字段、策略类型、版本和 6FB 观察位契约。用原场景的权重初始化训练时，可显式指定 `--init-model`。

## 已保存模型与复现

当前选用 `rml63ii_6fb/vision_transfer_step768/model.zip`。沿用旧视觉 PPO 的冻结权重，在新机型上重做完整回合验收。本次无新增 PPO 优化器更新。固定位置和半径 2 cm 内随机位置各取一个样本，均通过原有物理标准，结束时分别离台 29.671 mm 和 29.690 mm，物理保持 3.570 s 和 3.593 s，视觉保持均为 3 s。[6FB 复测结果](../outputs/grasp/ppo/tabletop_cartesian/rml63ii_6fb/vision_transfer_step768/validation.json)

旧 6F-V 场景的 768 步检查点曾通过固定 3/3 和随机 3/3。原配置、模型、源码归档和评估保留在 `vision_rgbd_v1/`。[历史验证报告](validation/TABLETOP_VISION_VALIDATION.md) · [历史回放](../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/review/full_rgbd_grasp_step768.gif)

以下命令从仓库根目录执行，输出路径须选择尚未使用的新文件：

```bash
conda activate realman_rl
VISION_MODEL=outputs/grasp/ppo/tabletop_cartesian/rml63ii_6fb/vision_transfer_step768/model.zip
python -I scripts/evaluate/evaluate_ppo_tabletop_cartesian.py \
  --model "$VISION_MODEL" --episodes 1 --seed-start 997000 \
  --json /tmp/rgbd_6fb_fixed_retest.json
python -I scripts/evaluate/evaluate_ppo_tabletop_cartesian.py \
  --model "$VISION_MODEL" --object-radius .02 --episodes 1 --seed-start 997100 \
  --json /tmp/rgbd_6fb_random2cm_retest.json
python -I scripts/demos/render_ppo_tabletop_cartesian.py \
  --model "$VISION_MODEL" --seed 997000 --output /tmp/rgbd_6fb_full_grasp.gif
```

评估从 checkpoint 自动恢复视觉模式、相机配置及动作尺度，实际执行零位启动，禁用启动缓存。策略物体信息来自 RGB-D，仿真真值用于物理验收。回放以 4 倍仿真时间播放，包含 22.70 s 传统启动和约 76 s PPO 任务。

继续训练的示例（新目录，不覆盖现有模型）：

```bash
python -I scripts/train/train_ppo_tabletop_cartesian.py --stage full \
  --observation-source rgbd_hsv --observation-pose table_down --control-mode position \
  --init-model "$VISION_MODEL" --jaw-control delta --jaw-delta-scale .0015 \
  --reset-exploration --std .025 --jaw-std .1 --learning-rate .00001 \
  --n-envs 1 --n-steps 256 --batch-size 128 --timesteps 1536 --eval-every 768 \
  --seed 2031 --run-dir outputs/grasp/ppo/tabletop_cartesian/rml63ii_6fb/continue_new
```

这是权重迁移并重建优化器的新增训练，不是完整优化器状态的跨进程恢复；进一步训练后的模型需要重新评估。

## 真机接口准备

部署设备已确认是 **RML63II-6FB**。实机法兰到夹爪和法兰到腕部相机的外参待标定。[实机本体](../assets/robots/rml63/SOURCE.md)

### 标定实物方块的 HSV

把方块放到实际抓取的桌面上，用部署时的彩色相机、观测角度和灯光标定。在桌面会话中打开终端，从仓库根目录运行以下命令。界面使用 Tkinter。如果尚未安装 OpenCV，先执行 `python -m pip install -r requirements-vision.txt`。

```bash
conda activate realman_rl
python -I scripts/demos/calibrate_cube_hsv.py
```

1. 在 **摄像头** 下拉框中选设备。也可以直接输入设备编号并点击 **打开**。插入新设备后点击 **刷新列表**。
2. 左侧是相机原图，右侧是二值掩膜。拖动 H、S、V 的六个上下限滑块，直到方块为白色，桌面和夹爪为黑色。H 采用 OpenCV 的 `0–179`，S 和 V 为 `0–255`。调参时可把方块挪到几个典型位置，观察掩膜是否稳定。
3. 点击 **保存 HSV**。窗口显示 `hsv.json` 的路径，默认在 `outputs/calibration/` 下按保存时间创建目录。启动时可以加 `--camera 2` 直接打开指定编号，或者加 `--output-dir outputs/calibration/cube_hsv_01` 指定首次保存的新目录。

单个色相区间可把 `hsv.json` 的 `vision_hsv_low` 和 `vision_hsv_high` 传给 `CartesianTabletopConfig` 的同名参数，或传给 `OrangeCubeDetector` 的 `hsv_low` 和 `hsv_high` 参数。需要匹配跨过 `179/0` 的红色色相时，把 H 下限调到高于 H 上限。界面会合并两段掩膜，`hsv_ranges` 会保存两组边界，部署代码可对两次 `inRange` 的结果按位或。

### 三维定位接口

`perception.RgbdFrame` 只需要 RGB、对齐的米制深度、内参 K、相机到基座的刚性变换及时间戳，不依赖 MuJoCo。真实相机可由硬件适配器提供同样数据：

1. 深度对齐到 RGB，统一深度单位为米，提供 RGB 相机实际内参和畸变系数。
2. 通过手眼标定获得工具到相机变换，并与关节编码器/FK 得到的工具位姿组合成相机到基座变换。
3. 使用真实采集时间和匹配的机器人位姿，提供实际夹爪开口/参考等机器人自身状态。
4. 按实际方块尺寸和光照校准尺寸先验、HSV 阈值；目标丢失时停止，禁止补入模拟器物体状态。

当前使用 MuJoCo 的相机内参与仿真安装变换。实机只读查询核实了本体型号和运动学参数。真实相机的内参、手眼外参、噪声和时间同步仍待验证。

接口依据：[OpenCV HSV 分割](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)、[OpenCV 相机投影约定](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)、[MuJoCo 相机约定](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)。
