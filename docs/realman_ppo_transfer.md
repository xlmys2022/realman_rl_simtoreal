# RealMan PPO 迁移准备

`realman_transfer` 是固定俯视的 MuJoCo 迁移场景，供训练、评估和离线回放显式选择。它使用实机位姿、D435 安装和夹爪方向。`legacy`、`realman_nominal` 和 `realman_hand_eye` 可继续用于原实验。真机侧提供影子观测与米制动作解码，执行仍由现有现场流程逐段人工确认。

## 场景依据

- 使用控制器已核对的 RML63II-6FB 运动学、256 mm `Grip_plus` TCP 和 108.469765 mm 名义连接件。MuJoCo 世界系与 Base 的轴平行，Base 在世界 Z 0.790 m，桌面在世界 Z 0.750 m。在 Base 中桌面为 −40 mm，静止 20 mm 方块的几何中心为 −30 mm。[安装与 FK 验证](realman_installation.md)
- 旧 MuJoCo 夹爪沿 TCP X 闭合。现场 RGB-D 对两侧夹指回投，测得闭合轴与 `Grip_plus` Y 的夹角在 1.49° 内。迁移场景把夹爪子树绕法兰 Z 转 90°，同时反转 `grasp_site` 的局部旋转，保留 TCP 原点和姿态。整套夹指、碰撞几何和关节随装配旋转。[夹指方向记录](../outputs/validation/realman_installation/onsite_grasp_jaw_axis_20261127.json)
- 固定观察位的 Base TCP 约为 `(316.57, −9.11, 45.96) mm`，取自已配对的俯视录像。MuJoCo 用 FK 反解出的等效关节角启动，手腕保持该姿态。默认方块 XY 约为 `(334.23, −9.65) mm`，来自同一高位批次。这个中心的 10 mm 和 20 mm 边界各有 24/24 个采样点可见，30 mm 边界为 20/24，50 mm 边界为 16/24。因此该场景支持 `--object-radius` 至 20 mm，超界配置在创建物理环境前报错。[现场观察位记录](../outputs/validation/realman_installation/onsite_grasp_flat_observation_20261127.npz) · [启动视野扫描](../outputs/validation/realman_transfer/sim_visibility_probe.json)
- 手眼读取最新 `calibration.20261127.json`。渲染相机使用 `T_flange_tcp @ T_tcp_color`，OpenCV 光学轴转到 MuJoCo 时翻转 Y、Z。真机回放对重建的相机位姿左乘 `base_leveling`，该项校正基座水平基准，渲染装配只使用物理手眼变换。实机 HSV、640×480 D435 内参及顶面估计器统一。仿真每 50 ms 一次控制，平均每秒 15 次视觉更新，深度取 1 mm 分辨率。现场序列平均 14.50–14.60 Hz。深度噪声、对齐边缘缺测和曝光变化仍由真实数据验证。[现场标定](../assets/calibration/realman/calibration.20261127.json) · [配对回放](../outputs/validation/realman_transfer/rgbd_comparison.json)
- 张满实测夹指净距 71 mm，原 CAD 机构的名义上限 66.217 mm。仿真继续使用已验证的关节行程、安全边距及 60 mm 初始夹距。反馈 999/1001（张满）、−2（空夹）以及 241/264（持物）随状态存档。台上夹持的两次力反馈为 140 g 与 787 g，缺少全行程米制映射、夹爪动态和摩擦测量。控制器工具配置为 0.906 kg、法兰系质心 `(10, 44, 204) mm`，模拟附件保留总计 0.1052992 kg 的既有惯性模型。仿真方块名义质量 10 g，实物质量待称。该差异限制对速度跟踪、滑移和持物能力的外推。[现场夹爪回读](realman_visual_grasp.md) · [附件惯性核对](../outputs/validation/realman_installation/geometry.json)

## 统一策略接口

`VisualPolicyProprioception` 的坐标都以 Base 表达，时间用与相机同域的主机单调时间，夹距、速度和位置用 SI 单位。仿真和影子接口共用 `advance_visual_observation`，结合 `RgbdFrame` 跟踪值编码 `float32[29]`。字段顺序沿用视觉 PPO，依次是 TCP 位置和速度、方块相对 TCP 和目标相对方块的位移、方块速度和表面法向、夹距与速度及开口参考、视觉夹持置信度与检测置信度、测量年龄、视觉保持进度、TCP 参考跟踪误差及 IK 接受比例。抬升目标由首次有效视觉测量建立。仿真真值用于物理审计和既有奖励计算。视觉过期时仿真终止回合，影子接口拒绝继续生成观测。[共用代码](../src/realman_rl_simtoreal/tasks/grasp/tabletop_policy_io.py)

动作是 `float32[4]`、值域 `[-1, 1]`。前三项是相对上一周期 TCP 参考的 Base XYZ 增量，第四项是配置指定的绝对开口或相对上一开口参考的增量。两种模式均使用既有的米制夹距归一化尺度。迁移档限制 XYZ 合成请求位移为每 50 ms 最多 0.25 mm，对应 5 mm/s 的现场分段保护上限。仿真仍按原平滑、IK、关节限位和碰撞预检执行。设备的 20 Hz 跟踪能力及实时执行时序有待独立验收。[动作接口与尺度](../src/realman_rl_simtoreal/tasks/grasp/tabletop_policy_io.py)

真实帧可由已验证的 `RealCubeLocalizer` 提供。回放使用原始彩色、米制对齐深度、内参及同时段法兰位姿。

```python
from realman_rl_simtoreal.deployment.policy_io import RealVisualPolicyIO
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig

config = CartesianTabletopConfig(scene_profile='realman_transfer', jaw_control='delta')
interface = RealVisualPolicyIO(config)
interface.ingest_frame(localized.sample.to_frame(localized.T_base_color))
policy_input = interface.observation(proprioception)
request = interface.decode_action(policy_output, proprioception.jaw_reference_m)
```

示例中的 `localized` 是已配对的实际定位结果，`proprioception` 是同域、完整的本体实测和控制器参考。现有录像可回放视觉通道，仍缺米制夹距连续值、夹距速度、开口参考、连续控制所需的 TCP 参考、IK 接受比例及同步动态 TCP 速度，因此无法构成真实策略输入。`request` 仅包含数据。现场 `grasp_real_cube.py` 的运动授权、净空和保护另有验收流程。[影子接口](../src/realman_rl_simtoreal/deployment/policy_io.py) · [传统控制现场流程](realman_visual_grasp.md)

## 配对回放

在项目根目录使用 `realman_rl` 环境。输出路径应取新文件。

```bash
LIVE=outputs/validation/realman_installation
MUJOCO_GL=egl python -I scripts/diagnostics/validate_realman_transfer.py \
  --recording "$LIVE/onsite_grasp_aligned_stable_20261127.npz" \
  --recording "$LIVE/onsite_pregrasp_low_rgbd_20261127.npz" \
  --recording "$LIVE/onsite_descend_low_rgbd_20261127.npz" \
  --reference-xy-m .329164 -.021707 \
  --json outputs/validation/realman_transfer/rgbd_comparison_repeat.json
```

配对渲染使用真机预抓取时的无遮挡视觉 XY 作为静止方块参考，比较时只切换仿真夹爪装配。三组真机中心检测分别为 20/20、16/16、15/16，Z 中值比几何中心高 1.88、1.04、1.61 mm。旧装配在两个低位均无法定位，新装配三个位姿均可检测。下降样本相对预抓取的 Y 偏移约 −3.77 mm，仿真偏移约 −4.06 mm。现场已确认方块在下降时保持静止。低位有检测结果时仍可能有遮挡偏差，几何及视觉通过标准继续独立检查。[配对回放报告](../outputs/validation/realman_transfer/rgbd_comparison.json)

## 策略评估和短程训练

冻结评估使用旧模型原始权重和特征配置。跨场景评估记录动作限幅及视觉契约变化。微调新建优化器和学习率调度，保留兼容动作头与既有固定归一化，无运行均值或方差可继承。入口按视觉契约变化重置价值网络，从零分支按现有 PPO 初始化。

```bash
MODEL=outputs/grasp/ppo/tabletop_cartesian/rml63ii_6fb/vision_transfer_step768/model.zip
RUN=outputs/validation/realman_transfer/repeat

MUJOCO_GL=egl python -I scripts/evaluate/evaluate_ppo_tabletop_cartesian.py \
  --model "$MODEL" --scene-profile realman_transfer --episodes 1 \
  --seed-start 997100 --compact --json "${RUN}_frozen.json"

MUJOCO_GL=egl python -I scripts/train/train_ppo_tabletop_cartesian.py \
  --scene-profile realman_transfer --stage full --init-model "$MODEL" \
  --n-envs 1 --n-steps 16 --batch-size 16 --timesteps 16 --max-steps 16 \
  --eval-every 16 --cache-startup --run-dir "${RUN}_finetune"

MUJOCO_GL=egl python -I scripts/train/train_ppo_tabletop_cartesian.py \
  --scene-profile realman_transfer --stage full --jaw-control delta --jaw-delta-scale .0015 \
  --n-envs 1 --n-steps 16 --batch-size 16 --timesteps 16 --max-steps 16 \
  --eval-every 16 --cache-startup --run-dir "${RUN}_scratch"

tensorboard --logdir outputs/validation/realman_transfer
```

传统控制用已知仿真目标检查接触和抬升。按现场夹持 TCP 高度下降后形成双侧夹持，TCP 上移约 10 mm，方块上移 8.39 mm，稳定承托力为 0 N。该试验确认名义接触路径可执行，也提供约 1.6 mm 的相对位移记录。[仿真传统控制验证](../outputs/validation/realman_transfer/classical_grasp_probe.json)

冻结旧权重使用种子 997100，旧场景为 1/1，迁移场景为 0/1。迁移场景在第 828 步因视觉过期终止，失去检测的步占 2.05%，Reach、Lift 均未通过，碰撞率和 IK 失败率为 0。停止时 TCP Z 约为 −9.62 mm，夹距已接近 2 mm，方块仍在桌面上。固定该仿真位姿和方块位置，2 mm 夹距无法检测，60 mm 时恢复，确认了过早收窄对可见性的影响。[旧场景冻结对照](../outputs/validation/realman_transfer/frozen_legacy_step768_seed997100.json) · [迁移场景冻结评估](../outputs/validation/realman_transfer/frozen_step768_seed997100.json) · [末段状态](../outputs/validation/realman_transfer/frozen_failure_trace.json) · [夹距对照](../outputs/validation/realman_transfer/sim_visibility_probe.json)

从零和微调各完成 16 步、1 次优化器更新，保存的 checkpoint 重新加载后通过契约校验。短程训练只验收模型、采样、更新和保存流程。示例的 16 步回合用于快速验收，正式实验应使用完整回合步数。[从零冒烟](../outputs/validation/realman_transfer/smoke_scratch/summary.json) · [微调冒烟](../outputs/validation/realman_transfer/smoke_finetune/summary.json)

最终离线全量回归按测试模块并行执行，573 项通过。测试覆盖历史控制与完成契约、坐标与工具装配、RGB-D 过期、原始夹爪量拒绝、采样范围及两端相同输入的逐位一致性。[回归日志](../outputs/validation/realman_transfer/full_regression.log)

```bash
MUJOCO_GL=egl python -I scripts/demos/move_to_observation_pose.py \
  --scene-profile realman_transfer --json "${RUN}_setup.json" --output "${RUN}_setup.gif"
MUJOCO_GL=egl python -I -m unittest discover -s tests
```

下一阶段需要成对记录分段 TCP 指令与时间同步回读、测量夹指全行程米制开口和延迟，以及在低位记录闭合后和持物抬升时的可见性。第五关节曾有现场可见晃动，缺少幅度和频谱测量，仍作为机械误差来源记录。附件各部件称重和惯性测量可用于后续补齐动力学。
