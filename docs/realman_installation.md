# RML63II-6FB 实机安装适配

## 本体与装配依据

本轮通过 `192.168.1.18:8080` 实时读取本体枚举 `RM_MODEL_RM_63_II_E`、力版 `ISF` 和独立的三代控制器字段。实机改进 DH 的 `d1=172.0 mm`、`d6=132.2 mm`。SDK 1.1.7 离线库的 II 和 III 型 `d1` 分别为 172.0 和 162.5 mm，两者的名义 `d6` 均为 132.3 mm。现有 II 型 URDF、MJCF、mesh 和控制器 DH 一致，关节方向、六轴零位、法兰轴和限位继承原模型。第三轴范围为 −178° 至 +145°，第六轴为 ±360°，其余四轴为 ±178°。[本轮控制器读回](../outputs/validation/realman_installation/live_controller.json) · [厂家型号枚举](https://develop.realman-robotics.com/robot/apic/type/) · [控制器字段](https://develop.realman-robotics.com/robot/apic/struct/robotInfo/) · [RML63 本体与 D-H](https://develop.realman-robotics.com/robot/robotParameter/RML63OntologyParameters/)

`legacy` 继续读取原场景和原观察位。显式选择 `realman_nominal` 可使用新的桌面安装几何、D435 内参和绿色方块，腕部相机安装位沿用旧几何。`realman_hand_eye` 在同一安装模型上单独导入手眼候选，报告记录父坐标系和标定文件 SHA-256。后者用于坐标和投影试验，实机一致性另行验收。

- 世界系桌面为 `z=0.750 m`，新基座安装面为 `z=0.790 m`。基座轴与世界轴平行，`p_base = p_world - [0, 0, 0.790]`。桌面、静止方块中心在基座系分别为 `−0.040 m` 和 `−0.030 m`，方块世界系中心为 `0.760 m`。
- 原桌面场景的安装板为 8 mm，用户装配记录的名义法兰至抓取中心为 155.530235 mm。原桌面生成器按 20 mm 开口重新取得夹垫中点，实际编译的 `grasp_site` 距法兰 154.158676 mm。新场景用 147.530235 mm 名义夹爪局部抓取中心反推 **108.469765 mm 连接结构总长**，圆柱外径 57 mm 取自原安装板代理。安装面沿法兰 +Z 移至该总长，整个夹爪子树随之移动，`grasp_site` 位于法兰 +Z **256 mm**。20 mm 开口下实际仿真夹垫中点距控制 TCP 约 1.372 mm，接触验收继续使用物理夹垫。
- 工作空间下界保持世界系 `0.755 m`，即新基座系 `−0.035 m`。方块初始化、奖励和成功条件仍根据实际桌面与支撑净空计算，抬升区间与终止条件沿用原任务。测试覆盖以 `(0.40, 0)` m 为中心的 50 mm 采样边界。
- 新俯视 TCP 取世界系 `(0.19, 0, 1.10)` m 以适应 D435 视野。使用原前视点 `(0.15, 0, 1.15)` m 进行直达时遇到约 3.46 mm 桌面穿透，前视预设改为 `(0.15, 0, 1.19)` m 并重新逆解。新观测点仍与随机目标位置无关，控制器按原限速和碰撞规则从六轴零位启动。
- 设备反馈的夹爪最大开口原始量为 1001。仿真仍用夹垫间距米值与原 EG2-4C2 关节模型，真实间距与 SDK 位置字段的映射待现场测量。

[修改前后参数对照 CSV](../outputs/validation/realman_installation/parameters.csv) · [当前姿态下 MuJoCo FK 与实机 Grip_plus 对照](../outputs/validation/realman_installation/live_fk_check.json) · [SDK II/III 对照](../outputs/validation/realman_installation/fk_ii_iii.json)

## 末端动力学

示教器活动工具 `Grip_plus` 读回 `pose=[0,0,0.256,0,0,0]`、`payload=0.906 kg`，质心字段为 `[10,44,204] mm`。厂家结构体给出负载 kg、质心 mm 的字段单位。当前审计保留示教器原始质心读数，作为整体工具配置的记录。[厂家负载与质心接口](https://develop.realman-robotics.com/robot/apic/struct/frame/)

新旧场景的有质量附件均合计 0.1052992 kg，其中夹爪 CAD 质量 0.0942992 kg、原安装板名义质量 0.011 kg。新场景仍沿用该板局部惯量和原夹爪各连杆惯量，圆柱延伸部分只提供几何与碰撞。相机和支架的质量分布待实测。使用原局部惯量和安装变换计算后，20 mm 开口下组件法兰系质心 Z 从 56.501 mm 变为 146.475 mm。新几何的抓取验证采用这套已知动力学基线，真实总负载和惯量差异仍是后续部署约束。[组件质心与惯量审计](../outputs/validation/realman_installation/geometry.json)

## 手眼矩阵与视觉诊断

`T_A_B` 把 B 系的点变换到 A 系。法兰系由本机关节读数和控制器 DH 得到。实测 TCP 变换为 `T_flange_tcp`，组合为 `T_base_tcp = T_base_flange @ T_flange_tcp`。SDK 当前位姿是活动工具的工作系位姿，先乘 `T_base_work` 再和 `T_base_flange @ T_flange_active_tool` 对照。当前 `Base` 是单位工作系，`Grip_plus` 的 256 mm 已包含在 SDK 位姿中。

压缩包的 `collect_data.py` 原样记录当时的控制器活动工具位姿，`compute_in_hand.py` 将其传入 OpenCV 求解相机到该工具系的矩阵。现用矩阵与压缩包 README 的示例逐元素相同。确认父系需要当时的采样记录，或同一静止方块在多姿态下的一致性数据。当前本地配置的 `flange` 解释和矩阵按原样保留。固定父系为法兰时使用 `T_base_flange @ T_flange_color`，父系为 TCP 时使用 `T_base_flange @ T_flange_tcp @ T_tcp_color`。

RealSense 彩色光学系和 OpenCV 一致，X 向右、Y 向下、Z 沿视线向前。MuJoCo 的相机局部 Y 向上、视线沿 −Z。导入 MJCF 候选时旋转右乘 `diag(1,-1,-1)`，反投影再换回光学系。D435 的 `K` 为 `fx=605.111755`、`fy=605.160339`、`cx=303.913635`、`cy=248.024475` px，640×480 彩色与深度均为 30 FPS。畸变模型为 `inverse_brown_conrady` 且五项为零。测量深度尺度约 0.001 m，MuJoCo 直接生成米制光轴深度。新场景在 20 Hz 策略周期上以相位累加采样，平均约 15 Hz。渲染器按已知点验证视锥偏移和反投影。[MuJoCo 相机内参](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera) · [仿真投影证据](../outputs/validation/realman_installation/sim_rgbd_geometry.json)

真机定位默认在基座系寻找有足够点支持的水平顶面，用其平面包围矩形估计 XY，并从顶面 Z 下移半个边长。几何信息不足时不输出中心。`--estimator legacy` 可用同一数据复查原算法。逐帧日志包含相机系中心、基座系中心、法兰、TCP、相机和控制器位姿。名义仿真 PPO 保留原估计器，便于比较更换装配与传感器参数对冻结权重的影响。

由操作员固定方块，用示教器完成 A→B 平移。两位置各只读采集 12 组，约 15 Hz。法兰平移 `[47.39, −82.79, 4.63]` mm，旋转差 0.0041°。两位置的 `FK @ Grip_plus` 与控制器 XYZ 的最大差为 0.0161 mm。现有法兰方向解释下，顶面算法的静止方块坐标变化 189.49 mm，原算法为 189.08 mm。取逆只用于离线对照时，顶面算法的变化降至 3.56 mm，但 A、B 绝对中心高度分别比预期高 31.20 和 27.98 mm。把同一矩阵直接视作 TCP 外参时，两位置的绝对高度均偏低约 285–292 mm。这组数据把主要差异定位到当前导入的旋转解释，同时保留绝对平移和来源待核实。[实机 A/B 诊断](../outputs/validation/realman_installation/overhead_audit.json)

同一静止方块还采集了 12 组斜视 C。C 的工具轴偏离竖直向下 25.25°，从 A 到 C 的法兰旋转差 37.54°。A/C 的中心差在现有解释下为 170.49 mm，取逆候选仍有 97.43 mm。当前解释的顶面估计输出为 A 10/12、B 12/12、C 8/12，三个静止组的各轴标准差在约 1.14 mm 内。静态重复性和当前 FK 精度均可复现，绝对高度及跨姿态一致性仍未通过。标定原文件保持原始内容，手眼候选继续单独使用。[A/B/C 全部诊断](../outputs/validation/realman_installation/three_pose_audit.json)

```bash
python -I scripts/diagnostics/check_realman_arm.py --ip 192.168.1.18 --gripper --kinematics
python -I scripts/demos/locate_real_cube.py --validate-calibration
python -I scripts/demos/locate_real_cube.py --ip 192.168.1.18 --no-preview --frames 60

python -I scripts/diagnostics/audit_realman_vision.py capture --ip 192.168.1.18 \
  --output outputs/validation/realman_installation/overhead_A.npz
# 由操作员用示教器完成平移，保持角度与方块位置，机械臂静止后再采 B
python -I scripts/diagnostics/audit_realman_vision.py capture --ip 192.168.1.18 \
  --output outputs/validation/realman_installation/overhead_B.npz
python -I scripts/diagnostics/audit_realman_vision.py compare \
  outputs/validation/realman_installation/overhead_A.npz \
  outputs/validation/realman_installation/overhead_B.npz \
  --output outputs/validation/realman_installation/overhead_audit.json
# 斜视复查时，在 compare 命令的位置参数中追加斜视的 .npz 采样文件
```

## 仿真复现

```bash
conda activate realman_rl
mkdir -p outputs/validation/realman_installation_replay
OUT=outputs/validation/realman_installation_replay
MODEL=outputs/grasp/ppo/tabletop_cartesian/rml63ii_6fb/vision_transfer_step768/model.zip

python -I scripts/models/build_tabletop_cartesian_scene.py --realman
python -I scripts/demos/move_to_observation_pose.py --scene-profile realman_nominal \
  --json "$OUT/setup.json" --output "$OUT/setup.gif"
python -I scripts/diagnostics/validate_realman_tabletop.py --episodes 3 --object-radius .05 \
  --json "$OUT/rule_random.json"
python -I scripts/diagnostics/compare_fk.py --hardware-profile --json "$OUT/fk.json"
python -I scripts/diagnostics/compare_fk.py --generation all --sensor ISF --json "$OUT/fk_ii_iii.json"
python -I scripts/diagnostics/compare_ik.py --hardware-profile --repeats 1 --json "$OUT/ik.json"
MUJOCO_GL=egl python -I -m unittest discover -s tests

for PROFILE in legacy realman_nominal; do
  python -I scripts/evaluate/evaluate_ppo_tabletop_cartesian.py --model "$MODEL" \
    --scene-profile "$PROFILE" --episodes 3 --seed-start 997100 --object-radius .02 \
    --compact --json "$OUT/ppo_$PROFILE.json"
done
python -I scripts/plots/plot_tabletop_comparison.py --baseline "$OUT/ppo_legacy.json" \
  --corrected "$OUT/ppo_realman_nominal.json" --out "$OUT/comparison.png"
```

评估完整保留老任务的抬升与保持门槛。评估脚本对未全成功的批次返回 1，同时写出结果。Reach 按实际 TCP 距夹取参考不超过 6 mm 计，Lift 按双侧接触和任务离台阈值计。碰撞率为出现意外物理接触的回合比例，另记碰撞预检拒绝步率。IK 失败率用实际求解调用为分母，关节限位率用受关节限位预检拦截的控制步为分子。动作平滑性为相邻裁剪后归一化动作的平均 L2 差。滑移为连续双侧夹持期间的方块相对夹垫最大位移，未夹持的回合用空值表示。

最终全量单元和回归测试 534 项通过，其中本阶段新增 5 项安装验证。历史 CLI、奖励与观测契约及冻结权重检查均通过。[全量回归日志](../outputs/validation/realman_installation/full_regression_final.log)

- 传统分段控制完成固定点 1/1、50 mm 半径随机点 3/3，随机点的平均最大离台高度 28.82 mm，平均最大滑移 1.23 mm。8 个采样边界位置均通过 TCP IK 和碰撞预检。[固定传统抓取](../outputs/validation/realman_installation/rule_fixed.json) · [随机传统抓取](../outputs/validation/realman_installation/rule_random.json) · [新观察位回放](../outputs/validation/realman_installation/startup_down.gif)
- 旧权重用 20 mm 随机半径和种子 997100–997102 成对评估。旧环境 3/3 完整成功，新环境 0/3。新环境 Reach 3/3、Lift 0/3。两次为夹爪速度超过 1 rad/s，一次为方块在桌面上的穿透超过 0.2 mm。原物理保护在三次失败时均生效。[旧环境指标](../outputs/validation/realman_installation/ppo_legacy.json) · [新环境指标](../outputs/validation/realman_installation/ppo_nominal.json) · [冻结权重对照图](../outputs/validation/realman_installation/ppo_comparison.png)
- 仿真在两个固定俯视平移点与一个 15° 斜视点测试同一静止方块。已知顶面点反投影误差最高 0.463 mm。中心相对首姿态的最大变化，名义场景约 2.38 mm，手眼候选场景约 0.61 mm。[三姿态投影与定位明细](../outputs/validation/realman_installation/sim_rgbd_geometry.json)
- 名义 SDK FK 与仿真在法兰端存在约 0.10 mm 的 `d6` 差，使用实机 DH 时两者数值一致。收敛测试中，MuJoCo 局部 DLS 自身目标 27/36，失败的 9 例均从远端初值开始。厂家 SDK 自身目标 36/36，边界自 FK 目标 11/12。现场安装场景使用邻近种子通过实际采样边界检查，通用远种子和边界结果单独保留。[实机 DH 与模型 FK](../outputs/validation/realman_installation/fk.json) · [厂商和仿真 IK 复核](../outputs/validation/realman_installation/ik.json)

## PPO 训练命令

两个入口使用相同 29 维视觉观测和 4 维动作。训练配置按原 SI 比例归一化，历史权重的固定特征尺度和网络形状可直接加载。微调加载权重及特征缓冲区，创建新优化器、新学习率和新调度状态。项目采用固定尺度编码，没有可跨场景继承的运行均值、方差。随机初始化使用原视觉网络架构和任务结构先验，训练开始时没有源 checkpoint。

```bash
conda activate realman_rl
MODEL=outputs/grasp/ppo/tabletop_cartesian/rml63ii_6fb/vision_transfer_step768/model.zip
OUT=outputs/grasp/ppo/tabletop_cartesian/realman_nominal_training

# 由操作员选定训练预算后运行，可先采用末尾所列冒烟预算
python -I scripts/train/train_ppo_tabletop_cartesian.py --scene-profile realman_nominal \
  --observation-source rgbd_hsv --stage full --init-model "$MODEL" \
  --object-radius .02 --run-dir "$OUT/finetune" --timesteps 32768 --n-envs 4 --n-steps 256

python -I scripts/train/train_ppo_tabletop_cartesian.py --scene-profile realman_nominal \
  --observation-source rgbd_hsv --stage full --jaw-control delta \
  --object-radius .02 --run-dir "$OUT/scratch" --timesteps 32768 --n-envs 4 --n-steps 256

tensorboard --logdir "$OUT" --port 6006
```

本轮两个分支各运行了 32 个策略步。复现短程验证时，将采样半径设为 0，预算替换为 `--timesteps 32 --n-envs 1 --n-steps 32 --batch-size 16 --max-steps 32 --eval-every 32 --cache-startup`，为两次运行指定新目录。

微调和随机初始化均完成一次 PPO rollout 和优化，Actor 与 Critic 参数发生更新。[微调冒烟](../outputs/validation/realman_installation/smoke_finetune/summary.json) · [随机初始化冒烟](../outputs/validation/realman_installation/smoke_scratch/summary.json)

## 后续部署顺序

1. 在原采样输出中确认相机矩阵的方向、当时的活动工具系及实际 D435 序列号。固定俯视版本先用已知基座参考位置复查绝对高度和 XY，再复查平移与斜视一致性。
2. 测量真实夹爪开口映射、连接件与支架的几何、总载荷的部件分布和惯量。用这些实测数据更新名义模型，复查定姿态传统抓取。
3. 以几何和动力学验收通过的仿真场景重新训练。重点复查夹持接触、滑移、抬升和保护触发率，再准备固定俯视部署。

## 修改文件

- `hardware/realman_config.py` 集中现场安装量、负载记录、D435 参数和反推结构尺寸。`hardware/realman.py` 增加只读 DH 与关节配置查询。
- `paths.py`、`scripts/models/build_tabletop_cartesian_scene.py` 和 `assets/scenes/tabletop_cartesian_realman.xml` 声明并生成新场景资产。
- `tasks/grasp/tabletop_cartesian.py` 定义场景与传感器默认项。`simulation/runtime/observation_presets.py` 和 `observation_startup.py` 按实际安装选择安全观察点。`simulation/runtime/tabletop.py` 应用刚体变换和手眼候选。
- `simulation/devices/camera_geometry.py` 与 `mujoco_rgbd_camera.py` 处理校准针孔模型和光学轴。`perception/rgbd.py` 增加顶面估计并保留原检测器。
- `simulation/physics/grasp_contact.py` 和 `envs/grasp/tabletop_cartesian_env.py` 从实际几何读取方块半边长。`envs/grasp/tabletop_cartesian_vision_env.py` 给新场景配置相机采样频率。
- `simulation/motion/cartesian_controller.py` 记录原预检与求解结果。`evaluation/tabletop_cartesian.py` 加入按阶段和拒绝原因统计。`rl/ppo_cartesian.py` 根据 checkpoint 核对对应场景并提供显式跨场景评估。
- `apps/cli/tabletop_installation.py` 注册附加参数。`apps/cli/tabletop_cartesian.py`、`apps/training/tabletop_cartesian.py` 和 `apps/visualization/tabletop_cartesian.py` 将场景选项贯通到训练、评估和回放。
- `deployment/calibration.py` 对外复用 HSV 文件解析。`deployment/vision.py`、`apps/cli/real_hardware.py` 与 `scripts/demos/locate_real_cube.py` 暴露中间位姿和新旧估计器选项。
- `scripts/diagnostics/check_realman_arm.py`、`audit_realman_vision.py` 和 `validate_realman_tabletop.py` 分别提供实机只读核对、RGB-D 采样及仿真传统抓取验证。`scripts/plots/plot_tabletop_comparison.py` 绘制成对结果。
- `tests/test_realman_installation.py` 覆盖实机 DH、256 mm TCP、雅可比、装配几何、路径与 RGB-D 跨姿态定位。`tests/test_package.py` 精确核对已安装的四组 extras。`README.md`、`docs/realman_deployment.md`、`docs/FILE_MAP.md` 和本文提供设备与仿真入口。
