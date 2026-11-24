# 桌面末端控制任务 tabletop_cartesian

本功能与现有 reach、grasp_lift、tabletop_pick 并列，使用同一个工程和 `realman_rl` 环境。原有任务、控制方式和模型保留。

新增 `rgbd_hsv` 视觉模式已完成完整抓取：固定位置 3/3、半径 2 cm 随机位置 3/3，输入中的全部物体信息来自腕部 RGB-D 相机，详见 [视觉接口、模型和命令](tabletop_vision.md)及[视觉验证报告](validation/TABLETOP_VISION_VALIDATION.md)。下文的 r13、54/60 和接触力观测均属于保留的默认 `state` 模式，不是视觉模型成绩。

当前完整模型为 `outputs/grasp/ppo/tabletop_cartesian/fixed_observation_v2/full_table_down_seed2029_r13_secure_grip/checkpoints/step_00003072/model.zip`。固定方块 `(0.40,0)` m、`table_down` 起点独立复测 **3/3** 成功，零安全终止；成功时离台 28.962 mm，连续稳定保持 2.009 秒。每次包含实际零位传统启动，推理期间参数不变。模型使用预训练 Reach PPO 及夹持门控先验，再进行 PPO 微调。

随机位置已独立验证 **54/60（90%）**：以 `(0.40,0)` m 为中心，半径 1、2、5 cm 的圆盘各采样 20 个未训练位置，三档均为 18/20。六次失败为三次抬升不足超时和三次夹爪速度保护终止；完整标准、固定启动和模型参数不变。当前有一定位置泛化能力，尚未稳定全通过；平视起点、随机物体姿态、视觉定位误差和真机尚未验证。

## 固定观测姿态与交接

执行顺序固定为：**六轴零位 → 传统控制到所选观测姿态 → PPO 接管并执行任务**。

观测姿态是预先定义的六轴关节角及末端朝向，不根据物体位置、距离或当前训练课程改变。默认保持 60 mm 夹爪开口。机器人不会先移到方块附近再开始观测，也不会在交接后自动接近或转向。若固定启动路径被阻挡，启动会终止。

两个预设位于 `simulation/runtime/observation_presets.py`，坐标相对于当前固定场景：桌面高 0.75 m，基座 +X 为正前方。

- `table_down`：TCP `(0.10, 0.00, 1.10)` m，距基座投影 10 cm、桌面上方 35 cm，夹爪朝下，虚拟相机面向桌面。默认用于桌面任务。
- `forward`：TCP `(0.15, 0.00, 1.15)` m，距基座投影 15 cm、桌面上方 40 cm，夹爪沿 +X 水平伸出、与桌面平行，虚拟相机平视正前方。

零位到观测姿态使用关节空间五次时间插值，经过真实执行器动力学、速度/力矩及碰撞检查。到位稳定后，策略时间和步数才从零开始；第一次策略观测就在该固定姿态生成，随后的任务运动来自 PPO 动作。

物体位置不参与观测姿态目标的计算。默认 `state` 模式在交接后读取仿真物体状态；新增 `rgbd_hsv` 模式在同一个固定姿态进行相机检测，只把视觉估计送给策略。

虚拟相机使用固定光轴约定：`camera_link` 局部 +Z 沿夹爪方向、局部 +X 为图像上方。两种姿态通过机器人本身的关节运动切换，相机模型不单独旋转伪造视角。RGB-D 视觉模式使用该相机及仿真内参，尚未进行硬件光学或手眼标定；当前视觉 PPO 只验证面向桌面的四维位置接口。

在仓库根目录直接查看两种启动过程：

```bash
conda activate realman_rl
python -I scripts/demos/move_to_observation_pose.py --observation-pose table_down \
  --json /tmp/observe_down.json --output /tmp/observe_down.gif
python -I scripts/demos/move_to_observation_pose.py --observation-pose forward \
  --json /tmp/observe_forward.json --output /tmp/observe_forward.gif
```

## 末端控制接口

默认 `--control-mode auto` 随所选姿态选择接口：

- `table_down` 使用原四维动作：XYZ 位移和绝对夹爪开口，29 维观测；保持朝下的末端方向。
- `forward` 使用七维**笛卡尔动作**：XYZ 位移、世界坐标旋转向量增量、绝对夹爪开口，38 维观测。PPO 负责从平视姿态转向桌面，不在启动后插入脚本转向。

也可对 `table_down` 显式选择 `--control-mode pose` 学习末端姿态。四维动作锁定朝向，因此桌面抓取的 `forward` 起点要求姿态控制接口。七维笛卡尔控制与旧七维关节控制是不同接口，模型不能混用。

XYZ 动作单位缩放为每轴 3 mm；姿态动作每轴为 3° 的世界坐标旋转增量。参考累加使用 0.3 平滑系数，再由独立 IK 与关节速度限制决定本次可执行幅度。夹爪动作 −1 为闭合、+1 为张开。零位移/旋转动作保持已发出的末端参考。

夹爪有两种接口：原 `absolute` 把动作映射为目标开口；`--jaw-control delta` 累加开口增量，零动作保持已有参考。`--jaw-delta-scale` 指定每个控制周期的最大参考增量，单位为米；兼容默认值为 2 mm，完整抓取校准使用 0.5 mm，避免对中前过早闭合。两种接口及缩放量均写入 checkpoint，评估自动恢复。增量参考始终受夹爪范围和原执行器速度约束。

工作区逐轴限幅：Z 到达边界时仍可执行有效的 X/Y 请求。默认 TCP 下界为桌面上方 5 mm，给张开及闭合夹爪的几何包络留出余量；该限制不计算抓取路径或物体目标。

29 维状态包含 TCP 相对基座位置及线速度、物体相对 TCP 及目标误差、物体线/角速度、夹爪宽度/速度/参考宽度、接触力、保持进度、末端跟踪误差及 IK 接受比例。姿态控制额外提供末端朝下目标的旋转误差、TCP 角速度、角度参考跟踪误差，共 9 维。

观测和奖励均不包含 IK 目标关节角、关节跟踪奖励或目标动作标签。PPO 输出末端请求，独立 FK/IK 与共享执行器负责关节执行。每步共享推进 `50 × 1 ms` 动力学；关节参考每 50 ms 最多变化 0.5°，物体通过真实接触摩擦抬起。

夹持观测使用两侧接触法向力向各自夹爪内向轴的投影；开口不足 15 mm 时不把对 20 mm 方块的压力计为有效夹持。原始法向力保留在调试记录。由顶面向下挤压产生的力不能充当成功条件中的双侧夹持力。

## 训练与评估

可用已训练的原 PPO Reach 初始化末端分支。新结构包含原 256/256 神经末端网络、独立夹爪网络，以及可学习的夹持力到抬升目标门控；任务输入仍为 29 维，FK/IK 继续独立执行。目标特征为末端的抓取净空和抬升笛卡尔目标，不包含 IK 解或动作标签。门控由 PPO 更新。`--freeze-arm-steps` 可在初期保留原末端权重，同时学习夹爪和目标门控；其后自动解冻。

门控采用线性层加 sigmoid，默认初始权重为 80、偏置为 −40；双侧较小内向夹力约 0.1 N 时处于过渡中点。它使初始策略在夹稳后抬升，参数仍参与 PPO 优化。`--reset-goal-gate` 可在完整任务续训时恢复这组初始化；不加此参数时保留 checkpoint 中学到的权重。它只适用于 Reach 初始化结构，不改变独立 IK，也不在执行层注入抓取动作。

```bash
python -I scripts/train/train_ppo_tabletop_cartesian.py --stage full \
  --observation-pose table_down --control-mode position --jaw-control delta \
  --init-model outputs/grasp/ppo/tabletop_cartesian/fixed_observation_v2/full_table_down_seed2029_r12_reach_delta/checkpoints/step_00012288/model.zip \
  --jaw-delta-scale .0005 --reset-goal-gate --reset-exploration \
  --freeze-arm-steps 2048 --timesteps 4096 --learning-rate .00001 \
  --n-envs 2 --n-steps 256 --batch-size 128 --std .05 --jaw-std .3 \
  --eval-every 1024 --seed 2029 \
  --run-dir outputs/grasp/ppo/tabletop_cartesian/secure_grip_new
```

上面是本次成功微调的实际配置，继承 r12 已学到的夹爪权重。首次初始化可使用 `--init-reach-model outputs/reach/ppo/random_20cm_seed2027_continue/final/model.zip`；此选项用于朝下的位置控制，平视/旋转控制继续使用普通笛卡尔 PPO。`--arm-speed-factor` 在首次初始化时缩放原末端输出层，以适应当前控制周期和稳定保持要求。后续用 `--init-model` 继承完整任务 checkpoint，原 Reach 模型文件保持不变。

所有课程从同一个所选观测预设开始。`easy` 只把成功门槛设为抬升 5 mm、保持 0.2 秒；`full` 为抬升 30 mm（2 mm 容差）、保持 2 秒。初始开口、XYZ 动作范围和观测位置不随课程改变。旧命令的 `pinch/approach/lift` 名称保留为门槛别名，已移除原来的近距离起点和 `near/half` 课程。

```bash
python -I scripts/train/train_ppo_tabletop_cartesian.py --observation-pose table_down \
  --stage easy --run-dir outputs/grasp/ppo/tabletop_cartesian/fixed_down_new --timesteps 65536
python -I scripts/train/train_ppo_tabletop_cartesian.py --observation-pose forward \
  --stage easy --run-dir outputs/grasp/ppo/tabletop_cartesian/fixed_forward_new --timesteps 65536
```

训练默认每次复位实际执行零位到观测姿态的传统控制。`--cache-startup` 是显式的训练加速选项：缓存实际验证过的固定机器人姿态，仍不靠近物体或覆盖新物体位置；日志会标明缓存复位。正式评估和回放始终关闭此选项。

当前奖励 `cartesian_task_v10` 使用非负有界势函数和接近、对中、闭合、有效夹持、抬升的逐级正向回报，折扣因子为 0.999；另外计入步数、过早闭合、动作、速度等代价。远处静止的回报为负，正确接近后逐级改善。失败不会优于先张开空爪再继续的可执行恢复路径；完成奖励高于无限期停留在已抬升状态的累计回报，分别防止提前失败和拖延完成。

对中几何反馈用物体相对 TCP 的 XY 误差平滑调节期望净空：未对中时在物体上方约 5 cm，对中后靠近物体中心上方 3 mm。双侧指垫包围方块的几何接近程度与对中后的实际闭合进度提供额外势函数；只有姿态、位置合适才鼓励闭合。过早闭合的代价高于对中后继续张开的代价，避免安全的张开接近被奖励压制。上述几何量不生成动作、关节目标、路径或持久阶段状态，也不改变固定启动；所有 XYZ、旋转与夹爪动作仍由 PPO 输出。

默认策略时限为 80 秒（1,600 个控制周期），给固定观测姿态的长距离接近后留出对中、闭合、抬升时间；传统启动不计入策略时限。完整成功门槛仍为 30 mm/2 秒，碰撞、力矩、夹爪约束和接触要求保持不变。不同奖励版本的原始回报不能直接比较。

默认 `--exploration-profile grasp_lift` 在未对中时减小夹爪噪声，对中后探索闭合；真实内向夹持力建立后沿剩余物体目标误差增加平移探索，并在目标附近恢复精细采样。旋转噪声与夹爪噪声在夹持阶段保持较小。这个过程只调整采样标准差，全部动作均值继续由网络输出；PPO 采样和更新使用同一个高斯分布。

`--std` 设置初始运动标准差，`--jaw-std` 可单独设置初始夹爪标准差。`--exploration-profile distance` 保留原距离方案，旧 checkpoint 缺少新字段时也按原距离方案恢复。可用 `--no-adaptive-exploration` 关闭状态相关幅度。更换奖励时，`--reset-critic` 仅重置价值分支，`--reset-exploration` 重置探索幅度，均保留已有 Actor 均值权重。

从头训练时，夹爪输出初始化为保持固定观测姿态的初始开口。迁移旧权重时可用 `--reset-jaw` 单独重新初始化夹爪输出头；这会改变初始夹爪均值，保留末端其它动作、共享隐层和价值网络。该输出头继续参与 PPO 更新，不在执行中固定开口。

新 checkpoint 使用 schema 2，并记录固定预设、六轴目标及交接方式。以前的近物体课程模型不满足该启动契约，评估/回放会明确拒绝。兼容的旧四维网络可以通过 `--init-model` 作为四维训练的权重初始化；新的运行会从固定观测姿态重训并记录 `startup_changed`。四维网络不能直接用于七维姿态接口。

接触测量版本也写入 checkpoint。旧的法向力总和观测与当前内向力观测含义不同，因此旧模型只允许作为 `--init-model` 重新训练；历史评估应使用该次运行保存的 `source_snapshot.zip`，避免将新环境结果误记为原运行成绩。每次训练记录 Actor/Critic 哈希、优化器更新次数、每回合最大抬升/有效夹力及最小末端距离。

```bash
python -I scripts/evaluate/evaluate_ppo_tabletop_cartesian.py \
  --model outputs/grasp/ppo/tabletop_cartesian/fixed_observation_v2/full_table_down_seed2029_r13_secure_grip/checkpoints/step_00003072/model.zip \
  --episodes 3 --json /tmp/fixed_observation_eval.json
python -I scripts/demos/render_ppo_tabletop_cartesian.py \
  --model outputs/grasp/ppo/tabletop_cartesian/fixed_observation_v2/full_table_down_seed2029_r13_secure_grip/checkpoints/step_00003072/model.zip \
  --output /tmp/fixed_observation_task.gif
```

默认完整评估保留 checkpoint 所选的固定观测姿态，并恢复 30 mm/2 秒任务。`--saved-task` 只保留课程成功门槛，不能恢复近物体起点。姿态控制模型可用 `--observation-pose` 显式选择另一个预设。

随机位置评估可显式传入 `--object-radius`（米，范围 0–0.05）。方块以配置中的 `(0.40,0)` m 为中心，在圆盘内按面积均匀采样：角度均匀，距离为 `R * sqrt(U)`；这不是 X/Y 各自独立的正方形范围。完整模式保留这个显式随机范围，同时恢复完整成功标准；不传参数时仍默认固定位置。每回合记录实际初始物体坐标 `initial_object_position_m`，启动始终是同一个固定观测姿态。

```bash
python -I scripts/evaluate/evaluate_ppo_tabletop_cartesian.py \
  --model outputs/grasp/ppo/tabletop_cartesian/fixed_observation_v2/full_table_down_seed2029_r13_secure_grip/checkpoints/step_00003072/model.zip \
  --object-radius .02 --episodes 20 --seed-start 870000 \
  --json /tmp/random_position_radius2cm.json
```

回放支持相同的 `--object-radius` 和 `--seed`，可复现某个具体位置。评估和回放不会更新模型；存在未完成任务的回合时返回码为 1，同时保存结果文件。

两种物理启动、物体位置独立性、PPO 接管边界及两种动作接口已验证；固定方块完整任务为 3/3，随机位置为 54/60。随机位置验证期间仅增加评估随机范围参数和实际位置记录，没有训练或修改策略。训练时随机动作采样的两个完整回合未通过；随机物体位置与随机动作噪声是不同测试。过去近距离课程的 3/3 成功不计为当前任务成绩。详情见 [验证记录](validation/TABLETOP_CARTESIAN_VALIDATION.md)。
