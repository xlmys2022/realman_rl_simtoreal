# 零位起点的桌面 PPO 抓取抬升

完整任务已经通过仿真验证：机械臂从用户批准的六轴全零开发姿态、60 mm 夹爪开口开始，接近水平桌面的方块，夹取、抬升并连续稳定保持 2 秒。零位是临时开发起点；示教器打包姿态的准确角度仍可通过 `--initial-q-deg` 配置。

## 场景和策略

独立场景 `assets/scenes/tabletop_pick.xml` 包含高 0.75 m、尺寸 1.2 m × 0.8 m 的固定水平桌面，机械臂基座固定在桌面上。20 mm、10 g 方块中心初始位于 `(0.4,0,0.76)` m，使用自由关节、重力和接触摩擦；物体没有焊接、重力补偿或外力注入。位置初始化仅在 reset 中发生，完整 reset 的仿真时间、关节角和速度均为零。

PPO 同时输出六轴绝对角度（度）和夹爪绝对开口，共 7 维动作，输入为 56 维状态。运动学提供接近、夹取和抬升的目标特征；v6 奖励包含关节目标动作正则及远处张开夹爪的动作正则。子目标切换更新观测与奖励，执行动作来自神经网络。训练没有 BC 或示范轨迹。

选中模型使用方块上方 60 mm 的接近子目标，横向误差门槛为 5 mm，竖直容差 15 mm，姿态误差门槛 0.03 rad。探索采用完整协方差高斯分布、每步重采样，关节参考平滑系数为 0.05。关节参考共用插值比例，每 50 ms 最多变化 0.5°；每个控制周期包含 50 个 1 ms 物理步。

## 冻结评估结果

开发集选择后冻结 checkpoint，再使用 `--full` 恢复完整任务要求，对种子 360000–360019 复测：

- 20/20 成功，零安全终止。
- 离台间隙 28.679 mm，世界竖直抬升 28.856 mm。
- 连续稳定保持 2.031 秒，每回合 432 步、21.6 秒仿真时间。
- 最终左右指垫法向力约 0.131 N，桌面支撑力为零。

目标抬升为 30 mm，高度容差为 2 mm。成功还要求双侧指垫法向力大于 5 mN、支撑力小于 1 mN、横向误差小于 3 mm、线速度小于 2 mm/s、角速度小于 0.1 rad/s，并在完整 2 秒内保持相对滑移小于 2 mm；逐物理步检查这些条件。

方块位置固定，20 个种子复测相同几何条件，用于验证确定性复现。随机方块位置和更换初始姿态的泛化不包含在本结果中。

[冻结模型](../../outputs/grasp/ppo/tabletop_zero_seed2027/final/model.zip) · [完整回放](../../outputs/grasp/ppo/tabletop_zero_seed2027/learned_zero_pick.gif) · [逐回合数据](../../outputs/grasp/ppo/tabletop_zero_seed2027/heldout_360000_360019.json) · [简明结果](../../outputs/grasp/ppo/tabletop_zero_seed2027/summary.json)。运行命令见 [README](../../README.md#水平桌面的完整抓取任务)。

## 来源和物理检查

选中来源链共 112640 步，包含 7 个已完成 PPO 更新的课程 checkpoint；各阶段 actor 均值和 critic 均有真实更新。权重迁移仅按配置重设探索方差，重建优化器和计数器。统计只计入实际继承的 checkpoint 前缀，未选中的试验和后续更新另行保留。

核对模型文件 SHA-256、源码、资产、来源模型权重、初始和最终网络参数及优化步数。评估前后的模型参数、优化器及训练计数不变；逐步记录的七维原始网络动作与实际 Box 裁剪一致。原始 checkpoint 和各阶段初始权重随冻结结果归档。

复测观测到的最大实际关节速度为 10.422°/s，最大 mimic 残差为 0.000688 rad，最终横向误差为 0.453 mm。原有 30°/s 关节速度、夹爪 1 rad/s、mimic 0.001 rad 和六轴总驱动力矩边界均保留。预检保持不安全参考，真实接触和抬升仍由物理执行器产生。

[选择记录](../../outputs/grasp/ppo/tabletop_zero_seed2027/selection.json) · [来源链](../../outputs/grasp/ppo/tabletop_zero_seed2027/lineage_audit.json) · [完整核对](../../outputs/grasp/ppo/tabletop_zero_seed2027/verification.json)。早期试验按输出目录规则归档到 `outputs/grasp/ppo/tabletop_early_experiments/`，1193 个原始文件的大小和 SHA-256 均保持一致，历史路径见该目录的 `MIGRATION.json`。

## 回归测试

在 `realman_rl` 中执行 `python -I -m unittest discover -s tests`，437 项测试全部通过。新增覆盖桌面厚度、精确零位、七维动作、同步限速、接触预检、自由物体、接近子目标、完整协方差、PPO 真实更新、模型重载、只读推理及训练参数继承；旧 Reach 和四维局部抓取测试继续通过。

[测试结果](../../outputs/grasp/ppo/tabletop_zero_seed2027/regression.json) · [完整日志](../../outputs/grasp/ppo/tabletop_zero_seed2027/regression.log)。
