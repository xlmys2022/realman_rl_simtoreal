# 腕部 RGB-D 视觉 PPO 完整抓取验证

2026-11-17，在已有 `realman_rl` 环境、同一工程中完成。视觉模式名为 `rgbd_hsv`，当前观测契约 `wrist_rgbd_hsv_cube_v4`。保留原 Reach、夹取、七维关节控制及 `state` 桌面末端模式。未连接真实机械臂。

## 验证结论和选定模型

冻结视觉 PPO 后，固定位置独立复测 **3/3**，半径 **2 cm** 的圆盘内三个随机位置 **3/3**，均通过原完整物理标准、无安全终止。三次固定位置是确定性重复复现；随机样本量为三，不能据此声称普遍 100% 泛化。原真值模式的 54/60 成绩不计入视觉结果。

选定模型为 [R4 第 768 步 checkpoint](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/checkpoints/step_00000768/model.zip)，SHA256 为 `64045c9a8ebadf1ce4cdd8d0faf750439ed1f69722412865786747434cae3542`。类型为 `VisualReachGraspPPOPolicy` / `VisualReachGraspNetworks`。该模型由既有 PPO 迁移，当前运行新增 768 步、60 次优化器更新；末端和夹爪参数实际更新，目标门控在这个早期 checkpoint 保持继承值。[检查点更新核对](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/review/selected_update_audit.json)。

R4 整轮完成 1,536 步、120 次优化器更新，Actor、Critic、末端、夹爪和门控参数均有更新，实际训练中的一个完整采样回合也通过。正式独立成绩使用上述 768 步检查点，未把最终模型和选定模型的成绩混用。[训练汇总](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/summary.json)及[训练配置与来源](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/config.json)保留完整记录。

## 实际完整流程与通过条件

每回合实际执行六轴全零 → 传统控制到固定 `table_down` 观测姿态 → PPO。零位是用户授权的开发起点，不标为厂家的打包姿态。传统启动约 23.45 秒、469 个控制周期，启动期间 PPO 动作数为零；TCP 观测目标为世界坐标 `(0.10,0,1.10)` m，六轴目标不随物体位置变化，未使用启动缓存。

交接后 PPO 输出 XYZ 与夹爪增量四维动作，独立 FK/IK 与原执行器推进 `50 × 1 ms` 物理周期。XYZ 缩放 3 mm、参考平滑 0.3、关节参考每周期最多 0.5°；该视觉模型夹爪增量尺度为 1.5 mm，仍经过原开口、速度、力矩和接触安全检查。未焊接物体、修改重力或注入外力。

视觉目标由第一帧相机物体位置加 30 mm 高度生成。相机判断 XY 误差 <3 mm、Z 误差 <1 mm、视觉速度 <1.5 mm/s、视觉夹持置信度 >0.5，并连续稳定 3 秒，才宣布视觉执行结束。三秒包含原两秒保持加一秒稳定余量，不依靠仿真成功标记计时。超过 0.25 秒失去有效检测会终止为 `vision_lost`。

相机宣布完成后，独立物理审计仍要求离台至少 28 mm（30 mm 目标、2 mm 容差）、双侧内向夹力各 >0.005 N、桌面支撑 <0.001 N、水平位移 <3 mm、物体线速度 <2 mm/s、角速度 <0.1 rad/s、相对滑移 <2 mm，且连续保持至少 2 秒。相机宣布完成但未通过物理审计会立即失败为 `visual_false_success`，不会继续等待真值成功。

独立复测数据：

- [固定位置 997000–997002](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/review/independent_fixed_997000.json)：3/3，结束离台 29.669 mm，物理连续保持 3.584 秒，视觉连续保持 3.000 秒；每回合 1,513 个 PPO 控制周期。
- [随机位置 997100–997102](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/review/independent_random2cm_997100.json)：3/3；结束离台分别 29.674、29.613、29.648 mm，物理连续保持 3.511、3.559、3.571 秒，视觉均为 3.000 秒；策略分别执行 1,521、1,481、1,549 个周期。初始基座平面坐标约为 `(0.401574,-0.005444)`、`(0.388288,0.005561)`、`(0.412037,-0.014368)` m。
- [完整回放](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/review/full_rgbd_grasp_step768.gif)及[对应物理记录](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/full_table_down_seed2031_r4_zero_anchored/review/full_rgbd_grasp_step768.json)：独立种子 997200，1/1，通过完整标准，离台 28.686 mm、物理保持 3.570 秒、视觉保持 3.000 秒。包含实际零位传统启动、策略 RGB 和 Mask；496 帧、960×720、4 倍仿真时间播放，全部帧解码成功。回放采集的相机初始估计与独立固定位置测试存在亚毫米差异，仍通过任务。

## 图像、深度与信息隔离

RGB 与米制光轴深度来自原 `camera_link/wrist_rgb`，局部相机位置保持 `(0,0,0)`。相同物理时刻采集 RGB/深度，640×480、10 Hz；策略控制 20 Hz。只使用 OpenCV RGB→HSV 阈值、连通区域及相应像素的深度，未使用物体 ID、语义分割标签或物体真值。像素通过实际内参反投影，再使用机器人相机的基座变换得到三维位置。

当前定位器结合可见深度表面及已知 20 mm 正立方块尺寸恢复中心，处理夹爪遮挡造成的 Mask 质心偏移；物体 Z 来自实时深度，未把物体高度锁定在桌面。视觉速度来自带时间戳测量差分与滤波。短暂丢失时仅保留有年龄标记的相机测量，不把目标附着到 TCP，不回退到仿真状态。

29 维 Actor 输入包含机器人自身 TCP、夹爪与参考跟踪状态，以及相机推导的相对物体位置、目标误差、速度、可见表面法向、检测置信度、年龄、视觉夹持置信度和视觉保持进度。视觉夹持置信度还使用机器人的夹爪编码器开口和参考，不包含真实接触力。物体真值位置、线/角速度、接触法向力、桌面支撑、真值保持进度均退出 Actor 输入。仿真真值仅用于物理、安全、训练奖励与独立物理审计；记录中的真值字段属于审计数据。

[机器可读验证](../../outputs/vision/validation_v4/verification.json)核对七个独立回合的 **10,601 个控制周期**：

- 从每次实际送入网络的 `policy_observation` 反算全部物体通道，逐项匹配当时可用的相机位置、目标、速度、法向、置信度、年龄及视觉保持进度。
- 使用保存模型对全部实际输入重新推理，原始网络动作最大重算误差 `4.18e-7`，Box 裁剪后的动作也一致，证明记录的执行请求来自网络。
- 推理前后模型参数、优化器计数与训练计数不变；当前执行源码与训练归档指纹匹配，旧 FK/IK 文件和两份旧桌面模型哈希不变。
- 重放原始 RGB、深度和 K/T 标定可精确复现检测及 Mask；原始 Mask 的唯一数值为 `0,255`，相机局部位置未移动。[原始 RGB](../../outputs/vision/validation_v4/detection_seed998000/rgb.png)、[二值 Mask](../../outputs/vision/validation_v4/detection_seed998000/mask.png)、[检测与标定](../../outputs/vision/validation_v4/detection_seed998000/detection.json)。

这七回合的新相机帧相对仿真真值的 X/Y/Z 位置 RMSE 约为 `0.195/0.104/0.034 mm`，只用于离线审计。这是理想渲染和当前尺寸先验下的仿真数值，不是实物深度相机精度结论。

## PPO 结构与开发失败记录

模型沿用已有 Reach/抓取 PPO 权重，在视觉观测契约下微调，没有从随机权重重新证明完整学习。原真实力、角速度与真值保持输入对应的夹爪权重在初次迁移时清除，价值网络重建，跨版本变换写入配置。独立 FK/IK 不参与神经网络监督，也不提供目标关节角、示范或关节动作标签。

视觉目标特征将相机物体目标误差送入末端网络，可学习门控融合接近和抬升特征。为消除预训练网络在零目标误差时的漂移，同一个末端网络的当前响应减去其保持参考响应；XY 1 mm、Z 0.5 mm 精度带内保持既有参考。夹爪仍由网络输出。该结构参与 PPO 采样、梯度和对数似然计算，没有在 rollout 或执行层替换动作为脚本。

保留以下早期失败，未计为当前成绩：

- R1 的旧夹爪尺度闭合过慢，完整任务未通过；修改尺度的运行时检查只是校准，不作为新保存模型的成绩。
- R2 使用两秒视觉保持曾在物理保持不足时结束，正确记为 `visual_false_success`。
- R3 加入一秒稳定余量仍暴露末端网络零误差漂移，同样未通过物理审计。随后实现上述网络结构，正式训练 R4，再冻结并完成本报告的独立测试。运行时结构校准单独保留，未冒充 PPO 更新。

四轮训练和各自的源码归档、配置、失败评估保存在 [视觉训练目录](../../outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/)。

## 测试与使用范围

79 项相关检查通过：

- [视觉专项 17 项](../../outputs/vision/vision_completion_v4_tests.log)：HSV 顺序与阈值、内参/外参和深度单位、遮挡、目标丢失、真实相机位置、模拟器调试字段和目标被改写后观测不变、相机完成与物理审计隔离、网络保持行为及 PPO 似然/梯度。
- [包与入口 10 项](../../outputs/vision/package_v4_final.log)：导入图、依赖边界和命令入口。
- [原桌面模式回归 28 项](../../outputs/vision/state_mode_regression.log)：旧控制、完整任务物理、模型保存/加载、显式随机范围保留。
- [学习结构回归 24 项](../../outputs/vision/learning_regression.log)：原奖励、分布与参数变换。

`realman_rl` 的 `pip check` 通过。上述是相关检查，不冒称全仓库全部测试重跑。完整测试和独立模拟验收完成后仅补充文档，未改变模型执行源码。

当前验证范围：20 mm、10 g 正立方块，固定桌面，`table_down` 位置控制，理想对齐 RGB-D。未验证任意物体、未知尺寸、明显旋转、真实光照和深度噪声、平视起点视觉抓取或真机。真实适配只需提供中立 `RgbdFrame` 的 RGB、对齐米制深度、内参/畸变、手眼与 FK 组合的相机到基座变换及时间戳；硬件采集驱动、标定和时间同步仍需接入并独立验收。[接口与复现命令](../tabletop_vision.md)。
