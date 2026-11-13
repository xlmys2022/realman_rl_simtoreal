# 验证报告索引

以下历史报告保留原始实验数据、结论和旧版路径；历史命令对应报告当时的结构。04-F2及后续报告使用重构后的入口。当前运行入口见 [README](../../README.md)，完整路径对照见 [重构报告](PROJECT_ORGANIZATION_REFACTOR.md)。

- [项目结构重构（本次）](PROJECT_ORGANIZATION_REFACTOR.md)
- [初始模型检查](VALIDATION.md)
- [动力学](DYNAMICS_VALIDATION.md)
- [运动学](KINEMATICS_VALIDATION.md)
- [控制与重力补偿](CONTROL_VALIDATION.md)
- [厂家运动学对照](REALMAN_KINEMATICS_VALIDATION.md)
- [厂家 IK](REALMAN_IK_VALIDATION.md)
- [02-D 早期 src 工程重构](REFACTOR_VALIDATION.md)
- [ReachEnv](REACH_ENV_VALIDATION.md)
- [Reach 任务与基线](REACH_TASK_VALIDATION.md)
- [夹爪](GRIPPER_VALIDATION.md)
- [抓取接触](GRASP_CONTACT_VALIDATION.md)
- [SAC](SAC_REACH_VALIDATION.md)
- [SAC 诊断](SAC_DIAGNOSTICS_VALIDATION.md)
- [BC](BC_REACH_VALIDATION.md)
- [BC 初始化 SAC](SAC_BC_INIT_VALIDATION.md)
- [纯 Reach](PURE_REACH_VALIDATION.md)
- [PPO](PPO_REACH_VALIDATION.md)
- [04-F2 PPO三维随机Reach训练准备与泛化验证](PPO_RANDOM_REACH_VALIDATION.md)
- [04-F3 PPO目标敏感性、两方向课程与六方向预检](PPO_GOAL_CONDITIONED_VALIDATION.md)
- [Reach双观测模式与Fetch-like 12维验证及用户Pilot](REACH_FETCH_OBSERVATION_VALIDATION.md)：314项开发测试通过；16384步随机Reach独立100目标成功3%，方向偏置尚未解决。
