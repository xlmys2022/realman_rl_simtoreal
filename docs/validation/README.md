# 验证报告索引

以下历史报告保留原始实验数据、结论和旧版路径；历史命令对应报告当时的结构。04-F2及后续报告使用重构后的入口。当前运行入口见 [README](../../README.md)，完整路径对照见 [重构报告](PROJECT_ORGANIZATION_REFACTOR.md)。

- [夹取与抬升环境及运行入口](../../README.md#夹取与抬升)：四维控制、真实接触夹持、沿支撑法线抬升 30 mm 和连续稳定保持 2 秒。独立 20 个随机位置全部成功，平均离台 29.206 mm；380 项回归通过，1 ms 配置另通过 40 项相关复测，保留不闭合、不抬升、低摩擦和 30 g 失败对照。
- [随机起点与连续多目标 Reach](../../README.md#随机起点与连续多目标-reach)支持 20 cm 可达区域内的随机起点、每条 5 个目标和连续物理状态，新增训练 49152 步。独立测试完成 12/20 条轨迹、72/100 个计划目标，另有 7 次限位终止、1 次超时；60 次目标切换状态连续，364 项回归测试通过。
- [20 cm 随机 Reach 续训及验证](../../README.md#20-cm-随机-reach-续训)使用 goal_relative_v3 观测、限速笛卡尔参考轨迹和 PPO 权重迁移，新增 49152 步训练。独立 100 目标成功 97/100，2 次超时、1 次 IK 失败；目标半径 3.19–19.99 cm，成功回合平均误差 0.773 mm，351 项回归测试通过。
- [5 mm 三维随机 Reach 验证及复现命令](../../README.md#5-mm-三维随机-reach-验证)采用 goal_relative_v1 观测与 reach_progress_v1 奖励。训练种子 2026、2027、2028 的模型在相同 1000 个新目标上分别成功 1000/1000，共计 3000 回合，八个空间象限均通过，平均实际末端误差 0.859–0.862 mm，零超时、零安全终止。
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
- [Reach双观测模式与Fetch-like 12维预算诊断](REACH_FETCH_OBSERVATION_VALIDATION.md)：314项开发测试通过；16384/131072步的独立成功率分别为3%/1%（不同目标集），长训练固定开发集未持续改善，方向偏置和落点集中仍在。
- [Reach奖励诊断与进度奖励验证](REACH_REWARD_OPTIMIZATION_VALIDATION.md)：旧公式无明确错误；339项测试及一次512步Smoke通过。用户16384步双方向Pilot最终0/2成功，+X/-X误差2.531/7.335mm，方向偏置与共同落点仍在；已追加检查点、奖励分项、优化指标和冻结12维网络复核，尚未通过进入随机任务的条件。
