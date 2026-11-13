# 实验历史索引

本目录保留重构前的完整研究叙述；历史命令和 Python 导入路径属于各自原版本。当前操作请使用仓库 [README](../../README.md)、[FILE_MAP](../FILE_MAP.md) 和 [迁移表](../validation/PROJECT_ORGANIZATION_REFACTOR.md)。不要直接以已有 run 目录重新训练或覆盖旧评估结果。

- [历史 README 完整快照](README_BEFORE_REFACTOR.md)：从模型、控制、FK/IK、夹爪接触到 SAC、BC、纯 Reach 与 PPO 的原说明、手动实验及结果。
- [历史架构完整快照](ARCHITECTURE_BEFORE_REFACTOR.md)：原模块设计、单位、物理与算法约束，保留细节以便解释历史产物。
- [SAC 04-A](../validation/SAC_REACH_VALIDATION.md)：官方 SAC 实验与原训练结果。
- [SAC 04-B 诊断及边界对照](../validation/SAC_DIAGNOSTICS_VALIDATION.md)：冻结策略、奖励反例及 Replay Q/TD 分析。
- [BC 04-C](../validation/BC_REACH_VALIDATION.md)：示范数据、独立监督学习、闭环对照与覆盖局限。
- [BC→SAC 04-D1](../validation/SAC_BC_INIT_VALIDATION.md)：历史 Actor 初始化与短期退化证据，当前不作为纯 RL 默认流程。
- [纯 Reach 04-E1](../validation/PURE_REACH_VALIDATION.md)：固定 / 随机任务、奖励探针、两组手动 5000 步 SAC Pilot 及失败结果。
- [PPO 04-F1](../validation/PPO_REACH_VALIDATION.md)：完整 Rollout 审计、用户手动 16384 步固定目标 Pilot、独立评估与局限。
- [PPO 04-F2 随机Reach](../validation/PPO_RANDOM_REACH_VALIDATION.md)：旧33维用户手动16384步Pilot，90000–90099独立100目标成功4%，方向偏置及预算诊断。
- [PPO 04-F3 两方向课程](../validation/PPO_GOAL_CONDITIONED_VALIDATION.md)：旧33维±X5mm候选与Pilot，+X成功、-X超时，课程未通过。
- [Fetch-like 12维观测与Pilot](../validation/REACH_FETCH_OBSERVATION_VALIDATION.md)：双模式314项开发验收；用户随后完成16384步随机Reach，88000–88099独立100目标成功3%、97次超时、0次安全失败，±X首动作方向偏置仍在。旧33维4%使用不同目标，尚非配对A/B。

物理、控制和环境的验证记录见[验证索引](../validation/README.md)。代码组织重构本身没有新增算法实验；其后由用户完成的训练另记在对应报告中，保留开发验收和历史失败记录。更新日志只读复核已有产物，不重新训练或覆盖模型。
