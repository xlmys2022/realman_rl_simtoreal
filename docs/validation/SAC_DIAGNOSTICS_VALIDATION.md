# 04-B：SAC失败诊断与训练边界对照

日期：2026-11-07。保留04-A全部模型、配置、Replay Buffer、日志和失败报告；仅新增显式训练边界实验。未改动力学、IK、控制增益、动作尺度、观测结构/缩放、目标采样或厂家SDK，未训练夹爪。

结论：同预算10k的最终100目标，原A成功0/100，新B成功1/100，平均末端距离7.458/3.229mm。误差、往复和价值残差有所降低，**仍未学得可靠Reach策略**。开发集均0/20，按预先标准没有追加第二seed；本轮训练到此停止。

## 实验范围与文件

- `experiments/sac_diagnostics.py`：只读加载、网络指纹、动作/物理轨迹记录、按安装版SB3公式计算soft TD目标及分类残差。
- `scripts/diagnose_sac_reach.py`：原30k SAC及规则策略的20目标完整轨迹、TD数组、静止/接近/越界轨迹；输出目录必须为空。
- `envs/reach_env.py`：可选boundary_mode及对应拒绝成本；原terminate仍为默认，原safe_dense基底不变。
- `experiments/sac_reach.py`、`train_sac_reach.py`：复用原SAC训练，允许显式传入边界配置和开发种子，追加拒绝计数/原因日志；未复制算法或引入新Replay Buffer。
- `scripts/compare_sac_boundary.py`：冻结A10k、从头B10k、开发20目标、TD对照和最终100目标；校验环境仅有声明的两个配置差异及预处理/超参数一致。
- `tests/test_sac_diagnostics.py`及`tests/fixtures/reach_terminate_04a.json`：18项新测试；fixture在改动前生成，包含原固定成功、随机超时和局部越界的完整观测/奖励/info/物理结果。
- README、architecture和本报告：语义、运行方法、证据及局限。

无新依赖。全部运行使用realman_rl及python -I，仍为MuJoCo3.17.0、Gymnasium1.4.0、SB3 2.9.2、PyTorch2.14.0+cpu。原始产物校验清单在`outputs/rl/sac_boundary/baseline_manifest.json`。所有新输出位于该目录，不能覆盖非空实验目录。

## 原30k策略的20目标动作诊断

开发目标固定为10000–10019，SAC deterministic=True，规则策略使用同一真实目标及原0.25归一化向量上限。所有诊断通过env.step及原执行器，不改实际qpos。

逐步记录目标、前后实际位置、前后命令位置、动作、两种误差、动作在误差方向的投影/夹角、请求与实际命令增量、实际末端位移、距离改善、完整关节目标、限速后的ctrl、限速残差、ctrl对应FK位置及接触/终止等字段。投影单位m，方向分别使用步前`g-p_actual`和`g-p_command`，不混为同一误差。近零动作方向标为null，不虚构角度0°。

原30k SAC：20个全部超时，4000步，最终距离均值7.08984mm。朝实际目标2020/4000=50.5%，背离1980/4000；朝命令目标1957/4000=48.925%，背离2043/4000；实际距离改善1953/4000=48.825%（改善>10nm的计数定义）。相邻反向1641/4000=41.025%。相邻对局部抵消长度总和约.7530m，属于重叠的相邻对统计，不能当成全局净路程扣除。

规则：20/20成功，300步；两个方向的动作均100%朝向目标，实际距离100%改善，相邻反向0。最终距离均值.86421mm。

SAC累计命令路径1.53979m、实际路径.68399m，比值.44421；规则.07500/.06679m，比值.89056。SAC关节指令变化率限制仅228/4000=5.7%实际触发；命令点与限速后ctrl FK的差均值.01432mm、最大.68914mm。规则未触发限速。这证明限速确实有作用，但不能把全部失败都归因于限速。

为避免微小抖动扭曲方向计数，另列请求增量≥10μm的动作：原30k 3451步，朝实际目标51.956%、朝命令目标50.130%；规则仍100%。实际距离改善>1μm的全步比例分别42.4%和100%。所有未过滤原数据保留，未只选择有利阈值。

已支持：原SAC方向一致性接近随机的一半，往复抵消明显，实际运动小于命令路径。不能只用`g-p_actual`判断反馈；独立命令误差也没有稳定被策略消除。原控制的平滑、PD响应和关节限速会滤掉一部分高频命令，但各因素贡献没有通过本实验完全因果分离。

## soft TD与终止语义核验

公式对照**本环境安装的SB3 2.9.2源码**，源文本归档在输出目录；[官方VecEnv文档](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)说明自动reset、terminal_observation和TimeLimit.truncated。诊断没有优化器、反向传播或网络更新。

对每个Replay transition，使用原存储动作查询当前两个Critic；从同一个冻结Actor随机采样下一动作，并由Target Critic和当前冻结温度计算：

```text
d_terminal = stored_done * (1 - timeout)
a_next ~ frozen_actor(s_next)
v_next = min(Q_target_1, Q_target_2)(s_next, a_next) - alpha*log_pi(a_next|s_next)
y = reward + gamma*(1-d_terminal)*v_next
delta_i = y - Q_i(s, replay_action)
critic_loss = 0.5*(mean(delta_1²)+mean(delta_2²))
```

每条transition一个固定seed随机下一动作，分批计算全buffer；不是历史训练minibatch loss，也不是无熵Monte Carlo监督。Torch RNG隔离并恢复，NumPy不取随机样本，参数指纹/步数不变。Replay的27维字段已经是固定wrapper预处理后的数据，不能再次normalize；两个goal键仍为SI。当前Q使用replay动作，和04-A的确定性Actor动作Q诊断不是同一查询。

原30k：30000条；stored_done=187、timeout=137、effective termination=50，其中39个安全终止、11个成功。所有下一状态goal均保持原Episode目标，terminal goal mismatch=0；源码确认SB3用terminal_observation替换自动reset后的观测。没有将timeout错误当terminated，也没有把新Episode状态接到旧目标上。

冻结全buffer诊断（一次随机下一动作/transition）：

- 普通29813条：奖励均值-.32060，当前Q均值167.645，TD目标168.177，残差RMSE5.4555。
- timeout137条：当前Q167.728，TD目标167.704，RMSE4.2655。正确bootstrap比错误使用原始done多约168.033的TD目标。
- 安全终止39条：目标就是奖励，均值-201.411；当前Q均值-63.291，RMSE185.214，单类MSE约34304。
- 成功终止11条：目标均值-.04210，当前Q145.556，RMSE146.511。成功终止的价值锚点同样拟合很差。
- 全部残差RMSE9.0624，MSE82.126；当前/Target Critic在同一存储动作的差绝对均值约1.926/1.161，二者都处于高值区，不能把偏高简单归因为“Target没有更新”。

原模型下一状态的熵项`-alpha*log_pi`平均约-.74394，并非平均正奖励；因此不能单靠“Q包含熵”合理化约168的正值。普通转移的TD残差相对小、终止样本误差极大，支持自举高值和稀少终止锚点拟合不良的判断。正确的timeout bootstrap会传递已有高值，但**这是约定下正确的继续任务语义，不是已证明的SB3实现错误**。把timeout改为吸收终止会改变任务，本次没有这样做。

无熵任务回报只用于任务行为比较，不作为上述soft Q的精确目标。是否改有限时域价值定义、熵目标或其他因素均未在本轮训练中改变。

## 原奖励与折扣轨迹

safe_dense基底保留：普通`-min(d/.020,1)`，终止安全失败额外`-(H+1)`（H=200时-201），gamma=.99。原未折扣排序不能直接当任意折扣轨迹的全局证明。

同一目标seed10000、相同1mm动作尺度的真实轨迹：

- 静止：200步超时，原始回报-36.38282，折扣-15.75413。
- 规则接近：13步成功，原始-1.47011，折扣-1.40543。
- 连续+X故意越界：terminate第11步workspace_limit，原始-204.11337，折扣-184.71334。
- 连续XYZ正向：terminate第5步joint_limit，原始-202.23197，折扣-194.28457。

这些实际案例没有发现“故意提前失败比静止/正常接近更高折扣回报”的优势，不能据低成功率宣称策略在骗终止。静止/无效往复比达到目标差，却远小于探索遇到-201的损失，可能形成保守行为；具体优化成因不能只凭回报大小确定。

正常非终止状态d>1mm，至少有.05的普通成本。在默认3–5mm初始目标、200步和.99折扣下，故意失败最晚一步的理想成本下界约31.5，而理想静止于home的最坏成本约21.65。这是本局部任务和稳定home假设下的补充分析，不是所有环境的通用排序保证，也不是soft Q精确监督。

## reject_and_hold的精确定义

默认`boundary_mode="terminate"`完整保留；只有显式`reject_and_hold`启用新行为：

1. 请求的Cartesian点超出home半径10mm：标记workspace_limit拒绝，不调用IK。
2. IK成功、6维有限且严格在模型URDF限位内，但候选角超home±5°：标记joint_command_limit拒绝，不裁剪解，不执行该候选。
3. 保留最近合法Cartesian命令及已发出的关节ctrl参考，用原advance_control_cycle继续25个2ms物理步。没有额外settle或第二次积分，也没有重算/修补实际qpos。
4. IK不收敛、NaN/Inf、维数错误、模型限位非法、已有参考非法继续终止，不泛化为可恢复拒绝。
5. 实际关节越限、碰撞、非有限、速度、总驱动力和1°实际姿态保护仍走原终止路径，物理故障优先于命令拒绝。

默认额外拒绝成本为1.0个普通奖励上限成本，即`r_B=r_safe_dense-1.0*is_action_rejected`；不对正常奖励整体缩放，没有成功奖金或进展奖励。1来自原普通成本最大值1，不沿用201，也未根据成功率调参。legacy distance模式对应.020m额外成本，保证单位一致，但不用于本训练。

B的info含is_action_rejected、rejection_reason和boundary_mode；拒绝本身不设is_safety_failure，物理故障仍设它。若拒绝后持有期间发生姿态故障，两标记可同时为True，仍额外有原-201终止惩罚。B的compute_reward要求独立布尔拒绝标记，支持dict/记录列表/布尔数组批量重算，与step一致，不使用旧目标的成功标记。A的原调用/奖励/默认info逐值不变。

拒绝也计一步并推进50ms，H步后超时，不存在无限免费尝试。具体+X轨迹在B中190次workspace拒绝、200步超时，原始回报-289.43729、折扣-118.78347，明显差于同模式静止-15.75413。XYZ轨迹第5步虽然拒绝了候选joint命令，实际姿态随后超过1°，仍以orientation_limit终止，折扣-195.28716。这个反例确认没有把物理保护伪装成普通拒绝。

## 受控10k实验设计

A：冻结04-A `formal_seed2026/checkpoints/step_010000/model.zip`（10000步/9900更新）。该checkpoint未单独存buffer，故只读使用30k未环回buffer的前10000条，其顺序在100k容量内未被覆盖；若buffer环回脚本会拒绝。没有把未来20000条用于A10k TD统计，更没有用于B训练。

B：seed2026，从头初始化标准SAC/空buffer，训练恰好10000步、9900次更新。完全复用原训练入口，网络、所有SAC参数、CPU线程1、33维预处理、动作尺度1mm、控制50ms、200步、1mm成功阈值、模型和目标分布一致。唯一配置差异是boundary_mode和必要的rejection_penalty=1.0，比较脚本逐项验证。

开发集10000–10019，5k/10k独立验证；最终报告集100–199不用于选参数或追加实验。首轮后是否使用seed2027的标准在protocol.json中预先记录：成功率增加≥20个百分点、平均距离降低且无真实物理保护失败。未达此标准就停止，不修改惩罚或其他因素。

本轮开发集A0/20、B0/20，均200步超时。B末端误差从A的7.76541mm降为3.74168mm，但未达到1mm成功区。因此未启动第二训练seed，不能宣称跨seed稳定改进。

开发动作对照：A10k朝实际目标49.148%、朝命令目标48.869%、距离改善47.35%、相邻反向84.25%；B分别51.039%、51.039%、48.825%、35.85%。两者方向一致性仍接近一半。A/B关节限速触发19.1%/1.3%，实际路径/命令路径比.0980/.3738。B主要减少了无效大动作，没有建立可靠目标反馈。两组开发评估都没有拒绝，故该评估差异不是靠运行时允许越界后多活几步获得。

B训练：60个完整Episode，15成功（随机探索训练成功率25%）、2次orientation_limit、43超时，另1步未完成Episode。101次拒绝，其中72 joint_command_limit、29 workspace_limit。真实碰撞、数值、速度、驱动力、实际joint_limit和IK终止均0；2个真实姿态保护保留，不能把旧任务级故障减少说成一切物理故障消失。

B训练原始平均回报-46.5154、平均长度166.65、成功Episode平均72.1333步，末端误差均值3.8690mm。冻结A10k训练76个完整Episode、7成功（9.21%随机探索训练成功率）、39超时、18 workspace_limit、8 joint_limit、4 orientation_limit；原始平均回报-115.9813、平均长度130.1842，另106步未完成Episode。两个机制的回报不得直接排名，主要看成功率/距离及分类事件；这些训练成功率也不能替代确定性独立评估。B训练263.436s、37.9598步/秒；两次开发验证129.715s，含存档总393.365s。原A10k报告约35.84步/秒；墙钟受本机同期负载影响，不把速度差宣称为机制带来的性能优化。

B训练回报中普通奖励中位数-.205625，P1约-1.13632（拒绝成本进入分布），最低-201.2238来自真实姿态故障，未移除该大惩罚。

Actor/Critic均实际更新，checkpoint参数全部有限，未触发数值发散停止。B的5k/10k checkpoint记录Critic minibatch loss约.16816/.55514，Actor loss约-15.21385/-10.28734，自动温度约.23488/.05637；这些是训练logger最后一个更新的值，不等于下一节冻结全Replay的TD均方误差。原Actor、双Critic、Target Critic和Replay均可读取；原模型参数指纹在诊断前后不变。

## 同预算TD对照

A10k：10000条，76 done、39 timeout、37真正终止（30任务/安全失败、7成功）。B10k：10000条，60 done、43 timeout、17真正终止（2物理失败、15成功）；101条拒绝保持非终止，由单独日志索引识别，而不是猜测奖励。

- A普通RMSE8.5212，B普通.43323。
- A安全终止RMSE188.532，B真实安全终止142.363（仅2条，仍拟合很差）。
- A成功终止RMSE124.959，B9.8262，仍未准确锚定近零终止值。
- B拒绝非终止101条，RMSE.56216；原A没有对应非终止类别。
- 全体RMSE A13.7743、B2.09446；当前Q均值A161.510、B10.111，soft TD目标均值A162.366、B10.228。
- 当前冻结alpha A约.371751、B.056355；下一状态熵项均值A-1.14765、B+.01934。二者都用相同auto温度配置，学得的温度不同，不是人为调参。

残差下降和Q值降低是已观察到的变化，但样本分布、终止/拒绝成本以及学得的温度都随该机制变化，不能只凭这些标量宣布学习问题解决。开发成功率仍0，成功终止/物理故障的价值拟合仍有偏差。

## 测试与证据边界

修改前189/189（110.587s）；目前完整回归207/207（原189+新增18，116.371s），未改原测试断言或安全阈值。默认A固定成功、随机超时、局部失败的冻结fixture观测、奖励、info及物理数值完全一致。B固定/随机与两种reward均通过Gymnasium官方check_env，旧无限Box提示保留，不关检查。

新测试覆盖：默认仍终止、workspace和合法模型内±5°候选拒绝、非法IK不裁剪、持有参考、一次25步物理推进、无qpos瞬移、真实关节/速度/碰撞/力矩/姿态/数值故障、标量与批量奖励、success/timeout/rejection优先级、重复拒绝成本、只读网络/计数/RNG和手算熵TD目标、评估不训练。

已支持的原因：原策略方向反馈差且大量往复；201稀疏终止样本和成功终止锚点拟合很差；timeout正确bootstrap传播了已有高值；新机制确实降低残差/大动作和开发末端距离，但仍未学会Reach。不能把原失败都归咎于关节限速，也没有证据证明终止观测接错或SAC代码实现错误。

仍为假设：动作尺度与控制带宽、累积命令参考滞后、没有进展信息的距离奖励、策略输入结构、有限预算/温度动态等各自的因果贡献。本轮没有同时改这些因素，也没有因为低成功率追加训练。

## 最终100目标：冻结模型、不再调参

开发实验结束后，使用种子100–199，deterministic=True，两组逐项相同的100个物理目标；未根据此集合选惩罚、追加训练或选择其他checkpoint。

- A10k terminate：成功0/100、超时100，平均最终距离7.458405mm；平均长度200，原始回报-73.597648、折扣回报-31.383214。没有成功Episode，成功平均步数为null。
- B10k reject_and_hold：成功1/100、超时99，平均最终距离3.228512mm；平均长度198.11，原始回报-32.304129、折扣回报-14.034330。唯一成功Episode为11步，不能据此推断成功速度的一般分布。
- 两组workspace/joint/IK/orientation/collision/numerical/velocity/drive等终止均0，拒绝均0。因此评估差异不是运行时多次拒绝后继续存活造成的假成功。B训练中确实有101次拒绝和2次真实姿态保护，不能以最终评估0故障抹去训练记录。
- A/B环境step墙钟均值16.034/16.026ms，中位16.000/15.996ms，P95 16.325/16.321ms；含reset/记录的交互速率60.549/60.602步/秒，完整评估330.312/326.903s。这不同于带网络更新的训练37.960步/秒，也不同于每step的50ms仿真时间。

04-A冻结30k的历史最终结果仍为1/100、6.712937mm；这是不同训练预算，不替代本轮同预算A10k。规则策略开发20/20，历史最终100/100也保留。B相比A10k的平均距离和往复改善，没有转化为稳定成功，不能据单seed的1次成功宣布方案有效。

另存的动作/实际位移滞后统计没有跨Episode拼接。原30k的能量归一化向量一致性在lag=0时.530、lag=2个50ms周期时.660，与响应延迟相容；规则动作长期同向，其lag统计不适合估计纯物理延迟。该结果不构成控制滞后为主要失败原因的因果证明，原始数据在lag_alignment.json。

## 完整性和可重复运行

`integrity_and_terminal_audit.json`确认：04-A原155个输出文件逐字节SHA256一致；assets、simulation、vendor及原测试文件也与基线清单一致。B初始网络权重与A原始初始网络逐张量一致，证明同seed从头开始，未混入旧模型或buffer。A30k的187个已结束Episode、B10k的60个已结束Episode，其末状态距离、done及timeout与训练日志逐项一致，错误数0。原30k仍30000步/29900更新，B10000步/9900更新。

所有产物在Git忽略的`outputs/rl/sac_boundary/`：

- `baseline_manifest.json`、`baseline_tests.log`：修改前资产/代码/旧实验清单和189项基线。
- `original_diagnosis/`：20目标SAC/规则完整轨迹、奖励反例、30k全buffer的TD摘要及NPZ原始数组。
- `controlled_seed2026/protocol.json`：训练前实验协议；`A/`、`B/`保存开发轨迹和各自TD数组；`comparison.json`记录未达到复验条件。
- `controlled_seed2026/training/`：独立B模型、原始Episode/拒绝日志、SB3损失、依赖、配置、初始模型、checkpoint和Replay Buffer。最终可独立加载模型为`training/final/model.zip`，同目录配置/预处理须一起保留。
- `final_evaluation/comparison.json`：两个冻结模型的配对100目标逐Episode、动作与汇总，不含重新训练。
- `reward_comparison.json`、`additional_action_statistics.json`、`lag_alignment.json`：补充轨迹、有效动作分层、滞后数据。
- `all_tests.log`、`integrity_and_terminal_audit.json`及SB3源码文本：207项回归、终止审计、实际使用公式来源。

在项目根目录执行；新目录名需未使用，不要覆盖已完成实验：

```bash
conda activate realman_rl
python -I scripts/diagnose_sac_reach.py --out outputs/rl/sac_boundary/new_diagnosis
# 下行会从头运行恰好10000步；本轮结果已保存，不需为读取报告再次训练
python -I scripts/compare_sac_boundary.py --out outputs/rl/sac_boundary/new_seed2026
# 仅读取已保存模型做最终评估；不会训练
python -I scripts/compare_sac_boundary.py --final --b outputs/rl/sac_boundary/controlled_seed2026/training/final/model.zip --out outputs/rl/sac_boundary/new_final_evaluation
python -I -m unittest discover -s tests -v
```

`pip check`通过（No broken requirements found），`git diff --check`通过。本次无Viewer要求，未进行新Viewer验证；没有新增依赖、提交或推送。当前结果不足以把该SAC策略用于可靠Reach控制；后续可以单独研究动作/命令动态或价值学习，但本次不开展下一因素实验。
