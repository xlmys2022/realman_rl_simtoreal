# Reach 奖励诊断与 reach_progress_v1 验证

日期：2026-11-14（Asia/Shanghai）。版本由用户管理，未创建分支、提交或推送。
启动时在读取附件的同时误做一次只读Git状态查询，已即时告知用户；随后停止全部Git操作，改用文件哈希核对。

开发诊断、339项测试与512步Smoke记录按完成时状态保留。用户随后手动完成新奖励的
16384步±X Pilot及独立评估；2026-11-14只读复核追加于文末：最终0/2成功，方向课程尚未通过。

## 第一阶段：修改奖励前的诊断

本节在实现新奖励前完成。实际程序为 `scripts/diagnostics/diagnose_reach_rewards.py`：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I scripts/diagnostics/diagnose_reach_rewards.py \
  --out outputs/rl/reach_reward_validation/20261114/before
```

该程序不训练策略。使用fetch_like_v1、±X5mm已验证目标、action_scale=0.00025m、
control_period=0.05s；每个动作/序列均从独立且相同的Home reset开始，不写qpos。
共54条旧奖励轨迹，覆盖3种旧奖励×2目标×（单步/短序列各3方向，以及成功/超时/保护终止3种整集探针）。
原始记录在 `outputs/rl/reach_reward_validation/20261114/before/diagnostics.json`。

### 现有公式与终止语义

- distance：`r=-d`，单位沿用米；reject_and_hold另按原规则扣0.02×rejection_penalty。
- safe_dense：`r=-min(d/0.02,1)-(H+1)*failure`，拒绝另扣原rejection_penalty。
- reach_shaped_v1：`Phi(d)=exp(-d/sigma)-1`，
  `r=gamma*Phi_next-Phi_previous+K*(success-failure)-(1-gamma)*rejected`。
  sigma=0.005m、gamma=0.99与PPO一致；H=200，K=1.01/gamma^199=7.463072298541118。
  正常成功/安全失败的Phi_next为0；TimeLimit保留后继势函数，且SB3按原流程做timeout价值自举。
  成功由传入goal重新计算，不复用历史is_success；安全/拒绝由真实info标记提供。

非终止非拒绝的势函数差保守范围为[-gamma,1]；加入拒绝成本后下界再减0.01。
成功步为K-Phi_previous，安全失败为-K-Phi_previous（如同时拒绝，再扣0.01）。
这些是数学范围，不代表机械臂每周期都能达到任意距离变化。

### 同状态动作方向与控制延迟

| 目标 | 动作方向 | 单步实际X位移(mm) | 动作后距离(mm) | reach_shaped_v1奖励 |
|---|---|---|---|---|
| +X5mm | +X | +0.022137 | 4.977869 | 0.0079368240 |
| +X5mm | -X | -0.022138 | 5.022143 | 0.0047118566 |
| +X5mm | 零 | 0 | 5.000000 | 0.0063212056 |
| -X5mm | +X | +0.022137 | 5.022143 | 0.0047118969 |
| -X5mm | -X | -0.022138 | 4.977868 | 0.0079368642 |
| -X5mm | 零 | 0 | 5.000000 | 0.0063212056 |

末端命令X位移为±0.25mm，单步实际约±0.02214mm，存在明显执行滞后；没有用命令位移代替实际进度。
正确动作的奖励在两个目标下均高于零动作，再高于错误动作，单步正误差约0.003225。
短期固定命令中，正确方向均在18步成功；错误方向25步后距目标约9.999mm。

### 解析数值与真实函数交叉核对

| 情况：前后距离 | 旧塑形奖励 |
|---|---|
| 5→4.75mm | +0.0249941720 |
| 5→5.25mm | -0.0114410696 |
| 5→5mm不动 | +0.0063212056 |
| 1.251→1.001mm，尚未成功 | +0.0417363146 |
| 1.25→1.00mm，成功 | +7.6842715155 |
| 5→5mm，安全失败 | -6.8309517397 |
| 5→5mm，正常TimeLimit | +0.0063212056 |

全部9个解析场景（另含阈值内成功、拒绝）与compute_reward逐项一致，误差为0。
静止的正值来自gamma<1与负势函数，不能单独当作实现错误。折扣累计具有
`sum gamma^t*(gamma*Phi_next-Phi_previous)=-Phi_initial+gamma^T*Phi_final` 的望远镜形式。

### 典型实际Episode及诊断结论

±X对称结果近似相同。以+X目标为例：正确脉冲18步成功，未折扣/折扣回报8.172627/6.923067；
静止200步超时，1.264241/0.547429；持续反向41步workspace_limit，-6.498605/-4.360464。
正确方向的典型回报优于停留，停留优于主动触发保护；本轮没有发现明确的公式、符号、goal重算或终止实现错误。

数学上正确不等于容易学习：真实单步进度很小、正确/错误动作即时差异小、静止有小幅正塑形值，
成功/失败的终止项相对大，可能增加信用分配和价值估计难度，但本次尚未证明它们是训练失败的原因。
此前131072步出现共同落点及连续开发集0成功，是继续调查学习信号的依据，不是旧奖励错误的证据。

后续阶段将保留上述三种旧公式，仅新增用户指定的reach_progress_v1实验模式。

## 第二阶段：reach_progress_v1 实验设计

旧奖励不是已证实的错误实现。本次增加一种学习信号形式，保留原默认reward_mode、
全部旧公式和旧模型加载路径，不据此预先宣称成功率改善。

设 `d_prev=norm(previous_achieved_goal-desired_goal)`、`d=norm(achieved_goal-desired_goal)`，
均由实际末端的世界坐标米计算。新公式为：

```text
r = w_progress * (d_prev-d) / progress_scale
    - w_distance * d / distance_scale
    - step_cost
    + success_bonus * success
    - safety_penalty * is_safety_failure
    - rejection_cost * is_action_rejected
```

| 参数 | 初始值 | 作用 |
|---|---|---|
| w_progress | 1.0 | 奖励实际距离减少，惩罚实际距离增加 |
| progress_scale | 0.001m | 1mm实际进度对应1单位进度奖励 |
| w_distance | 0.05 | 持续反映剩余距离 |
| distance_scale | 0.005m | 原5mm任务尺度，未改变目标分布 |
| step_cost | 0.01 | 每次有效env.step的固定成本，包括成功/安全失败步 |
| success_bonus | 10.0 | 当前传入goal满足阈值且无安全失败时的奖金 |
| safety_penalty | 20.0 | 仅根据真实安全失败标记扣除 |
| rejection_cost | 0.1 | 仅根据真实拒绝标记扣除 |

这些是用户指定的实验初值，集中在 `tasks/rewards.py`，不是经过调优的最优参数。
环境可接收 `reward_progress_parameters` 字典；task/训练函数支持同名可选参数，验证未知键、
非有限/负值和非法尺度。CLI的 `--reward-mode reach_progress_v1` 使用上述完整默认值，
本次没有增加超参数搜索、修改默认PPO配置或自动运行正式实验。

### 奖励接口和审计

- `compute_reward(achieved_goal, desired_goal, info)`可重算标量、批量、info字典数组或info记录数组。
- 真实step的info同时提供previous_achieved_goal、is_safety_failure和is_action_rejected；旧塑形信息保持原语义。
- 成功由当前传入goal及原success_threshold重新计算，不复用历史is_success/distance/组件；安全失败抑制成功奖金。
- `reward_components()`是纯查询，不写info或仿真。六个signed分项加起来等于total_reward，
  并附previous_distance_m、current_distance_m、distance_change_m。距离变化取“之前减当前”。
- step的 `info['reward_components']`、训练actions.jsonl、评估每个Episode的逐步组件及组件累计均可审计。
  分项是signed值，distance_penalty等已带负号，求和时不再减一次。
- 组件与动作/目标方向统计不进入observation。12维定义、原世界坐标goal、动作与全部物理保护不变。
- VecEnv仍按官方流程把奖励转成float32，timeout仍按SB3自举。新组件日志保留环境原始float64回报，
  并核对它的float32值等于采样奖励；没有改RolloutBuffer、GAE或优化逻辑。

### 配置保存与模型加载

完整8项参数同时保存在 `config.environment.reward_progress_parameters` 和顶层 `reward_parameters`。
新PPO模型通过原生SB3序列化保存一个普通metadata字典 `reach_reward_configuration`，
不改变策略类、网络、损失或分布。保存/加载核对环境参数、顶层参数及模型metadata；
将progress模型错误标成旧奖励或改错参数会明确报错。
旧模型没有这个metadata时仍按原奖励配置正常加载；实际旧33维模型冻结预测测试仍通过。
独立评估直接读取保存的环境和参数，无需用户再次指定奖励名。

## 同轨迹数值对比、奖励尺度与局限

在新实现之前冻结的18条reach_shaped_v1物理轨迹上，对两种奖励重新计分；
比较过程不执行物理step。旧奖励逐值保持原值，新旧奖励使用完全相同的实际位置、目标及安全/拒绝标记。
另外单独运行新模式真实诊断，18组轨迹的660个物理转移与修改前完全一致。

| 解析场景 | reach_shaped_v1 | reach_progress_v1 |
|---|---|---|
| 5→4.75mm | +0.0249941720 | +0.192500 |
| 5→5.25mm | -0.0114410696 | -0.312500 |
| 5→5mm | +0.0063212056 | -0.060000 |
| 1.251→1.001mm | +0.0417363146 | +0.229990 |
| 1.25→1.00mm成功 | +7.6842715155 | +10.230000 |
| 5→5mm安全失败 | -6.8309517397 | -20.060000 |
| 5→5mm时间耗尽 | +0.0063212056 | -0.060000 |
| 5→5mm拒绝 | -0.0036787944 | -0.160000 |

若真实减少0.25mm，进度项为+0.25，持续距离项仅-0.0475、步成本-0.01，进度项占主要量级。
但实际控制首步只减少约0.02213mm：新进度项+0.022131、距离项-0.049779、步成本-0.01，
总值约-0.037647；此时距离与步成本大于微小进度，不能把“方向正确”误写成“每个正确动作都获得正总奖励”。
旧新公式在两方向下都满足正确>零>错误；新首步为约-0.03765、-0.06000、-0.08236，
正误差约0.044717。这个尺度变大不等于PPO必然更容易学习。

在上述实际轨迹样本中，旧塑形逐步奖励范围为[-6.517579,7.676844]，新奖励为[-20.155483,10.230493]。
新进度与距离项对任意有限世界goal没有固定的全局数值上/下界；这些只是本次实际样本范围，不是全局证明。

以+X目标的同一物理轨迹为例：

| 轨迹 | 旧未折扣回报 | 旧折扣回报 | 新未折扣回报 | 新折扣回报 |
|---|---|---|---|---|
| 18步正确脉冲成功 | 8.172627 | 6.923067 | 13.317017 | 11.435098 |
| 200步静止超时 | 1.264241 | 0.547429 | -12.000000 | -5.196122 |
| 41步反向触发workspace_limit | -6.498605 | -4.360464 | -33.976466 | -24.736848 |

这些实际典型轨迹没有鼓励主动触发安全失败，但固定20惩罚**不保证所有轨迹的全局安全偏好**。
一个明确的解析反例是：从抽象15mm距离保持200步，未折扣回报-32；同距离立即安全失败为-20.16，
后者更高。这个抽象状态不是本任务5mm Home reset或已执行的物理轨迹；在相同gamma=.99下，
该静止超时的折扣回报约-13.86，仍高于-20.16。报告保留这个边界，不修改参数掩盖它。

### 往返与终止奖励漏洞检查

5→4.75→5mm往返，新默认奖励依次为+0.1925、-0.3100，未折扣合计-0.1175，
gamma=.99折扣合计-0.1144；额外测试三段往返亦为负，没有在这些默认参数例子中观察到刷分收益。
若只保留纯进度项（诊断用w_distance=0、step_cost=0，不用于Smoke），两步为+0.25/-0.25，
折扣合计却为+0.0025。这确认简单距离差不是折扣一致的势函数差，不能宣称任意往返都严格抵消。
本次不因此叠加额外惩罚。成功后原环境立即结束，再step抛ResetNeeded，不能重复领取奖金。

### 四张实际诊断图

图来自同一份冻结的真实MuJoCo轨迹，没有正式训练成功率曲线：

- [同轨迹旧新逐步奖励](../../outputs/rl/reach_reward_validation/20261114/comparison/reward_trace_comparison.png)
- [新奖励signed分项](../../outputs/rl/reach_reward_validation/20261114/comparison/new_reward_components.png)
- [实际末端距离随时间变化](../../outputs/rl/reach_reward_validation/20261114/comparison/distance_trace.png)
- [两目标下正确/错误/零动作奖励](../../outputs/rl/reach_reward_validation/20261114/comparison/direction_reward_comparison.png)

## 自动化测试及唯一一次PPO Smoke

首轮新奖励单元测试命令：

```bash
/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python \
  -m unittest discover -s tests -p test_reach_progress_reward.py -k ProgressRewardTests -v
```

首轮20项通过，9.371s。之后新增配置/独立CLI边界测试，连同原回归执行下列完整命令：

```bash
env \
  REALMAN_PPO_TEST_ARTIFACTS=outputs/rl/ppo_reach_validation/smoke_seed2026 \
  REALMAN_AXIS_PPO_SMOKE_ARTIFACTS=outputs/rl/ppo_goal_conditioned_validation/20261113_preparation/axis_pair_smoke_seed2026 \
  REALMAN_RANDOM_PPO_SMOKE_ARTIFACTS=outputs/rl/ppo_random_reach_validation/20261113_preparation/random_smoke_seed2026 \
  REALMAN_FETCH_PPO_TEST_ARTIFACTS=outputs/rl/reach_fetch_observation_validation/20261113/smoke_seed2026 \
  REALMAN_PROGRESS_PPO_TEST_ARTIFACTS=outputs/rl/reach_reward_validation/20261114/smoke_seed2026 \
  /home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python \
  -m unittest discover -s tests -v
```

实际输出保存在 `outputs/rl/reach_reward_validation/20261114/full_tests.log`。
结果：**339/339通过，210.967s；0失败、0错误、0跳过**。其中原314项、新增25项；
旧PPO四组Smoke均复用已有产物，新PPO只进行下面一次512步。
原BC/SAC测试的微型训练在临时目录运行，不进入新PPO实验。

覆盖A–H：方向排序、静止无正进度、当前goal重算成功、超时无误判、安全失败与拒绝、
典型安全偏好/反例、往返及折扣反例、成功后禁止重复step、三种旧奖励冻结样本与黄金轨迹。
另覆盖多维batch、参数校验、旧模型预测、物理一致性、两种观测模式Gym/SB3 checker、
逐步分项求和、模型/config错配拒绝与独立进程评估。
旧AST回归仅忽略新增progress分派/审计分支及必要的模式判断扩展，保留原期望哈希；未改黄金fixture。

### 512步链路结果

使用axis_pair_reach、fetch_like_v1、reach_progress_v1、seed2026、原PPO_CONFIG，
完成512步/1个Rollout/10个epoch/80次Adam更新，Actor和Critic均真实更新，Rollout数组及Loss均有限。
value_loss=0.279526、policy_gradient_loss=-0.00881443、approx_kl=0.00785175。
训练actions.jsonl的512条记录均有组件，signed分项之和等于环境原始返回值。
训练与内置评估共1312个env.step、32800个真实物理步，逐物理步检查没有qpos捷径；阻断专家动作和BC/SAC/SDK导入。

initial、step_000000、step_000512、final均保存完整参数和模型metadata；final与step_000512确定性预测相同。
独立进程按保存的新奖励评估两个候选：成功0/2、均200步超时、安全失败0，均值最终距离6.787405mm。
独立评估参数不变，逐步组件和Episode回报一致。这里只验证链路，不能解释为训练收敛或新奖励优于旧奖励。

新增代码编译检查通过。完整物理/观测源码、原PPO_CONFIG、TASKS与ENV_CONFIG保持原语义；
18对轨迹的660个物理转移一致；44个物理资产/控制/观测文件及680个历史产物SHA256保持不变。
原始证据、测试结果、同轨迹comparison.json、前置诊断冻结报告和完整8参数配置均位于
`outputs/rl/reach_reward_validation/20261114/`。本次没有执行16384步或131072步正式训练。

## 用户后续手动实验命令

以下命令仅供用户执行；三个任务与A/B运行须分别使用新目录。

### 实验A：先验证±X5mm方向学习

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal

python -I scripts/train/train_ppo_reach.py \
  --task axis_pair_reach --observation-mode fetch_like_v1 --reward-mode reach_progress_v1 \
  --action-scale 0.00025 --timesteps 16384 --seed 2026 \
  --eval-every 2048 --checkpoint-every 2048 --eval-episodes 2 --eval-seed-start 97000 \
  --run-dir outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot

python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/final/model.zip \
  --observation-mode fetch_like_v1 --episodes 2 --seed-start 98000 \
  --json outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/distinct_goals.json
```

这里始终是两个不同候选，不把重复seed当随机泛化；分别查看±X成功率、距离、首动作、路径和保护终止。
如果一侧长期失败或重新出现共同落点，不自动进入随机任务或追加更长预算。

### 实验B：双方向表现合理后再做随机目标

2026-11-14状态：实验A最终两方向均超时，暂未满足进入实验B的条件；以下仍为条件性手动命令，未执行。

```bash
python -I scripts/train/train_ppo_reach.py \
  --task random_reach --observation-mode fetch_like_v1 --reward-mode reach_progress_v1 \
  --action-scale 0.00025 --timesteps 16384 --seed 2026 \
  --eval-every 2048 --checkpoint-every 2048 --eval-episodes 20 --eval-seed-start 97000 \
  --run-dir outputs/rl/ppo_reward/random_progress_seed2026_pilot

python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reward/random_progress_seed2026_pilot/final/model.zip \
  --observation-mode fetch_like_v1 --episodes 100 --seed-start 98000 \
  --json outputs/rl/ppo_reward/random_progress_seed2026_pilot/heldout_98000_98099.json
```

### 实验C：等预算旧新奖励对照

对每个任务、每个seed分别以reach_shaped_v1和reach_progress_v1从头训练，
至少比较2026/2027/2028三个seed。两组均为fetch_like_v1、16384步、0.00025m动作、
原PPO配置、相同开发目标和评估频率，使用不同输出目录；冻结模型后用同一组目标配对评估。
已有131072步旧奖励结果仅作历史背景，不能与新奖励16384步作等预算收敛比较。

以成功率、末端距离/初始到最终改善、首动作方向、±X及随机XYZ方向组、成功步数、
超时/安全失败、Box裁剪和Actor/Critic指标为主；观察三维目标与实际终点是否再次集中。
奖励公式不同，不能直接以更高原始Return认定更好。只在多seed出现可复现改善后再讨论增加预算。
当前CLI仍不提供严格跨进程完整续训，这些命令均从头训练。

## 设计局限与交付文件

更大的单步奖励不保证更好的PPO学习；直接进度差可能引入折扣相关的偏好。
固定失败惩罚仅验证了典型轨迹，没有全局安全排序保证。初始执行滞后时，残余距离成本
可能大于微小进度，正确动作的总奖励仍可为负。新奖励不能补足全部关节/控制参考状态，
也不能解决所有观测、控制或探索问题。正式受控训练前不能证明它优于旧奖励。

| 文件 | 修改 |
|---|---|
| src/realman_rl_simtoreal/tasks/rewards.py | 集中参数、分项键、配置/模型奖励一致性检查 |
| src/realman_rl_simtoreal/envs/reach_env.py | 新奖励分派、纯组件计算、真实previous与组件info |
| src/realman_rl_simtoreal/tasks/reach.py | 可选新奖励参数配置，目标定义不变 |
| src/realman_rl_simtoreal/rl/ppo.py | 原始奖励分项审计、模型metadata检查，无优化改动 |
| src/realman_rl_simtoreal/rl/common/evaluation.py | 逐步组件、累计分项与日志 |
| scripts/train/train_ppo_reach.py | CLI模式与完整参数/metadata保存 |
| scripts/diagnostics/diagnose_reach_rewards.py | 前置真实动作诊断、解析交叉核对、同轨迹重算及四图 |
| tests/test_reach_progress_reward.py | 25项新契约与单次共享PPO Smoke |
| tests/fixtures/reach_reward_legacy_values.json | 修改前18个实际旧奖励单步冻结样本 |
| tests/test_ppo_reach.py | 旧方法AST核对仅排除新增奖励分派/审计 |
| README.md | 新奖励状态和用户手动入口 |
| docs/architecture.md | 奖励组件与保存/评估协议 |
| docs/FILE_MAP.md | 新增文件与当前数量 |
| docs/experiments/README.md | 诊断/验证索引 |
| docs/validation/README.md | 新报告索引 |
| docs/validation/REACH_REWARD_OPTIMIZATION_VALIDATION.md | 本报告 |

`simulation/`、模型资产、观测变换与独立评估CLI未修改；后者已有保存配置加载流程，可直接评估新模型。
本次不提交或推送，由用户检查工作区并自行管理全部版本操作。

## 用户手动16384步奖励Pilot复核（2026-11-14）

### 范围与数据来源

用户已完成 `outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/` 的训练与独立评估。
本节只读已有产物及冻结模型，不重新训练，不reset/step仿真，不执行任何Git命令，不修改算法/控制代码。
复核脚本、结构化结果、输入哈希和四张图存于
`outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/`。

- 任务：axis_pair_reach，Home的±X5mm，两个目标相距10mm。
- 观测：fetch_like_v1；普通字段6维、两个Goal各3维，默认CombinedExtractor，Actor/Critic均接收12维。
- 奖励：reach_progress_v1；完整8参数与本报告实验初值一致，训练/检查点/独立评估配置一致。
- 动作/控制：action_scale=0.00025m，control_period=0.05s，原200步时限和1mm成功阈值。
- PPO：seed2026，16384个实际训练步骤，32个完整Rollout、320个epoch、2560次Adam更新，Actor/Critic均更新；全部PPO_CONFIG保持原值。
- 开发评估：0、2048、4096、6144、8192、10240、12288、14336、16384步，每次枚举两个目标，seed97000/97001。
- 独立进程评估：final/model.zip，seed98000/98001，仍是同一对固定候选。不同seed仅用于复现标识，**不代表新目标位置泛化**。
- 训练约271.35s，含开发评估总计约327.68s；状态completed，无报错。

原始证据：
[配置](../../outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/config.json)、
[训练汇总](../../outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/summary.json)、
[九次开发评估](../../outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/evaluations.json)、
[独立评估完整轨迹](../../outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot/distinct_goals.json)、
[本次结构化分析](../../outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/analysis.json)。

### 结论：未学会稳定的目标方向条件控制

最终确定性评估**0/2成功、2/2超时、0次安全失败、0次动作拒绝**。
平均末端误差4.933122mm，相对两目标各5mm的初始误差仅下降1.34%；平均值掩盖了方向间的抵消。

| 最终模型指标 | +X5mm | -X5mm |
|---|---:|---:|
| 初始距离(mm) | 5.000000 | 5.000000 |
| 最终距离(mm) | 2.531277 | 7.334967 |
| 距离改善：初始减最终(mm) | +2.468723 | -2.334967 |
| 全轨迹最近距离(mm) | 2.414284，第92步 | 4.998995，第1步 |
| 结果 / 步数 | time_limit / 200 | time_limit / 200 |
| 首步X动作 | +0.164402 | +0.131315 |
| 最终实际X相对Home位移(mm) | +2.486951 | +2.325465 |
| 最后20步真实路径总长(mm) | 0.000000144 | 0.001117821 |
| Episode Return | -5.224282 | -18.415711 |

两个真实终点只相距**0.176062mm**，约为目标间距的1.76%；再次出现不同目标落到近似共同位置的现象。
两条路径都向+X偏移后基本停滞。+X虽有部分距离改善，始终未进入1mm成功半径；
-X除第一步约1微米的短暂距离改善外，很快远离目标。因此不是“两侧都已接近成功、只差少量Episode时间”。
首步命令方向与单步实际位移需区分：两目标首步命令X都为正，实际X均有微小负漂移，
后续轨迹才呈稳定的正X移动；没有用命令位移代替真实物理轨迹。

### 检查点：短暂单侧成功随后消失

| 训练步数 | +X最终距离(mm) | -X最终距离(mm) | 成功数/2 | 首步X：+X目标 / -X目标 |
|---:|---:|---:|---:|---:|
| 0 | 5.040788 | 4.959167 | 0 | -0.000959 / -0.000964 |
| 2048 | 8.643052 | 4.274377 | 0 | -0.030809 / -0.030846 |
| 4096 | 8.471664 | 3.547984 | 0 | -0.226784 / -0.226900 |
| 6144 | 10.496789 | 0.967931 | 1 | -0.458981 / -0.460955 |
| 8192 | 7.157669 | 3.035458 | 0 | -0.164523 / -0.168448 |
| 10240 | 4.828474 | 5.122218 | 0 | +0.066075 / +0.055997 |
| 12288 | 2.940478 | 7.033131 | 0 | +0.296733 / +0.281050 |
| 14336 | 5.084891 | 4.844728 | 0 | -0.063453 / -0.073314 |
| 16384 | 2.531277 | 7.334967 | 0 | +0.164402 / +0.131315 |

6144步时-X在65步、0.967931mm处成功，但同一模型+X误差达10.496789mm，且两侧首动作都指向-X。
8192步起连续五次评估再无成功。九个检查点的首步X动作均在两个目标间保持同号；
表现更接近**共同动作偏置随训练摆动**，尚未形成按目标正负方向切换动作的策略。
6144步模型可保留作单侧行为诊断，不能作为“两方向已学会”的模型，也不支持把其50%称为稳定成功率。

![检查点成功、距离、首动作与终点分离](../../outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/checkpoint_learning.png)

![最终距离轨迹与三维目标/实际轨迹](../../outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/final_trajectories.png)

### 训练采样：少量早期成功，没有后续保持

83个完整Episode中成功3次（3.61%），76次超时（91.57%），4次安全失败（4.82%）；
另有186步未结束Episode。安全失败为2次joint_limit、2次workspace_limit，保护按原机制生效；
后续没有把它们归类为普通超时或忽略惩罚。

| 训练目标 | 完整Episode | 成功 | 超时 | 安全失败 | 平均最终距离(mm) |
|---|---:|---:|---:|---:|---:|
| +X | 43 | 0 | 40 | 3 | 7.016563 |
| -X | 40 | 3 | 36 | 1 | 4.555749 |

三次成功全部来自-X，分别在累计4901、5012、5798步结束，耗时51、111、186步；
从5798步后至16384步训练结束，没有新的成功。目标数量43/40没有严重失衡，不能把该现象解释为没有采到-X或+X目标。
训练是随机高斯采样，独立评估是确定性均值动作，三次训练成功不代表最终策略能复现成功。

训练动作Box向量裁剪67.19%，分量裁剪31.05%；高斯std均值最终0.942728（初始1.0）。
最终确定性评估Box裁剪为0，因此最终失败不能归因于评估动作被裁剪。
这也不能单独证明高裁剪是失败原因；本轮没有改探索分布、std、动作尺度或PPO参数。

### 奖励与优化审计

对全部16384条训练记录，从真实前后末端位置、当前目标与真实标记重新计算六个奖励项，
与已保存分项的最大绝对差为0；所有分项之和等于真实环境奖励。独立评估400步也逐项求和、
按真实位置核对距离；新奖励确实生效，没有观察到误用旧公式、命令进度或错误目标的现象。

| 最终评估Episode奖励分项累计 | +X | -X |
|---|---:|---:|
| progress_reward | +2.468723 | -2.334967 |
| distance_penalty | -5.693005 | -14.080743 |
| step_penalty | -2.000000 | -2.000000 |
| success_bonus / safety_penalty / rejection_penalty | 0 / 0 / 0 | 0 / 0 / 0 |
| total_reward | -5.224282 | -18.415711 |

奖励正确区分接近与远离：+X进度为正、-X为负；停留后仍有剩余距离与步成本。
实际数据不足以说明“把新奖励再放大”就能解决问题，也不能因一次运行失败就认定公式错误。
训练所有步骤的平均绝对进度项约0.07503、平均距离惩罚约-0.05690，
不能用正负进度抵消后的累计值误判距离项在每个步骤都压过进度项。

训练指标均有限；最终value_loss=0.318960、approx_kl=0.013713、
PPO似然比clip_fraction=0.147266、explained_variance=0.128735。
EV范围[-0.603780,0.638837]，价值估计波动；loss下降和参数更新并不证明目标控制已收敛。
PPO的clip_fraction与前述动作Box裁剪是两种不同统计。

![最终评估奖励分项与逐步奖励](../../outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/reward_audit.png)

![原生PPO优化指标与探索动作统计](../../outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/optimization.png)

### 冻结网络目标敏感性

加载九个检查点，复用保存的Home实际位置与零初速度，只切换原已验证的±X5mm目标；
不reset/step仿真，也不更新权重。重现全部18个已记录首步动作（容差1e-7）；
网络实际特征仅第3项desired_goal_x改变，差为0.01m，其余11项一致。
默认CombinedExtractor的features_dim为12，Actor和Critic首层in_features均为12；
两个Goal正常进入网络，未添加相对位置特征或自定义提取器。

最终模型两侧首步X均值为0.147858，目标对比造成约±0.016544的变化：
**能读取目标，但目标的影响未超过共同方向偏置**。局部只读梯度探针显示，在+X Home输入处：

- desired_goal_x每增加1mm，a_x的一阶局部变化约+0.003307。
- 普通observation中的实际x通过原0.01m尺度变化1mm，a_x的一阶局部变化约-0.097051。

这只量化该模型、该输入附近的敏感性，提示可继续检查目标条件影响与状态反馈强弱；
不能据此断定归一化是根因，也不能外推全空间，未改变Goal世界坐标或任何预处理。

final与step_016384压缩文件字节不同，但政策参数哈希、更新计数及确定性预测完全相同；
独立评估的summary/per_goal与最后一次开发评估逐值相同，未发现加载了其他检查点的情况。

### 复核命令、保全与下一步

实际执行的只读分析命令：

```bash
/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python -I \
  outputs/rl/reach_reward_validation/20261114/axis_pair_pilot_review/analyze.py
```

脚本拒绝覆盖既有analysis.json；新复核应使用新目录并保留此版本。复核核对16384条训练、
400条独立评估、18次冻结模型前向；最后Rollout所有数组有限。运行配置中24项源码/资产哈希
与当前文件一致；原实验68个文件及src/scripts/tests/assets共125个文件复核前后SHA-256一致。
四张PNG均已打开检查。没有运行新增训练、物理step或Git操作；代码未改，本轮未重跑原339项开发测试。

本轮**两方向课程未通过，暂不进入random_reach，也不据此直接启动131072步**。
16384步可能仍不足，但本轮没有稳定成功趋势，当前证据不支持把失败只归因于训练时间。
后续优先使用已有检查点复核目标条件/优势信号，并在同一axis_pair任务、fetch_like_v1、
相同预算、相同seed与评估目标下对比reach_shaped_v1/reach_progress_v1；沿用上文A/B约束。
待多seed对照出现可复现的两侧方向正确与成功改善后，再讨论扩预算或随机目标。

尚无同任务、同12维观测、同预算的新旧奖励配对训练；旧33维轴向实验及旧12维随机实验
都存在其他因素差异，不能直接用来宣称新奖励优于或劣于旧奖励。本次只得出当前配置单seed
Pilot未解决方向偏置的结论，未证明任意更长训练必然失败。
