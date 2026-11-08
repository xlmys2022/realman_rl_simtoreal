# 04-F1：官方PPO Reach训练系统验证

日期：2026-11-08。开发及验证均使用`realman_rl`，没有启动16384步Pilot或其他正式训练。只进行**一次512步完整PPO Rollout及其10个epoch更新**；14项新增测试共享该次结果，其余原252项回归测试原样运行。不使用BC、SAC权重、专家数据、专家纠正或自定义PPO损失。

上述及下文开发验收部分描述交付时状态。用户随后已手动完成16384步固定目标Pilot；结果及只读复核见文末“用户手动Pilot结果记录”。该次复核没有启动新训练。

## 文件与职责

- `src/realman_rl_simtoreal/experiments/ppo_reach.py`：官方PPO构造、固定任务配置、原始/实际动作裁剪统计、只读Rollout审计、参数指纹、完整更新指标、checkpoint及独立评估/加载。
- `scripts/train_ppo_reach.py`：CLI、完整Rollout调度、初始/阶段/最终模型、配置/依赖快照、独立评估及SB3 CSV/JSON/TensorBoard日志。
- `scripts/evaluate_ppo_reach.py`：加载已标识为PPO的checkpoint及保存的预处理/环境，独立确定性推理。拒绝误用SAC checkpoint。
- `scripts/plot_ppo_training.py`：只读已保存JSON，导出11张PNG和`plot_data.json`；缺少测量数据时跳过对应图，不从回报推算成功率。
- `tests/test_ppo_reach.py`：14项接口、真实参数更新、保存加载、裁剪、时序、冻结评估、无专家、物理推进和监控验证。
- `experiments/sac_reach.py`：仅让已有评估协议接受`ppo`名称，复用相同物理环境、预处理和汇总；SAC原分支/默认行为保留。
- README及`docs/architecture.md`：新增命令、职责和调度说明。本报告保存实测结果。

机器人资产、simulation、ReachEnv、IK、vendor、BC及SAC训练入口均未修改。原实验输出未覆盖。基线清单覆盖1020个现有文件；最终只有上述共享`sac_reach.py`在清单中变化，模型资产和历史输出逐文件哈希保持相同。

## 环境与算法配置

Python3.11、MuJoCo3.17.0、Gymnasium1.4.0、NumPy2.4.6、Stable-Baselines3 2.9.2、PyTorch2.14.0+cpu、TensorBoard2.21.0、Matplotlib3.11.2已经安装。不安装/升级任何包，前后完整依赖映射一致，`python -I -m pip check`通过。

已核对安装版`PPO.__init__`、`OnPolicyAlgorithm.collect_rollouts/learn`及`PPO.train`实际源码，归档于`outputs/rl/ppo_reach_validation/`。官方参考：[PPO参数](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、[完整Rollout与learn实现](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html)。以已安装2.9.2实现和锁定依赖为复现依据。

主要参数：

```text
algorithm=PPO, policy=MultiInputPolicy, seed=2026, device=cpu
learning_rate=0.0003, n_steps=512, batch_size=64, n_epochs=10
gamma=0.99, gae_lambda=0.95, clip_range=0.2
ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5
net_arch={pi:[256,256], vf:[256,256]}
native activation=Tanh, orthogonal_init=True
native DiagGaussian, log_std_init=0 (std=1), squash_output=False
native normalize_advantage=True, target_kl=None, use_sde=False
torch_num_threads=1, deterministic_algorithms=True
```

没有照搬SAC的ReLU/tanh压缩输出；采用PPO官方策略默认值。特征仍按`achieved_goal(3), desired_goal(3), observation(27)`展开，共33维float32。固定仿射变换与04-A逐项一致；goal键保持米制，原观测单位与环境计算不变。策略为33→256/Tanh→256/Tanh→3维高斯均值，加三个独立可训练log_std；价值网络为33→256/Tanh→256/Tanh→1。策略75270、价值74753，共150023个参数；不使用双Q或Target Critic。

初始环境配置：

```text
task=fixed_reach, target_mode=fixed (+世界X 5mm)
reward_mode=reach_shaped_v1, reward_sigma_m=0.005, reward_gamma=0.99
action_scale=0.00025 m/轴/环境步
control_period=0.05 s, timestep=0.002 s, 每步25个物理步
max_episode_steps=200, success_threshold=0.001 m
boundary_mode=terminate, gravity_compensation=True
```

仍使用原六轴模型，不加载夹爪场景。累积命令参考、DLS、执行器、增益、限位、±5°/10mm保护和奖励公式保持原样。后续可显式选择`easy_random_reach`或`random_reach`，本次不训练这些任务。

## Rollout、更新与保存语义

每个完整512步Rollout之后，官方PPO执行10个epoch，每epoch有8个64样本minibatch，即80个Adam步。驱动不在半个Rollout调用更新，也不以SAC的分段步数逻辑修改Buffer。当前仅使用常数学习率及clip范围，在同一进程/环境/优化器中按完整Rollout调用learn，保留未结束Episode；没有跨进程继续训练接口。

`total_timesteps`是下界：例如请求128，实际512。默认16384恰为32个Rollout；正常完成时320个epoch、2560个Adam步。`epoch_updates`是SB3的`_n_updates`，不能解释成Rollout或梯度步数。

评估/checkpoint间隔是阈值，越过后在**已完成更新**处执行；同时保存/评估第0步和最终步。Smoke故意设`eval_every=300`、`checkpoint_every=400`，实际阶段仅0、512，没有在300/400步保存伪造的“已完成更新”。中断/严重物理故障保存时标记`partial_or_interrupted`，不把未完成更新计入Rollout完成数。

输出包含：`initial_model.zip`、`initial/`、`checkpoints/step_XXXXXX/`、`final/`、环境/预处理/超参数/版本及源文件/资产校验配置、`actions.jsonl`、`training_episodes.jsonl`、`metrics.jsonl`、`stages.json`、`evaluations.json`、`summary.json`及`sb3/`日志。`final/last_rollout.npz`只是官方最后一次Rollout数组快照，不是Replay Buffer或专家数据。

真实环境奖励在SB3的timeout价值bootstrap之前记录；官方Buffer中的timeout奖励遵循原SB3实现，可包含终止观测价值。自动reset前的真实终止位置通过`terminal_observation`读取。原始高斯动作/log_prob仍按官方方式存入Buffer；实际进入环境的动作是原Box裁剪结果。

## 唯一Smoke实测

产物：`outputs/rl/ppo_reach_validation/smoke_seed2026/`。种子2026，请求128步但实际512步，完成1个Rollout、10个epoch、80次Adam更新。

- 策略和价值参数指纹均变化；优势、回报、log_prob、网络参数、优化器状态及损失均为有限值。
- 实际训练及审计墙钟7.820秒，约65.48交互步/秒；两个独立评估约6.003秒，总约13.847秒。含逐步日志及测试qpos断言，不能直接当作正式训练速度。
- 训练2个完成Episode，均200步超时，另有112步未完成Episode；均无安全失败/动作拒绝。平均原始回报1.31246，不代表成功率改善。
- 训练动作朝实际目标比例47.07%，真实末端累计路径74.957mm、命令路径153.021mm，路径长度比0.48985；随机高频动作存在控制响应滞后，不能把命令路径当实际运动。
- 带物理状态审计的子进程共执行912个标准env.step（训练512+两次200步评估）、22800次mj_step。逐物理步前后核验live qpos，只允许mj_step改变正常运动状态；reset初始化不在此限制内。
- 阻断BC/初始化/厂家模块导入，专家动作与SAC模型构造函数均替换为会报错的哨兵，训练仍完成。未加载历史权重或数据。

独立确定性评估（每阶段1个固定目标）：

- 第0步：0/1成功，200步超时，最终5.094956mm；无安全失败、拒绝或动作裁剪。
- 第512步：0/1成功，200步超时，最终5.474828mm；无安全失败、拒绝或动作裁剪。
- 从`/tmp`使用`python -I`及绝对路径，独立加载`step_000512/model.zip`并改用seed88000，结果仍为5.474828mm，预测与保存前逐元素相同；评估不改网络、优化计数或Torch RNG。

固定目标换seed仍是同一目标。以上仅验证训练和推理链路，不是随机目标泛化实验，不支持“PPO已经学会Reach”结论。

更新后指标：

```text
policy_gradient_loss = -0.02859145
value_loss           =  0.00357663
entropy_loss         = -4.25687836
approx_kl            =  0.01573157
clip_fraction        =  0.21718750
explained_variance   = -0.86603701
policy std (3轴)     = [0.98367077, 1.01274657, 1.00014400]
```

只有一次更新，不能据此判断长期价值拟合或收敛。原生explained_variance比较Rollout收集时的价值预测与GAE returns，不是更新后价值网络的全新误差；approx_kl沿用SB3最后一个epoch的汇总。负微分熵损失不等于策略已经确定性。`ent_coef=0`也不意味着随机采样被关闭。

### 动作裁剪与PPO Clip的区别

512个采样动作中342个向量含越界分量（66.796875%）；1536个分量中466个被裁剪（30.338542%）。原始绝对最大值3.360730，实际输入绝对最大值1。它是原生std≈1高斯分布与[-1,1]动作Box造成的裁剪，保留原实现并记录，不额外缩放或改写分布。每轴实际物理命令仍最多0.25mm。

PPO的`clip_fraction=21.71875%`表示似然比超出目标clip区间的样本比例，与上述Box裁剪含义不同。即使确定性评估没有Box裁剪，也不代表随机训练探索没有裁剪。Pilot应同时检查这两项及std、熵、KL，不能混用。

## TensorBoard与PNG

`sb3/progress.csv`、`progress.json`和事件文件包含以下独立命名指标：

- `evaluation/`：独立成功率、最终距离mm、Episode长度/回报、成功步数（仅存在成功样本时）、超时、安全失败、拒绝比例及方向/位移统计。
- `training/`：已完成Episode成功率、平均回报/长度、各终止原因累计次数、拒绝/安全失败比例及方向/运动路径；不与独立评估混称。
- `train/`：官方policy_gradient_loss、value_loss、entropy_loss、approx_kl、clip_fraction、explained_variance等。
- `policy/action_std_*`：三轴及平均标准差；`monitor/`：实际步数、完成Rollout数、epoch及Adam更新数。
- `actions/box_clip_rollout/`与`actions/box_clip_total/`：每Rollout/累计向量、分量裁剪率及原始/实际动作最大值。

TensorBoard `--inspect`确认0、512步事件及正确指标；未打开浏览器验证仪表盘。TensorFlow未安装但不影响这些PyTorch标量日志，无需为此安装TensorFlow。

已在`smoke_seed2026/plots/`导出并检查11张PNG：独立成功率、距离、Episode长度，policy loss、value loss、KL、PPO clip_fraction、explained variance、entropy loss、std及Box裁剪。原始序列在`plot_data.json`；只有一次更新的损失图只显示一个点，未伪造训练曲线。缺失评估的反例不会产生成功率图。

## 自动化与完整性

修改前252项测试全部通过（135.514秒），日志`outputs/rl/ppo_reach_validation/baseline_tests.log`。修改后**266/266通过**（154.361秒），原252个测试及断言未修改，新增14个测试不以成功率达标作为断言。完整日志`outputs/rl/ppo_reach_validation/all_tests.log`。

测试覆盖Dict/33维特征顺序与原预处理、原生Tanh网络、动作合法性及实际裁剪、完整Rollout和实际计数、策略/价值更新、有限优势/参数/指标、保存加载一致、独立评估不更新网络、固定目标语义、禁止BC/专家/SAC权重、禁止live qpos捷径、物理配置/资产、TensorBoard实测数据、PNG与缺评估反例、非对齐调度、错误算法/非空目录拒绝。

`integrity.json`保存1020文件清单对比及参数量；`dependencies_before.json`和`dependencies_after.json`证明未改变依赖。原控制、厂家、Reach、BC/SAC等测试均通过。未修改已有失败记录、精度阈值或安全限制。Git未提交、未推送。

## 用户手动执行的首次Pilot

以下命令使用现有realman_rl，输出目录须为新的空目录。不需要重新安装依赖。本次仅验证了CLI参数、单Rollout及后续独立评估/绘图，**没有执行此Pilot**。

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I scripts/train_ppo_reach.py --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/fixed_seed2026_pilot --eval-every 2048 --eval-episodes 1 --eval-seed-start 87000 --checkpoint-every 2048 --tensorboard
```

另开终端查看：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach --host 127.0.0.1 --port 6006
```

训练完成后检查汇总、独立加载checkpoint并导出PNG：

```bash
python -I -m json.tool outputs/rl/ppo_reach/fixed_seed2026_pilot/summary.json
python -I scripts/evaluate_ppo_reach.py --model outputs/rl/ppo_reach/fixed_seed2026_pilot/checkpoints/step_016384/model.zip --episodes 1 --seed-start 88000 --json outputs/rl/ppo_reach/fixed_seed2026_pilot/loaded_evaluation.json
python -I scripts/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/fixed_seed2026_pilot
```

若提前安全停止，按实际存在的checkpoint路径评估，并先查看summary.status/error；不要假定一定存在016384。也可加载`final/model.zip`，其state.json明确标记完整或部分阶段。需要再次评估/绘图时用新的`--json`/`--out`路径，避免覆盖原记录。评估、绘图均支持从其他目录通过绝对脚本路径及`python -I`运行，无需手改PYTHONPATH。

## 开发交付时尚未验证的内容与下一步边界

尚未验证16384步是否学会固定目标、长期std/裁剪/价值拟合变化、多seed稳定性或随机目标泛化。单Rollout速度不等于长训练吞吐，当前不支持跨进程完整续训。当前PPO与历史SAC的网络激活、算法和选定动作尺度不同，不能把历史结果当作只改变算法的严格对照。

Pilot应先确认独立成功率及距离持续改善，再看是否过早静止、随机动作过多裁剪、KL/熵/std变化和价值拟合。若失败，保留全部数据并先分析曲线；不自动追加100000步或进入随机目标，不修改奖励/物理约束制造成功。当前源码editable安装仍依赖仓库，机器人/TCP/动力学尚未真机标定，全部结果仅适用于此MuJoCo局部Reach任务。

## 用户手动Pilot结果记录（2026-11-08）

用户已完成`outputs/rl/ppo_reach/fixed_seed2026_pilot/`。本次读取现有模型和日志、独立推理、复放两条保护失败轨迹、导出PNG并更新文档，**新增训练步数为0**。未更改任何训练代码、环境、奖励、IK、控制器或模型，未覆盖原产物，未启动下一阶段。

### 配置、步数与完整性

实际配置与上述首次Pilot命令一致：`fixed_reach / reach_shaped_v1 / 0.00025m / terminate`，seed2026、CPU、33维固定预处理、256×256/Tanh，`init_mode=scratch`、`pure_rl=True`，不使用BC或专家数据。

- `status=completed`，请求/实际均16384步，32个完整Rollout、320个epoch、2560次Adam更新。
- 训练和更新墙钟274.138秒，59.765步/秒；独立阶段评估7.576秒，总计281.898秒，约4分42秒。
- 逐步动作日志有16384行；456个完成Episode共16374步，另有10步未完成Episode。33个指标阶段为0及每512步，9次评估为0及每2048步，计数一致。
- 最终模型与16384步checkpoint的参数指纹、实际步数、epoch和Adam计数一致。网络、优化器状态、最后Rollout优势/returns及全部数值日志均有限。
- 保存的资产/实验源文件校验值与当前文件一致；复核前后原训练目录68个文件的SHA-256完全相同。全部新复核数据放在`outputs/rl/ppo_reach_validation/pilot_review_20261108/`。

### 固定目标的独立确定性评估

每次阶段评估均为seed87000、**同一个+世界X 5mm目标**，各1个Episode。结果按训练步列出：

```text
训练步数   成功/集数   Episode步数   最终误差(mm)
       0      0/1          200          5.094956  (超时)
    2048      1/1           45          0.988010
    4096      1/1           59          0.978546
    6144      1/1           34          0.957573
    8192      1/1           20          0.990181
   10240      1/1           18          0.961236
   12288      1/1           18          0.972923
   14336      1/1           18          0.961867
   16384      1/1           18          0.960968
```

从第一个2048步评估checkpoint起，后续8次评估均成功，无安全失败或动作拒绝。完成时间最终为18×50ms=0.9秒物理仿真时间。并非每次都更快，4096步时曾为59步；不能把这些稀疏评估描述成严格单调改善。

随后独立进程加载`final/model.zip`，用seed88000复验，仍为18步、0.960968mm、成功、无安全失败，逐步确定性动作及汇总与原16384步评估完全一致。评估核验网络指纹及优化计数不变，结果在`pilot_review_20261108/loaded_evaluation.json`。

误差保持约0.96–0.99mm与1mm成功阈值下立即终止一致，不能据此要求曲线继续降至零，也没有验证到达后持续保持。换seed的固定目标并非新位置，以上不是100个随机目标测试或随机目标泛化成功率。

### 随机训练采样与保护终止

456个已完成训练Episode中：成功429个（94.0789%），超时17个，`workspace_limit`8个，`joint_limit`2个；IK失败、姿态失败、碰撞、数值异常、速度及驱动力保护均为0。动作拒绝为0，因使用terminate模式而不是reject_and_hold。平均原始回报7.627153、平均Episode长度35.9079，成功Episode平均26.8065步；这些是整个学习过程中随机采样的数据，不等同于最终确定性策略成功率。

最近100个已完成训练Episode均成功，平均19.1步、最终误差0.871470mm；仍是相同固定目标重复，不是100个不同目标。

10次被环境标记为安全失败的终止都发生在1529–9956步：8次10mm候选命令范围保护，2次`joint_limit`分别在1529和3517步。全部在该次动作执行前停止（`physics_steps=0`），没有把非法目标送入执行器。对两条joint_limit Episode按原日志动作通过标准env.step独立复放，所有实际末端位置与日志匹配，最后错误均为`Demo targets must stay within 5 degrees of home`；实际关节处于模型限位内，IK候选目标超过home ±5°。这属于原任务级关节命令保护，不能描述成真实关节越限或物理碰撞。详细信息在`joint_limit_replay.json`。

训练动作朝实际目标比例77.52%、朝命令目标比例77.39%，使实际距离改善的比例80.41%；相邻反向比例33.09%。实际累计路径3.164064m，命令累计路径5.260735m，路径长度比0.60145。最终确定性评估18步全部朝目标、距离逐步改善、无相邻反向，实际路径4.415754mm、命令路径4.933736mm，比值0.89501。训练随机动作与确定性动作的控制响应不同，应分别分析。

### 价值拟合、策略标准差和裁剪

全部32次更新的指标均为有限值。最终值：

```text
policy_gradient_loss = -0.00705351
value_loss           =  0.00065523
entropy_loss         = -3.86990708
approx_kl            =  0.01347215
clip_fraction        =  0.12714844
explained_variance   =  0.98893082
policy std (3轴)     = [0.72846413, 0.96044731, 0.96460646]
policy std mean      =  0.88450599
```

KL在所有更新中的范围为0.005398–0.022966；value_loss曾升至3.511952后降至最终0.000655，explained_variance从早期负值改善到0.988931。支持当前固定任务的价值拟合改善，未发现非有限数值或持续发散，但explained_variance仍是原生Rollout统计，不是模型在新目标上的价值准确率。

动作Box裁剪仍明显：训练累计12938/16384个向量（78.97%）、19626/49152个分量（39.93%）被裁剪；最后512步为492个向量（96.09%）、864个分量（56.25%）。最终确定性评估18个动作向量均含裁剪，54个分量中19个被裁剪（35.19%），原始动作最大绝对值2.370535，实际最大值1。不能因为std下降便认定动作裁剪减少；动作均值也会越出Box。它与PPO目标`clip_fraction=12.71%`不是同一指标。

该策略在固定方向任务中经Box裁剪形成较强的饱和命令，当前在原保护范围内完成了目标；这既不代表驱动力超限，也不能证明已学会根据三维目标改变方向。需要在后续单独研究饱和及随机目标表现，本次未更改分布、动作尺度、熵系数或奖励。

### 日志与本次验证范围

TensorBoard事件、CSV和JSON对应关系已逐项核验：33行阶段指标（含第0步）、9次独立评估及全部policy/value/KL/clip/EV指标一致。已在`pilot_review_20261108/plots/`导出11张PNG及`plot_data.json`，检查了Episode长度、价值拟合及动作裁剪图；没有覆盖原日志或伪造缺失数据。`audit.json`保存机器可读汇总、失败列表、指标范围和完整性核验，前后哈希在`training_files_before.json`及`training_files_after.json`。

本次只读复核和文档更新没有重新运行完整266项测试；266/266是前文开发验收的回归结果。实际新增验证是模型离线加载推理、两次保护轨迹复放、数值/配置/计数/日志/资产及哈希审计、TensorBoard读取和PNG导出，全部通过，没有执行PPO更新。

结论：**纯随机初始化PPO已在本轮单seed、单固定起点/目标条件下学会到达任务**。尚不能宣称随机Reach、不同初始姿态、多seed稳定性、持续保持或真机能力已验证。与此前SAC实验的预算、网络及动作尺度不同，不是严格只改变算法的对照。本次保留所有产物及原失败记录，停止于日志整理与复核，不自动进入easy_random_reach或更长训练。
