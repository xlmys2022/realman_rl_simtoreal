# 04-F2：PPO三维随机Reach训练准备与泛化验证

日期：2026-11-13。当前仅完成开发、采样验证、测试和**一次完整512步随机目标PPO更新**。没有执行16384步或更长正式训练，没有评估90000–90099最终测试目标，没有使用BC、专家动作/数据或SAC权重。以下命令由用户随后手动执行；当前不能报告正式随机Reach成功率。

以上及下文准备部分记录开发交付时状态。用户随后已手动完成16384步Pilot及100目标测试；结果与“是否训练太短”的分析追加在文末。本次结果复核没有执行新训练。

## 现有能力与最小修改

已经阅读当前`docs/architecture.md`、`docs/FILE_MAP.md`、`tasks/reach.py`、ReachEnv、`rl/common/{env_factory,observation,evaluation,metrics}.py`、`rl/ppo.py`和重构后的训练/评估/绘图入口，核对04-F1历史报告。

**现有代码可以直接训练随机目标**：`--task random_reach`已经设置random及5mm半径；训练环境的VecEnv自动reset会恢复home并重新采样。33维固定预处理、20目标独立评估、完整Rollout调度、checkpoint、TensorBoard、保存加载和PNG均已具备。无需新物理环境、任务配置、训练器、网络或训练入口。

实际缺口是逐方向结果与裁剪诊断，本次只补充：

- `rl/ppo.py`：仅扩展`evaluate_ppo`的只读输出。每个Episode额外保存`raw_actions`、`actual_positions_before_m`及`box_clipping`。已有`actions`仍是实际送入env.step的动作，顺序和语义不变。
- `scripts/evaluate/analyze_reach_results.py`：新增只读分析入口，读取现有评估JSON，不加载策略/模型、不调用IK或仿真。输出六个方向、八象限、两段半径的样本量/成功率/误差/步数，以及动作偏置、裁剪比例和方向变化。
- `tests/test_ppo_random_reach.py`：16项新测试，覆盖200个目标、观测/reset契约、分析边界和一次共享随机PPO Smoke。
- README、FILE_MAP、architecture及验证索引更新相应职责和现行命令。

保留`simulation / envs / tasks / rl / scripts/{train,evaluate,plots}`分层，没有恢复`experiments`代码目录。原266项测试及断言不修改。原始URDF/STL、动力学、IK、PD、重力补偿和全部保护保持原字节。PPO配置、构造及除评估函数外的全部原有Python语法树保持相同，没有修改动作分布、损失、采样、优势或优化器。

## 随机目标采样与观测检查

使用正式环境参数，对83000–83199这200个种子各采样一次。另有少量重复reset用于种子序列复现检查。目标候选由Gymnasium的`np_random`生成：标准正态向量归一化得到方向，半径由`cbrt(U(r_min^3, r_max^3))`采样；`r_min=(success_threshold+target_radius)/2=3mm`，`r_max=5mm`。接受分布是体积球壳经原IK、home ±5°及路径/姿态/碰撞预检条件化后的结果，不承诺各方向精确均匀。默认每次最多64个候选，失败明确报错。

200个接受目标均为不同位置，距home最小3.028260mm、最大4.992066mm、平均4.168357mm；3–4mm为69个，4–5mm为131个。候选共201个，其中1个因关节命令限制拒绝，没有取消筛选或修改实际状态来接受目标。

```text
世界坐标相对home的分量范围 (mm)
X: -4.701442 .. +4.700219    +X 97 / -X 103
Y: -4.565961 .. +4.543007    +Y 85 / -Y 115
Z: -4.911190 .. +4.612845    +Z 87 / -Z 113

互不重叠的XYZ八象限计数
+++:19   ++-:17   +-+:26   +--:35
-++:20   -+-:29   --+:22   ---:32
```

这里的六个符号组允许重叠，每个非轴面目标同时属于三个符号组；八象限总和200。上述是目标分布覆盖，不是策略成功率。

每个接受目标重新用原DLS核验：位移误差≤10µm、姿态误差≤1e-4rad，模型关节限位及home ±5°范围、原`check_motion`全部通过。每次reset后实际qpos/ctrl为home、qvel为零、物理时间为零；独立计算不改变live qpos。固定seed能复现目标及后续无seed reset序列，不同seed/后续reset产生不同目标。

真实step测试确认Episode内目标保持不变，3个零动作推进75个2ms物理步；逐步审计qpos只由mj_step改变。VecEnv短Episode测试确认终止观测仍保留旧目标，自动reset后的新观测使用新目标，并恢复home/时间零。没有在step中重新采样。

观测沿用schema 2：27维observation中的目标相对位置索引15:18按10mm尺度缩放，XYZ三个符号和数值保持；两个goal键仍是米制。SB3按achieved_goal(3)、desired_goal(3)、observation(27)展开为33维float32，未丢失方向信息。

完整逐目标数据和IK残差：`outputs/rl/ppo_random_reach_validation/20261113_preparation/target_sampling.json`。

## 正式Pilot配置与种子划分

```text
task=random_reach, target_mode=random, target_radius=0.005m
reward_mode=reach_shaped_v1, action_scale=0.00025m/轴
control_period=0.05s = 25 * 0.002s
max_episode_steps=200, success_threshold=0.001m
boundary_mode=terminate, gravity_compensation=True
learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10
gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0
vf_coef=0.5, max_grad_norm=0.5
MultiInputPolicy, 独立Actor/Critic各256×256 Tanh, CPU, seed=2026
```

算法版本和依赖沿用现有SB3 2.9.2、PyTorch2.14.0+cpu、MuJoCo3.17.0、Gymnasium1.4.0、NumPy2.4.6。无安装/升级，`pip check`通过。

- 采样验证：83000–83199；其他测试序列84000附近。
- 512步Smoke：训练seed2026，两个独立目标seed86000–86001；加载检查86050，20目标API测试86020–86039。
- 正式开发评估：87000–87019，共20个随机目标，各checkpoint相同，允许用于开发期比较。
- 最终冻结评估：90000–90099，共100个新目标，本次未生成/使用，不用于开发或模型选择。

正式Pilot请求/正常实际为16384步，32个Rollout、320个epoch、2560个Adam步；每4096步完整更新后保存并评估20集，另有第0步评估。目标方向来自环境采样，动作全部来自随机初始化后的官方PPO，不加载固定目标PPO模型，也不使用规则策略纠正。

奖励和1mm成功阈值不变；训练随机采样成功率与独立deterministic=True评估分开记录。没有实现跨进程完整续训；重新启动CLI是独立scratch实验，不能描述成无损恢复。

## 诊断输出及解释边界

分析入口根据`target_position_m - initial_position_m`分类；ReachEnv的初始末端即reset home，因此对应世界XYZ相对home方向，不把绝对世界Z高度当作+Z目标。

- 六个符号组可重叠；八象限互斥，坐标分量精确为零的目标单列`on_coordinate_plane`，不丢弃。
- 半径为`[3,4)mm`与`[4,5]mm`，范围外单列。每类记录样本数、成功数/率、平均误差、Episode长度、成功步数、终止原因；空组率为null，不伪造0%或100%。
- 默认`--min-samples 10`只给小样本标记，不是可调整的成功门槛或置信区间。100目标在8象限中不保证每类达到10，不对少数样本推断泛化。
- 每类记录平均首动作（每Episode一个样本）和全部动作均值/标准差/各轴正负比例/近零比例，帮助检查目标方向变化与持续+X偏置。全部动作指标按步池化，长Episode权重较大。
- 新JSON的原始动作与实际动作逐项核对Box裁剪，计算向量/分量裁剪率、raw→applied夹角均值/P95/最大值（degree），及对当前实际目标误差的投影符号是否翻转。目标误差使用逐步实际位置，不拿固定初始误差代替。
- 历史JSON缺raw或位置轨迹时明确标记不可计算，不从动作最大值猜测裁剪率、不事后编造轨迹。PPO `train/clip_fraction`与动作Box裁剪继续区分。

现有TensorBoard名称原样复用，没有重复增加等价标签：

```text
独立成功率                 evaluation/success_rate
平均最终误差(mm)           evaluation/final_distance_mm
平均Episode/成功步数        evaluation/episode_length_mean, evaluation/success_steps_mean
安全失败/超时              evaluation/safety_failures, evaluation/timeouts
朝实际目标动作/距离改善     evaluation/motion/toward_actual_fraction,
                           evaluation/motion/distance_improving_fraction
策略标准差                 policy/action_std_mean, policy/action_std_0..2
Box裁剪                    actions/box_clip_rollout/{vector_fraction,component_fraction}
                           actions/box_clip_total/{vector_fraction,component_fraction}
训练Episode成功率           training/episode_success_rate
原生PPO优化指标             train/value_loss, train/explained_variance,
                           train/approx_kl, train/clip_fraction等
```

`success_steps_mean`在没有成功Episode时不写入，不伪造零步成功。分方向/象限诊断在只读分析JSON中，不重复写训练标签。

## 短测试与自动化结果

先运行原266项测试，全部通过（148.863秒），固定目标PPO测试显式复用历史Smoke产物，没有为基线新增PPO训练。新增16项后，**282/282通过**（177.263秒）；原测试和精度/安全断言不变。

新测试共享唯一一次512步随机目标PPO Smoke：

- 完成1个Rollout、10个epoch、80次Adam更新，Actor/Critic参数均确实改变；原始随机初始化指纹与同seed的旧随机初始模型一致，与旧训练结果不同。
- 3个完成训练Episode：随机采样1次成功、2次超时，另有40步未结束；不能把1/3偶然成功当成最终策略成功率。无安全失败/拒绝，优势、returns、网络及优化日志均有限。
- 第0步两目标确定性评估0/2、平均4.208082mm；第512步仍0/2、平均3.249891mm，各200步超时。仅86000/86001两个目标，不足以判断三维控制或学习趋势。
- 训练采样向量裁剪343/512=66.99%，分量467/1536=30.40%；确定性两目标评估向量/分量裁剪均0，raw最大0.144954。前者是随机探索，不与后者混称。
- 更新后value_loss=0.521007，explained_variance=-0.098549，approx_kl=0.006580，PPO clip_fraction=0.028906，平均std=0.994036，均有限。一个更新不用于下长期收敛结论。
- 子进程阻断本项目SAC、imitation、SDK导入及专家动作，审计1312次env.step（训练512+两阶段各400评估步）没有live qpos捷径。模型运动由原执行器和mj_step推进。
- 独立加载保存模型，seed86050运行200步超时、2.814015mm；保存加载、推理和冻结评估均正常。20目标接口另用H=1做20次真实交互检查，不把它当H=200正式评估或成功率结果。

所有新产物在`outputs/rl/ppo_random_reach_validation/20261113_preparation/`：基线/完整测试日志、target_sampling.json、random_smoke_seed2026、独立加载JSON、smoke_direction_analysis.json、TensorBoard检查和`smoke_plots/`。

原模型、控制、环境、任务配置及原测试文件的字节校验一致；原固定目标Pilot的68个文件均未改变。依赖映射前后一致，完整性证据在`integrity.json`。正式随机目标模型不存在，因此本次没有执行最终100目标评估。没有提交或推送GitHub。

## 用户手动执行的六项命令

先激活现有环境并进入仓库。以下训练目录必须是新的空目录，不能覆盖已有实验。参数已检查`--help`，训练参数通过非空目录拒绝路径验证，评估/绘图/分析入口从`/tmp`用绝对路径短测通过；未运行下述正式训练/100目标命令。

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal

# 1. 全新随机初始化PPO随机目标Pilot（由用户手动执行）
python -I scripts/train/train_ppo_reach.py \
  --task random_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/random_seed2026_pilot_04f2 \
  --eval-every 4096 --eval-episodes 20 --eval-seed-start 87000 \
  --checkpoint-every 4096 --tensorboard

# 2. 另开同环境终端实时监控
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach/random_seed2026_pilot_04f2/sb3 --host 127.0.0.1 --port 6006

# 3. 训练结果
python -I -m json.tool outputs/rl/ppo_reach/random_seed2026_pilot_04f2/summary.json

# 4. 模型冻结后独立100目标评估；不能用此集合反复选择checkpoint
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/random_seed2026_pilot_04f2/checkpoints/step_016384/model.zip \
  --episodes 100 --seed-start 90000 \
  --json outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final_test_90000_90099.json

# 5. 训练曲线PNG及原始序列
python -I scripts/plots/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/random_seed2026_pilot_04f2

# 6. 六方向、八象限、半径及动作裁剪分析
python -I scripts/evaluate/analyze_reach_results.py \
  --evaluation outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final_test_90000_90099.json \
  --json outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final_test_direction_analysis.json
```

开发期间可将分析入口的`--evaluation`改为某个`checkpoints/step_XXXXXX/evaluation.json`，只分析87000–87019开发目标。提前停止时先检查summary，使用实际存在的checkpoint。再次评估或绘图需新的`--json`/`--out`位置；默认拒绝覆盖。

TensorBoard事件已读出原有优化/评估标签，11张PNG及plot_data.json已导出，图像检查见Smoke产物；没有打开浏览器验证仪表盘。原始数据保留，曲线不由训练回报推断成功率。

## 未验证事项和停止边界

尚未验证16384步随机Reach成功率、各方向/象限真实能力、不同seed稳定性、长期饱和及目标条件策略是否泛化。当前200目标只证明分布和可达性，512步只证明训练路径可运行；都不能代替用户正式Pilot。此前固定目标96%裁剪及成功结果继续保留，不能移用为随机任务成绩。

用户可先查看20目标开发曲线、方向统计、限制终止、价值拟合与裁剪。若有学习趋势，再由用户决定是否进行32768步的独立实验；当前CLI不提供严格跨进程续训。不自动追加步数、调奖励、改分布/网络/IK或扩大工作空间。完成本次准备后停止，等待用户手动训练反馈。

## 用户手动16384步Pilot结果与预算诊断（2026-11-13）

用户已完成`outputs/rl/ppo_reach/random_seed2026_pilot_04f2/`及最终100目标测试。配置与准备命令一致：random_reach、reach_shaped_v1、每轴0.25mm、50ms周期、terminate、CPU、seed2026、scratch。状态completed，16384步/32个Rollout/320个epoch/2560次Adam更新；没有用旧固定目标或BC权重初始化。

原训练目录63个文件在本次复核前后保持相同哈希；新增只读审计在`outputs/rl/ppo_random_reach_validation/pilot_review_20261113/audit.json`。本次没有更新模型、运行新训练、重复选择checkpoint或改动环境/控制/奖励。

### 实际训练覆盖和开发曲线

16384步共完成99个Episode，另有54步未结束：成功9、超时58、workspace_limit 23、joint_limit 9，训练成功率9.09%。训练采样不是固定评估，其成功数在四个4096步窗口分别为5/27、3/27、1/23、0/22，没有持续增加。

训练/更新墙钟273.654秒，评估270.862秒，总544.826秒。约一半墙钟用于独立评估，不产生训练样本；判断预算应看交互步数、Rollout数及Episode覆盖，不能只看进程运行了几分钟。

同一开发集87000–87019的确定性结果：

```text
训练步数   成功/20   平均最终误差(mm)   保护终止   超时
       0      0          4.104525           0       20
    4096      0         10.102286          20        0
    8192      0          4.264055           0       20
   12288      1          6.173555           0       19
   16384      0          6.586518          10       10
```

12288步的一次成功只是1/20，后续又为0/20；不能称作明确的持续学习趋势。最终平均误差也比同集第0步更大。本轮预算仅覆盖99个完整Episode，可能仍不足，但曲线不支持仅以“已接近收敛，继续加步数即可”解释当前失败。

### 已完成的100目标结果和方向偏置

已核对最终JSON来自本随机任务16384步模型、保存的同一环境，种子为90000–90099。结果成功4/100、平均末端误差6.197187mm、平均139.73步、成功平均20.75步；52次超时，44次joint_limit保护，无碰撞、IK失败或数值失败。此处joint_limit是日志中的保护类别；现有评估JSON没有保存具体detail，不能仅凭名称宣称实际关节突破了模型物理限位。

4次成功全部在`+X,-Y,+Z`象限（该象限共14例）。全部100个目标的首步动作分量符号也都是`+X,-Y,+Z`，尽管目标有49个在-X、55个在+Y、48个在-Z。动作幅度会随目标变化，因此不能说网络完全忽略目标；但起始方向没有随目标符号切换，是目标条件控制尚未学好的直接表现。全程动作也不是一直+X，后续X分量会改变符号，不能将首步偏置误写为整条轨迹恒定方向。

最终评估中仅43.92%的非零动作朝当前实际目标，42.49%的步骤使距离改善；相邻动作反向率为0，不支持直接套用历史SAC“高频往复”的诊断。实际路径/命令路径长度比为0.9891，说明机器人确实在执行运动，但该比值不排除瞬时控制滞后，也不证明策略指令正确。

确定性100目标评估的Box裁剪率为0，raw最大绝对值0.6121。因此固定目标阶段的严重动作饱和不是本轮确定性失败的直接原因。训练随机动作累计裁剪率仍为66.42%，探索裁剪对学习的影响未被排除。

最终value_loss为1.18408、explained_variance为-0.01582、approx_kl为0.01188、平均std为0.94920；未见非有限数值。EV接近零提示当前Rollout价值预测对returns的拟合较弱，不能由此单独认定Critic实现有错或确定某个奖励问题。现有结果支持方向控制不足、最终Rollout价值拟合偏弱，不能只从这一个seed定位数据预算、探索与信用分配的因果贡献。

### 对“训练时间是否太短”的判断

**可能预算不足，但不能认定只是训练太短，也没有证据保证加长便会成功。** 32个Rollout、99个完整随机目标Episode是较小的覆盖；同时开发集无持续改善、后段训练成功减少、固定首步方向和较弱价值拟合，均提示需要受控验证。

[SB3官方训练建议](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)指出无模型RL通常需要较多交互，训练结果也有随机性和不稳定性；这只是一般背景，不能推导本任务在某个固定步数必定收敛。

若用户要检验预算不足假设，可以单独做32768步、保持当前配置与seed2026不变的**从头独立实验**，使用新输出目录、同一20目标开发集和4096步评估间隔，只改变预算。它是诊断对照，不是已经确认的修复。本次不执行此训练，不直接建议盲目扩至100000步。当前没有严格跨进程续训，不能把重新加载当前checkpoint称为完整恢复训练。

若增加预算后开发集仍接近零、首动作方向仍固定、EV仍接近零，应停止单纯追加预算，单独审查目标条件映射、价值学习和奖励/探索信号，而不是同时更改多个因素。不得取消原物理保护或混入专家纠正。

90000–90099现已被测试并用于诊断。后续若根据本结果调参或选模型，不能继续将同一集合称作新的未见最终测试集；开发仍用87000–87019，另保留一组尚未查看的目标作未来最终验证。本次仅更新日志，没有重新运行282项回归测试、没有启动训练或提交/推送GitHub。
