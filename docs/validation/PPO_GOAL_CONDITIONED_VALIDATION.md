# 04-F3：PPO目标敏感性与渐进式Reach训练准备

日期：2026-11-13；分支`feat/ppo-random_reach`。本次完成代码、冻结策略诊断、可达性检查、自动化测试及**唯一一次512步PPO Smoke更新**。没有执行16384/32768步正式训练，没有使用BC、专家动作/数据、SAC或旧固定目标权重初始化新策略，没有提交/推送GitHub。

以上及下文准备部分是开发交付时记录。用户随后已手动完成16384步两方向Pilot，结果和只读复核追加在文末；复核没有执行新训练。

## 旧失败Actor是否读取目标

冻结模型为`outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final/model.zip`（16384步）。`scripts/diagnostics/probe_ppo_goals.py`仅构造原模型的ReachEnv，通过正常reset取得观测再调用确定性predict，不执行物理step，不手造目标/关节状态，也不改权重。

先核验5mm轴向候选；±Y未通过原home±5°命令包络，故不能把六方向5mm当合法reset输入。随后**显式选择3mm作为诊断距离**，六个候选全部经过原IK、限位、姿态和路径碰撞预检，再逐目标reset。此距离没有替换任何训练任务默认值。

六次reset的qpos、qvel、ctrl、实际末端与命令参考、仿真时间完全相同。输入顺序仍为achieved_goal(3)、desired_goal(3)、observation(27)。33维中仅索引`3,4,5`（目标世界坐标）与`21,22,23`（缩放的goal-ee）发生变化，其余特征完全相同；末端相对目标保留正确符号/量级。

```text
目标  world XYZ (m，显示到9位)                 预处理goal-ee     Actor确定性动作 XYZ          动作/目标cos
+X   [0.008616691,-0.000145159,1.060227293]     [+.3,0,0]        [+.398270,-.041936,+.421448]   +.685049
-X   [0.002616691,-0.000145159,1.060227293]     [-.3,0,0]        [+.315457,-.068625,+.388205]   -.624789
+Y   [0.005616691,+0.002854841,1.060227293]     [0,+.3,0]        [+.312384,-.019922,+.372396]   -.040951
-Y   [0.005616691,-0.003145159,1.060227293]     [0,-.3,0]        [+.400848,-.089609,+.436789]   +.149453
+Z   [0.005616691,-0.000145159,1.063227293]     [0,0,+.3]        [+.439858,-.063279,+.530830]   +.766779
-Z   [0.005616691,-0.000145159,1.057227293]     [0,0,-.3]        [+.270700,-.046130,+.273961]   -.706280
```

另外对合法的±X5mm，目标坐标分别是`[0.010616691,-0.000145159,1.060227293]`和`[0.000616691,-0.000145159,1.060227293]`，预处理向量为`[+.5,0,0]`/`[-.5,0,0]`；动作分别为`[+.424006,-.033210,+.430682]`和`[+.286646,-.077527,+.375449]`，cos分别+.700505/-.598822。

结论是**Actor对目标有响应，但部分初始方向错误**：六个目标动作幅度不同，不能说完全忽略目标；符号却全部保持+X、-Y、+Z，只有3/6朝向目标。没有把“初始方向朝向目标”当作闭环成功，也未用规则策略修正输出。模型参数指纹/优化计数不变，完整33维特征、原始坐标及动作保存在`sensitivity_six_3mm.json`和`sensitivity_pair_5mm.json`。

## 候选可达性及课程边界

所有检查仍使用当前模型、home朝向、原DLS、实际关节限位、home±5°实验命令限制和原check_motion，未修改几何/惯量/控制增益。

- +X5mm：IK位置误差0.000140mm、姿态3.05e-7rad，原检查通过。
- -X5mm：IK位置误差0.000157mm、姿态5.63e-7rad，原检查通过。
- ±Y5mm：IK收敛，但返回关节解被`Demo targets must stay within 5 degrees of home`拒绝。这不是物理不可达的证明，也没有穷举所有IK分支；当前配置必须失败。
- ±Z5mm：IK及原检查通过。
- 显式3mm六方向：全部检查通过，其中±Y最大关节变化约3.2305°，仅用于敏感性诊断。

`axis_pair_reach`在home的+X和-X各5mm，每次reset由Gymnasium np_random均匀选择其中一个，Episode内不变。`axis_six_reach`保留六个5mm候选作为后续阶段配置，但当前会同时报告candidate 2/+Y与candidate 3/-Y失败并拒绝整个任务，不丢弃候选、不缩短目标、不启用只含4方向的伪六方向任务。

课程顺序保持两方向→六方向→原3–5mm球壳；第二阶段必须先明确解决目标配置合法性，本次没有把3mm诊断配置变成隐藏训练阶段。未实现`ppo_transfer`；首次两方向只有scratch，PPO权重迁移不是当前CLI功能，也不能冒充严格跨进程续训。

## 实现与文件职责

- `tasks/reach.py`：在原三任务外增加两个轴向候选配置；未改原目标分布。
- `envs/reach_env.py`：同一环境增加`target_mode="discrete"`及米制target_offsets。构造时必须验证全部候选，失败整体报错。候选位置/报告通过副本返回；reset随机选候选或显式`options={"goal_index":i}`，仅reset调用官方home keyframe。正常step、奖励及安全逻辑不变。
- `rl/common/evaluation.py`：离散模式逐一枚举所有候选，要求恰好候选数个Episode，记录per_goal、独立运动统计和结束详情；逐目标TensorBoard值来自真实结果。原固定/随机评估路径和指标保留。
- `rl/ppo.py`：新增离散评估范围说明，参数/网络/分布/更新流程保持原样。
- `scripts/train/train_ppo_reach.py`、`scripts/evaluate/evaluate_ppo_reach.py`：离散任务默认自动评估全部候选；明确给错误数量报错。旧任务默认1集，随机任务仍可显式20/100集。阶段JSON保存per_goal，独立CLI分别打印两方向结果。
- `scripts/evaluate/analyze_reach_results.py`：只读分析增加discrete_goals；相同reset seed可用于不同显式候选，重复候选索引仍拒绝，不能把重复目标叫泛化。
- `scripts/plots/plot_ppo_training.py`：离散任务额外导出逐目标成功/距离/首动作XYZ、随机训练成功率、评估保护/超时图；旧模式仍为原11图。
- 新增`scripts/diagnostics/probe_ppo_goals.py`：合法reset的冻结Actor目标敏感性，不执行step/训练。
- 新增`tests/test_goal_conditioned_reach.py`及`fixtures/reach_env_legacy_contract.json`：目标、原语义逐值回归、唯一512步更新和完整评估/监控测试。
- 原`probe_reach_env.py`及两项旧任务遍历测试明确保持原固定/球壳任务覆盖；离散任务由新测试检查，六方向预期拒绝有单独负例。原PPO源码哈希测试改为物理资源精确哈希＋旧step/奖励/安全方法AST摘要，详见下节。

没有恢复experiments目录，没有新增环境、控制器、方向标签、奖励、算法或依赖。goal_index只在reset选项/info里，未加入动作或27+3+3观测。原模型及所有历史输出原字节保留。

## 评估、TensorBoard及PNG语义

两方向每次评估恰好+X和-X各一次，不依赖seed碰巧抽中不同目标。reset seed是标识和复现标签，不构成随机位置泛化。六方向合法时同理六个不同候选各一次。显式`--episodes 100`或`--eval-episodes 20`用于两方向会清晰报错；省略时自动采用2，旧任务省略仍为1。

每个方向分别输出success、目标/实际位置及误差、Episode长度、first_action XYZ、termination_reason/detail、实际路径长度、动作朝当前实际目标比例。完整几何路径由原`actual_positions_before_m`加final_position_m还原，仍来自执行器的真实物理运动。

TensorBoard保留原loss/EV/KL/std/Box裁剪/训练Episode及全局评估标签，并新增：

```text
evaluation/goals/plus_x|minus_x/success
evaluation/goals/plus_x|minus_x/distance_mm
evaluation/goals/plus_x|minus_x/episode_length
evaluation/goals/plus_x|minus_x/first_action_x|y|z
evaluation/goals/plus_x|minus_x/action_direction_fraction
evaluation/goals/plus_x|minus_x/actual_path_mm
evaluation/goals/plus_x|minus_x/termination/<reason>
```

结束分类按0/1写每个原因，避免某次失败标签残留为“始终失败”。成功是单个目标的布尔结果，不给两个位置重复100次产生虚假样本量。训练Episode成功率与确定性评估继续分开。PNG为原11图加7张课程图，共18张；原始曲线保留plot_data.json，没有评估数据时不造成功率图。

## 验证结果及唯一Smoke

修改前原282项测试全部通过（150.698秒），基线运行显式复用原固定/随机PPO Smoke产物，没有新增PPO训练。修改后**297/297通过**（176.573秒），新增15项；另定向验证了相同seed枚举不同候选的分析分支，未重复训练。

原测试中对ReachEnv整个文件的历史哈希不能用于允许新增reset功能的本次变更。因此保留模型/mesh/所有simulation源码精确哈希，冻结原`step`、`compute_reward`、`_shaped_reward`、`_sample_goal`、`_observation`、`render`、`close`七方法AST，并用改动前捕获的固定/随机轨迹逐值检查qpos/qvel/ctrl/观测/奖励/终止/物理时间。两方向+X与原固定+X相同动作下也逐值相同。未放宽任何精度、力矩、碰撞、速度或限位阈值；原282测试名/数量保留。

唯一新Smoke在`axis_pair_smoke_seed2026/`，随机初始化、原33维输入与256×256/Tanh、原PPO_CONFIG不变。512个环境交互、1个Rollout、10个epoch、80次Adam更新，Actor与价值网络指纹均改变；优势/returns/参数和指标有限。训练2次200步超时，另112步未结束；无安全失败。子进程阻断本项目SAC、BC、SDK及专家动作，1312次env.step（训练512＋两次各400步评估）逐物理步审计live qpos只由mj_step改变。

```text
阶段         方向    结果       步数   误差(mm)   实际路径(mm)  朝目标动作比例
第0步        +X      timeout    200    5.094956    0.129255          0%
第0步        -X      timeout    200    4.940708    0.060645        100%
第512步      +X      timeout    200    2.039788    6.401666         73%
第512步      -X      timeout    200   11.305812    7.101503          0%
```

第512步首动作：+X目标`[+.142070,-.004069,+.011527]`，-X目标`[+.105747,-.038060,+.130439]`。短测试两方向都未成功，只验证更新和监控链路，不推断课程正式训练效果。训练向量Box裁剪66.80%、分量30.34%；两目标确定性评估裁剪为0。未因此调整std、奖励或控制参数。

已独立加载保存模型，分别得到相同±X结果；TensorBoard事件检查通过，18张PNG和只读逐目标分析已输出。CLI --help、完整Pilot参数及非空目录拒绝、两目标错误重复100次的失败入口均验证；评估/绘图从/tmp用绝对路径及python -I运行，不依赖手工路径设置。未打开浏览器/Viewer验证GUI。

全部本次产物位于`outputs/rl/ppo_goal_conditioned_validation/20261113_preparation/`：原文件哈希、两组候选预检、敏感性JSON、基线/完整测试日志、唯一Smoke、加载评估、逐目标分析、TensorBoard检查和axis_smoke_plots。依赖保持原版本，无安装升级。

## 用户手动训练与检查命令

当前CLI仍从头创建PPO；以下Pilot没有由Codex执行，run-dir必须是新的空目录。保持3e-4学习率、512/64/10、gamma .99、GAE .95、clip .2、ent_coef 0、vf .5、max_grad_norm .5、CPU及seed2026。16384步正常为32个完整Rollout，评估/保存仅在完整更新后，每2048步一次。

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I scripts/train/train_ppo_reach.py \
  --task axis_pair_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3 \
  --eval-every 2048 --eval-episodes 2 --eval-seed-start 87000 \
  --checkpoint-every 2048 --tensorboard

# 另开同环境终端
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/sb3 --host 127.0.0.1 --port 6006

python -I -m json.tool outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/summary.json
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/checkpoints/step_016384/model.zip \
  --episodes 2 --json outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/distinct_goal_evaluation.json
python -I scripts/evaluate/analyze_reach_results.py \
  --evaluation outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/distinct_goal_evaluation.json \
  --json outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/distinct_goal_analysis.json
python -I scripts/plots/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3
```

如需重新只读诊断旧失败Actor，用新的输出JSON（不能覆盖本次产物）：

```bash
python -I scripts/diagnostics/probe_ppo_goals.py \
  --model outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final/model.zip \
  --task axis_pair_reach --distance-mm 5 --json /tmp/realman_goal_pair_probe_new.json
python -I scripts/diagnostics/probe_ppo_goals.py \
  --model outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final/model.zip \
  --task axis_six_reach --distance-mm 3 --json /tmp/realman_goal_six3_probe_new.json
```

## 未验证与停止边界

尚未验证正式两方向Pilot是否学会方向切换、多seed稳定性或更复杂目标泛化。只有+X成功时先看目标输入与动作映射；两方向都失败时先看奖励/价值估计，不盲目扩大预算。两方向均稳定后再讨论六方向合法配置，当前5mm配置不能直接用作下一轮训练；本次没有开展六方向正式训练、PPO权重迁移、BC融合或新的随机目标训练。所有旧PPO/SAC/BC模型及失败日志保留，等待用户手动Pilot反馈。

## 用户手动两方向Pilot结果（2026-11-13）

用户已完成`outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/`，并运行了独立评估、分析与18图导出。本次只读取日志、独立加载最终模型复验两个目标、执行无物理step的同状态目标敏感性诊断并更新文档，没有新增训练、调参、模型选择或进入六方向阶段。

### 配置、计数与训练采样

配置与交付命令一致：axis_pair_reach，±X5mm，scratch/seed2026，reach_shaped_v1，每轴0.25mm/50ms，原PPO_CONFIG。状态completed，16384步、32个Rollout、320个epoch、2560个Adam步。训练/更新276.689秒，约59.21步/秒；阶段评估41.905秒，总318.831秒。

已完成121个随机训练Episode，成功49（40.50%）、超时63、workspace_limit 9；另有29步未结束。无joint_limit、碰撞、IK、姿态、速度、驱动力或数值保护终止。按真实日志目标和Episode边界分别统计：

- +X：64集，49次成功、10次超时、5次workspace_limit，共5068步。
- -X：57集，**0次成功**、53次超时、4次workspace_limit，共11287步。

因此-X不是没有被采样，也不是训练时间步更少。该结果不能证明57集已经足够学习，但可排除“忘了采样-X”这一解释。训练总体40.50%不能当作两方向均已掌握。

### 逐目标确定性评估

每个checkpoint均为+X、-X各一次，不按重复seed扩大样本量。

```text
训练步    +X结果          步数   +X误差mm   -X结果          步数   -X误差mm
    0     timeout         200    5.094956   timeout         200    4.940708
 2048     workspace_limit  88    5.273871   workspace_limit 145   14.437174
 4096     timeout         200    2.954262   timeout         200    9.602824
 6144     timeout         200    2.015079   timeout         200    8.980317
 8192     success          27    0.984122   timeout         200    9.244163
10240     success          23    0.953480   workspace_limit 161   13.807155
12288     success          21    0.897397   timeout         200   10.815211
14336     success          18    0.977740   timeout         200   10.523596
16384     success          19    0.929137   timeout         200   10.241481
```

+X从8192步起后续5个评估checkpoint持续成功；-X没有一次成功。最终独立评估与重新加载final/model.zip、换reset标签89000/89001复验完全一致，参数指纹和优化计数保持不变。换标签仍是同两个目标，不是新的泛化样本。

最终两目标平均值“成功率50%、误差5.585309mm”掩盖了方向不对称，验收应分别读取：

- +X：19步成功，0.929137mm；首步实际动作`[1,-0.597289,-0.132355]`，实际路径4.453517mm，100%非零动作投影朝当前实际目标。
- -X：200步超时，10.241481mm；首步实际动作`[1,-0.255647,+0.560247]`，实际路径6.684003mm，49%非零动作投影朝当前实际目标。

最终-X是超时，不是被关节或碰撞保护提前中止。目标距离误差超过10mm也不等于超出home的10mm命令区域；策略向目标的反侧移动，仍可能处于原命令区域内。

### 目标输入、运动符号与停滞

再次通过合法reset对最终模型作冻结敏感性检查：物理初始状态相同，33维中仅`3,21`两个X目标相关特征改变；预处理相对目标分别为`[+.5,0,0]`和`[-.5,0,0]`，单位/符号正确。Actor的Y/Z分量及原始X幅值会随目标变化，故不能说完全忽略目标；但-X的首步X方向仍然错误。

首步原始动作X分别为+1.705876和+1.019922，经原Box裁剪后均为+1。裁剪没有把负命令翻成正命令，网络原本输出的就是正值。+X向量裁剪18/19（94.74%），-X为7/200（3.5%）；合并25/219=11.42%不能代替分方向分析。

对-X Episode，所有实际输入动作累加得到净命令位移`[+4.911008,+1.297469,+2.246205]mm`，实际净末端位移为`[+4.908470,+1.293061,+2.244583]mm`。控制器基本执行了策略给出的正方向命令，不支持“负X动作被底层执行成正X”的解释。此比较不排除瞬时滞后，但主要错误已经存在于策略命令中。

-X共有121/200步动作范数小于0.01（归一化单位，60.5%），单步笛卡尔命令范数相应小于2.5µm；最后一步动作约`[.002984,.002597,.000461]`。表现为先向错误方向移动，然后在离目标仍远时接近静止。49%的“朝目标”投影中可包含极小动作，不能仅凭方向比例宣称完成有效纠偏。

### 价值函数与结论边界

最终value_loss=0.020222、explained_variance=0.972720、approx_kl=0.008765、PPO clip_fraction=0.107031，平均std=0.923240；网络/优化日志保持有限值。相比旧三维随机任务，价值拟合指标改善，但Actor仍不能执行-X。这说明高EV并不等于正确的目标条件控制；该指标衡量当前Rollout价值预测与returns的关系，不验证策略是否选择了正确方向。

本轮可以确认：+X学会、-X未学会；目标被正常采样、观测保留正确符号、Actor有响应但X方向映射不足，随后发生远离目标的停滞。尚不能据这些现象单独确定奖励信用分配、两方向更新竞争或其他学习因素的因果作用。

**两方向阶段尚未通过，不进入六方向，也不自动做PPO权重迁移或追加训练。** 后续应按本阶段原定标准重点诊断-X输入到Actor动作的映射及分方向学习信号；不以总体50%成功或高EV作为阶段成功依据。六方向5mm的±Y预检限制仍保留，未因本轮结果放宽。

复核证据在`outputs/rl/ppo_goal_conditioned_validation/pilot_review_20261113/`：audit.json、独立加载评估、goal_sensitivity.json及训练目录前后哈希。原训练目录90个文件保持原字节。只读复核没有重新运行297项回归测试；297/297是前文开发验收记录。本次未改Python代码、模型或依赖，没有新训练、提交或推送GitHub。
