# 04-E1：纯SAC Reach配置、奖励与渐进验证

日期：2026-11-08。本次恢复纯强化学习研究；没有加载BC初始化/冻结参照、专家Replay、BC损失或专家动作纠正。BC代码、模型、旧实验全部保留，仅作历史记录。没有运行正式5k/30k训练；唯一独立训练Smoke为128步，测试中的短更新只验证接口。

以上及下文开发验收部分记录交付时状态。用户随后已手动完成两组5000步Pilot，最新结果见文末“用户手动Pilot结果记录”；本次日志更新没有启动新训练。

## 现有问题与本次边界

04-A原30k独立100目标仅1%；04-B同预算10k的terminate/reject_hold分别0%/1%；用户手动04-D1的固定20监控目标中，BC初始化从0步100%退至最终0%，scratch也为0%。这些历史预算、目标、机制不同，不能冒充本轮奖励对照。原日志和数值不覆盖。

旧1mm是**每轴**50ms增量，三维动作最大向量长度1.732mm；累计命令参考不是实际末端位置。理想无界独立均匀随机动作在H步的命令RMS距离为`action_scale*sqrt(H)`，200步约14.14mm，超过10mm实验命令球；.25mm尺度约3.54mm。这只是忽略限位/IK/控制响应的探索尺度估算，不是真实工作空间结论。

控制还含每周期.5°关节指令变化上限、PD响应及IK。04-B已观察实际路径小于命令路径、策略往复，不能靠改变动作尺度宣称控制延迟消失。本次不改PD、IK、动力学、姿态/速度/力矩/碰撞阈值或实际关节限位。

旧safe_dense的普通距离成本在5mm时-.25，终止失败额外-201，存在明显数值尺度差异；旧折扣轨迹没有证明策略在利用终止。本轮既保留旧模式，也不把低成功率或高Q直接归咎于某个唯一因素。

## 三种显式任务

`experiments.reach_tasks.TASKS`配置同一个ReachEnv：

- `fixed_reach`：原home世界+X5mm，构造时已用原IK/路径检查验证；用于消除目标条件变化。
- `easy_random_reach`：home周围2–3mm体积均匀球壳，再经过原可达性、home ±5°、姿态及51点碰撞预检；最多64次重试。初始距离大于1mm成功阈值。
- `random_reach`：原3–5mm条件分布，采样器和种子机制不变。

每个任务都从home keyframe重置，目标朝向为home；保持六轴模型、25个2ms物理步/50ms控制周期、gravity compensation、200步、1mm成功阈值、三维Box(-1,1)动作及27+3+3 Dict观测。没有第二套机械臂环境。

实测每个任务20个reset目标：easy半径2.05456–2.99354mm，XYZ范围[-2.50349,+2.55847]、[-2.47478,+2.10632]、[-2.94819,+2.69175]mm；random半径3.12338–4.98801mm。fixed恰为+X5mm。它们是经原检查接受后的局部条件分布，不是完整工作空间。

CLI独立支持task、reward-mode、action-scale；原timesteps/seed/eval/checkpoint/run-dir继续使用。旧不传新参数的入口保持原random/safe_dense/1mm行为。任何新任务/奖励/动作配置均强制scratch且禁止bc-model，包括只读BC参照；没有隐式续训或阶段权重迁移。历史BC入口保留供旧实验复现，不用于本轮。

## 为什么不用直接负指数加小惩罚

候选`r=-(1-exp(-d/.005))+1*success-1*failure`在5mm静止的gamma=.99、200步回报约-54.743，而第一步故意失败约-1.632，明显鼓励提前退出。此反例基于真实静止/越界状态重新计算，未把该候选安装为环境奖励。

只缩小旧-201并不足够。新可选`reach_shaped_v1`把指数距离用作**势函数**，并显式处理折扣和终止；旧safe_dense和distance代码路径保留。它不是新的默认奖励，也不宣称等价于旧safe_dense的最优策略。

## 新奖励公式、单位与回报边界

设当前实际末端为p，下一实际位置p'，目标g，距离单位m；sigma默认.005m，gamma=.99，H=200，成功阈值epsilon=.001m：

```text
Phi(p,g) = -(1-exp(-||p-g||/sigma)), in [-1,0]
K = 1.01 / gamma**(H-1)                 # default 7.4630722985
c_reject = 1-gamma                     # default .01
success = (||p'-g|| <= epsilon) and not safety_failure
Phi_next = 0 if success or safety_failure else Phi(p',g)
r = gamma*Phi_next - Phi(p,g)
    + K*(success - safety_failure) - c_reject*is_action_rejected
```

奖励无量纲。成功奖励/失败惩罚同为K，1.01是严格排序的.01余量，不按训练成功率拟合；拒绝成本由gamma推导。新模式不沿用旧reject_and_hold的1.0额外成本，后者仅属于旧safe_dense约定；新模式实际参数另存config.reward_parameters。sigma/gamma可通过Python环境参数独立设置，训练器强制shaping gamma与SAC gamma相等；不改原SAC gamma。

Phi的折扣差形式参考[Ng等的势函数奖励研究](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)。这里的任务终止与惩罚另作如下有限Episode验证，不能仅引用理论就声称本项目已学会。

令初始C0=-Phi(p0,g)，拒绝折扣成本J=(1-gamma)*sum(gamma**t*rejected_t)。同一个初始位置/目标下：

- 成功于T≤H：`G=C0+gamma**(T-1)*K-J > C0+.01`。
- 超时：保留下一势，`G=C0+gamma**H*Phi(pH,g)-J`，位于`[C0-1,C0]`。
- 安全失败于T≤H：`G=C0-gamma**(T-1)*K-J ≤ C0-1.01`。

因此在声明gamma和H的**不含策略熵的折扣Episode回报**下，任意成功严格优于任意超时，任意超时严格优于任意提前失败，包含反复拒绝的最坏情况。它避免了借提前失败少付未来成本的优势。K随H增大而增长，不能把此设计无条件推广到很长horizon；非有限/过大尺度明确报错。

静止单步奖励可能为小正数，因为gamma*Phi-Phi是折扣势差；不能把它当作“越远越好”。同起点的累计回报已包含势的望远镜抵消，成功仍严格更优。200步会正常超时，没有无限Episode。不把timeout错误设为吸收终止，保留原SB3 bootstrap约定。SAC还优化熵项且学习Critic，以上不是soft Q的精确监督，也不是训练必然收敛的保证。

奖励需要info.previous_achieved_goal（原始世界米制XYZ）、is_safety_failure及is_action_rejected；scalar/dict批量/记录列表均支持。成功从传入的新goal重新判断，不读取旧info.is_success或旧distance。step把实际步前位置写入info并调用同一个compute_reward。原观测形状不变。

未来HER仍需同步重标记observation中的相对目标并重算成功终止；本次只保留可重算奖励所需的原始前后位置，未实现HER。

## 真实轨迹与动作尺度验证

固定+X5mm目标、相同初态和物理参数。成功探针只是先5次固定+X动作再保持，用于奖励验收，**不作为训练策略、标签或Replay来源**。越界/拒绝探针显式使用每轴10mm、[1,1,1]以确保第一步只拒绝命令、不会执行该位移；训练仍使用下述1mm/.25mm。

实测gamma=.99折扣回报：

- safe_dense：成功6步-.910944；静止200步超时-21.650508；反复拒绝200步-108.252541；第一步越界-201.25。
- reach_shaped_v1：相同成功轨迹7.729428；静止.547429；反复拒绝-.318591；第一步越界-6.830952。

两种奖励的实际位置和终止原因逐步完全一致；新奖励逐轨迹核验望远镜恒等式。静止/拒绝都低于成功，没有通过修改qpos、关闭重力或放宽保护取得排序。原始每步动作、位置、奖励、物理步数、原因保存在reward_and_action_probes/results.json。

动作尺度对照只改每轴scale，沿用原random目标、安全机制及独立均匀随机动作种子，4个配对Episode：

- .001m：17/14/34/36步均joint_limit，平均每步实际路径约.44593mm，实际/命令路径比约.377–.511。
- .00025m：4个均200步time_limit，无安全失败，但也无成功；平均每步实际路径约.11604mm，路径比约.445–.488。

这些joint_limit是候选命令局部限制，未取消实际关节保护。小尺度减少这4例的提前失败，未解决任务，也未明显消除跟踪滞后。样本很少，不作为训练效果或普遍成功率。首次奖励比较仍保留1mm，随后才独立研究.25mm，不能同时换奖励、尺度和目标来归因。

## Smoke、监控与安全

唯一独立Smoke：`fixed_reach + reach_shaped_v1 + .001m`，随机SAC seed2026，128交互、28官方更新、Replay128，无BC/专家。成功率0；2个完整探索Episode均joint_limit，另21步未完成集。没有碰撞/数值/速度/驱动力故障或NaN/Inf。测试不要求少量更新学会任务。

训练用时约2.384s（短run约53.69交互/秒，不能外推5k吞吐）；独立评估约10.131s。每次完整控制仍25物理步。实际路径.053225m、命令路径.115819m，比.459557；动作朝实际目标53.125%、朝命令目标52.344%，符合随机探索，而非专家导向。

继续记录TensorBoard的成功率/距离/长度/训练回报/Actor-Critic loss/alpha/安全失败/拒绝；新增training和evaluation的实际位移、实际/命令路径、朝实际/命令目标比例、距离改善及反向比例。训练callback正确使用terminal_observation，避免把VecEnv自动reset后的home跳变计入运动。只读MotionStats不产生控制指令。

TD类别对新奖励必须使用Episode安全/成功/timeout标记，不能把“不低于-200的终止奖励”都当成功。本轮纯RL快照已改用显式标记；原safe_dense诊断的旧默认保留供历史复现。旧reward_distribution.safety_sized_rewards仍是<-200计数，新奖励通常为0，不等于没有真实失败，应看safety_failures/termination_counts。

已生成TensorBoard事件和4种PNG，事件标量与真实JSON核对，无bc/*标签；缺少评估时原绘图程序仍不伪造成功曲线。模型/预处理/配置/Replay均保存且可以独立加载。

## 回归与文件职责

原234项基线通过（132.214s）。新测试涵盖公式、标量/批量、goal重算、最坏折扣排序、负指数反例、真实超时/拒绝/数值保护、物理不变、可达采样、动作范围、25步且无qpos瞬移、Gymnasium/SB3 checker、禁止BC/SDK导入和专家调用、保存加载、纯RL TensorBoard及显式TD标签。旧测试和安全阈值不删不改。

回归曾发现通用探针的函数内导入造成静态依赖环；已把它放回定义ReachTransform的通用sac_reach，消除环，保留旧初始化模块的导入接口。新目标测试最初误把1e-5m位置容差用于姿态；已按原IK的1e-4rad姿态容差核验，没有更改求解器或旧安全断言。最终测试数量和结果见下文完成记录。

新增：

- `experiments/reach_tasks.py`：3种任务定义、无策略输出的SI运动统计；不导入BC或SDK。
- `scripts/validate_reach_training.py`：有界真实奖励、随机动作尺度和目标采样探针，不训练网络。
- `tests/test_pure_reach_training.py`及本报告：18项纯RL/奖励契约与实际证据。

修改：ReachEnv只加可选奖励及其必要info，原路径保持；train_sac_reach增加task/reward/action配置并禁止新实验BC；sac_reach增加通用探针/可选运动统计；sac_monitoring延迟加载历史BC辅助，新增纯RL运动/失败频率；sac_diagnostics接受显式终止标签；sac_initialization仅复用通用探针，旧移植功能保留作历史复现；README与architecture记录当前方向。simulation、vendor、URDF/STL/MJCF、依赖版本不变。

## 用户手动Pilot与渐进阶段（开发交付时未执行）

每次CLI都是独立随机初始化、空Replay/优化器，没有SAC续训或权重迁移。建议先完成同条件固定目标奖励对照：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal

# A1：固定目标，旧奖励对照
python -I scripts/train_sac_reach.py --task fixed_reach --reward-mode safe_dense --action-scale 0.001 --run-dir outputs/rl/pure_reach/fixed_safe_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 1 --eval-seed-start 85000 --tensorboard

# A2：只改变奖励
python -I scripts/train_sac_reach.py --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.001 --run-dir outputs/rl/pure_reach/fixed_shaped_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 1 --eval-seed-start 85000 --tensorboard
```

fixed目标与seed无关，确定性评估1集即可检查这个固定起点/目标，不代表随机目标成功率。先看A1/A2实际结果再选奖励；不要用不同奖励累计回报直接排名，应比较成功、距离、长度和故障。

仅在奖励对照后，再单独比较动作尺度。以下示例以实验奖励为候选，不表示已经证明它更好：

```bash
# A3：相对于A2只改动作尺度
python -I scripts/train_sac_reach.py --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 --run-dir outputs/rl/pure_reach/fixed_shaped_small_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 1 --eval-seed-start 85000 --tensorboard

# 阶段B：相对于A3只改变目标分布，仍从头训练
python -I scripts/train_sac_reach.py --task easy_random_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 --run-dir outputs/rl/pure_reach/easy_shaped_small_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 20 --eval-seed-start 85000 --tensorboard

# 阶段C：相对于B只扩大目标球壳，仍从头训练
python -I scripts/train_sac_reach.py --task random_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 --run-dir outputs/rl/pure_reach/random_shaped_small_seed2026_pilot --timesteps 5000 --seed 2026 --init-mode scratch --eval-at-start --eval-every 1000 --checkpoint-every 1000 --eval-episodes 20 --eval-seed-start 85000 --tensorboard
```

**不要连续自动运行所有阶段。** 若固定目标尚未稳定成功，停止扩大任务，先检查奖励/探索/终止；阶段间学习参数也未自动迁移。是否扩大到30k由用户根据Pilot决定，本次没有执行。

监控、结果查看、独立加载评估与PNG（示例A2目录）：

```bash
python -I -m tensorboard.main --logdir outputs/rl/pure_reach --host 127.0.0.1 --port 6006
python -I -m json.tool outputs/rl/pure_reach/fixed_shaped_seed2026_pilot/evaluations.json
python -I scripts/evaluate_sac_reach.py --model outputs/rl/pure_reach/fixed_shaped_seed2026_pilot/final/model.zip --episodes 1 --seed-start 86000 --policies sac --json outputs/rl/pure_reach/fixed_shaped_seed2026_pilot/loaded_evaluation.json
python -I scripts/plot_sac_training.py --run-dir outputs/rl/pure_reach/fixed_shaped_seed2026_pilot
```

easy/random阶段可在新种子上评估100个目标；fixed不同seed仍是同一个目标，不能冒称100目标泛化。以上新参数已以128步Smoke及短测试验证，正式5k命令未执行。现有可复现实验探针为：

```bash
python -I scripts/validate_reach_training.py --out outputs/rl/pure_reach_validation/new_probes
python -I -m unittest discover -s tests -v
```

## 开发完成记录

最终冻结代码后在新进程运行完整回归：**252/252通过（234原有+18新增，135.953s）**，日志为`outputs/rl/pure_reach_validation/frozen_all_tests.log`。新增18项单独复验也通过。修复依赖环期间的旧测试进程曾保留移动前模块，出现make_probe导入缓存不一致；已重新启动完整回归，不把该失败日志当作最终结果。

650项基线文件清单中，改变的已有代码仅为本报告列明的6个实验/环境入口源文件；机器人资产、simulation、vendor、原BC网络代码/模型、原测试和所有历史输出逐字节保持。没有新增依赖或改其他Conda环境；pip check和git diff --check通过。

Smoke TensorBoard包含45个标量tag，实际位移/方向字段与评估数据可读，无bc/* tag；4个PNG的成功率序列与evaluations.json完全一致。真实模型已在独立评估入口重新加载：固定目标1集超时、最终误差4.15749mm、原始回报1.25026；正回报不等于成功，也未被报告为学会。

CLI负例显式传入`--task fixed_reach --bc-model ...`返回1，在创建run目录前拒绝，不读取BC模型。没有运行正式5000步或更长训练，没有提交或推送GitHub，没有进入抓取训练。

## 用户手动Pilot结果记录（2026-11-08）

用户已自行完成A1/A2；本次仅读取产物、补齐两组PNG并写入结果，没有追加训练、环境交互或修改算法。原始模型、Replay、配置、CSV/JSON/TensorBoard日志均保留。机器可读汇总为`outputs/rl/pure_reach/pilot_review_20261108.json`，包含源文件SHA256与全部阶段指标。

### 配置核对

- A1目录：`outputs/rl/pure_reach/fixed_safe_seed2026_pilot`，safe_dense。
- A2目录：`outputs/rl/pure_reach/fixed_shaped_seed2026_pilot`，reach_shaped_v1。
- 两组均completed、5000交互、4900次更新、Replay5000条；随机scratch、seed2026、无BC模型/参照。
- 任务均为固定home世界+X5mm，scale=.001m/轴、周期.05s、H=200、成功阈值.001m、原重力补偿及安全规则。
- SAC参数、预处理、目标、资产哈希和实验源代码哈希逐项一致。环境差异仅为reward_mode及新奖励专用sigma=.005m/gamma=.99参数，符合奖励单因素对照。
- 每1000步评估一次，包含0步；每次1集、seed85000。固定起点/目标下的确定性检查不代表随机目标总体成功率，换seed也不产生新的固定目标。

### 独立评估轨迹

两组在0/1000/2000/3000/4000/5000的所有评估中均未成功：

- A1最终距离依次为3.68067、5.10886、6.08529、7.45031、8.12997、12.58722mm；前5次均200步超时，5000步模型第36个环境动作触发workspace_limit。
- A2最终距离依次为3.68067、10.40660、5.92275、6.93643、6.72978、7.44793mm；全部200步超时，没有评估安全失败。

A2最终误差小于A1，但两者末态对应200步与36步，不能称为相同运动时长下的精度对照。A2最终误差也大于相同初始化网络的3.68067mm，未形成有效的确定性Reach反馈。

### 训练探索统计与数值状态

- A1：44个完整Episode，成功0，15次workspace_limit、7次joint_limit、1次orientation_limit、21次超时，另33步未完成集；训练成功率0%，安全标记23次。
- A2：33个完整Episode，成功3（9.09%），7次workspace_limit、4次joint_limit、2次orientation_limit、17次超时，另64步未完成集；安全标记13次。
- 这些训练成功来自随机探索，不能替代独立评估。workspace/局部joint命令限制与实际姿态保护不同；两组均无碰撞、数值、实际速度或驱动力故障，所有checkpoint参数有限。
- 纯训练墙钟A1约129.711s（38.55步/秒），A2约128.261s（38.98步/秒）；含独立评估与保存总耗时约147.337/148.023s。

最终固定探针Q均值A1约70.6435、A2约19.6665；最后minibatch Critic loss约100.2498/.130343，alpha约.362693/.233561。新奖励降低了数值尺度及部分故障，但**不同奖励下的Q、loss和累计回报不是可直接排名的任务成绩**，也不能单凭Q或loss宣称SAC实现错误。

### 动作与执行诊断

A2最终200步评估：

- 动作范数<.01的比例86.5%，即当前1mm尺度下请求位移向量<10μm；最后20步平均动作范数约.001264，对应约1.264μm/步。
- 动作朝实际目标和朝命令目标的比例均37.5%，实际距离改善步数占17%，相邻反向比例约3.02%。主要表现不是此前高频往复，而是移向错误位置后动作趋近零。
- 累计命令路径3.62047mm、实际路径3.51094mm，实际/命令路径比.96975。该统计不证明无延迟，但不支持“控制器没有执行命令”作为本次主要解释。
- A1最终36步评估约30.56%动作朝实际目标，末态越界；A1训练过程的反向比例约66.62%，A2约51.33%，这是包含随机探索的5000步统计，不能与上述确定性末次评估混用。

A2最终未折扣回报为正的1.39469，但仍然超时；不含熵的折扣任务回报.528349，低于同起点原地静止超时的.547429。正奖励或Critic loss降低都不能解释为目标到达。此处静止比较采用同一奖励/目标/gamma/H，不把普通任务回报当成SAC soft Q的精确监督。

### 结论与下一步边界

本轮证据支持：奖励单因素改变后数值尺度和部分失败有所减少，但固定目标仍未学会；最终策略远离成功区后趋于静止。尚未分离动作尺度、探索/熵、价值学习和控制时间动态各自的因果贡献。

暂不进入easy_random_reach/random_reach，也不直接增加至30000步。若继续既定计划，下一项仅将A2动作尺度改为.00025m，保留固定目标、奖励、seed、预算及其余SAC配置；这仍是未经验证的候选，不宣称新奖励已胜出，也不保证能解决过早停止。正式训练仍由用户手动决定和执行。

两组已分别导出成功率、距离、Actor/Critic loss和温度4个PNG，位于各run的plots目录，plot_data.json保留真实绘图序列。日志更新不需要重跑模型/自动化测试，仅核验JSON和文档；没有提交或推送GitHub。
