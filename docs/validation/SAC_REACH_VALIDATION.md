# 04-A：首次SB3 SAC Reach实验

结论：训练与保存/独立推理流程完成，但未获得可靠Reach策略。正式30,000步后，固定100目标成功1/100（初始2/100、随机4/100、规则100/100），99次超时、无安全失败。189项自动化测试通过不等于策略任务成功；本轮未追加步数或修改原环境。

日期：2026-11-07。只使用原六轴ReachEnv，不加载EG2、物体或抓取场景，不调用厂家SDK。没有实现自定义SAC、Replay Buffer、反向传播、HER、PPO或真机接口。

## 环境、安装与隔离

目标解释器为`/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python`，Python3.11.17。全部安装、测试和训练通过realman_rl及python -I执行，忽略继承的ROS PYTHONPATH和用户包。原MuJoCo3.17.0、NumPy2.4.6、Gymnasium1.4.0及其他既有包版本不变。

预检后仅新增8个包：stable_baselines3=2.9.2、torch=2.14.0+cpu、filelock=4.2.9、Jinja2=3.1.6、MarkupSafe=3.0.4、mpmath=1.3.0、networkx=3.6.1、sympy=1.14.0。SB3来自PyPI，Torch来自官方CPU轮子索引，torch.version.cuda=None；未安装CUDA、GPU框架、pandas、matplotlib或TensorBoard。pip check通过，安装计划/下载来源和完整前后依赖快照保存于outputs/rl/sac_reach。

基础requirements.txt和requirements-vendor.txt不变；新增requirements-rl.txt锁定本次运行依赖，pyproject增加rl可选组。CPU版Torch需要安装命令中的官方CPU索引；不能只从默认PyPI索引解析`+cpu`版本。editable元数据用--no-deps更新，无依赖升级。

参考：[SB3官方SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)、[官方VecEnv语义](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)、[PyTorch官方CPU安装](https://pytorch.org/get-started/locally/)。实际版本/轮子URL以保存的安装报告为准，不依赖未来变化的网页。

## 文件职责

- `scripts/train_sac_reach.py`：CPU训练入口，独立run目录、seed、5k分段训练、验证、checkpoint、配置和版本保存；每次CLI调用从头开始，不隐式覆盖/续训。
- `scripts/evaluate_sac_reach.py`：独立进程加载模型及预处理，deterministic=True，按相同种子比较原规则、随机、SAC。
- `src/realman_rl_simtoreal/experiments/sac_reach.py`：可选实验辅助，固定观测变换、官方SB3环境封装、只读训练/Q审计、评估和保存；不实现算法或物理控制。
- `experiments/common.py`与旧`evaluate_reach_baselines.py`：原rule_action原样迁入共享实验模块，旧入口导入同一实现，避免训练模块导入脚本。
- `requirements-rl.txt`、pyproject：独立可选CPU依赖，基础仿真不要求安装RL框架。
- `tests/test_sac_reach.py`：12项官方API、真实采样/参数更新、保存/重载、RNG、SI单位、固定种子、SDK隔离和不改实际qpos验证。
- `tests/test_package.py`：原元数据测试现在分别核对vendor和rl依赖组，仍验证原基础/vendor文件，新增RL文件的完整一致性；没有删除原测试或改变物理/安全断言。
- README、architecture、本报告：安装、命令、职责、结果、限制。大型产物仍由原outputs/忽略规则排除，没有提交或推送。

## 固定环境与预处理

环境参数：target_mode=random、reward_mode=safe_dense、action_scale=.001m/轴、control_period=.05s、max_episode_steps=200、success_threshold=.001m、gravity_compensation=True。保留默认3–5mm局部随机可达目标、64次有界采样、home末端朝向、原IK、±5°/10mm实验限制、碰撞和总驱动力检查。nq=nv=nu=6；不存在夹爪动作。每个有效环境动作仍通过原执行器、25个2ms物理步产生运动；被安全预检拒绝的交互仍计作一个环境step，可能不推进物理时间。

SB3的向量输入预处理会转float32，但不会自动将米、弧度和毫米级误差标准化。本次用**固定仿射变换**，不做VecNormalize在线估计，也不缩放奖励：

- q和6个ctrl减home，除以5°（先转rad）。
- qvel除以30°/s（rad/s）。
- ee和command减home位置，除以.01m。
- goal-ee相对向量除以.01m。
- achieved_goal、desired_goal继续保留原SI米制float64，原compute_reward合同不变；SB3仅在网络张量入口将它们转float32。

因此主要27维字段量级接近1，目标相对位移在约.3–.5量级，另6个原SI goal特征保留。所有shape仍为27+3+3，总计33。没有裁剪观测或动作，没有把缩放后的位置送回FK、IK或原奖励接口。schema=2的全部offset、scale和dtype保存到每个模型旁的JSON，推理必须加载同一配置；无统计量需要继续更新。训练/评估共享相同变换，评估不拟合目标分布。

早期包装器将goal字段也缩放，SB3 GoalEnv检查器随后要求wrapper.compute_reward与原step逐项一致；该实现撤回。最终保留SI goal键，并将wrapper.compute_reward委托给原环境，原/包装后的SB3 check_env均通过。此修正在Pilot之前完成；早期smoke产物单独保留，不用于正式比较。

## 网络与训练参数

官方SAC MultiInputPolicy，标准CombinedExtractor展开Dict。Actor：33→256→256，ReLU，输出3维均值和3维log_std，通过标准SAC tanh分布输出Box[-1,1]动作。两个独立Critic：拼接33维状态和3维动作，各36→256→256→1/ReLU；标准target Critic和Adam由SB3维护。没有自定义网络学习规则。

固定参数：learning_rate=.0003、batch_size=256、buffer_size=100000、learning_starts=100、gamma=.99、tau=.005、train_freq=1、gradient_steps=1、ent_coef=auto、target_entropy=auto（3维动作对应-3）、target_update_interval=1、use_sde=False、policy net_arch=[256,256]、device=cpu、seed=2026、Torch线程1、确定性算法开启。

除Replay Buffer容量外采用SB3 SAC默认值。100k容量足够本次30k预算，避免默认1M容量的无用分配和存档；整个正式实验没有填满或环回buffer，也没有尝试参数网格。未修改环境以适应算法。

模型每5k交互返回一次learn并检查/评估，下一段reset_num_timesteps=False，保持同一进程模型、优化器、buffer和环境状态。checkpoint在learn返回、最后一次参数更新完成后保存，避免callback步数已增加而样本/更新尚未保存的歧义。每1k还在rollout结束时只读诊断，更新前后以日志scope和updates区分。NumPy诊断采样后恢复RNG；确定性Actor不消耗Torch RNG，测试已验证。

## 实验阶段和审计标准

Smoke最终配置：160次环境交互、60次更新，Actor和Critic权重均改变，官方DictReplayBuffer有160条，模型与压缩buffer均能重载；独立进程推理通过。它不要求短训练高成功率。最初使用gzip.GzipFile直接传入官方save_replay_buffer失败，改为SB3支持的BytesIO后无损压缩；没有自行实现buffer/序列化格式，失败日志保留。

Pilot：独立从头5,000步，4,900次更新，38.809步/秒（训练128.835s，独立验证32.201s）。完成44个Episode：3成功、12 workspace_limit、8 joint_limit、21超时；训练成功率6.818%，平均长度112.227，平均回报-122.417。安全失败20次/5000交互，最低单步奖励-201.628。独立验证10个目标全部超时，0成功、0安全失败、平均最终误差6.900mm、平均回报-68.518。近零动作占62.55%，已有保守/停留倾向。

Pilot参数有限、模型可重载、无碰撞/数值异常，没有基础采样或更新错误；诊断Critic loss约116.7–640.2，Q有明显乐观偏差，故不把“有限”称作价值估计正确。按用户允许的预算进行同配置独立30k实验，检查趋势而非修改奖励。

正式实验是**另一独立从头30,000步运行**，不是Pilot之后累计35,000步，也不是不完整的跨进程续训。两次初始网络及第5k checkpoint所有网络权重逐项相同，见independent_prefix_check.json。后续在正式进程内部继续各5k段。未根据测试集选择最优模型，最终评估固定使用最后30k模型。

基本保护：VecCheckNan检查输入/动作/奖励；每步检查已记录Actor/Critic loss与温度的有限性，每1k检查所有神经参数/Q/std；Q绝对值>1e6、Critic loss>1e8为粗略发散停止线（不是性能验收线），物理碰撞/数值失败立即停止；连续100次安全失败且平均长度≤2也停止。其他原环境安全终止逐Episode记录，不能因为早期随机探索安全失败较多就静默删除样本。进展、任务成功和价值高估持续独立审查。

## 奖励、折扣和诊断解释

原safe_dense完全保留：`-min(distance/.02,1)-201*is_safety_failure`，量纲为无量纲成本。未折扣“任何安全失败差于任何安全Episode”的全局排序证明，不能直接当成gamma=.99下任意两条轨迹的排序证明。

具体示例：若始终距目标4mm且安全超时200步，折扣回报约-17.3204（未折扣-40）；第一步同样距离就安全失败约-201.2；相同距离轨迹到最后一步再失败约-44.5223。它仍强烈惩罚立即失败，但失败时刻和状态分布改变了折扣贡献，且静止成本远小于探索时遇到的失败尖峰。

1mm距离改进只改变约.05单步奖励，而一次安全失败增加201成本。Pilot失败尖峰仅20/5000=.4%交互，却给Critic带来很大的平方误差；单个尚未拟合的失败样本约201²≈40401，因此数百级Critic loss本身不等于数值发散。Actor loss受soft Q/熵项影响，不能把它变得更负解释成成功率上升。

SAC的Q含未来熵奖励，不直接等于评估的无熵Episode回报；但明显过高的Q仍需报告。固定当前温度alpha、动作范围[-1,1]^3、原奖励≤0时，最大微分熵为3log2，理想soft Q的乐观上界约为`gamma*alpha*3log(2)/(1-gamma)`。可用这个只针对当前冻结策略/温度的尺度检查识别Critic乐观偏差，而不是把正Q一律解释成奖励为正。本轮没有据此追改奖励、增益、约束或网络参数。

## 评估方法和产物

训练验证使用10个固定种子10000–10009，每5k一次。最终对照独立使用100–199共100个目标，三个策略相同环境参数、相同真实位置目标；初始未训练网络也在此集合测量，未利用该集合训练、调参或挑选checkpoint。

随机策略用独立default_rng([2026,episode_seed])生成合法均匀动作；规则策略复用原命令滞后修正和0.25归一化向量范数上限；SAC deterministic=True，冻结权重/预处理。评估不调用learn，不写buffer，不动厂家配置。逐Episode记录目标、动作、奖励、长度、距离、终止/截断原因、物理步数和墙钟时间。

标准产物目录：outputs/rl/sac_reach/{smoke_ready_seed2026,pilot_seed2026,formal_seed2026}/。正式模型位于formal_seed2026/final/model.zip，旁边有config.json、observation_transform.json、state.json及官方replay_buffer.pkl.gz；初始网络initial_model.zip和每5k checkpoint也保留。training_episodes.jsonl、diagnostics.jsonl、sb3/progress.csv、sb3/progress.json、Monitor CSV分别保存Episode、诊断及官方训练日志。dependencies.json记录完整运行版本，config记录模型/环境文件哈希和实验源码哈希。

训练速度以真实learn墙钟测量（环境、网络更新和审计），独立验证和checkpoint另计，不用50ms仿真周期推算速度。所有步骤仅为仿真，没有推送GitHub。

## 复现

```bash
conda activate realman_rl
python -I -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -c requirements.txt -r requirements-rl.txt
python -I -m pip install --no-deps --no-build-isolation -e .
python -I scripts/train_sac_reach.py --run-dir outputs/rl/sac_reach/new_pilot --timesteps 5000 --label pilot
# 检查Pilot后才运行独立正式实验；已有目录不会被覆盖。
python -I scripts/train_sac_reach.py --run-dir outputs/rl/sac_reach/new_formal --timesteps 30000 --label formal
python -I scripts/evaluate_sac_reach.py --model outputs/rl/sac_reach/new_formal/final/model.zip --episodes 100 --seed-start 100
python -I -m unittest discover -s tests -v
```

压缩buffer可用gzip.open读取到BytesIO后交给官方model.load_replay_buffer；独立推理无需buffer，只需要model.zip及保存的同一预处理配置。当前CLI有意不提供跨进程resume，避免错误恢复环境轨迹、计数或随机状态。

## 正式30k训练结果

已完成30,000次交互、29,900次参数更新，buffer确实包含30,000条，Actor与两个Critic均实际更新。共187个完整Episode（另47步未完Episode不混入均值）：11成功、26 workspace_limit、8 joint_limit、5 orientation_limit、137 time_limit；ik_failure、collision、numerical_failure、unreachable_target、velocity_limit、drive_failure均0。这里joint_limit包含原±5°局部命令保护，不等于机械臂实际撞到硬限位。

完整Episode训练成功率5.8824%，平均长度160.1765，平均累计回报-93.2950，成功Episode平均55.4545步，平均末端误差6.5737mm。训练随机探索成功率不是最终deterministic评估成功率。

每5k独立验证均为0/10成功：5k、10k、15k、20k、25k、30k均未出现可靠Reach表现。没有用中途较好训练回报挑选模型。训练墙钟807.072s、37.1714交互步/秒；独立验证196.968s，含验证/存档的总时间约1004.456s（不含前置模型创建/依赖安装）。

诊断日志采样范围：Actor loss约[-188.612,-11.908]，最终-168.135；Critic loss约[23.794,640.173]，最终134.560；温度从初始1逐渐调整，最后0.249457。采样Q最大值峰值285.412，最后样本范围[128.361,217.794]，均值172.788；所有参数/Q/std有限，无已检测数值发散。这些是日志采样范围，不伪称记录了每次更新的全局极值。

当前温度下理想冻结soft Q乐观上界约51.354，实际Critic预测明显更高，说明价值估计有严重乐观偏差/尚未收敛。正Q并非奖励正值；Actor loss变得很负也不能证明任务改善。与5k比，训练平均回报改善但成功率没有改善，主要是安全失败频率降低、Episode更长。

安全惩罚尖峰39/30000=.13%交互，范围最低-201.628，普通奖励中位数约-.324923，P1约-.520343，P99约-.113751。仅显式失败惩罚合计39×201=7839，占完整Episode累计成本的重要部分；稀少大幅负奖励影响TD误差和探索倾向。未降低失败惩罚、归一化奖励、增大控制增益或放宽任何保护。

最终官方buffer成功重载：188个不同训练目标与独立100个目标无一逐项完全相同；状态检查确认30k步、29900更新和所有参数有限。证明保存在heldout_and_checkpoint_audit.json。最终网络结构原文保存于formal_seed2026/network.txt。

## 自动化与原功能保持

修改前177/177，99.908s。最终完整回归189/189（原177加新增12），123.590s。既有安全、物理、数值阈值没有改动。唯一旧测试改动是将“所有可选依赖仅vendor”的元数据假设扩展为分别精确核对vendor和rl组；没有删掉原vendor验证。新测试包括同一训练对象跨两个learn段后128步/128条buffer/28次更新的计数一致性，以及检查正常env.step中实际qpos只能由mj_step改变。

48个既有asset、simulation、envs及vendor受保护文件哈希不变；原requirements两份版本锁定文件不变。新增RL逻辑不进入envs/simulation，原177项功能继续工作。模型能够在不导入厂家SDK的独立进程加载、输出合法动作和执行原环境step；无需训练Replay Buffer即可推理。

## 独立100目标最终对照（种子100–199）

四组目标位置逐项完全一致，模型评估使用deterministic=True；SAC模型从磁盘在独立进程加载。基线先测、正式模型后测，环境参数相同，最终比较合并后再次检查目标一致性。训练步数、更新次数、buffer和所有模型文件字节均未被评估改变。

- 随机策略：4/100成功，平均累计回报-204.6891，平均63.46步，成功Episode平均29.25步，最终位置误差均值6.9695mm。失败为workspace19、joint71、orientation2，另4超时；IK、碰撞、数值失败均0。
- 原规则策略：100/100成功，平均累计回报-1.8829，平均及成功Episode均14.59步，最终误差均值0.8742mm；所有失败/超时类别均0。
- 未训练SAC初始网络：2/100成功，平均累计回报-159.8892，平均187.24步，成功Episode平均86步，最终误差均值9.5638mm。workspace失败52、超时46，其余类别均0。
- 最终30k SAC：**1/100成功，99超时，0安全失败**。平均累计回报-66.2082、平均折扣回报-28.1089、平均198.08步、最终误差均值6.7129mm（最小0.8894mm、最大12.1136mm）。唯一成功为seed122、8步；仅一个成功Episode的条件均值不具代表性。workspace_limit、joint_limit、ik_failure、orientation_limit、collision、numerical_failure、unreachable_target、velocity_limit、drive_failure均0，time_limit=99。

训练前后成功率从2%到1%，没有证据表明本轮提升了Reach能力；在此固定100目标集合上也未优于随机策略，明显没有达到规则策略的可靠性。平均回报从-159.889改善至-66.208，同时安全失败从52降到0，主要改善的是避免惩罚，并非完成目标。不能把此回报改善包装成训练成功。

平均评估Episode墙钟：随机约1.128s、规则.252s、初始SAC3.331s、最终SAC3.242s；最后一组约324.18秒。评估无更新，速度与训练37.17步/秒的统计范围不同。

## 具体性能诊断与未解决事项

最终策略并非简单输出全零：平均归一化动作范数约.4228，平均Episode中范数<.01的动作比例约9.845%。99个超时Episode平均累计命令路径约83.803mm（始终限制在原局部区域内），相邻动作点积为负的平均比例约46.16%，末50步动作范数均值约.3792。这说明存在无效指令往复，而不是有效接近目标。

为核实物理反应，额外只诊断第一条seed100（不改变100目标评分、不训练）：末50步实际末端XYZ范围约[.000669,.021415,.000421]mm，而命令参考范围约[.024497,.415744,.091753]mm；最终仍距目标4.7501mm并超时。原执行器/平滑抑制了大部分指令抖动，机械臂在错误位置附近有界停留，不能把“没有撞击或失控”当作“学会Reach”。动作往复、错误位置保持与价值高估同时存在，不能只凭动作均值判断策略是否静止。

已证实的事实：规则100/100证明当前目标可达；同一原物理环境下SAC仍99次超时；-201尖峰只占.13%训练交互，但其显式惩罚约占完整Episode总成本44.93%（Pilot约74.63%）；Critic明显高估，Actor优化的估计目标与真实任务回报脱节。不能把问题归因于IK无法达到这些目标，也不能说API测试通过就说明RL策略有效。

待独立验证的原因包括：较大失败成本与距离/熵尺度失配、稀少失败样本对TD误差的影响、bootstrapping乐观偏差、短预算下未学到命令滞后和稳定反馈。仅此单seed/单配置实验不能分离这些因素，也不能证明单纯多训练就能解决。

后续若研究改善，应另建明确标识的对照实验，优先审计TD目标/熵与实际回报尺度，再考虑一致缩放奖励与温度或依据折扣回报重新设计训练奖励；保留原环境、原安全终止及原始奖励日志作为基线。此次**没有实施这些修改，没有追加超过30k的正式训练，没有开展抓取训练**。

目前局限：没有得到可替代规则策略的可靠Reach神经控制器；只验证单训练seed和home附近的局部目标，未研究多seed稳定性、完整工作空间、实物模型/TCP标定、sim-to-real或真机部署。原环境接口/控制功能正常，SAC学习效果仍不满足任务性能要求。

## 结果文件索引

- `outputs/rl/sac_reach/formal_seed2026/final/model.zip`：最终30k网络，约3.3MiB。
- 同目录config.json、observation_transform.json、state.json、replay_buffer.pkl.gz（约7.5MiB）。
- `before_and_baselines.json`：相同100种子的随机/规则/未训练SAC逐Episode记录。
- `final_sac_evaluation.json`：最后30k模型独立推理逐Episode记录。
- `final_comparison.json`：四组配对目标、完整指标和动作；`experiment_summary.json`为小型汇总。
- `command_behavior.json`、`final_behavior_trace.json`、`policy_behavior_analysis.json`：命令往复和实际末端行为证据。
- `heldout_and_checkpoint_audit.json`、`final_artifact_integrity.json`、`independent_prefix_check.json`：目标隔离、模型/数据完整性及独立运行复现证明。
- `baseline_tests.log`、`all_tests.log`、安装/训练console、训练原始JSONL/CSV保留。`analyze_results.py`可从保存日志重新生成最终比较和小型摘要。
