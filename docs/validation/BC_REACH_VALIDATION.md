# 04-C：Reach行为克隆与网络可学习性

日期：2026-11-07。仅新增监督学习实验；机器人资产、simulation、vendor、ReachEnv、规则策略、04-A预处理、SAC代码及历史产物保持不变。没有新增SAC训练、预训练、微调或在线专家纠正。

结论：在新保留目标20000–20099上，BC **100/100成功**，未训练网络1/100，规则策略100/100。BC平均最终误差.86449mm、成功平均15.15步，没有安全失败或超时。当前33维输入/两层256网络具备表示本局部Reach任务有效策略的能力，但尚未证明停止/持续保持能力，也不能据此确定SAC失败的唯一原因。

## 实验协议与数据隔离

训练前保存`outputs/bc_reach/protocol.json`和基线哈希。全程使用现有realman_rl、Python 3.11、MuJoCo 3.17.0、Gymnasium 1.4.0、NumPy 2.4.6、SB3 2.9.2、PyTorch 2.14.0+cpu；未安装或升级依赖。

- 训练：seed 30000–30239，240个完整Episode，3587个样本。
- 监督验证：30240–30299，60个完整Episode，860个样本。按Episode划分，不把同一轨迹的相邻步拆到训练/验证。
- 开发闭环：40000–40019，仅模型冻结后检查；没有据此换网络或参数。
- 最终保留：20000–20099，未用于监督训练、选择epoch、网络选型或调整参数。
- 固定训练预算300个epoch；仅按监督验证动作MSE选择最优checkpoint。未根据最终成功率重新训练或选择checkpoint。

环境完全沿用04-A配置：random目标、safe_dense奖励、action_scale=.001m、control_period=.05s、H=200、success_threshold=.001m、重力补偿开启，默认terminate边界。未使用04-B reject_and_hold。每次正常step仍25个2ms物理步，局部10mm/home ±5°、IK/碰撞/姿态/速度/总驱动力保护均不变。

## 示范与分布

直接调用原`experiments.common.rule_action`，只读取公开Dict观测和动作尺度。规则为`(goal-actual)-(command-actual)=goal-command`，归一化动作向量范数上限.25。真实运动仅经过env.step；没有隐藏关节标签或独立IK标签。

300/300专家Episode成功，共4447个样本。每个NPZ样本包含原始Dict（float64）、04-A固定预处理Dict（observation为float32，两个goal仍SI float64）、33维网络特征、float32动作、实际/命令/目标世界位置、Episode/step/seed/split、返回奖励、成功及terminated/truncated、仿真时间。状态和动作位于step之前；step_success为该转移返回的成功标记，episode_success是整集结局。索引从0开始。清单保存逻辑数组摘要及NPZ文件SHA256，读取时复核。

实际接受目标是经过原IK与安全预检后的3–5mm球壳条件分布：

- 半径3.00010–4.99809mm，均值4.15992mm。
- 相对home的XYZ范围分别为[-4.71638,+4.79178]、[-4.65964,+4.54053]、[-4.65639,+4.81303]mm。
- 八方向象限Episode数：+++42，++-40，+-+48，+--38，-++35，-+-27，--+35，---35。不是完整机器人工作空间。
- 动作各分量范围约[-.24711,+.24676]、[-.24980,+.24743]、[-.24761,+.24829]，八个动作象限均覆盖。
- 动作范数全部约.25（浮点范围.249999985–.250000000），移动样本100%；近零停止（norm≤.01）和小动作（.01<norm<.1）均0。

**覆盖限制**：规则每步最多命令.25mm，而Episode在实际进入1mm成功区时结束，本数据没有运行到command恰好等于goal后的零增量等待阶段。因此不能用本实验声称学会精确停车或亚毫米持续保持。未伪造停止标签或在已结束Episode上继续step。停止/小动作误差返回count=0、MSE/MAE=null，不虚报为0误差。实际距目标≤1.5mm的近目标状态仍有600个，其中验证120个。

## 网络、预处理与优化

BC为独立PyTorch网络：33→256/ReLU→256/ReLU→3/tanh，共75267个可训练参数，输出原归一化动作。输入拼接顺序为`achieved_goal, desired_goal, observation`，与Gymnasium排序后的Dict及SB3 CombinedExtractor一致；所有字段在网络入口转为float32。完全复用04-A ReachTransform：关节/home中心、5°、30°/s和10mm尺度；两个goal保留SI米制，不额外构造goal-command特征，不改变环境单位。

这与[SAC Actor确定性均值通路](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/policies.html)的主要容量和计算一致。测试把BC的latent_pi/mu权重复制到安装版SB3 Actor，仅作离线数值对照，确定性输出逐元素一致；没有用这些权重创建、预训练或更新SAC模型。BC省略Gaussian log_std分支、分布采样、Critic、熵、自举和Replay Buffer，因此训练系统并不等同于SAC。安装版Actor/CombinedExtractor/分布源码归档于输出目录。

损失`L=mean((a_BC-a_expert)^2)`，三动作分量等权；使用[PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)：lr=.001、batch=256、betas=(.9,.999)、eps=1e-8、weight_decay=0。训练seed2026、CPU单线程、确定性算法，单独的固定seed minibatch洗牌；无数据增强、重采样、额外损失或在线采样。

300个epoch、每epoch15次更新，共4500次，训练约10.513s；按验证MSE选第267个epoch，另保留未训练初始网络和第300个epoch模型。训练只读取冻结NPZ；测试在拟合期间禁止构造ReachEnv，证明没有隐式闭环监督。所有权重有限且实际更新。

## 独立监督验证

以下误差均为原归一化动作单位，不是米或末端误差：

- 训练3587样本：MSE1.05342e-6、MAE.000815661，方向余弦均值.99998504，范数MAE.000879029。
- 验证860样本：MSE3.67032e-6、MAE.001463335，方向余弦均值.99995407、最小.99961366，范数MAE.001783825。
- 验证预测动作朝实际目标及朝命令目标的比例均100%。非零移动组包含全部860个样本，其误差与总误差相同。
- 近目标验证120样本：MSE4.00650e-6、MAE.001559983，预测范数均值.249031；没有通过输出零动作获得低MSE。
- 真实停止和小动作组均没有样本，相关泛化能力未验证。动作预测准不等于闭环成功，后续评估独立执行。

## 冻结后的闭环评估

模型先按监督验证选择并冻结，再检查开发种子40000–40019：BC20/20成功，最终误差均值.91708mm，平均15.1步。没有据此调整epoch、超参数或网络。最终使用全新的20000–20099；三组的物理目标逐项一致，与300个监督目标的实际位置也没有重叠，不仅是种子编号不同。

- **规则**：100/100成功，成功/全体平均15.10步，最终距离均值.873636mm，原始累计奖励均值-1.996152；无安全失败或超时。
- **未训练同结构网络**：1/100成功，成功Episode129步，全体平均187.67步，最终距离均值9.906651mm，累计奖励均值-180.262524；61次workspace_limit、38次time_limit。没有joint_limit、IK、碰撞、姿态、速度、驱动力或数值故障。此处61个is_safety_failure是任务命令球边界终止，不能称为61次真实物理碰撞。
- **BC**：100/100成功，成功/全体平均15.15步，最终距离均值.864490mm，累计奖励均值-1.998940；所有失败类别、超时均0。

动作方向分别相对步前`goal-actual`和`goal-command`计算，近零分母样本不虚构方向；相邻反向比例分母为每个Episode内L-1个相邻动作对，Episode之间不拼接。

- 规则：两种朝目标比例均100%，反向0；累计命令/实际路径.377500/.336594m，实际/命令比.891639。
- 未训练：朝实际目标15.0477%、朝命令目标14.4989%，反向0；命令/实际路径.964469/.953721m，比.988856。它主要持续朝错误方向漂移，而非原SAC式往复；高路径跟踪比例本身不代表任务成功。
- BC：两种朝目标比例均100%，反向0；命令/实际路径.378410/.337562m，比.892053，与规则接近。结束前仍有正常执行器响应滞后，但没有因此失去任务成功。

BC没有闭环失败案例可作失败归因；所有100条完整轨迹均保留，而非只存成功摘要。示例seed20000/20001/20002：BC分别16/15/11步成功，末端误差.968680/.786185/.984280mm。未训练网络在相同目标下分别200步超时、167步命令越界、200步超时，最终误差12.575528/13.032071/9.597905mm；唯一偶然成功为seed20066。没有在测试时调用专家修正神经网络，也没有通过qpos瞬移或改变成功阈值取得结果。

闭环墙钟：规则25.486s，未训练307.016s，BC25.213s；BC的env.step均值约15.987ms、含reset/记录约60.09步/秒。仿真控制周期仍50ms，不以它推算真实执行速度。本轮没有做性能优化。

## 与SAC历史结果及结论边界

04-A原30k仍为1/100、6.712937mm；04-B同预算A10k/B10k仍为0/100、7.458405mm及1/100、3.228512mm。这些冻结历史使用100–199目标，本次BC三组使用20000–20099，**不是同目标集合或同监督信息预算的SAC/BC公平排名**。没有重新评估或训练SAC以挑选有利结果。

BC在完全未参加监督拟合的局部随机目标上达到100%，且离线误差、朝向目标比例和实际执行一致，支持当前输入信息和确定性均值通路容量能够表示有效的局部策略；单纯“网络太小/无法表达”不足以解释现有SAC结果。BC拥有直接专家动作标签，而SAC依赖奖励、探索、Critic和熵优化，数据分布与学习问题不同。不能证明奖励或探索是唯一原因，也不能证明SAC算法实现错误。

下一步应优先受控研究价值估计/终止样本学习、奖励与熵的相对尺度、探索和动作时间一致性；先保持此BC作为可重复的表达能力基线，而非扩大网络或盲目追加SAC步数。本轮不执行这些研究，也不做SAC预训练、DAgger或其他在线纠正。

已知限制：单个网络初始化seed、同一小范围目标分布、同一精确仿真模型；没有扰动恢复、局部范围外泛化、停止/持续保持、真实机械臂或夹爪学习验证。100/100有限样本成功不等于所有目标或真实系统100%可靠。

## 回归、完整性与产物

基线原207/207通过（113.931s）；新增15/15通过；完整回归**222/222通过**（129.480s），未改原测试或断言。数据及短训练重复运行逐数组/逐预测一致；复制权重后的SB3确定性Actor输出逐元素一致。官方ReachEnv安全/API测试随原207项继续运行。单元测试只检验数据/网络/物理契约，没有把BC高成功率写成必须达到的断言。

新增测试还覆盖：纯监督训练期间禁止构造环境、参数确实更新且有限、验证MSE选checkpoint、权重保存加载与固定seed复现、标量/批量预测、无专家fallback、逐物理步前qpos未被改写、25步/50ms一次推进、无厂家SDK。首次测试运行中的NaN反例曾误改共享测试数组的view，已将反例改为独立copy；这是测试隔离问题，没有影响正式采集文件或训练数据。

283项基线文件SHA256一致（旧src、assets、原测试及outputs/rl历史实验）；全部4447标签逐条复核，差异0；环境配置和预处理与冻结04-A逐项相同。保存的初始/最优权重指纹和训练源文件哈希均一致，最终实际目标重叠0。`pip check`、`git diff --check`通过。新脚本从/tmp以绝对路径、python -I读取帮助成功；BC在独立解释器从/tmp加载并实际执行一次25物理步/50ms控制成功，无需数据文件或专家。

负例：评估请求训练seed30000明确返回1且不创建输出目录；尝试覆盖专家数据目录明确返回1，不改已有文件。所有大型NPZ和模型均被Git忽略。

输出目录`outputs/bc_reach/`包含：protocol与baseline_manifest；demonstrations/{manifest.json,demonstrations.npz}；fit_seed2026/{config.json,dependencies.json,history.json,summary.json,untrained_bc.pt,last_epoch_bc.pt,best_bc.pt}；development_bc.json；final_evaluation/{rule.json,untrained.json,bc.json,comparison.json}；baseline_tests/new_tests/all_tests日志；integrity_audit.json、independent_inference.json和experiment_summary.json。history保存全部300个epoch，最优模型来自267（4005次更新），总训练4500次更新，两者不混为同一状态。

没有新的Viewer验证要求，本轮无图形运行；未修改依赖或其他Conda环境，未提交或推送GitHub。完成后停止，不启动新的SAC或夹爪训练。

## 文件与可复现命令

新增文件职责：

- `experiments/bc_reach.py`：数据采集/校验、固定特征拼接、独立BC网络/保存加载、纯监督拟合、分组误差及标准env.step闭环评估。simulation和envs不依赖它。
- `scripts/collect_reach_demonstrations.py`：专家采集CLI，默认240训练/60验证。
- `scripts/train_bc_reach.py`：只读NPZ、Adam监督训练与验证checkpoint选择。
- `scripts/evaluate_bc_reach.py`：同目标规则/未训练/BC配对评估，禁止与监督种子重叠；神经分支不调用规则函数。
- `tests/test_bc_reach.py`：数据复现、标签、SB3输入/均值通路、保存加载、物理推进、无专家兜底/无SDK及有限数值等验证；不要求小样本BC达到某个成功率。
- 本报告、README及architecture：实验边界、实际结果、接口和运行方法。

大文件仅放在Git忽略的`outputs/bc_reach/`。已有目录拒绝覆盖；以下新路径用于复现，不会覆盖本轮结果：

```bash
conda activate realman_rl
python -I scripts/collect_reach_demonstrations.py --out outputs/bc_reach/new_demonstrations
python -I scripts/train_bc_reach.py --data outputs/bc_reach/new_demonstrations/demonstrations.npz --out outputs/bc_reach/new_fit
python -I scripts/evaluate_bc_reach.py --model outputs/bc_reach/new_fit/best_bc.pt --out outputs/bc_reach/new_evaluation
python -I -m unittest discover -s tests -v
```

独立推理只需BC文件、现有editable包及依赖，不需NPZ、优化器、专家、SAC模型或Replay：

```python
from realman_rl_simtoreal.experiments.bc_reach import load_policy
from realman_rl_simtoreal.envs.reach_env import ReachEnv
policy, config = load_policy("outputs/bc_reach/fit_seed2026/best_bc.pt")
with ReachEnv(**config["environment"]) as env:
    obs, info = env.reset(seed=20000)
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

文件内明确标记behavior_cloning_not_SAC，使用weights_only加载；模型不是已完成强化学习的SAC策略。当前模型只覆盖局部Reach，不用于夹爪、真机或工作空间外控制。
