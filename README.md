# realman_rl_simtoreal

基于 MuJoCo 的 RealMan **RML63II-6F-V** 机械臂强化学习仿真项目，长期方向为 Reach → Grasp → 视觉 → 真机部署。当前采用 Python 3.11、Gymnasium、Stable-Baselines3 和 PyTorch；代码组织与算法实验分别维护。

## 已实现功能

- 六轴动力学、关节与末端控制、FK / DLS IK、厂家离线 FK / IK 对照。
- EG2-4C2 夹爪模型、同步控制和接触 / 抬升 / 释放案例。
- 固定及随机目标 ReachEnv，PPO、SAC 训练、独立评估、checkpoint 和 TensorBoard。
- 独立 BC 示范采集、训练和推理；保留历史 BC→SAC Actor 初始化、边界对照与 Q/TD 诊断。
- 自动化回归测试、环境探针及只读训练曲线绘图。

## 夹取与抬升

`GraspLiftEnv` 将机械臂和 EG2-4C2 夹爪接入同一个物理控制周期。默认任务使用 20 mm、10 g 的自由方块，物体位置在支撑面内随机偏移最多 2 mm；机械臂从距夹取位置 20 mm 处接近，缓慢闭合夹爪，沿支撑面法线抬升 30 mm，再稳定保持。支撑面约倾斜 30°，因此离台高度和世界竖直抬升量分别记录。演示采用带横向对中反馈的分阶段控制器，对中每周期最多 0.05 mm、累计最多 2 mm。

成功要求左右指垫法向力均大于 5 mN、支撑力小于 1 mN、离台间隙至少 28 mm、横向位置误差小于 3 mm、物体线速度小于 2 mm/s、角速度小于 0.1 rad/s，并连续保持 2 秒且相对指垫滑移小于 2 mm。物体由接触与摩擦带动；正常 step 只写控制量，位置初始化仅发生在 reset。每步共享 50 个 1 ms 物理步，继续检查力矩、速度、mimic 残差和分类碰撞；该局部任务的关节命令范围为 home±25°。

在 `realman_rl` 环境运行夹取抬升，输出文件须选择新路径：

```bash
conda activate realman_rl
python -I scripts/demos/run_grasp_lift.py --episodes 1 --seed-start 2027 \
  --json /tmp/realman_pick_lift.json
```

追加 `--viewer` 可显示仿真，`--record-physics` 保存逐 1 ms 的物体运动和接触力。`--no-close` 与 `--no-lift` 是负对照；未完成夹取抬升会返回状态码 1，同时保存失败原因。`--mass`、`--friction`、`--object-radius` 和 `--lift-height` 支持实验参数。当前参数针对这一局部方块场景验证，夹爪模型与摩擦参数仍为未标定仿真值。

RL 接口为 4 维 `Box(-1,1)`：前三维是世界 XYZ 增量，每轴最多 1 mm；第四维为绝对开口，-1 闭合、+1 达到模型最大开口约 66.2 mm。41 维 float32 观测包含机械臂状态、物体相对位置及速度、开口、双侧/支撑接触力和保持进度，可直接交给 SB3 `PPO("MlpPolicy", env)`。接口测试覆盖真实 Rollout、PPO 更新、模型保存和重载；本文演示结果来自分阶段控制器。

独立测试种子 230000–230019 的 **20/20 个随机位置全部夹取抬升成功**，平均离台间隙 29.206 mm、世界竖直抬升 25.243 mm，最短连续保持 2.032 秒，零安全终止。447000 次物理积分的力矩、速度和时间记录核对通过，mimic 残差峰值为 0.0002174 rad，低于原 0.001 rad 保护阈值。全套 380 项回归通过；将抓取任务改为 1 ms 后，重新通过 16 项抓取专项和 24 项接触回归。见[逐回合结果](outputs/grasp_lift/validation_1ms_20/evaluation.json)、[核对记录](outputs/grasp_lift/validation_1ms_20/verification.json)及[实测动图](outputs/grasp_lift/pick_lift_1ms.gif)。

负对照中，不闭合和不抬升均在 700 步超时、保持计时为零；摩擦降为 0.05 时触发 mimic 保护，30 g 方块未达到高度与保持条件、在 700 步超时。失败记录分别保存在 [no_close](outputs/grasp_lift/controls_1ms/no_close.json)、[no_lift](outputs/grasp_lift/controls_1ms/no_lift.json)、[low_friction](outputs/grasp_lift/controls_1ms/low_friction.json)和[heavy](outputs/grasp_lift/controls_1ms/heavy.json)。开发阶段的 2 ms 失败案例与控制时序诊断仍保存在 `outputs/grasp_lift/`。

实现入口为 [GraspLiftEnv](src/realman_rl_simtoreal/envs/grasp_lift_env.py)、[任务配置](src/realman_rl_simtoreal/tasks/grasp_lift.py)、[夹取控制器](src/realman_rl_simtoreal/simulation/grasp_controller.py)与[回合评估](src/realman_rl_simtoreal/rl/grasp_lift.py)。

## 随机起点与连续多目标 Reach

任务 `random_start_multi_reach_20cm` 每回合从一个经过 IK 和碰撞预检的随机姿态开始，初始速度为零，并连续执行 5 个目标。起点和目标均位于以 home 为中心、半径 20 cm 的可达区域，连续目标的计划间距至少为 10 mm。完整目标计划在 reset 时用 Gymnasium RNG 生成，同一 seed 对应相同的起点和目标计划。

实际末端进入当前目标的 1 mm 范围后即切换下一目标，关节位置、速度、控制量与仿真时间保持连续。每个目标的时间上限为 1000 步，一条 5 目标轨迹最多 5000 步；任一目标失败或超时会结束当前轨迹。运行时 IK 求解预算为 600 次迭代，姿态跟踪容差为 5°。起点姿态的初始化发生在回合 reset 中。

该任务启用 `track_issued_command`：笛卡尔命令跟随实际发出的限速参考点，避免累计尚未执行的位移；增量 IK 失败时最多尝试 6 次缩短步长。运行时 DLS 阻尼为 0.001、初值正则权重为 0.000001，求解结果仍须通过固定姿态、关节限位、速率及碰撞检查。目标采样分布保持不变。

观测使用 `goal_relative_v3`。到点的那个 transition 按已到达的目标计算进度与成功奖励，返回的 observation 包含下一目标；`info.transition_desired_goal` 保存本步奖励所用的目标，`compute_reward` 和批量重算读取该上下文。`info.is_goal_success` 表示单个目标到达，`info.is_success` 表示整条轨迹完成。

本次从 20 cm 单目标 PPO 权重开始，先训练 32768 步，再修复连续运动中的命令累计及 IK 数值问题，追加 16384 步，共新增 49152 步多目标训练。最终开发集 180000–180003 完成 4/4 条轨迹、20/20 个目标，零安全终止。actor 和 critic 均有更新，训练配置、来源权重和当前源码资源核对通过，全部 364 项回归测试通过。见[训练记录](outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/summary.json)与[训练核对](outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/training_verification.json)。

冻结最终模型后，独立测试 191000–191019 完成 **12/20 条完整轨迹**，成功 **72/100 个计划目标**。实际尝试 80 个目标，单目标成功率为 72/80（90%）；提前结束后未尝试的 20 个目标计入计划目标完成率的分母。8 条失败轨迹中，7 条因求解出的关节目标触及限位而停止，1 条超时，碰撞、数值失败及 IK 失败均为 0。当前仍未达到全部随机连续轨迹成功。

72 个成功目标平均实际误差为 0.794 mm，最大为 0.996 mm。20 个随机起点距 home 6.39–19.74 cm，100 个计划目标距 home 4.83–19.94 cm，覆盖八个空间象限。全部 60 次目标切换均保留非零速度，切换前后的关节位置、速度、控制量和时间逐项相同。原始数据见[独立评估](outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/heldout_191000_191019.json)、[逐目标分析](outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/heldout_191000_191019_analysis.json)、[连续性核对](outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/heldout_verification.json)与[失败明细](outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/heldout_failures.json)。

使用本次模型继续训练，可在 `realman_rl` 环境运行以下命令，输出目录需为新目录。

```bash
conda activate realman_rl
python -I scripts/train/train_ppo_reach.py --task random_start_multi_reach_20cm \
  --init-model outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/final/model.zip \
  --timesteps 16384 --seed 2027 --eval-every 8192 --eval-episodes 4 \
  --eval-seed-start 180000 --checkpoint-every 8192 --tensorboard \
  --run-dir outputs/rl/ppo_reach/random_start_multi_20cm_next
```

独立评估中的 `--episodes 20` 表示 20 条轨迹，共计划 100 个目标。`sequence_success_rate` 为整条轨迹完成率，`goal_success_rate` 为成功目标数除以实际尝试目标数，`goal_completion_rate` 为成功目标数除以计划目标数。逐回合 `goal_results` 记录每个目标的动作、位置、结束原因及起止物理状态，成功目标的 `end_state` 与下一目标的 `start_state` 可逐项核对。

```bash
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/random_start_multi_20cm_seed2027_continue/final/model.zip \
  --episodes 20 --seed-start 191000 --json /tmp/realman_multi_goal_evaluation.json
```

训练日志增加 `training_goals.jsonl`，TensorBoard 记录单目标事件与轨迹完成率。保存配置通过 `envs.factory.make_reach_env` 自动还原随机起点、多目标环境。分析脚本将轨迹拆成各目标段，距离分组统计每段实际行程，目标区域半径单独统计。

## 20 cm 随机 Reach 续训

任务 `random_reach_20cm` 在以 home 为中心、半径 20 cm 的球体内采样目标，最小采样半径为 5 mm，采样按体积均匀分布，再经过固定末端姿态的 IK、模型关节限位和碰撞预检。每回合从 home 开始，实际末端误差进入 1 mm 即成功，回合上限 600 步。实际可采到的位置受机械臂当前 home 和固定姿态的可达性约束。

该配置使用 1 mm 的每轴动作尺度、22 cm 命令范围和 5° 实际姿态跟踪容差。控制器沿笛卡尔路径求解满足每周期关节速率限制的参考点，关节命令速率为 10°/s，真实速度上限为 30°/s，执行过程检查驱动力、限位和碰撞。`goal_relative_v3` 将目标与命令、命令与实际位置的两个三维差值变换为 `asinh(delta / 0.001)`，GoalEnv 的目标字段以米为单位。

本次由种子 2027 的 5 mm PPO 权重开始迁移，先完成 32768 步大范围训练，再使用上述观测和参考轨迹追加 16384 步，20 cm 任务共新增 49152 步。两阶段权重哈希与父模型记录相符，actor 和 critic 均有更新。最终模型在开发集 130000–130011 上成功 12/12；独立测试 140000–140099 成功 **97/100**，另有 2 次超时、1 次 IK 失败，碰撞与数值失败为 0。

独立目标的实际半径为 3.19–19.99 cm，八个空间象限均有样本，15–20 cm 分组成功 52/55。100 回合平均最终误差为 2.692 mm；97 个成功回合的平均误差为 0.773 mm，平均 241.47 步到达。三个失败目标均位于 +X 方向，IK 失败记录为迭代次数耗尽，两个超时回合在 600 步结束时的误差为 53.68、45.21 mm。完整数据见[独立评估](outputs/rl/ppo_reach/random_20cm_seed2027_continue/heldout_140000_140099.json)、[方向与距离分组](outputs/rl/ppo_reach/random_20cm_seed2027_continue/heldout_140000_140099_analysis.json)及[失败复核](outputs/rl/ppo_reach/random_20cm_seed2027_continue/heldout_failures.json)。全部 351 项回归测试通过。

在 `realman_rl` 环境中使用已训练权重继续训练，可执行以下命令。`--run-dir` 需选择新的目录。

```bash
conda activate realman_rl
python -I scripts/train/train_ppo_reach.py --task random_reach_20cm \
  --init-model outputs/rl/ppo_reach/random_20cm_seed2027_continue/final/model.zip \
  --timesteps 16384 --seed 2027 --eval-every 4096 --eval-episodes 12 \
  --eval-seed-start 130000 --checkpoint-every 4096 --tensorboard \
  --run-dir outputs/rl/ppo_reach/random_20cm_seed2027_next
```

`--init-model` 加载 PPO 的 actor、critic 和 log-std 权重，为本次任务创建优化器、RolloutBuffer 和计数器。配置记录来源模型哈希、来源训练步数和观测变换；`--timesteps` 表示本次新增训练步数。训练完成后，可独立复测 100 个新目标，`--json` 需使用新路径。

```bash
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/random_20cm_seed2027_continue/final/model.zip \
  --episodes 100 --seed-start 140000 --json /tmp/realman_20cm_evaluation.json
```

## 5 mm 三维随机 Reach 验证

在 `realman_rl` 环境中，`random_reach` 每回合从距 home 3–5 mm 的球壳中采样一个通过逆运动学与安全预检的目标。实际末端进入目标的 1 mm 球内即成功，每回合上限 200 步。

种子 2026 的 PPO 从头训练 16384 步后，在旧模型的同一组 100 个目标（92000–92099）上从 1/100 提高到 100/100，另一组新目标（110000–110099）也达到 100/100。两组合计平均 12.44 步，终止时平均实际误差 0.857 mm，零超时、零安全终止。冻结同一策略单独复测 ±X 5 mm，两个方向均为 18 步。以上为 MuJoCo 仿真结果。

补充的 1000 个新目标（120000–120999）全部成功，平均 12.138 步，终止时平均实际误差 0.859 mm，零超时、零安全终止。目标覆盖全部八个空间象限，每个象限有 105–141 个样本且全部成功，两个距离分组 3–4 mm 与 4–5 mm 也全部成功。原始位置、动作、奖励和结束原因见[逐回合结果](outputs/rl/ppo_reach/random_goal_relative_seed2026/heldout_120000_120999.json)，分方向统计见[目标覆盖分析](outputs/rl/ppo_reach/random_goal_relative_seed2026/heldout_120000_120999_analysis.json)。

当前代码从头训练的种子 2027、2028 模型各完成 16384 步，在相同 1000 个目标上分别成功 1000/1000，零超时、零安全终止；平均分别为 12.157、12.106 步，终止时平均实际误差分别为 0.862、0.859 mm。三个训练种子共计 3000 回合评估，八个象限均全部成功。新模型、环境资源哈希、配置与 TensorBoard 记录保存在各自的训练目录，评估原始记录见[种子 2027](outputs/rl/ppo_reach/random_goal_relative_seed2027/heldout_120000_120999.json)和[种子 2028](outputs/rl/ppo_reach/random_goal_relative_seed2028/heldout_120000_120999.json)。

种子 2026 也已在当前源码下重新训练 16384 步，初始、4096、8192、12288、16384 步五个检查点的网络参数均与历史运行逐项一致。此次重跑记录的环境源码和机器人资源哈希与当前工作区匹配，另用 121000–121019 的 20 个新目标复测，成功 20/20。记录见[复现训练](outputs/rl/ppo_reach/random_goal_relative_seed2026_verified/summary.json)和[复现模型独立评估](outputs/rl/ppo_reach/random_goal_relative_seed2026_verified/heldout_121000_121019.json)。随机 Reach 回归测试覆盖默认训练配置、真实物理步进、模型重载和独立 CLI 推理，共 17 项通过。

用工作区中按当前代码训练的种子 2027 模型复测这 1000 个目标：

```bash
conda activate realman_rl
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/random_goal_relative_seed2027/final/model.zip \
  --episodes 1000 --seed-start 120000 --json /tmp/realman_random_reach_eval.json
```

从干净检出重新训练可执行以下命令。训练和评估均须使用新的输出位置。评估自己训练的模型时，将上面的 `--model` 换成对应目录下的 `final/model.zip`。

```bash
python -I scripts/train/train_ppo_reach.py --task random_reach \
  --timesteps 16384 --seed 2027 --eval-every 4096 --eval-episodes 20 \
  --checkpoint-every 4096 --run-dir outputs/rl/ppo_reach/random_goal_relative_retrain
```

变目标 PPO 默认使用 `goal_relative_v1` 观测与 `reach_progress_v1` 奖励。策略每步获得两个世界坐标系下的三维差值，即「目标位置 − 已发命令位置」和「已发命令位置 − 实际末端位置」。策略输入按 1 mm 缩放，GoalEnv 的目标字段、逆运动学和奖励计算使用的位置均以米为单位。训练从随机权重开始，动作由 PPO 产生。评估自动加载模型保存的观测和奖励配置。

## 历史实验

用户已将12维 `fetch_like_v1` 随机Reach从头训练到131072步：完成256个Rollout，独立100目标成功1%、99次超时、0次安全失败，平均最终距离4.571mm。固定20目标开发集从32768步起连续13个检查点均0成功；99次超时末端集中在Home相对位移约[+0.61,-1.20,+2.08]mm，首动作仍固定偏向-X,-Y,+Z。实际Actor/Critic输入均为12维，增加预算尚未带来可靠的目标条件Reach。训练曲线、落点图、方向探针及预算判断见[Fetch观测验证报告](docs/validation/REACH_FETCH_OBSERVATION_VALIDATION.md)。

此前同seed的12维16384步Pilot独立100目标成功3%，旧33维随机Pilot成功4%；三次独立测试使用不同目标，不能把1%/3%/4%的差值直接作为配对对照结论。长训练在16384步时的参数与12维短实验final完全一致，可以比较其后在同一开发集上的变化。

奖励方向诊断已完成，未发现旧reach_shaped_v1的明确实现错误。新增显式实验模式reach_progress_v1，
保留原默认奖励；339项测试通过，开发只新增一次512步PPO Smoke。用户随后完成16384步±X Pilot，
最终两目标均超时（误差2.531/7.335mm），方向课程尚未通过，暂不进入随机任务；尚无等条件新旧奖励优劣结论。
实际数值、折扣往返与安全偏好边界见[奖励诊断报告](docs/validation/REACH_REWARD_OPTIMIZATION_VALIDATION.md)。

历史33维纯PPO两方向Pilot：+X19步成功、误差0.929mm；-X200步超时、误差10.241mm，训练-X57集均未成功，课程尚未通过，不进入六方向。旧固定目标Pilot成功。两方向±X5mm均通过原保护；六方向5mm中的±Y未通过home±5°预检，该配置仍明确拒绝启动。全部旧产物保留；历史逐目标结果见[目标条件验证](docs/validation/PPO_GOAL_CONDITIONED_VALIDATION.md)、[PPO验证](docs/validation/PPO_REACH_VALIDATION.md)和[实验索引](docs/experiments/README.md)。

## 安装环境

在已有环境中使用 editable 安装；已经安装的环境无需因本次目录迁移重新安装：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I -m pip install --no-build-isolation -e .
python -I -m pip check
```

基础依赖锁在 `requirements.txt`；可选训练依赖锁在 `requirements-rl.txt`，厂家离线 SDK 锁在 `requirements-vendor.txt`。需要补装时分别使用：

```bash
python -I -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -c requirements.txt -r requirements-rl.txt
python -I -m pip install -r requirements-vendor.txt
```

训练依赖安装命令显式指定官方 PyTorch CPU 索引，并沿用基础版本约束。普通仿真不需要 SDK 或训练框架。`-I` 避免继承 ROS / PYTHONPATH 干扰；不需要修改 `sys.path`。资源按包所在仓库定位，仅支持保留源码与 `assets/` 的 editable 使用方式，普通 wheel 脱离仓库安装尚不保证资源可用。

## 快速开始

以下命令均从仓库根目录执行；Viewer 需要图形桌面，控制演示不加 `--viewer` 时可无界面运行：

```bash
python -I scripts/models/inspect_model.py
python -I scripts/demos/view_robot.py
python -I scripts/demos/control_joints.py --viewer
python -I scripts/demos/control_ee.py --viewer
python -I scripts/demos/demo_gripper.py --viewer
python -I scripts/diagnostics/smoke_reach_env.py --check
python -I -m unittest discover -s tests -v
```

接触案例入口为 `scripts/demos/run_grasp_contact_cases.py`；环境与奖励探针入口为 `scripts/diagnostics/probe_reach_env.py`。前者是演示，后者不训练网络。其他脚本参数见 `--help`；三个 `scripts/models/build_*.py` 沿用无参数生成器接口。

资源目录内的 `SOURCE.md` 保留原版本来源记录，其中旧生成路径属于历史结构。需要重新生成模型时，当前入口为：

```bash
python -I scripts/models/build_dynamics_model.py
python -I scripts/models/build_gripper_model.py
python -I scripts/models/build_grasp_scene.py
```

## PPO Reach 训练

以下为手动训练示例。每次换一个未使用的 `--run-dir`，保留已有结果：

```bash
python -I scripts/train/train_ppo_reach.py \
  --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/fixed_seed2026_new \
  --eval-every 2048 --eval-episodes 1 --eval-seed-start 87000 \
  --checkpoint-every 2048 --tensorboard
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach --host 127.0.0.1 --port 6006
```

PPO 保留官方 512 步 Rollout、64 样本 minibatch、10 个 epoch；评估和保存发生在完整更新之后，因此实际步数可能超过请求下界。原始高斯动作与环境 Box 裁剪动作分别记录，动作裁剪率与 PPO `clip_fraction` 含义不同。固定目标更换 seed 不等于泛化验证。

### 三维随机目标Pilot（04-F2）

开发阶段完成200个随机目标检查及短测试；用户随后已手动完成16384步Pilot。最终100目标成功4%、平均误差6.197mm，开发集未呈持续改善，当前尚未学会可靠的三维Reach。结果、方向偏置及训练预算分析见[验证报告](docs/validation/PPO_RANDOM_REACH_VALIDATION.md)。沿用同一ReachEnv，每个Episode在reset采样home周围3–5mm的可达目标，Episode内保持固定。全新PPO随机初始化，不加载固定目标、SAC或BC权重。以下保留Pilot复现参数；输出目录必须未使用，不能覆盖已有实验：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I scripts/train/train_ppo_reach.py \
  --task random_reach --observation-mode legacy --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/random_seed2026_pilot_04f2 \
  --eval-every 4096 --eval-episodes 20 --eval-seed-start 87000 \
  --checkpoint-every 4096 --tensorboard
```

开发期checkpoint始终用87000–87019这20个目标比较。最终模型冻结后再评估90000–90099，不据此反复选模型或调参。监控、查看结果、最终100目标评估、导出曲线和分方向统计：

```bash
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach/random_seed2026_pilot_04f2/sb3 --host 127.0.0.1 --port 6006
python -I -m json.tool outputs/rl/ppo_reach/random_seed2026_pilot_04f2/summary.json
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/random_seed2026_pilot_04f2/checkpoints/step_016384/model.zip \
  --episodes 100 --seed-start 90000 \
  --json outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final_test_90000_90099.json
python -I scripts/plots/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/random_seed2026_pilot_04f2
python -I scripts/evaluate/analyze_reach_results.py \
  --evaluation outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final_test_90000_90099.json \
  --json outputs/rl/ppo_reach/random_seed2026_pilot_04f2/final_test_direction_analysis.json
```

提前停止时先检查summary，再选择实际存在的checkpoint。评估JSON、绘图目录及分析JSON均不得覆盖已有结果。方向分析同时输出六个可重叠符号组、八个互斥象限、3–4/4–5mm半径组及每组样本量；不足10个标记小样本。新PPO评估JSON额外保留原始动作和步前实际位置，可计算裁剪角度、动作是否朝实际目标和+X偏置；历史JSON缺少字段时明确报告不可计算。环境、网络、损失及超参数不变。详情见 [PPO随机Reach验证](docs/validation/PPO_RANDOM_REACH_VALIDATION.md)。

### 两方向课程准备（04-F3）

同一ReachEnv新增离散候选模式：每次reset用Gymnasium RNG选择+X或-X 5mm，Episode内固定；两个目标均在初始化时通过原IK、限位、姿态及碰撞预检。动作仍为三维增量，观测仍为原33维，没有额外方向标签、BC或专家指导。PPO从随机权重开始，奖励及全部超参数不变。开发仅执行一次512步Smoke，用户随后已完成正式Pilot；以下保留复现命令，已有目录不得覆盖：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I scripts/train/train_ppo_reach.py \
  --task axis_pair_reach --observation-mode legacy --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3 \
  --eval-every 2048 --eval-episodes 2 --eval-seed-start 87000 \
  --checkpoint-every 2048 --tensorboard
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/sb3 --host 127.0.0.1 --port 6006
```

评估会通过合法reset选项逐一枚举两个候选，保证每次恰好+X、-X各一次；种子只是reset标签，不是不同随机目标。离散任务省略`--eval-episodes`/`--episodes`时自动覆盖所有候选，显式要求100次重复会报错。随机任务的多seed评估语义不变。

```bash
python -I -m json.tool outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/summary.json
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/checkpoints/step_016384/model.zip \
  --episodes 2 --json outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/distinct_goal_evaluation.json
python -I scripts/evaluate/analyze_reach_results.py \
  --evaluation outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/distinct_goal_evaluation.json \
  --json outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3/distinct_goal_analysis.json
python -I scripts/plots/plot_ppo_training.py --run-dir outputs/rl/ppo_reach/axis_pair_seed2026_pilot_04f3
```

JSON、终端和TensorBoard分别报告±X结果、误差、步数、首动作、结束原因、实际路径和动作方向比例；PNG增加两方向曲线及训练成功率/保护分类。所有输出目录须为新位置，不覆盖原产物。两方向均稳定成功后才考虑六方向；`axis_six_reach`5mm目前会拒绝±Y，不能直接启动。显式3mm六方向仅用于已完成的合法reset敏感性诊断，没有偷偷替换训练配置。证据、文件说明和课程边界见[04-F3验证报告](docs/validation/PPO_GOAL_CONDITIONED_VALIDATION.md)。尚未实现`ppo_transfer`或严格跨进程续训。

纯 SAC 的手动入口仍保留：

```bash
python -I scripts/train/train_sac_reach.py \
  --task fixed_reach --reward-mode safe_dense --action-scale 0.001 \
  --timesteps 5000 --seed 2026 --init-mode scratch \
  --run-dir outputs/rl/pure_reach/fixed_safe_seed2026_new \
  --eval-at-start --eval-every 1000 --checkpoint-every 1000 \
  --eval-episodes 1 --eval-seed-start 85000 --tensorboard
```

`--task fixed_reach|easy_random_reach|random_reach` 使用同一环境，不自动升级任务难度。历史 BC 初始化只在明确指定 `--init-mode bc_actor --bc-model PATH` 的旧实验分支加载，纯 RL 任务拒绝 BC 参照或初始化。

## Reach 奖励实验

PPO通过 `--reward-mode reach_progress_v1` 选择实际距离进度、持续距离成本、步成本、成功奖金、
安全失败及动作拒绝成本。8项实验初值集中管理并完整保存到config与模型metadata；独立评估自动读取。
普通observation仍为12维，不加入奖励分项；原distance、safe_dense、reach_shaped_v1保持可用。

以下仅提供用户手动执行的±X5mm Pilot，输出目录须为新目录；本次没有执行正式训练：

```bash
python -I scripts/train/train_ppo_reach.py \
  --task axis_pair_reach --observation-mode fetch_like_v1 --reward-mode reach_progress_v1 \
  --action-scale 0.00025 --timesteps 16384 --seed 2026 \
  --eval-every 2048 --checkpoint-every 2048 --eval-episodes 2 --eval-seed-start 97000 \
  --run-dir outputs/rl/ppo_reward/axis_pair_progress_seed2026_pilot
```

先检查两方向是否都能选择正确动作，再考虑random_reach。新旧奖励应以相同观测、seed、预算和目标配对，
至少复核三个训练seed；不同奖励的原始Return不直接比较。完整评估命令、四张诊断图和A/B方案见
[奖励诊断与验证报告](docs/validation/REACH_REWARD_OPTIMIZATION_VALIDATION.md)。

## Reach 观测模式

原始 ReachEnv 和固定目标 PPO 默认使用 `legacy` 的 `27+3+3=33` 维观测及固定预处理。PPO 可显式选择
`--observation-mode fetch_like_v1`：普通 observation 为实际末端世界位置和
`末端世界线速度 × control_period`，各 3 维；加上 achieved_goal 和 desired_goal，
官方 MultiInputPolicy 实际输入共 12 维。运动特征单位为米，包装器将普通位置相对
home、运动特征按固定 0.01m 尺度处理；两个 goal 始终保持世界坐标米。

用户已按以下参数完成训练与独立评估，产物位于
`outputs/rl/ppo_reach/random_fetch_v1_seed2026_ab_new/`。以下保留已完成实验的命令；
复现时必须更换 `--run-dir` 和 `--json`，不能覆盖该目录或原评估文件：

```bash
python -I scripts/train/train_ppo_reach.py \
  --task random_reach --observation-mode fetch_like_v1 \
  --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --eval-every 2048 --checkpoint-every 2048 \
  --eval-episodes 20 --eval-seed-start 87000 \
  --run-dir outputs/rl/ppo_reach/random_fetch_v1_seed2026_ab_new

python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/random_fetch_v1_seed2026_ab_new/final/model.zip \
  --observation-mode fetch_like_v1 --episodes 100 --seed-start 88000 \
  --json outputs/rl/ppo_reach/random_fetch_v1_seed2026_ab_new/heldout_88000.json
```

训练也支持 fixed_reach、axis_pair_reach 和 easy_random_reach；离散两方向评估仍须
恰好 2 个 Episode。评估默认读取 checkpoint 的模式，显式传入不一致模式会报错。
旧 schema=1 / transform schema=2 checkpoint 继续识别为 legacy，不迁移权重。
原命令滞后规则基线、BC 和 SAC 专用实验/诊断目前仅支持 legacy，新模式会明确拒绝。
配置契约、依赖适配、测试证据和 A/B 建议见
[Fetch观测验证报告](docs/validation/REACH_FETCH_OBSERVATION_VALIDATION.md)。本轮训练完成32次Rollout、
320个epoch、2560次Adam更新；开发集87000–87019最终成功2/20，独立集88000–88099成功3/100。
独立集平均初始距离4.122mm、最终距离5.888mm，100次首动作X分量均为正。
后续131072步实验位于 `outputs/rl/ppo_reach/random_fetch_v1_seed2026_131072/`，
使用相同PPO配置和开发集，每8192步评估/保存，独立测试为92000–92099，成功1/100。
12维接口和训练链路验证通过，但两档预算均未学会可靠的随机Reach；在诊断目标映射和固定落点前，不建议直接扩大到262144步。

## 模型评估与绘图

使用已有 PPO Pilot 模型进行独立只读推理；评估 JSON 和图片使用新的输出位置：

```bash
python -I scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/ppo_reach/fixed_seed2026_pilot/checkpoints/step_016384/model.zip \
  --episodes 1 --seed-start 88000 --json /tmp/realman_ppo_evaluation_new.json
python -I scripts/plots/plot_ppo_training.py \
  --run-dir outputs/rl/ppo_reach/fixed_seed2026_pilot --out /tmp/realman_ppo_plots_new
```

SAC、BC 分别使用 `scripts/evaluate/evaluate_sac_reach.py`、`scripts/evaluate/evaluate_bc_reach.py`。评估读取保存的环境与观测变换；绘图只读 JSON，不加载策略、不推断未测量的成功率。早期 schema 1 SAC smoke 文件不受当前 schema 2 加载器支持；本次未改写这些历史文件，详见 [重构报告](docs/validation/PROJECT_ORGANIZATION_REFACTOR.md)。

## 项目结构

- `assets/`：机器人、夹爪与场景资源。
- `src/realman_rl_simtoreal/simulation/`、`envs/`：物理控制与 Gymnasium 环境。
- `tasks/`：算法无关的 Reach 配置；`rl/common/`：公共观测、包装、评估、统计和日志。
- `rl/`：PPO / SAC、SAC 监控诊断；`imitation/`：BC 与显式 BC→SAC 转移。
- `utils/`、`vendor/`：共享几何 / 演示工具和厂家离线适配器。
- `scripts/{train,evaluate,demos,models,diagnostics,plots}/`：按用途分类的正式入口。
- `tests/`：独立回归测试；`docs/`：架构、文件索引、实验及验证记录。
- `outputs/`：Git 忽略的模型、数据和日志，保留历史产物。

包内目录的完整关系见 [架构](docs/architecture.md)，逐文件职责和调用者见 [FILE_MAP](docs/FILE_MAP.md)。旧平铺脚本和 `experiments.*` 不再作为 Python 接口保留。

## 文档索引

- [工程架构与依赖边界](docs/architecture.md)
- [完整 Python 文件地图](docs/FILE_MAP.md)
- [实验历史与历史 README](docs/experiments/README.md)
- [历次验证报告索引](docs/validation/README.md)
- [本次重构、迁移表、验证及新命令](docs/validation/PROJECT_ORGANIZATION_REFACTOR.md)
- [机器人资源来源与限制](assets/robots/rml63/SOURCE.md) · [夹爪来源](assets/grippers/eg2_4c2/SOURCE.md)

## 已知限制

Reach 仍是 home 附近的小范围三维平移任务；没有 GraspEnv、视觉观测、HER 或真机通信。训练入口保留同进程分段训练，不提供跨进程完整续训。实物代际、TCP、零位、惯量和碰撞几何尚需校准；厂家 SDK 与 MuJoCo 模型不完全一致，仿真重力补偿精度不代表真机精度。历史实验细节和失败结论保存在文档中。
