# realman_rl_simtoreal

基于 MuJoCo 的 RealMan **RML63II-6F-V** 机械臂强化学习仿真项目，长期方向为 Reach → Grasp → 视觉 → 真机部署。当前采用 Python 3.11、Gymnasium、Stable-Baselines3 和 PyTorch；代码组织与算法实验分别维护。

## 已实现功能

- 六轴动力学、关节与末端控制、FK / DLS IK、厂家离线 FK / IK 对照。
- EG2-4C2 夹爪模型、同步控制和接触 / 抬升 / 释放案例。
- 固定及随机目标 ReachEnv，PPO、SAC 训练、独立评估、checkpoint 和 TensorBoard。
- 独立 BC 示范采集、训练和推理；保留历史 BC→SAC Actor 初始化、边界对照与 Q/TD 诊断。
- 自动化回归测试、环境探针及只读训练曲线绘图。

## 当前状态

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
  --task random_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 \
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
  --task axis_pair_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 \
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

默认 `legacy` 完整保留原 `27+3+3=33` 维观测及固定预处理。PPO 可显式选择
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
