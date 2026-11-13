# Reach 双观测模式与 Fetch-like 12 维验证

日期：2026-11-13（Asia/Shanghai）。基线：main `f0d69828a85ccded5a390bb72f1d029e078e1b0e`。
开发分支：`feat/observation-improve`。

本次完成观测接口、固定预处理和下游兼容适配。默认仍为 legacy；只有显式设置
`observation_mode="fetch_like_v1"` 才启用新模式。全量 **314 项测试通过，0 失败、0 跳过**。
新增 PPO 实验仅进行一次 512 步 Smoke，未执行正式训练，不声称随机 Reach 成功率改善。

初始工作区干净。指定分支已经存在，且与 main 同指上述提交、没有独有提交；切到 main
执行 `git pull --ff-only origin main`，确认最新后复用该分支。所有实现修改位于此分支。
旧实验、模型与历史验证报告均保留；本次没有推送、创建 PR 或合并 main。

## 第一部分：原 33 维观测设计

原始环境 Dict 中 `observation` 为 float64 的 27 维向量，另外两个 goal 各 3 维。
其顺序、单位、数值和固定预处理完整保留。

| observation 索引 | 内容及单位 | 提供的信息与预处理 |
|---|---|---|
| 0:6 | joint_1..joint_6 实际关节角，rad | 描述构型、限位和运动学映射；减 home_q，除以 5°对应弧度 |
| 6:12 | 对应关节实际速度，rad/s | 描述关节动态状态；除以 30°/s 对应弧度每秒 |
| 12:15 | ee_site 实际世界位置，m | 当前末端实际位置；减 home_position，除以 0.01m |
| 15:18 | goal - 实际末端位置，m | 显式方向/距离信息；除以 0.01m |
| 18:21 | 当前笛卡尔命令位置，世界坐标 m | 累积增量命令、尚未完成的追踪目标；减 home_position，除以 0.01m |
| 21:27 | 六个执行器 ctrl 关节位置参考，rad | 经过命令速率限制的控制参考；减 home_q，除以 5°对应弧度 |

`achieved_goal` 是 ee_site 实际世界位置，float64、米；`desired_goal` 是当前 Episode
的目标世界位置，float64、米。两者原样保留，不做中心化或缩放。

冗余与保留价值应分别理解：

- observation 的末端位置与 achieved_goal 重复。
- 显式相对位置可由 desired_goal - achieved_goal 重建。
- 实际末端位置也可从完整关节构型和机器人模型计算，因此存在运动学上的关联。
- 关节构型、关节速度、笛卡尔命令和执行器参考并非简单的重复信息。相同末端位置可能对应不同构型；
  命令与参考还能描述控制器内部累积状态和执行滞后，删除它们具有代价。

这些关联可能增加网络处理输入的复杂性，但本次没有证明其导致原来的动作方向偏置。
`_legacy_observation()` 保留原 `_observation()` 的完整函数主体；旧黄金轨迹逐项精确回归。

## 第二部分：FetchReach 参考设计与实现

参考官方实现：

- [fetch_env.py：_get_obs 与 generate_mujoco_observations](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/fetch_env.py)
- [reach.py：FetchReach 的任务参数和 Goal-aware 接口](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/reach.py)
- [MuJoCo Site Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jacsite)

官方 Reach 使用 `has_object=False`、`block_gripper=True`。`_get_obs()` 中物体相关数组为空，
普通 observation 是末端位置 3、夹爪位置 2、末端线速度特征 3、夹爪速度特征 2，共 10 维；
achieved_goal 从实际 grip Site 位置复制，desired_goal 从当前 `self.goal` 复制，各 3 维。
完整 Dict 共 16 维。

这一接口把当前运动状态放在普通 observation，把已经达到的位置和期望位置单独提供给
goal-conditioned 策略与奖励接口。位置描述当前位置，速度特征提供运动方向和变化趋势。
这是对接口作用的解释，并非官方保证这些量对任意机器人都构成充分状态。
保留 GoalEnv 格式也不等于必须使用 HER；本次仅使用官方 PPO MultiInputPolicy。

官方 `generate_mujoco_observations()` 从 MuJoCo Site 获取 grip 实际世界位置和线速度，
随后乘 `dt = n_substeps * model.opt.timestep`。本项目去掉全部 4 维夹爪状态，得到：

```python
{
    "observation": np.r_[ee_world_position_m, ee_world_velocity_m_s * control_period_s],
    "achieved_goal": ee_world_position_m,
    "desired_goal": episode_target_world_position_m,
}
```

普通 observation 的索引 `0:3` 为真实 ee_site 世界位置，`3:6` 为末端世界线速度乘控制周期。
三个 Dict 字段的原始形状是 `(6,)`、`(3,)`、`(3,)`，均为 float64，不存在其他观测字段。
achieved_goal 既不是 IK 解算目标，也不是笛卡尔命令；desired_goal 在当前 Episode 内固定。

### 真实速度、自由度与数值单位

在当前实际 `MjData` 上调用 `mj_jacSite(model, data, jacp, None, ee_site_id)`，
`jacp.shape == (3, model.nv)`；计算 `jacp @ data.qvel`。
Jacobian 列与完整广义速度逐列对应，不切成前六列，不使用 qpos 索引或假定关节 DOF 连续。
计算结果是世界坐标线速度 m/s，再乘原有 `control_period`（默认 0.05s）。

因此，最终运动特征是**当前瞬时线速度按一个控制周期缩放得到的位移尺度，单位 m**。
它不是未经缩放的 m/s，也不等于上一控制周期真实轨迹的积分或相邻位置差分。
实现不调用 mj_step、mj_forward、FK 或 IK，不写入 qpos/qvel/ctrl，不推进时间。
测试同时覆盖 0.05s/0.1s、MuJoCo 世界/局部速度变换、独立私有 MjData 的 FK 差分、
含额外关节和 freejoint（nq≠nv）的合成模型，以及读取前后完整积分状态不变。

### 原始物理观测与网络固定预处理

环境返回原始物理量；`ScaledReachObservation` 中的 `ReachTransform` 负责：

```python
processed_observation[:3] = (raw_ee_position - home_position) / 0.01
processed_observation[3:] = (raw_ee_world_velocity * control_period) / 0.01
# achieved_goal / desired_goal 仍为世界坐标米，float64
```

普通 processed observation 为 float32。SB3 在进入网络时按照其默认流程将全部字段转为 float32。
没有 VecNormalize、动态均值/方差、额外裁剪或奖励归一化。归一化结果不会传给 IK、FK、
物理控制、成功判定或安全逻辑；包装器 compute_reward 始终接收未归一化的 goal。

legacy 的 transform schema=2 和生成配置保持原样；fetch_like_v1 使用 schema=3，额外明确
记录模式、运动特征含义与 `control_period_s`，固定位置尺度仍为 0.01m。

## 第三部分：新旧观测、网络及依赖对比

| 指标 | Legacy | Fetch-like |
|---|---|---|
| 环境完整 Dict 观测维度 | 33（27+3+3） | 12（6+3+3） |
| 实测神经网络输入维度 | 33 | 12 |
| 实测特征张量形状，batch=1 | (1, 33) | (1, 12) |
| 关节角度 | 有 | 无 |
| 关节角速度 | 有 | 无 |
| 末端位置 | 有 | 有 |
| 末端速度特征 | 无 | 有，世界速度×控制周期，单位 m |
| 显式相对目标位置 | 有 | 无 |
| 控制命令位置 | 有 | 无 |
| 关节控制参考值 | 有 | 无 |
| achieved_goal | 有，世界坐标 m | 有，世界坐标 m |
| desired_goal | 有，世界坐标 m | 有，世界坐标 m |
| 固定预处理版本 | schema=2 | schema=3 |
| 默认模式 | 是 | 否，须显式开启 |

### 官方网络的实际输入

PPO 仍是 SB3 `MultiInputActorCriticPolicy`，使用原生 `CombinedExtractor` 和三个 `Flatten`。
没有自定义提取器、9 维筛选或额外相对坐标拼接。现有 `features()` 仅用于 BC/诊断核对，
不接管 PPO 的特征提取。

SB3 按 Gymnasium Dict 的键顺序 `achieved_goal, desired_goal, observation` 拼接。
fetch_like_v1 的实际 12 维顺序是：

1. `0:3` achieved_goal 世界坐标米。
2. `3:6` desired_goal 世界坐标米。
3. `6:9` `(实际末端位置-home_position)/0.01`。
4. `9:12` `实际末端世界线速度*control_period/0.01`。

测试在 Actor 和 Critic 第一隐藏层注册只读 hook，比较实际收到的两个张量与上述完整向量，
确认两者相同；分别改变三个 Dict 字段，均能改变实际提取特征。
Actor/Critic 隐藏层仍各 `[256,256]`、Tanh；Actor 输出合法 XYZ 三维动作。
`PPO_CONFIG` 的 AST 与基线一致，官方 PPO 采样、损失、优化与分布代码均未修改。

### 修改前依赖搜索与适配边界

修改前对 src/scripts/tests 搜索了 `ReachTransform`、`ScaledReachObservation`、`features()`、
`MultiInputPolicy`、`rule_action`、所有 observation 索引及固定 27/33/features_dim。
处理结果如下：

| 消费者 | 处理 |
|---|---|
| ReachEnv | 模式校验与独立分派；legacy 主体原样保留 |
| ReachTransform / wrapper | 旧 schema=2；新 schema=3；形状、批次和模式检查 |
| env_factory / make_probe | 已将环境配置完整传入 ReachEnv，并从原始 env 生成变换，无需改变包装栈 |
| PPO 训练/评估/加载 | 增加 CLI 模式、保存契约、实测 features_dim、模式及模型空间核对 |
| 共享评估与 MotionStats | 命令位置从已有 reset/step info 取，避免读 observation[18:21]；不进入策略输入 |
| PPO 训练运动审计 | 原实现已从 info 获取命令状态，保持原样 |
| 原 rule_action | 保留 command-lag 控制器，仅 legacy；新模式明确报错，不替换控制器 |
| SAC 训练/加载/独立评估/专用轨迹诊断 | legacy 继续支持；fetch_like_v1 提前明确拒绝 |
| BC 策略/数据/统计/评估、BC→SAC | 固定 33 维网络保持原样；不兼容配置提前明确拒绝 |
| probe_ppo_goals | 支持 12/33 维；新模式仅 desired_goal 索引允许随同状态目标变化 |
| probe_reach_env | 运动统计从已有 info 获取命令，不再读取旧索引 |
| 历史 baseline 验证脚本 | 保留 legacy，进入旧索引检查前明确拒绝其他模式 |
| 绘图、结果分析、无固定索引的环境 Smoke | 使用结果 JSON/公共字段，无需改变 |

没有新增另一种规则基线，也没有把命令位置加回新观测。SAC/BC 的拒绝是本次范围内的显式
支持边界；PPO 的四个目标任务均可使用新模式，目标分布未变。原六方向 5mm 候选预检仍会
拒绝不合法方向，本次没有改成更小半径或绕过检查。

### Checkpoint 区分和旧模型兼容

新 PPO 配置顶层 schema=2，每个 checkpoint config 保存：

- `observation_mode`、`observation_metadata_version=1`。
- 原始 `observation_space` 和 `processed_observation_space`：每个键的形状、dtype、无界空间语义。
- `features_dim`、SB3 `feature_keys`、各字段逐标量 `observation_field_order`。
- `observation_transform`、`normalization_version`、完整 `normalization_config`。

保存及加载均核对配置一致性；加载后再核对模型内部 observation_space 和实际提取器维度。
显式请求错误模式、错配模型与配置、删除新元数据、错配控制周期或错误形状均会报错。
独立评估默认使用保存模式，也允许 `--observation-mode` 作为一致性断言。

历史 PPO `config schema=1 + observation_transform schema=2` 即使缺少所有新增字段，也按
legacy 正常加载；不修改配置文件、权重或层尺寸。此前已不支持的更早 transform 格式
不在本次重构中迁移。SAC 新生成的 legacy 配置也补充相同观测元数据。

## 第四部分：实际测试结果

运行环境：Conda `realman_rl`，Python 3.11；MuJoCo 3.17.0、Gymnasium 1.4.0、
NumPy 2.4.6、SB3 2.9.2、PyTorch 2.14.0+cpu。未安装新依赖或改变锁定版本。

### 命令与数量

以下命令均在仓库根目录执行。首轮不训练专项测试：

```bash
/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python \
  -m unittest discover -s tests -p test_fetch_observation.py -k Observation -v
```

结果：14 项通过，14.093s。

实际全量回归命令（旧 PPO 三种 Smoke 使用已有产物，避免重复训练）：

```bash
env \
  REALMAN_PPO_TEST_ARTIFACTS=outputs/rl/ppo_reach_validation/smoke_seed2026 \
  REALMAN_AXIS_PPO_SMOKE_ARTIFACTS=outputs/rl/ppo_goal_conditioned_validation/20261113_preparation/axis_pair_smoke_seed2026 \
  REALMAN_RANDOM_PPO_SMOKE_ARTIFACTS=outputs/rl/ppo_random_reach_validation/20261113_preparation/random_smoke_seed2026 \
  REALMAN_FETCH_PPO_TEST_ARTIFACTS=outputs/rl/reach_fetch_observation_validation/20261113/smoke_seed2026 \
  /home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python \
  -m unittest discover -s tests -v > /tmp/realman_fetch_full_tests.log 2>&1
```

结果：**314 项通过，198.150s；0 失败、0 错误、0 跳过**。其中新增专项 17 项，旧回归 297 项。
全量回归也执行了原有 BC/SAC 的微型单元测试训练；它们使用测试临时目录，没有将 BC/SAC
引入新观测 PPO 实验。新增 PPO 仅训练一次 512 步，未启动其他 PPO 训练或正式实验。
后续重跑时保留上述环境变量即可复用本次产物。

`python -m compileall -q src scripts tests` 和 `git diff --check` 也通过。
Gymnasium 的无界 Box 提示属于原设计保留的警告，不影响空间包含检查；没有观测裁剪。

### A–G 验收覆盖

| 测试 | 实测结果 |
|---|---|
| A 维度与空间 | 四类目标任务×两种模式；27+3+3=33 / 6+3+3=12；raw/wrapped 均满足空间 |
| B 物理状态 | 真实 Site/目标一致；世界速度、局部转世界、时间缩放、有限值及无状态写入均通过 |
| C 网络实际输入 | 原生 CombinedExtractor，features_dim=33/12；Actor/Critic 实际输入逐值一致；合法三维动作 |
| D 目标方向 | 同 Home 的 ±X5mm，achieved_goal 和普通观测相同，desired_goal 正确不同；仅目标输入索引变化 |
| E 物理一致性 | 四任务×三奖励×两边界×正常/边界动作，共48组，再加成功轨迹；逐项精确相等 |
| F PPO Smoke | 初始化、采样、完整 Rollout、Actor/Critic 更新、保存、重载、独立 CLI 评估全部通过 |
| G 旧模型 | 历史 schema 载入测试与真实旧模型 6 组冻结预测逐值相等；模型/配置哈希不变 |

E 使用 seed=83001 和相同动作序列，比较实际轨迹、qpos/qvel/qacc/ctrl/驱动力、奖励、
terminated/truncated、完整 info（含全部安全记录）、仿真时间及真实 mj_step 计数。
实际采用 `assert_array_equal`，比一般浮点容差更严格；同时检查物理步之间没有 qpos 改写。
与 main 比较确认 `_info`、`reset`、`_sample_goal`、`compute_reward`、`_shaped_reward`、
`step`、`render`、`close` 及目标候选校验的 AST 均未改变。
资产、simulation 目录、依赖配置文件也没有差异。
原黄金 fixture 未改；原 AST 测试仅将新函数名 `_legacy_observation` 规范回原名称后比较原哈希。

同 Home 的实际世界坐标（m）：

- achieved_goal：`[0.005616691094786186, -0.00014515865649906284, 1.060227293451642]`。
- +X desired_goal：`[0.010616691094786187, -0.00014515865649906284, 1.060227293451642]`。
- -X desired_goal：`[0.0006166910947861858, -0.00014515865649906284, 1.060227293451642]`。

### 唯一一次新增 PPO Smoke

通过新测试类调用原训练入口：

```python
train(
    "outputs/rl/reach_fetch_observation_validation/20261113/smoke_seed2026",
    timesteps=512, task="axis_pair_reach", observation_mode="fetch_like_v1",
    seed=2026, eval_every=512, checkpoint_every=512,
)
```

PPO_CONFIG 原样：n_steps=512、batch_size=64、n_epochs=10、学习率3e-4、原隐藏层等均未调整。
实测 512 timesteps、1 次完整 Rollout、10 epoch、80 Adam 步；Actor 和 Critic 参数哈希均变化，
Rollout 数组及训练损失均有限。训练/内置评估共调用 env.step 1312 次，真实物理步 32800 次；
其中训练为 512 步，另外是初始/结束独立评估，不计入优化预算。
测试阻断规则动作调用，并逐物理步确认没有 qpos 写入捷径；配置为 scratch、pure_rl。

独立新进程 CLI 实际使用：

```bash
/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python -I \
  scripts/evaluate/evaluate_ppo_reach.py \
  --model outputs/rl/reach_fetch_observation_validation/20261113/smoke_seed2026/final/model.zip \
  --observation-mode fetch_like_v1 --episodes 2 --seed-start 88000 \
  --json outputs/rl/reach_fetch_observation_validation/20261113/smoke_seed2026/independent_test_evaluation.json
```

两方向均200步超时，成功率0%，安全失败0；模型参数和优化计数保持不变。
这只验证链路，不能说明训练收敛或方向偏置得到解决。
initial、step_000000、step_000512、final 的模型与配置均保存；step_000512 与 final 确定性预测相同。

### 真实旧 checkpoint 证据

改动前对 `outputs/rl/ppo_reach_validation/smoke_seed2026/final/model.zip` 保存了两种 seed 下
共6组 raw Dict、processed Dict 和确定性动作。改动后全部逐值相同，33维提取器保持正常。
这些预先冻结的数据保存在 `tests/fixtures/reach_legacy_policy_predictions.json`。

- 旧模型 SHA256：`c4038d0d298a3749b2ee4c48be1bddd51c8e44f17ab3491e5517ad8f6513f0b6`。
- 旧 config SHA256：`b51cbf8bd34d5751558e0bda8d216cd5df288de7e0091d35781e8a09747ab2a9`。

CI 没有本地历史模型时，这项外部实物核对会单独跳过；不依赖旧产物的 schema=1 保存/加载
测试仍会执行。本次本地314项没有跳过，实际旧模型已验证。

### 本地证据路径

全部新实验保存在 Git 忽略的 `outputs/rl/reach_fetch_observation_validation/20261113/`：

- `unittest_full.log`：全量测试原始输出。
- `test_results.json`：测试数量与本次 PPO 训练次数。
- `integrity_and_dimensions.json`：原始观测、实际网络张量、两方向世界坐标、版本与源码一致性。
- `smoke_seed2026/summary.json`、`config.json`、`metrics.jsonl`、`actions.jsonl`。
- `smoke_seed2026/test_verification.json`：物理步及无专家/无 qpos 捷径证据。
- `smoke_seed2026/final/`：模型、配置、变换、状态及 last_rollout.npz。
- `smoke_seed2026/independent_test_evaluation.json`：独立评估结果。

## 第五部分：设计局限

1. 12维不一定比33维更好；本次没有正式 A/B 成功率证据。
2. 末端位置和线速度可能无法完整描述关节动力学、姿态、接近限位的程度或执行器状态。
3. 不同关节构型可能具有相同末端位置，末端观测可能存在状态混淆。
4. 删除笛卡尔命令和关节控制参考后，策略处理累积命令、执行延迟和追踪误差的能力可能降低。
5. 维度减少不等于一定更容易训练；没有显式相对位置时，网络仍需从两个世界坐标 goal 学到关系。
6. 本次既删除了部分状态，也新增末端运动特征，因此不是单纯改变维度数量，不能把未来效果归因
   为“少了21维”这一单一因素。
7. 本方案只针对现有局部 Reach 的观测对照，不能据此认为已验证真机部署表现。

## 后续手动训练、评估与 A/B 建议

以下命令只提供给用户后续执行，本次未启动。先激活 realman_rl 并进入仓库；使用全新目录：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal

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

A/B 中仅改变 `--observation-mode legacy|fetch_like_v1` 与输出目录，分别从随机权重开始。
每对运行固定任务、seed、预算、奖励、action_scale、PPO_CONFIG、评估目标和checkpoint选择规则。
建议至少比较2026/2027/2028三个训练seed，分别报告结果及跨seed波动，不选择最好的一次代替整体。
可先用 fixed_reach 做回归，再以 axis_pair_reach 分别看±X结果，随后评估 easy_random_reach /
random_reach；不要自动改变目标分布。axis_pair 的训练内与独立评估须各为2个Episode。

随机任务使用独立 held-out seeds，报告成功率、距离、步数、安全失败、方向/象限覆盖、
动作朝目标比例与动作偏置。固定目标和两方向候选重复次数不代表随机目标泛化。
预算不足时可明确将运行记为 Pilot；不要以调奖励、PPO超参数或BC初始化的方式改善某组。
当前没有跨进程完整续训支持，不能把新的训练命令当成恢复旧权重训练。

## 文件修改清单

| 文件 | 本次修改 |
|---|---|
| `src/realman_rl_simtoreal/envs/reach_env.py` | 双模式与完整Site Jacobian运动特征；保留旧观测主体 |
| `src/realman_rl_simtoreal/rl/common/observation.py` | 版本化固定变换、维度/顺序描述、checkpoint验证、legacy边界 |
| `src/realman_rl_simtoreal/rl/common/baselines.py` | 原规则策略明确限制legacy |
| `src/realman_rl_simtoreal/rl/common/evaluation.py` | info运动统计与规则策略配置检查 |
| `src/realman_rl_simtoreal/rl/ppo.py` | 模式配置、模型空间检查和兼容加载/保存 |
| `src/realman_rl_simtoreal/rl/sac.py` | 新观测模式提前拒绝 |
| `src/realman_rl_simtoreal/rl/sac_diagnostics.py` | legacy检查与模型契约核对 |
| `src/realman_rl_simtoreal/tasks/reach.py` | 可选观测配置，目标分布保持原样 |
| `src/realman_rl_simtoreal/imitation/bc.py` | BC各索引消费者的明确legacy限制 |
| `src/realman_rl_simtoreal/imitation/bc_to_sac.py` | 33维初始化兼容边界 |
| `scripts/train/train_ppo_reach.py` | CLI模式、完整元数据、实测网络维度 |
| `scripts/train/train_sac_reach.py` | CLI明确拒绝新模式；legacy配置补充元数据 |
| `scripts/evaluate/evaluate_ppo_reach.py` | 自动读取/显式校验保存模式 |
| `scripts/evaluate/evaluate_sac_reach.py` | legacy及模型空间校验 |
| `scripts/evaluate/evaluate_reach_baselines.py` | 历史固定索引检查前明确限制模式 |
| `scripts/diagnostics/probe_ppo_goals.py` | 33/12维目标敏感性审计 |
| `scripts/diagnostics/probe_reach_env.py` | 从info读取命令用于运动统计 |
| `tests/test_fetch_observation.py` | 17项新增A–G契约/物理/PPO测试 |
| `tests/test_ppo_reach.py` | 旧观测AST核对适应函数改名，原期望哈希不变 |
| `tests/fixtures/reach_legacy_policy_predictions.json` | 重构前真实旧模型的6组冻结预测 |
| `README.md` | 双模式与训练/评估使用说明 |
| `docs/FILE_MAP.md` | 当前文件职责及新增测试索引 |
| `docs/validation/README.md` | 新验证报告索引 |
| `docs/validation/REACH_FETCH_OBSERVATION_VALIDATION.md` | 本报告 |

`rl/common/env_factory.py`、`metrics.py`、PPOAudit 与全部 simulation/模型文件均已检查，
不需要修改。没有增加训练框架、算法或控制器重构。
