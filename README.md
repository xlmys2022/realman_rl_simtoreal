# realman_rl_simtoreal

基于 MuJoCo 的 RealMan **RML63II-6F-V** 机械臂强化学习仿真项目，长期方向为 Reach → Grasp → 视觉 → 真机部署。当前采用 Python 3.11、Gymnasium、Stable-Baselines3 和 PyTorch；代码组织与算法实验分别维护。

## 已实现功能

- 六轴动力学、关节与末端控制、FK / DLS IK、厂家离线 FK / IK 对照。
- EG2-4C2 夹爪模型、同步控制和接触 / 抬升 / 释放案例。
- 固定及随机目标 ReachEnv，PPO、SAC 训练、独立评估、checkpoint 和 TensorBoard。
- 独立 BC 示范采集、训练和推理；保留历史 BC→SAC Actor 初始化、边界对照与 Q/TD 诊断。
- 自动化回归测试、环境探针及只读训练曲线绘图。

## 当前状态

当前入口为 PPO Reach。已有 seed2026、16384 步固定目标 Pilot：最终独立评估 18 步成功、误差约 0.961 mm；这是已有实验结果，不代表随机目标泛化。纯 SAC Pilot 未稳定成功，BC 融合暂作历史复现。完整证据见 [PPO 验证](docs/validation/PPO_REACH_VALIDATION.md) 和 [实验索引](docs/experiments/README.md)。本次结构重构没有重新执行正式训练或改变算法、模型及物理参数。

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

以下为手动训练示例，本次重构未执行。每次换一个未使用的 `--run-dir`，保留已有结果：

```bash
python -I scripts/train/train_ppo_reach.py \
  --task fixed_reach --reward-mode reach_shaped_v1 --action-scale 0.00025 \
  --timesteps 16384 --seed 2026 --run-dir outputs/rl/ppo_reach/fixed_seed2026_new \
  --eval-every 2048 --eval-episodes 1 --eval-seed-start 87000 \
  --checkpoint-every 2048 --tensorboard
python -I -m tensorboard.main --logdir outputs/rl/ppo_reach --host 127.0.0.1 --port 6006
```

PPO 保留官方 512 步 Rollout、64 样本 minibatch、10 个 epoch；评估和保存发生在完整更新之后，因此实际步数可能超过请求下界。原始高斯动作与环境 Box 裁剪动作分别记录，动作裁剪率与 PPO `clip_fraction` 含义不同。固定目标更换 seed 不等于泛化验证。

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
