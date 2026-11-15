# 输出目录与实验产物

当前入口：[outputs 本地导航](../outputs/README.md)。整理日期：2026-11-15。

输出按用途划分为五类，保留训练 run 的原名称和内部文件。根目录只保留五个分类目录与 README；旧目录链接已移除。历史 JSON / 日志内的路径快照原样保存，查找文件使用本文迁移表。本说明随仓库保留；`outputs/` 中的模型、导航、清单和审计仍按原 `.gitignore` 约定只保存在本地。

## 正式目录

```text
outputs/
├── README.md                         # 本地导航与主要结果
├── grasp/                            # PPO 课程、控制器、接触实验
├── reach/                            # PPO / SAC、奖励对照、规则基线
├── imitation/                        # BC 数据、权重和历史迁移
├── simulation/                       # 关节 / 末端控制、FK / IK、夹爪、截图
└── validation/                       # 专项验证、测试和工程审计
    └── output_organization/
        ├── CATALOG.json              # 机器可读 run 索引
        ├── legacy_paths.md           # 已停用旧路径的对照表
        ├── manifest_before.json      # 全部原始文件的位置和指纹
        └── root_cleanup_verification.json  # 根目录收拢后的内容校验
```

各分类的子目录见 [本地导航](../outputs/README.md)。原先的 `rl/` 兼容目录与七个顶层旧目录链接已经移除，索引与迁移记录统一收在 `validation/output_organization/`。所有模型、配置、训练和评估数据仍保存在对应分类中。

## 常用入口

- [PPO 夹取抬升模型](../outputs/grasp/ppo/pick_lift_12mm_seed2027/final/model.zip)：12 mm 起点；[独立评估](../outputs/grasp/ppo/pick_lift_12mm_seed2027/heldout_270000_270019.json)为 20/20，[动图](../outputs/grasp/ppo/pick_lift_12mm_seed2027/learned_pick_lift.gif)。默认 20 mm 开发测试的 0/2 失败记录仍在对应 run。
- [20 cm 单目标 PPO](../outputs/reach/ppo/random_20cm_seed2027_continue/final/model.zip)：[独立评估](../outputs/reach/ppo/random_20cm_seed2027_continue/heldout_140000_140099.json)为 97/100。
- [随机起点连续 5 目标 PPO](../outputs/reach/ppo/random_start_multi_20cm_seed2027_continue/final/model.zip)：[独立评估](../outputs/reach/ppo/random_start_multi_20cm_seed2027_continue/heldout_191000_191019.json)为 12/20 条轨迹、72/100 个计划目标。
- [5 mm 随机 Reach PPO](../outputs/reach/ppo/random_goal_relative_seed2027/final/model.zip)：[独立评估](../outputs/reach/ppo/random_goal_relative_seed2027/heldout_120000_120999.json)为 1000/1000。
- [Reach 全部 PPO run 对照](../outputs/reach/ppo/README.md) · [抓取全部 PPO 课程对照](../outputs/grasp/ppo/README.md)。

这些结果摘自已有证据；输出整理只做目录迁移和只读检查，没有重新测量独立成功率。

## 保存与查看规则

1. 新实验使用正式目录和新 run 名称，例如 `outputs/reach/ppo/random_reach_seed2027_20261115/`，保留原 run。
2. 模型与相邻的 `config.json`、`state.json`、`observation_transform.json` 一起保留。不要单独移动一个 checkpoint 的 `model.zip`。
3. 独立评估使用新文件名，例如 `<run>/heldout_<首seed>_<末seed>.json`；图像与 TensorBoard 留在对应 run。专项测试证据进入 `validation/<主题>/<日期>/`。
4. `summary.json` 的 `completed` 只表示训练流程完成；模型效果应查看独立评估的任务范围、样本数和结束原因。
5. 用对应任务目录作为 TensorBoard 根目录。日常命令统一采用新路径；历史记录中的旧地址按迁移表查找。
6. 48 个训练 run、统计容量与索引是本次快照。新增或调整实验后，同步更新本地 `outputs/README.md`、相关 PPO 索引和 `validation/output_organization/CATALOG.json` 中对应记录。

```bash
# 抓取 PPO 指标
python -I -m tensorboard.main --logdir outputs/grasp/ppo --host 127.0.0.1 --port 6006
# Reach PPO 指标
python -I -m tensorboard.main --logdir outputs/reach/ppo --host 127.0.0.1 --port 6006
# 独立加载既有抓取模型；评估使用新文件
python -I scripts/evaluate/evaluate_ppo_grasp_lift.py \
  --model outputs/grasp/ppo/pick_lift_12mm_seed2027/final/model.zip \
  --episodes 20 --seed-start 270000 --json /tmp/realman_ppo_pick_lift_new.json
```

九个相关 CLI 的默认输入 / 输出位置已同步到新目录：`evaluate_reach_baselines.py`、`compare_controllers.py`、`compare_fk.py`、`compare_ik.py`、`run_grasp_contact_cases.py`、`run_grasp_lift.py`、`train_sac_reach.py`、`diagnose_sac_reach.py` 和 `compare_sac_boundary.py`。命令行参数、计算与保存格式沿用原实现。

## 旧目录对照

下表路径均相对 `outputs/`。旧路径不再保留为顶层入口；现有 checkpoint 的来源链记录不改写，查阅来源时按表中正式路径定位文件。

| 原目录 | 正式目录 |
|---|---|
| `rl/ppo_grasp/` | [grasp/ppo/](../outputs/grasp/ppo/) |
| `grasp_lift/` | [grasp/controller/](../outputs/grasp/controller/) |
| `grasp_contact/` | [grasp/contact/](../outputs/grasp/contact/) |
| `gripper/` | [simulation/gripper/](../outputs/simulation/gripper/) |
| `rl/ppo_reach/` | [reach/ppo/](../outputs/reach/ppo/) |
| `rl/ppo_reward/` | [reach/ppo_reward/](../outputs/reach/ppo_reward/) |
| `rl/pure_reach/` | [reach/sac_reward_comparison/](../outputs/reach/sac_reward_comparison/) |
| `rl/sac_reach/` | [reach/sac/](../outputs/reach/sac/) |
| `rl/sac_boundary/` | [reach/sac_boundary/](../outputs/reach/sac_boundary/) |
| `bc_reach/` | [imitation/bc/](../outputs/imitation/bc/) |
| `rl/sac_bc_compare/` | [imitation/bc_to_sac/](../outputs/imitation/bc_to_sac/) |
| `reach_env/` | [validation/reach_env/](../outputs/validation/reach_env/) |
| `reach_task/` | [validation/reach_task/](../outputs/validation/reach_task/) |
| `refactor/` | [validation/project_refactor/](../outputs/validation/project_refactor/) |
| `rl/ppo_reach_validation/` | [validation/ppo_reach/](../outputs/validation/ppo_reach/) |
| `rl/ppo_random_reach_validation/` | [validation/ppo_random_reach/](../outputs/validation/ppo_random_reach/) |
| `rl/ppo_goal_conditioned_validation/` | [validation/ppo_goal_conditioned/](../outputs/validation/ppo_goal_conditioned/) |
| `rl/reach_fetch_observation_validation/` | [validation/fetch_observation/](../outputs/validation/fetch_observation/) |
| `rl/reach_reward_validation/` | [validation/reach_reward/](../outputs/validation/reach_reward/) |
| `rl/pure_reach_validation/` | [validation/pure_reach/](../outputs/validation/pure_reach/) |
| `rl/sac_bc_init_validation/` | [validation/bc_to_sac/](../outputs/validation/bc_to_sac/) |

## 原根目录散文件对照

下表中的文件全部迁移，原文件名保持不变。旧散文件路径作为历史记录保留在报告正文；按此表查找新文件，根目录不保留 45 个散文件副本或链接。

| 原文件（相对 outputs） | 当前文件（相对 outputs） |
|---|---|
| `control_all_tests.txt` | [simulation/gravity_compensation/control_all_tests.txt](../outputs/simulation/gravity_compensation/control_all_tests.txt) |
| `control_comparison.json` | [simulation/gravity_compensation/control_comparison.json](../outputs/simulation/gravity_compensation/control_comparison.json) |
| `control_summary.txt` | [simulation/gravity_compensation/control_summary.txt](../outputs/simulation/gravity_compensation/control_summary.txt) |
| `control_viewer_duration.txt` | [simulation/joints/control_viewer_duration.txt](../outputs/simulation/joints/control_viewer_duration.txt) |
| `control_viewer_initial.png` | [simulation/joints/control_viewer_initial.png](../outputs/simulation/joints/control_viewer_initial.png) |
| `control_viewer_log.txt` | [simulation/joints/control_viewer_log.txt](../outputs/simulation/joints/control_viewer_log.txt) |
| `control_viewer_moved.png` | [simulation/joints/control_viewer_moved.png](../outputs/simulation/joints/control_viewer_moved.png) |
| `control_viewer_sigint.txt` | [simulation/joints/control_viewer_sigint.txt](../outputs/simulation/joints/control_viewer_sigint.txt) |
| `control_viewer_window_close.txt` | [simulation/joints/control_viewer_window_close.txt](../outputs/simulation/joints/control_viewer_window_close.txt) |
| `ee_gui_checks.txt` | [simulation/end_effector/ee_gui_checks.txt](../outputs/simulation/end_effector/ee_gui_checks.txt) |
| `ee_results.json` | [simulation/end_effector/ee_results.json](../outputs/simulation/end_effector/ee_results.json) |
| `ee_viewer.json` | [simulation/end_effector/ee_viewer.json](../outputs/simulation/end_effector/ee_viewer.json) |
| `ee_viewer.txt` | [simulation/end_effector/ee_viewer.txt](../outputs/simulation/end_effector/ee_viewer.txt) |
| `ee_viewer_close.txt` | [simulation/end_effector/ee_viewer_close.txt](../outputs/simulation/end_effector/ee_viewer_close.txt) |
| `ee_viewer_duration.txt` | [simulation/end_effector/ee_viewer_duration.txt](../outputs/simulation/end_effector/ee_viewer_duration.txt) |
| `ee_viewer_final.png` | [simulation/end_effector/ee_viewer_final.png](../outputs/simulation/end_effector/ee_viewer_final.png) |
| `ee_viewer_final.txt` | [simulation/end_effector/ee_viewer_final.txt](../outputs/simulation/end_effector/ee_viewer_final.txt) |
| `ee_viewer_initial.png` | [simulation/end_effector/ee_viewer_initial.png](../outputs/simulation/end_effector/ee_viewer_initial.png) |
| `ee_viewer_moved.png` | [simulation/end_effector/ee_viewer_moved.png](../outputs/simulation/end_effector/ee_viewer_moved.png) |
| `ee_viewer_sigint.txt` | [simulation/end_effector/ee_viewer_sigint.txt](../outputs/simulation/end_effector/ee_viewer_sigint.txt) |
| `ee_x5.json` | [simulation/end_effector/ee_x5.json](../outputs/simulation/end_effector/ee_x5.json) |
| `ee_x5.txt` | [simulation/end_effector/ee_x5.txt](../outputs/simulation/end_effector/ee_x5.txt) |
| `ee_xminus5.json` | [simulation/end_effector/ee_xminus5.json](../outputs/simulation/end_effector/ee_xminus5.json) |
| `ee_xminus5.txt` | [simulation/end_effector/ee_xminus5.txt](../outputs/simulation/end_effector/ee_xminus5.txt) |
| `ee_z5.json` | [simulation/end_effector/ee_z5.json](../outputs/simulation/end_effector/ee_z5.json) |
| `ee_z5.txt` | [simulation/end_effector/ee_z5.txt](../outputs/simulation/end_effector/ee_z5.txt) |
| `fk_comparison.json` | [simulation/kinematics/fk_comparison.json](../outputs/simulation/kinematics/fk_comparison.json) |
| `fk_comparison.log` | [simulation/kinematics/fk_comparison.log](../outputs/simulation/kinematics/fk_comparison.log) |
| `fk_ik_consistency.json` | [simulation/kinematics/fk_ik_consistency.json](../outputs/simulation/kinematics/fk_ik_consistency.json) |
| `gravity_force_checks.json` | [simulation/gravity_compensation/gravity_force_checks.json](../outputs/simulation/gravity_compensation/gravity_force_checks.json) |
| `ik_comparison.json` | [simulation/kinematics/ik_comparison.json](../outputs/simulation/kinematics/ik_comparison.json) |
| `ik_comparison.log` | [simulation/kinematics/ik_comparison.log](../outputs/simulation/kinematics/ik_comparison.log) |
| `joint_control_log.txt` | [simulation/joints/joint_control_log.txt](../outputs/simulation/joints/joint_control_log.txt) |
| `joint_control_results.json` | [simulation/joints/joint_control_results.json](../outputs/simulation/joints/joint_control_results.json) |
| `joint_control_tests.txt` | [simulation/joints/joint_control_tests.txt](../outputs/simulation/joints/joint_control_tests.txt) |
| `kinematics_all_tests.txt` | [simulation/kinematics/kinematics_all_tests.txt](../outputs/simulation/kinematics/kinematics_all_tests.txt) |
| `kinematics_metrics.txt` | [simulation/kinematics/kinematics_metrics.txt](../outputs/simulation/kinematics/kinematics_metrics.txt) |
| `reach_baselines.json` | [reach/baselines/reach_baselines.json](../outputs/reach/baselines/reach_baselines.json) |
| `reach_baselines.profile.txt` | [reach/baselines/reach_baselines.profile.txt](../outputs/reach/baselines/reach_baselines.profile.txt) |
| `tests_02_c1.log` | [simulation/kinematics/tests_02_c1.log](../outputs/simulation/kinematics/tests_02_c1.log) |
| `tests_02_c2.log` | [simulation/kinematics/tests_02_c2.log](../outputs/simulation/kinematics/tests_02_c2.log) |
| `tests_ik_only.log` | [simulation/kinematics/tests_ik_only.log](../outputs/simulation/kinematics/tests_ik_only.log) |
| `viewer_initial.png` | [simulation/viewer/viewer_initial.png](../outputs/simulation/viewer/viewer_initial.png) |
| `viewer_rotated.png` | [simulation/viewer/viewer_rotated.png](../outputs/simulation/viewer/viewer_rotated.png) |
| `viewer_zoomed.png` | [simulation/viewer/viewer_zoomed.png](../outputs/simulation/viewer/viewer_zoomed.png) |

## 本次验证

整理前清单记录 **4033 个文件，2463953402 字节（约 2.29 GiB）**。迁移后逐一核对大小及 SHA-256。首次整理保留的 21 个兼容链接已在后续根目录收拢中移除，未删除它们指向的数据。全部模型文件（728 个 `.zip`、3 个 BC `.pt`）与历史配置 / 日志按同一清单保持内容一致。

5 个代表模型（抓取 PPO、20 cm PPO、多目标 PPO、SAC、BC）按同一 seed 重建 reset 观测，再比较固定输入、预处理和预测动作：30 个数组逐元素相同，最大差为 0；模型内容没有改变。该推理对照记录属于首次迁移阶段；根目录收拢后使用正式路径加载模型。没有为目录整理执行训练。

本次另通过 10 项包 / 入口专项测试、29 个 CLI 的 `--help` 检查，以及 15 份导航 / 文档的 354 个本地链接检查；源码语法树核对确认九个脚本只改变输出默认路径。没有重跑含训练的 396 项完整回归。

[完整审计与验证状态](../outputs/validation/output_organization/README.md) · [逐文件迁移及原始 SHA-256](../outputs/validation/output_organization/manifest_before.json) · [预测对照](../outputs/validation/output_organization/prediction_verification.json)。

根目录收拢后再次核对全部 4033 个原始文件（包含 731 个模型），大小与 SHA-256 全部一致。10 项包测试与 1 项实际历史 checkpoint 推理测试通过，0 失败、0 跳过；测试仅同步两处路径，输入和期望预测不变。[最新收拢校验](../outputs/validation/output_organization/root_cleanup_verification.json)和[测试记录](../outputs/validation/output_organization/root_cleanup_checks.json)已保存。
