# 04-D1：BC初始化SAC与训练监控

日期：2026-11-08。完成显式Actor初始化、独立评估、checkpoint、TensorBoard和离线PNG；只运行第0步验证、128步Smoke Test及必要测试，**未运行10k/30k或更长正式训练**。

结果：第0步在新seed50000–50019上20/20成功、误差均值.878671mm。初始化动作最大差5.96046e-8（容差1e-7）。短训练已有能力退化证据：额外配对seed60000–60002，第0步3/3、128步1/3，另外2次关节命令范围终止。保留失败，不继续加步数或调参。

## 参数移植与保护

`experiments.sac_initialization.initialize_actor_from_bc(model,path,transform,probe)`仅接受0步、0更新、空Replay及空Actor/Critic优化器状态的标准SAC。BC通过weights_only加载，模型缺失、标识错误或配置不兼容直接报错，无随机初始化兜底。

验证包含：固定预处理配置逐项相同；Dict次序`achieved_goal(3), desired_goal(3), observation(27)`；各Box shape、dtype及边界；原float32三维Box(-1,1)动作；CombinedExtractor中按此顺序的Flatten、33维输入；两层256 Linear/ReLU；3维Linear mu；非SDE及SquashedDiagGaussian/tanh动作映射。相同shape但激活、提取顺序或动作空间不同仍拒绝。

复制仅为：

- `BC.latent_pi.{0,2}.{weight,bias}` → `SAC.actor.latent_pi.{0,2}.{weight,bias}`。
- `BC.mu.{weight,bias}` → `SAC.actor.mu.{weight,bias}`。

使用原参数对象的strict state_dict复制，优化器仍指向同一批Actor参数。`log_std`保留SB3非SDE nn.Linear按本次seed初始化的权重/bias，不把它改成常数。由于latent已改变，随机动作的标准差输出会变，但该head参数未被覆盖。双Critic完全从头初始化，Target Critic由SB3原机制复制；复制前后分别验证log_std、Critic、Target指纹和熵系数不变。没有专家Replay、Critic预训练、Actor冻结或BC辅助损失。

当前实现依据[SB3 Actor源码](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/sac/policies.html)，未修改SB3库。BC仍是独立监督模型，不将其原文件改名为SAC。仅通过显式`--init-mode bc_actor --bc-model ...`启用；缺省scratch保留。

## 第0步与短训练结果

原BC文件SHA256为`af059a26c78f0cb5eb40075ddfdf16ad1e1e18821660396ac44bd9dfe04eb8b8`，网络SHA256为`2be1479af9bd5aad90b984bc1ffe8cd660b4b5f2a71b39d06f537fafbb9ffe42`。完整指纹、复制映射、保护参数指纹及配置独立保存在run的initialization.json。

固定探针及独立环境都使用相同04-A预处理。BC与SAC deterministic动作最大差5.96e-8、探针MAE3.65e-8。SAC.predict包含动作反缩放，Box(-1,1)数学上为恒等但仍有float32舍入，因此使用严格1e-7绝对阈值。随机SAC样本不是该比较对象。

- 新的20目标seed50000–50019：第0步20/20成功，平均14.7步，最终距离.878671mm，无安全失败或超时；这些目标未参加BC监督训练/验证。
- Smoke：seed2026、128步、28次官方SAC更新、Replay128条。eval_every=64、checkpoint_every=32，阶段0/32/64/96/128分别保存，独立评估在0/64/128。原SAC超参数全部保留。
- Smoke监控目标50000–50002：0及64步均3/3成功、平均15.667步、.868228mm；128步仍3/3，但平均23步、.983022mm。成功率没有体现出的速度/精度退化也保留。
- 固定探针MAE从3.65e-8增至.0428532，最大差.0808070；标准差输出均值.971513→.886784，温度1→.991632。Actor/Critic均实际更新，权重与所有统计有限。
- 额外新目标60000–60002，分别加载第0步和最终模型进行配对：0步3/3、.857191mm、成功平均16.333步；128步1/3、2.880425mm、唯一成功22步，2次joint_limit终止。两者目标逐项相同，无碰撞/数值故障。该joint_limit是局部关节命令保护，不是允许非法目标继续执行。
- 训练探索与确定性评估分开：训练完整Episode仅2个，成功0，均joint_limit终止，另21步未完成集；平均训练回报-216.172。没有用这些训练回报宣称Reach改善。
- 最后一个minibatch Actor loss约-1.97100、Critic loss790.522；冻结128条Replay的soft TD残差RMSE25.1566。二者不是同一个统计量，不能仅凭loss大小断言算法错误。

这只是短测试，不能估计长期训练收益或跨seed可靠性。它已支持“训练更新可改变并损害BC能力”，未证明是某一个Critic、熵或探索因素单独造成。本次不加入任何补救算法。尤其保持learning_starts=100：前100条是官方均匀随机warmup动作，之后仍按随机SAC策略采样；初始化确定性均值不意味着训练rollout等于BC策略。

默认scratch兼容性另行用修改前入口及新入口各运行128步、同seed74、同64步评估间隔：Actor/双Critic/Target全部policy张量和温度逐元素一致，均28更新。只读诊断没有改变原随机采样序列或优化结果。

## 监控与保存

`--tensorboard`显式启用SB3自带TensorBoardOutputFormat，与CSV/JSON共同写在`run_dir/sb3`；默认旧scratch仍不强制使用TensorBoard。官方日志用法参考[SB3 TensorBoard文档](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)。

主要标量：monitor/total_timesteps、training/episode_reward_mean、training/episode_length_mean、training/episode_success_rate、training/action_rejection_fraction、training/safety_failures_total；原train/actor_loss、train/critic_loss、train/ent_coef；evaluation/success_rate、final_distance_mm、success_steps_mean、episode_length_mean、episode_reward_mean、timeouts、safety_failures、action_rejection_fraction；bc/fixed_probe_*及bc/evaluation_states_*；固定Q与soft TD摘要。训练Episode窗口成功率与独立评估成功率使用不同命名，缺少成功集时不伪造平均成功步数。

BC偏差有两种：固定reset观测探针便于跨阶段比较；当前SAC自身评估轨迹上的动作偏差反映实际访问状态，但包含状态分布变化。冻结BC只做只读参照，不修正动作。探针与TD计算不消耗训练RNG，不更新网络或插入Replay。TD按现有timeout-aware公式，取前至多256个存储槽，不伪称历史minibatch或全buffer统计；每阶段也保存空buffer时可用的固定Q值。

TensorBoard事件已通过EventAccumulator与保存评估JSON逐项对照，`--inspect`确认0至128步、5个事件step、无乱序。TensorFlow未安装，标量所需功能正常；没有为此安装TensorFlow。

每次run保存：config.json、observation_transform.json、dependencies.json、initialization.json、fixed_probe.npz、initial_model.zip及initial/{model.zip,state.json,config,transform}；checkpoints/step_*；evaluations.json、checkpoints.json、training_episodes/rejected_actions/diagnostics/bc_diagnostics日志、CSV/JSON/TensorBoard；final完整模型及官方Replay压缩文件。第0步独立保存、无prefill。非空目录拒绝覆盖。

eval_every与checkpoint_every可独立配置；不重合时未评估的checkpoint不捏造评估数据，评估结果保存于独立evaluations/step_*。总预算末步强制保存/评估（原安全/数值中止只保存故障状态，不继续评估坏模型）。Ctrl+C由原异常路径保存partial final。仍不提供跨进程完整续训，不能仅加载模型后把Replay/环境/RNG状态当成自动恢复。

## PNG入口

`scripts/plot_sac_training.py`只读evaluations.json/checkpoints.json/diagnostics.jsonl，输出独立评估成功率、平均最终距离、Actor/Critic loss及熵系数四个PNG。横轴为实际训练步数，缺数据时显式skip，不把训练回报转换为成功曲线。Actor/Critic分别使用子图，避免大Critic loss淹没Actor；少量loss测量只画实际点，不补造之前的曲线。使用Agg，无GUI依赖，缓存放于输出目录；显式MPLCONFIGDIR优先。

已实际导出并查看Smoke PNG，plot_data.json保留用于画图的准确序列。原始CSV/JSON不删除。训练结束后可独立导出，不需要加载环境、SDK或模型。

## 验证、依赖与文件

原222/222基线通过（127.200s）；新增12项通过，完整**234/234通过（133.011s）**，未改原测试断言。覆盖参数映射、确定性动作、错误预处理/同shape错误激活/顺序/空间、缺失模型、log_std、双Critic实际更新、空状态/不可重复初始化、保存加载、RNG、事件数据一致、独立评估/checkpoint周期、无评估时不造曲线。新测试初次执行时误用类属性run遮蔽unittest方法，已改为run_dir；不是训练实现故障。

只在realman_rl安装TensorBoard2.21.0、Matplotlib3.11.2及共15项新包，锁定在requirements-rl.txt；所有原有依赖版本逐项不变，pip check通过。安装前dry-run无升级既有包。333项基线清单中，仅允许原训练入口和实验evaluate扩展有变化；资产、envs、simulation、vendor、旧测试、04-A/B/C全部输出逐字节保持。没有提交或推送。

新增sac_initialization.py负责校验/移植，sac_monitoring.py负责只读监控，plot_sac_training.py负责离线PNG，test_sac_bc_initialization.py负责12项验证；train_sac_reach.py增加显式参数和阶段调度，sac_reach.evaluate仅增加可选冻结参照。README、architecture、requirements-rl及本报告更新。核心控制和SAC损失均未改。

本次证据集中在`outputs/rl/sac_bc_init_validation/`：基线清单/依赖计划、测试日志、smoke_bc_seed2026、step_zero_20.json、配对加载评估、scratch兼容性两组128步、TensorBoard inspect、PNG及完整性审计。网页交互、正式10k及长期效果未验证。本次所有实验训练调用至多128步；测试中的更短更新仅验证接口。

## 用户手动运行：本次未执行以下正式训练

先激活环境并进入项目。A/B仅init-mode及目录/标签不同；两组均加载同一BC作为只读监控参照，环境默认terminate，完全相同的SAC_CONFIG、seed、目标及预算。不要重复使用已有非空目录。

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal

# A：原随机Actor；BC仅作监控参照
python -I scripts/train_sac_reach.py --run-dir outputs/rl/sac_bc_compare/manual_scratch_seed2026 --timesteps 10000 --seed 2026 --init-mode scratch --bc-model outputs/bc_reach/fit_seed2026/best_bc.pt --boundary-mode terminate --eval-at-start --eval-every 1000 --eval-episodes 20 --eval-seed-start 50000 --checkpoint-every 1000 --tensorboard --label scratch

# B：只用BC初始化Actor均值通路
python -I scripts/train_sac_reach.py --run-dir outputs/rl/sac_bc_compare/manual_bc_actor_seed2026 --timesteps 10000 --seed 2026 --init-mode bc_actor --bc-model outputs/bc_reach/fit_seed2026/best_bc.pt --boundary-mode terminate --eval-at-start --eval-every 1000 --eval-episodes 20 --eval-seed-start 50000 --checkpoint-every 1000 --tensorboard --label bc_actor
```

每1000步评估并保存，因此包含0/1000/2000/5000/10000；评估另耗墙钟时间但不计训练步数。若BC从第0步明显退化，应Ctrl+C并分析保存的Actor/Critic/TD证据，勿盲目扩大预算。

另开终端启动TensorBoard，在浏览器访问http://127.0.0.1:6006：

```bash
conda activate realman_rl
cd /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I -m tensorboard.main --logdir outputs/rl/sac_bc_compare --host 127.0.0.1 --port 6006
```

查看真实指标、独立加载评估、导出图片：

```bash
python -I -m json.tool outputs/rl/sac_bc_compare/manual_bc_actor_seed2026/evaluations.json
python -I -m json.tool outputs/rl/sac_bc_compare/manual_bc_actor_seed2026/summary.json
python -I scripts/evaluate_sac_reach.py --model outputs/rl/sac_bc_compare/manual_bc_actor_seed2026/final/model.zip --episodes 100 --seed-start 70000 --policies sac --json outputs/rl/sac_bc_compare/manual_bc_actor_seed2026/heldout_evaluation.json
python -I scripts/plot_sac_training.py --run-dir outputs/rl/sac_bc_compare/manual_bc_actor_seed2026
python -I scripts/plot_sac_training.py --run-dir outputs/rl/sac_bc_compare/manual_scratch_seed2026
```

相同CLI参数已以128步、3个监控Episode、加载评估和PNG短测试实际运行；JSON查看和TensorBoard inspect/help已执行。正式10k命令未执行，TensorBoard网页未打开。历史04-A/B/C预算或配置不同时仅作参考，不当作本轮严格A/B对照。
