"""关节桌面 PPO 的策略、持续相关噪声、checkpoint 操作及评估适配。"""
import copy
from pathlib import Path
import json

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import StateDependentNoiseDistribution

from ..envs.grasp.tabletop_pick_env import TabletopPickEnv, PPOTabletopEnv, ENCODING, encode, ACTION_CONTRACT
from ..paths import PROJECT_ROOT,TABLETOP_PICK_SCENE
from ..utils.artifacts import file_hashes
from ..tasks.grasp.tabletop_joint import TABLETOP_PICK_CONFIG, CURRICULUM, REWARD, environment_config, features, potential, shaped_reward
from ..evaluation.grasp_lift import rollout
from ..evaluation.metrics import summarize_grasp_episodes
from .common.env_factory import grasp_training_vector
from .common.artifacts import append_json, write_json, parameter_hashes, optimizer_steps

PPO_PARAMETERS=dict(learning_rate=3e-4,n_steps=256,batch_size=256,n_epochs=10,
    gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.001,vf_coef=.5,max_grad_norm=.5,
    target_kl=.05,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])),device='cpu')


class CorrelatedPositionNoise(StateDependentNoiseDistribution):
    """在 SB3 持续探索噪声上施加可逆的七维线性变换。

    base 提供 gSDE 的原取样机制，transform 为满秩 7×7 矩阵，log_std 为原对数标准差。
    proba_distribution、log_prob 和 entropy 使用变换后的完整协方差。
    """
    def __init__(self,base,transform,log_std):
        super().__init__(base.action_dim,full_std=base.full_std,use_expln=base.use_expln,
                         squash_output=False,learn_features=False,epsilon=0.)
        self.latent_sde_dim=base.latent_sde_dim
        matrix=np.asarray(transform,dtype=np.float32)
        if matrix.shape!=(7,7) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
            raise ValueError('A finite invertible seven-action exploration transform is required')
        self.transform=torch.as_tensor(matrix)
        self.sample_weights(log_std)

    def proba_distribution(self,mean_actions,log_std,latent_sde):
        super().proba_distribution(mean_actions,log_std,latent_sde)
        matrix=self.transform.to(mean_actions.device)
        # 线性变换同时作用于采样与打分，协方差按 AΣAᵀ 变换。
        covariance=(matrix*self.distribution.variance.unsqueeze(-2))@matrix.T
        self.distribution=torch.distributions.MultivariateNormal(mean_actions,covariance_matrix=covariance)
        return self

    def get_noise(self,latent_sde):
        noise=super().get_noise(latent_sde)
        return noise@self.transform.to(noise.device).T

    def log_prob(self,actions,gaussian_actions=None):return self.distribution.log_prob(actions)

    def entropy(self):return self.distribution.entropy()


class SmoothJointPolicy(ActorCriticPolicy):
    """预测六轴绝对位置（度）与夹爪归一化开口。

    exploration_transform 定义相关噪声的坐标变换，action_anchor 为可选动作均值锚点。
    Actor 使用 MLP 预测动作，Critic 保留独立价值网络。持续噪声的固定单位范数输入
    让接触抖动不再调制同一噪声刷新区间的采样偏移。
    """
    def __init__(self,*args,exploration_transform=None,action_anchor=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.exploration_transform=exploration_transform
        self.has_affine_action_mean=action_anchor is not None
        if exploration_transform is not None:
            if not isinstance(self.action_dist,StateDependentNoiseDistribution):
                raise ValueError('Correlated exploration requires the SDE sampler')
            self.action_dist=CorrelatedPositionNoise(self.action_dist,exploration_transform,self.log_std)
        if action_anchor is not None:
            anchor=np.asarray(action_anchor,dtype=np.float32)
            if anchor.shape!=(7,) or not np.isfinite(anchor).all() or exploration_transform is None:
                raise ValueError('Affine action means require a finite anchor and a full-rank transform')
            self.register_buffer('action_anchor',torch.as_tensor(anchor))
            self.register_buffer('action_mean_transform',torch.as_tensor(np.asarray(exploration_transform,dtype=np.float32)))

    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters()
        result['exploration_transform']=self.exploration_transform
        result['action_anchor']=self.action_anchor.detach().cpu().tolist() if self.has_affine_action_mean else None
        return result

    def _get_action_dist_from_latent(self,latent_pi):
        if isinstance(self.action_dist,StateDependentNoiseDistribution):
            mean=self.action_net(latent_pi)
            if self.has_affine_action_mean:mean=self.action_anchor+mean@self.action_mean_transform.T
            stable=torch.ones_like(latent_pi)/(latent_pi.shape[-1]**.5)
            return self.action_dist.proba_distribution(mean,self.log_std,stable)
        return super()._get_action_dist_from_latent(latent_pi)


def make_training_env(config,n_envs,seed,reward=None):
    if not 1<=n_envs<=16:raise ValueError('1..16 environments required')
    return grasp_training_vector(lambda: PPOTabletopEnv(config,reward), n_envs, seed)


def reset_exploration_std(model, observation, *, joint_std, gripper_std, smooth):
    """按对角高斯或 gSDE 的尺度重置各动作通道的探索标准差。"""
    with torch.no_grad():
        if model.policy.log_std.ndim == 1:
            model.policy.log_std[:6].fill_(float(np.log(joint_std)))
            model.policy.log_std[6].fill_(float(np.log(gripper_std)))
        else:
            tensor, _ = model.policy.obs_to_tensor(observation)
            features = model.policy.extract_features(tensor)
            latent, _ = model.policy.mlp_extractor(features)
            norm = 1. if smooth else max(float(latent.square().sum().sqrt()), 1e-6)
            model.policy.log_std[:, :6].fill_(float(np.log(joint_std / norm)))
            model.policy.log_std[:, 6].fill_(float(np.log(gripper_std / norm)))


def make_model(env, seed, parameters, *, smooth, initial_observation, reference_rad,
               joint_std, gripper_std, gripper_mean):
    """创建关节位置 PPO，把初始动作均值锚定到 reset 的关节参考。"""
    model = PPO(SmoothJointPolicy if smooth else 'MlpPolicy', env, seed=seed, verbose=0, **parameters)
    reset_exploration_std(model, initial_observation, joint_std=joint_std,
                         gripper_std=gripper_std, smooth=smooth)
    with torch.no_grad():
        model.policy.action_net.weight[:6].zero_()
        if smooth and model.policy.has_affine_action_mean:
            model.policy.action_net.bias[:6].zero_()
        else:
            model.policy.action_net.bias[:6] = torch.as_tensor(np.rad2deg(reference_rad), dtype=torch.float32)
        model.policy.action_net.bias[6] = gripper_mean
    return model


def metadata():
    assets={TABLETOP_PICK_SCENE}
    for folder in ('assets/robots/rml63','assets/grippers/eg2_4c2'):
        assets.update(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file())
    sources=[PROJECT_ROOT/'src/realman_rl_simtoreal'/p for p in (
        'envs/grasp/tabletop_pick_env.py','tasks/grasp/tabletop_joint.py','envs/grasp/grasp_lift_env.py','rl/ppo_tabletop.py',
        'simulation/physics/model.py','simulation/physics/joint_control.py','simulation/motion/kinematics.py',
        'simulation/physics/grasp_contact.py','simulation/devices/mujoco_gripper.py','evaluation/grasp_lift.py')]
    return dict(assets=file_hashes(sorted(assets),root=PROJECT_ROOT),
        sources=file_hashes(sorted(sources),root=PROJECT_ROOT),observation_dim=56,action_dim=7,
        action_semantics=[f'joint_{j}_position_deg' for j in range(1,7)]+['absolute_gripper_opening'])


def validate_model(model,config):
    if config.get('task')!='tabletop_pick' or config.get('algorithm')!='PPO':raise ValueError('A tabletop PPO checkpoint is required')
    if model.observation_space.shape!=(56,) or model.action_space.shape!=(7,):raise ValueError('Expected 56 observations and seven actions')
    if config.get('action_contract')!=ACTION_CONTRACT:raise ValueError('Incompatible joint-action semantics')
    if not np.array_equal(model.action_space.low,config['action_space_low']) or not np.array_equal(model.action_space.high,config['action_space_high']):
        raise ValueError('Joint-action bounds mismatch')
    if config.get('encoding')!=ENCODING or model.gamma!=config['reward']['gamma']:raise ValueError('Tabletop observation/discount contract mismatch')
    if getattr(model,'tabletop_contract',None)!=dict(encoding=ENCODING,reward=config['reward'],action_contract=ACTION_CONTRACT):
        raise ValueError('Tabletop model contract mismatch')
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):raise ValueError('Non-finite policy parameters')
    std=model.policy.log_std.detach().exp()
    if not torch.isfinite(std).all() or not (std>0).all():raise ValueError('Invalid action standard deviations')


def save_checkpoint(model,folder,config,*,complete=True):
    validate_model(model,config);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.save(folder/'model.zip');write_json(folder/'config.json',config)
    write_json(folder/'state.json',dict(actual_timesteps=int(model.num_timesteps),epoch_updates=int(model._n_updates),
        optimizer_steps=optimizer_steps(model),parameter_hashes=parameter_hashes(model),
        completed_update_stage=complete and model.num_timesteps>0,
        stage_kind='initial' if model.num_timesteps==0 else 'post_update' if complete else 'partial_or_interrupted'))


def load_checkpoint(path):
    path=Path(path);config=json.loads((path.parent/'config.json').read_text());torch.set_num_threads(1)
    # Earlier checkpoints used a fixed 5 mm spherical waypoint test.
    # Preserve their vertical tolerance unless a new training run overrides it.
    config['environment'].setdefault('approach_waypoint_tolerance',.005)
    model=PPO.load(path,device='cpu');validate_model(model,config)
    return model,config


def evaluate_model(model,config,seeds,*,full=False,tick=None):
    validate_model(model,config)
    environment=copy.deepcopy(config['environment'])
    if full:
        for key in ('start_fraction','initial_width','pregrasp_height','lift_height','height_tolerance',
                    'hold_seconds','max_episode_steps','object_radius','goal_height'):
            environment[key]=TABLETOP_PICK_CONFIG[key]
    before=parameter_hashes(model);counters=(model.num_timesteps,model._n_updates,optimizer_steps(model));raw=[]
    class Policy:
        def predict(self,obs,deterministic):
            if not deterministic:raise ValueError('Deterministic evaluation required')
            encoded=encode(obs);action,state=model.predict(encoded,deterministic=True)
            with torch.no_grad():
                tensors,_=model.policy.obs_to_tensor(encoded)
                value=model.policy.get_distribution(tensors).get_actions(deterministic=True).cpu().numpy().reshape(7)
            np.testing.assert_array_equal(np.clip(value,model.action_space.low,model.action_space.high),action);raw.append(value.tolist())
            return action,state
    with TabletopPickEnv(**environment) as env:
        episodes=[rollout(env,seed,model=Policy(),tick=tick) for seed in seeds]
    offset=0
    for episode in episodes:
        for record,value in zip(episode['records'],raw[offset:offset+episode['length']]):record['raw_action']=value
        offset+=episode['length']
        if environment['start_fraction']==1:
            np.testing.assert_allclose(episode['records'][0]['initial_q_deg'],environment['initial_q_deg'],rtol=0,atol=1e-10)
    if before!=parameter_hashes(model) or counters!=(model.num_timesteps,model._n_updates,optimizer_steps(model)):
        raise ValueError('Evaluation changed the PPO model')
    return dict(schema=1,task='tabletop_pick',policy='PPO',environment=environment,
        seeds=[e['seed'] for e in episodes],summary=summarize_grasp_episodes(episodes),episodes=episodes,
        parameters_unchanged=True,staged_controller_used=False,encoding=ENCODING,**metadata())


class TabletopAudit(BaseCallback):
    def __init__(self,folder,n_envs):
        super().__init__();self.folder=Path(folder);self.episodes=[];self.rollouts=0;self.abort_reason=None
        self.returns=np.zeros(n_envs);self.lengths=np.zeros(n_envs,dtype=int)
        self.height=np.zeros(n_envs);self.grip=np.zeros(n_envs);self.hold=np.zeros(n_envs)
        self.rejected=0;self.samples=0;self.box_clipped=0

    def _on_step(self):
        raw=np.asarray(self.locals['actions']);applied=np.asarray(self.locals['clipped_actions']);infos=self.locals['infos']
        low,high=self.model.action_space.low,self.model.action_space.high
        if raw.shape[-1]!=7 or not np.isfinite(raw).all() or not np.array_equal(applied,np.clip(raw,low,high)):
            raise ValueError('Invalid stock PPO seven-action clipping')
        self.samples+=len(raw);self.box_clipped+=int(((raw<low)|(raw>high)).any(axis=1).sum())
        self.rejected+=sum(i['action_rejected'] for i in infos)
        self.lengths+=1
        for rank,info in enumerate(infos):
            self.returns[rank]+=info['ppo_reward'];self.height[rank]=max(self.height[rank],info['support_clearance_m'])
            self.grip[rank]=max(self.grip[rank],min(info['contact_forces_n'][:2]));self.hold[rank]=max(self.hold[rank],info['max_hold_time_s'])
            if not info['observation_valid'] or info['termination_reason']=='numerical_failure':self.abort_reason='numerical_failure'
            if self.locals['dones'][rank]:
                row=dict(end_step=int(self.model.num_timesteps),rank=rank,length=int(self.lengths[rank]),
                    return_=float(self.returns[rank]),success=info['is_success'],reason=info['termination_reason'],detail=info['detail'],
                    safety_failure=info['is_safety_failure'],peak_clearance_m=float(self.height[rank]),
                    peak_bilateral_force_n=float(self.grip[rank]),max_hold_time_s=float(self.hold[rank]),
                    initial_q_deg=info['initial_q_deg'],start_fraction=info['start_fraction'])
                self.episodes.append(row);append_json(self.folder/'training_episodes.jsonl',row)
                self.returns[rank]=0;self.lengths[rank]=0;self.height[rank]=self.grip[rank]=self.hold[rank]=0
        append_json(self.folder/'actions.jsonl',dict(step=int(self.model.num_timesteps),raw_actions=raw.tolist(),
            clipped_actions=applied.tolist(),joint_targets=[i['command_joint_position_rad'] for i in infos],
            arm_action_rejected=[i['action_rejected'] for i in infos],physics_steps=[i['physics_steps'] for i in infos]))
        return self.abort_reason is None

    def _on_rollout_end(self):
        if not self.model.rollout_buffer.full:raise ValueError('Incomplete PPO rollout')
        self.rollouts+=1
        for key in ('advantages','returns','values','log_probs','rewards','actions'):
            if not np.isfinite(getattr(self.model.rollout_buffer,key)).all():raise ValueError('Non-finite PPO '+key)

    def summary(self):
        tail=self.episodes[-100:]
        return dict(episodes=len(self.episodes),successes=sum(e['success'] for e in self.episodes),
            recent_success_rate=float(np.mean([e['success'] for e in tail])) if tail else None,
            contact_episodes=sum(e['peak_bilateral_force_n']>.005 for e in self.episodes),
            lifted_episodes=sum(e['peak_clearance_m']>.003 for e in self.episodes),
            max_clearance_m=max((e['peak_clearance_m'] for e in self.episodes),default=0),
            max_hold_time_s=max((e['max_hold_time_s'] for e in self.episodes),default=0),
            rejected_fraction=self.rejected/self.samples if self.samples else 0,
            box_clip_fraction=self.box_clipped/self.samples if self.samples else 0,
            collected_rollouts=self.rollouts,abort_reason=self.abort_reason)
