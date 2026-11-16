"""PPO-only joint-space tabletop pickup, goal features and immutable evaluation."""
import copy
import hashlib
from pathlib import Path
import json

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecCheckNan

from ..envs.tabletop_pick_env import TabletopPickEnv
from ..paths import PROJECT_ROOT,TABLETOP_PICK_SCENE
from ..tasks.tabletop_pick import TABLETOP_PICK_CONFIG
from ..evaluation.grasp_lift import rollout
from ..evaluation.metrics import summarize_grasp_episodes
from .ppo_grasp import parameter_hashes,optimizer_steps
from .common.artifacts import append_json,write_json

ENCODING=dict(name='joint_goal_tabletop_v1',input_dim=56,output_dim=56,
    position_scale_m=.01,command_lag_scale_m=.001,joint_goal_scale_rad=.05,running_normalization=False)
ACTION_CONTRACT=dict(name='absolute_joint_positions_deg_v1',dimension=7,
    joint_units='degree',jaw_units='normalized_absolute_opening',rate_limited=True)
REWARD=dict(name='tabletop_pick_action_regularized_v6',gamma=.995,scale=.01,success=1000.,safety=50.,drop=10.,
    joint_potential=10.,reach_potential=1.,orientation_potential=.5,aperture_potential=1.,
    grip_potential=2.,height_potential=8.,hold_potential=2.,step=.01,rejection=.05,
    dense_joint=1.,dense_reach=.2,dense_aperture=.2,dense_grip=.5,dense_height=.8,dense_hold=.1,speed_cost=.1,
    binary_contact_joint_goal=True,pregrasp_joint_weight=.1,
    approach_joint_scale=.2,grasp_joint_scale=.03,action_target_weight=1.,action_target_scale_rad=.1,
    far_jaw_action_weight=.5)
CURRICULUM=dict(
    pinch=dict(start_fraction=0.,pregrasp_height=.003,initial_width=.022,lift_height=.005,
               height_tolerance=.0005,hold_seconds=.2,max_episode_steps=300,object_radius=0.),
    approach=dict(start_fraction=0.,pregrasp_height=.03,initial_width=.06,lift_height=.005,
                  height_tolerance=.0005,hold_seconds=.2,max_episode_steps=500,object_radius=0.),
    lift=dict(start_fraction=0.,pregrasp_height=.03,object_radius=0.,max_episode_steps=700),
    unfold25=dict(start_fraction=.25,object_radius=0.,max_episode_steps=900),
    unfold50=dict(start_fraction=.5,object_radius=0.,max_episode_steps=1000),
    unfold75=dict(start_fraction=.75,object_radius=0.,max_episode_steps=1100),
    full=dict(),full_random=dict(object_radius=.02))
PPO_PARAMETERS=dict(learning_rate=3e-4,n_steps=256,batch_size=256,n_epochs=10,
    gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.001,vf_coef=.5,max_grad_norm=.5,
    target_kl=.05,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])),device='cpu')


class CorrelatedPositionNoise(StateDependentNoiseDistribution):
    """Full-rank correlated Gaussian with the stock persistent-noise sampler."""
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
        covariance=(matrix*self.distribution.variance.unsqueeze(-2))@matrix.T
        self.distribution=torch.distributions.MultivariateNormal(mean_actions,covariance_matrix=covariance)
        return self

    def get_noise(self,latent_sde):
        noise=super().get_noise(latent_sde)
        return noise@self.transform.to(noise.device).T

    def log_prob(self,actions,gaussian_actions=None):return self.distribution.log_prob(actions)

    def entropy(self):return self.distribution.entropy()


class SmoothJointPolicy(ActorCriticPolicy):
    """MLP action means with persistent Gaussian position exploration.

    Stock PPO/gSDE still samples and scores every action. A constant unit-norm
    exploration feature prevents contact-force jitter from feeding back into
    the sampled position offset between scheduled noise refreshes.
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


def environment_config(stage='full',**overrides):
    if stage not in CURRICULUM:raise ValueError('Unknown tabletop curriculum')
    result=copy.deepcopy(TABLETOP_PICK_CONFIG);result.update(CURRICULUM[stage]);result.update(overrides)
    return result


def encode(observation):
    value=np.asarray(observation,dtype=np.float32).copy()
    if value.shape[-1:]!=(56,) or not np.isfinite(value).all():raise ValueError('Finite 56-vector required')
    value[...,12:21]=np.arcsinh(value[...,12:21]*5)
    value[...,38:41]=np.arcsinh(value[...,38:41]*10)
    value[...,44:56]=np.arcsinh(value[...,44:56]*2)
    return value


def features(info,config,reward=REWARD):
    grip=float(np.clip(min(info['contact_forces_n'][:2])/.05,0,1))
    distance=info['grasp_distance_m']
    fraction=float(np.clip((distance-.005)/.035,0,1))
    approaching=not info.get('approach_complete',True)
    if approaching:fraction=1.
    if info['bilateral_contact']:fraction=0.
    desired=.02+.04*fraction
    aperture=float(np.exp(-((info['actual_width_m']-desired)/.02)**2))
    goal_height=config.get('goal_height') or config['lift_height']
    height=float(np.clip(info['support_clearance_m']/goal_height,0,1))
    held=float(np.clip(info['hold_time_s']/config['hold_seconds'],0,1))
    error=np.asarray(info['object_position_m'])-info['object_goal_m']
    alignment=float(np.exp(-np.linalg.norm(error[:2])/.005))
    if reward.get('binary_contact_joint_goal',False):
        joint=info['lift_joint_error_rad'] if info['bilateral_contact'] else info.get('reach_joint_error_rad',info['grasp_joint_error_rad'])
    else:joint=(1-grip)*info['grasp_joint_error_rad']+grip*info['lift_joint_error_rad']
    joint_weight=1. if info['bilateral_contact'] or approaching else reward.get('pregrasp_joint_weight',1.)
    return dict(joint=joint,reach=float(np.exp(-info.get('reach_goal_distance_m',distance)/.08)),
        orientation=float(np.exp(-(info['orientation_error_rad']/.5)**2)),
        aperture=aperture,grip=grip,height=height*alignment,hold=held,joint_weight=joint_weight,
        joint_scale=reward.get('approach_joint_scale',.2) if approaching else reward.get('grasp_joint_scale',.03))


def potential(info,config,reward=REWARD):
    f=features(info,config,reward)
    return float(-reward['joint_potential']*f['joint']+reward['reach_potential']*f['reach']
        +reward['orientation_potential']*f['orientation']+reward['aperture_potential']*f['aperture']
        +reward['grip_potential']*f['grip']+reward['height_potential']*f['grip']*f['height']
        +reward['hold_potential']*f['hold'])


def shaped_reward(previous,info,terminated,config,reward=REWARD,*,action=None,action_info=None):
    f=features(info,config,reward)
    terms=dict(potential_change=reward['gamma']*(0 if terminated else potential(info,config,reward))-previous,
        step=-reward['step'],joint_distance=-.02*f['joint'],
        joint_alignment=reward.get('dense_joint',0)*f['joint_weight']*np.exp(-f['joint']/f['joint_scale']),
        reach=reward['dense_reach']*f['reach'],aperture=reward['dense_aperture']*f['aperture'],
        grip=reward['dense_grip']*f['grip'],height=reward['dense_height']*f['grip']*f['height'],
        hold=reward['dense_hold']*f['hold'],
        speed=-reward['speed_cost']*f['grip']*(f['height']>.9)*min(np.linalg.norm(info['object_velocity_m_s'])/.01,1),
        rejected=-reward['rejection']*info['action_rejected'],
        success=reward['success']*info['is_success'],safety=-reward['safety']*info['is_safety_failure'],
        dropped=-reward['drop']*(info['termination_reason'] in ('dropped','object_lost')))
    if reward.get('action_target_weight',0):
        if action is None or action_info is None:raise ValueError('Action regularization requires the observed pre-action goal')
        target=action_info['lift_goal_joint_rad'] if action_info['bilateral_contact'] else action_info['reach_goal_joint_rad']
        error=(np.deg2rad(np.asarray(action[:6],dtype=float))-target)/reward['action_target_scale_rad']
        terms['action_target']=-reward['action_target_weight']*float(error@error)
        if (not action_info['bilateral_contact'] and
                (not action_info.get('approach_complete',True) or action_info['grasp_distance_m']>.03)):
            from ..simulation.gripper import MAX_WIDTH
            open_action=2*config['initial_width']/MAX_WIDTH-1
            terms['far_jaw_action']=-reward.get('far_jaw_action_weight',0)*(float(action[6])-open_action)**2
    terms={k:float(v*reward['scale']) for k,v in terms.items()}
    result=float(sum(terms.values()))
    if not np.isfinite(result):raise ValueError('Non-finite tabletop reward')
    return result,terms


class PPOTabletopEnv(gym.Wrapper):
    def __init__(self,config,reward=None):
        super().__init__(TabletopPickEnv(**config))
        self.reward_config=copy.deepcopy(REWARD if reward is None else reward)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(56,),dtype=np.float32)

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self.previous=potential(info,self.env.config,self.reward_config)
        self.action_info=copy.deepcopy(info)
        return encode(obs),info

    def step(self,action):
        self.env.recorder.phase='ppo'
        obs,base,terminated,truncated,info=self.env.step(action)
        reward,terms=shaped_reward(self.previous,info,terminated,self.env.config,self.reward_config,
                                  action=action,action_info=self.action_info)
        self.previous=potential(info,self.env.config,self.reward_config)
        self.action_info=copy.deepcopy(info)
        info.update(environment_reward=base,ppo_reward=reward,reward_components=terms)
        return encode(obs),reward,terminated,truncated,info


def make_training_env(config,n_envs,seed,reward=None):
    if not 1<=n_envs<=16:raise ValueError('1..16 environments required')
    def factory():return Monitor(PPOTabletopEnv(config,reward))
    vector=DummyVecEnv([factory]) if n_envs==1 else SubprocVecEnv([factory for _ in range(n_envs)],start_method='fork')
    vector.seed(seed)
    return VecCheckNan(vector,raise_exception=True)


def metadata():
    assets={TABLETOP_PICK_SCENE}
    for folder in ('assets/robots/rml63','assets/grippers/eg2_4c2'):
        assets.update(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file())
    sources=[PROJECT_ROOT/'src/realman_rl_simtoreal'/p for p in (
        'envs/tabletop_pick_env.py','tasks/tabletop_pick.py','envs/grasp_lift_env.py','rl/ppo_tabletop.py',
        'simulation/model.py','simulation/control.py','simulation/kinematics.py',
        'simulation/grasp_contact.py','simulation/gripper.py','evaluation/grasp_lift.py')]
    def hashes(paths):return {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return dict(assets=hashes(assets),sources=hashes(sources),observation_dim=56,action_dim=7,
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
