"""PPO learns all four grasp actions; fixed encodings and potential shaping.

Training and inference never construct the staged grasp controller. Curriculum
stages change only the declared reset distance, lift height and hold duration.
"""
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

from ..envs.grasp_lift_env import GraspLiftEnv
from ..utils.artifacts import experiment_metadata
from ..tasks.grasp_lift import GRASP_LIFT_CONFIG, CURRICULUM, REWARD, aperture_score, goal_alignment, potential, shaped_reward
from ..evaluation.grasp_lift import evaluate
from .common.artifacts import append_json, write_json, parameter_hashes, optimizer_steps


ENCODING=dict(name='grasp_relative_v1',input_dim=41,output_dim=41,
    method='asinh on position errors/displacements at 1mm; all other features unchanged',
    position_slice=[12,21],command_lag_slice=[38,41],running_normalization=False)
PPO_PARAMETERS=dict(learning_rate=3e-4,n_steps=128,batch_size=256,n_epochs=10,
    gamma=.995,gae_lambda=.95,clip_range=.2,ent_coef=.001,vf_coef=.5,max_grad_norm=.5,
    target_kl=.05,policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128]),log_std_init=-1.),device='cpu')


def environment_config(stage='full'):
    if stage not in CURRICULUM:raise ValueError('Unknown grasp curriculum stage')
    return dict(GRASP_LIFT_CONFIG,**CURRICULUM[stage])


def encode(observation):
    value=np.asarray(observation,dtype=np.float32).copy()
    if value.shape[-1:]!=(41,) or not np.isfinite(value).all():
        raise ValueError('PPO grasp requires finite 41-dimensional observations')
    value[...,12:21]=np.arcsinh(value[...,12:21]*50.)
    value[...,38:41]=np.arcsinh(value[...,38:41]*10.)
    return value


class PPOGraspEnv(gym.Wrapper):
    """Fixed observation encoding and discount-consistent task reward."""
    def __init__(self,config,reward=None,randomize_height=False,randomize_approach=False):
        super().__init__(GraspLiftEnv(**config))
        self.task_config=self.env.config.copy()
        self.reward_config=copy.deepcopy(REWARD if reward is None else reward)
        self.randomize_height=bool(randomize_height)
        self.randomize_approach=bool(randomize_approach)
        if self.randomize_approach and self.task_config['approach_distance']<.005:
            raise ValueError('Random approach curriculum requires a maximum distance of at least 5mm')
        self._curriculum_rng=None
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(41,),dtype=np.float32)

    def reset(self,**kwargs):
        if self.randomize_height or self.randomize_approach:
            if kwargs.get('seed') is not None or self._curriculum_rng is None:
                seed=kwargs.get('seed')
                self._curriculum_rng=np.random.default_rng(None if seed is None else [int(seed),38117])
        if self.randomize_height:
            height=float(self._curriculum_rng.uniform(.005,self.task_config['lift_height']))
            self.env.config['lift_height']=height
            self.env.config['height_tolerance']=height*self.task_config['height_tolerance']/self.task_config['lift_height']
        if self.randomize_approach:
            self.env.config['approach_distance']=float(self._curriculum_rng.uniform(.005,self.task_config['approach_distance']))
        obs,info=self.env.reset(**kwargs)
        self.episode_config=self.env.config.copy()
        info['training_goal_height_m']=self.episode_config['lift_height']
        info['training_approach_distance_m']=self.episode_config['approach_distance']
        self.previous_phi=potential(info,self.episode_config,self.reward_config)
        return encode(obs),info

    def step(self,action):
        obs,base,terminated,truncated,info=self.env.step(action)
        reward,terms=shaped_reward(self.previous_phi,info,terminated,self.episode_config,self.reward_config)
        self.previous_phi=potential(info,self.episode_config,self.reward_config)
        info.update(environment_reward=base,environment_reward_components=info['reward_components'],
                    reward_components=terms,ppo_reward=reward,reward_mode=self.reward_config['name'],
                    training_goal_height_m=self.episode_config['lift_height'])
        info['training_approach_distance_m']=self.episode_config['approach_distance']
        return encode(obs),reward,terminated,truncated,info


def make_training_env(config,n_envs,seed,reward=None,randomize_height=False,randomize_approach=False):
    if not 1<=n_envs<=16:raise ValueError('Expected 1..16 environments')
    def factory():return Monitor(PPOGraspEnv(config,reward,randomize_height,randomize_approach))
    vector=DummyVecEnv([factory]) if n_envs==1 else SubprocVecEnv([factory for _ in range(n_envs)],start_method='fork')
    vector.seed(seed)
    return VecCheckNan(vector,raise_exception=True)


def validate_model(model,config):
    if model.observation_space.shape!=(41,) or model.action_space.shape!=(4,):
        raise ValueError('Expected a 41-observation, four-action grasp PPO checkpoint')
    if not np.all(model.action_space.low==-1) or not np.all(model.action_space.high==1):
        raise ValueError('PPO grasp requires the normalized four-action Box')
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise ValueError('Non-finite PPO grasp network parameters')
    std=model.policy.log_std.detach().exp()
    if not torch.isfinite(std).all() or not (std>0).all():raise ValueError('Invalid PPO action standard deviations')
    if (config.get('encoding')!=ENCODING or model.gamma!=config['reward']['gamma']
            or getattr(model,'grasp_ppo_contract',None)!=dict(encoding=ENCODING,reward=config['reward'])):
        raise ValueError('PPO checkpoint grasp observation/reward contract mismatch')


def save_checkpoint(model,folder,config,*,complete=True):
    validate_model(model,config)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.save(folder/'model.zip')
    write_json(folder/'config.json',config)
    write_json(folder/'state.json',dict(actual_timesteps=int(model.num_timesteps),epoch_updates=int(model._n_updates),
        optimizer_steps=optimizer_steps(model),completed_update_stage=complete and model.num_timesteps>0,
        stage_kind='initial' if model.num_timesteps==0 else 'post_update' if complete else 'partial_or_interrupted',
        parameter_hashes=parameter_hashes(model)))


def load_checkpoint(path):
    path=Path(path).resolve()
    config=json.loads((path.parent/'config.json').read_text())
    if config.get('algorithm')!='PPO' or config.get('task')!='grasp_lift':raise ValueError('Expected a grasp PPO checkpoint')
    torch.set_num_threads(1)
    model=PPO.load(path,device='cpu')
    validate_model(model,config)
    return model,config


def evaluate_model(model,config,seeds):
    validate_model(model,config)
    before=parameter_hashes(model);counters=(model.num_timesteps,model._n_updates,optimizer_steps(model))
    raw_actions=[]
    class EncodedPolicy:
        def predict(self,obs,deterministic):
            if not deterministic:raise ValueError('Independent evaluation must be deterministic')
            encoded=encode(obs)
            action,state=model.predict(encoded,deterministic=True)
            with torch.no_grad():
                tensors,_=model.policy.obs_to_tensor(encoded)
                raw=model.policy.get_distribution(tensors).get_actions(deterministic=True).cpu().numpy().reshape(4)
            if not np.array_equal(np.clip(raw,-1,1),action):raise ValueError('Unexpected PPO inference clipping')
            raw_actions.append(raw.tolist())
            return action,state
    result=evaluate(config['environment'],list(seeds),model=EncodedPolicy())
    if before!=parameter_hashes(model) or counters!=(model.num_timesteps,model._n_updates,optimizer_steps(model)):
        raise ValueError('Evaluation changed PPO parameters or counters')
    result.update(parameters_unchanged=True,encoding=ENCODING,model_actual_timesteps=int(model.num_timesteps),
        evaluation_reward_mode='original environment reward; checkpoint selection uses physical success')
    offset=0
    for episode in result['episodes']:
        for record,raw in zip(episode['records'],raw_actions[offset:offset+episode['length']]):record['raw_action']=raw
        offset+=episode['length']
    raw=np.asarray(raw_actions).reshape(-1,4)
    result['box_clipping']=dict(samples=len(raw),vector_fraction=float((abs(raw)>1).any(axis=1).mean()),
        component_fraction=float((abs(raw)>1).mean()),raw_abs_max=float(abs(raw).max()))
    return result


class GraspAudit(BaseCallback):
    def __init__(self,folder,n_envs):
        super().__init__()
        self.folder=Path(folder);self.episodes=[];self.rollouts=0;self.abort_reason=None
        self.returns=np.zeros(n_envs);self.lengths=np.zeros(n_envs,dtype=int)
        self.max_height=np.zeros(n_envs);self.max_contact=np.zeros(n_envs);self.max_hold=np.zeros(n_envs)
        self.clipped=0;self.samples=0

    def _on_step(self):
        raw=np.asarray(self.locals['actions']);applied=np.asarray(self.locals['clipped_actions'])
        if raw.shape[-1]!=4 or not np.isfinite(raw).all() or not np.array_equal(applied,np.clip(raw,-1,1)):
            raise ValueError('Invalid stock PPO four-action Box clipping')
        self.samples+=len(raw);self.clipped+=int((abs(raw)>1).any(axis=1).sum())
        infos=self.locals['infos']
        self.lengths+=1
        for rank,info in enumerate(infos):
            self.returns[rank]+=info['ppo_reward']
            self.max_height[rank]=max(self.max_height[rank],info['support_clearance_m'])
            self.max_contact[rank]=max(self.max_contact[rank],min(info['contact_forces_n'][:2]))
            self.max_hold[rank]=max(self.max_hold[rank],info['max_hold_time_s'])
            if not info['observation_valid'] or info['termination_reason']=='numerical_failure':
                self.abort_reason='numerical_failure'
            if self.locals['dones'][rank]:
                row=dict(end_step=int(self.model.num_timesteps),rank=rank,length=int(self.lengths[rank]),
                    return_=float(self.returns[rank]),success=info['is_success'],reason=info['termination_reason'],
                    detail=info['detail'],safety_failure=info['is_safety_failure'],
                    peak_clearance_m=float(self.max_height[rank]),peak_bilateral_force_n=float(self.max_contact[rank]),
                    max_hold_time_s=float(self.max_hold[rank]),final_clearance_m=info['support_clearance_m'])
                row['goal_height_m']=info['training_goal_height_m']
                row['approach_distance_m']=info['training_approach_distance_m']
                self.episodes.append(row);append_json(self.folder/'training_episodes.jsonl',row)
                self.returns[rank]=0;self.lengths[rank]=0
                self.max_height[rank]=self.max_contact[rank]=self.max_hold[rank]=0
        append_json(self.folder/'actions.jsonl',dict(step=int(self.model.num_timesteps),
            raw_actions=raw.tolist(),applied_actions=applied.tolist(),
            clearance_m=[i['support_clearance_m'] for i in infos],
            bilateral=[i['bilateral_contact'] for i in infos],
            hold_time_s=[i['hold_time_s'] for i in infos],physics_steps=[i['physics_steps'] for i in infos]))
        return self.abort_reason is None

    def _on_rollout_end(self):
        if not self.model.rollout_buffer.full:raise ValueError('Partial PPO rollout')
        self.rollouts+=1
        for key in ('advantages','returns','values','log_probs','rewards','actions'):
            if not np.isfinite(getattr(self.model.rollout_buffer,key)).all():raise ValueError('Non-finite rollout '+key)

    def summary(self):
        tail=self.episodes[-100:]
        return dict(episodes=len(self.episodes),recent_success_rate=float(np.mean([e['success'] for e in tail])) if tail else None,
            successes=sum(e['success'] for e in self.episodes),
            contact_episodes=sum(e['peak_bilateral_force_n']>.005 for e in self.episodes),
            lifted_episodes=sum(e['peak_clearance_m']>.003 for e in self.episodes),
            max_clearance_m=max((e['peak_clearance_m'] for e in self.episodes),default=0),
            box_clip_fraction=self.clipped/self.samples if self.samples else None,
            collected_rollouts=self.rollouts,partial_episode_steps=self.lengths.tolist(),abort_reason=self.abort_reason)
