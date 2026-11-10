"""Official PPO Reach helpers and read-only rollout audit; no custom loss/buffer."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .common.metrics import MotionStats
from ..tasks.reach import task_environment, ENV_CONFIG
from .common.observation import ReachTransform
from .common.evaluation import evaluate, summarize_episodes
from .common.artifacts import write_json, append_json


PPO_CONFIG=dict(learning_rate=3e-4,n_steps=512,batch_size=64,n_epochs=10,gamma=.99,
    gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
    policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256])),device='cpu')
TRAIN_METRICS=('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction',
               'explained_variance','std','loss','learning_rate','clip_range')


def environment_config(task='fixed_reach',reward_mode='reach_shaped_v1',action_scale=.00025):
    return task_environment(task,dict(ENV_CONFIG,boundary_mode='terminate'),
                            reward_mode=reward_mode,action_scale=action_scale)


def make_ppo(env,seed,*,n_steps=512,batch_size=64,n_epochs=10):
    if seed<0 or min(n_steps,batch_size)<2 or n_epochs<1 or n_steps%batch_size:
        raise ValueError('Nonnegative seed, n_steps/batch>=2, positive epochs and full minibatches required')
    if env.num_envs!=1:raise ValueError('This experiment currently audits one environment')
    parameters=copy.deepcopy(PPO_CONFIG)
    parameters.update(n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs)
    return PPO('MultiInputPolicy',env,seed=seed,verbose=0,**parameters),parameters


def parameter_hashes(model):
    groups={key:hashlib.sha256() for key in ('actor','critic','all')}
    for name,tensor in model.policy.state_dict().items():
        data=name.encode()+tensor.detach().cpu().contiguous().numpy().tobytes()
        groups['all'].update(data)
        if name.startswith(('mlp_extractor.policy_net.','action_net.','log_std')):groups['actor'].update(data)
        if name.startswith(('mlp_extractor.value_net.','value_net.')):groups['critic'].update(data)
    return {key:value.hexdigest() for key,value in groups.items()}


def optimizer_steps(model):
    # Adam's own per-parameter counters: different from PPO._n_updates (epochs).
    return max((int(state['step'].item()) for state in model.policy.optimizer.state.values() if 'step' in state),default=0)


def crossed_thresholds(previous,current,interval):
    if interval<1:raise ValueError('Positive scheduling interval required')
    return list(range((previous//interval+1)*interval,current+1,interval))


class ActionClipStats:
    """Environment Box clipping; never PPO likelihood-ratio clip_fraction."""
    def __init__(self):
        self.samples=self.components=self.clipped_vectors=self.clipped_components=0
        self.raw_abs_max=self.applied_abs_max=0.

    def observe(self,raw,applied,low,high):
        raw,applied=np.asarray(raw).reshape(-1,3),np.asarray(applied).reshape(-1,3)
        if not np.isfinite(raw).all() or not np.isfinite(applied).all():
            raise FloatingPointError('Non-finite PPO action')
        if not np.array_equal(applied,np.clip(raw,low,high)):
            raise ValueError('Actual PPO env action differs from stock Box clipping')
        clipped=(raw<low)|(raw>high)
        self.samples+=len(raw);self.components+=raw.size
        self.clipped_vectors+=int(np.any(clipped,axis=1).sum());self.clipped_components+=int(clipped.sum())
        self.raw_abs_max=max(self.raw_abs_max,float(np.abs(raw).max()))
        self.applied_abs_max=max(self.applied_abs_max,float(np.abs(applied).max()))

    def summary(self):
        return dict(samples=self.samples,clipped_vectors=self.clipped_vectors,clipped_components=self.clipped_components,
            vector_fraction=self.clipped_vectors/self.samples if self.samples else None,
            component_fraction=self.clipped_components/self.components if self.components else None,
            raw_abs_max=self.raw_abs_max,applied_abs_max=self.applied_abs_max)


class PPOAudit(BaseCallback):
    def __init__(self,folder,action_scale):
        super().__init__()
        self.folder=Path(folder);self.action_scale=action_scale
        self.episodes=[];self.partial_rewards=[];self.partial_actions=[]
        self.motion=MotionStats();self.clipping=ActionClipStats();self.rollout_clipping=ActionClipStats()
        self.complete_rollouts=0;self.abort_reason=None;self.partial_rejections=0;self.rejections=0

    def _on_rollout_start(self):
        self.rollout_clipping=ActionClipStats()

    def _on_step(self):
        raw=np.asarray(self.locals['actions'][0]);applied=np.asarray(self.locals['clipped_actions'][0])
        for stats in (self.clipping,self.rollout_clipping):
            stats.observe(raw,applied,self.model.action_space.low,self.model.action_space.high)
        reward=float(self.locals['rewards'][0]);info=self.locals['infos'][0]
        if not np.isfinite(reward):raise FloatingPointError('Non-finite environment reward')
        done=bool(self.locals['dones'][0]);old=self.model._last_obs
        next_obs=info.get('terminal_observation',self.locals['new_obs'])
        before=np.asarray(old['achieved_goal'][0]);after=np.asarray(next_obs['achieved_goal']).reshape(-1,3)[0]
        goal=old['desired_goal'][0]
        command=np.asarray(info['attempted_position_m'])-applied*self.action_scale
        self.motion.observe(before,after,goal,command,info['command_position_m'],applied,done)
        self.partial_rewards.append(reward);self.partial_actions.append(float(np.linalg.norm(applied)))
        rejected=bool(info.get('is_action_rejected',False));self.rejections+=int(rejected);self.partial_rejections+=int(rejected)
        append_json(self.folder/'actions.jsonl',dict(step=int(self.model.num_timesteps),
            raw_action=raw.tolist(),applied_action=applied.tolist(),box_clipped=bool(np.any(raw!=applied)),
            actual_before_m=before.tolist(),actual_after_m=after.tolist(),goal_m=goal.tolist(),
            command_before_m=command.tolist(),command_after_m=info['command_position_m'].tolist(),
            reward=reward,physics_steps=int(info['physics_steps']),simulation_time_s=info['simulation_time_s'],
            reason=info['termination_reason'],is_success=info['is_success'],
            is_safety_failure=info['is_safety_failure'],is_action_rejected=rejected,
            terminated=done and not bool(info.get('TimeLimit.truncated',False)),
            truncated=bool(info.get('TimeLimit.truncated',False))))
        if done:
            rewards=self.partial_rewards
            row=dict(end_step=int(self.model.num_timesteps),length=len(rewards),return_=float(sum(rewards)),
                discounted_return=float(np.dot(rewards,self.model.gamma**np.arange(len(rewards)))),
                success=bool(info['is_success']),safety_failure=bool(info['is_safety_failure']),
                reason=info['termination_reason'],final_distance_m=info['distance_m'],
                rejected_actions=self.partial_rejections,mean_action_norm=float(np.mean(self.partial_actions)))
            self.episodes.append(row);append_json(self.folder/'training_episodes.jsonl',row)
            self.partial_rewards=[];self.partial_actions=[];self.partial_rejections=0
        if info['termination_reason'] in ('collision','numerical_failure'):
            self.abort_reason='physical '+info['termination_reason']
        return self.abort_reason is None

    def _on_rollout_end(self):
        buffer=self.model.rollout_buffer
        if not buffer.full:raise ValueError('PPO reported a partial rollout as complete')
        for name in ('advantages','returns','values','log_probs','rewards','actions'):
            if not np.isfinite(getattr(buffer,name)).all():
                raise FloatingPointError('Non-finite PPO rollout '+name)
        self.complete_rollouts+=1

    def summary(self):
        result=summarize_episodes(self.episodes)
        result.update(partial_episode_steps=len(self.partial_rewards),rejected_actions=self.rejections,
                      collected_rollouts=self.complete_rollouts,abort_reason=self.abort_reason,
                      motion=self.motion.summary(),box_clipping=self.clipping.summary())
        return result

    def record_scalars(self):
        summary=self.summary();tail=self.episodes[-100:]
        if tail:
            for tag,key in [('episode_reward_mean','return_'),('episode_length_mean','length'),('episode_success_rate','success')]:
                self.logger.record('training/'+tag,float(np.mean([e[key] for e in tail])))
        for reason,count in summary['termination_counts'].items():
            self.logger.record('training/termination/'+reason,count)
        self.logger.record('training/action_rejection_fraction',self.rejections/max(1,self.model.num_timesteps))
        self.logger.record('training/safety_failure_fraction',summary['safety_failures']/max(1,self.model.num_timesteps))
        for key,value in self.motion.summary().items():
            if value is not None:self.logger.record('training/motion/'+key,value)
        for prefix,stats in [('actions/box_clip_total/',self.clipping),('actions/box_clip_rollout/',self.rollout_clipping)]:
            for key,value in stats.summary().items():
                if value is not None:self.logger.record(prefix+key,value)


def post_update_metrics(model,audit):
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise FloatingPointError('Non-finite PPO network parameter')
    result=dict(step=int(model.num_timesteps),epoch_updates=int(model._n_updates),optimizer_steps=optimizer_steps(model),
                collected_rollouts=audit.complete_rollouts,box_clipping=audit.clipping.summary(),
                rollout_box_clipping=audit.rollout_clipping.summary())
    for key in TRAIN_METRICS:
        value=model.logger.name_to_value.get('train/'+key)
        if value is not None:
            value=float(value)
            if not np.isfinite(value):
                if key=='explained_variance' and np.var(model.rollout_buffer.returns)==0:
                    result[key]=None;result['explained_variance_status']='undefined: constant returns'
                    model.logger.name_to_value.pop('train/'+key,None)
                    continue
                raise FloatingPointError('Non-finite PPO training metric '+key)
            result[key]=value
    std=model.policy.log_std.detach().exp().cpu().numpy()
    if not np.isfinite(std).all():raise FloatingPointError('Non-finite PPO action std')
    result['policy_std']=std.tolist();result['policy_std_mean']=float(std.mean())
    model.logger.record('policy/action_std_mean',result['policy_std_mean'])
    for i,value in enumerate(std):model.logger.record(f'policy/action_std_{i}',float(value))
    return result


def evaluate_ppo(model,transform,environment,seeds):
    before=parameter_hashes(model);counters=(model.num_timesteps,model._n_updates,optimizer_steps(model))
    clipping=ActionClipStats()
    class ObservedPolicy:
        def predict(self,obs,deterministic):
            if not deterministic:raise ValueError('Independent PPO evaluation must be deterministic')
            action,state=model.predict(obs,deterministic=True)
            with torch.no_grad():
                tensors,_=model.policy.obs_to_tensor(obs)
                raw=model.policy.get_distribution(tensors).get_actions(deterministic=True).cpu().numpy()
            clipping.observe(raw,action,model.action_space.low,model.action_space.high)
            return action,state
    measured=evaluate('ppo',list(seeds),model=ObservedPolicy(),transform=transform,config=environment,
                      gamma=model.gamma,collect_motion=True)
    if parameter_hashes(model)!=before or counters!=(model.num_timesteps,model._n_updates,optimizer_steps(model)):
        raise ValueError('Evaluation changed PPO parameters or optimization counters')
    measured.update(box_clipping=clipping.summary(),parameters_unchanged=True,
        goal_scope='one fixed target; repeated seeds do not test target generalization' if environment['target_mode']=='fixed'
                   else 'independent seeded random reachable targets')
    return measured


def save_ppo_checkpoint(model,folder,config,trained_rollouts,*,completed_stage):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.save(folder/'model.zip')
    write_json(folder/'config.json',config)
    write_json(folder/'observation_transform.json',config['observation_transform'])
    write_json(folder/'state.json',dict(algorithm='PPO',actual_timesteps=int(model.num_timesteps),
        epoch_updates=int(model._n_updates),optimizer_steps=optimizer_steps(model),
        trained_rollouts=trained_rollouts,completed_update_stage=completed_stage and model.num_timesteps>0,
        stage_kind='initial' if model.num_timesteps==0 else 'post_update' if completed_stage else 'partial_or_interrupted'))


def load_ppo(path):
    path=Path(path).resolve();config=json.loads((path.parent/'config.json').read_text())
    if config.get('algorithm')!='PPO':raise ValueError('Expected an explicitly marked PPO checkpoint')
    torch.set_num_threads(config['torch_num_threads'])
    model=PPO.load(path,device='cpu')
    return model,ReachTransform(config['observation_transform']),config
