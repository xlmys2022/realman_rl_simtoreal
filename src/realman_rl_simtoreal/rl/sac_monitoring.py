"""SAC-only training audits and read-only Q monitoring."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from .common.artifacts import append_json
from .common.evaluation import summarize_episodes
from .common.metrics import MotionStats
from .sac_diagnostics import q_diagnostic


class TrainingAudit(BaseCallback):
    """Read SB3 diagnostics and original env infos, never alter actions/rewards."""
    def __init__(self, folder, *, diagnostic_every=1000):
        super().__init__()
        self.folder = Path(folder)
        self.diagnostic_every = diagnostic_every
        self.episodes, self.diagnostics, self.rewards = [], [], []
        self.partial_rewards = []
        self.partial_actions = []
        self.abort_reason = None
        self.rejected_actions = 0
        self.rejection_reasons = Counter()
        self.partial_rejections = 0
        self.wall_start = perf_counter()

    def _on_step(self):
        rewards = self.locals['rewards']
        actions = self.locals['actions']
        if not np.isfinite(rewards).all() or not np.isfinite(actions).all():
            raise ValueError('Non-finite training reward/action')
        self.rewards.append(float(rewards[0]))
        self.partial_rewards.append(float(rewards[0]))
        self.partial_actions.append(float(np.linalg.norm(actions[0])))
        info = self.locals['infos'][0]
        if info.get('is_action_rejected', False):
            self.rejected_actions += 1
            self.partial_rejections += 1
            self.rejection_reasons[info['rejection_reason']] += 1
            append_json(self.folder/'rejected_actions.jsonl',
                        dict(step=int(self.num_timesteps), reason=info['rejection_reason'],
                             reward=float(rewards[0]), physics_steps=int(info['physics_steps'])))
        for key in ('train/actor_loss', 'train/critic_loss', 'train/ent_coef'):
            value = self.model.logger.name_to_value.get(key)
            if value is not None and not np.isfinite(value):
                self.abort_reason = f'nonfinite {key}'
        critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0)
        if critic_loss > 1e8:
            self.abort_reason = 'critic_loss exceeded predeclared 1e8 divergence guard'
        if info['termination_reason'] in ('numerical_failure', 'collision'):
            self.abort_reason = f"physical {info['termination_reason']}"
        if self.locals['dones'][0]:
            rewards = self.partial_rewards
            row = dict(end_step=int(self.num_timesteps), return_=float(sum(rewards)), length=len(rewards),
                       discounted_return=float(np.dot(rewards, self.model.gamma**np.arange(len(rewards)))),
                       success=bool(info['is_success']), safety_failure=bool(info['is_safety_failure']),
                       final_distance_m=float(info['distance_m']), reason=info['termination_reason'],
                       min_reward=min(rewards), mean_action_norm=float(np.mean(self.partial_actions)))
            row['rejected_actions'] = self.partial_rejections
            self.episodes.append(row)
            append_json(self.folder/'training_episodes.jsonl', row)
            self.partial_rewards, self.partial_actions = [], []
            self.partial_rejections = 0
            tail = self.episodes[-100:]
            if len(tail) == 100 and all(e['safety_failure'] for e in tail) and np.mean([e['length'] for e in tail]) <= 2:
                self.abort_reason = '100 consecutive safety-failed episodes of mean length <=2'
        return self.abort_reason is None

    def _on_rollout_end(self):
        if self.num_timesteps % self.diagnostic_every == 0:
            self.record_diagnostic()

    def record_diagnostic(self, scope='rollout_before_update'):
        result = dict(step=int(self.num_timesteps), updates=int(self.model._n_updates),
                      scope=scope, elapsed_wall_s=perf_counter()-self.wall_start, **q_diagnostic(self.model))
        for key, value in self.model.logger.name_to_value.items():
            if key.startswith('train/') and np.isscalar(value):
                result[key] = float(value)
        self.diagnostics.append(result)
        append_json(self.folder/'diagnostics.jsonl', result)
        return result

    def summary(self):
        values = np.asarray(self.rewards)
        summary = summarize_episodes(self.episodes)
        summary.update(rejected_actions=self.rejected_actions,rejection_reasons=dict(self.rejection_reasons))
        return dict(**summary, abort_reason=self.abort_reason,
                    transitions=len(values), partial_episode_steps=len(self.partial_rewards),
                    reward_distribution=dict(min=float(values.min()), median=float(np.median(values)),
                                             p01=float(np.percentile(values, 1)), p99=float(np.percentile(values, 99)),
                                             safety_sized_rewards=int(np.sum(values < -200))) if len(values) else None)


class MonitoredTrainingAudit(TrainingAudit):
    def __init__(self, folder, *, reference=None, transform=None, probe=None, diagnostic_every=1000,
                 collect_motion=False,action_scale=.001):
        super().__init__(folder,diagnostic_every=diagnostic_every)
        self.reference,self.transform,self.probe=reference,transform,probe
        self.motion=MotionStats() if collect_motion else None
        self.action_scale=action_scale

    def _on_step(self):
        result = super()._on_step()
        if self.motion is not None:
            info=self.locals['infos'][0]
            old=self.model._last_obs
            next_obs=info.get('terminal_observation',self.locals['new_obs'])
            action=np.asarray(self.locals['actions'][0])
            command=np.asarray(info['attempted_position_m'])-action*self.action_scale
            self.motion.observe(old['achieved_goal'][0],np.asarray(next_obs['achieved_goal']).reshape(-1,3)[0],
                old['desired_goal'][0],command,info['command_position_m'],action,bool(self.locals['dones'][0]))
        if self.locals['dones'][0]:
            self.record_training_scalars()
        return result

    def record_training_scalars(self):
        tail=self.episodes[-100:]
        self.logger.record('monitor/total_timesteps',self.model.num_timesteps)
        self.logger.record('training/action_rejection_fraction',self.rejected_actions/max(1,len(self.rewards)))
        self.logger.record('training/safety_failure_fraction',sum(e['safety_failure'] for e in self.episodes)/max(1,len(self.rewards)))
        if self.motion is not None:
            for key,value in self.motion.summary().items():
                if value is not None:self.logger.record('training/motion/'+key,value)
        if tail:
            self.logger.record('training/episode_reward_mean',np.mean([e['return_'] for e in tail]))
            self.logger.record('training/episode_length_mean',np.mean([e['length'] for e in tail]))
            self.logger.record('training/episode_success_rate',np.mean([e['success'] for e in tail]))
            self.logger.record('training/safety_failures_total',sum(e['safety_failure'] for e in self.episodes))

    def record_diagnostic(self, scope='rollout_before_update'):
        result=super().record_diagnostic(scope)
        self.record_training_scalars()
        if self.reference is not None:
            from ..imitation.bc_to_sac import action_deviation
            deviation=action_deviation(self.model,self.transform,self.reference,self.probe)
            result['bc_fixed_probe_deviation']=deviation
            append_json(self.folder/'bc_diagnostics.jsonl',dict(step=int(self.model.num_timesteps),
                        scope=scope,bc_fixed_probe_deviation=deviation))
            for key in ('mse','mae','max_abs'):
                self.logger.record('bc/fixed_probe_'+key,deviation[key])
        return result

    def summary(self):
        result=super().summary()
        if self.motion is not None:result['motion']=self.motion.summary()
        return result


def probe_q(model, transform, probe):
    """Finite Q on a fixed observation/action probe, including step 0 empty replay."""
    with torch.no_grad():
        obs={k:torch.as_tensor(v,device=model.device).float() for k,v in transform.encode(probe).items()}
        action=model.actor(obs,deterministic=True)
        q=torch.cat(model.critic(obs,action),dim=1)
        target=torch.cat(model.critic_target(obs,action),dim=1)
    if not torch.isfinite(q).all() or not torch.isfinite(target).all():
        raise ValueError('Non-finite probe Critic value')
    return dict(samples=len(action),q_mean=float(q.mean()),q_min=float(q.min()),q_max=float(q.max()),
                target_q_mean=float(target.mean()),current_target_mae=float((q-target).abs().mean()))
