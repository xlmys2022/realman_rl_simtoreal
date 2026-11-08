"""Experiment-only scalar monitoring; no changes to SAC optimization/control."""
import numpy as np
import torch

from .sac_reach import TrainingAudit,append_json
from .reach_tasks import MotionStats


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
            from .sac_initialization import action_deviation
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


def record_evaluation(logger, evaluation):
    summary=evaluation['summary']
    for tag,key in [('success_rate','success_rate'),('episode_length_mean','mean_episode_length'),
                    ('episode_reward_mean','mean_return'),('success_steps_mean','mean_success_steps'),
                    ('safety_failures','safety_failures')]:
        if summary[key] is not None:
            logger.record('evaluation/'+tag,summary[key])
    logger.record('evaluation/final_distance_mm',1000*summary['mean_final_distance_m'])
    logger.record('evaluation/timeouts',summary['termination_counts'].get('time_limit',0))
    logger.record('evaluation/action_rejection_fraction',summary['rejected_actions']/sum(e['length'] for e in evaluation['episodes']))
    if evaluation.get('bc_action_deviation') is not None:
        for key in ('mse','mae','max_abs'):
            logger.record('bc/evaluation_states_'+key,evaluation['bc_action_deviation'][key])
    for key,value in evaluation.get('motion',{}).items():
        if value is not None:logger.record('evaluation/motion/'+key,value)
