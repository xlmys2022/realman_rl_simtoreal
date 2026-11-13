"""Shared deterministic Reach evaluation, episode summaries and logger tags."""
from collections import Counter
from time import perf_counter
import numpy as np

from ...envs.reach_env import ReachEnv
from ...tasks.reach import ENV_CONFIG
from .baselines import rule_action
from .metrics import MotionStats
from .observation import ReachTransform


REASONS = ('success', 'workspace_limit', 'joint_limit', 'ik_failure', 'orientation_limit',
           'collision', 'numerical_failure', 'time_limit', 'unreachable_target',
           'velocity_limit', 'drive_failure')


def summarize_episodes(episodes):
    counts = Counter(e['reason'] for e in episodes)
    successful = [e for e in episodes if e['success']]
    return dict(episodes=len(episodes), success_rate=len(successful)/len(episodes) if episodes else None,
                mean_return=float(np.mean([e['return_'] for e in episodes])) if episodes else None,
                mean_discounted_return=float(np.mean([e['discounted_return'] for e in episodes])) if episodes else None,
                mean_episode_length=float(np.mean([e['length'] for e in episodes])) if episodes else None,
                mean_success_steps=float(np.mean([e['length'] for e in successful])) if successful else None,
                mean_final_distance_m=float(np.mean([e['final_distance_m'] for e in episodes])) if episodes else None,
                safety_failures=sum(e['safety_failure'] for e in episodes),
                rejected_actions=sum(e.get('rejected_actions',0) for e in episodes),
                termination_counts={k: counts[k] for k in dict.fromkeys((*REASONS, *counts))})


def evaluate(name, seeds, *, model=None, transform=None, config=None, action_seed=2026, gamma=.99,
             reference_policy=None,collect_motion=False):
    """Raw SI environment; shared deterministic model.predict protocol for SAC/PPO."""
    if name not in ('random', 'rule', 'sac', 'ppo') or not seeds:
        raise ValueError('Expected policy name and nonempty seeds')
    if name in ('sac','ppo') and (model is None or transform is None):
        raise ValueError(name.upper()+' requires model AND its saved observation transform')
    episodes, step_times, reference_errors = [], [], []
    motion=MotionStats() if collect_motion else None
    start_total = perf_counter()
    with ReachEnv(**(ENV_CONFIG if config is None else config)) as env:
        expected_transform = ReachTransform.from_env(env).config
        if transform is not None and transform.config != expected_transform:
            raise ValueError('Saved transform does not match this model/home and units')
        discrete=env.target_mode=='discrete'
        if discrete and len(seeds)!=len(env.goal_candidates):
            raise ValueError(f'Discrete evaluation requires exactly {len(env.goal_candidates)} episodes, one per candidate; repeats are not generalization')
        for episode_index,seed in enumerate(seeds):
            started = perf_counter()
            obs, reset_info = env.reset(seed=int(seed),options={'goal_index':episode_index} if discrete else None)
            goal, initial = obs['desired_goal'].copy(), obs['achieved_goal'].copy()
            rng = np.random.default_rng([action_seed, int(seed)])
            rewards, actions = [], []
            rejected_actions, rejection_reasons = 0, Counter()
            physics_steps = 0
            episode_motion=MotionStats() if discrete and collect_motion else None
            for _ in range(env.max_episode_steps):
                if name in ('sac','ppo'):
                    action, _ = model.predict(transform.encode(obs), deterministic=True)
                    if reference_policy is not None:
                        reference_errors.append(action.astype(float)-reference_policy.predict(obs))
                elif name == 'rule':
                    action = rule_action(obs, env.action_scale)
                else:
                    action = rng.uniform(-1, 1, 3).astype(np.float32)
                if not env.action_space.contains(action) or not np.isfinite(action).all():
                    raise ValueError('Policy output outside finite action space')
                step_start = perf_counter()
                before_obs=obs
                obs, reward, terminated, truncated, info = env.step(action)
                if motion is not None:
                    motion.observe(before_obs['achieved_goal'],obs['achieved_goal'],before_obs['desired_goal'],
                        before_obs['observation'][18:21],info['command_position_m'],action,terminated or truncated)
                if episode_motion is not None:
                    episode_motion.observe(before_obs['achieved_goal'],obs['achieved_goal'],before_obs['desired_goal'],
                        before_obs['observation'][18:21],info['command_position_m'],action,terminated or truncated)
                step_times.append(perf_counter()-step_start)
                if not env.observation_space.contains(obs) or not all(np.isfinite(x).all() for x in obs.values()):
                    raise ValueError('Invalid physical observation')
                if reward != env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info):
                    raise ValueError('Reward changed during evaluation')
                physics_steps += info['physics_steps']
                rewards.append(float(reward))
                actions.append(action.tolist())
                rejected_actions += int(info.get('is_action_rejected', False))
                if info.get('is_action_rejected', False):
                    rejection_reasons[info['rejection_reason']] += 1
                if terminated or truncated:
                    break
            episodes.append(dict(seed=int(seed), target_position_m=goal.tolist(), initial_position_m=initial.tolist(),
                                 goal_sampling=reset_info['goal_sampling'], return_=float(sum(rewards)),
                                 discounted_return=float(np.dot(rewards, gamma**np.arange(len(rewards)))),
                                 length=len(rewards), success=bool(info['is_success']),
                                 safety_failure=bool(info['is_safety_failure']), reason=info['termination_reason'],
                                 terminated=bool(terminated), truncated=bool(truncated),
                                 final_position_m=obs['achieved_goal'].tolist(), final_distance_m=info['distance_m'],
                                 min_reward=min(rewards), mean_action_norm=float(np.linalg.norm(actions, axis=1).mean()),
                                 rejected_actions=rejected_actions, rejection_reasons=dict(rejection_reasons),
                                 near_zero_action_fraction=float(np.mean(np.linalg.norm(actions, axis=1) < .01)),
                                 physics_steps=physics_steps, episode_wall_s=perf_counter()-started, actions=actions))
            if discrete:
                episodes[-1]['goal_index']=episode_index
                episodes[-1]['termination_detail']=info.get('detail')
                offset=goal-initial;axis=int(np.argmax(abs(offset)))
                others=np.delete(offset,axis)
                episodes[-1]['goal_label']=('+' if offset[axis]>0 else '-')+'XYZ'[axis] if np.all(abs(others)<1e-10) else f'goal_{episode_index}'
                if episode_motion is not None:episodes[-1]['motion']=episode_motion.summary()
    elapsed = perf_counter()-start_total
    result = dict(policy=name, deterministic=name in ('sac','ppo'), summary=summarize_episodes(episodes), episodes=episodes,
                timing=dict(wall_s=elapsed, interaction_steps_per_s=len(step_times)/elapsed,
                            mean_step_s=float(np.mean(step_times)), median_step_s=float(np.median(step_times)),
                            p95_step_s=float(np.percentile(step_times, 95))))
    if reference_policy is not None:
        errors=np.asarray(reference_errors)
        result['bc_action_deviation']=(dict(samples=len(errors),mse=float(np.mean(errors**2)),
                                          mae=float(np.abs(errors).mean()),max_abs=float(np.abs(errors).max()))
                                       if errors.size else None)
    if motion is not None:result['motion']=motion.summary()
    if discrete:
        result['per_goal']={e['goal_label']:dict(goal_index=e['goal_index'],goal_position_m=e['target_position_m'],
            final_position_m=e['final_position_m'],
            success=e['success'],final_distance_m=e['final_distance_m'],episode_length=e['length'],
            first_action=e['actions'][0],termination_reason=e['reason'],termination_detail=e.get('termination_detail'),
            actual_path_m=e.get('motion',{}).get('actual_path_m'),
            action_direction_fraction=e.get('motion',{}).get('toward_actual_fraction')) for e in episodes}
    return result


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
    for label,row in evaluation.get('per_goal',{}).items():
        tag=label.replace('+','plus_').replace('-','minus_').lower()
        for key in ('success','episode_length','action_direction_fraction'):
            if row[key] is not None:logger.record(f'evaluation/goals/{tag}/{key}',float(row[key]))
        logger.record(f'evaluation/goals/{tag}/distance_mm',row['final_distance_m']*1000)
        if row['actual_path_m'] is not None:logger.record(f'evaluation/goals/{tag}/actual_path_mm',row['actual_path_m']*1000)
        for reason in REASONS:logger.record(f'evaluation/goals/{tag}/termination/'+reason,int(row['termination_reason']==reason))
        for axis,value in zip('xyz',row['first_action']):logger.record(f'evaluation/goals/{tag}/first_action_{axis}',value)
