"""Independent Reach episodes; policies are supplied by callers, never trained here."""
from collections import Counter
from time import perf_counter
import numpy as np

from ..envs.factory import make_reach_env
from ..tasks.reach import ENV_CONFIG
from ..tasks.rewards import REWARD_COMPONENT_KEYS
from ..rl.common.observation import ReachTransform, require_legacy
from ..utils.metrics import MotionStats
from .baselines import rule_action
from .metrics import summarize_episodes, summarize_bc_episodes
from ..envs.reach_env import ReachEnv
from ..utils.metrics import cosine


def evaluate(name, seeds, *, model=None, transform=None, config=None, action_seed=2026, gamma=.99,
             reference_policy=None,collect_motion=False,predict_action=None):
    """Raw SI environment; shared deterministic model.predict protocol for SAC/PPO."""
    if predict_action is not None and not callable(predict_action):
        raise TypeError('predict_action must map one raw observation to one action')
    if (name not in ('random', 'rule', 'sac', 'ppo') and predict_action is None) or not seeds:
        raise ValueError('Expected policy name and nonempty seeds')
    if predict_action is None and name in ('sac','ppo') and (model is None or transform is None):
        raise ValueError(name.upper()+' requires model AND its saved observation transform')
    if name == 'rule':
        require_legacy(ENV_CONFIG if config is None else config, 'Command-lag rule baseline')
    episodes, step_times, reference_errors = [], [], []
    motion=MotionStats() if collect_motion else None
    start_total = perf_counter()
    with make_reach_env(ENV_CONFIG if config is None else config) as env:
        expected_transform = ReachTransform.from_env(env).config
        if transform is not None and transform.config != expected_transform:
            raise ValueError('Saved transform does not match this model/home and units')
        discrete=env.target_mode=='discrete'
        if discrete and len(seeds)!=len(env.goal_candidates):
            raise ValueError(f'Discrete evaluation requires exactly {len(env.goal_candidates)} episodes, one per candidate; repeats are not generalization')
        for episode_index,seed in enumerate(seeds):
            started = perf_counter()
            obs, reset_info = env.reset(seed=int(seed),options={'goal_index':episode_index} if discrete else None)
            previous_info = reset_info
            goal, initial = obs['desired_goal'].copy(), obs['achieved_goal'].copy()
            rng = np.random.default_rng([action_seed, int(seed)])
            rewards, actions = [], []
            reward_components=[]
            goal_results=[]
            rejected_actions, rejection_reasons = 0, Counter()
            physics_steps = 0
            episode_motion=MotionStats() if discrete and collect_motion else None
            for _ in range(getattr(env,'episode_step_limit',env.max_episode_steps)):
                if predict_action is not None:
                    action = predict_action(obs)
                    if reference_policy is not None:
                        reference_errors.append(action.astype(float)-reference_policy.predict(obs))
                elif name in ('sac','ppo'):
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
                command_before=np.asarray(previous_info['command_position_m']).copy()
                obs, reward, terminated, truncated, info = env.step(action)
                if motion is not None:
                    motion.observe(before_obs['achieved_goal'],obs['achieved_goal'],before_obs['desired_goal'],
                        command_before,info['command_position_m'],action,terminated or truncated or info.get('goal_changed',False))
                if episode_motion is not None:
                    episode_motion.observe(before_obs['achieved_goal'],obs['achieved_goal'],before_obs['desired_goal'],
                        command_before,info['command_position_m'],action,terminated or truncated)
                previous_info=info
                step_times.append(perf_counter()-step_start)
                if not env.observation_space.contains(obs) or not all(np.isfinite(x).all() for x in obs.values()):
                    raise ValueError('Invalid physical observation')
                if reward != env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info):
                    raise ValueError('Reward changed during evaluation')
                physics_steps += info['physics_steps']
                rewards.append(float(reward))
                if 'reward_components' in info:
                    components=info['reward_components']
                    if sum(components[k] for k in REWARD_COMPONENT_KEYS)!=reward or components['total_reward']!=reward:
                        raise ValueError('Evaluation reward component sum mismatch')
                    reward_components.append(components.copy())
                actions.append(action.tolist())
                if info.get('goal_result') is not None:
                    goal_results.append(dict(info['goal_result']))
                rejected_actions += int(info.get('is_action_rejected', False))
                if info.get('is_action_rejected', False):
                    rejection_reasons[info['rejection_reason']] += 1
                if terminated or truncated:
                    break
            if not (terminated or truncated):
                raise RuntimeError('Reach episode exceeded its declared step limit without ending')
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
            if 'goals_planned' in reset_info:
                for goal_result in goal_results:
                    begin,end=goal_result['start_step'],goal_result['end_step']
                    leg_rewards=rewards[begin:end]
                    goal_result.update(actions=actions[begin:end],return_=float(sum(leg_rewards)),
                        discounted_return=float(np.dot(leg_rewards,gamma**np.arange(len(leg_rewards)))))
                episodes[-1].update(initial_target_position_m=goal.tolist(),
                    target_position_m=obs['desired_goal'].tolist(),goal_results=goal_results,
                    goals_completed=info['goals_completed'],goals_planned=info['goals_planned'],
                    goals_attempted=len(goal_results),goal_plan_m=reset_info['goal_plan_m'],
                    start_sampling=reset_info['start_sampling'],workspace_center_m=reset_info['workspace_center_m'])
            if reward_components:
                episodes[-1]['reward_components']=reward_components
                episodes[-1]['reward_component_totals']={k:float(sum(r[k] for r in reward_components))
                    for k in (*REWARD_COMPONENT_KEYS,'total_reward')}
            if discrete:
                episodes[-1]['goal_index']=episode_index
                episodes[-1]['termination_detail']=info.get('detail')
                offset=goal-initial;axis=int(np.argmax(abs(offset)))
                others=np.delete(offset,axis)
                episodes[-1]['goal_label']=('+' if offset[axis]>0 else '-')+'XYZ'[axis] if np.all(abs(others)<1e-10) else f'goal_{episode_index}'
                if episode_motion is not None:episodes[-1]['motion']=episode_motion.summary()
    elapsed = perf_counter()-start_total
    result = dict(policy=name, deterministic=predict_action is not None or name in ('sac','ppo'), summary=summarize_episodes(episodes), episodes=episodes,
                timing=dict(wall_s=elapsed, interaction_steps_per_s=len(step_times)/elapsed,
                            mean_step_s=float(np.mean(step_times)), median_step_s=float(np.median(step_times)),
                            p95_step_s=float(np.percentile(step_times, 95))))
    if reference_policy is not None:
        errors=np.asarray(reference_errors)
        result['bc_action_deviation']=(dict(samples=len(errors),mse=float(np.mean(errors**2)),
                                          mae=float(np.abs(errors).mean()),max_abs=float(np.abs(errors).max()))
                                       if errors.size else None)
    if motion is not None:result['motion']=motion.summary()
    if any('reward_component_totals' in e for e in episodes):
        result['reward_component_totals']={k:float(sum(e['reward_component_totals'][k] for e in episodes))
            for k in (*REWARD_COMPONENT_KEYS,'total_reward')}
    if discrete:
        result['per_goal']={e['goal_label']:dict(goal_index=e['goal_index'],goal_position_m=e['target_position_m'],
            final_position_m=e['final_position_m'],
            success=e['success'],final_distance_m=e['final_distance_m'],episode_length=e['length'],
            first_action=e['actions'][0],termination_reason=e['reason'],termination_detail=e.get('termination_detail'),
            actual_path_m=e.get('motion',{}).get('actual_path_m'),
            action_direction_fraction=e.get('motion',{}).get('toward_actual_fraction')) for e in episodes}
    return result


def evaluate_policy(predict_action, seeds, *, name='policy', config=None, transform=None,
                    gamma=.99, collect_motion=False, state_digest=None):
    """Evaluate a raw-observation -> action callable, independently of its training.

    Callers select deterministic inference and any needed preprocessing. For
    example BCPolicy.predict is already a raw-observation callable; an SB3
    policy uses a lambda encoding obs and selecting predict(..., deterministic=True)[0].
    An optional caller-owned state_digest verifies parameter/counter invariance
    without importing its algorithm. Historical JSON schemas stay unchanged.
    """
    before=state_digest() if state_digest is not None else None
    result=evaluate(name, seeds, config=config, transform=transform, gamma=gamma,
                    collect_motion=collect_motion, predict_action=predict_action)
    if state_digest is not None and state_digest()!=before:
        raise ValueError('Evaluation changed policy parameters or counters')
    return result


def evaluate_bc(name, seeds, *, policy=None, environment=None, state_digest=None):
    """BC execution never calls expert; only the rule comparison uses rule_action."""
    if name not in ('rule','untrained','bc') or not len(seeds):
        raise ValueError('Expected rule/untrained/bc and nonempty seeds')
    if name != 'rule' and policy is None:
        raise ValueError('Learned/untrained policy required')
    config=dict(ENV_CONFIG if environment is None else environment)
    require_legacy(config, 'BC evaluation')
    before=state_digest() if state_digest is not None else None
    episodes,records,step_times=[],[],[]
    started=perf_counter()
    with ReachEnv(**config) as env:
        if policy is not None and policy.transform.config != ReachTransform.from_env(env).config:
            raise ValueError('Saved transform does not match environment')
        for seed in seeds:
            obs,_=env.reset(seed=int(seed))
            goal=obs['desired_goal'].copy();initial=obs['achieved_goal'].copy()
            rewards=[];previous=None;reversed_steps=0;physical_steps=0
            begin=len(records)
            for step in range(env.max_episode_steps):
                actual=obs['achieved_goal'].copy();command=obs['observation'][18:21].copy()
                # No expert correction, blending or fallback in either neural branch.
                action=rule_action(obs,env.action_scale) if name == 'rule' else policy.predict(obs)
                if not env.action_space.contains(action):
                    raise ValueError('Illegal policy action')
                raw={k:v.copy() for k,v in obs.items()}
                tick=perf_counter()
                obs,reward,terminated,truncated,info=env.step(action)
                step_times.append(perf_counter()-tick)
                if not env.observation_space.contains(obs) or not all(np.isfinite(v).all() for v in obs.values()):
                    raise ValueError('Invalid evaluation observation')
                if reward != env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info):
                    raise ValueError('Inconsistent evaluation reward')
                reversal=previous is not None and float(previous@action) < -1e-9
                reversed_steps+=int(reversal);previous=action.copy()
                records.append(dict(seed=int(seed),step=step+1,raw_observation=raw['observation'].tolist(),
                    actual_before_m=actual.tolist(),command_before_m=command.tolist(),goal_m=goal.tolist(),
                    actual_after_m=obs['achieved_goal'].tolist(),command_after_m=info['command_position_m'].tolist(),
                    action=action.tolist(),reversed_from_previous=bool(reversal),
                    reward=reward,distance_m=info['distance_m'],reason=info['termination_reason'],
                    physics_steps=info['physics_steps'],simulation_time_s=info['simulation_time_s']))
                rewards.append(reward);physical_steps+=info['physics_steps']
                if terminated or truncated:
                    break
            rows=records[begin:]
            actual_error=goal-np.array([r['actual_before_m'] for r in rows])
            command_error=goal-np.array([r['command_before_m'] for r in rows])
            actions=np.array([r['action'] for r in rows])
            actual_cos,command_cos=cosine(actions,actual_error),cosine(actions,command_error)
            command_path=sum(np.linalg.norm(np.subtract(r['command_after_m'],r['command_before_m'])) for r in rows)
            actual_path=sum(np.linalg.norm(np.subtract(r['actual_after_m'],r['actual_before_m'])) for r in rows)
            episodes.append(dict(seed=int(seed),goal_m=goal.tolist(),initial_position_m=initial.tolist(),
                final_position_m=obs['achieved_goal'].tolist(),success=info['is_success'],
                safety_failure=info['is_safety_failure'],reason=info['termination_reason'],
                terminated=bool(terminated),truncated=bool(truncated),length=len(rows),
                return_=float(sum(rewards)),discounted_return=float(np.dot(rewards,.99**np.arange(len(rewards)))),
                final_distance_m=info['distance_m'],physical_steps=physical_steps,
                actual_toward=int(np.sum(actual_cos > 0)),actual_directions=len(actual_cos),
                command_toward=int(np.sum(command_cos > 0)),command_directions=len(command_cos),
                reversed_steps=reversed_steps,reversal_pairs=max(0,len(rows)-1),
                command_path_m=command_path,actual_path_m=actual_path))
    if state_digest is not None and state_digest() != before:
        raise ValueError('Evaluation changed network parameters')
    summary=summarize_bc_episodes(episodes)
    wall=perf_counter()-started
    return dict(policy=name,environment=config,summary=summary,episodes=episodes,records=records,
                parameters_unchanged=True if policy is None or state_digest is not None else None,timing=dict(wall_s=wall,mean_step_s=float(np.mean(step_times)),
                                                     interaction_steps_per_s=len(step_times)/wall))
