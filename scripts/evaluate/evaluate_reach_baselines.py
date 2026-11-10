#!/usr/bin/env python3
"""Paired, headless random/rule policy evaluation without training or joint access."""

import argparse
from collections import Counter
import cProfile
import io
import json
from pathlib import Path
import platform
import pstats
import sys
from time import perf_counter

import gymnasium
import mujoco
import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import OUTPUTS


from realman_rl_simtoreal.rl.common.baselines import rule_action


def validate_transition(env, observation, reward, info):
    if (not env.observation_space.contains(observation)
            or not all(np.isfinite(v).all() for v in observation.values())):
        raise ValueError("Invalid observation during evaluation")
    if not np.array_equal(observation['observation'][15:18],
                          observation['desired_goal']-observation['achieved_goal']):
        raise ValueError("Goal-relative observation is inconsistent")
    if reward != env.compute_reward(observation['achieved_goal'], observation['desired_goal'], info):
        raise ValueError("step and compute_reward disagree")


def latency_summary(samples):
    values = np.asarray(samples, dtype=float)
    if not len(values):
        return dict(count=0, mean_s=None, median_s=None, p95_s=None, total_s=0.0)
    return dict(count=len(values), mean_s=float(np.mean(values)), median_s=float(np.median(values)),
                p95_s=float(np.percentile(values, 95)), total_s=float(np.sum(values)))


def goal_distribution(episodes):
    deltas = np.array([e['target_delta_m'] for e in episodes])
    octants = Counter(''.join('+' if x >= 0 else '-' for x in row) for row in deltas)
    return dict(min_xyz_m=deltas.min(axis=0).tolist(), max_xyz_m=deltas.max(axis=0).tolist(),
                mean_xyz_m=deltas.mean(axis=0).tolist(), octant_counts=dict(octants),
                min_radius_m=float(np.min(np.linalg.norm(deltas, axis=1))),
                max_radius_m=float(np.max(np.linalg.norm(deltas, axis=1))),
                total_sampling_attempts=sum(e['goal_sampling']['attempts'] for e in episodes),
                sampling_rejections=dict(sum((Counter(e['goal_sampling']['rejections']) for e in episodes), Counter())))


def evaluate_policy(name, seeds, *, action_seed=2026, max_episode_steps=200,
                    target_radius=.005, reward_mode='safe_dense'):
    if name not in ('random', 'rule') or not seeds:
        raise ValueError("Expected random/rule policy and nonempty seeds")
    episodes, all_step_times, full_cycle_times = [], [], []
    with ReachEnv(target_mode='random', reward_mode=reward_mode, target_radius=target_radius,
                  max_episode_steps=max_episode_steps) as env:
        # Warm-up is excluded. Episode seeds below reset the target RNG exactly.
        env.reset(seed=2**31-1)
        env.step(np.zeros(3, dtype=np.float32))
        for seed in seeds:
            begin = perf_counter()
            observation, reset_info = env.reset(seed=int(seed))
            reset_time = perf_counter()-begin
            initial = observation['achieved_goal'].copy()
            goal = observation['desired_goal'].copy()
            rng = np.random.default_rng([action_seed, int(seed)])
            rewards, actions, step_times = [], [], []
            physics_steps = 0
            peak_torque, peak_speed = np.zeros(6), np.zeros(6)
            max_contacts = 0
            for _ in range(max_episode_steps):
                action = (rng.uniform(-1, 1, 3).astype(np.float32) if name == 'random'
                          else rule_action(observation, env.action_scale))
                if not env.action_space.contains(action):
                    raise ValueError("Policy action outside action_space")
                start = perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                elapsed = perf_counter()-start
                step_times.append(elapsed)
                all_step_times.append(elapsed)
                if info['physics_steps'] == env.physics_steps:
                    full_cycle_times.append(elapsed)
                validate_transition(env, observation, reward, info)
                rewards.append(reward)
                actions.append(action.tolist())
                physics_steps += info['physics_steps']
                if info['control'] is not None:
                    audit = info['control']['forces']
                    peak_torque = np.maximum(peak_torque, audit['peak_total_drive_nm'])
                    peak_speed = np.maximum(peak_speed, audit['peak_speed_deg_s'])
                    max_contacts = max(max_contacts, info['control']['max_contacts'])
                if terminated or truncated:
                    break
            episode_time = perf_counter()-begin
            episodes.append(dict(seed=int(seed), target_position_m=goal.tolist(), target_delta_m=(goal-initial).tolist(),
                                 goal_sampling=reset_info['goal_sampling'], success=info['is_success'],
                                 return_=float(sum(rewards)), length=len(rewards), reason=info['termination_reason'],
                                 terminated=terminated, truncated=truncated, final_distance_m=info['distance_m'],
                                 actions=actions, step_times_s=step_times, episode_time_s=episode_time,
                                 reset_time_s=reset_time, physics_steps=physics_steps, max_contacts=max_contacts,
                                 peak_total_drive_nm=peak_torque.tolist(), peak_speed_deg_s=peak_speed.tolist()))
    successful = [e for e in episodes if e['success']]
    counts = Counter(e['reason'] for e in episodes)
    summary = dict(episodes=len(episodes), success_rate=len(successful)/len(episodes),
                   mean_return=float(np.mean([e['return_'] for e in episodes])),
                   mean_episode_length=float(np.mean([e['length'] for e in episodes])),
                   mean_success_steps=float(np.mean([e['length'] for e in successful])) if successful else None,
                   mean_final_distance_m=float(np.mean([e['final_distance_m'] for e in episodes])),
                   termination_counts=dict(counts), ik_failures=counts['ik_failure'],
                   command_boundary_failures=counts['workspace_limit'], joint_limit_failures=counts['joint_limit'],
                   collisions=counts['collision'], numerical_failures=counts['numerical_failure'],
                   timeouts=counts['time_limit'])
    episode_total = sum(e['episode_time_s'] for e in episodes)
    timing = dict(step=latency_summary(all_step_times), full_cycle_step=latency_summary(full_cycle_times),
                  reset=latency_summary([e['reset_time_s'] for e in episodes]),
                  episode=latency_summary([e['episode_time_s'] for e in episodes]),
                  interaction_steps_per_s=len(all_step_times)/episode_total,
                  step_only_calls_per_s=len(all_step_times)/sum(all_step_times),
                  scope='Episode time includes reset, policy and transition audits; excludes creation, warm-up and JSON I/O')
    return dict(policy=name, summary=summary, timing=timing, goal_distribution=goal_distribution(episodes), episodes=episodes)


def reward_comparison(max_episode_steps=200):
    """Concrete shortcut counterexample with identical dynamics/action scales."""
    rows = []
    for mode in ('distance', 'safe_dense'):
        for case in ('reach', 'deliberate_boundary_failure'):
            with ReachEnv(action_scale=.01, reward_mode=mode, max_episode_steps=max_episode_steps) as env:
                env.reset(seed=0)
                rewards = []
                for index in range(max_episode_steps):
                    action = ([1, 1, 1] if case == 'deliberate_boundary_failure'
                              else [.1, 0, 0] if index < 5 else [0, 0, 0])
                    _, reward, t, tr, info = env.step(action)
                    rewards.append(reward)
                    if t or tr:
                        break
                rows.append(dict(reward_mode=mode, case=case, length=len(rewards), return_=sum(rewards),
                                 reason=info['termination_reason'], distance_m=info['distance_m']))
    return rows


def profile_components(seeds, *, max_episode_steps=200, target_radius=.005):
    """Separate diagnostic run. cProfile overhead is excluded from main timings."""
    profiler, steps, wall = cProfile.Profile(), 0, 0.0
    with ReachEnv(target_mode='random', max_episode_steps=max_episode_steps, target_radius=target_radius) as env:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            for _ in range(max_episode_steps):
                action = rule_action(obs, env.action_scale)
                start = perf_counter()
                profiler.enable()
                try:
                    obs, _, t, tr, _ = env.step(action)
                finally:
                    profiler.disable()
                wall += perf_counter()-start
                steps += 1
                if t or tr:
                    break
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stages = {}
    for label, function in [('ik', 'solve_ik'), ('collision_precheck', 'check_motion'),
                            ('physics_and_audits', 'advance_control_cycle'), ('native_mj_step', 'mj_step')]:
        entries = [v for k, v in stats.stats.items() if k[2] == function
                   or (label == 'native_mj_step' and 'mj_step' in k[2])]
        cumulative = sum(v[3] for v in entries)
        stages[label] = dict(calls=sum(v[1] for v in entries), cumulative_s=cumulative,
                             mean_per_env_step_s=cumulative/steps, share=cumulative/wall)
    other = wall-sum(stages[label]['cumulative_s'] for label in ('ik', 'collision_precheck', 'physics_and_audits'))
    stages['other'] = dict(cumulative_s=other, mean_per_env_step_s=other/steps, share=other/wall)
    stats.sort_stats('cumulative').print_stats(30)
    return dict(steps=steps, profiled_step_wall_s=wall, stages=stages,
                scope='Separate rule episodes; only step calls profiled, reset excluded; native_mj_step is a subset of physics_and_audits'), stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--action-seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--target-radius-mm', type=float, default=5)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--json', type=Path, default=OUTPUTS/'reach_baselines.json')
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.seed_start < 0 or args.action_seed < 0:
        parser.error('Episodes/max-steps must be positive and seeds nonnegative')
    seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    try:
        results = {}
        for name in ('random', 'rule'):
            results[name] = evaluate_policy(name, seeds, action_seed=args.action_seed,
                                            max_episode_steps=args.max_steps, target_radius=args.target_radius_mm/1000)
            print(name, json.dumps(results[name]['summary']), flush=True)
        paired = all(a['target_position_m'] == b['target_position_m']
                     for a, b in zip(results['random']['episodes'], results['rule']['episodes']))
        if not paired:
            raise ValueError('Policies did not receive identical goal seeds/positions')
        report = dict(configuration=dict(seeds=seeds, action_seed=args.action_seed, max_episode_steps=args.max_steps,
                                         target_radius_m=args.target_radius_mm/1000, target_mode='random', reward_mode='safe_dense'),
                      versions=dict(python=platform.python_version(), python_executable=sys.executable,
                                    gymnasium=gymnasium.__version__, mujoco=mujoco.__version__, numpy=np.__version__),
                      policies=dict(random='Independent uniform Box(-1,1) actions',
                                    rule='Observation feedback with pending-command correction; normalized norm capped at .25'),
                      paired_goals_identical=paired, results=results, reward_comparison=reward_comparison(args.max_steps))
        args.json.parent.mkdir(parents=True, exist_ok=True)
        if args.profile:
            report['profile'], text = profile_components(seeds[:10], max_episode_steps=args.max_steps,
                                                        target_radius=args.target_radius_mm/1000)
            args.json.with_suffix('.profile.txt').write_text(text)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(f'[PASS] Paired evaluation and audits completed: {args.json}', flush=True)
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f'[FAIL] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
