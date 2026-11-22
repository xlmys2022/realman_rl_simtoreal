"""Public-observation rule baseline; never used to guide pure RL training."""
import cProfile
import io
import pstats
from time import perf_counter
import numpy as np

from ..envs.reach.reach_env import ReachEnv
from ..rl.common.observation import require_legacy
from .metrics import latency_summary, goal_distribution, summarize_baseline_episodes


def rule_action(observation, action_scale):
    """Position feedback minus already-commanded tracking lag (anti-windup).

    delta = (goal-actual) - (command-actual) = goal-command. Once the command
    reaches the goal we send zero increments while the actuators catch up.
    The normalized vector norm is capped at .25 (0.25mm by default). This
    leaves margin for coupled wrist motion under the existing joint rate cap.
    Only public observations are read; all motion goes through env.step.
    """
    if np.asarray(observation['observation']).shape != (27,):
        raise ValueError("Legacy command-lag rule baseline requires observation_mode='legacy'; "
                         "fetch_like_v1 has no command position")
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("action_scale must be positive")
    actual = observation['achieved_goal']
    error = observation['desired_goal']-actual
    pending = observation['observation'][18:21]-actual
    action = (error-pending)/action_scale
    action *= min(1.0, .25/max(float(np.linalg.norm(action)), 1e-30))
    return np.clip(action, -1, 1).astype(np.float32)


def validate_transition(env, observation, reward, info):
    require_legacy(dict(observation_mode=env.observation_mode), 'Historical baseline validation')
    if (not env.observation_space.contains(observation)
            or not all(np.isfinite(v).all() for v in observation.values())):
        raise ValueError("Invalid observation during evaluation")
    if not np.array_equal(observation['observation'][15:18],
                          observation['desired_goal']-observation['achieved_goal']):
        raise ValueError("Goal-relative observation is inconsistent")
    if reward != env.compute_reward(observation['achieved_goal'], observation['desired_goal'], info):
        raise ValueError("step and compute_reward disagree")


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
    summary = summarize_baseline_episodes(episodes)
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
