"""Optional SB3 experiment helpers. No algorithm, replay buffer or physics reimplementation."""
from collections import Counter
import gzip
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

from ..envs.reach_env import ReachEnv
from ..paths import PROJECT_ROOT
from .common import rule_action

ENV_CONFIG = dict(target_mode='random', reward_mode='safe_dense', action_scale=.001,
                  control_period=.05, max_episode_steps=200, success_threshold=.001,
                  gravity_compensation=True)
SAC_CONFIG = dict(learning_rate=3e-4, batch_size=256, buffer_size=100_000,
                  learning_starts=100, gamma=.99, tau=.005, train_freq=1,
                  gradient_steps=1, ent_coef='auto', target_entropy='auto',
                  policy_kwargs=dict(net_arch=[256, 256]), device='cpu')
REASONS = ('success', 'workspace_limit', 'joint_limit', 'ik_failure', 'orientation_limit',
           'collision', 'numerical_failure', 'time_limit', 'unreachable_target',
           'velocity_limit', 'drive_failure')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append_json(path, value):
    with Path(path).open('a') as file:
        file.write(json.dumps(value, allow_nan=False)+'\n')


def versions():
    import sys
    return dict(python=sys.version, executable=sys.executable, torch_cuda=torch.version.cuda,
                packages={d.metadata['Name']: d.version for d in metadata.distributions()})


class ReachTransform:
    """Fixed affine SI -> dimensionless inputs; no online/reward normalization.

    Position centre = nominal home, scale = original 10mm command envelope.
    Angles use original 5deg local command envelope, velocity uses 30deg/s.
    Original Dict shapes remain 27+3+3. Goal keys stay SI float64 for the
    GoalEnv compute_reward contract; relative XYZ inside observation is scaled.
    SB3 itself casts network tensors to float32. Nothing is clipped or sent to IK.
    """
    def __init__(self, config):
        if config.get('schema') != 2:
            raise ValueError('Unsupported observation transform')
        self.config = config
        self.offset = {k: np.asarray(v, dtype=np.float64) for k, v in config['offset'].items()}
        self.scale = {k: np.asarray(v, dtype=np.float64) for k, v in config['scale'].items()}
        if config.get('output_dtype') != dict(observation='float32', achieved_goal='float64', desired_goal='float64'):
            raise ValueError('Policy wrapper must retain SI goal dtype/semantics')
        for key, shape in [('observation', (27,)), ('achieved_goal', (3,)), ('desired_goal', (3,))]:
            if (self.offset[key].shape != shape or self.scale[key].shape != shape
                    or not np.isfinite(self.offset[key]).all() or not np.isfinite(self.scale[key]).all()
                    or np.any(self.scale[key] <= 0)):
                raise ValueError('Invalid observation offset/scale')
        for key in ('achieved_goal', 'desired_goal'):
            if np.any(self.offset[key] != 0) or np.any(self.scale[key] != 1):
                raise ValueError('Goal keys must retain SI coordinates for compute_reward')

    @classmethod
    def from_env(cls, env):
        q, p = env.kinematics.home_q, env.kinematics.home_position
        offset = dict(observation=np.r_[q, np.zeros(6), p, np.zeros(3), p, q].tolist(),
                      achieved_goal=[0.]*3, desired_goal=[0.]*3)
        scale = dict(observation=np.r_[np.full(6, np.deg2rad(5)), np.full(6, np.deg2rad(30)),
                                         np.full(9, .01), np.full(6, np.deg2rad(5))].tolist(),
                     achieved_goal=[1.]*3, desired_goal=[1.]*3)
        return cls(dict(schema=2, method='fixed_affine_observation_keep_si_goals',
                        output_dtype=dict(observation='float32', achieved_goal='float64', desired_goal='float64'), clipping=False,
                        offset=offset, scale=scale, reward_normalization=False))

    def encode(self, observation):
        result = {k: ((np.asarray(observation[k])-self.offset[k])/self.scale[k]).astype(self.config['output_dtype'][k])
                  for k in self.offset}
        if not all(np.isfinite(x).all() for x in result.values()):
            raise ValueError('Non-finite policy input')
        return result


class ScaledReachObservation(gym.ObservationWrapper):
    def __init__(self, env, transform=None):
        super().__init__(env)
        self.transform = ReachTransform.from_env(env.unwrapped) if transform is None else transform
        self.observation_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, shape=v.shape,
                                                            dtype=self.transform.config['output_dtype'][k])
                                              for k, v in env.observation_space.spaces.items()})

    def observation(self, observation):
        return self.transform.encode(observation)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # The GoalEnv keys deliberately retain metres; never feed normalized
        # physical positions into the original environment's reward/FK/IK.
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)


def training_env(config, monitor_file=None):
    raw = ReachEnv(**config)
    wrapped = ScaledReachObservation(raw)
    transform = wrapped.transform
    monitored = Monitor(wrapped, filename=None if monitor_file is None else str(monitor_file),
                        info_keywords=('is_success', 'distance_m', 'termination_reason', 'is_safety_failure'))
    return VecCheckNan(DummyVecEnv([lambda: monitored]), raise_exception=True), transform


def make_model(env, seed, *, overrides=None):
    params = dict(SAC_CONFIG)
    if overrides:
        params.update(overrides)
    return SAC('MultiInputPolicy', env, seed=seed, verbose=0, **params)


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


def evaluate(name, seeds, *, model=None, transform=None, config=None, action_seed=2026, gamma=.99):
    """Raw SI environment for every policy; transform only the SAC network input."""
    if name not in ('random', 'rule', 'sac') or not seeds:
        raise ValueError('Expected policy name and nonempty seeds')
    if name == 'sac' and (model is None or transform is None):
        raise ValueError('SAC requires model AND its saved observation transform')
    episodes, step_times = [], []
    start_total = perf_counter()
    with ReachEnv(**(ENV_CONFIG if config is None else config)) as env:
        expected_transform = ReachTransform.from_env(env).config
        if transform is not None and transform.config != expected_transform:
            raise ValueError('Saved transform does not match this model/home and units')
        for seed in seeds:
            started = perf_counter()
            obs, reset_info = env.reset(seed=int(seed))
            goal, initial = obs['desired_goal'].copy(), obs['achieved_goal'].copy()
            rng = np.random.default_rng([action_seed, int(seed)])
            rewards, actions = [], []
            rejected_actions, rejection_reasons = 0, Counter()
            physics_steps = 0
            for _ in range(env.max_episode_steps):
                if name == 'sac':
                    action, _ = model.predict(transform.encode(obs), deterministic=True)
                elif name == 'rule':
                    action = rule_action(obs, env.action_scale)
                else:
                    action = rng.uniform(-1, 1, 3).astype(np.float32)
                if not env.action_space.contains(action) or not np.isfinite(action).all():
                    raise ValueError('Policy output outside finite action space')
                step_start = perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
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
    elapsed = perf_counter()-start_total
    return dict(policy=name, deterministic=name == 'sac', summary=summarize_episodes(episodes), episodes=episodes,
                timing=dict(wall_s=elapsed, interaction_steps_per_s=len(step_times)/elapsed,
                            mean_step_s=float(np.mean(step_times)), median_step_s=float(np.median(step_times)),
                            p95_step_s=float(np.percentile(step_times, 95))))


def q_diagnostic(model):
    """Inspect ordinary SB3 networks/buffer; preserve the training sampling RNG."""
    if model.replay_buffer.size() == 0:
        return dict(buffer_size=0)
    state = np.random.get_state()
    try:
        batch = model.replay_buffer.sample(min(128, model.replay_buffer.size()))
    finally:
        np.random.set_state(state)
    with torch.no_grad():
        actions = model.actor(batch.observations, deterministic=True)
        values = torch.cat(model.critic(batch.observations, actions), dim=1).cpu().numpy()
        _, log_std, _ = model.actor.get_action_dist_params(batch.observations)
        std = torch.exp(log_std).cpu().numpy()
        finite = all(torch.isfinite(p).all().item() for p in model.policy.parameters())
        if model.log_ent_coef is not None:
            finite = finite and torch.isfinite(model.log_ent_coef).all().item()
    result = dict(buffer_size=int(model.replay_buffer.size()), q_min=float(values.min()), q_max=float(values.max()),
                  q_abs_max=float(np.abs(values).max()), q_mean=float(values.mean()), parameters_finite=bool(finite),
                  action_abs_mean=float(actions.abs().mean()), policy_std_mean=float(std.mean()))
    if not finite or not np.isfinite(values).all() or not np.isfinite(std).all():
        raise ValueError('Non-finite neural parameters/Q/std')
    if result['q_abs_max'] > 1e6:
        raise ValueError('Q magnitude exceeded the predeclared 1e6 divergence guard')
    return result


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


def save_checkpoint(model, folder, *, replay=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder/'model.zip')
    if replay:
        # SB3 accepts BytesIO, but does not accept gzip.GzipFile directly.
        serialized = io.BytesIO()
        model.save_replay_buffer(serialized)
        with gzip.open(folder/'replay_buffer.pkl.gz', 'wb') as file:
            file.write(serialized.getbuffer())
    write_json(folder/'state.json', dict(num_timesteps=int(model.num_timesteps), n_updates=int(model._n_updates),
                                      replay_size=int(model.replay_buffer.size())))


def asset_hashes():
    files = [p for p in (PROJECT_ROOT/'assets/robots/rml63').rglob('*') if p.is_file()]
    files += [PROJECT_ROOT/'src/realman_rl_simtoreal'/name for name in
              ('envs/reach_env.py', 'simulation/kinematics.py', 'simulation/control.py', 'simulation/model.py')]
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
