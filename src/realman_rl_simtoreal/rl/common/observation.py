"""Versioned fixed transforms; raw physics and SI GoalEnv keys stay separate."""
import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np


FEATURE_KEYS = ('achieved_goal', 'desired_goal', 'observation')
OBSERVATION_DIMS = dict(legacy=27, fetch_like_v1=6, goal_relative_v1=6)


def observation_fields(mode):
    """Scalar order inside each Dict key; SB3 concatenates FEATURE_KEYS."""
    if mode == 'legacy':
        ordinary = ([f'joint_{i}_position_rad' for i in range(1, 7)]
                    + [f'joint_{i}_velocity_rad_s' for i in range(1, 7)]
                    + [f'ee_{a}_world_m' for a in 'xyz']
                    + [f'goal_minus_ee_{a}_m' for a in 'xyz']
                    + [f'command_{a}_world_m' for a in 'xyz']
                    + [f'joint_{i}_control_reference_rad' for i in range(1, 7)])
    elif mode == 'fetch_like_v1':
        ordinary = ([f'ee_{a}_world_m' for a in 'xyz']
                    + [f'ee_velocity_{a}_world_times_control_period_m' for a in 'xyz'])
    elif mode == 'goal_relative_v1':
        ordinary = ([f'goal_minus_command_{a}_m' for a in 'xyz']
                    + [f'command_minus_ee_{a}_m' for a in 'xyz'])
    else:
        raise ValueError('Unknown observation_mode: '+str(mode))
    return dict(observation=ordinary, achieved_goal=[f'ee_{a}_world_m' for a in 'xyz'],
                desired_goal=[f'target_{a}_world_m' for a in 'xyz'])


class ReachTransform:
    """Fixed affine SI -> dimensionless inputs; no online/reward normalization.

    Position centre = nominal home, scale = original 10mm command envelope.
    Angles use original 5deg local command envelope, velocity uses 30deg/s.
    Legacy schema=2 is preserved byte-for-byte in generated configuration.
    Fetch-like schema=3 centres ee position on home and scales both XYZ and
    velocity*control_period by .01m. Goal-relative schema=4 expresses both
    residuals in millimetres. Goal keys stay SI float64 for compute_reward.
    SB3 itself casts network tensors to float32. Nothing is clipped or sent to IK.
    """
    def __init__(self, config):
        self.observation_mode = config.get('observation_mode', 'legacy')
        if ((config.get('schema'), self.observation_mode)
                not in ((2, 'legacy'), (3, 'fetch_like_v1'), (4, 'goal_relative_v1'))):
            raise ValueError('Unsupported observation transform')
        self.config = copy.deepcopy(config)
        self.offset = {k: np.asarray(v, dtype=np.float64) for k, v in config['offset'].items()}
        self.scale = {k: np.asarray(v, dtype=np.float64) for k, v in config['scale'].items()}
        if config.get('output_dtype') != dict(observation='float32', achieved_goal='float64', desired_goal='float64'):
            raise ValueError('Policy wrapper must retain SI goal dtype/semantics')
        self.features_dim = OBSERVATION_DIMS[self.observation_mode] + 6
        for key, shape in [('observation', (self.features_dim-6,)), ('achieved_goal', (3,)), ('desired_goal', (3,))]:
            if (self.offset[key].shape != shape or self.scale[key].shape != shape
                    or not np.isfinite(self.offset[key]).all() or not np.isfinite(self.scale[key]).all()
                    or np.any(self.scale[key] <= 0)):
                raise ValueError('Invalid observation offset/scale')
        for key in ('achieved_goal', 'desired_goal'):
            if np.any(self.offset[key] != 0) or np.any(self.scale[key] != 1):
                raise ValueError('Goal keys must retain SI coordinates for compute_reward')
        if self.observation_mode == 'fetch_like_v1':
            if (config.get('method') != 'fixed_affine_observation_keep_si_goals'
                    or config.get('clipping') is not False or config.get('reward_normalization') is not False
                    or config.get('motion_feature') != 'world_site_linear_velocity_times_control_period_m'
                    or not np.isfinite(config.get('control_period_s', np.nan))
                    or config['control_period_s'] <= 0
                    or np.any(self.offset['observation'][3:] != 0)
                    or np.any(self.scale['observation'] != .01)):
                raise ValueError('Invalid fetch_like_v1 fixed normalization/motion units')
        if self.observation_mode == 'goal_relative_v1':
            if (config.get('method') != 'fixed_affine_observation_keep_si_goals'
                    or config.get('clipping') is not False or config.get('reward_normalization') is not False
                    or np.any(self.offset['observation'] != 0) or np.any(self.scale['observation'] != .001)):
                raise ValueError('goal_relative_v1 uses zero offsets and millimetre-scale relative features')

    @classmethod
    def from_env(cls, env):
        q, p = env.kinematics.home_q, env.kinematics.home_position
        if env.observation_mode == 'goal_relative_v1':
            return cls(dict(schema=4, observation_mode=env.observation_mode,
                method='fixed_affine_observation_keep_si_goals',
                output_dtype=dict(observation='float32', achieved_goal='float64', desired_goal='float64'),
                clipping=False, reward_normalization=False,
                offset=dict(observation=[0.]*6, achieved_goal=[0.]*3, desired_goal=[0.]*3),
                scale=dict(observation=[.001]*6, achieved_goal=[1.]*3, desired_goal=[1.]*3)))
        if env.observation_mode == 'fetch_like_v1':
            return cls(dict(schema=3, observation_mode='fetch_like_v1',
                method='fixed_affine_observation_keep_si_goals',
                motion_feature='world_site_linear_velocity_times_control_period_m',
                control_period_s=env.control_period,
                output_dtype=dict(observation='float32', achieved_goal='float64', desired_goal='float64'),
                clipping=False, reward_normalization=False,
                offset=dict(observation=np.r_[p, np.zeros(3)].tolist(), achieved_goal=[0.]*3, desired_goal=[0.]*3),
                scale=dict(observation=[.01]*6, achieved_goal=[1.]*3, desired_goal=[1.]*3)))
        offset = dict(observation=np.r_[q, np.zeros(6), p, np.zeros(3), p, q].tolist(),
                      achieved_goal=[0.]*3, desired_goal=[0.]*3)
        scale = dict(observation=np.r_[np.full(6, np.deg2rad(5)), np.full(6, np.deg2rad(30)),
                                         np.full(9, .01), np.full(6, np.deg2rad(5))].tolist(),
                     achieved_goal=[1.]*3, desired_goal=[1.]*3)
        return cls(dict(schema=2, method='fixed_affine_observation_keep_si_goals',
                        output_dtype=dict(observation='float32', achieved_goal='float64', desired_goal='float64'), clipping=False,
                        offset=offset, scale=scale, reward_normalization=False))

    def encode(self, observation):
        if set(observation) != set(FEATURE_KEYS):
            raise ValueError('Expected only observation, achieved_goal and desired_goal')
        batch_shape = None
        for key in FEATURE_KEYS:
            value = np.asarray(observation[key])
            if value.ndim < 1 or value.shape[-1:] != self.offset[key].shape:
                raise ValueError(f'{self.observation_mode} expects {key} shape (..., {len(self.offset[key])}); '
                                 'observation modes cannot be mixed')
            if batch_shape is not None and value.shape[:-1] != batch_shape:
                raise ValueError('Observation Dict batch shapes differ')
            batch_shape = value.shape[:-1]
        result = {k: ((np.asarray(observation[k])-self.offset[k])/self.scale[k]).astype(self.config['output_dtype'][k])
                  for k in self.offset}
        if not all(np.isfinite(x).all() for x in result.values()):
            raise ValueError('Non-finite policy input')
        return result


class ScaledReachObservation(gym.ObservationWrapper):
    def __init__(self, env, transform=None):
        super().__init__(env)
        self.transform = ReachTransform.from_env(env.unwrapped) if transform is None else transform
        if self.transform.config != ReachTransform.from_env(env.unwrapped).config:
            raise ValueError('Observation transform does not match environment mode/home/units/control period')
        self.observation_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, shape=v.shape,
                                                            dtype=self.transform.config['output_dtype'][k])
                                              for k, v in env.observation_space.spaces.items()})

    def observation(self, observation):
        return self.transform.encode(observation)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # The GoalEnv keys deliberately retain metres; never feed normalized
        # physical positions into the original environment's reward/FK/IK.
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)


def features(encoded):
    """Diagnostic/BC flatten only; PPO uses the stock SB3 CombinedExtractor."""
    ordinary = np.asarray(encoded['observation'])
    if (set(encoded) != set(FEATURE_KEYS) or ordinary.ndim < 1
            or ordinary.shape[-1] not in OBSERVATION_DIMS.values()
            or any(np.asarray(encoded[k]).shape != ordinary.shape[:-1]+(3,)
                   for k in ('achieved_goal', 'desired_goal'))):
        raise ValueError('Expected a complete Reach observation Dict with 33 or 12 features')
    result = np.concatenate([encoded[k] for k in FEATURE_KEYS], axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Expected finite processed Dict features')
    return result


def observation_metadata(env, transform):
    """JSON checkpoint contract, including raw versus processed units/dtypes."""
    if transform.config != ReachTransform.from_env(env).config:
        raise ValueError('Transform and environment observation contracts differ')
    raw = {k: dict(shape=list(v.shape), dtype=str(v.dtype), bounds='unbounded')
           for k, v in env.observation_space.spaces.items()}
    processed = {k: dict(v, dtype=transform.config['output_dtype'][k]) for k, v in raw.items()}
    return dict(observation_metadata_version=1, observation_mode=env.observation_mode,
                observation_space=raw, processed_observation_space=processed,
                features_dim=transform.features_dim, feature_keys=list(FEATURE_KEYS),
                observation_field_order=observation_fields(env.observation_mode),
                normalization_version=transform.config['schema'], normalization_config=transform.config)


def validate_model_observation(model, transform):
    space = model.observation_space
    if not isinstance(space, spaces.Dict) or tuple(space.spaces) != FEATURE_KEYS:
        raise ValueError('Model observation Dict field order differs')
    # Older checkpoints predate goal_relative_v1. Their only six-component
    # ordinary observation was fetch_like_v1. New PPO zips store the mode.
    historical_mode = 'fetch_like_v1' if space['observation'].shape == (6,) else 'legacy'
    if getattr(model, 'reach_observation_mode', historical_mode) != transform.observation_mode:
        raise ValueError('Model observation_space and transform observation_mode mismatch')
    for key in FEATURE_KEYS:
        box = space[key]
        if (not isinstance(box, spaces.Box) or box.shape != transform.offset[key].shape
                or box.dtype != np.dtype(transform.config['output_dtype'][key])
                or not np.all(np.isneginf(box.low)) or not np.all(np.isposinf(box.high))):
            raise ValueError('Model observation_space does not match '+transform.observation_mode+': '+key)
    policy = model.policy
    extractors = ([policy.features_extractor] if hasattr(policy, 'features_extractor')
                  and policy.features_extractor is not None else
                  [policy.actor.features_extractor, policy.critic.features_extractor])
    if any(extractor.features_dim != transform.features_dim for extractor in extractors):
        raise ValueError('Model features_dim does not match observation_mode')


def checkpoint_transform(config, model=None, *, observation_mode=None):
    """Recognize historical schema=1 + transform=2 as legacy; never migrate weights."""
    transform = ReachTransform(config['observation_transform'])
    mode = transform.observation_mode
    if (config.get('observation_mode', mode) != mode
            or config['environment'].get('observation_mode', 'legacy') != mode
            or observation_mode is not None and observation_mode != mode):
        raise ValueError('Checkpoint observation_mode mismatch; use the saved mode, never migrate weights')
    if config.get('observation_metadata_version') is None:
        if config.get('schema') != 1 or mode != 'legacy':
            raise ValueError('Missing versioned checkpoint observation metadata')
    else:
        expected_raw = {k: dict(shape=list(v.shape), dtype='float64', bounds='unbounded')
                        for k, v in transform.offset.items()}
        expected_processed = {k: dict(v, dtype=transform.config['output_dtype'][k])
                              for k, v in expected_raw.items()}
        expected = dict(observation_metadata_version=1, observation_space=expected_raw,
                        processed_observation_space=expected_processed, features_dim=transform.features_dim,
                        feature_keys=list(FEATURE_KEYS), observation_field_order=observation_fields(mode),
                        normalization_version=transform.config['schema'], normalization_config=transform.config,
                        observation_mode=mode)
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError('Checkpoint observation metadata/normalization is inconsistent')
    if mode == 'fetch_like_v1' and config['environment'].get('control_period', .05) != transform.config['control_period_s']:
        raise ValueError('Checkpoint motion feature control_period does not match environment')
    if model is not None:
        validate_model_observation(model, transform)
    return transform


def require_legacy(environment, consumer):
    """Fail before a legacy-only consumer reads fixed observation indices."""
    if environment.get('observation_mode', 'legacy') != 'legacy':
        raise ValueError(f"{consumer} supports only observation_mode='legacy' (33 dimensions); "
                         "use the PPO training/evaluation path for other modes")
