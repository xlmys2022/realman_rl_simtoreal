"""Fixed Reach observation transform and shared 33-dimensional feature order."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


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


FEATURE_KEYS = ('achieved_goal', 'desired_goal', 'observation')


def features(encoded):
    """Same sorted Dict/Flatten order and float32 cast as SB3 CombinedExtractor."""
    result = np.concatenate([encoded[k] for k in FEATURE_KEYS], axis=-1).astype(np.float32)
    if result.shape[-1:] != (33,) or not np.isfinite(result).all():
        raise ValueError('Expected finite 33-dimensional processed Dict features')
    return result
