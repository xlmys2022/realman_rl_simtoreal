"""Reach reward mode names and fixed experimental progress-reward parameters."""
import math
from numbers import Real
from types import MappingProxyType


REWARD_MODES = ('distance', 'safe_dense', 'reach_shaped_v1', 'reach_progress_v1')
REACH_PROGRESS_V1_DEFAULTS = MappingProxyType(dict(
    w_progress=1.0, progress_scale=.001, w_distance=.05, distance_scale=.005,
    step_cost=.01, success_bonus=10.0, safety_penalty=20.0, rejection_cost=.1,
))
REWARD_COMPONENT_KEYS = ('progress_reward', 'distance_penalty', 'step_penalty',
                         'success_bonus', 'safety_penalty', 'rejection_penalty')


def progress_parameters(overrides=None):
    """Copy and validate SI parameters; these defaults are experimental, not tuned."""
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError('reward_progress_parameters must be a parameter dictionary')
    if overrides is not None and set(overrides)-set(REACH_PROGRESS_V1_DEFAULTS):
        raise ValueError('Unknown reach_progress_v1 reward parameter')
    result=dict(REACH_PROGRESS_V1_DEFAULTS)
    result.update({} if overrides is None else overrides)
    for key,value in result.items():
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or value<0:
            raise ValueError('Reward parameter must be finite and nonnegative: '+key)
        if key in ('progress_scale','distance_scale') and value<=0:
            raise ValueError('Reward scales must be positive SI metres: '+key)
        result[key]=float(value)
    return result


def validate_checkpoint_reward(config, model=None):
    """New progress checkpoints must save exactly the parameters used by the env."""
    environment=config['environment']
    if environment.get('reward_mode')=='reach_progress_v1':
        saved=environment.get('reward_progress_parameters')
        if (not isinstance(saved,dict) or set(saved)!=set(REACH_PROGRESS_V1_DEFAULTS)
                or progress_parameters(saved)!=saved or config.get('reward_parameters')!=saved):
            raise ValueError('Checkpoint reach_progress_v1 reward parameters are missing or inconsistent')
    if model is not None:
        signature=getattr(model,'reach_reward_configuration',None)
        if signature is not None:
            if signature!=dict(mode=environment.get('reward_mode'),parameters=config.get('reward_parameters')):
                raise ValueError('Model and checkpoint reward configuration differ')
        elif environment.get('reward_mode')=='reach_progress_v1':
            raise ValueError('Progress model is missing its saved reward configuration')
