"""Build the environment described by a saved Reach task configuration."""
from .reach_env import ReachEnv
from .sequence_reach_env import SequenceReachEnv


def make_reach_env(config):
    if 'start_mode' in config or 'goals_per_episode' in config:
        return SequenceReachEnv(**config)
    return ReachEnv(**config)


def make_tabletop_cartesian_env(config=None,**kwargs):
    from ..tasks.tabletop_cartesian import CartesianTabletopConfig
    from .tabletop_cartesian_env import TabletopCartesianEnv
    config=CartesianTabletopConfig() if config is None else config
    if config.observation_source=='rgbd_hsv':
        from .tabletop_cartesian_vision_env import VisionTabletopCartesianEnv
        return VisionTabletopCartesianEnv(config,**kwargs)
    return TabletopCartesianEnv(config,**kwargs)
