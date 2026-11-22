"""将保存的任务配置分派给对应的 Gymnasium 环境，供训练与独立评估共用。"""
from .reach.reach_env import ReachEnv
from .reach.sequence_reach_env import SequenceReachEnv


def make_reach_env(config):
    """按起点和目标序列字段选择单目标 Reach 或连续多目标 Reach。"""
    if 'start_mode' in config or 'goals_per_episode' in config:
        return SequenceReachEnv(**config)
    return ReachEnv(**config)


def make_tabletop_cartesian_env(config=None,**kwargs):
    """按配置中的观测来源构造真值或腕部 RGB-D 桌面环境。"""
    from ..tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
    from .grasp.tabletop_cartesian_env import TabletopCartesianEnv
    config=CartesianTabletopConfig() if config is None else config
    if config.observation_source=='rgbd_hsv':
        from .grasp.tabletop_cartesian_vision_env import VisionTabletopCartesianEnv
        return VisionTabletopCartesianEnv(config,**kwargs)
    return TabletopCartesianEnv(config,**kwargs)
