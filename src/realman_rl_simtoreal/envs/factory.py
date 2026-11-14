"""Build the environment described by a saved Reach task configuration."""
from .reach_env import ReachEnv
from .sequence_reach_env import SequenceReachEnv


def make_reach_env(config):
    if 'start_mode' in config or 'goals_per_episode' in config:
        return SequenceReachEnv(**config)
    return ReachEnv(**config)
