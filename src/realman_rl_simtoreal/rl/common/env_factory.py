"""Construct the unchanged training wrapper stack and raw observation probes."""
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

from ...envs.factory import make_reach_env
from .observation import ReachTransform, ScaledReachObservation


def make_probe(environment, seeds):
    """Public reset observations only; no BC, experts, actions or training time."""
    with make_reach_env(environment) as env:
        observations=[env.reset(seed=int(s))[0] for s in seeds]
        transform=ReachTransform.from_env(env)
    return {key:np.stack([obs[key] for obs in observations])
            for key in ('achieved_goal','desired_goal','observation')},transform


def training_env(config, monitor_file=None):
    raw = make_reach_env(config)
    wrapped = ScaledReachObservation(raw)
    transform = wrapped.transform
    monitored = Monitor(wrapped, filename=None if monitor_file is None else str(monitor_file),
                        info_keywords=('is_success', 'distance_m', 'termination_reason', 'is_safety_failure'))
    return VecCheckNan(DummyVecEnv([lambda: monitored]), raise_exception=True), transform
