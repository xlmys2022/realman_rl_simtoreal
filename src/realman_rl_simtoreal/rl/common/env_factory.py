"""构造 Reach 与抓取的 SB3 环境包装栈，复用原 Monitor、向量化与数值检查。"""
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

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


def grasp_training_vector(make_env, n_envs, seed):
    """构造局部及关节桌面抓取共用的 Monitor、VecEnv 和 NaN 检查栈。"""
    def factory():
        return Monitor(make_env())

    vector = (DummyVecEnv([factory]) if n_envs == 1 else
              SubprocVecEnv([factory for _ in range(n_envs)], start_method='fork'))
    vector.seed(seed)
    return VecCheckNan(vector, raise_exception=True)
