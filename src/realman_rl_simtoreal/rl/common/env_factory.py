"""构造 Reach 与抓取的 SB3 环境包装栈，复用原 Monitor、向量化与数值检查。"""
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

from ...envs.factory import make_reach_env
from .observation import ScaledReachObservation


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
