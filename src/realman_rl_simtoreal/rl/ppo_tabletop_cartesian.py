"""兼容历史笛卡尔 PPO 模型操作、评估及训练的导入入口。"""
from ..tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config
from ..tasks.grasp.tabletop_cartesian import EASY, STAGES, environment_config
from ..tasks.grasp.tabletop_cartesian_reward import REWARD
from .common.artifacts import optimizer_steps, optimizer_state_hash
from .ppo_cartesian import (
    Audit, capture_sources, deterministic_action, evaluate, hashes, load, provenance, rollout,
    save, write_json,
)


def train(args):
    """按旧入口调用现行桌面训练流程。"""
    from ..apps.training.tabletop_cartesian import train as train_tabletop

    return train_tabletop(args)
