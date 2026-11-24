"""保存固定观察位关节目标和 TCP 姿态，供传统启动及 checkpoint 契约校验。"""
from dataclasses import dataclass, asdict
from types import MappingProxyType

from ...paths import ROBOT_MODEL


@dataclass(frozen=True)
class ObservationPose:
    """预设名称、世界系 TCP 位置（m）、工具旋转矩阵与六轴角度（度）。"""
    name: str
    tcp_m: tuple
    tool_rotation: tuple
    joints_deg: tuple

    def to_dict(self):
        values = asdict(self)
        return {key: [list(row) if isinstance(row, tuple) else row for row in value]
                if isinstance(value, tuple) else value for key, value in values.items()}


# 固定的世界系抓取 TCP 目标。工具 +Z 沿夹指，关节角由 6FB 场景离线逆解，
# 启动时与方块位置无关。
OBSERVATION_POSES = MappingProxyType({
    'table_down': ObservationPose(
        'table_down', (.10, 0., 1.10),
        ((0., 1., 0.), (1., 0., 0.), (0., 0., -1.)),
        (0., -31.72666510, 91.96456974, 0., 119.76209534, 90.)),
    'forward': ObservationPose(
        'forward', (.15, 0., 1.15),
        ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)),
        (0., -92.67821475, 136.26946672, 0., 46.40874803, 90.)),
})


def startup_contract(profile_name):
    """返回可写入 checkpoint 配置的固定观察位约定及策略交接标记。"""
    return dict(version=3, robot_model=ROBOT_MODEL, initial_joints_deg=[0.] * 6,
                observation_pose=OBSERVATION_POSES[profile_name].to_dict(),
                object_independent=True, handoff='policy_at_observation_pose')
