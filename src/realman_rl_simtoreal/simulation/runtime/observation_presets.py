"""保存固定观察位关节目标和 TCP 姿态，供传统启动及 checkpoint 契约校验。"""
from dataclasses import dataclass, asdict
from types import MappingProxyType


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


# Tool +Z points along the fingers. These joint targets are offline solutions
# for this robot/desk, never solved or moved in response to a detected object.
OBSERVATION_POSES = MappingProxyType({
    'table_down': ObservationPose(
        'table_down', (.10, 0., 1.10),
        ((0., 1., 0.), (1., 0., 0.), (0., 0., -1.)),
        (.16898727934312463, -29.746924071027728, 85.81340526214663,
         -.00016727079258500456, 123.93267679564669, 90.07808700350067)),
    'forward': ObservationPose(
        'forward', (.15, 0., 1.15),
        ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)),
        (.045395134798094594, -99.54833210944871, 134.7752447227696,
         .055716692466343115, 54.7722588880776, 89.96761867329418)),
})


def startup_contract(profile_name):
    """返回可写入 checkpoint 配置的固定观察位约定及策略交接标记。"""
    return dict(version=2, initial_joints_deg=[0.] * 6,
                observation_pose=OBSERVATION_POSES[profile_name].to_dict(),
                object_independent=True, handoff='policy_at_observation_pose')
