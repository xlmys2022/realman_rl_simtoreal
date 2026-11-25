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


SCENE_PROFILES = ("legacy", "realman_nominal", "realman_hand_eye")
REALMAN_OBSERVATION_POSES = MappingProxyType({
    # D435 的实际视野更窄，向桌心移动俯视 TCP 后重新用 256 mm 工具逆解。
    'table_down': ObservationPose(
        'table_down', (.19, 0., 1.10), OBSERVATION_POSES['table_down'].tool_rotation,
        (0., -15.6176177683, 72.9938851013, 0., 122.6237326659, 90.)),
    # 旧前视目标的直插路径会碰桌面。上移 40 mm 保持原基座相对高度并通过预检。
    'forward': ObservationPose(
        'forward', (.15, 0., 1.19), OBSERVATION_POSES['forward'].tool_rotation,
        (0., -109.6226236394, 130.0044695218, 0., 69.6181541129, 90.)),
})


def observation_preset(name, scene_profile="legacy"):
    if scene_profile not in SCENE_PROFILES:
        raise ValueError(f"Unknown tabletop scene profile: {scene_profile}")
    presets = OBSERVATION_POSES if scene_profile == "legacy" else REALMAN_OBSERVATION_POSES
    return presets[name]


def startup_contract(profile_name, scene_profile="legacy"):
    """返回可写入 checkpoint 配置的固定观察位约定及策略交接标记。"""
    contract = dict(version=3, robot_model=ROBOT_MODEL, initial_joints_deg=[0.] * 6,
                    observation_pose=observation_preset(profile_name, scene_profile).to_dict(),
                    object_independent=True, handoff='policy_at_observation_pose')
    if scene_profile != "legacy":
        contract.update(version=4, scene_profile=scene_profile)
    return contract
