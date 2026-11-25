"""实机示教器版本和控制器只读回包中的构型、运动学与关节限位。"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SoftwareVersion:
    version: str
    revision: str | None = None
    build_time: str | None = None


@dataclass(frozen=True)
class ModifiedDH:
    """控制器报告的改进 DH 参数。平移为米，扭角和机械零位偏置为度。"""

    d_m: tuple[float, ...]
    a_m: tuple[float, ...]
    alpha_deg: tuple[float, ...]
    offset_deg: tuple[float, ...]


@dataclass(frozen=True)
class RealManHardwareConfig:
    recorded_on: str
    model: str
    arm_dof: int
    arm_generation: Literal["II", "III"]
    sdk_arm_model_value: int
    sdk_force_type: str
    controller_generation: int
    platform_name: str
    platform: SoftwareVersion
    pendant_server: SoftwareVersion
    pendant_web: SoftwareVersion
    pendant_build_time: str
    controller_plan: SoftwareVersion
    controller_control: SoftwareVersion
    controller_algorithm: SoftwareVersion
    dynamics_model_version: int
    joint_firmware: tuple[str, ...]
    tool_interface_board_version: str
    controller_api2_release: str
    observed_sdk_version: str
    controller_dh: ModifiedDH
    joint_min_deg: tuple[float, ...]
    joint_max_deg: tuple[float, ...]
    joint_max_speed_deg_s: tuple[float, ...]
    packing_joints_deg: tuple[float, ...] | None


# 示教器截图提供构建版本。控制器查询返回原始型号枚举 3（RML63II）和力版 3（ISF）。
# 构建时间按设备显示记录，未加时区。
REALMAN_HARDWARE = RealManHardwareConfig(
    recorded_on="2026-11-24",
    model="RML63-6FB",
    arm_dof=6,
    arm_generation="II",
    sdk_arm_model_value=3,
    sdk_force_type="ISF",
    controller_generation=3,
    platform_name="RM_ARM",
    platform=SoftwareVersion("V1.7.0", build_time="2025-03-10T20:42:49"),
    pendant_server=SoftwareVersion("V1.7.0", "15db53f"),
    pendant_web=SoftwareVersion("V1.7.0", "6fc7c5e6"),
    pendant_build_time="2025-03-10T20:43:59",
    controller_plan=SoftwareVersion("V1.7.0", "b34c631", "2025-03-10T20:43:18"),
    controller_control=SoftwareVersion("V1.7.0", "1a610b9", "2025-03-10T20:43:02"),
    controller_algorithm=SoftwareVersion("V1.5.2", "0cabec5b"),
    dynamics_model_version=2,
    joint_firmware=("Vd5.1.0", "Vd5.1.0", "Vd5.1.0", "Vd5.1.0", "Vd5.1.0", "Ve5.1.0"),
    tool_interface_board_version="V1.9.8",
    # 厂家发布表与本机已验证的只读通信版本分别记录。
    controller_api2_release="1.0.7",
    observed_sdk_version="1.1.7",
    # 把读取的 float32 米制数据归整至微米。当前设备 d6 为 132.2 mm。
    controller_dh=ModifiedDH(
        d_m=(.172, 0., 0., .405, 0., .1322),
        a_m=(0., -.086, .380, .069, 0., 0.),
        alpha_deg=(0., -90., 0., 90., -90., -90.),
        offset_deg=(0., -90., 90., 0., 180., 180.),
    ),
    joint_min_deg=(-178., -178., -178., -178., -178., -360.),
    joint_max_deg=(178., 178., 145., 178., 178., 360.),
    joint_max_speed_deg_s=(180., 180., 225., 225., 225., 225.),
    packing_joints_deg=None,
)


@dataclass(frozen=True)
class TabletopInstallation:
    """现场桌面装配参数。长度统一用米，负载使用 kg。"""

    base_above_table_m: float = 0.040
    tcp_name: str = "Grip_plus"
    flange_tcp_m: float = 0.256  # 实测 TCP 沿法兰 +Z，旋转为单位阵。
    tool_payload_kg: float = 0.906
    tool_center_of_mass_m: tuple[float, float, float] = (0.010, 0.044, 0.204)
    gripper_open_feedback_raw: int = 1001
    # 由原装配的名义夹指中心反推总长。半径沿用旧 57 mm 安装板代理。
    gripper_mount_to_nominal_tcp_m: float = 0.147530235
    adapter_radius_m: float = 0.0285
    cube_edge_m: float = 0.020
    camera_serial: str = "152122072347"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    vision_effective_hz: float = 15.0  # 静止时实测约 14–15 Hz，仿真取上沿。
    camera_fx_px: float = 605.1117553710938
    camera_fy_px: float = 605.1603393554688
    camera_cx_px: float = 303.91363525390625
    camera_cy_px: float = 248.02447509765625
    camera_distortion: str = "inverse_brown_conrady"
    camera_distortion_coeffs: tuple[float, ...] = (0., 0., 0., 0., 0.)
    depth_unit_m: float = 0.001

    @property
    def adapter_length_m(self):
        return self.flange_tcp_m - self.gripper_mount_to_nominal_tcp_m


REALMAN_TABLETOP = TabletopInstallation()
