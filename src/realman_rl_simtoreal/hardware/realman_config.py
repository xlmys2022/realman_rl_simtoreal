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
