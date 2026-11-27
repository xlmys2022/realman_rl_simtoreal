"""加载现场验收的传统抓取参数。仿真场景参数仅用于离线运动学预检。"""

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path

from ..hardware.safety import GripperLimits, GripperOpeningLimits, MotionLimits

GRASP_STAGES = ("open", "align", "pregrasp", "descend", "close", "lift")


@dataclass(frozen=True)
class GraspGeometry:
    table_z_base_m: float
    grasp_tcp_z_base_m: float
    jaw_open_gap_m: float
    pregrasp_clearance_m: float
    lift_height_m: float
    closing_axis_tcp: str
    max_tool_tilt_deg: float
    max_jaw_yaw_error_deg: float
    max_visual_z_error_m: float
    max_target_shift_m: float
    speed_pct: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if name != "closing_axis_tcp" and (type(value) not in (int, float) or not isfinite(value)):
                raise ValueError(f"grasp.{name} 需要现场确认的有限数值")
        if self.grasp_tcp_z_base_m <= self.table_z_base_m:
            raise ValueError("经实物验收的夹持 TCP Z 必须高于桌面")
        if min(self.jaw_open_gap_m, self.lift_height_m,
               self.max_visual_z_error_m, self.max_target_shift_m) <= 0:
            raise ValueError("抓取开口、距离和视觉容差需要正值")
        if self.pregrasp_clearance_m <= 0.001:
            raise ValueError("预抓取高度差须超过 1 mm 的 TCP 到位容差")
        if self.closing_axis_tcp not in ("x", "y"):
            raise ValueError("closing_axis_tcp 须填写实测的 TCP x 或 y 夹紧轴")
        if not (0 < self.max_tool_tilt_deg <= 5 and 0 < self.max_jaw_yaw_error_deg <= 15):
            raise ValueError("固定俯视流程要求工具倾角容差至多 5°、夹指对边容差至多 15°")
        if type(self.speed_pct) is not int or self.speed_pct <= 0:
            raise ValueError("grasp.speed_pct 必须为正整数")


@dataclass(frozen=True)
class GraspSite:
    motion: MotionLimits
    gripper: GripperLimits | GripperOpeningLimits
    geometry: GraspGeometry


def read_site(path):
    """只读取配置。允许按验收进度逐节填写，未用到的节无需填值。"""
    path = Path(path).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取现场抓取配置 {path} · {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("现场抓取配置必须为 JSON 对象")
    return data


def _section(data, name, cls, *, fields=None, **extra):
    values = data.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"现场抓取配置缺少 {name} 节")
    if fields is not None:
        values = {field: values.get(field) for field in fields}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"现场参数未填写 · {', '.join(f'{name}.{key}' for key in missing)}")
    try:
        return cls(**values, **extra)
    except TypeError as exc:
        raise ValueError(f"{name} 节的字段不完整或重复 · {exc}") from exc


def motion_limits(data, calibration):
    return _section(data, "motion", MotionLimits, T_flange_tcp=calibration.T_flange_tcp)


def gripper_limits(data):
    return _section(data, "gripper", GripperLimits)


def gripper_opening_limits(data):
    return _section(data, "gripper", GripperOpeningLimits,
                    fields=("open_speed_raw", "open_feedback_raw", "completion_timeout_s"))


def grasp_site(data, calibration, *, until="lift"):
    if calibration.camera_parent != "tcp" or not calibration.camera_serial:
        raise ValueError("固定俯视流程需要含相机序列号的 Grip_plus TCP 手眼标定")
    if until not in GRASP_STAGES:
        raise ValueError("无效抓取阶段")
    allowed_stage = data.get("max_grasp_stage", "lift")
    if allowed_stage not in GRASP_STAGES:
        raise ValueError("现场 max_grasp_stage 无效")
    if GRASP_STAGES.index(until) > GRASP_STAGES.index(allowed_stage):
        raise ValueError(f"当前现场配置只批准执行到 {allowed_stage}")
    motion = motion_limits(data, calibration)
    gripper = gripper_limits(data) if until in ("close", "lift") else gripper_opening_limits(data)
    geometry = _section(data, "grasp", GraspGeometry)
    if (geometry.jaw_open_gap_m <= calibration.cube_edge_m
            or max(geometry.max_visual_z_error_m, geometry.max_target_shift_m) > calibration.cube_edge_m / 2):
        raise ValueError("夹指净开口须大于方块边长，视觉误差上限须在半边长内")
    if geometry.lift_height_m - geometry.max_target_shift_m <= 0.001:
        raise ValueError("最小可判定抬升高度须超过 1 mm 的 TCP 到位容差")
    # 预抓取试验的下界按该段实际终点设定。夹持端点仅在验收范围到达下降段时检查。
    lowest = (geometry.grasp_tcp_z_base_m + geometry.pregrasp_clearance_m
              if until in ("open", "align", "pregrasp") else geometry.grasp_tcp_z_base_m)
    if not (geometry.table_z_base_m <= motion.workspace_min_m[2] <= lowest <= motion.workspace_max_m[2]
            and geometry.speed_pct <= motion.max_speed_pct):
        raise ValueError("夹持高度、现场工作空间 Z 下界或抓取限速不一致")
    return GraspSite(motion, gripper, geometry)
