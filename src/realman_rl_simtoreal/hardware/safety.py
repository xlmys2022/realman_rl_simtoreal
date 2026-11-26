"""低速直线调试所需的现场安全边界。"""

from dataclasses import dataclass

import numpy as np

from ..utils.geometry import validate_transform
from . import HardwareError


class MotionSafetyError(HardwareError):
    """运动安全检查拦截或停止。"""


@dataclass(frozen=True)
class MotionLimits:
    """显式提供基座系 TCP 工作空间、速度、单次位移、时限及实测 TCP。"""

    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    T_flange_tcp: np.ndarray
    max_speed_pct: int
    max_linear_speed_m_s: float
    max_step_m: float
    completion_timeout_s: float
    max_path_error_m: float = 0.002

    def __post_init__(self):
        low = np.asarray(self.workspace_min_m, dtype=float)
        high = np.asarray(self.workspace_max_m, dtype=float)
        if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                or np.any(low >= high)):
            raise ValueError("工作空间需要三轴有限且有序的基座系边界（m）")
        if type(self.max_speed_pct) is not int or not 1 <= self.max_speed_pct <= 100:
            raise ValueError("max_speed_pct 必须为 1–100 的整数")
        for name in ("max_linear_speed_m_s", "max_step_m", "completion_timeout_s", "max_path_error_m"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        object.__setattr__(self, "workspace_min_m", tuple(low))
        object.__setattr__(self, "workspace_max_m", tuple(high))
        object.__setattr__(self, "T_flange_tcp", validate_transform(self.T_flange_tcp, "T_flange_tcp"))

    def require_in_workspace(self, position_m, label):
        xyz = np.array(position_m, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise MotionSafetyError(f"{label} 必须为有限的基座系 XYZ（m）")
        if np.any(xyz < self.workspace_min_m) or np.any(xyz > self.workspace_max_m):
            raise MotionSafetyError(f"{label} 超出 TCP 工作空间 · {xyz.tolist()}")
        return xyz

    def require_speed(self, speed_pct, controller_max_m_s):
        if type(speed_pct) is not int or not 1 <= speed_pct <= self.max_speed_pct:
            raise MotionSafetyError(f"速度百分比必须为 1–{self.max_speed_pct} 的整数")
        if not np.isfinite(controller_max_m_s) or controller_max_m_s <= 0:
            raise MotionSafetyError("控制器末端线速度上限无效")
        if controller_max_m_s * speed_pct / 100 > self.max_linear_speed_m_s:
            raise MotionSafetyError("按控制器读回的线速度上限换算后，指令超出现场限速")

    def require_segment(self, start_m, target_m):
        start = self.require_in_workspace(start_m, "实测 TCP")
        target = self.require_in_workspace(target_m, "目标 TCP")
        distance = float(np.linalg.norm(target - start))
        if distance > self.max_step_m:
            raise MotionSafetyError("单次 TCP 位移超出现场限幅")
        return distance


@dataclass(frozen=True)
class GripperLimits:
    """限定本机试验过的指令及读回范围。夹爪开口反馈与指令采用不同字段。"""

    open_speed_raw: int
    close_speed_raw: int
    close_force_raw: int
    open_feedback_raw: tuple[int, int]
    held_feedback_raw: tuple[int, int]
    empty_closed_feedback_raw: int
    held_force_g: tuple[int, int]
    completion_timeout_s: float

    def __post_init__(self):
        for name, low, high in (("open_speed_raw", 1, 1000), ("close_speed_raw", 1, 1000),
                                ("close_force_raw", 50, 1000)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} 必须为 SDK 支持的 {low}–{high} 整数")
        for name, minimum in (("open_feedback_raw", 0), ("held_feedback_raw", 0), ("held_force_g", 1)):
            value = getattr(self, name)
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(type(v) is not int for v in value) or not minimum <= value[0] <= value[1]):
                raise ValueError(f"{name} 需要两个现场实测的有序整数")
            object.__setattr__(self, name, tuple(value))
        empty = self.empty_closed_feedback_raw
        if (type(empty) is not int or not 0 <= empty < self.held_feedback_raw[0]
                or self.held_feedback_raw[1] >= self.open_feedback_raw[0]):
            raise ValueError("空夹闭合、实物夹持、张满反馈必须能相互区分")
        timeout = self.completion_timeout_s
        if type(timeout) not in (int, float) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("夹爪动作时限必须为有限正数（s）")
